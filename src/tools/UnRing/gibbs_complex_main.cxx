#ifndef _GIBBSCOMPLEXMAIN_CXX_
#define _GIBBSCOMPLEXMAIN_CXX_

#include "defines.h"
#include "unring.h"
#include "pf_geometry.h"
#include "pocs.h"
#include "gibbs_complex_parser.h"
#include "itkMultiThreaderBase.h"
#include "itkImageIOFactory.h"
#include "itkImageIOBase.h"
#include "itkImageRegionIteratorWithIndex.h"
#include <cmath>
#include <complex>

// Swap the two in-plane axes. DetectPFGeometry, ApplyPOCS and the SuShi 1D
// pass are all written against a vertical phase-encode axis, so --pe_dir 0
// data is transposed on the way in and back on the way out -- the same trick
// gibbs_main.cxx uses.
ImageType4DComplex::Pointer TransposeInPlane(ImageType4DComplex::Pointer img)
{
    ImageType4DComplex::SizeType osz = img->GetLargestPossibleRegion().GetSize();

    ImageType4DComplex::SizeType nsz;
    nsz[0]=osz[1];
    nsz[1]=osz[0];
    nsz[2]=osz[2];
    nsz[3]=osz[3];

    ImageType4DComplex::IndexType start; start.Fill(0);
    ImageType4DComplex::RegionType reg(start,nsz);

    ImageType4DComplex::Pointer out= ImageType4DComplex::New();
    out->SetRegions(reg);
    out->Allocate();

    itk::ImageRegionIteratorWithIndex<ImageType4DComplex> it(out,out->GetLargestPossibleRegion());
    for(it.GoToBegin();!it.IsAtEnd();++it)
    {
        ImageType4DComplex::IndexType ind4= it.GetIndex();
        ImageType4DComplex::IndexType old=ind4;
        old[0]=ind4[1];
        old[1]=ind4[0];
        it.Set(img->GetPixel(old));
    }
    return out;
}

int main(int argc, char* argv[])
{
    GibbsComplex_PARSER *parser = new GibbsComplex_PARSER(argc,argv);

    TORTOISE t;

    // The TORTOISE constructor sets omp_set_num_threads() to the host core
    // count and ignores OMP_NUM_THREADS, so any cap has to be applied after
    // it. Mirrors gibbs_main.cxx and DRBUDDI_main.cxx.
    int ncores = parser->getNumberOfCores();
    if(ncores>0)
    {
        itk::MultiThreaderBase::SetGlobalDefaultNumberOfThreads(ncores);
        omp_set_num_threads(ncores);
        TORTOISE::SetNAvailableCores(ncores);
    }
    if(parser->getDisableITKThreads())
    {
        itk::MultiThreaderBase::SetGlobalDefaultNumberOfThreads(1);
    }

    std::string input_name  = parser->getInputImageName();
    std::string output_name = parser->getOutputImageName();
    std::string mag_name    = parser->getOutputMagnitudeName();
    int   pe_dir     = parser->getPEDir();
    bool  do_pocs    = parser->getDoPOCS();
    bool  force_pf   = parser->getForcePF();
    float zero_tol   = parser->getZeroTol();
    float pf_factor_arg = parser->getPFFactor();
    std::string pf_side_arg = parser->getPFSide();

    POCSParams pocs_params;
    pocs_params.iters = parser->getPOCSIters();
    pocs_params.tol   = parser->getPOCSTol();

    int nsh  = parser->getNsh();
    int minW = parser->getMinW();
    int maxW = parser->getMaxW();

    // Refuse magnitude input rather than silently treating it as complex.
    {
        itk::ImageIOBase::Pointer io = itk::ImageIOFactory::CreateImageIO(
            input_name.c_str(), itk::ImageIOFactory::FileModeEnum::ReadMode);
        if(io.IsNull())
        {
            std::cout<<"Could not read "<<input_name<<" . Exiting..."<<std::endl;
            return EXIT_FAILURE;
        }
        io->SetFileName(input_name);
        io->ReadImageInformation();
        if(io->GetPixelType()!=itk::IOPixelEnum::COMPLEX)
        {
            std::cout<<"Input image "<<input_name<<" is not complex-valued (datatype: "
                     <<io->GetPixelTypeAsString(io->GetPixelType())
                     <<"). GibbsComplex requires a COMPLEX64 or COMPLEX128 NIFTI. "
                     <<"For magnitude-only data use the Gibbs command. Exiting..."<<std::endl;
            return EXIT_FAILURE;
        }
    }

    ImageType4DComplex::Pointer dwis = readImageD<ImageType4DComplex>(input_name);

    ImageType4DComplex::DirectionType orig_dir = dwis->GetDirection();
    ImageType4DComplex::SpacingType   orig_spc = dwis->GetSpacing();
    ImageType4DComplex::PointType     orig_org = dwis->GetOrigin();
    ImageType4DComplex::RegionType    orig_reg = dwis->GetLargestPossibleRegion();

    if(pe_dir==0)
        dwis = TransposeInPlane(dwis);

    // After the transpose the phase-encode axis is always 1.
    const int pe_axis = 1;

    // ------------------------------------------------------------------
    // Diagnostics. These REPORT; they no longer decide.
    //
    // The partial-Fourier geometry comes from acquisition metadata via --pf_factor: a
    // BIDS sidecar knows the acquired factor, and a reconstructed image does not
    // reliably reveal it. Whether the exported image still LOOKS zero-filled is a
    // separate question, answered by the conjugate-mirror energy ratio. POCS is enabled
    // only on explicit request, never by image-derived detection alone -- re-running a
    // partial-Fourier reconstruction on top of the scanner's would corrupt data that was
    // already correct.
    // ------------------------------------------------------------------
    PFGeometry geom = DetectPFGeometry(dwis, pe_axis, zero_tol, 8);

    PFDiagnostics diag;
    diag.valid = false;
    if(pf_factor_arg>0)
        diag = ComputePFDiagnostics(dwis, pe_axis, pf_factor_arg, 8, 16);

    std::cout<<"---- partial-Fourier diagnostics ----"<<std::endl;
    if(pf_factor_arg>0)
        std::cout<<"  Declared PF factor:       "<<pf_factor_arg<<std::endl;
    else
        std::cout<<"  Declared PF factor:       none given (pass --pf_factor from the BIDS sidecar)"<<std::endl;

    if(diag.valid)
    {
        std::cout<<"  Expected missing lines:   "<<diag.n_missing<<" of "<<diag.n_pe<<std::endl;
        std::cout<<"  Likely missing side:      "<<(diag.side==PFGeometry::Low?"low":"high")
                 <<" (asymmetry "<<diag.side_asymmetry<<"x)"<<std::endl;
        std::cout<<"  Band/mirror energy:       "<<diag.mirror_ratio
                 <<" (p10 "<<diag.mirror_p10<<", p90 "<<diag.mirror_p90
                 <<", n="<<diag.n_samples<<")"<<std::endl;
        std::cout<<"  Zero-fill compatibility:  "
                 <<(diag.zero_fill_compatible ? "strong -- consistent with zero filling"
                                              : "weak -- does NOT look zero-filled")<<std::endl;
    }
    else if(pf_factor_arg>0)
    {
        std::cout<<"  Zero-fill compatibility:  not computable for the declared factor"<<std::endl;
    }

    if(geom.status==PFGeometry::SymmetricBand)
        std::cout<<"  Literal zero-band scan:   empty bands at BOTH edges -- looks like symmetric "
                 <<"zero-padding (matrix interpolation), not partial Fourier"<<std::endl;
    else if(geom.is_partial_fourier)
        std::cout<<"  Literal zero-band scan:   "<<geom.n_missing<<" exactly-empty lines, "
                 <<(geom.side==PFGeometry::Low?"low":"high")<<" side"<<std::endl;
    else
        std::cout<<"  Literal zero-band scan:   no exactly-empty band (normal for real data)"<<std::endl;
    std::cout<<"-------------------------------------"<<std::endl;

    // ------------------------------------------------------------------
    // POCS: opt-in, and only with declared geometry.
    // ------------------------------------------------------------------
    bool run_pocs = false;
    PFGeometry pocs_geom;
    pocs_geom.status = PFGeometry::DetectedPF;
    pocs_geom.is_partial_fourier = true;
    pocs_geom.n_pe = 0; pocs_geom.n_missing = 0; pocs_geom.factor = 1.f;
    pocs_geom.side = PFGeometry::Low; pocs_geom.zero_band_energy_ratio = 0.f;

    if(do_pocs)
    {
        if(pf_factor_arg<=0)
        {
            std::cout<<"--pocs 1 requires --pf_factor, the partial-Fourier factor from the "
                     <<"acquisition metadata. POCS is never enabled from image-derived detection "
                     <<"alone: running a partial-Fourier reconstruction on data the scanner has "
                     <<"already reconstructed corrupts it. Exiting..."<<std::endl;
            return EXIT_FAILURE;
        }
        if(!diag.valid)
        {
            std::cout<<"Could not evaluate zero-fill compatibility for --pf_factor "<<pf_factor_arg
                     <<". Exiting..."<<std::endl;
            return EXIT_FAILURE;
        }
        if(!diag.zero_fill_compatible && !force_pf)
        {
            std::cout<<"Refusing to run POCS. The un-acquired band holds "<<diag.mirror_ratio
                     <<" of its conjugate mirror's energy, above the "<<PF_ZEROFILL_MAX_MIRROR_RATIO
                     <<" threshold, so this image does not look zero-filled -- the scanner has "
                     <<"most likely reconstructed it already and POCS would overwrite that "
                     <<"reconstruction with synthesised content. Pass --force_pf 1 to override. "
                     <<"Exiting..."<<std::endl;
            return EXIT_FAILURE;
        }
        if(!diag.zero_fill_compatible)
            std::cout<<"--force_pf given: running POCS despite weak zero-fill compatibility."<<std::endl;

        pocs_geom.n_pe = diag.n_pe;
        pocs_geom.n_missing = diag.n_missing;
        pocs_geom.factor = pf_factor_arg;
        pocs_geom.side = diag.side;
        pocs_geom.zero_band_energy_ratio = diag.mirror_ratio;

        if(pf_side_arg!=std::string(""))
        {
            const PFGeometry::Side declared_side =
                (pf_side_arg==std::string("low")) ? PFGeometry::Low : PFGeometry::High;
            if(declared_side != diag.side)
                std::cout<<"NOTE: --pf_side "<<pf_side_arg<<" overrides the inferred side ("
                         <<(diag.side==PFGeometry::Low?"low":"high")<<")."<<std::endl;
            pocs_geom.side = declared_side;
        }
        run_pocs = true;
    }

    if(run_pocs)
    {
        std::cout<<"Running POCS partial-Fourier reconstruction ("<<pocs_geom.n_missing
                 <<" lines, "<<(pocs_geom.side==PFGeometry::Low?"low":"high")<<" side)..."<<std::endl;
        POCSResult res = ApplyPOCS(dwis, pocs_geom, pe_axis, pocs_params);
        std::cout<<"POCS finished after at most "<<res.iters_run
                 <<" iterations, final relative change "<<res.final_rel_change<<"."<<std::endl;
        if(res.final_rel_change > pocs_params.tol)
            std::cout<<"WARNING: POCS reached the iteration cap without meeting the requested "
                     <<"tolerance ("<<pocs_params.tol<<"). Consider raising --pocs_iters."<<std::endl;
    }
    else if(diag.valid && diag.zero_fill_compatible)
    {
        std::cout<<"POCS not enabled. This image IS compatible with zero filling, so residual "
                 <<"partial-Fourier ringing may remain along the phase encoding direction. The "
                 <<"magnitude-domain RPG method in the Gibbs command addresses that case."<<std::endl;
    }

    dwis = UnRingFullComplex(dwis, nsh, minW, maxW);

    if(pe_dir==0)
        dwis = TransposeInPlane(dwis);

    dwis->SetDirection(orig_dir);
    dwis->SetSpacing(orig_spc);
    dwis->SetOrigin(orig_org);

    // The complex output is the primary deliverable and is written first, so
    // a subsequent magnitude write failure can never lose it. An unwritable
    // path (bad directory, permissions, disk full, ...) must produce a clear
    // error and a non-zero exit, not let the ITK exception escape main() and
    // abort the process.
    try
    {
        writeImageD<ImageType4DComplex>(dwis, output_name);
    }
    catch(itk::ExceptionObject &ex)
    {
        std::cout<<"Could not write output image "<<output_name<<" : "<<ex<<std::endl;
        return EXIT_FAILURE;
    }

    if(mag_name!=std::string(""))
    {
        ImageType4D::Pointer mag = ImageType4D::New();
        mag->SetRegions(orig_reg);
        mag->SetDirection(orig_dir);
        mag->SetSpacing(orig_spc);
        mag->SetOrigin(orig_org);
        mag->Allocate();
        mag->FillBuffer(0);

        itk::ImageRegionIteratorWithIndex<ImageType4D> it(mag,mag->GetLargestPossibleRegion());
        for(it.GoToBegin();!it.IsAtEnd();++it)
            it.Set( std::abs( dwis->GetPixel(it.GetIndex()) ) );

        try
        {
            writeImageD<ImageType4D>(mag, mag_name);
        }
        catch(itk::ExceptionObject &ex)
        {
            std::cout<<"Could not write output magnitude image "<<mag_name<<" : "<<ex<<std::endl;
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}

#endif
