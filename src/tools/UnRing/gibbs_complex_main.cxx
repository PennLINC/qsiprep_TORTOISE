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

    PFGeometry geom = DetectPFGeometry(dwis, pe_axis, zero_tol, 8);

    if(geom.status==PFGeometry::ImplausibleFactor && !force_pf)
    {
        std::cout<<"Detected a k-space zero band of "<<geom.n_missing<<" lines out of "
                 <<geom.n_pe<<" (factor "<<geom.factor<<"), which is below 0.5 and "
                 <<"implausible for partial Fourier. Exiting..."<<std::endl;
        return EXIT_FAILURE;
    }

    if(pf_factor_arg>0 || pf_side_arg!=std::string(""))
    {
        PFGeometry declared = geom;
        declared.status = PFGeometry::DetectedPF;
        declared.is_partial_fourier = true;

        if(pf_factor_arg>0)
        {
            declared.factor = pf_factor_arg;
            declared.n_missing = (int)(llround((1.-(double)pf_factor_arg)*(double)declared.n_pe));
        }
        if(pf_side_arg==std::string("low"))
            declared.side = PFGeometry::Low;
        if(pf_side_arg==std::string("high"))
            declared.side = PFGeometry::High;

        bool mismatch = (!geom.is_partial_fourier)
                        || (geom.n_missing != declared.n_missing)
                        || (geom.side != declared.side);

        if(mismatch)
        {
            std::cout<<"Declared and detected partial-Fourier geometry disagree."<<std::endl;
            std::cout<<"  detected: pf="<<(geom.is_partial_fourier?"yes":"no")
                     <<" n_missing="<<geom.n_missing<<" factor="<<geom.factor
                     <<" side="<<(geom.side==PFGeometry::Low?"low":"high")<<std::endl;
            std::cout<<"  declared: n_missing="<<declared.n_missing
                     <<" factor="<<declared.factor
                     <<" side="<<(declared.side==PFGeometry::Low?"low":"high")<<std::endl;
            if(!force_pf)
            {
                std::cout<<"Pass --force_pf 1 to proceed with the declared geometry. Exiting..."<<std::endl;
                return EXIT_FAILURE;
            }
            std::cout<<"--force_pf given: proceeding with the declared geometry."<<std::endl;
        }
        geom = declared;
    }

    if(geom.is_partial_fourier)
    {
        std::cout<<"Partial Fourier detected: "<<geom.n_missing<<" of "<<geom.n_pe
                 <<" k-space lines empty (factor "<<geom.factor<<", "
                 <<(geom.side==PFGeometry::Low?"low":"high")<<" side, band energy ratio "
                 <<geom.zero_band_energy_ratio<<")."<<std::endl;
    }
    else
    {
        std::cout<<"No partial-Fourier zero band found. The data does not look "
                 <<"zero-filled -- it may be full Fourier, or already reconstructed "
                 <<"with homodyne or POCS."<<std::endl;
    }

    if(do_pocs && geom.is_partial_fourier)
    {
        std::cout<<"Running POCS partial-Fourier reconstruction..."<<std::endl;
        POCSResult res = ApplyPOCS(dwis, geom, pe_axis, pocs_params);
        std::cout<<"POCS finished after at most "<<res.iters_run
                 <<" iterations, final relative change "<<res.final_rel_change<<"."<<std::endl;
    }
    else if(do_pocs && !geom.is_partial_fourier)
    {
        std::cout<<"Skipping POCS."<<std::endl;
    }
    else if(!do_pocs && geom.is_partial_fourier)
    {
        std::cout<<"WARNING: POCS disabled on partial-Fourier data. Residual "
                 <<"partial-Fourier ringing may remain along the phase encoding "
                 <<"direction. The magnitude-domain RPG method in the Gibbs command "
                 <<"addresses that case."<<std::endl;
    }

    dwis = UnRingFullComplex(dwis, nsh, minW, maxW);

    if(pe_dir==0)
        dwis = TransposeInPlane(dwis);

    dwis->SetDirection(orig_dir);
    dwis->SetSpacing(orig_spc);
    dwis->SetOrigin(orig_org);

    writeImageD<ImageType4DComplex>(dwis, output_name);

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

        writeImageD<ImageType4D>(mag, mag_name);
    }

    delete parser;
    return EXIT_SUCCESS;
}

#endif
