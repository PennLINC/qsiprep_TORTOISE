#ifndef _GIBBSMAIN_CXX_
#define _GIBBSMAIN_CXX_


#include "defines.h"
#include "unring.h"
#include "itkMultiThreaderBase.h"
#include <vector>


int main(int argc, char* argv[])
{
    // Pull the optional flags out of the argument list so the positional
    // arguments below keep their fixed order regardless of where a flag sits.
    int ncores = -1;
    bool disable_itk_threads = false;
    std::vector<std::string> pos;
    for(int i=1; i<argc; i++)
    {
        std::string arg(argv[i]);
        if(arg=="--ncores" && i+1<argc)
        {
            ncores = atoi(argv[++i]);
            continue;
        }
        if(arg=="--disable_itk_threads")
        {
            disable_itk_threads = true;
            continue;
        }
        pos.push_back(arg);
    }

    if(pos.size()<4)
    {
        std::cout<<"Usage: Gibbs input_nifti  output_nifti kspace_coverage(1,0.875,0.75) phase_encoding_dir(0: horizontal, 1:vertical) nsh(optional) minW(optional) maxW(optional) [--ncores N] [--disable_itk_threads] "<<std::endl;
        return EXIT_FAILURE;
    }

    TORTOISE t;

    // The TORTOISE constructor sets omp_set_num_threads() to the host core count
    // (getNCores() = sysconf(_SC_NPROCESSORS_ONLN); it ignores OMP_NUM_THREADS),
    // so any cap has to be applied after it. Mirroring DRBUDDI_main.cxx, --ncores
    // sets the ITK and OpenMP thread counts together; --disable_itk_threads then
    // pins ITK to one thread so the two layers can't multiply (ncores x ncores)
    // under a scheduler allocation.
    if(ncores>0)
    {
        itk::MultiThreaderBase::SetGlobalDefaultNumberOfThreads(ncores);
        omp_set_num_threads(ncores);
        TORTOISE::SetNAvailableCores(ncores);
    }
    if(disable_itk_threads)
    {
        itk::MultiThreaderBase::SetGlobalDefaultNumberOfThreads(1);
    }

    std::string input_name(pos[0]);
    std::string output_name(pos[1]);
    float gibbs_kspace_coverage= atof(pos[2].c_str());
    int gibbs_nsh=25;
    int gibbs_minW=1;
    int gibbs_maxW=3;

    short phase=atoi(pos[3].c_str());

    if(pos.size()>4)
        gibbs_nsh=atoi(pos[4].c_str());
    if(pos.size()>5)
        gibbs_minW=atoi(pos[5].c_str());
    if(pos.size()>6)
        gibbs_maxW=atoi(pos[6].c_str());


    float ks_cov= gibbs_kspace_coverage;

    ImageType4D::DirectionType orig_dir;
    ImageType4D::SpacingType orig_spc;
    ImageType4D::RegionType orig_reg;
    ImageType4D::PointType orig_or;


    ImageType4D::Pointer dwis= readImageD<ImageType4D>(input_name);
    if(phase==0)
    {
        orig_dir=dwis->GetDirection();
        orig_spc=dwis->GetSpacing();
        orig_reg=dwis->GetLargestPossibleRegion();
        orig_or =dwis->GetOrigin();

        ImageType4D::SizeType new_size;
        new_size[0]=dwis->GetLargestPossibleRegion().GetSize()[1];
        new_size[1]=dwis->GetLargestPossibleRegion().GetSize()[0];
        new_size[2]=dwis->GetLargestPossibleRegion().GetSize()[2];
        new_size[3]=dwis->GetLargestPossibleRegion().GetSize()[3];

        ImageType4D::IndexType start; start.Fill(0);
        ImageType4D::RegionType reg(start,new_size);

        ImageType4D::Pointer dwis2= ImageType4D::New();
        dwis2->SetRegions(reg);
        dwis2->Allocate();
        dwis2->FillBuffer(0);

        itk::ImageRegionIteratorWithIndex<ImageType4D> it(dwis2,dwis2->GetLargestPossibleRegion());
        for(it.GoToBegin();!it.IsAtEnd();++it)
        {
            ImageType4D::IndexType ind4= it.GetIndex();
            ImageType4D::IndexType ind4_old=ind4;
            ind4_old[0]=ind4[1];
            ind4_old[1]=ind4[0];
            it.Set(dwis->GetPixel(ind4_old));
        }
        dwis=dwis2;
    }


    if(ks_cov>=0.9375)
    {
        std::cout<<"Gibbs correction with full k-space coverage"<<std::endl;
        dwis=UnRingFull(dwis,gibbs_nsh,gibbs_minW,gibbs_maxW);
    }
    else
    {
        if(ks_cov<0.9375 && ks_cov >=0.8125)
        {
            std::cout<<"Gibbs correction with 7/8 k-space coverage"<<std::endl;
            dwis=UnRing78(dwis,gibbs_nsh,gibbs_minW,gibbs_maxW);
        }
        else
        {
            if(ks_cov>0.65)
            {
                std::cout<<"Gibbs correction with 6/8 k-space coverage"<<std::endl;
                dwis=UnRing68(dwis,gibbs_nsh,gibbs_minW,gibbs_maxW);
            }
            else
            {
                std::cout<<"K-space coverage in the data is less than 65\%. Skipping Gibbs ringing correction. "<<std::endl;
            }
        }
    }

    if(phase==0)
    {
        ImageType4D::Pointer dwis2= ImageType4D::New();
        dwis2->SetRegions(orig_reg);
        dwis2->SetSpacing(orig_spc);
        dwis2->SetDirection(orig_dir);
        dwis2->SetOrigin(orig_or);
        dwis2->Allocate();
        dwis2->FillBuffer(0);

        itk::ImageRegionIteratorWithIndex<ImageType4D> it(dwis2,dwis2->GetLargestPossibleRegion());
        for(it.GoToBegin();!it.IsAtEnd();++it)
        {
            ImageType4D::IndexType ind4= it.GetIndex();
            ImageType4D::IndexType ind4_old=ind4;
            ind4_old[0]=ind4[1];
            ind4_old[1]=ind4[0];
            it.Set(dwis->GetPixel(ind4_old));
        }
        dwis=dwis2;
    }

    writeImageD<ImageType4D>(dwis,output_name);

}


#endif
