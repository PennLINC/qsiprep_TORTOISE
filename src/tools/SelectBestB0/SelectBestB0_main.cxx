#ifndef SELECTBESTB0MAIN_CXX
#define SELECTBESTB0MAIN_CXX

#include "defines.h"
#include "select_best_b0.h"
#include "itkMultiThreaderBase.h"

#include <string>
#include <vector>


int main(int argc, char * argv[])
{
    // Pull the optional flags out of the argument list so the positional
    // arguments keep their fixed order regardless of where a flag sits.
    int ncores = -1;
    float b0_threshold = -1;
    std::string report_name;
    std::vector<std::string> pos;
    for(int i=1; i<argc; i++)
    {
        std::string arg(argv[i]);
        if(arg=="--ncores" && i+1<argc)
        {
            ncores = atoi(argv[++i]);
            continue;
        }
        if(arg=="--b0_threshold" && i+1<argc)
        {
            b0_threshold = atof(argv[++i]);
            continue;
        }
        if(arg=="--report" && i+1<argc)
        {
            report_name = argv[++i];
            continue;
        }
        pos.push_back(arg);
    }

    if(pos.size()<3)
    {
        std::cout<<"Usage: SelectBestB0 NIFTI_file bvals_file output_name [--b0_threshold B] [--ncores N] [--report path.tsv]"<<std::endl;
        std::cout<<"  --b0_threshold B: treat volumes with b-value <= B as b=0 candidates (default: within 10 of the smallest b-value)"<<std::endl;
        std::cout<<"  --ncores N:       cap OpenMP and ITK threads at N (default: all detected cores)"<<std::endl;
        std::cout<<"  --report path:    write per-volume scores and rigid parameters to this TSV (default: output_name with the image extension replaced by .tsv)"<<std::endl;
        return EXIT_FAILURE;
    }

    TORTOISE t;

    // The TORTOISE constructor sets omp_set_num_threads() to the host core count
    // (getNCores() ignores OMP_NUM_THREADS and cgroup limits), so the cap has to
    // be applied after it. Mirrors gibbs_main.cxx.
    if(ncores>0)
    {
        itk::MultiThreaderBase::SetGlobalDefaultNumberOfThreads(ncores);
        omp_set_num_threads(ncores);
        TORTOISE::SetNAvailableCores(ncores);
    }

    std::string nii_name(pos[0]);
    std::string bvals_name(pos[1]);
    std::string output_name(pos[2]);

    if(report_name=="")
    {
        report_name = output_name;
        if(report_name.size()>7 && report_name.substr(report_name.size()-7)==".nii.gz")
            report_name = report_name.substr(0,report_name.size()-7);
        else if(report_name.size()>4 && report_name.substr(report_name.size()-4)==".nii")
            report_name = report_name.substr(0,report_name.size()-4);
        report_name += ".tsv";
    }

    ImageType4D::Pointer img =readImageD<ImageType4D>(nii_name);
    int Nvols = img->GetLargestPossibleRegion().GetSize()[3];

    vnl_vector<double> bvals(Nvols);
    std::ifstream infileb(bvals_name.c_str());
    infileb>>bvals;
    infileb.close();

    ImageType3D::Pointer best_b0_img;

    int id = select_best_b0(img,bvals,best_b0_img,b0_threshold,report_name);
    if(id<0)
        return EXIT_FAILURE;

    std::cout<<"Selected b=0 volume id: "<<id<<std::endl;
    std::cout<<"Selection report: "<<report_name<<std::endl;

    writeImageD<ImageType3D>(best_b0_img,output_name);
    return EXIT_SUCCESS;

}



#endif
