#ifndef _EPIREG_CXX
#define _EPIREG_CXX



#include "EPIREG.h"
#include "../utilities/read_bmatrix_file.h"
#include "../utilities/read_3Dvolume_from_4D.h"
#include "registration_settings.h"

#include "../tools/EstimateTensor/estimate_tensor_wlls.h"
#include "create_mask.h"
//#include "../tools/ComputeFAMap/compute_fa_map.h"
//#include "../tools/RotateBMatrix/rotate_bmatrix.h"

#include "rigid_register_images.h"
#include "itkResampleImageFilter.h"

#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkResampleImageFilter.h"
#include "itkImageDuplicator.h"
#include "itkKdTreeGenerator.h"
#include "itkKdTree.h"
#include "itkListSample.h"
#include "itkAddImageFilter.h"
#include "itkDivideImageFilter.h"
#include "itkTransformFileWriter.h"
#include "itkBSplineInterpolateImageFunction.h"

#include "DRBUDDI_Diffeo.h"


EPIREG::EPIREG(std::string uname,std::vector<std::string> str_names,json mjson)
{
    this->up_nii_name=uname;

    this->structural_names=str_names;
    my_json=mjson;

    std::string json_PE= my_json["PhaseEncodingDirection"];      //get phase encoding direction
    if(json_PE.find("j")!=std::string::npos)
        PE_string="vertical";
    else
        if(json_PE.find("i")!=std::string::npos)
            PE_string="horizontal";
        else
            PE_string="slice";

    this->proc_folder = fs::path(up_nii_name).parent_path().string();

    this->stream= TORTOISE::stream;
    (*stream)<<"Starting EPIREG Processing..."<<std::endl;
}


void EPIREG::Process()
{
    Step0_CreateImages();
    Step1_RigidRegistration();
    Step2_DiffeoRegistration();
    Step3_WriteOutput();
}

void EPIREG::Step0_CreateImages()
{
    ImageType3D::Pointer dummy;
    CreateCorrectionImage(this->up_nii_name,this->b0_up,dummy);

    std::string gradnonlin_field_name= parser->getGradNonlinInput();
    if(gradnonlin_field_name!="" && parser->getNOGradWarp()==false)
    {
        // The field TORTOISE converted and wrote at import, as DRBUDDI reads it (not the raw
        // --grad_nonlin argument, which may be a coefficient file).
        std::string up_name = this->parser->getUpInputName();
        std::string basename= fs::path(up_name).filename().string();
        basename=basename.substr(0,basename.rfind(".nii"));
        std::string gradnonlin_name_inv= this->proc_folder + std::string("/") + basename + std::string("_proc_gradnonlin_field_inv.nii");
        DisplacementFieldType::Pointer field= readImageD<DisplacementFieldType>(gradnonlin_name_inv);

        DisplacementFieldTransformType::Pointer gradwarp_trans=DisplacementFieldTransformType::New();
        gradwarp_trans->SetDisplacementField(field);

        using ResampleImageFilterType= itk::ResampleImageFilter<ImageType3D, ImageType3D> ;
        {
            ResampleImageFilterType::Pointer resampleFilter = ResampleImageFilterType::New();
            resampleFilter->SetOutputParametersFromImage(b0_up);
            resampleFilter->SetInput(b0_up);
            resampleFilter->SetTransform(gradwarp_trans);
            resampleFilter->Update();
            this->b0_up=resampleFilter->GetOutput();
        }

    }

    writeImageD<ImageType3D>(this->b0_up,proc_folder+"/blip_up_b0.nii");
}




void EPIREG::Step1_RigidRegistration()
{
    //Create and write b0_up quad image
    CreateBlipUpQuadImage();
    writeImageD<ImageType3D>(this->b0_up_quad,proc_folder+"/blip_up_b0_quad.nii");



    // Register the structural to the least distorted b=0 available: the quad b=0, unwarped by
    // --EPIREG_initial_field when one is given (e.g. from a GRE field map). A rigid fit to the
    // distorted b=0 is biased by the distortion; on a TRXScan fixture with known truth it placed
    // the T2w about 3 degrees off.
    ImageType3D::Pointer str_target= this->b0_up_quad;
    std::string init_field_name = parser->GetEPIREGInitialField();
    if(init_field_name!="")
    {
        DisplacementFieldType::Pointer init_field= readImageD<DisplacementFieldType>(init_field_name);
        DisplacementFieldTransformType::Pointer init_trans= DisplacementFieldTransformType::New();
        init_trans->SetDisplacementField(init_field);

        using ResampleImageFilterType= itk::ResampleImageFilter<ImageType3D, ImageType3D> ;
        ResampleImageFilterType::Pointer resampleFilter = ResampleImageFilterType::New();
        resampleFilter->SetOutputParametersFromImage(this->b0_up_quad);
        resampleFilter->SetInput(this->b0_up_quad);
        resampleFilter->SetTransform(init_trans);
        resampleFilter->SetDefaultPixelValue(0);
        resampleFilter->Update();
        str_target= resampleFilter->GetOutput();
        (*stream)<<"Registering the structural to the b=0 unwarped by "<<init_field_name<<std::endl;
    }
    writeImageD<ImageType3D>(str_target,proc_folder+"/b0_str_registration_target.nii");

    // and finally rigid register all structural images to the kind of corrected b0 image
    int Nstr= parser->getNumberOfStructurals();

    for(int str=0;str<Nstr;str++)
    {
        (*stream)<<"Rigidly registering structural image id: " <<str<<" to b0_up quad..."<<std::endl;

        ImageType3D::Pointer str_img = readImageD<ImageType3D>(parser->getStructuralNames(str));
        RigidTransformType::Pointer rigid_trans;
        if(parser->getDisableInitRigid())
        {
            // The caller has already placed the structural on the b=0; only resample it onto the quad grid.
            (*stream)<<"--DRBUDDI_disable_initial_rigid: structural image used at the given pose"<<std::endl;
            rigid_trans= RigidTransformType::New();
            rigid_trans->SetIdentity();
        }
        else
        {
            // The same CC/MI comparison and forward/backward consistency check as DRBUDDI's structural
            // rigid, instead of a single unchecked registration.
            rigid_trans= RegisterStructuralToB0(str_target, str_img);
            (*stream)<<"Rigid transformation: " << rigid_trans->GetParameters()<<std::endl;
        }

        {
            using ResampleImageFilterType = itk::ResampleImageFilter<ImageType3D, ImageType3D> ;
            ResampleImageFilterType::Pointer resampleFilter3 = ResampleImageFilterType::New();
            resampleFilter3->SetOutputParametersFromImage(this->b0_up_quad);
            resampleFilter3->SetInput(str_img);
            resampleFilter3->SetTransform(rigid_trans);
            resampleFilter3->SetDefaultPixelValue(0);
            resampleFilter3->Update();
            ImageType3D::Pointer structural_used= resampleFilter3->GetOutput();
            itk::ImageRegionIterator<ImageType3D> it(structural_used,structural_used->GetLargestPossibleRegion());
            for(it.GoToBegin(); !it.IsAtEnd(); ++it)
            {
                if(it.Get()<0)
                    it.Set(0);
            }
            structural_imgs.push_back(structural_used);

            char dummy_name[100]={0};
            if(Nstr>1)
                sprintf(dummy_name,"/structural_used_%d.nii",str);
            else
                sprintf(dummy_name,"/structural_used.nii");
            std::string new_str_name= this->proc_folder + std::string(dummy_name);
            writeImageD<ImageType3D>(structural_used,new_str_name);
        }
    }
}


void EPIREG::Step2_DiffeoRegistration()
{    
    this->b0_up_quad=readImageD<ImageType3D>(proc_folder+"/blip_up_b0_quad.nii");


    int Nstr= parser->getNumberOfStructurals();
    for(int str=0;str<Nstr;str++)
    {
        {
            char dummy_name[100]={0};
            if(Nstr>1)
                sprintf(dummy_name,"/structural_used_%d.nii",str);
            else
                sprintf(dummy_name,"/structural_used.nii");
            std::string new_str_name= this->proc_folder + std::string(dummy_name);

            ImageType3D::Pointer str_img =readImageD<ImageType3D>(new_str_name);
            structural_imgs.push_back(str_img);
        }
    }

    vnl_vector<double> phase_vector(3,0);
    if(this->PE_string=="vertical")
        phase_vector[1]=1;
    if(this->PE_string=="horizontal")
        phase_vector[0]=1;
    if(this->PE_string=="slice")
        phase_vector[2]=1;
    // The phase vector stays in index space: the metrics apply the direction matrix.


    std::vector<DRBUDDIStageSettings> stages;
    stages.resize(6);
    {
        stages[0].niter=200;
        stages[0].img_smoothing_std=3.;
        stages[0].downsample_factor=8;
        stages[0].learning_rate=0.25;
        stages[0].update_gaussian_sigma=9.;
        stages[0].total_gaussian_sigma=0.25;
        stages[0].restrct=1;
        stages[0].constrain=0;
        DRBUDDIMetric metric;
        metric.SetMetricType(DRBUDDIMetricEnumeration::CC);
        metric.weight=1;
        stages[0].metrics.push_back(metric);
    }
    {
        stages[1].niter=200;
        stages[1].img_smoothing_std=2.;
        stages[1].downsample_factor=6;
        stages[1].learning_rate=0.4;
        stages[1].update_gaussian_sigma=9.;
        stages[1].total_gaussian_sigma=0.25;
        stages[1].restrct=1;
        stages[1].constrain=0;
        DRBUDDIMetric metric;
        metric.SetMetricType(DRBUDDIMetricEnumeration::CC);
        metric.weight=1;
        stages[1].metrics.push_back(metric);
    }
    {
        stages[2].niter=200;
        stages[2].img_smoothing_std=2.;
        stages[2].downsample_factor=4;
        stages[2].learning_rate=0.5;
        stages[2].update_gaussian_sigma=9.;
        stages[2].total_gaussian_sigma=0.25;
        stages[2].restrct=1;
        stages[2].constrain=0;
        DRBUDDIMetric metric;
        metric.SetMetricType(DRBUDDIMetricEnumeration::CC);
        metric.weight=1;
        stages[2].metrics.push_back(metric);
    }
    {
        stages[3].niter=200;
        stages[3].img_smoothing_std=1.;
        stages[3].downsample_factor=2;
        stages[3].learning_rate=0.75;
        stages[3].update_gaussian_sigma=9.;
        stages[3].total_gaussian_sigma=0.25;
        stages[3].restrct=1;
        stages[3].constrain=0;
        DRBUDDIMetric metric;
        metric.SetMetricType(DRBUDDIMetricEnumeration::CC);
        metric.weight=1;
        stages[3].metrics.push_back(metric);
    }
    {
        stages[4].niter=200;
        stages[4].img_smoothing_std=0.;
        stages[4].downsample_factor=1;
        stages[4].learning_rate=1.25;
        stages[4].update_gaussian_sigma=9.;
        stages[4].total_gaussian_sigma=0.25;
        stages[4].restrct=1;
        stages[4].constrain=0;
        DRBUDDIMetric metric;
        metric.SetMetricType(DRBUDDIMetricEnumeration::CC);
        metric.weight=1;
        stages[4].metrics.push_back(metric);
    }
    {
        stages[5].niter=20;
        stages[5].img_smoothing_std=0.;
        stages[5].downsample_factor=1;
        stages[5].learning_rate=0.75;
        stages[5].update_gaussian_sigma=9.;
        stages[5].total_gaussian_sigma=0.25;
        stages[5].restrct=0;
        stages[5].constrain=0;
        DRBUDDIMetric metric;        
        metric.SetMetricType(DRBUDDIMetricEnumeration::CC);
        metric.weight=1;
        stages[5].metrics.push_back(metric);

    }



    DRBUDDI_Diffeo *myEPIREG_processor = new DRBUDDI_Diffeo;
    myEPIREG_processor->SetB0UpImage(this->b0_up_quad);
    myEPIREG_processor->SetB0DownImage(this->b0_up_quad);
    myEPIREG_processor->SetFAUpImage(this->b0_up_quad);
    myEPIREG_processor->SetFADownImage(structural_imgs[0]);


    myEPIREG_processor->SetStructuralImages(structural_imgs);
    myEPIREG_processor->SetUpPEVector(phase_vector);
    myEPIREG_processor->SetDownPEVector(phase_vector);
    myEPIREG_processor->SetParser(parser);
    if(parser->getNumberOfStages()==0)
        myEPIREG_processor->SetStagesFromExternal(stages);
    else
        (*stream)<<"Using the "<<parser->getNumberOfStages()<<" --DRBUDDI_stage settings for the EPI registration instead of the built-in schedule"<<std::endl;
    std::string init_field_name = parser->GetEPIREGInitialField();
    if(init_field_name!="")
    {
        (*stream)<<"Initializing the EPI registration with "<<init_field_name<<std::endl;
        DisplacementFieldType::Pointer init_field= readImageD<DisplacementFieldType>(init_field_name);
        {
            // Onto the quad grid the registration runs on. The filter's default linear
            // interpolator handles vector pixels; drbuddi_image_utilities' helper is not
            // compiled into the CUDA target.
            using VecResampleType= itk::ResampleImageFilter<DisplacementFieldType, DisplacementFieldType>;
            VecResampleType::Pointer resampler= VecResampleType::New();
            resampler->SetOutputParametersFromImage(this->b0_up_quad);
            resampler->SetInput(init_field);
            DisplacementFieldType::PixelType zero; zero.Fill(0);
            resampler->SetDefaultPixelValue(zero);
            resampler->Update();
            init_field= resampler->GetOutput();
        }
        myEPIREG_processor->SetInitialFieldsFromExternal(init_field, nullptr);
    }
    myEPIREG_processor->Process();

    this->def_FINV=myEPIREG_processor->getUp2DownINV();

    delete myEPIREG_processor;
}

void EPIREG::Step3_WriteOutput()
{
    (*stream)<<"Writing EPIREG output files..."<<std::endl;


    writeImageD<DisplacementFieldType>(this->def_FINV,proc_folder+"/deformation_FINV.nii.gz");

    ImageType3D::Pointer b0_up_corrected=nullptr;

    using ResampleImageFilterType = itk::ResampleImageFilter<ImageType3D, ImageType3D> ;
    {
        DisplacementFieldTransformType::Pointer trans = DisplacementFieldTransformType::New();
        trans->SetDisplacementField(this->def_FINV);
        ResampleImageFilterType::Pointer resampleFilter3 = ResampleImageFilterType::New();
        resampleFilter3->SetOutputParametersFromImage(this->b0_up_quad);;
        resampleFilter3->SetInput(this->b0_up_quad);
        resampleFilter3->SetTransform(trans);
        resampleFilter3->Update();
        b0_up_corrected= resampleFilter3->GetOutput();
        writeImageD<ImageType3D>(b0_up_corrected,proc_folder+"/blip_up_b0_corrected.nii");
    }

    ImageType3D::Pointer b0_up_corrected_JAC= JacobianTransformImage(b0_up_quad,def_FINV, b0_up_quad);
    writeImageD<ImageType3D>(b0_up_corrected_JAC,proc_folder+"/blip_up_b0_corrected_JAC.nii");

}



#endif

