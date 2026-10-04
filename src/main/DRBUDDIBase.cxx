#ifndef _DRBUDDIBase_CXX
#define _DRBUDDIBase_CXX


#include "DRBUDDIBase.h"
#include "../utilities/read_bmatrix_file.h"
#include "../utilities/read_3Dvolume_from_4D.h"
#include "registration_settings.h"

#include "../tools/EstimateTensor/estimate_tensor_wlls.h"
#include "create_mask.h"
#include "../tools/ComputeFAMap/compute_fa_map.h"

#include "rigid_register_images.h"
#include "../tools/ResampleDWIs/resample_dwis.h"
#include "itkImageToHistogramFilter.h"
#include "itkIntensityWindowingImageFilter.h"
#include "itkHistogramMatchingImageFilter.h"

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
#include "../tools/EstimateMAPMRI/MAPMRIModel.h"



void DRBUDDIBase::Process()
{
    Step0_CreateImages();
    Step1_RigidRegistration();
    Step2_DiffeoRegistration();
    Step3_WriteOutput();
}


void DRBUDDIBase::CreateBlipUpQuadImage()
{
    std::vector<std::string> str_names= parser->getStructuralNames();
    ImageType3D::SpacingType new_spacing;

    if(str_names.size())
    {
        ImageType3D::Pointer str_img = readImageD<ImageType3D>(str_names[0]);
        new_spacing = str_img->GetSpacing();
    }
    else
    {
        new_spacing= b0_up->GetSpacing();
    }
    float epi_working_res = RegistrationSettings::get().getValue<float>("epi_working_res");  // QSIPREP_EPIRES
    if(epi_working_res > 0)
    {
        // Explicit isotropic working grid. Set directly rather than iterating the
        // default rule below: that rule only ever REFINES spacing, so a coarser
        // target (qsiprep --sloppy) would be ignored whenever the structural is
        // already sub-millimetre -- which it is for the T2w used here.
        new_spacing[0]=epi_working_res;
        new_spacing[1]=epi_working_res;
        new_spacing[2]=epi_working_res;
        (*stream)<<"Using requested EPI working resolution "<<epi_working_res<<" mm"<<std::endl;
    }
    else
    {
        float avg_spacing = (new_spacing[0] + new_spacing[1] + new_spacing[2])/3.;
        while(avg_spacing>1)
        {
            new_spacing=new_spacing/1.3;
            avg_spacing = (new_spacing[0] + new_spacing[1] + new_spacing[2])/3.;
        }
    }

    ImageType3D::SizeType new_size;
    new_size[0] = (int)ceil(1.0*b0_up->GetLargestPossibleRegion().GetSize()[0] * b0_up->GetSpacing()[0]/new_spacing[0]);
    new_size[1] = (int)ceil(1.0*b0_up->GetLargestPossibleRegion().GetSize()[1] * b0_up->GetSpacing()[1]/new_spacing[1]);
    new_size[2] = (int)ceil(1.0*b0_up->GetLargestPossibleRegion().GetSize()[2] * b0_up->GetSpacing()[2]/new_spacing[2]);
    ImageType3D::IndexType start; start.Fill(0);
    ImageType3D::RegionType nreg(start,new_size);

    ImageType3D::SpacingType new_spc;
    new_spc[0]=  1.0*b0_up->GetLargestPossibleRegion().GetSize()[0] * b0_up->GetSpacing()[0] / new_size[0];
    new_spc[1]=  1.0*b0_up->GetLargestPossibleRegion().GetSize()[1] * b0_up->GetSpacing()[1] / new_size[1];
    new_spc[2]=  1.0*b0_up->GetLargestPossibleRegion().GetSize()[2] * b0_up->GetSpacing()[2] / new_size[2];

    itk::ContinuousIndex<double,3> ind;
    ind[0]=-0.5;
    ind[1]=-0.5;
    ind[2]=-0.5;
    ImageType3D::PointType pt;
    this->b0_up->TransformContinuousIndexToPhysicalPoint(ind,pt);

    vnl_vector<double> vec(3);
    vec[0]= ind[0]* new_spc[0];
    vec[1]= ind[1]* new_spc[1];
    vec[2]= ind[2]* new_spc[2];
    vnl_vector<double> nvec= b0_up->GetDirection().GetVnlMatrix() * vec;
    ImageType3D::PointType new_orig;
    new_orig[0]=pt[0]-nvec[0];
    new_orig[1]=pt[1]-nvec[1];
    new_orig[2]=pt[2]-nvec[2];

    ImageType3D::Pointer b0_up_ref_img =  ImageType3D::New();
    b0_up_ref_img->SetDirection(b0_up->GetDirection());
    b0_up_ref_img->SetOrigin(new_orig);
    b0_up_ref_img->SetSpacing(new_spc);
    b0_up_ref_img->SetRegions(nreg);

    int pad=16;
    //if(parser->getDisableInitRigid())
      //  pad=0;

    ind[0]=-pad/2;
    ind[1]=-pad/2;
    ind[2]=0;
    b0_up_ref_img->TransformContinuousIndexToPhysicalPoint(ind,pt);

    new_size[0]=new_size[0]+pad;
    new_size[1]=new_size[1]+pad;
    new_size[2]=new_size[2];
    nreg.SetSize(new_size);
    nreg.SetIndex(start);

    b0_up_ref_img =  ImageType3D::New();
    b0_up_ref_img->SetDirection(b0_up->GetDirection());
    b0_up_ref_img->SetOrigin(pt);
    b0_up_ref_img->SetSpacing(new_spc);
    b0_up_ref_img->SetRegions(nreg);

    itk::IdentityTransform<double,3>::Pointer  id=itk::IdentityTransform<double,3>::New();
    id->SetIdentity();

    using InterpolatorType= itk::BSplineInterpolateImageFunction<ImageType3D,double,double>;
    InterpolatorType::Pointer interp=InterpolatorType::New();
    interp->SetSplineOrder(3);


    using ResampleImageFilterType = itk::ResampleImageFilter<ImageType3D, ImageType3D> ;
    ResampleImageFilterType::Pointer resampleFilter2 = ResampleImageFilterType::New();
    resampleFilter2->SetOutputParametersFromImage(b0_up_ref_img);
    resampleFilter2->SetInput(b0_up);
    resampleFilter2->SetTransform(id);
    resampleFilter2->SetInterpolator(interp);
    resampleFilter2->Update();
    this->b0_up_quad= resampleFilter2->GetOutput();



    itk::ImageRegionIterator<ImageType3D> it(this->b0_up_quad,this->b0_up_quad->GetLargestPossibleRegion());
    for(it.GoToBegin(); !it.IsAtEnd(); ++it)
    {
        if(it.Get()<0)
            it.Set(0);
    }
}

ImageType3D::Pointer DRBUDDIBase::JacobianTransformImage(ImageType3D::Pointer img,DisplacementFieldType::Pointer field,ImageType3D::Pointer ref_img)
{
    DisplacementFieldTransformType::Pointer disp_trans = DisplacementFieldTransformType::New();
    disp_trans->SetDisplacementField(field);

    using InterpolatorType = itk::BSplineInterpolateImageFunction<ImageType3D,double>;
    InterpolatorType::Pointer interpolator = InterpolatorType::New();
    interpolator->SetSplineOrder(3);

    using ResampleImageFilterType = itk::ResampleImageFilter<ImageType3D, ImageType3D> ;
    ResampleImageFilterType::Pointer resampleFilter2 = ResampleImageFilterType::New();
    resampleFilter2->SetOutputParametersFromImage(ref_img);
  //  resampleFilter2->SetInterpolator(interpolator);
    resampleFilter2->SetInput(img);
    resampleFilter2->SetTransform(disp_trans);
    resampleFilter2->Update();
    ImageType3D::Pointer trans_img= resampleFilter2->GetOutput();

    vnl_vector<double> phase_vector(3,0);
    if(this->PE_string=="vertical")
        phase_vector[1]=1;
    if(this->PE_string=="horizontal")
        phase_vector[0]=1;
    if(this->PE_string=="slice")
        phase_vector[2]=1;
    phase_vector= this->b0_up->GetDirection().GetVnlMatrix() * phase_vector;
    int phase_id=0;
    if(  (fabs(phase_vector[1])>fabs(phase_vector[0])) &&  (fabs(phase_vector[1])>fabs(phase_vector[0])) )
          phase_id=1;
    if(  (fabs(phase_vector[2])>fabs(phase_vector[0])) &&  (fabs(phase_vector[2])>fabs(phase_vector[1])) )
          phase_id=2;

    itk::ImageRegionIteratorWithIndex<ImageType3D> it(trans_img, trans_img->GetLargestPossibleRegion());
    it.GoToBegin();
    while(!it.IsAtEnd())
    {
        ImageType3D::IndexType index = it.GetIndex();
        double det= ComputeJacobianDetAtIndex(field,index,phase_id);
        if(det>0)
        {
            it.Set(it.Get()*det);
        }
        else
        {
            ImageType3D::IndexType tind=index;
            int Ntot=0;
            double tot=0;
            for(int k=-1;k<=1;k++)
            {
                tind[2]=index[2]+k;
                for(int j=-2;j<=2;j++)
                {
                    tind[1]=index[1]+j;
                    for(int i=-2;i<=2;i++)
                    {
                        tind[0]=index[0]+i;
                        double det2=  ComputeJacobianDetAtIndex(field,tind,phase_id);
                        if(det2>0)
                        {
                            Ntot++;
                            tot+=det2;
                        }
                    }
                }
            }
            if(Ntot>0)
            {
                det=tot/Ntot;
                it.Set(it.Get()*det);
            }
            else
                it.Set(0);
        }
        ++it;
    }
    return trans_img;
}

double  DRBUDDIBase::ComputeJacobianDetAtIndex(DisplacementFieldType::Pointer disp_field, DisplacementFieldType::IndexType index, int phase)
{
    const int h=1;
    if(index[phase]<h || index[phase]> disp_field->GetLargestPossibleRegion().GetSize()[phase]-h-1)
        return 1.;

    ImageType3D::SpacingType d_spc= disp_field->GetSpacing();

    vnl_vector<double> phase_vec(3,0);
    phase_vec[phase]=1;
    vnl_vector<double> new_phase = disp_field->GetDirection().GetVnlMatrix()*phase_vec;

    int phase_xyz;
    if( (fabs(new_phase[0]) > fabs(new_phase[1]))  && (fabs(new_phase[0]) > fabs(new_phase[2])))
        phase_xyz=0;
    else if( (fabs(new_phase[1]) > fabs(new_phase[0]))  && (fabs(new_phase[1]) > fabs(new_phase[2])))
        phase_xyz=1;
    else phase_xyz=2;


    double grad;

    ImageType3D::IndexType tind3=index;
    tind3[phase]+=1;
    double valp=disp_field->GetPixel(tind3)[phase] ;
    tind3[phase]-=2;
    double valm=disp_field->GetPixel(tind3)[phase] ;


    grad=0.5*(valp-valm)/d_spc[phase]/h;

    vnl_vector<double> temp(3,0);
    temp[phase]=grad;
    vnl_vector<double> temp2= disp_field->GetDirection().GetVnlMatrix() * temp;


    //return  temp2[phase_xyz];
     return 1+ temp2[phase_xyz];

}

InternalMatrixType DRBUDDIBase::ComputeJacobianAtIndex(DisplacementFieldType::Pointer disp_field, DisplacementFieldType::IndexType index)
{
    InternalMatrixType A;
    A.set_identity();

    if(index[0]<=0 || index[0]>= disp_field->GetLargestPossibleRegion().GetSize()[0]-1)
        return A;

    if(index[1]<=0 || index[1]>= disp_field->GetLargestPossibleRegion().GetSize()[1]-1)
        return A;

    if(index[2]<=0 || index[2]>= disp_field->GetLargestPossibleRegion().GetSize()[2]-1)
        return A;

    bool do_second_order=false;
    if(index[0]==1 || index[0]== disp_field->GetLargestPossibleRegion().GetSize()[0]-2)
        do_second_order=false;
    if(index[1]==1 || index[1]== disp_field->GetLargestPossibleRegion().GetSize()[1]-2)
        do_second_order=false;
    if(index[2]==1 || index[2]== disp_field->GetLargestPossibleRegion().GetSize()[2]-2)
        do_second_order=false;

    if(do_second_order)
    {
        for(int dim=0;dim<3;dim++)   // derivative w.r.t.
        {
            DisplacementFieldType::IndexType rind=index;
            DisplacementFieldType::IndexType lind=index;
            DisplacementFieldType::IndexType rrind=index;
            DisplacementFieldType::IndexType llind=index;
            rind[dim]++;
            lind[dim]--;
            rrind[dim]+=2;
            llind[dim]-=2;

            DisplacementFieldType::PixelType lval = disp_field->GetPixel(lind);
            DisplacementFieldType::PixelType rval = disp_field->GetPixel(rind);
            DisplacementFieldType::PixelType llval = disp_field->GetPixel(llind);
            DisplacementFieldType::PixelType rrval = disp_field->GetPixel(rrind);

            DisplacementFieldType::PixelType deriv= (-rrval+8.*rval-8.*lval+llval)/12./disp_field->GetSpacing()[dim];

            A.set_column(dim,deriv.GetVnlVector());
        }
    }
    else
    {
        for(int dim=0;dim<3;dim++)   // derivative w.r.t.
        {
            DisplacementFieldType::IndexType rind=index;
            DisplacementFieldType::IndexType lind=index;
            rind[dim]++;
            lind[dim]--;

            DisplacementFieldType::PixelType lval = disp_field->GetPixel(lind);
            DisplacementFieldType::PixelType rval = disp_field->GetPixel(rind);

            DisplacementFieldType::PixelType deriv= 0.5*(rval-lval)/disp_field->GetSpacing()[dim];

            A.set_column(dim,deriv.GetVnlVector());
        }
    }

    vnl_vector<double> phys_vec(3);
    phys_vec= disp_field->GetDirection().GetVnlMatrix()*A.get_row(0);
    A.set_row(0,phys_vec);
    phys_vec= disp_field->GetDirection().GetVnlMatrix()*A.get_row(1);
    A.set_row(1,phys_vec);
    phys_vec= disp_field->GetDirection().GetVnlMatrix()*A.get_row(2);
    A.set_row(2,phys_vec);
    A(0,0)+=1;
    A(1,1)+=1;
    A(2,2)+=1;


    return A;
}

void DRBUDDIBase::CreateCorrectionImage(std::string nii_filename,ImageType3D::Pointer &b0_img, ImageType3D::Pointer &FA_img)
{
    std::string bmtxt_name= nii_filename.substr(0,nii_filename.rfind(".nii"))+std::string(".bmtxt");
    vnl_matrix<double> Bmatrix= read_bmatrix_file(bmtxt_name);
    int Nvols= Bmatrix.rows();

     bool use_tensor=true;
     int ndwi=0;
     vnl_vector<double> bvals = Bmatrix.get_column(0)+Bmatrix.get_column(3)+Bmatrix.get_column(5);
     for(int v=0;v<bvals.size();v++)
         if(bvals[v]>200)
             ndwi++;
     if(ndwi<12)
         use_tensor=false;
     vnl_svd<double> msvd(Bmatrix);
     use_tensor=true;
     vnl_diag_matrix<double> W = msvd.W();
     for(int v=0;v<W.cols();v++)
     {
         if(W(v,v)<1E-50)
         {
             use_tensor=false;
             break;
         }
     }
     if(use_tensor)
     {
         (*stream)<<"TENSOR fitting data for b=0 image generation!!"<<std::endl;
     }
     else
     {
         (*stream)<<"NOT USING TENSOR fitting for b=0 image generation!!"<<std::endl;
     }



     std::string inc_name= nii_filename.substr(0,nii_filename.rfind(".nii"))+std::string("_inc.nii");
     std::vector<ImageType3DBool::Pointer> final_inclusion_imgs;
     std::vector<ImageType3D::Pointer> final_data;
     final_data.resize(Nvols);     
     if(Nvols==1)
     {
         final_data[0]=readImageD<ImageType3D>(nii_filename);
     }
     else
     {
         for(int v=0;v<Nvols;v++)
         {
             final_data[v]=read_3D_volume_from_4D(nii_filename,v);
         }
     }

     if(fs::exists(inc_name))
     {
         final_inclusion_imgs.resize(Nvols);
         for(int v=0;v<Nvols;v++)
         {
             final_inclusion_imgs[v]=read_3D_volume_from_4DBool(inc_name,v);
         }
     }



     // Non-shelled (CS-DSI) data: the direct tensor fit below is invalid, so fit
     // MAPMRI and synthesize a tensor-fittable shell to derive b=0/FA from.
     // Called once per phase-encoding direction by DRBUDDI::Step0_CreateImages().
     float synth_bval = RegistrationSettings::get().getValue<float>("DRBUDDI_synth_shell_bval");
     if(synth_bval > 0)
     {
         int ndirs = RegistrationSettings::get().getValue<int>("DRBUDDI_synth_shell_ndirs");
         if(ndirs < 6)
             ndirs = 30;
         (*stream)<<"Synthesizing a b="<<synth_bval<<" shell ("<<ndirs<<" directions) with MAPMRI for the DRBUDDI target..."<<std::endl;

         ImageType3D::Pointer mask_img = main_mask_img ? main_mask_img : create_mask(final_data[0]);

         // The MAPMRI fit needs a DTI initializer built from the low-b volumes
         // only; using the whole grid is the very thing we are avoiding.
         std::vector<int> dt_indices, all_indices;
         for(int v=0;v<bvals.size();v++)
         {
             all_indices.push_back(v);
             if(bvals[v] <= 1.05*synth_bval)
                 dt_indices.push_back(v);
         }
         dt_indices = all_indices;   // QSIPREP_ALLVOL: see comment above

         float small_delta = RegistrationSettings::get().getValue<float>("small_delta");
         float big_delta   = RegistrationSettings::get().getValue<float>("big_delta");
         if(small_delta <= 0 || big_delta <= 0)
         {
             // DIFFPREP's fallback when the JSON carries no timings.
             double max_bval = bvals.max_value();
             double gyro = 267.51532*1E6;
             double G = 40*1E-3; G *= 2;
             double temp = max_bval/gyro/gyro/G/G/2.*1E6;
             small_delta = pow(temp,1./3.)*1000.;
             big_delta   = small_delta*3;
             (*stream)<<"heuristic deltas "<<small_delta<<"/"<<big_delta<<" ms"<<std::endl;
         }

         MAPMRIModel mapmri;
         mapmri.SetMAPMRIDegree(4);
         mapmri.SetBmatrix(Bmatrix);
         mapmri.SetDWIData(final_data);
         mapmri.SetMaskImage(mask_img);
         mapmri.SetVolIndicesForFitting(all_indices);
         mapmri.SetDTIIndices(dt_indices);
         mapmri.SetSmallDelta(small_delta);
         mapmri.SetBigDelta(big_delta);
         mapmri.PerformFitting();

         // A deterministic, near-uniform set of directions (golden-angle spiral),
         // plus one b=0, forms a clean single-shell acquisition.
         std::vector<ImageType3D::Pointer> synth_data;
         vnl_matrix<double> synth_bmat(ndirs+1, 6, 0.0);

         vnl_vector<double> b0row(6, 0.0);
         synth_data.push_back(mapmri.SynthesizeDWI(b0row));

         const double ga = M_PI * (3.0 - sqrt(5.0));
         for(int d=0; d<ndirs; d++)
         {
             double z = 1.0 - 2.0*(d + 0.5)/ndirs;
             double r = sqrt(std::max(0.0, 1.0 - z*z));
             double th = ga * d;
             double gx = r*cos(th), gy = r*sin(th), gz = z;

             vnl_vector<double> row(6, 0.0);
             row[0]= synth_bval*gx*gx;   row[1]= 2*synth_bval*gx*gy;
             row[2]= 2*synth_bval*gx*gz; row[3]= synth_bval*gy*gy;
             row[4]= 2*synth_bval*gy*gz; row[5]= synth_bval*gz*gz;
             synth_bmat.set_row(d+1, row);
             synth_data.push_back(mapmri.SynthesizeDWI(row));
         }

         std::vector<int> synth_fit_indices;
         for(int v=0;v<(int)synth_data.size();v++)
             synth_fit_indices.push_back(v);

         std::vector<ImageType3DBool::Pointer> no_inclusion;
         ImageType3D::Pointer synth_A0=nullptr;
         DTImageType::Pointer synth_dt = EstimateTensorWLLS_sub_nomm(synth_data, synth_bmat, synth_fit_indices, synth_A0, nullptr, no_inclusion);
         FA_img = compute_fa_map(synth_dt);

         itk::ImageRegionIteratorWithIndex<ImageType3D> sit(synth_A0, synth_A0->GetLargestPossibleRegion());
         for(sit.GoToBegin(); !sit.IsAtEnd(); ++sit)
         {
             ImageType3D::IndexType ind3 = sit.GetIndex();
             float val = sit.Get();
             if(val!=val || val<0)
                 sit.Set(0);
             if(mask_img->GetPixel(ind3)==0)
                 FA_img->SetPixel(ind3,0);
         }
         b0_img = synth_A0;
         return;
     }

     if(use_tensor)
     {
         int DWI_bval = RegistrationSettings::get().getValue<int>("DRBUDDI_DWI_bval_tensor_fitting");

         std::vector<int> dummy;
         if(DWI_bval==0)
         {
             for(int i=0;i<bvals.size();i++)
             {
                 dummy.push_back(i);
             }
         }
         else
         {
             for(int i=0;i<bvals.size();i++)
             {
                 if(bvals[i]<= 1.05*DWI_bval)
                     dummy.push_back(i);
             }
         }

         //int b0_vol_id = my_json["B0VolId"];
         ImageType3D::Pointer mask_img= nullptr;
         if(main_mask_img)
             mask_img=main_mask_img;
         else
             mask_img= create_mask(final_data[0]);
         ImageType3D::Pointer A0_image=nullptr;
         DTImageType::Pointer  dt_image=nullptr;
         if(dummy.size()<15)
             dummy.resize(0);
         dt_image= EstimateTensorWLLS_sub_nomm(final_data,Bmatrix,dummy,A0_image,nullptr,final_inclusion_imgs);
         FA_img = compute_fa_map(dt_image);


         itk::ImageRegionIteratorWithIndex<ImageType3D> mit(A0_image,A0_image->GetLargestPossibleRegion());
         mit.GoToBegin();
         while(!mit.IsAtEnd())
         {
             ImageType3D::IndexType ind3= mit.GetIndex();
             float val =mit.Get();
             if(val!=val || val<0)
             {
                 mit.Set(0);
             }
             if(mask_img->GetPixel(ind3)==0)
                 FA_img->SetPixel(ind3,0);

             ++mit;
         }

         b0_img=A0_image;

     }
     else
     {
         FA_img=nullptr;

         float b0bval = bvals.min_value();
         std::vector<int> b0_indices;
         for(int v=0;v<bvals.size();v++)
         {
             if(bvals[v]<=1.05*b0bval)
             {
                 b0_indices.push_back(v);
             }
         }

         b0_img= final_data[b0_indices[0]];
         for(int v=1;v<b0_indices.size();v++)
         {
             ImageType3D::Pointer im2= final_data[b0_indices[v]];

             typedef itk::AddImageFilter<ImageType3D,ImageType3D,ImageType3D> AdderType;
             AdderType::Pointer adder= AdderType::New();
             adder->SetInput1(b0_img);
             adder->SetInput2(im2);
             adder->Update();
             b0_img= adder->GetOutput();
         }
         if(b0_indices.size()>1)
         {
             typedef itk::DivideImageFilter<ImageType3D,ImageType3D,ImageType3D> DividerType;
             DividerType::Pointer divider= DividerType::New();
             divider->SetInput1(b0_img);
             divider->SetConstant(1.*b0_indices.size());
             divider->Update();
             b0_img=divider->GetOutput();
         }

         itk::ImageRegionIterator<ImageType3D> it(b0_img,b0_img->GetLargestPossibleRegion());
         it.GoToBegin();
         while(!it.IsAtEnd())
         {
             if(it.Get()<0)
                 it.Set(0);
             ++it;
         }
     }
}




#endif


ImageType3D::Pointer DRBUDDIBase::PreprocessImage(  ImageType3D::ConstPointer  inputImage,
                                              ImageType3D::PixelType lowerScaleValue,
                                              ImageType3D::PixelType upperScaleValue,
                                              float winsorizeLowerQuantile, float winsorizeUpperQuantile,
                                              ImageType3D::ConstPointer histogramMatchSourceImage )
{
    typedef itk::Statistics::ImageToHistogramFilter<ImageType3D>   HistogramFilterType;
    typedef  HistogramFilterType::InputBooleanObjectType InputBooleanObjectType;
    typedef  HistogramFilterType::HistogramSizeType      HistogramSizeType;
    typedef  HistogramFilterType::HistogramType          HistogramType;

    HistogramSizeType histogramSize( 1 );
    histogramSize[0] = 256;

    InputBooleanObjectType::Pointer autoMinMaxInputObject = InputBooleanObjectType::New();
    autoMinMaxInputObject->Set( true );

    HistogramFilterType::Pointer histogramFilter = HistogramFilterType::New();
    histogramFilter->SetInput( inputImage );
    histogramFilter->SetAutoMinimumMaximumInput( autoMinMaxInputObject );
    histogramFilter->SetHistogramSize( histogramSize );
    histogramFilter->SetMarginalScale( 10.0 );
    histogramFilter->Update();

    float lowerValue = histogramFilter->GetOutput()->Quantile( 0, winsorizeLowerQuantile );
    float upperValue = histogramFilter->GetOutput()->Quantile( 0, winsorizeUpperQuantile );

    typedef itk::IntensityWindowingImageFilter<ImageType3D, ImageType3D> IntensityWindowingImageFilterType;

    IntensityWindowingImageFilterType::Pointer windowingFilter = IntensityWindowingImageFilterType::New();
    windowingFilter->SetInput( inputImage );
    windowingFilter->SetWindowMinimum( lowerValue );
    windowingFilter->SetWindowMaximum( upperValue );
    windowingFilter->SetOutputMinimum( lowerScaleValue );
    windowingFilter->SetOutputMaximum( upperScaleValue );
    windowingFilter->Update();

    ImageType3D::Pointer outputImage = nullptr;
    if( histogramMatchSourceImage )
    {
        typedef itk::HistogramMatchingImageFilter<ImageType3D, ImageType3D> HistogramMatchingFilterType;
        HistogramMatchingFilterType::Pointer matchingFilter = HistogramMatchingFilterType::New();
        matchingFilter->SetSourceImage( windowingFilter->GetOutput() );
        matchingFilter->SetReferenceImage( histogramMatchSourceImage );
        matchingFilter->SetNumberOfHistogramLevels( 256 );
        matchingFilter->SetNumberOfMatchPoints( 12 );
        matchingFilter->ThresholdAtMeanIntensityOn();
        matchingFilter->Update();

        outputImage = matchingFilter->GetOutput();
        outputImage->Update();
        outputImage->DisconnectPipeline();
    }
    else
    {
        outputImage = windowingFilter->GetOutput();
        outputImage->Update();
        outputImage->DisconnectPipeline();
    }
    return outputImage;
}


DRBUDDIBase::RigidTransformType::Pointer DRBUDDIBase::RegisterStructuralToB0(ImageType3D::Pointer b0_img, ImageType3D::Pointer str_img_orig)
{
    ImageType3D::Pointer str_img= create_mask(str_img_orig);

    {
        itk::ImageRegionIteratorWithIndex<ImageType3D> it(str_img,str_img->GetLargestPossibleRegion());
        for(it.GoToBegin();!it.IsAtEnd();++it)
        {
            ImageType3D::IndexType ind3= it.GetIndex();
            it.Set(it.Get()* str_img_orig->GetPixel(ind3)*5 + str_img_orig->GetPixel(ind3) );
        }
    }

    str_img=PreprocessImage(str_img,0,1,0,1);
    b0_img=PreprocessImage(b0_img,0,1,0,1);

    RigidTransformType::Pointer rigid_trans1= RigidRegisterImagesEuler( b0_img,  str_img, "CC",parser->getRigidLR());
    RigidTransformType::Pointer rigid_trans2= RigidRegisterImagesEuler( b0_img,  str_img,"MI",parser->getRigidLR());

    auto params1= rigid_trans1->GetParameters();
    auto params2= rigid_trans2->GetParameters();
    auto p1=params1-params2;

    double diff=0;
    diff+= p1[0]*p1[0] + p1[1]*p1[1] +  p1[2]*p1[2] +
            p1[3]*p1[3]/400. + p1[4]*p1[4]/400. + p1[5]*p1[5]/400. ;

    RigidTransformType::Pointer rigid_trans=nullptr;
    (*stream)<<"R1: "<< params1<<std::endl;
    (*stream)<<"R2: "<< params2<<std::endl;
    (*stream)<<"MI vs CC diff: "<< diff<<" (tolerance "<<parser->getStructuralRigidTolerance()<<")"<<std::endl;
    if(diff<parser->getStructuralRigidTolerance())
        rigid_trans=rigid_trans2;
    else
    {
        (*stream)<<"Could not compute the rigid transformation from the structural imageto b=0 image... Starting multistart.... This could take a while"<<std::endl;
        (*stream)<<"Better be safe than sorry, right?"<<std::endl;

        RigidTransformType::Pointer rigid_trans1a= RigidRegisterImagesEuler( str_img, b0_img,  "CC",parser->getRigidLR(),false);
        RigidTransformType::ParametersType b1= rigid_trans1a->GetParameters();

        p1[0]= params1[0]+ b1[0];
        p1[1]= params1[1]+ b1[1];
        p1[2]= params1[2]+ b1[2];

        double diff1= p1[0]*p1[0] + p1[1]*p1[1] +  p1[2]*p1[2] ;
        RigidTransformType::Pointer rigid_trans2a= RigidRegisterImagesEuler( str_img, b0_img,  "MI",parser->getRigidLR(),false);
        RigidTransformType::ParametersType b2= rigid_trans2a->GetParameters();

        (*stream)<< "Trans CC F" << rigid_trans1->GetParameters()<<std::endl;
        (*stream)<< "Trans CC B" << rigid_trans1a->GetParameters()<<std::endl;
        (*stream)<< "Trans MI F" << rigid_trans2->GetParameters()<<std::endl;
        (*stream)<< "Trans MI B" << rigid_trans2a->GetParameters()<<std::endl;


        p1[0]= params2[0]+ b2[0];
        p1[1]= params2[1]+ b2[1];
        p1[2]= params2[2]+ b2[2];

        double diff2= p1[0]*p1[0] + p1[1]*p1[1] +  p1[2]*p1[2] ;
        (*stream)<< "diff1 "<<diff1 << " diff2 " <<diff2 <<std::endl;

        std::string new_metric_type="MI";
        if(diff1 < diff2)
        {
            (*stream)<< "CC was determined to be more robust than MI. Switching..."<<std::endl;
            new_metric_type="CC";
        }

        if(diff1<0.001)
        {
            b1[0]= (params1[0] - b1[0])/2.;
            b1[1]= (params1[1] - b1[1])/2.;
            b1[2]= (params1[2] - b1[2])/2.;
            b1[3]= (params1[3] );
            b1[4]= (params1[4] );
            b1[5]= (params1[5] );
            rigid_trans1->SetParameters(b1);

            rigid_trans= RigidRegisterImagesEuler( b0_img,  str_img, "CC",parser->getRigidLR(),true, rigid_trans1);
        }
        else
        {
            if(diff2<0.001)
            {
                b2[0]= (params2[0] - b2[0])/2.;
                b2[1]= (params2[1] - b2[1])/2.;
                b2[2]= (params2[2] - b2[2])/2.;
                b2[3]= (params2[3] );
                b2[4]= (params2[4] );
                b2[5]= (params2[5] );
                rigid_trans2->SetParameters(b2);

                rigid_trans= RigidRegisterImagesEuler( b0_img,  str_img, "MI",parser->getRigidLR(),true,rigid_trans2);
            }
            else
            {
                std::vector<float> new_res; new_res.resize(3);
                new_res[0]= b0_img->GetSpacing()[0] * 2;
                new_res[1]= b0_img->GetSpacing()[1] * 2;
                new_res[2]= b0_img->GetSpacing()[2] * 2;
                std::vector<float> dummy;
                ImageType3D::Pointer b02= resample_3D_image(b0_img,new_res,dummy,"Linear");
                new_res[0]= str_img->GetSpacing()[0] * 2;
                new_res[1]= str_img->GetSpacing()[1] * 2;
                new_res[2]= str_img->GetSpacing()[2] * 2;
                ImageType3D::Pointer str2= resample_3D_image(str_img,new_res,dummy,"Linear");

                rigid_trans1=MultiStartRigidSearch(b02,  str2,new_metric_type);
                // The multistart result must go to in_trans; passed fifth it bound to the bool gd and the
                // refinement restarted from the moments initializer. Fixed upstream in 1ffb8f3.
                rigid_trans= RigidRegisterImagesEuler( b0_img,  str_img, new_metric_type,parser->getRigidLR(),true,rigid_trans1);
            }
        }

    }

    return rigid_trans;
}
