#ifndef SELECTBESTB0_HXX
#define SELECTBESTB0_HXX

#include "defines.h"
#include "../utilities/extract_3Dvolume_from_4D.h"
#include "../utilities/TORTOISE_Utilities.h"
#include "TORTOISE.h"

#include "itkResampleImageFilter.h"
#include "itkImageDuplicator.h"

#include "rigid_register_images.h"

#include "../utilities/write_3D_image_to_4D_file.h"

#include "vnl/vnl_math.h"
#include "vnl/vnl_matrix.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

double cross_correlation(ImageType3D::Pointer img1,ImageType3D::Pointer img2)
{
    ImageType3D::Pointer m_MetricImage= ImageType3D::New();
    m_MetricImage->SetRegions(img1->GetLargestPossibleRegion());
    m_MetricImage->Allocate();
    m_MetricImage->SetSpacing(img1->GetSpacing());
    m_MetricImage->SetOrigin(img1->GetOrigin());
    m_MetricImage->SetDirection(img1->GetDirection());
    m_MetricImage->FillBuffer(0);

    ImageType3D::SizeType sz= img1->GetLargestPossibleRegion().GetSize();

    int R=3;
    double LIM=1E10*itk::NumericTraits< double >::epsilon();

    #pragma omp parallel for
    for( int k=0; k<(int)sz[2];k++)
    {
        TORTOISE::EnableOMPThread();

        ImageType3D::IndexType index;
        index[2]=k;

        for(unsigned int j=0; j<sz[1];j++)
        {
            index[1]=j;
            for(unsigned int i=0; i<sz[0];i++)
            {
                index[0]=i;

                double suma2 = 0.0;
                double suma = 0.0;
                double  sumac=0;
                double sumc2 = 0.0;
                double sumc = 0.0;

                ImageType3D::SizeType size;
                ImageType3D::IndexType start;

                start[2]= std::max(0,(int)index[2]-R);
                size[2]=   std::min((int)sz[2]-1,(int)index[2]+R)-start[2]+1;

                start[1]= std::max(0,(int)index[1]-R);
                size[1]=   std::min((int)sz[1]-1,(int)index[1]+R)-start[1]+1;

                start[0]= std::max(0,(int)index[0]-R);
                size[0]=   std::min((int)sz[0]-1,(int)index[0]+R)-start[0]+1;

                int count = size[0]*size[1]*size[2];
                ImageType3D::RegionType reg(start,size);

                typedef itk::ImageRegionConstIterator<ImageType3D> IterType;
                IterType Fiter(img1,reg);
                IterType Miter(img2,reg);
                Fiter.GoToBegin();
                Miter.GoToBegin();

                while(!Fiter.IsAtEnd())
                {
                    double f = Fiter.Get();
                    double m=  Miter.Get();

                    suma2 += f * f;
                    suma += f;
                    sumc2 += m * m;
                    sumc += m;
                    sumac += f*m;

                    ++Fiter;
                    ++Miter;
                }

                double FMean = suma / count;
                double MMean = sumc / count;

                double sFF = suma2 -     FMean*suma;
                double sMM = sumc2 -     MMean*sumc;
                double sFM = sumac - FMean * sumc;

                double sFF_sMM = sFF * sMM;

                if(fabs(sFF_sMM) > LIM && fabs(sFF) > LIM && fabs(sMM) > LIM)
                    m_MetricImage->SetPixel(index, sFM*sFM/ sFF_sMM );
            }
        }

        TORTOISE::DisableOMPThread();
    }

    double value = 0;
    typedef itk::ImageRegionIterator<ImageType3D>  ItType;
    ItType it(m_MetricImage,m_MetricImage->GetLargestPossibleRegion());
    it.GoToBegin();
    while( !it.IsAtEnd() )
    {
        value+= it.Get();
        ++it;
    }
    value/=(sz[0]*sz[1]*sz[2]);


    return value;
}

void write_b0_selection_report(std::string report_path,
                               const std::vector<int> &b0_ids,
                               const vnl_vector<double> &bvals,
                               const std::vector<RigidTransformType::Pointer> &transforms,
                               const vnl_matrix<double> &CCs,
                               int best_r, int best_c)
{
    std::ofstream outfile(report_path.c_str());
    if(!outfile.good())
    {
        std::cout<<"Could not write the b=0 selection report to "<<report_path<<std::endl;
        return;
    }

    int n = (int)b0_ids.size();
    const double RADTODEG= 180.0/vnl_math::pi;

    outfile<<"# Rigid parameters are ITK Euler3DTransform (ZYX, LPS mm/deg, fixed-to-moving) mapping the registration reference to each volume."<<std::endl;
    outfile<<"volume_index\tbval\tregistered_to\ttrans_x_mm\ttrans_y_mm\ttrans_z_mm"
           <<"\trot_x_deg\trot_y_deg\trot_z_deg\ttranslation_total_mm\trotation_total_deg"
           <<"\tmean_cc\tin_best_pair\tselected";
    for(int j=0;j<n;j++)
        outfile<<"\tcc_vol"<<b0_ids[j];
    outfile<<std::endl;

    for(int i=0;i<n;i++)
    {
        std::string ref_label= "first_b0";
        if(n==1)
            ref_label="none";
        else if(i==0)
            ref_label="mean_b0";

        RigidTransformType::TranslationType trans;
        trans.Fill(0);
        double rots[3]={0,0,0};
        double total_rot=0;
        if(transforms[i])
        {
            trans= transforms[i]->GetTranslation();
            rots[0]= transforms[i]->GetAngleX()*RADTODEG;
            rots[1]= transforms[i]->GetAngleY()*RADTODEG;
            rots[2]= transforms[i]->GetAngleZ()*RADTODEG;
            auto R= transforms[i]->GetMatrix();
            double cos_arg= (R(0,0)+R(1,1)+R(2,2)-1.)/2.;
            cos_arg= std::max(-1.,std::min(1.,cos_arg));
            total_rot= acos(cos_arg)*RADTODEG;
        }
        double total_trans= sqrt(trans[0]*trans[0]+trans[1]*trans[1]+trans[2]*trans[2]);

        double cc_sum=0;
        for(int j=0;j<n;j++)
        {
            if(j!=i)
                cc_sum+= CCs(i,j);
        }

        outfile<<b0_ids[i]<<"\t"<<std::fixed<<std::setprecision(1)<<bvals[b0_ids[i]]<<"\t"<<ref_label;
        outfile<<std::setprecision(4);
        outfile<<"\t"<<trans[0]<<"\t"<<trans[1]<<"\t"<<trans[2];
        outfile<<"\t"<<rots[0]<<"\t"<<rots[1]<<"\t"<<rots[2];
        outfile<<"\t"<<total_trans<<"\t"<<total_rot;
        outfile<<std::setprecision(6);
        if(n>1)
            outfile<<"\t"<<cc_sum/(n-1);
        else
            outfile<<"\tNaN";
        outfile<<"\t"<<(int)(i==best_r || i==best_c);
        outfile<<"\t"<<(int)(i==best_r);
        for(int j=0;j<n;j++)
        {
            if(j==i)
                outfile<<"\tNaN";
            else
                outfile<<"\t"<<CCs(i,j);
        }
        outfile<<std::endl;
    }
    outfile.close();
}

int select_best_b0(ImageType4D::Pointer img4d, vnl_vector<double> bvals, ImageType3D::Pointer &avg_best_b0_img,
                   float b0_threshold=-1, std::string report_path="")
{
    using OkanQuadraticTransformType=TORTOISE::OkanQuadraticTransformType;

    std::vector<int> b0_ids;
    if(b0_threshold>=0)
    {
        for(int v=0;v<bvals.size();v++)
        {
            if(bvals[v]<=b0_threshold)
                b0_ids.push_back(v);
        }
        if(b0_ids.size()==0)
        {
            std::cout<<"No volumes with b-value <= "<<b0_threshold<<" in the dataset."<<std::endl;
            return -1;
        }
    }
    else
    {
        float b0_val= bvals.min_value();         //get b=0 images' volume id
        for(int v=0;v<bvals.size();v++)
        {
            if(fabs(bvals[v]-b0_val)<10)
                b0_ids.push_back(v);
        }
    }

    std::vector<RigidTransformType::Pointer> b0_to_ref_trans(b0_ids.size(),nullptr);

    if(b0_ids.size() ==1)
    {
        avg_best_b0_img = extract_3D_volume_from_4D(img4d, b0_ids[0]);
        if(report_path!="")
        {
            vnl_matrix<double> CCs(1,1,0.);
            write_b0_selection_report(report_path,b0_ids,bvals,b0_to_ref_trans,CCs,0,0);
        }
        return b0_ids[0];
    }


    std::vector<ImageType3D::Pointer> registered_imgs;    //put the b=0 images in a vector
    for(int v=0;v<b0_ids.size();v++)
    {
        ImageType3D::Pointer b0_img = extract_3D_volume_from_4D(img4d, b0_ids[v]);
        registered_imgs.push_back(b0_img);
    }



    #pragma omp parallel for
    for(int v=1;v<b0_ids.size();v++)
    {
        TORTOISE::EnableOMPThread();
        // Same computation as RigidRegisterImages, unwrapped so the Euler
        // parameters survive for the selection report.
        RigidTransformType::Pointer curr_rigid= RigidRegisterImagesEuler(registered_imgs[0],registered_imgs[v]);
        CompositeTransformType::Pointer curr_composite= CompositeTransformType::New();
        curr_composite->AddTransform(curr_rigid);
        OkanQuadraticTransformType::Pointer curr_trans= CompositeLinearToQuadratic(curr_composite,"vertical");
        b0_to_ref_trans[v]=curr_rigid;

        typedef itk::ResampleImageFilter<ImageType3D, ImageType3D> ResampleImageFilterType;
        ResampleImageFilterType::Pointer resampleFilter = ResampleImageFilterType::New();
        resampleFilter->SetOutputParametersFromImage(registered_imgs[0]);
        resampleFilter->SetInput(registered_imgs[v]);
        resampleFilter->SetTransform(curr_trans);
        int NITK= TORTOISE::GetAvailableITKThreadFor();
        resampleFilter->SetNumberOfWorkUnits(NITK);
        resampleFilter->Update();
        registered_imgs[v]= resampleFilter->GetOutput();
        TORTOISE::DisableOMPThread();
    }

    using DupType=itk::ImageDuplicator<ImageType3D> ;
    DupType::Pointer dup= DupType::New();
    dup->SetInputImage(registered_imgs[0]);
    dup->Update();
    ImageType3D::Pointer avg_img = dup->GetOutput();
    avg_img->FillBuffer(0);

    itk::ImageRegionIteratorWithIndex<ImageType3D> it(avg_img,avg_img->GetLargestPossibleRegion());
    it.GoToBegin();
    while(!it.IsAtEnd())
    {
        ImageType3D::IndexType ind3=it.GetIndex();
        double val=0;
        for(int v=1;v<b0_ids.size();v++)
            val+= registered_imgs[v]->GetPixel(ind3);
        val/=(b0_ids.size()-1);
        it.Set(val);
        ++it;
    }

    {
        RigidTransformType::Pointer curr_rigid= RigidRegisterImagesEuler(avg_img,registered_imgs[0]);
        CompositeTransformType::Pointer curr_composite= CompositeTransformType::New();
        curr_composite->AddTransform(curr_rigid);
        OkanQuadraticTransformType::Pointer curr_trans= CompositeLinearToQuadratic(curr_composite,"vertical");
        b0_to_ref_trans[0]=curr_rigid;

        typedef itk::ResampleImageFilter<ImageType3D, ImageType3D> ResampleImageFilterType;
        ResampleImageFilterType::Pointer resampleFilter = ResampleImageFilterType::New();
        resampleFilter->SetOutputParametersFromImage(avg_img);
        resampleFilter->SetInput(registered_imgs[0]);
        resampleFilter->SetTransform(curr_trans);
        resampleFilter->Update();
        registered_imgs[0]= resampleFilter->GetOutput();
    }


    vnl_matrix<double> CCs(b0_ids.size(),b0_ids.size(),0.);
    int best_r=0, best_c=0;
    double best_CC=-1;
    for(int r=0;r<b0_ids.size();r++)
    {
        for(int c=r+1;c<b0_ids.size();c++)
        {
            double CC= cross_correlation(registered_imgs[r],registered_imgs[c]);
            CCs(r,c)=CC;
            CCs(c,r)=CC;
            if(CC>best_CC)
            {
                best_CC=CC;
                best_r=r;
                best_c=c;
            }
        }
    }

    if(report_path!="")
        write_b0_selection_report(report_path,b0_ids,bvals,b0_to_ref_trans,CCs,best_r,best_c);


    avg_best_b0_img = ImageType3D::New();
    avg_best_b0_img->SetRegions(registered_imgs[0]->GetLargestPossibleRegion());
    avg_best_b0_img->Allocate();
    avg_best_b0_img->SetSpacing(registered_imgs[0]->GetSpacing());
    avg_best_b0_img->SetDirection(registered_imgs[0]->GetDirection());
    avg_best_b0_img->SetOrigin(registered_imgs[0]->GetOrigin());
    avg_best_b0_img->FillBuffer(0);

    itk::ImageRegionIteratorWithIndex<ImageType3D> it2(avg_best_b0_img,avg_best_b0_img->GetLargestPossibleRegion());
    for(it2.GoToBegin();!it2.IsAtEnd();++it2)
    {
        ImageType3D::IndexType ind3= it2.GetIndex();
        float val = 0.5 * (registered_imgs[best_r]->GetPixel(ind3) +registered_imgs[best_c]->GetPixel(ind3) );
        it2.Set(val);
    }

    return b0_ids[best_r];
}

#endif
