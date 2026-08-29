#include "defines.h"
#include "TORTOISE.h"
#include "../unring.h"
#include "itkImageIOFactory.h"
#include "itkImageIOBase.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << "  "        \
                      << #cond << std::endl;                                   \
            return 1;                                                          \
        }                                                                      \
    } while (0)

// Allocate a complex 4D image with a deterministic, non-symmetric pattern in
// both components so a swapped or dropped channel cannot pass unnoticed.
static ImageType4DComplex::Pointer MakeComplexImage(int nx, int ny, int nz, int nt)
{
    ImageType4DComplex::SizeType sz;
    sz[0] = nx; sz[1] = ny; sz[2] = nz; sz[3] = nt;
    ImageType4DComplex::IndexType start; start.Fill(0);
    ImageType4DComplex::RegionType reg(start, sz);

    ImageType4DComplex::Pointer img = ImageType4DComplex::New();
    img->SetRegions(reg);
    img->Allocate();

    ImageType4DComplex::IndexType ind;
    for (int t = 0; t < nt; t++) {
        ind[3] = t;
        for (int z = 0; z < nz; z++) {
            ind[2] = z;
            for (int y = 0; y < ny; y++) {
                ind[1] = y;
                for (int x = 0; x < nx; x++) {
                    ind[0] = x;
                    float re = 1.0f * x + 10.0f * y + 100.0f * z + 1000.0f * t;
                    float im = -2.0f * x + 3.0f * y - 5.0f * z + 7.0f * t;
                    img->SetPixel(ind, std::complex<float>(re, im));
                }
            }
        }
    }
    return img;
}

static int test_complex_io_roundtrip()
{
    const std::string fname = "/tmp/test_complex_roundtrip.nii";

    ImageType4DComplex::Pointer img = MakeComplexImage(4, 5, 3, 2);
    writeImageD<ImageType4DComplex>(img, fname);

    // The on-disk datatype must be COMPLEX64, not two-component float.
    itk::ImageIOBase::Pointer io = itk::ImageIOFactory::CreateImageIO(
        fname.c_str(), itk::ImageIOFactory::FileModeEnum::ReadMode);
    CHECK(io.IsNotNull());
    io->SetFileName(fname);
    io->ReadImageInformation();
    CHECK(io->GetPixelType() == itk::IOPixelEnum::COMPLEX);
    CHECK(io->GetComponentType() == itk::IOComponentEnum::FLOAT);

    ImageType4DComplex::Pointer back = readImageD<ImageType4DComplex>(fname);

    ImageType4DComplex::SizeType sz = back->GetLargestPossibleRegion().GetSize();
    CHECK(sz[0] == 4 && sz[1] == 5 && sz[2] == 3 && sz[3] == 2);

    ImageType4DComplex::IndexType ind;
    for (int t = 0; t < 2; t++) {
        ind[3] = t;
        for (int z = 0; z < 3; z++) {
            ind[2] = z;
            for (int y = 0; y < 5; y++) {
                ind[1] = y;
                for (int x = 0; x < 4; x++) {
                    ind[0] = x;
                    std::complex<float> a = img->GetPixel(ind);
                    std::complex<float> b = back->GetPixel(ind);
                    CHECK(a.real() == b.real());
                    CHECK(a.imag() == b.imag());
                }
            }
        }
    }

    std::remove(fname.c_str());
    std::cout << "PASS complex_io_roundtrip" << std::endl;
    return 0;
}

// With a zero imaginary channel, unring_2d's imaginary intermediates are all
// zero, so UnRingFullComplex must reproduce UnRingFull's real output. This is
// the regression anchor: it is the only way a marshalling error in the complex
// path can be caught, because such an error is silent on real data.
static int test_magnitude_equivalence()
{
    const int nx = 64, ny = 64, nz = 2, nt = 1;

    ImageType4D::SizeType sz;
    sz[0] = nx; sz[1] = ny; sz[2] = nz; sz[3] = nt;
    ImageType4D::IndexType start; start.Fill(0);
    ImageType4D::RegionType reg(start, sz);

    ImageType4D::Pointer real_img = ImageType4D::New();
    real_img->SetRegions(reg); real_img->Allocate(); real_img->FillBuffer(0);

    ImageType4DComplex::SizeType csz;
    csz[0] = nx; csz[1] = ny; csz[2] = nz; csz[3] = nt;
    ImageType4DComplex::IndexType cstart; cstart.Fill(0);
    ImageType4DComplex::RegionType creg(cstart, csz);

    ImageType4DComplex::Pointer cplx_img = ImageType4DComplex::New();
    cplx_img->SetRegions(creg); cplx_img->Allocate();

    // A hard-edged square: the sharpest thing available, so SuShi has ringing
    // to actually remove and the two paths have something to disagree about.
    ImageType4D::IndexType ind;
    ImageType4DComplex::IndexType cind;
    for (int t = 0; t < nt; t++) {
        ind[3] = t; cind[3] = t;
        for (int z = 0; z < nz; z++) {
            ind[2] = z; cind[2] = z;
            for (int y = 0; y < ny; y++) {
                ind[1] = y; cind[1] = y;
                for (int x = 0; x < nx; x++) {
                    ind[0] = x; cind[0] = x;
                    float v = (x >= 22 && x < 42 && y >= 22 && y < 42) ? 100.0f : 0.0f;
                    real_img->SetPixel(ind, v);
                    cplx_img->SetPixel(cind, std::complex<float>(v, 0.0f));
                }
            }
        }
    }

    ImageType4D::Pointer real_out = UnRingFull(real_img, 25, 1, 3);
    ImageType4DComplex::Pointer cplx_out = UnRingFullComplex(cplx_img, 25, 1, 3);

    double max_rel = 0.0;
    for (int t = 0; t < nt; t++) {
        ind[3] = t; cind[3] = t;
        for (int z = 0; z < nz; z++) {
            ind[2] = z; cind[2] = z;
            for (int y = 0; y < ny; y++) {
                ind[1] = y; cind[1] = y;
                for (int x = 0; x < nx; x++) {
                    ind[0] = x; cind[0] = x;
                    double a = real_out->GetPixel(ind);
                    double b = cplx_out->GetPixel(cind).real();
                    double rel = fabs(a - b) / std::max(1.0, fabs(a));
                    if (rel > max_rel) max_rel = rel;
                }
            }
        }
    }

    std::cout << "max relative difference: " << max_rel << std::endl;
    CHECK(max_rel < 1e-9);

    std::cout << "PASS magnitude_equivalence" << std::endl;
    return 0;
}

int main(int argc, char *argv[])
{
    // Later tasks' tests (unring.h, pocs.h) call TORTOISE::EnableOMPThread(),
    // which indexes into Nthreads_per_OMP_thread, a std::vector<uint> that is
    // declared empty in TORTOISE_global.cxx and only sized by SetThreadArray(),
    // called solely from the TORTOISE default constructor. Without this, those
    // tests write out of bounds on an empty vector. Do not remove as "unused".
    TORTOISE t;

    if (argc < 2) {
        std::cerr << "Usage: GibbsComplexTests <test_name>" << std::endl;
        return 1;
    }
    std::string name(argv[1]);

    if (name == "complex_io_roundtrip") return test_complex_io_roundtrip();
    if (name == "magnitude_equivalence") return test_magnitude_equivalence();

    std::cerr << "Unknown test: " << name << std::endl;
    return 1;
}
