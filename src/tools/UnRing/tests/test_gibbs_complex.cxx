#include "defines.h"
#include "TORTOISE.h"
#include "../unring.h"
#include "../pf_geometry.h"
#include "../pocs.h"
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

// Build a complex image whose k-space has an exactly-zero band of band_len
// lines starting at shifted PE index band_start. band_len == 0 gives full
// k-space. Values decay as 1/r from DC so the energy profile has realistic
// dynamic range instead of being flat.
static ImageType4DComplex::Pointer MakeBandedImage(int nx, int ny, int band_start,
                                                   int band_len, int pe_axis)
{
    const int npix = nx * ny;
    fftw_complex *K  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npix);
    fftw_complex *im = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npix);
    fftw_plan p = fftw_plan_dft_2d(ny, nx, K, im, FFTW_BACKWARD, FFTW_ESTIMATE);

    const int band_end = band_start + band_len - 1;

    for (int sy = 0; sy < ny; sy++) {
        for (int sx = 0; sx < nx; sx++) {
            const int uy = ShiftedToUnshifted(sy, ny);
            const int ux = ShiftedToUnshifted(sx, nx);
            const int pe = (pe_axis == 0) ? sx : sy;

            double re = 0.0, imv = 0.0;
            if (band_len == 0 || pe < band_start || pe > band_end) {
                const double dy = sy - ny / 2;
                const double dx = sx - nx / 2;
                const double r = sqrt(dx * dx + dy * dy) + 1.0;
                re  = cos(0.7 * sx + 0.3 * sy) / r;
                imv = sin(0.4 * sx - 0.9 * sy) / r;
            }
            K[nx * uy + ux][0] = re;
            K[nx * uy + ux][1] = imv;
        }
    }
    fftw_execute(p);

    ImageType4DComplex::SizeType sz;
    sz[0] = nx; sz[1] = ny; sz[2] = 1; sz[3] = 1;
    ImageType4DComplex::IndexType start; start.Fill(0);
    ImageType4DComplex::RegionType reg(start, sz);
    ImageType4DComplex::Pointer img = ImageType4DComplex::New();
    img->SetRegions(reg);
    img->Allocate();

    ImageType4DComplex::IndexType ind; ind[2] = 0; ind[3] = 0;
    for (int y = 0; y < ny; y++) {
        ind[1] = y;
        for (int x = 0; x < nx; x++) {
            ind[0] = x;
            img->SetPixel(ind, std::complex<float>((float)im[nx * y + x][0],
                                                   (float)im[nx * y + x][1]));
        }
    }

    fftw_destroy_plan(p);
    fftw_free(K);
    fftw_free(im);
    return img;
}

// Build a complex image whose k-space has exact-zero bands of edge_len lines
// at BOTH edges of shifted PE space -- the signature of symmetric zero-
// padding / matrix-interpolated ("ZIP") reconstructions, as opposed to the
// single-edge band of true partial-Fourier truncation.
static ImageType4DComplex::Pointer MakeSymmetricBandedImage(int nx, int ny, int edge_len,
                                                             int pe_axis)
{
    const int npix = nx * ny;
    fftw_complex *K  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npix);
    fftw_complex *im = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npix);
    fftw_plan p = fftw_plan_dft_2d(ny, nx, K, im, FFTW_BACKWARD, FFTW_ESTIMATE);

    const int n_pe = (pe_axis == 0) ? nx : ny;

    for (int sy = 0; sy < ny; sy++) {
        for (int sx = 0; sx < nx; sx++) {
            const int uy = ShiftedToUnshifted(sy, ny);
            const int ux = ShiftedToUnshifted(sx, nx);
            const int pe = (pe_axis == 0) ? sx : sy;

            double re = 0.0, imv = 0.0;
            if (pe >= edge_len && pe < n_pe - edge_len) {
                const double dy = sy - ny / 2;
                const double dx = sx - nx / 2;
                const double r = sqrt(dx * dx + dy * dy) + 1.0;
                re  = cos(0.7 * sx + 0.3 * sy) / r;
                imv = sin(0.4 * sx - 0.9 * sy) / r;
            }
            K[nx * uy + ux][0] = re;
            K[nx * uy + ux][1] = imv;
        }
    }
    fftw_execute(p);

    ImageType4DComplex::SizeType sz;
    sz[0] = nx; sz[1] = ny; sz[2] = 1; sz[3] = 1;
    ImageType4DComplex::IndexType start; start.Fill(0);
    ImageType4DComplex::RegionType reg(start, sz);
    ImageType4DComplex::Pointer img = ImageType4DComplex::New();
    img->SetRegions(reg);
    img->Allocate();

    ImageType4DComplex::IndexType ind; ind[2] = 0; ind[3] = 0;
    for (int y = 0; y < ny; y++) {
        ind[1] = y;
        for (int x = 0; x < nx; x++) {
            ind[0] = x;
            img->SetPixel(ind, std::complex<float>((float)im[nx * y + x][0],
                                                   (float)im[nx * y + x][1]));
        }
    }

    fftw_destroy_plan(p);
    fftw_free(K);
    fftw_free(im);
    return img;
}

static int test_pf_detection_positive()
{
    struct Case { int nx, ny, n_missing; PFGeometry::Side side; int pe_axis; const char *label; };
    Case cases[] = {
        { 64, 64, 16, PFGeometry::Low,  1, "6/8 along y, low side"  },
        { 64, 64,  8, PFGeometry::High, 1, "7/8 along y, high side" },
        { 64, 63, 16, PFGeometry::Low,  1, "odd PE size"            },
        { 64, 64, 16, PFGeometry::High, 0, "PF along x"             },
    };

    for (int c = 0; c < 4; c++) {
        const int n_pe = (cases[c].pe_axis == 0) ? cases[c].nx : cases[c].ny;
        const int band_start = (cases[c].side == PFGeometry::Low) ? 0 : n_pe - cases[c].n_missing;

        ImageType4DComplex::Pointer img =
            MakeBandedImage(cases[c].nx, cases[c].ny, band_start, cases[c].n_missing, cases[c].pe_axis);

        PFGeometry g = DetectPFGeometry(img, cases[c].pe_axis, 1e-6f, 8);

        std::cout << cases[c].label << ": n_missing=" << g.n_missing
                  << " factor=" << g.factor << " side=" << (int)g.side << std::endl;

        CHECK(g.status == PFGeometry::DetectedPF);
        CHECK(g.is_partial_fourier);
        CHECK(g.n_pe == n_pe);
        CHECK(g.n_missing == cases[c].n_missing);
        CHECK(g.side == cases[c].side);
        CHECK(fabs(g.factor - (float)(n_pe - cases[c].n_missing) / (float)n_pe) < 1e-4f);
    }

    std::cout << "PASS pf_detection_positive" << std::endl;
    return 0;
}

static int test_pf_detection_negative()
{
    // Full k-space: nothing to detect.
    {
        ImageType4DComplex::Pointer img = MakeBandedImage(64, 64, 0, 0, 1);
        PFGeometry g = DetectPFGeometry(img, 1, 1e-6f, 8);
        CHECK(!g.is_partial_fourier);
        CHECK(g.status == PFGeometry::NoZeroBand);
        CHECK(g.n_missing == 0);
    }

    // A zero band floating in the interior is not asymmetric truncation.
    // Reporting it as PF would be worse than reporting nothing.
    {
        ImageType4DComplex::Pointer img = MakeBandedImage(64, 64, 20, 8, 1);
        PFGeometry g = DetectPFGeometry(img, 1, 1e-6f, 8);
        CHECK(!g.is_partial_fourier);
        CHECK(g.status == PFGeometry::InteriorBand);
    }

    // More than half of k-space missing is implausible for PF.
    {
        ImageType4DComplex::Pointer img = MakeBandedImage(64, 64, 0, 40, 1);
        PFGeometry g = DetectPFGeometry(img, 1, 1e-6f, 8);
        CHECK(!g.is_partial_fourier);
        CHECK(g.status == PFGeometry::ImplausibleFactor);
    }

    // Zero bands at BOTH edges (ZIP / matrix-interpolated reconstruction,
    // e.g. n_pe=256 acquired [64,191]). This must NOT be reported as
    // partial Fourier -- ApplyPOCS would otherwise pin the genuinely empty
    // high-ky band to zero while synthesising invented content into the
    // low-ky band. n_pe=64 with 8 zero lines at each end (acquired [8,55])
    // is the test-sized analogue of the 256/128 ZIP case.
    {
        ImageType4DComplex::Pointer img = MakeSymmetricBandedImage(64, 64, 8, 1);
        PFGeometry g = DetectPFGeometry(img, 1, 1e-6f, 8);
        CHECK(!g.is_partial_fourier);
        CHECK(g.status == PFGeometry::SymmetricBand);
    }

    std::cout << "PASS pf_detection_negative" << std::endl;
    return 0;
}

// Ground-truth complex phantom: hard-edged structure (so truncation produces
// real ringing) with a smooth low-frequency phase (the assumption POCS relies
// on).
static ImageType4DComplex::Pointer MakePhantom(int nx, int ny)
{
    ImageType4DComplex::SizeType sz;
    sz[0] = nx; sz[1] = ny; sz[2] = 1; sz[3] = 1;
    ImageType4DComplex::IndexType start; start.Fill(0);
    ImageType4DComplex::RegionType reg(start, sz);
    ImageType4DComplex::Pointer img = ImageType4DComplex::New();
    img->SetRegions(reg);
    img->Allocate();

    const double two_pi = 6.283185307179586;
    ImageType4DComplex::IndexType ind; ind[2] = 0; ind[3] = 0;
    for (int y = 0; y < ny; y++) {
        ind[1] = y;
        for (int x = 0; x < nx; x++) {
            ind[0] = x;
            double mag = 0.0;
            if (x >= 16 && x < 48 && y >= 14 && y < 50) mag = 100.0;
            if (x >= 28 && x < 36 && y >= 26 && y < 38) mag = 40.0;
            const double ph = 0.5 * cos(two_pi * x / nx) + 0.3 * sin(two_pi * y / ny);
            img->SetPixel(ind, std::complex<float>((float)(mag * cos(ph)),
                                                   (float)(mag * sin(ph))));
        }
    }
    return img;
}

// Zero-fill the given shifted PE band of an image's k-space, in place.
static void TruncateKSpace(ImageType4DComplex::Pointer img, int band_start, int band_len, int pe_axis)
{
    ImageType4DComplex::SizeType sz = img->GetLargestPossibleRegion().GetSize();
    const int nx = (int)sz[0], ny = (int)sz[1];
    const int npix = nx * ny;

    fftw_complex *a = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npix);
    fftw_complex *b = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npix);
    fftw_plan pf = fftw_plan_dft_2d(ny, nx, a, b, FFTW_FORWARD,  FFTW_ESTIMATE);
    fftw_plan pb = fftw_plan_dft_2d(ny, nx, b, a, FFTW_BACKWARD, FFTW_ESTIMATE);

    ImageType4DComplex::IndexType ind; ind[2] = 0; ind[3] = 0;
    for (int y = 0; y < ny; y++) { ind[1] = y;
        for (int x = 0; x < nx; x++) { ind[0] = x;
            std::complex<float> v = img->GetPixel(ind);
            a[nx * y + x][0] = v.real();
            a[nx * y + x][1] = v.imag();
        } }

    fftw_execute(pf);

    const int n_pe = (pe_axis == 0) ? nx : ny;
    for (int s = band_start; s < band_start + band_len; s++) {
        const int u = ShiftedToUnshifted(s, n_pe);
        for (int q = 0; q < ((pe_axis == 0) ? ny : nx); q++) {
            const int idx = (pe_axis == 0) ? (nx * q + u) : (nx * u + q);
            b[idx][0] = 0.0;
            b[idx][1] = 0.0;
        }
    }

    fftw_execute(pb);

    const double nfac = 1.0 / (double) npix;
    for (int y = 0; y < ny; y++) { ind[1] = y;
        for (int x = 0; x < nx; x++) { ind[0] = x;
            img->SetPixel(ind, std::complex<float>((float)(a[nx * y + x][0] * nfac),
                                                   (float)(a[nx * y + x][1] * nfac)));
        } }

    fftw_destroy_plan(pf); fftw_destroy_plan(pb);
    fftw_free(a); fftw_free(b);
}

static double MagnitudeRMSE(ImageType4DComplex::Pointer a, ImageType4DComplex::Pointer b)
{
    ImageType4DComplex::SizeType sz = a->GetLargestPossibleRegion().GetSize();
    double acc = 0.0; long n = 0;
    ImageType4DComplex::IndexType ind; ind[2] = 0; ind[3] = 0;
    for (int y = 0; y < (int)sz[1]; y++) { ind[1] = y;
        for (int x = 0; x < (int)sz[0]; x++) { ind[0] = x;
            const double d = std::abs(a->GetPixel(ind)) - std::abs(b->GetPixel(ind));
            acc += d * d; n++;
        } }
    return sqrt(acc / (double) n);
}

static int test_pocs_consistency()
{
    const int nx = 64, ny = 64, n_missing = 16;

    ImageType4DComplex::Pointer img = MakePhantom(nx, ny);
    TruncateKSpace(img, 0, n_missing, 1);   // Low side, 6/8

    // Snapshot the measured k-space before POCS.
    PFGeometry geom = DetectPFGeometry(img, 1, 1e-6f, 8);
    CHECK(geom.status == PFGeometry::DetectedPF);
    CHECK(geom.n_missing == n_missing);
    CHECK(geom.side == PFGeometry::Low);

    const int npix = nx * ny;
    fftw_complex *a  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npix);
    fftw_complex *k0 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npix);
    fftw_complex *k1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npix);
    fftw_plan pf = fftw_plan_dft_2d(ny, nx, a, k0, FFTW_FORWARD, FFTW_ESTIMATE);

    ImageType4DComplex::IndexType ind; ind[2] = 0; ind[3] = 0;
    for (int y = 0; y < ny; y++) { ind[1] = y;
        for (int x = 0; x < nx; x++) { ind[0] = x;
            std::complex<float> v = img->GetPixel(ind);
            a[nx * y + x][0] = v.real(); a[nx * y + x][1] = v.imag();
        } }
    fftw_execute(pf);

    POCSParams params;
    POCSResult res = ApplyPOCS(img, geom, 1, params);
    std::cout << "POCS iters=" << res.iters_run
              << " final_rel_change=" << res.final_rel_change << std::endl;
    CHECK(res.iters_run > 0);

    for (int y = 0; y < ny; y++) { ind[1] = y;
        for (int x = 0; x < nx; x++) { ind[0] = x;
            std::complex<float> v = img->GetPixel(ind);
            a[nx * y + x][0] = v.real(); a[nx * y + x][1] = v.imag();
        } }
    fftw_execute_dft(pf, a, k1);

    double maxmag = 0.0;
    for (int q = 0; q < npix; q++)
        maxmag = std::max(maxmag, sqrt(k0[q][0] * k0[q][0] + k0[q][1] * k0[q][1]));

    std::vector<char> acquired; std::vector<double> window;
    BuildPOCSMasks(geom, acquired, window);

    double worst = 0.0;
    for (int ky = 0; ky < ny; ky++) {
        if (!acquired[ky]) continue;
        for (int kx = 0; kx < nx; kx++) {
            const int q = nx * ky + kx;
            const double dr = k1[q][0] - k0[q][0];
            const double di = k1[q][1] - k0[q][1];
            worst = std::max(worst, sqrt(dr * dr + di * di) / maxmag);
        }
    }
    std::cout << "worst acquired-line deviation: " << worst << std::endl;
    CHECK(worst < 1e-5);

    fftw_destroy_plan(pf);
    fftw_free(a); fftw_free(k0); fftw_free(k1);

    // Full k-space must be left strictly alone.
    {
        ImageType4DComplex::Pointer full = MakePhantom(nx, ny);
        ImageType4DComplex::Pointer copy = MakePhantom(nx, ny);
        PFGeometry g = DetectPFGeometry(full, 1, 1e-6f, 8);
        CHECK(!g.is_partial_fourier);
        POCSResult r = ApplyPOCS(full, g, 1, params);
        CHECK(r.iters_run == 0);
        CHECK(MagnitudeRMSE(full, copy) == 0.0);
    }

    std::cout << "PASS pocs_consistency" << std::endl;
    return 0;
}

static int test_pocs_accuracy()
{
    const int nx = 64, ny = 64, n_missing = 16;

    ImageType4DComplex::Pointer truth = MakePhantom(nx, ny);
    ImageType4DComplex::Pointer zf    = MakePhantom(nx, ny);
    TruncateKSpace(zf, 0, n_missing, 1);

    ImageType4DComplex::Pointer pocs = MakePhantom(nx, ny);
    TruncateKSpace(pocs, 0, n_missing, 1);

    PFGeometry geom = DetectPFGeometry(pocs, 1, 1e-6f, 8);
    CHECK(geom.status == PFGeometry::DetectedPF);

    POCSParams params;
    ApplyPOCS(pocs, geom, 1, params);

    const double rmse_zf   = MagnitudeRMSE(zf, truth);
    const double rmse_pocs = MagnitudeRMSE(pocs, truth);
    std::cout << "RMSE zero-filled=" << rmse_zf << "  POCS=" << rmse_pocs
              << "  ratio=" << (rmse_pocs / rmse_zf) << std::endl;

    CHECK(rmse_zf > 0.0);
    CHECK(rmse_pocs < 0.9 * rmse_zf);

    std::cout << "PASS pocs_accuracy" << std::endl;
    return 0;
}


// A global phase offset is an arbitrary receiver convention, so unringing must commute
// with it:  U(S*e^{i.theta}) == U(S)*e^{i.theta}.
// The legacy TV criterion sums |dRe| + |dIm| -- an L1 norm in the complex plane. That is
// invariant when Re and Im are merely swapped, i.e. at multiples of pi/2, but not at any
// other angle. A test that only checked pi/2 would therefore PASS against the broken
// implementation, so several angles are checked here on purpose.
static int test_complex_phase_equivariance()
{
    const int nx = 96, ny = 96;
    const double PI_ = 3.14159265358979323846;

    ImageType4DComplex::Pointer base = MakePhantom(nx, ny);
    ImageType4DComplex::Pointer U0 = UnRingFullComplex(base, 25, 1, 3);

    ImageType4DComplex::IndexType ind; ind[2] = 0; ind[3] = 0;
    double peak = 0.0;
    for (int y = 0; y < ny; y++) { ind[1] = y;
        for (int x = 0; x < nx; x++) { ind[0] = x;
            peak = std::max(peak, (double) std::abs(U0->GetPixel(ind)));
        } }
    CHECK(peak > 0.0);

    const double thetas[5] = { PI_/8.0, PI_/4.0, PI_/3.0, PI_/2.0, 1.0 };
    for (int t = 0; t < 5; t++)
    {
        const double th = thetas[t];
        const std::complex<double> rot(cos(th), sin(th));

        ImageType4DComplex::Pointer rotimg = MakePhantom(nx, ny);
        for (int y = 0; y < ny; y++) { ind[1] = y;
            for (int x = 0; x < nx; x++) { ind[0] = x;
                std::complex<double> v = std::complex<double>(rotimg->GetPixel(ind)) * rot;
                rotimg->SetPixel(ind, std::complex<float>((float) v.real(), (float) v.imag()));
            } }

        ImageType4DComplex::Pointer Ur = UnRingFullComplex(rotimg, 25, 1, 3);

        double worst = 0.0;
        for (int y = 0; y < ny; y++) { ind[1] = y;
            for (int x = 0; x < nx; x++) { ind[0] = x;
                std::complex<double> a = std::complex<double>(Ur->GetPixel(ind)) * std::conj(rot);
                std::complex<double> b = std::complex<double>(U0->GetPixel(ind));
                worst = std::max(worst, std::abs(a - b));
            } }

        std::cout << "  theta=" << th << "  max relative deviation = " << (worst / peak) << std::endl;
        CHECK(worst / peak < 1e-6);
    }

    std::cout << "PASS complex_phase_equivariance" << std::endl;
    return 0;
}


// The zero-fill compatibility diagnostic. Unlike DetectPFGeometry it takes the band width
// from the DECLARED partial-Fourier factor and measures against the conjugate mirror, so it
// is scale-free: it does not care where the DC line sits or what the noise floor is.
static int test_pf_diagnostics()
{
    const int nx = 64, ny = 64, nm = 16;   // 6/8 partial Fourier along y

    // Zero-filled at the low end: the band should read as essentially empty.
    {
        ImageType4DComplex::Pointer img = MakeBandedImage(nx, ny, 0, nm, 1);
        PFDiagnostics d = ComputePFDiagnostics(img, 1, 0.75f, 8, 16);
        std::cout << "  zero-filled low: ratio=" << d.mirror_ratio
                  << " side=" << (int) d.side << " asym=" << d.side_asymmetry
                  << " compatible=" << d.zero_fill_compatible << std::endl;
        CHECK(d.valid);
        CHECK(d.n_missing == nm);
        CHECK(d.side == PFGeometry::Low);
        CHECK(d.mirror_ratio < 0.05f);
        CHECK(d.zero_fill_compatible);
        CHECK(d.side_asymmetry > 5.0f);
    }

    // Same, truncated at the high end: the inferred side must follow.
    {
        ImageType4DComplex::Pointer img = MakeBandedImage(nx, ny, ny - nm, nm, 1);
        PFDiagnostics d = ComputePFDiagnostics(img, 1, 0.75f, 8, 16);
        std::cout << "  zero-filled high: ratio=" << d.mirror_ratio
                  << " side=" << (int) d.side << std::endl;
        CHECK(d.valid);
        CHECK(d.side == PFGeometry::High);
        CHECK(d.zero_fill_compatible);
    }

    // Full k-space: nothing is missing, so the "band" holds as much as its mirror and the
    // image must NOT be reported as compatible with zero filling.
    {
        ImageType4DComplex::Pointer img = MakeBandedImage(nx, ny, 0, 0, 1);
        PFDiagnostics d = ComputePFDiagnostics(img, 1, 0.75f, 8, 16);
        std::cout << "  full k-space:    ratio=" << d.mirror_ratio
                  << " compatible=" << d.zero_fill_compatible << std::endl;
        CHECK(d.valid);
        CHECK(d.mirror_ratio > 0.2f);
        CHECK(!d.zero_fill_compatible);
    }

    // A nonsensical declared factor is rejected rather than silently producing a band.
    {
        ImageType4DComplex::Pointer img = MakeBandedImage(nx, ny, 0, nm, 1);
        CHECK(!ComputePFDiagnostics(img, 1, 1.0f,  8, 16).valid);   // nothing missing
        CHECK(!ComputePFDiagnostics(img, 1, 0.25f, 8, 16).valid);   // more than half missing
    }

    std::cout << "PASS pf_diagnostics" << std::endl;
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
    if (name == "pf_detection_positive") return test_pf_detection_positive();
    if (name == "pf_detection_negative") return test_pf_detection_negative();
    if (name == "pocs_consistency") return test_pocs_consistency();
    if (name == "pocs_accuracy")    return test_pocs_accuracy();
    if (name == "complex_phase_equivariance") return test_complex_phase_equivariance();
    if (name == "pf_diagnostics") return test_pf_diagnostics();

    std::cerr << "Unknown test: " << name << std::endl;
    return 1;
}
