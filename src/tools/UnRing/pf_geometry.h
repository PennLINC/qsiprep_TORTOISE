#ifndef _PF_GEOMETRY_H
#define _PF_GEOMETRY_H

#include "defines.h"
#include "fftw3.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

// Geometry of an asymmetrically truncated (partial-Fourier) acquisition,
// recovered from the zero-filled region of k-space.
struct PFGeometry
{
    // Low  : the most-negative-ky lines are missing.
    // High : the most-positive-ky lines are missing.
    enum Side { Low, High };

    enum Status
    {
        NoZeroBand,        // full k-space, or a reconstruction that already filled it
        InteriorBand,      // empty band away from the edges: not PF truncation
        ImplausibleFactor, // more than half of k-space missing
        DetectedPF
    };

    Status status;
    bool   is_partial_fourier;   // == (status == DetectedPF)
    int    n_pe;
    int    n_missing;
    float  factor;               // (n_pe - n_missing) / n_pe
    Side   side;
    float  zero_band_energy_ratio;
};

// Index mapping between fftshifted and raw FFT ordering. DC sits at raw index
// 0 and at shifted index n/2.
inline int ShiftedToUnshifted(int i, int n)
{
    return (i + n - n / 2) % n;
}

// Energy (|K|^2) summed along the readout axis, over all slices and up to
// max_volumes volumes, then fftshifted so DC lands at n_pe/2.
inline std::vector<double> ComputeShiftedPEEnergyProfile(ImageType4DComplex::Pointer img,
                                                         int pe_axis, int max_volumes)
{
    ImageType4DComplex::SizeType sz = img->GetLargestPossibleRegion().GetSize();
    const int nx = (int)sz[0], ny = (int)sz[1], nz = (int)sz[2], nt = (int)sz[3];
    const int n_pe = (pe_axis == 0) ? nx : ny;

    std::vector<double> profile(n_pe, 0.0);

    fftw_complex *buf = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny);
    fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny);
    // Same convention as unring.h: first plan dimension is y, buffer is [nx*y+x].
    fftw_plan p = fftw_plan_dft_2d(ny, nx, buf, out, FFTW_FORWARD, FFTW_ESTIMATE);

    int nvol = std::min(nt, max_volumes);
    if (nvol < 1) nvol = 1;

    ImageType4DComplex::IndexType ind;
    for (int t = 0; t < nvol; t++) {
        ind[3] = t;
        for (int z = 0; z < nz; z++) {
            ind[2] = z;
            for (int y = 0; y < ny; y++) {
                ind[1] = y;
                for (int x = 0; x < nx; x++) {
                    ind[0] = x;
                    std::complex<float> v = img->GetPixel(ind);
                    buf[nx * y + x][0] = (double) v.real();
                    buf[nx * y + x][1] = (double) v.imag();
                }
            }
            fftw_execute(p);
            for (int ky = 0; ky < ny; ky++) {
                for (int kx = 0; kx < nx; kx++) {
                    const double re = out[nx * ky + kx][0];
                    const double im = out[nx * ky + kx][1];
                    profile[(pe_axis == 0) ? kx : ky] += re * re + im * im;
                }
            }
        }
    }

    fftw_destroy_plan(p);
    fftw_free(buf);
    fftw_free(out);

    std::vector<double> shifted(n_pe, 0.0);
    for (int i = 0; i < n_pe; i++)
        shifted[i] = profile[ShiftedToUnshifted(i, n_pe)];
    return shifted;
}

// zero_tol is compared against the NORMALISED ENERGY profile, so the 1e-6
// default corresponds to roughly 1e-3 in amplitude.
inline PFGeometry DetectPFGeometry(ImageType4DComplex::Pointer img, int pe_axis,
                                   float zero_tol = 1e-6f, int max_volumes = 8)
{
    ImageType4DComplex::SizeType sz = img->GetLargestPossibleRegion().GetSize();
    const int n_pe = (pe_axis == 0) ? (int)sz[0] : (int)sz[1];

    PFGeometry geom;
    geom.status = PFGeometry::NoZeroBand;
    geom.is_partial_fourier = false;
    geom.n_pe = n_pe;
    geom.n_missing = 0;
    geom.factor = 1.0f;
    geom.side = PFGeometry::Low;
    geom.zero_band_energy_ratio = 0.0f;

    std::vector<double> shifted = ComputeShiftedPEEnergyProfile(img, pe_axis, max_volumes);

    double maxe = 0.0;
    for (int i = 0; i < n_pe; i++)
        maxe = std::max(maxe, shifted[i]);
    if (maxe <= 0.0)
        return geom;                       // all-zero image: nothing to say
    for (int i = 0; i < n_pe; i++)
        shifted[i] /= maxe;

    int best_len = 0, best_start = -1, cur_len = 0, cur_start = 0;
    for (int i = 0; i < n_pe; i++) {
        if (shifted[i] <= (double) zero_tol) {
            if (cur_len == 0) cur_start = i;
            cur_len++;
            if (cur_len > best_len) { best_len = cur_len; best_start = cur_start; }
        } else {
            cur_len = 0;
        }
    }

    if (best_len < 2)
        return geom;                       // no meaningful zero band

    const int best_end = best_start + best_len - 1;
    const bool at_low  = (best_start == 0);
    const bool at_high = (best_end == n_pe - 1);

    if (at_low && at_high)
        return geom;                       // whole profile empty

    // Asymmetric truncation always puts the empty band against one edge of
    // shifted k-space. A band in the interior is something else, and guessing
    // would be worse than declining.
    if (!at_low && !at_high) {
        geom.status = PFGeometry::InteriorBand;
        return geom;
    }

    if (best_len > n_pe / 2) {
        geom.status = PFGeometry::ImplausibleFactor;
        geom.n_missing = best_len;
        geom.factor = (float)(n_pe - best_len) / (float) n_pe;
        return geom;
    }

    geom.status = PFGeometry::DetectedPF;
    geom.is_partial_fourier = true;
    geom.n_missing = best_len;
    geom.factor = (float)(n_pe - best_len) / (float) n_pe;
    geom.side = at_low ? PFGeometry::Low : PFGeometry::High;

    double band = 0.0;
    for (int i = best_start; i <= best_end; i++)
        band += shifted[i];
    geom.zero_band_energy_ratio = (float)(band / (double) best_len);

    return geom;
}

#endif
