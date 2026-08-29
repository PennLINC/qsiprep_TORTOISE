#ifndef _PF_GEOMETRY_H
#define _PF_GEOMETRY_H

#include "defines.h"
#include "fftw3.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>
#include <cstdlib>

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
        SymmetricBand,     // distinct zero bands at BOTH edges: zero-padding/ZIP, not PF
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

    // Symmetric zero-padding (e.g. ZIP / matrix-interpolated reconstructions)
    // leaves a distinct exact-zero band at BOTH edges of shifted k-space --
    // not just the longest one picked up above. Only looking at whether the
    // single longest run touches both edges (the at_low && at_high case just
    // above) misses this, because the shorter opposite-edge run never became
    // "best". Scan the opposite edge explicitly for its own run.
    if (at_low != at_high) {
        int opp_len = 0;
        if (at_low) {
            // best run touches the low edge; scan inward from the high edge.
            for (int i = n_pe - 1; i >= 0 && shifted[i] <= (double) zero_tol; i--)
                opp_len++;
        } else {
            // best run touches the high edge; scan inward from the low edge.
            for (int i = 0; i < n_pe && shifted[i] <= (double) zero_tol; i++)
                opp_len++;
        }
        if (opp_len >= 2) {
            geom.status = PFGeometry::SymmetricBand;
            return geom;
        }
    }

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


// ---------------------------------------------------------------------------
// Zero-fill compatibility, measured against the conjugate mirror.
//
// DetectPFGeometry above normalises the energy profile by its MAXIMUM, i.e. the DC
// line. That makes its threshold "a fraction of DC", which does not survive real data:
// the DC line dwarfs every line at the edge of k-space, so a genuinely empty band still
// sits far above 1e-6 of DC once there is any noise floor. Real acquisitions that were
// verifiably zero-filled (51x to 403x suppressed relative to their conjugate mirrors)
// were all declined by that test.
//
// The scale-free question is: how much energy does the un-acquired band hold compared to
// the lines that mirror it about k-space centre? ~0 means zero-filled, ~1 means the
// scanner already reconstructed it. That ratio is what should gate POCS.
//
// This is a DIAGNOSTIC. It reports whether an image is *compatible with* zero filling;
// it is deliberately not wired up to enable POCS on its own.
// ---------------------------------------------------------------------------
struct PFDiagnostics
{
    bool  valid;                  // false if the geometry made no sense (e.g. nm < 1)
    int   n_pe;
    int   n_missing;              // from the DECLARED factor, not threshold-derived
    PFGeometry::Side side;        // inferred: whichever side is more suppressed
    float side_asymmetry;         // opposite-side ratio / chosen-side ratio; >>1 is confident
    float mirror_ratio;           // median over slices and volumes
    float mirror_p10, mirror_p90; // spread, so a single pooled number cannot mislead
    int   n_samples;
    bool  zero_fill_compatible;   // mirror_ratio below the threshold below
};

// A band holding under ~5% of its mirror's energy reads as zero-filled. Measured values
// on verified-zero-filled real data were 0.0025 to 0.0198; a vendor-reconstructed image
// should sit near 1.
const float PF_ZEROFILL_MAX_MIRROR_RATIO = 0.05f;

// pf_factor is the DECLARED partial-Fourier factor, normally from BIDS metadata. The band
// width comes from it rather than from a threshold, so the measurement does not depend on
// picking a cutoff. Ratios are formed per slice and per volume and then aggregated by
// median, so high-signal b=0 volumes cannot dominate a pooled energy sum.
inline PFDiagnostics ComputePFDiagnostics(ImageType4DComplex::Pointer img, int pe_axis,
                                          float pf_factor, int max_volumes = 8,
                                          int max_slices = 16)
{
    ImageType4DComplex::SizeType sz = img->GetLargestPossibleRegion().GetSize();
    const int nx = (int)sz[0], ny = (int)sz[1], nz = (int)sz[2], nt = (int)sz[3];
    const int n_pe = (pe_axis == 0) ? nx : ny;
    const int c    = n_pe / 2;

    PFDiagnostics d;
    d.valid = false; d.n_pe = n_pe; d.n_missing = 0;
    d.side = PFGeometry::Low; d.side_asymmetry = 1.0f;
    d.mirror_ratio = 1.0f; d.mirror_p10 = 1.0f; d.mirror_p90 = 1.0f;
    d.n_samples = 0; d.zero_fill_compatible = false;

    const int nm = (int)(llround((1.0 - (double)pf_factor) * (double)n_pe));
    if(nm < 1 || nm > n_pe/2)
        return d;
    d.n_missing = nm;

    fftw_complex *buf = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny);
    fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny);
    fftw_plan p = fftw_plan_dft_2d(ny, nx, buf, out, FFTW_FORWARD, FFTW_ESTIMATE);

    const int vstep = std::max(1, nt / std::max(1, max_volumes));
    const int zstep = std::max(1, nz / std::max(1, max_slices));

    std::vector<double> rlow, rhigh;
    ImageType4DComplex::IndexType ind;

    for(int t = 0; t < nt; t += vstep)
    {
        ind[3] = t;
        for(int z = 0; z < nz; z += zstep)
        {
            ind[2] = z;
            for(int y = 0; y < ny; y++) { ind[1] = y;
                for(int x = 0; x < nx; x++) { ind[0] = x;
                    std::complex<float> v = img->GetPixel(ind);
                    buf[nx*y+x][0] = (double) v.real();
                    buf[nx*y+x][1] = (double) v.imag();
                } }
            fftw_execute(p);

            std::vector<double> prof(n_pe, 0.0);
            for(int ky = 0; ky < ny; ky++)
                for(int kx = 0; kx < nx; kx++)
                {
                    const double re = out[nx*ky+kx][0], im = out[nx*ky+kx][1];
                    prof[(pe_axis == 0) ? kx : ky] += re*re + im*im;
                }
            std::vector<double> sh(n_pe, 0.0);
            for(int i = 0; i < n_pe; i++)
                sh[i] = prof[ShiftedToUnshifted(i, n_pe)];

            // Both candidate sides, so the inference is a comparison rather than a guess.
            // Index 0 is the Nyquist line and has no mirror, so it is skipped.
            double bl = 0.0, ml = 0.0, bh = 0.0, mh = 0.0;
            for(int k = 0; k < nm; k++)
            {
                const int il = k, ih = n_pe - 1 - k;
                const int jl = 2*c - il, jh = 2*c - ih;
                if(il != 0 && jl > 0 && jl < n_pe) { bl += sh[il]; ml += sh[jl]; }
                if(ih != 0 && jh > 0 && jh < n_pe) { bh += sh[ih]; mh += sh[jh]; }
            }
            if(ml > 0.0) rlow.push_back(bl/ml);
            if(mh > 0.0) rhigh.push_back(bh/mh);
        }
    }

    fftw_destroy_plan(p); fftw_free(buf); fftw_free(out);

    if(rlow.empty() || rhigh.empty())
        return d;

    std::sort(rlow.begin(), rlow.end());
    std::sort(rhigh.begin(), rhigh.end());
    const double medl = rlow [rlow.size()/2];
    const double medh = rhigh[rhigh.size()/2];

    const std::vector<double> &chosen = (medl <= medh) ? rlow : rhigh;
    d.side  = (medl <= medh) ? PFGeometry::Low : PFGeometry::High;
    const double med_chosen = (medl <= medh) ? medl : medh;
    const double med_other  = (medl <= medh) ? medh : medl;

    d.mirror_ratio   = (float) med_chosen;
    d.mirror_p10     = (float) chosen[(size_t)(0.10*(chosen.size()-1))];
    d.mirror_p90     = (float) chosen[(size_t)(0.90*(chosen.size()-1))];
    d.side_asymmetry = (med_chosen > 0.0) ? (float)(med_other/med_chosen) : 1e9f;
    d.n_samples      = (int) chosen.size();
    d.valid          = true;
    d.zero_fill_compatible = (d.mirror_ratio < PF_ZEROFILL_MAX_MIRROR_RATIO);
    return d;
}

#endif
