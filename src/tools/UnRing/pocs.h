#ifndef _POCS_H
#define _POCS_H

#include "defines.h"
#include "pf_geometry.h"
#include "TORTOISE.h"
#include "fftw3.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>
#include <omp.h>

struct POCSParams
{
    int   iters;
    float tol;
    POCSParams() : iters(10), tol(1e-4f) {}
};

struct POCSResult
{
    int   iters_run;
    float final_rel_change;
};

// Acquired-line mask and the Hamming window over the symmetric fully-sampled
// centre band, both indexed by RAW (unshifted) PE frequency index.
inline void BuildPOCSMasks(const PFGeometry &geom,
                           std::vector<char> &acquired,
                           std::vector<double> &window)
{
    const int n  = geom.n_pe;
    const int nm = geom.n_missing;
    const int c  = n / 2;                        // DC in shifted coordinates

    const int acq_start_s = (geom.side == PFGeometry::Low) ? nm : 0;
    const int acq_end_s   = (geom.side == PFGeometry::Low) ? n - 1 : n - 1 - nm;

    const int a = c - acq_start_s;               // acquired extent below DC
    const int b = acq_end_s - c;                 // acquired extent above DC
    const int m = std::min(a, b);                // symmetric half-width

    acquired.assign(n, 0);
    window.assign(n, 0.0);

    for (int s = acq_start_s; s <= acq_end_s; s++)
        acquired[ShiftedToUnshifted(s, n)] = 1;

    if (m > 0) {
        const double two_pi = 6.283185307179586;
        const int s0 = c - m, s1 = c + m;
        for (int s = s0; s <= s1; s++) {
            const double frac = double(s - s0) / double(s1 - s0);
            // Hamming taper: an untapered band would make the phase estimate
            // itself ring, which is the artifact we are trying to remove.
            window[ShiftedToUnshifted(s, n)] = 0.54 - 0.46 * cos(two_pi * frac);
        }
    } else {
        window[ShiftedToUnshifted(c, n)] = 1.0;  // degenerate: DC alone
    }
}

// POCS partial-Fourier reconstruction, in place. No-op when the geometry says
// there is nothing missing.
inline POCSResult ApplyPOCS(ImageType4DComplex::Pointer img, const PFGeometry &geom,
                            int pe_axis, const POCSParams &params)
{
    POCSResult result;
    result.iters_run = 0;
    result.final_rel_change = 0.0f;

    if (!geom.is_partial_fourier || geom.n_missing == 0)
        return result;

    ImageType4DComplex::SizeType sz = img->GetLargestPossibleRegion().GetSize();
    const int nx = (int)sz[0], ny = (int)sz[1], nz = (int)sz[2], nt = (int)sz[3];
    const int npix = nx * ny;
    const double nfac = 1.0 / double(npix);

    std::vector<char> acquired;
    std::vector<double> window;
    BuildPOCSMasks(geom, acquired, window);

    fftw_plan pf = fftw_plan_dft_2d(ny, nx, NULL, NULL, FFTW_FORWARD,  FFTW_ESTIMATE);
    fftw_plan pb = fftw_plan_dft_2d(ny, nx, NULL, NULL, FFTW_BACKWARD, FFTW_ESTIMATE);

    int    max_iters_run = 0;
    double max_rel = 0.0;

    #pragma omp parallel for
    for (int t = 0; t < nt; t++)
    {
        TORTOISE::EnableOMPThread();

        fftw_complex *x   = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npix);
        fftw_complex *K0  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npix);
        fftw_complex *tmp = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npix);
        fftw_complex *y   = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npix);
        std::vector<double> phase(npix);

        ImageType4DComplex::IndexType ind;
        ind[3] = t;

        for (int z = 0; z < nz; z++)
        {
            ind[2] = z;
            for (int j = 0; j < ny; j++) { ind[1] = j;
                for (int i = 0; i < nx; i++) { ind[0] = i;
                    std::complex<float> v = img->GetPixel(ind);
                    x[nx * j + i][0] = (double) v.real();
                    x[nx * j + i][1] = (double) v.imag();
                } }

            fftw_execute_dft(pf, x, K0);         // measured k-space

            // Low-frequency phase estimate from the symmetric centre band.
            for (int ky = 0; ky < ny; ky++) {
                for (int kx = 0; kx < nx; kx++) {
                    const double w = window[(pe_axis == 0) ? kx : ky];
                    tmp[nx * ky + kx][0] = K0[nx * ky + kx][0] * w;
                    tmp[nx * ky + kx][1] = K0[nx * ky + kx][1] * w;
                }
            }
            fftw_execute_dft(pb, tmp, y);
            for (int q = 0; q < npix; q++)
                phase[q] = atan2(y[q][1], y[q][0]);

            double rel = 0.0;
            int it = 0;
            for (it = 0; it < params.iters; it++)
            {
                for (int q = 0; q < npix; q++) {
                    const double mag = sqrt(x[q][0] * x[q][0] + x[q][1] * x[q][1]);
                    y[q][0] = mag * cos(phase[q]);
                    y[q][1] = mag * sin(phase[q]);
                }
                fftw_execute_dft(pf, y, tmp);

                // Data consistency: measured lines always win. This is what
                // bounds the worst case -- POCS can only alter the region that
                // was zero-filled to begin with.
                for (int ky = 0; ky < ny; ky++) {
                    for (int kx = 0; kx < nx; kx++) {
                        if (acquired[(pe_axis == 0) ? kx : ky]) {
                            tmp[nx * ky + kx][0] = K0[nx * ky + kx][0];
                            tmp[nx * ky + kx][1] = K0[nx * ky + kx][1];
                        }
                    }
                }
                fftw_execute_dft(pb, tmp, y);

                double num = 0.0, den = 0.0;
                for (int q = 0; q < npix; q++) {
                    const double nr = y[q][0] * nfac, ni = y[q][1] * nfac;
                    const double dr = nr - x[q][0], di = ni - x[q][1];
                    num += dr * dr + di * di;
                    den += x[q][0] * x[q][0] + x[q][1] * x[q][1];
                    x[q][0] = nr; x[q][1] = ni;
                }
                rel = (den > 0.0) ? sqrt(num / den) : 0.0;
                if (rel < params.tol) { it++; break; }
            }

            #pragma omp critical
            {
                if (it  > max_iters_run) max_iters_run = it;
                if (rel > max_rel)       max_rel = rel;
            }

            for (int j = 0; j < ny; j++) { ind[1] = j;
                for (int i = 0; i < nx; i++) { ind[0] = i;
                    img->SetPixel(ind, std::complex<float>((float) x[nx * j + i][0],
                                                           (float) x[nx * j + i][1]));
                } }
        }

        fftw_free(x); fftw_free(K0); fftw_free(tmp); fftw_free(y);
        TORTOISE::DisableOMPThread();
    }

    fftw_destroy_plan(pf);
    fftw_destroy_plan(pb);

    result.iters_run = max_iters_run;
    result.final_rel_change = (float) max_rel;
    return result;
}

#endif
