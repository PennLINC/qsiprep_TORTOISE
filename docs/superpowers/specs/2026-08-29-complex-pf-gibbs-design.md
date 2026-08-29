# Partial-Fourier-Aware Gibbs Correction from Complex-Valued Data

Design spec — 2026-08-29
Tracking issue: [PennLINC/qsiprep#1099](https://github.com/PennLINC/qsiprep/issues/1099)

## Problem

TORTOISE's `Gibbs` command corrects Gibbs ringing on magnitude images. For
partial-Fourier (PF) acquisitions it uses the RPG method of Lee et al., which
removes the extra ringing period caused by asymmetric k-space truncation
*without* access to phase or k-space data. That is the right method when phase
is unavailable, and the wrong one when it is not: the same paper reports that
PF reconstruction followed by local subvoxel shifts (SuShi) on the complex image
is more robust, and that RPG degrades at aggressive PF factors such as 5/8.

QSIPrep already carries BIDS `part-mag` / `part-phase` DWI pairs through to its
denoising stage and can assemble a complex image from them. Nothing downstream
can use that complex signal for unringing today.

## Goal

A new `GibbsComplex` command that takes a complex-valued 4D NIfTI, optionally
restores the un-acquired PF k-space region with POCS, and applies the existing
Kellner SuShi algorithm to the complex data rather than to its magnitude.

## Scope

In scope:

- A new `GibbsComplex` executable reading and writing COMPLEX64 4D NIfTI.
- Complex-valued SuShi, reusing the existing `unring_2d` / `unring_1D` core.
- Automatic detection of PF geometry (factor and truncated side) from k-space.
- POCS partial-Fourier reconstruction, enabled by default, disableable.
- ctest unit coverage and a Python phantom validation harness.

Out of scope:

- Slice-direction partial Fourier (3D acquisitions). In-plane phase encoding
  only; detection reports "not partial Fourier" rather than guessing.
- Any change to the existing `Gibbs` executable or to `UnRingFull`,
  `UnRing78`, `UnRing68`.
- Routing complex data through `TORTOISEProcess`.
- Magnitude + phase file pairs as input. QSIPrep assembles the complex NIfTI.
- Automatic fallback to the RPG path. When PF is present and POCS is off, the
  tool warns; it does not silently switch algorithms.

## Decisions

| Question | Decision |
|---|---|
| Algorithm | Complex SuShi as the base stage; POCS as a separately testable stage controlled by a flag. |
| Delivery | New `GibbsComplex` binary. The `Gibbs` binary and its positional-argument contract are untouched. |
| PF geometry | Auto-detected from the k-space zero band; CLI flags override; a declared-vs-detected mismatch is an error unless forced. |
| Output | COMPLEX64 NIfTI by default, with an optional magnitude output written in the same pass. |
| Testing | ctest unit tests inside the build, plus a Python end-to-end phantom harness. |
| Build | Docker, reusing the existing qsiprep TORTOISE build recipe against a bind-mounted checkout. |

The key enabling fact: `unring_1D` and `unring_2d` in `src/tools/UnRing/unring.h`
already operate on `fftw_complex` buffers end to end. Only the marshalling loops
in the `UnRing*` wrappers discard the imaginary part, setting `data[i][1] = 0` on
read and taking `res[i][0]` on write. Complex SuShi therefore needs no change to
the algorithm core.

## Architecture

Five units.

### 1. Complex image I/O

`defines.h` gains:

```cpp
using ImageType4DComplex = itk::Image<std::complex<float>, 4>;
```

with explicit `readImageD` / `writeImageD` instantiations in `defines.cxx`.

The bundled `itkNiftiImageIOHeader` already maps `NIFTI_TYPE_COMPLEX64` and
`NIFTI_TYPE_COMPLEX128` to ITK's `COMPLEX` pixel type in both directions
(`src/utilities/itkNiftiImageIOHeader.cxx` lines 953-960 and 1471-1484), so this
is plumbing rather than new IO code.

### 2. PF geometry detection — `src/tools/UnRing/pf_geometry.h`

```cpp
struct PFGeometry
{
    bool  is_partial_fourier;
    int   n_pe;                    // matrix size along the PE axis
    int   n_missing;               // number of zero-filled lines
    float factor;                  // (n_pe - n_missing) / n_pe
    enum Side { Low, High } side;  // Low = most-negative-ky end is missing
    float zero_band_energy_ratio;  // evidence, for the log
};

PFGeometry DetectPFGeometry(ImageType4DComplex::Pointer img,
                            int pe_axis,
                            float zero_tol = 1e-6f,
                            int max_volumes = 8);
```

Method: 2D FFT each in-plane slice; accumulate `profile[j] = sum |K(i,j,z,v)|^2`
over the readout index `i`, all slices `z`, and up to `max_volumes` sampled
volumes; normalise by the profile maximum; fftshift so DC sits at `n_pe/2`; find
the longest contiguous run below `zero_tol`.

The run must touch one end of the shifted profile — that is what asymmetric
truncation looks like. A below-threshold run in the interior means something
other than PF truncation, and detection reports `is_partial_fourier = false`.
`factor` and `side` follow from the run's length and which end it touches:
`Low` when the run sits at the start of the shifted profile (the most-negative
ky lines are missing), `High` when it sits at the end.

`zero_tol` is compared against the *normalised energy* profile, so the default
1e-6 corresponds to roughly 1e-3 in amplitude.

Guards: a run shorter than 2 lines is treated as no run; a run longer than
`n_pe / 2` (factor below 0.5) is refused as implausible.

On `pe_axis`: the driver transposes first, so in `GibbsComplex` this argument is
always 1. It stays a parameter so that `DetectPFGeometry` and `ApplyPOCS` can be
unit-tested on either axis without going through the transpose.

This one function serves two purposes. It supplies the geometry POCS needs, and
it answers the question the issue raises about vendor reconstructions: data that
has already been through homodyne or POCS reconstruction has no zero band, so
detection reports "not partial Fourier" and POCS is skipped rather than being
applied a second time.

### 3. POCS reconstruction — `src/tools/UnRing/pocs.h`

```cpp
struct POCSParams { int iters = 10; float tol = 1e-4f; };
struct POCSResult { int iters_run; float final_rel_change; };

POCSResult ApplyPOCS(ImageType4DComplex::Pointer img,  // modified in place
                     const PFGeometry &geom,
                     int pe_axis,
                     const POCSParams &params);
```

For each in-plane slice of each volume:

1. `K = FFT2(slice)`; `A` is the acquired line set from `geom`.
2. Symmetric band `S`: the region symmetric about DC that lies wholly inside
   `A`. With DC at `c` and `A` spanning `[c-a, c+b]`, `S = [c-m, c+m]` where
   `m = min(a, b)`.
3. Phase estimate `p = angle(IFFT2(K * W))`, `W` a Hamming window over `S` and
   zero outside. The taper keeps the phase estimate itself from ringing.
4. Iterate: `y = |x| * exp(i p)`; `K' = FFT2(y)`; `K'[A] = K[A]`;
   `x = IFFT2(K')`.
5. Stop at `iters` or when `||x_new - x|| / ||x|| < tol`.

Step 4's data-consistency substitution is what bounds the worst case: measured
k-space lines are always restored exactly, so POCS can only alter the region
that was zero-filled to begin with.

Parallelised over volumes with the same OpenMP structure the existing `UnRing*`
functions use, including `TORTOISE::EnableOMPThread()` / `DisableOMPThread()`.

### 4. Complex SuShi — addition to `src/tools/UnRing/unring.h`

```cpp
ImageType4DComplex::Pointer UnRingFullComplex(ImageType4DComplex::Pointer input_img,
                                              int nsh = 25, int minW = 1, int maxW = 3);
```

Structurally a copy of `UnRingFull`'s volume loop with two changes: the read loop
takes both components (`data[i][0] = real`, `data[i][1] = imag`) instead of
zeroing the imaginary part, and the write loop stores both components of
`res_complex` instead of only `[0]`. FFTW plan setup, the OpenMP loop, and the
call into `unring_2d` are unchanged.

`UnRingFull`, `UnRing78`, and `UnRing68` are not modified. The repeated FFTW
plan boilerplate across those three functions is worth factoring out on its own
merits, but they are used by `TORTOISEProcess`, have no test coverage today, and
refactoring them buys this feature nothing — so they stay as they are.

### 5. Driver — `gibbs_complex_main.cxx` + `gibbs_complex_parser.{h,cxx}`

Uses the existing `antsCommandLineParser`, matching the pattern of
`extract_dwi_subset_parser.cxx` and the other tool parsers, so the CLI is named
flags rather than the positional contract `Gibbs` uses.

New CMake target alongside the existing one:

```cmake
add_executable(GibbsComplex ../src/tools/UnRing/gibbs_complex_main.cxx
                            ../src/tools/UnRing/gibbs_complex_parser.cxx
                            ../src/tools/ResampleDWIs/resample_dwis.cxx ${SOURCES})
target_link_libraries(GibbsComplex ${ITK_LIBRARIES} ${Boost_LIBRARIES} fftw3)
```

`QSIPREP=1` gates only one block in `CMakeLists.txt` (line 179) and `Gibbs` is
added unconditionally, so `GibbsComplex` is present in the reduced qsiprep build.

## CLI contract

```
GibbsComplex -i <complex_4d.nii.gz> -o <corrected_complex.nii.gz> [options]

  -i, --input               Input complex-valued 4D NIfTI (COMPLEX64/128). Required.
  -o, --output              Output complex-valued 4D NIfTI (COMPLEX64). Required.
      --output_magnitude    Also write the magnitude to this path. Optional.
      --pe_dir              Phase encoding direction, 0 = horizontal, 1 = vertical.
                            Default 1. Same convention as the Gibbs command.
      --pocs                Run POCS PF reconstruction. Default 1.
      --pocs_iters          POCS iterations. Default 10.
      --pocs_tol            POCS relative-change stopping tolerance. Default 1e-4.
      --pf_factor           Override the detected PF factor (e.g. 0.75, 0.875).
      --pf_side             Override the detected truncated side: low | high.
      --force_pf            Proceed when declared and detected geometry disagree.
      --nsh                 SuShi subvoxel shifts. Default 25.
      --minW                SuShi minimum window. Default 1.
      --maxW                SuShi maximum window. Default 3.
      --ncores              Cap ITK and OpenMP thread counts.
      --disable_itk_threads Pin ITK to one thread.
```

`--ncores` and `--disable_itk_threads` replicate the handling added to
`gibbs_main.cxx`: applied *after* the `TORTOISE` constructor, which sets
`omp_set_num_threads()` to the host core count regardless of `OMP_NUM_THREADS`.

## Data flow

```
read complex 4D NIfTI
  -> if --pe_dir 0, transpose in-plane axes (the trick gibbs_main.cxx uses)
  -> DetectPFGeometry
  -> reconcile detection against --pf_factor / --pf_side
  -> if PF and --pocs: ApplyPOCS   (k-space now full)
  -> UnRingFullComplex
  -> if --pe_dir 0, transpose back and restore original direction/spacing/origin
  -> write COMPLEX64; optionally write magnitude
```

## Error handling

| Condition | Behaviour |
|---|---|
| Input is not a complex NIfTI | Error and exit. Name the datatype found and point at `Gibbs` for magnitude data. |
| Detection finds no zero band, `--pocs 1`, no override | Log that the data does not look zero-filled, skip POCS, run complex SuShi, exit 0. |
| Detection finds PF, `--pocs 0` | Log a warning that residual PF ringing may remain and name the RPG path in `Gibbs`. Run complex SuShi, exit 0. |
| Declared and detected geometry disagree | Error and exit, printing both, unless `--force_pf`, in which case the declared values win with a warning. |
| Detected factor below 0.5 | Error and exit as implausible. |
| Zero band in the interior of shifted k-space | Report not-PF; skip POCS. |
| POCS reaches the iteration cap without meeting `tol` | Log the final relative change and continue. Not an error. |
| `--output_magnitude` given but unwritable | Error after the complex output is written, so work is not lost. |

## Testing

No test infrastructure exists in this repository today — no `enable_testing()`,
no `add_test`, no CI. This adds the minimum that makes the new code verifiable,
without introducing a test-framework dependency: `enable_testing()`, a
`GibbsComplexTests` executable using plain asserts and a nonzero exit, and
`add_test`.

| # | Test | Asserts |
|---|---|---|
| 1 | Complex NIfTI round trip | Real and imaginary parts survive write-then-read exactly; on-disk datatype is `NIFTI_TYPE_COMPLEX64`. |
| 2 | Magnitude equivalence | `UnRingFullComplex` on an image with zero imaginary part matches `UnRingFull` to 1e-9 relative. |
| 3 | PF detection, positive | Correct factor and side for 6/8 and 7/8, both sides, even and odd `n_pe`. |
| 4 | PF detection, negative | Full k-space in, `is_partial_fourier == false` out. |
| 5 | POCS consistency and identity | Acquired lines bit-preserved after iteration; POCS on full k-space is a no-op. |
| 6 | POCS accuracy | On a known complex phantom, RMSE to ground truth is at least 10% below the zero-filled RMSE. Margin calibrated when the test is written, then fixed. |

Test 2 is the regression anchor. `unring_2d` already carries an imaginary
channel, so with zero imaginary input every intermediate imaginary value is zero
and the real output should be near-identical. A failure there means the
marshalling is wrong, which is the only place the complex path can go wrong
silently.

### Python phantom harness

Lives in `src/tools/UnRing/validation/`, run with `micromamba run -n linc311`
(numpy 1.26, nibabel 5.3, scipy 1.16 confirmed present).

1. Build a Shepp-Logan-style complex phantom with a smooth background phase.
2. Forward FFT; truncate to 6/8 and 7/8; zero-fill; inverse FFT.
3. Write COMPLEX64 NIfTI.
4. Run three arms: `GibbsComplex` with POCS, `GibbsComplex` without POCS, and
   the existing magnitude `Gibbs` RPG path.
5. Score RMSE against ground truth, a ringing metric (total variation in a band
   beside each sharp edge), and edge profiles. Write a short report.

Acceptance is the claim the issue actually makes: **complex + POCS beats
magnitude RPG, and the margin is larger at 6/8 than at 7/8.** Numeric thresholds
are pinned from the first clean run and recorded in the harness — inventing pass
marks before any measurement exists would make the harness worse, not better.

Validation on real complex DWI, if data is available, stays a manual check at
the end.

## Build environment

`/mnt/c/Users/tsalo/Documents/linc/qsiprep_build/Dockerfile_TORTOISE` already
builds this project:

- base `pennlinc/qsiprep-ants:26.1.2`, which carries ITK at
  `/tmp/ants/build/ITKv5-build`
- apt: `libfftw3-dev`, `libboost-{dev,iostreams,filesystem,system,regex}-dev`,
  `libeigen3-dev`, `zlib1g-dev`, `build-essential`, `cmake`
- `cmake . -DUSECUDA=0 -DQSIPREP=1 -DITK_DIR=/tmp/ants/build/ITKv5-build -DISDEBUG=0 && make`

Phase 0 adapts it into a dev image that bind-mounts this checkout instead of
cloning upstream, so edits are picked up and `ctest` runs in the same container.
The dev Dockerfile lives in the session scratchpad, not the repository.

This WSL host cannot build natively: no ITK anywhere under `/home/tsalo`,
`/usr/local`, or `/opt`; no Boost headers; only the fftw3 runtime
(`libfftw3.so.3`, no `fftw3.h`). `cmake` 3.28.3 and `g++` 13.3.0 are present,
Docker Desktop is reachable, and `pennlinc/qsiprep:unstable` is already pulled.

Bind-mounting from `/mnt/c` is slow. If compile times become painful, copy the
tree into the container filesystem and rsync results back.

## Repository hygiene

The working tree is a whole-repo CRLF/LF flip: 373 files, 136,183 insertions and
exactly 136,183 deletions. The tree is CRLF, HEAD blobs are LF,
`core.autocrlf=false`, and there is no `.gitattributes`.

Consequence: every commit in this work uses `git add` with explicit paths. A
`git add -A` would sweep the line-ending change into the commit and bury the
feature. New files are written with LF endings.

Fixing the CRLF situation is unrelated to this feature and is not part of this
work.

## Phases

Each phase is test-first and independently committable.

| Phase | Deliverable | Verified by |
|---|---|---|
| 0 | Dev Docker image bind-mounting this checkout | `Gibbs` builds and runs `--help` in the container |
| 1 | `ImageType4DComplex` + read/write instantiations | Test 1 |
| 2 | `UnRingFullComplex` | Test 2 |
| 3 | `DetectPFGeometry` | Tests 3, 4 |
| 4 | `ApplyPOCS` | Tests 5, 6 |
| 5 | `GibbsComplex` driver, parser, CMake target | Binary runs end to end on a synthetic input |
| 6 | Python phantom harness | Acceptance comparison against RPG; thresholds pinned |
| 7 | Docs | `GibbsComplex` section in `TORTOISEV4/tools/README.md` |

After phase 2 the tool is already useful on its own: complex SuShi with no PF
handling, equivalent to complex-input `mrdegibbs`.

Note for phase 7: the standalone `Gibbs` tool is not documented in
`TORTOISEV4/tools/README.md` today either. Adding an entry for it is optional
scope, excluded unless requested.

## Risks

**Motion-induced phase in diffusion data.** POCS assumes the image phase is
smooth enough to be captured by a low-frequency estimate. Diffusion-weighted
volumes, especially at high b-value, can carry rapid motion-induced phase that
violates this. The data-consistency step bounds the damage — measured lines are
always restored — but the restored region may be poor. This is the main reason
real-data validation matters and the main reason POCS is a flag rather than
unconditional.

**Vendor reconstruction.** Data that was zero-filled and then lightly filtered
has a near-zero rather than exactly-zero band. `zero_tol` is tunable and the
detection result is logged with its energy ratio so a user can see what was
decided and why.

**ITK complex NIfTI scaling.** `scl_slope` / `scl_inter` handling for complex
pixel types is worth confirming in phase 1 rather than assuming; test 1's exact
round-trip assertion is what catches it.

**Axis conventions.** The `--pe_dir 0` transpose, the fftshift convention in
detection, and which end of shifted k-space counts as "low" all have to agree.
Tests 3 and 4 cover both sides and both parities specifically for this reason.

## References

- Lee HH, Novikov DS, Fieremans E. Removal of partial Fourier-induced Gibbs
  (RPG) ringing artifacts in MRI. *Magn Reson Med.* 2021;86:2733-2750.
  [PMC9212190](https://pmc.ncbi.nlm.nih.gov/articles/PMC9212190/)
- Kellner E, Dhital B, Kiselev VG, Reisert M. Gibbs-ringing artifact removal
  based on local subvoxel-shifts. *Magn Reson Med.* 2016;76:1574-1581.
- Muckley MJ, Ades-Aron B, Papaioannou A, et al. Training a neural network for
  Gibbs and noise removal in diffusion MRI. *Magn Reson Med.* 2021.
  [PMC7722184](https://pmc.ncbi.nlm.nih.gov/articles/PMC7722184/)
- [mchiew/partial-fourier-tutorial](https://github.com/mchiew/partial-fourier-tutorial)
  — reference implementations of zero filling, conjugate synthesis, Margosian,
  and POCS.
