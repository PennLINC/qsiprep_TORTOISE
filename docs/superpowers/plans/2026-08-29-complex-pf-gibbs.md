# Partial-Fourier-Aware Complex Gibbs Correction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `GibbsComplex` command to TORTOISE that reads a complex-valued 4D NIfTI, optionally restores un-acquired partial-Fourier k-space with POCS, and runs the existing Kellner subvoxel-shift algorithm on the complex data instead of its magnitude.

**Architecture:** Five units — complex NIfTI I/O, k-space PF geometry detection, POCS reconstruction, a complex-valued SuShi wrapper, and a CLI driver. The SuShi core (`unring_2d` / `unring_1D`) already operates on `fftw_complex` buffers end to end and is **not modified**; only the marshalling into and out of it changes. The existing `Gibbs` binary and the `UnRingFull` / `UnRing78` / `UnRing68` functions are left untouched.

**Tech Stack:** C++14, ITK 5.3 (via `pennlinc/qsiprep-ants:26.1.2`), FFTW3, Boost, OpenMP, CMake 3.20+, ctest. Python 3.11 with numpy/nibabel/scipy in the `linc311` micromamba environment for the validation harness.

**Spec:** `docs/superpowers/specs/2026-08-29-complex-pf-gibbs-design.md`

## Global Constraints

- **Never run `git add -A` or `git add .`** The working tree is a whole-repo CRLF/LF flip: 373 files, 136,183 insertions and exactly 136,183 deletions, `core.autocrlf=false`, no `.gitattributes`. Every commit uses `git add` with explicit file paths. Write all new files with LF line endings.
- **Do not modify** `UnRingFull`, `UnRing78`, `UnRing68`, or `gibbs_main.cxx`. They are used by `TORTOISEProcess` and have no test coverage.
- Build flags are fixed: `-DUSECUDA=0 -DQSIPREP=1 -DITK_DIR=/tmp/ants/build/ITKv5-build -DISDEBUG=0`.
- C++ standard is C++14 (set by `CMAKE_CXX_FLAGS` in `TORTOISEV4/CMakeLists.txt`). No C++17 constructs.
- The `--pe_dir` convention matches the existing `Gibbs` command exactly: `0` = horizontal, `1` = vertical. Default `1`.
- Default parameter values, used identically in the parser and in every doc string: `nsh=25`, `minW=1`, `maxW=3`, `pocs=1`, `pocs_iters=10`, `pocs_tol=1e-4`, `zero_tol=1e-6`.
- `PFGeometry::Side::Low` means **the most-negative-ky lines are missing**. `High` means the most-positive-ky lines are missing.
- All FFT index arithmetic follows the convention already used in `unring.h`: `fftw_plan_dft_2d(sz[1], sz[0], ...)` paired with buffer indexing `buf[sz[0]*y + x]`, so the **first** plan dimension is the y/PE axis.
- `zero_tol` is compared against the **normalised energy** (`|K|^2`) profile, not amplitude.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `src/main/defines.h` | Modify | Add the `ImageType4DComplex` type alias. |
| `src/main/defines.cxx` | Modify | Explicit `readImageD` / `writeImageD` instantiations for the complex type. |
| `src/tools/UnRing/unring.h` | Modify (append only) | Add `UnRingFullComplex`. Existing functions untouched. |
| `src/tools/UnRing/pf_geometry.h` | Create | `PFGeometry` struct and `DetectPFGeometry`. No dependency on POCS or SuShi. |
| `src/tools/UnRing/pocs.h` | Create | `POCSParams`, `POCSResult`, `ApplyPOCS`. Depends only on `pf_geometry.h`. |
| `src/tools/UnRing/gibbs_complex_parser.h` | Create | CLI option declarations. |
| `src/tools/UnRing/gibbs_complex_parser.cxx` | Create | CLI option definitions and typed getters. |
| `src/tools/UnRing/gibbs_complex_main.cxx` | Create | Driver: read, transpose, detect, POCS, SuShi, untranspose, write. |
| `src/tools/UnRing/tests/test_gibbs_complex.cxx` | Create | All six unit tests, dispatched by name from `argv[1]`. |
| `src/tools/UnRing/validation/run_validation.py` | Create | Python phantom harness and scoring. |
| `TORTOISEV4/CMakeLists.txt` | Modify | `enable_testing()`, `GibbsComplex` target, `GibbsComplexTests` target, six `add_test` entries. |
| `TORTOISEV4/tools/README.md` | Modify | `GibbsComplex` documentation entry. |

`pf_geometry.h` and `pocs.h` are header-only, matching the existing `unring.h` pattern in this directory.

---

## Task 0: Dev build environment in Docker

This WSL host cannot build natively — no ITK, no Boost headers, only the fftw3 runtime. Everything downstream depends on this task.

**Files:**
- Create: `<scratchpad>/Dockerfile.dev` (scratchpad only, **not** committed)
- Create: `<scratchpad>/tbuild.sh` (scratchpad only, **not** committed)

**Interfaces:**
- Consumes: nothing.
- Produces: a `tbuild.sh <make-target>` command that compiles any target and leaves binaries in `/opt/bin` inside a persistent Docker volume, and `tbuild.sh ctest` to run the test suite. Every later task uses these.

- [ ] **Step 1: Write the dev Dockerfile**

Replace `<scratchpad>` with your actual scratchpad directory path.

```dockerfile
FROM pennlinc/qsiprep-ants:26.1.2

RUN apt update && apt install --no-install-recommends -y \
      zlib1g-dev \
      build-essential \
      libeigen3-dev \
      fftw3 \
      libfftw3-dev \
      cmake \
      cmake-data \
      libboost-dev \
      libboost-iostreams-dev \
      libboost-filesystem-dev \
      libboost-system-dev \
      libboost-regex-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /var/tmp/*

CMD ["/bin/bash"]
```

This mirrors `/mnt/c/Users/tsalo/Documents/linc/qsiprep_build/Dockerfile_TORTOISE`, minus the `git clone` — the source comes from a bind mount instead so edits are picked up without rebuilding the image.

- [ ] **Step 2: Build the image**

```bash
docker build -t tortoise-dev -f <scratchpad>/Dockerfile.dev <scratchpad>
```

Expected: `naming to docker.io/library/tortoise-dev`. The `pennlinc/qsiprep-ants:26.1.2` base pull runs once and is several GB.

- [ ] **Step 3: Write the build helper script**

```bash
#!/usr/bin/env bash
# <scratchpad>/tbuild.sh — build or test TORTOISE in the dev container.
# Usage: tbuild.sh <make-target> | tbuild.sh ctest [ctest-args...]
set -euo pipefail

REPO=/mnt/c/Users/tsalo/Documents/linc/qsiprep_TORTOISE

if [ "${1:-}" = "ctest" ]; then
    shift
    CMD="cd /opt/tbuild && ctest --output-on-failure $*"
else
    TARGET="${1:-all}"
    CMD="mkdir -p /opt/tbuild && cd /opt/tbuild \
         && cmake /src/TORTOISEV4/TORTOISEV4 \
              -DUSECUDA=0 -DQSIPREP=1 \
              -DITK_DIR=/tmp/ants/build/ITKv5-build \
              -DISDEBUG=0 \
         && make -j\$(nproc) ${TARGET}"
fi

docker run --rm \
    -v "${REPO}":/src/TORTOISEV4 \
    -v tortoise-build:/opt/tbuild \
    -v tortoise-bin:/opt/bin \
    tortoise-dev bash -lc "${CMD}"
```

Then `chmod +x <scratchpad>/tbuild.sh`.

Two things this arranges deliberately. The build tree lives in the named volume `tortoise-build`, not on the bind mount, so `cmake` never drops `CMakeCache.txt` or `CMakeFiles/` into the git tree and compilation does not pay the `/mnt/c` filesystem penalty. And because `TORTOISEV4/CMakeLists.txt` sets `CMAKE_RUNTIME_OUTPUT_DIRECTORY` to `${CMAKE_BINARY_DIR}/../bin/`, a build root of `/opt/tbuild` puts binaries in `/opt/bin` — building at `/build` would instead target the system `/bin`, which must be avoided.

- [ ] **Step 4: Verify the existing Gibbs target builds**

```bash
<scratchpad>/tbuild.sh Gibbs
```

Expected: CMake configures, then `[100%] Built target Gibbs`. First run takes several minutes.

- [ ] **Step 5: Verify the binary runs**

```bash
docker run --rm -v tortoise-bin:/opt/bin tortoise-dev /opt/bin/Gibbs
```

Expected: `Usage: Gibbs input_nifti  output_nifti kspace_coverage(1,0.875,0.75) phase_encoding_dir(0: horizontal, 1:vertical) nsh(optional) minW(optional) maxW(optional) [--ncores N] [--disable_itk_threads]`

- [ ] **Step 6: No commit**

Task 0 produces only scratchpad files. Nothing is committed. Confirm with `git status --short` that the repo is unchanged apart from the pre-existing CRLF noise.

---

## Task 1: Complex NIfTI I/O and the test harness

**Files:**
- Modify: `src/main/defines.h`
- Modify: `src/main/defines.cxx:37-56`
- Create: `src/tools/UnRing/tests/test_gibbs_complex.cxx`
- Modify: `TORTOISEV4/CMakeLists.txt`

**Interfaces:**
- Consumes: `tbuild.sh` from Task 0.
- Produces:
  - `ImageType4DComplex` = `itk::Image<std::complex<float>,4>`
  - `ImageType4DComplex::Pointer readImageD<ImageType4DComplex>(std::string)`
  - `void writeImageD<ImageType4DComplex>(ImageType4DComplex::Pointer, std::string)`
  - Test binary contract: `GibbsComplexTests <test_name>` returns 0 on pass, 1 on failure. Later tasks append cases to the same `main` dispatch.
  - Helper available to all later tests: `ImageType4DComplex::Pointer MakeComplexImage(int nx,int ny,int nz,int nt)`

- [ ] **Step 1: Write the failing test**

Create `src/tools/UnRing/tests/test_gibbs_complex.cxx`:

```cpp
#include "defines.h"
#include "itkImageIOFactory.h"
#include "itkImageIOBase.h"
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

int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: GibbsComplexTests <test_name>" << std::endl;
        return 1;
    }
    std::string name(argv[1]);

    if (name == "complex_io_roundtrip") return test_complex_io_roundtrip();

    std::cerr << "Unknown test: " << name << std::endl;
    return 1;
}
```

If `itk::IOPixelEnum::COMPLEX` or `itk::IOComponentEnum::FLOAT` fails to compile against this ITK build, substitute `itk::ImageIOBase::COMPLEX` and `itk::ImageIOBase::FLOAT`. Likewise if `itk::ImageIOFactory::FileModeEnum::ReadMode` fails, use `itk::ImageIOFactory::ReadMode`. These are the ITK 5.3-vs-5.4 enum-scoping differences the repo already carries a shim for (commit `0fb03ca`).

- [ ] **Step 2: Add the test target to CMake**

In `TORTOISEV4/CMakeLists.txt`, immediately after the `PROJECT(TORTOISEV4  )` line near the top, add:

```cmake
enable_testing()
```

Then immediately after the existing `Gibbs` target block (currently lines 255-256), add:

```cmake
add_executable(GibbsComplexTests ../src/tools/UnRing/tests/test_gibbs_complex.cxx ../src/tools/ResampleDWIs/resample_dwis.cxx ${SOURCES})
target_link_libraries(GibbsComplexTests ${ITK_LIBRARIES} ${Boost_LIBRARIES} fftw3 )

add_test(NAME complex_io_roundtrip COMMAND GibbsComplexTests complex_io_roundtrip)
```

- [ ] **Step 3: Run the test to verify it fails**

```bash
<scratchpad>/tbuild.sh GibbsComplexTests
```

Expected: compile error — `ImageType4DComplex` was not declared, and no `readImageD`/`writeImageD` instantiation exists for it. That failure is the point: it proves the test exercises code that does not yet exist.

- [ ] **Step 4: Add the type alias**

In `src/main/defines.h`, add `#include <complex>` alongside the existing includes, and add this line directly after the `ImageType4D` alias (currently line 34):

```cpp
using ImageType4DComplex=itk::Image<std::complex<float>,4>;
```

- [ ] **Step 5: Add the template instantiations**

In `src/main/defines.cxx`, after the existing `ImageType4D` instantiation pair (currently lines 44-45), add:

```cpp
template ImageType4DComplex::Pointer readImageD<ImageType4DComplex>(std::string) ;
template void writeImageD<ImageType4DComplex>(ImageType4DComplex::Pointer , std::string);
```

- [ ] **Step 6: Run the test to verify it passes**

```bash
<scratchpad>/tbuild.sh GibbsComplexTests && <scratchpad>/tbuild.sh ctest -R complex_io_roundtrip
```

Expected: `PASS complex_io_roundtrip` and `100% tests passed, 0 tests failed out of 1`.

If the datatype assertion fails while the value comparison passes, the writer chose a two-component vector encoding instead of COMPLEX64. That is a real bug for downstream consumers, not a test artifact — fix the writer path rather than relaxing the assertion.

- [ ] **Step 7: Commit**

```bash
git add src/main/defines.h src/main/defines.cxx \
        src/tools/UnRing/tests/test_gibbs_complex.cxx \
        TORTOISEV4/CMakeLists.txt
git commit -m "Add complex 4D image type and NIfTI round-trip test"
```

---

## Task 2: Complex-valued SuShi

After this task the tool is already useful on its own — complex subvoxel-shift unringing with no PF handling, equivalent to complex-input `mrdegibbs`.

**Files:**
- Modify: `src/tools/UnRing/unring.h` (append after the `ImageType3D` overload of `UnRingFull`, which currently ends at line 426)
- Modify: `src/tools/UnRing/tests/test_gibbs_complex.cxx`
- Modify: `TORTOISEV4/CMakeLists.txt`

**Interfaces:**
- Consumes: `ImageType4DComplex`, `readImageD`/`writeImageD` (Task 1); `unring_2d` and `my_plans_struct` (existing, unmodified).
- Produces: `ImageType4DComplex::Pointer UnRingFullComplex(ImageType4DComplex::Pointer input_img, int nsh=25, int minW=1, int maxW=3)`

- [ ] **Step 1: Write the failing test**

Add to `src/tools/UnRing/tests/test_gibbs_complex.cxx`. Add `#include "unring.h"` to the includes at the top, then add this function above `main`:

```cpp
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
```

And register it in `main` next to the existing dispatch line:

```cpp
    if (name == "magnitude_equivalence") return test_magnitude_equivalence();
```

- [ ] **Step 2: Register the test in CMake**

In `TORTOISEV4/CMakeLists.txt`, after the `complex_io_roundtrip` test line:

```cmake
add_test(NAME magnitude_equivalence COMMAND GibbsComplexTests magnitude_equivalence)
```

- [ ] **Step 3: Run the test to verify it fails**

```bash
<scratchpad>/tbuild.sh GibbsComplexTests
```

Expected: compile error — `UnRingFullComplex` was not declared in this scope.

- [ ] **Step 4: Implement UnRingFullComplex**

Append to `src/tools/UnRing/unring.h`, directly after the `ImageType3D` overload of `UnRingFull` ends (line 426) and before `SplitImageRows`:

```cpp
// Complex-valued variant of UnRingFull. unring_2d already carries both
// components through its fftw_complex buffers, so this differs from the real
// version only in the marshalling: the imaginary channel is read from the
// input instead of being zeroed, and both components are written back.
ImageType4DComplex::Pointer UnRingFullComplex(ImageType4DComplex::Pointer input_img, int nsh=25, int minW=1,int maxW=3)
{
    typedef itk::ImageDuplicator<ImageType4DComplex> DupType;
    DupType::Pointer dup= DupType::New();
    dup->SetInputImage(input_img);
    dup->Update();
    ImageType4DComplex::Pointer output_img= dup->GetOutput();
    ImageType4DComplex::SizeType sz= output_img->GetLargestPossibleRegion().GetSize();

    int dim_sz[4];
    dim_sz[0] = sz[0];
    dim_sz[1] = sz[1];
    dim_sz[2] = 1;
    dim_sz[3] = 1 ;

    auto stream = (TORTOISE::stream);
    if(stream)
        (*stream)<<  "Complex Gibbs ringing correction of volume: " <<  std::flush;
    else
        std::cout<<  "Complex Gibbs ringing correction of volume: " <<  std::flush;

    my_plans_struct my_plans;
    fftw_plan p = fftw_plan_dft_2d(dim_sz[1],dim_sz[0], NULL, NULL, FFTW_FORWARD, FFTW_ESTIMATE);
    my_plans.p2d = &p;
    fftw_plan pinv = fftw_plan_dft_2d(dim_sz[1],dim_sz[0], NULL, NULL, FFTW_BACKWARD, FFTW_ESTIMATE);
    my_plans.pinv2d= &pinv;
    fftw_plan p_tr = fftw_plan_dft_2d(dim_sz[0],dim_sz[1], NULL, NULL, FFTW_FORWARD, FFTW_ESTIMATE);
    my_plans.p_tr2d= &p_tr;
    fftw_plan pinv_tr = fftw_plan_dft_2d(dim_sz[0],dim_sz[1],  NULL, NULL, FFTW_BACKWARD, FFTW_ESTIMATE);
    my_plans.pinv_tr2d= &pinv_tr;

    fftw_plan p1d = fftw_plan_dft_1d(sz[0], NULL, NULL, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan pinv1d = fftw_plan_dft_1d(sz[0], NULL, NULL, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_plan p1dtr = fftw_plan_dft_1d(sz[1], NULL, NULL, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan pinv1dtr = fftw_plan_dft_1d(sz[1], NULL, NULL, FFTW_BACKWARD, FFTW_ESTIMATE);
    my_plans.p1d=&p1d;
    my_plans.pinv1d=&pinv1d;
    my_plans.ptr1d=&p1dtr;
    my_plans.pinvtr1d= &pinv1dtr;

    #pragma omp parallel for
    for(int t=0;t<sz[3];t++)
    {
        TORTOISE::EnableOMPThread();
        #pragma omp critical
        {
            if(stream)
                (*stream)<<  t <<", "<< std::flush;
            else
                std::cout<<  t <<", "<< std::flush;
        }

        fftw_complex *data_complex =  (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * sz[0]*sz[1]);
        fftw_complex *res_complex  =  (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * sz[0]*sz[1]);

        ImageType4DComplex::IndexType index;
        index[3]=t;

        for (int z=0; z<sz[2]; z++)
        {
            index[2]=z;
            for (int x = 0 ; x < sz[0];x++)
            {
                index[0]=x;
                for (int y = 0 ; y < sz[1];y++)
                {
                    index[1]=y;
                    std::complex<float> v = input_img->GetPixel(index);
                    data_complex[sz[0]*y+x][0] = (double) v.real();
                    data_complex[sz[0]*y+x][1] = (double) v.imag();
                }
            }
            unring_2d(data_complex,res_complex, dim_sz,nsh,minW,maxW,&my_plans);

            for (int x = 0 ; x < sz[0];x++)
            {
                index[0]=x;
                for (int y = 0 ; y < sz[1];y++)
                {
                    index[1]=y;
                    std::complex<float> val( (float)res_complex[sz[0]*y+x][0],
                                             (float)res_complex[sz[0]*y+x][1] );
                    output_img->SetPixel(index,val);
                }
            }
        }

        fftw_free(data_complex);
        fftw_free(res_complex);
        TORTOISE::DisableOMPThread();
    }

    fftw_destroy_plan(p);
    fftw_destroy_plan(pinv);
    fftw_destroy_plan(p_tr);
    fftw_destroy_plan(pinv_tr);

    fftw_destroy_plan(p1d);
    fftw_destroy_plan(pinv1d);
    fftw_destroy_plan(p1dtr);
    fftw_destroy_plan(pinv1dtr);

    if(stream)
        (*stream)<< std::endl;
    else
        std::cout<< std::endl;

    return output_img;
}
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
<scratchpad>/tbuild.sh GibbsComplexTests && <scratchpad>/tbuild.sh ctest -R "complex_io_roundtrip|magnitude_equivalence"
```

Expected: `max relative difference: 0` (or below 1e-15), `PASS magnitude_equivalence`, and `100% tests passed, 0 tests failed out of 2`.

A non-trivial difference means the marshalling indices disagree with `UnRingFull` — check that the buffer index is `sz[0]*y+x` and that the x and y loops are nested in the same order as the original.

- [ ] **Step 6: Commit**

```bash
git add src/tools/UnRing/unring.h \
        src/tools/UnRing/tests/test_gibbs_complex.cxx \
        TORTOISEV4/CMakeLists.txt
git commit -m "Add complex-valued SuShi unringing with magnitude-equivalence test"
```

---
## Task 3: Partial-Fourier geometry detection

**Files:**
- Create: `src/tools/UnRing/pf_geometry.h`
- Modify: `src/tools/UnRing/tests/test_gibbs_complex.cxx`
- Modify: `TORTOISEV4/CMakeLists.txt`

**Interfaces:**
- Consumes: `ImageType4DComplex` (Task 1).
- Produces:
  - `struct PFGeometry` with members `status`, `is_partial_fourier`, `n_pe`, `n_missing`, `factor`, `side`, `zero_band_energy_ratio`
  - `enum PFGeometry::Side { Low, High }` — `Low` means the most-negative-ky lines are missing
  - `enum PFGeometry::Status { NoZeroBand, InteriorBand, ImplausibleFactor, DetectedPF }`
  - `int ShiftedToUnshifted(int i, int n)`
  - `PFGeometry DetectPFGeometry(ImageType4DComplex::Pointer img, int pe_axis, float zero_tol=1e-6f, int max_volumes=8)`

> **Refinement against the spec.** The spec's `PFGeometry` carried only a `bool
> is_partial_fourier`, but its error-handling table needs to distinguish "no
> zero band" (skip POCS, continue, exit 0) from "detected factor below 0.5"
> (error and exit). A single bool cannot express that. `Status` is added for
> it; `is_partial_fourier` remains as the convenience predicate, always equal
> to `status == DetectedPF`.

- [ ] **Step 1: Write the failing test**

Add to `src/tools/UnRing/tests/test_gibbs_complex.cxx`. Add `#include "pf_geometry.h"` to the includes, then add above `main`:

```cpp
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

    std::cout << "PASS pf_detection_negative" << std::endl;
    return 0;
}
```

Register both in `main`:

```cpp
    if (name == "pf_detection_positive") return test_pf_detection_positive();
    if (name == "pf_detection_negative") return test_pf_detection_negative();
```

- [ ] **Step 2: Register the tests in CMake**

In `TORTOISEV4/CMakeLists.txt`, after the `magnitude_equivalence` test line:

```cmake
add_test(NAME pf_detection_positive COMMAND GibbsComplexTests pf_detection_positive)
add_test(NAME pf_detection_negative COMMAND GibbsComplexTests pf_detection_negative)
```

- [ ] **Step 3: Run the tests to verify they fail**

```bash
<scratchpad>/tbuild.sh GibbsComplexTests
```

Expected: compile error — `pf_geometry.h: No such file or directory`.

- [ ] **Step 4: Implement pf_geometry.h**

Create `src/tools/UnRing/pf_geometry.h`:

```cpp
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
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
<scratchpad>/tbuild.sh GibbsComplexTests && <scratchpad>/tbuild.sh ctest -R pf_detection
```

Expected: four lines of detected geometry, `PASS pf_detection_positive`, `PASS pf_detection_negative`, `100% tests passed, 0 tests failed out of 2`.

If `side` comes out inverted, the `ShiftedToUnshifted` direction is wrong — verify that `ShiftedToUnshifted(n/2, n) == 0`, which is the defining property (DC at shifted centre maps to raw index 0).

- [ ] **Step 6: Commit**

```bash
git add src/tools/UnRing/pf_geometry.h \
        src/tools/UnRing/tests/test_gibbs_complex.cxx \
        TORTOISEV4/CMakeLists.txt
git commit -m "Add partial-Fourier geometry detection from k-space zero band"
```

---

## Task 4: POCS partial-Fourier reconstruction

**Files:**
- Create: `src/tools/UnRing/pocs.h`
- Modify: `src/tools/UnRing/tests/test_gibbs_complex.cxx`
- Modify: `TORTOISEV4/CMakeLists.txt`

**Interfaces:**
- Consumes: `PFGeometry`, `ShiftedToUnshifted` (Task 3); `ImageType4DComplex` (Task 1); `TORTOISE::EnableOMPThread` / `DisableOMPThread` from `TORTOISE.h`.
- Produces:
  - `struct POCSParams { int iters; float tol; }` — constructed with `iters=10`, `tol=1e-4f`
  - `struct POCSResult { int iters_run; float final_rel_change; }`
  - `void BuildPOCSMasks(const PFGeometry &geom, std::vector<char> &acquired, std::vector<double> &window)`
  - `POCSResult ApplyPOCS(ImageType4DComplex::Pointer img, const PFGeometry &geom, int pe_axis, const POCSParams &params)` — modifies `img` in place

> **Correction against the spec.** The spec says acquired k-space lines are
> "bit-preserved" by the data-consistency step. That is not achievable: the
> substitution is exact, but the surrounding inverse FFT and the store back to
> `float` pixels introduce round-trip error of order 1e-7 relative. Test 5
> therefore asserts a 1e-5 relative bound, which is tight enough to catch a
> genuinely broken consistency step and honest about float precision.

- [ ] **Step 1: Write the failing test**

Add to `src/tools/UnRing/tests/test_gibbs_complex.cxx`. Add `#include "pocs.h"` to the includes, then add above `main`:

```cpp
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
```

Register in `main`:

```cpp
    if (name == "pocs_consistency") return test_pocs_consistency();
    if (name == "pocs_accuracy")    return test_pocs_accuracy();
```

- [ ] **Step 2: Register the tests in CMake**

```cmake
add_test(NAME pocs_consistency COMMAND GibbsComplexTests pocs_consistency)
add_test(NAME pocs_accuracy COMMAND GibbsComplexTests pocs_accuracy)
```

- [ ] **Step 3: Run the tests to verify they fail**

```bash
<scratchpad>/tbuild.sh GibbsComplexTests
```

Expected: compile error — `pocs.h: No such file or directory`.

- [ ] **Step 4: Implement pocs.h**

Create `src/tools/UnRing/pocs.h`:

```cpp
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
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
<scratchpad>/tbuild.sh GibbsComplexTests && <scratchpad>/tbuild.sh ctest -R pocs
```

Expected: `PASS pocs_consistency`, `PASS pocs_accuracy`, `100% tests passed, 0 tests failed out of 2`, with the printed RMSE ratio comfortably below 0.9.

If `pocs_accuracy` fails with a ratio near 1.0, the phase estimate is wrong — check `BuildPOCSMasks` gives a non-empty window and that `m > 0` for a 6/8 acquisition. If it fails with a ratio above 1.0, the iteration is diverging, which almost always means the `nfac` normalisation was dropped after an inverse transform.

- [ ] **Step 6: Run the whole suite**

```bash
<scratchpad>/tbuild.sh ctest
```

Expected: `100% tests passed, 0 tests failed out of 6`.

- [ ] **Step 7: Commit**

```bash
git add src/tools/UnRing/pocs.h \
        src/tools/UnRing/tests/test_gibbs_complex.cxx \
        TORTOISEV4/CMakeLists.txt
git commit -m "Add POCS partial-Fourier reconstruction for complex data"
```

---
## Task 5: GibbsComplex driver and CLI

**Files:**
- Create: `src/tools/UnRing/gibbs_complex_parser.h`
- Create: `src/tools/UnRing/gibbs_complex_parser.cxx`
- Create: `src/tools/UnRing/gibbs_complex_main.cxx`
- Create: `<scratchpad>/trun.sh` (scratchpad only, **not** committed)
- Modify: `TORTOISEV4/CMakeLists.txt`

**Interfaces:**
- Consumes: `UnRingFullComplex` (Task 2), `DetectPFGeometry` / `PFGeometry` (Task 3), `ApplyPOCS` / `POCSParams` (Task 4), `readImageD` / `writeImageD` for `ImageType4DComplex` (Task 1).
- Produces: the `GibbsComplex` executable and `class GibbsComplex_PARSER`.

- [ ] **Step 1: Write the parser header**

Create `src/tools/UnRing/gibbs_complex_parser.h`:

```cpp
#ifndef _GibbsComplex_PARSER_h
#define _GibbsComplex_PARSER_h

#include <iostream>
#include <string>

#include "antsCommandLineParser.h"

class GibbsComplex_PARSER : public itk::ants::CommandLineParser
{
public:
    GibbsComplex_PARSER( int argc , char * argv[] );
    ~GibbsComplex_PARSER();

    std::string getInputImageName();
    std::string getOutputImageName();
    std::string getOutputMagnitudeName();
    int         getPEDir();
    bool        getDoPOCS();
    int         getPOCSIters();
    float       getPOCSTol();
    float       getPFFactor();      // <= 0 when not supplied
    std::string getPFSide();        // "" when not supplied
    bool        getForcePF();
    float       getZeroTol();
    int         getNsh();
    int         getMinW();
    int         getMaxW();
    int         getNumberOfCores();
    bool        getDisableITKThreads();

private:
    void CreateParserandFillText(int argc , char * argv[] );
    void InitializeCommandLineOptions();
    bool checkIfAllRequiredParamsAreEntered();
};

#endif
```

- [ ] **Step 2: Write the parser implementation**

Create `src/tools/UnRing/gibbs_complex_parser.cxx`:

```cpp
#include "gibbs_complex_parser.h"
#include <algorithm>
#include <ctype.h>

GibbsComplex_PARSER::GibbsComplex_PARSER( int argc , char * argv[] )
{
    CreateParserandFillText(argc,argv);
    this->Parse(argc,argv);

    if( argc == 1 )
    {
        this->PrintMenu( std::cout, 5, false );
        exit(EXIT_FAILURE);
    }

    if(checkIfAllRequiredParamsAreEntered()==0)
    {
        std::cout<<"Not all the required Parameters are entered! Exiting!"<<std::endl;
        exit(EXIT_FAILURE);
    }
}

GibbsComplex_PARSER::~GibbsComplex_PARSER()
{
}

void GibbsComplex_PARSER::CreateParserandFillText(int argc, char* argv[])
{
    this->SetCommand( argv[0] );

    std::string commandDescription = std::string( "Partial-Fourier-aware Gibbs ringing correction for complex-valued DWIs. Reads a complex-valued (COMPLEX64) 4D NIFTI, optionally restores the un-acquired partial-Fourier k-space region with POCS, and applies the Kellner et al. local subvoxel-shift method to the complex data instead of its magnitude. For magnitude-only data, use the Gibbs command instead." );

    this->SetCommandDescription( commandDescription );
    this->InitializeCommandLineOptions();
}

void GibbsComplex_PARSER::InitializeCommandLineOptions()
{
    typedef itk::ants::CommandLineParser::OptionType OptionType;

    {
        std::string description = std::string( "Full path to the input complex-valued 4D NIFTI (COMPLEX64 or COMPLEX128). REQUIRED." );
        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'i');
        option->SetLongName( "input");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Full path to the output complex-valued 4D NIFTI (COMPLEX64). REQUIRED." );
        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'o');
        option->SetLongName( "output");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Optional. Full path for an additional magnitude-only output NIFTI, written in the same pass." );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "output_magnitude");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Phase encoding direction. 0: horizontal, 1: vertical. Same convention as the Gibbs command. Default: 1" );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "pe_dir");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Run POCS partial-Fourier reconstruction before unringing (0/1). Skipped automatically if the data does not look zero-filled. Default: 1" );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "pocs");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Maximum number of POCS iterations (int). Default: 10" );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "pocs_iters");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "POCS relative-change stopping tolerance (float). Default: 1e-4" );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "pocs_tol");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Override the detected partial-Fourier factor (float, e.g. 0.75 or 0.875). By default the factor is detected from the k-space zero band." );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "pf_factor");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Override the detected truncated side of k-space: low or high. low means the most-negative-ky lines are missing." );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "pf_side");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Proceed even when the declared and detected partial-Fourier geometry disagree, using the declared values (0/1). Default: 0" );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "force_pf");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Normalised-energy threshold below which a k-space line counts as empty (float). Default: 1e-6" );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "zero_tol");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Number of subvoxel shifts for the unringing (int). Default: 25" );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "nsh");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Minimum window size for the unringing (int). Default: 1" );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "minW");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Maximum window size for the unringing (int). Default: 3" );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "maxW");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Number of cores to use (int). Caps both the ITK and OpenMP thread counts." );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "ncores");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Pin ITK to a single thread so the ITK and OpenMP layers cannot multiply (0/1). Default: 0" );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "disable_itk_threads");
        option->SetDescription( description );
        this->AddOption( option );
    }
}

std::string GibbsComplex_PARSER::getInputImageName()
{
    OptionType::Pointer option = this->GetOption( "input");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
        return std::string("");
}

std::string GibbsComplex_PARSER::getOutputImageName()
{
    OptionType::Pointer option = this->GetOption( "output");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
        return std::string("");
}

std::string GibbsComplex_PARSER::getOutputMagnitudeName()
{
    OptionType::Pointer option = this->GetOption( "output_magnitude");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
        return std::string("");
}

int GibbsComplex_PARSER::getPEDir()
{
    OptionType::Pointer option = this->GetOption( "pe_dir");
    if(option->GetNumberOfFunctions())
        return (int)(atoi(option->GetFunction(0)->GetName().c_str()));
    else
        return 1;
}

bool GibbsComplex_PARSER::getDoPOCS()
{
    OptionType::Pointer option = this->GetOption( "pocs");
    if(option->GetNumberOfFunctions())
        return (bool)(atoi(option->GetFunction(0)->GetName().c_str()));
    else
        return true;
}

int GibbsComplex_PARSER::getPOCSIters()
{
    OptionType::Pointer option = this->GetOption( "pocs_iters");
    if(option->GetNumberOfFunctions())
        return (int)(atoi(option->GetFunction(0)->GetName().c_str()));
    else
        return 10;
}

float GibbsComplex_PARSER::getPOCSTol()
{
    OptionType::Pointer option = this->GetOption( "pocs_tol");
    if(option->GetNumberOfFunctions())
        return (atof(option->GetFunction(0)->GetName().c_str()));
    else
        return 1e-4;
}

float GibbsComplex_PARSER::getPFFactor()
{
    OptionType::Pointer option = this->GetOption( "pf_factor");
    if(option->GetNumberOfFunctions())
        return (atof(option->GetFunction(0)->GetName().c_str()));
    else
        return -1.;
}

std::string GibbsComplex_PARSER::getPFSide()
{
    OptionType::Pointer option = this->GetOption( "pf_side");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
        return std::string("");
}

bool GibbsComplex_PARSER::getForcePF()
{
    OptionType::Pointer option = this->GetOption( "force_pf");
    if(option->GetNumberOfFunctions())
        return (bool)(atoi(option->GetFunction(0)->GetName().c_str()));
    else
        return false;
}

float GibbsComplex_PARSER::getZeroTol()
{
    OptionType::Pointer option = this->GetOption( "zero_tol");
    if(option->GetNumberOfFunctions())
        return (atof(option->GetFunction(0)->GetName().c_str()));
    else
        return 1e-6;
}

int GibbsComplex_PARSER::getNsh()
{
    OptionType::Pointer option = this->GetOption( "nsh");
    if(option->GetNumberOfFunctions())
        return (int)(atoi(option->GetFunction(0)->GetName().c_str()));
    else
        return 25;
}

int GibbsComplex_PARSER::getMinW()
{
    OptionType::Pointer option = this->GetOption( "minW");
    if(option->GetNumberOfFunctions())
        return (int)(atoi(option->GetFunction(0)->GetName().c_str()));
    else
        return 1;
}

int GibbsComplex_PARSER::getMaxW()
{
    OptionType::Pointer option = this->GetOption( "maxW");
    if(option->GetNumberOfFunctions())
        return (int)(atoi(option->GetFunction(0)->GetName().c_str()));
    else
        return 3;
}

int GibbsComplex_PARSER::getNumberOfCores()
{
    OptionType::Pointer option = this->GetOption( "ncores");
    if(option->GetNumberOfFunctions())
        return (int)(atoi(option->GetFunction(0)->GetName().c_str()));
    else
        return 0;
}

bool GibbsComplex_PARSER::getDisableITKThreads()
{
    OptionType::Pointer option = this->GetOption( "disable_itk_threads");
    if(option->GetNumberOfFunctions())
        return (bool)(atoi(option->GetFunction(0)->GetName().c_str()));
    else
        return false;
}

bool GibbsComplex_PARSER::checkIfAllRequiredParamsAreEntered()
{
    if(this->getInputImageName()==std::string(""))
    {
        std::cout<<"Input image name not entered...Exiting..."<<std::endl;
        return 0;
    }
    if(this->getOutputImageName()==std::string(""))
    {
        std::cout<<"Output image name not entered...Exiting..."<<std::endl;
        return 0;
    }
    std::string side = this->getPFSide();
    if(side!=std::string("") && side!=std::string("low") && side!=std::string("high"))
    {
        std::cout<<"pf_side must be either low or high...Exiting..."<<std::endl;
        return 0;
    }
    return 1;
}
```

- [ ] **Step 3: Write the driver**

Create `src/tools/UnRing/gibbs_complex_main.cxx`:

```cpp
#ifndef _GIBBSCOMPLEXMAIN_CXX_
#define _GIBBSCOMPLEXMAIN_CXX_

#include "defines.h"
#include "unring.h"
#include "pf_geometry.h"
#include "pocs.h"
#include "gibbs_complex_parser.h"
#include "itkMultiThreaderBase.h"
#include "itkImageIOFactory.h"
#include "itkImageIOBase.h"
#include "itkImageRegionIteratorWithIndex.h"
#include <cmath>
#include <complex>

// Swap the two in-plane axes. DetectPFGeometry, ApplyPOCS and the SuShi 1D
// pass are all written against a vertical phase-encode axis, so --pe_dir 0
// data is transposed on the way in and back on the way out -- the same trick
// gibbs_main.cxx uses.
ImageType4DComplex::Pointer TransposeInPlane(ImageType4DComplex::Pointer img)
{
    ImageType4DComplex::SizeType osz = img->GetLargestPossibleRegion().GetSize();

    ImageType4DComplex::SizeType nsz;
    nsz[0]=osz[1];
    nsz[1]=osz[0];
    nsz[2]=osz[2];
    nsz[3]=osz[3];

    ImageType4DComplex::IndexType start; start.Fill(0);
    ImageType4DComplex::RegionType reg(start,nsz);

    ImageType4DComplex::Pointer out= ImageType4DComplex::New();
    out->SetRegions(reg);
    out->Allocate();

    itk::ImageRegionIteratorWithIndex<ImageType4DComplex> it(out,out->GetLargestPossibleRegion());
    for(it.GoToBegin();!it.IsAtEnd();++it)
    {
        ImageType4DComplex::IndexType ind4= it.GetIndex();
        ImageType4DComplex::IndexType old=ind4;
        old[0]=ind4[1];
        old[1]=ind4[0];
        it.Set(img->GetPixel(old));
    }
    return out;
}

int main(int argc, char* argv[])
{
    GibbsComplex_PARSER *parser = new GibbsComplex_PARSER(argc,argv);

    TORTOISE t;

    // The TORTOISE constructor sets omp_set_num_threads() to the host core
    // count and ignores OMP_NUM_THREADS, so any cap has to be applied after
    // it. Mirrors gibbs_main.cxx and DRBUDDI_main.cxx.
    int ncores = parser->getNumberOfCores();
    if(ncores>0)
    {
        itk::MultiThreaderBase::SetGlobalDefaultNumberOfThreads(ncores);
        omp_set_num_threads(ncores);
        TORTOISE::SetNAvailableCores(ncores);
    }
    if(parser->getDisableITKThreads())
    {
        itk::MultiThreaderBase::SetGlobalDefaultNumberOfThreads(1);
    }

    std::string input_name  = parser->getInputImageName();
    std::string output_name = parser->getOutputImageName();
    std::string mag_name    = parser->getOutputMagnitudeName();
    int   pe_dir     = parser->getPEDir();
    bool  do_pocs    = parser->getDoPOCS();
    bool  force_pf   = parser->getForcePF();
    float zero_tol   = parser->getZeroTol();
    float pf_factor_arg = parser->getPFFactor();
    std::string pf_side_arg = parser->getPFSide();

    POCSParams pocs_params;
    pocs_params.iters = parser->getPOCSIters();
    pocs_params.tol   = parser->getPOCSTol();

    int nsh  = parser->getNsh();
    int minW = parser->getMinW();
    int maxW = parser->getMaxW();

    // Refuse magnitude input rather than silently treating it as complex.
    {
        itk::ImageIOBase::Pointer io = itk::ImageIOFactory::CreateImageIO(
            input_name.c_str(), itk::ImageIOFactory::FileModeEnum::ReadMode);
        if(io.IsNull())
        {
            std::cout<<"Could not read "<<input_name<<" . Exiting..."<<std::endl;
            return EXIT_FAILURE;
        }
        io->SetFileName(input_name);
        io->ReadImageInformation();
        if(io->GetPixelType()!=itk::IOPixelEnum::COMPLEX)
        {
            std::cout<<"Input image "<<input_name<<" is not complex-valued (datatype: "
                     <<io->GetPixelTypeAsString(io->GetPixelType())
                     <<"). GibbsComplex requires a COMPLEX64 or COMPLEX128 NIFTI. "
                     <<"For magnitude-only data use the Gibbs command. Exiting..."<<std::endl;
            return EXIT_FAILURE;
        }
    }

    ImageType4DComplex::Pointer dwis = readImageD<ImageType4DComplex>(input_name);

    ImageType4DComplex::DirectionType orig_dir = dwis->GetDirection();
    ImageType4DComplex::SpacingType   orig_spc = dwis->GetSpacing();
    ImageType4DComplex::PointType     orig_org = dwis->GetOrigin();
    ImageType4DComplex::RegionType    orig_reg = dwis->GetLargestPossibleRegion();

    if(pe_dir==0)
        dwis = TransposeInPlane(dwis);

    // After the transpose the phase-encode axis is always 1.
    const int pe_axis = 1;

    PFGeometry geom = DetectPFGeometry(dwis, pe_axis, zero_tol, 8);

    if(geom.status==PFGeometry::ImplausibleFactor && !force_pf)
    {
        std::cout<<"Detected a k-space zero band of "<<geom.n_missing<<" lines out of "
                 <<geom.n_pe<<" (factor "<<geom.factor<<"), which is below 0.5 and "
                 <<"implausible for partial Fourier. Exiting..."<<std::endl;
        return EXIT_FAILURE;
    }

    if(pf_factor_arg>0 || pf_side_arg!=std::string(""))
    {
        PFGeometry declared = geom;
        declared.status = PFGeometry::DetectedPF;
        declared.is_partial_fourier = true;

        if(pf_factor_arg>0)
        {
            declared.factor = pf_factor_arg;
            declared.n_missing = (int)(llround((1.-(double)pf_factor_arg)*(double)declared.n_pe));
        }
        if(pf_side_arg==std::string("low"))
            declared.side = PFGeometry::Low;
        if(pf_side_arg==std::string("high"))
            declared.side = PFGeometry::High;

        bool mismatch = (!geom.is_partial_fourier)
                        || (geom.n_missing != declared.n_missing)
                        || (geom.side != declared.side);

        if(mismatch)
        {
            std::cout<<"Declared and detected partial-Fourier geometry disagree."<<std::endl;
            std::cout<<"  detected: pf="<<(geom.is_partial_fourier?"yes":"no")
                     <<" n_missing="<<geom.n_missing<<" factor="<<geom.factor
                     <<" side="<<(geom.side==PFGeometry::Low?"low":"high")<<std::endl;
            std::cout<<"  declared: n_missing="<<declared.n_missing
                     <<" factor="<<declared.factor
                     <<" side="<<(declared.side==PFGeometry::Low?"low":"high")<<std::endl;
            if(!force_pf)
            {
                std::cout<<"Pass --force_pf 1 to proceed with the declared geometry. Exiting..."<<std::endl;
                return EXIT_FAILURE;
            }
            std::cout<<"--force_pf given: proceeding with the declared geometry."<<std::endl;
        }
        geom = declared;
    }

    if(geom.is_partial_fourier)
    {
        std::cout<<"Partial Fourier detected: "<<geom.n_missing<<" of "<<geom.n_pe
                 <<" k-space lines empty (factor "<<geom.factor<<", "
                 <<(geom.side==PFGeometry::Low?"low":"high")<<" side, band energy ratio "
                 <<geom.zero_band_energy_ratio<<")."<<std::endl;
    }
    else
    {
        std::cout<<"No partial-Fourier zero band found. The data does not look "
                 <<"zero-filled -- it may be full Fourier, or already reconstructed "
                 <<"with homodyne or POCS."<<std::endl;
    }

    if(do_pocs && geom.is_partial_fourier)
    {
        std::cout<<"Running POCS partial-Fourier reconstruction..."<<std::endl;
        POCSResult res = ApplyPOCS(dwis, geom, pe_axis, pocs_params);
        std::cout<<"POCS finished after at most "<<res.iters_run
                 <<" iterations, final relative change "<<res.final_rel_change<<"."<<std::endl;
    }
    else if(do_pocs && !geom.is_partial_fourier)
    {
        std::cout<<"Skipping POCS."<<std::endl;
    }
    else if(!do_pocs && geom.is_partial_fourier)
    {
        std::cout<<"WARNING: POCS disabled on partial-Fourier data. Residual "
                 <<"partial-Fourier ringing may remain along the phase encoding "
                 <<"direction. The magnitude-domain RPG method in the Gibbs command "
                 <<"addresses that case."<<std::endl;
    }

    dwis = UnRingFullComplex(dwis, nsh, minW, maxW);

    if(pe_dir==0)
        dwis = TransposeInPlane(dwis);

    dwis->SetDirection(orig_dir);
    dwis->SetSpacing(orig_spc);
    dwis->SetOrigin(orig_org);

    writeImageD<ImageType4DComplex>(dwis, output_name);

    if(mag_name!=std::string(""))
    {
        ImageType4D::Pointer mag = ImageType4D::New();
        mag->SetRegions(orig_reg);
        mag->SetDirection(orig_dir);
        mag->SetSpacing(orig_spc);
        mag->SetOrigin(orig_org);
        mag->Allocate();
        mag->FillBuffer(0);

        itk::ImageRegionIteratorWithIndex<ImageType4D> it(mag,mag->GetLargestPossibleRegion());
        for(it.GoToBegin();!it.IsAtEnd();++it)
            it.Set( std::abs( dwis->GetPixel(it.GetIndex()) ) );

        writeImageD<ImageType4D>(mag, mag_name);
    }

    delete parser;
    return EXIT_SUCCESS;
}

#endif
```

If `io->GetPixelTypeAsString(...)` does not compile against this ITK version, drop that clause from the message and print only the file name.

- [ ] **Step 4: Add the CMake target**

In `TORTOISEV4/CMakeLists.txt`, directly after the existing `Gibbs` target block:

```cmake
add_executable(GibbsComplex ../src/tools/UnRing/gibbs_complex_main.cxx ../src/tools/UnRing/gibbs_complex_parser.cxx ../src/tools/ResampleDWIs/resample_dwis.cxx ${SOURCES})
target_link_libraries(GibbsComplex ${ITK_LIBRARIES} ${Boost_LIBRARIES} fftw3 )
```

- [ ] **Step 5: Build it**

```bash
<scratchpad>/tbuild.sh GibbsComplex
```

Expected: `[100%] Built target GibbsComplex`.

- [ ] **Step 6: Write the run helper**

Create `<scratchpad>/trun.sh`:

```bash
#!/usr/bin/env bash
# <scratchpad>/trun.sh — run a built TORTOISE binary against data in the
# scratchpad. The scratchpad data directory appears as /data in the container.
set -euo pipefail

REPO=/mnt/c/Users/tsalo/Documents/linc/qsiprep_TORTOISE
DATA=<scratchpad>/data
mkdir -p "${DATA}"

docker run --rm \
    -v "${REPO}":/src/TORTOISEV4 \
    -v tortoise-bin:/opt/bin \
    -v "${DATA}":/data \
    tortoise-dev "$@"
```

Then `chmod +x <scratchpad>/trun.sh`.

- [ ] **Step 7: Generate a synthetic complex input and run the binary end to end**

```bash
mkdir -p <scratchpad>/data
micromamba run -n linc311 python - <<'PY'
import numpy as np, nibabel as nb

nx, ny, nz, nt = 64, 64, 4, 2
x = np.arange(nx)[:, None, None, None]
y = np.arange(ny)[None, :, None, None]

mag = np.zeros((nx, ny, nz, nt))
mag[16:48, 14:50] = 100.0
mag[28:36, 26:38] = 40.0
phase = 0.5*np.cos(2*np.pi*x/nx) + 0.3*np.sin(2*np.pi*y/ny)
img = mag * np.exp(1j*phase)

# Zero-fill the 16 most-negative-ky lines: 6/8 partial Fourier, low side.
K = np.fft.fftshift(np.fft.fft2(img, axes=(0, 1)), axes=(0, 1))
K[:, :16] = 0
img_pf = np.fft.ifft2(np.fft.ifftshift(K, axes=(0, 1)), axes=(0, 1))

nb.save(nb.Nifti1Image(img_pf.astype(np.complex64), np.eye(4)),
        "<scratchpad>/data/pf_test.nii.gz")
print("wrote pf_test.nii.gz", img_pf.shape, img_pf.dtype)
PY

<scratchpad>/trun.sh /opt/bin/GibbsComplex \
    -i /data/pf_test.nii.gz \
    -o /data/pf_out.nii.gz \
    --output_magnitude /data/pf_out_mag.nii.gz \
    --pe_dir 1
```

Expected output includes `Partial Fourier detected: 16 of 64 k-space lines empty (factor 0.75, low side, ...)`, then `Running POCS partial-Fourier reconstruction...`, then the per-volume unringing progress, and exit status 0.

- [ ] **Step 8: Verify the outputs**

```bash
micromamba run -n linc311 python - <<'PY'
import nibabel as nb, numpy as np
c = nb.load("<scratchpad>/data/pf_out.nii.gz")
m = nb.load("<scratchpad>/data/pf_out_mag.nii.gz")
print("complex dtype:", c.get_data_dtype(), "shape:", c.shape)
print("magnitude dtype:", m.get_data_dtype(), "shape:", m.shape)
assert c.get_data_dtype() == np.complex64
assert np.allclose(np.abs(c.get_fdata(dtype=np.complex64)), m.get_fdata(), rtol=1e-5, atol=1e-4)
print("OK")
PY
```

Expected: `complex dtype: complex64`, matching shapes, and `OK`.

- [ ] **Step 9: Verify the guard rails**

```bash
# Magnitude input must be refused, not silently mangled.
<scratchpad>/trun.sh /opt/bin/GibbsComplex -i /data/pf_out_mag.nii.gz -o /data/nope.nii.gz ; echo "exit=$?"

# A declared geometry that contradicts the data must be an error.
<scratchpad>/trun.sh /opt/bin/GibbsComplex -i /data/pf_test.nii.gz -o /data/nope.nii.gz --pf_side high ; echo "exit=$?"

# ...and must proceed under protest when forced.
<scratchpad>/trun.sh /opt/bin/GibbsComplex -i /data/pf_test.nii.gz -o /data/forced.nii.gz --pf_side high --force_pf 1 ; echo "exit=$?"

# POCS off on PF data warns but still succeeds.
<scratchpad>/trun.sh /opt/bin/GibbsComplex -i /data/pf_test.nii.gz -o /data/nopocs.nii.gz --pocs 0 ; echo "exit=$?"
```

Expected, in order: `is not complex-valued` and `exit=1`; `Declared and detected partial-Fourier geometry disagree` and `exit=1`; `--force_pf given` and `exit=0`; `WARNING: POCS disabled on partial-Fourier data` and `exit=0`.

- [ ] **Step 10: Commit**

```bash
git add src/tools/UnRing/gibbs_complex_parser.h \
        src/tools/UnRing/gibbs_complex_parser.cxx \
        src/tools/UnRing/gibbs_complex_main.cxx \
        TORTOISEV4/CMakeLists.txt
git commit -m "Add GibbsComplex command for partial-Fourier-aware complex unringing"
```

---

## Task 6: Python validation harness

**Files:**
- Create: `src/tools/UnRing/validation/run_validation.py`

**Interfaces:**
- Consumes: the `GibbsComplex` and `Gibbs` binaries (Tasks 5 and pre-existing), invoked through a runner command prefix.
- Produces: a scored comparison report and, once calibrated, committed acceptance thresholds.

- [ ] **Step 1: Write the harness**

Create `src/tools/UnRing/validation/run_validation.py`:

```python
#!/usr/bin/env python
"""Validate GibbsComplex against magnitude RPG on synthetic partial-Fourier data.

Builds a complex phantom with known ground truth, simulates zero-filled partial
Fourier at 6/8 and 7/8, and scores three correction arms against the truth:

  complex+pocs : GibbsComplex with POCS
  complex      : GibbsComplex without POCS
  rpg          : the existing magnitude-domain Gibbs command

Run with:
  micromamba run -n linc311 python run_validation.py \
      --workdir <scratchpad>/data/validation \
      --container-workdir /data/validation \
      --runner <scratchpad>/trun.sh
"""

import argparse
import json
import os
import subprocess
import sys

import nibabel as nb
import numpy as np

# Acceptance thresholds, calibrated from the first clean run (see Step 3).
# None means "not yet calibrated": the harness reports numbers but does not
# gate on them.
THRESHOLDS = {
    "pf75_complex_pocs_vs_rpg_ratio": None,
    "pf875_complex_pocs_vs_rpg_ratio": None,
}

NX, NY, NZ, NT = 96, 96, 4, 2


def make_phantom():
    """Complex phantom: hard edges for ringing, smooth phase for POCS."""
    x = np.arange(NX)[:, None, None, None]
    y = np.arange(NY)[None, :, None, None]

    mag = np.zeros((NX, NY, NZ, NT))
    mag[20:76, 18:78] = 100.0
    mag[40:56, 38:58] = 40.0
    mag[46:50, 30:34] = 160.0
    phase = 0.5 * np.cos(2 * np.pi * x / NX) + 0.3 * np.sin(2 * np.pi * y / NY)
    return mag * np.exp(1j * phase)


def truncate(img, n_missing):
    """Zero-fill the n_missing most-negative-ky lines (low side, axis 1)."""
    k = np.fft.fftshift(np.fft.fft2(img, axes=(0, 1)), axes=(0, 1))
    k[:, :n_missing] = 0
    return np.fft.ifft2(np.fft.ifftshift(k, axes=(0, 1)), axes=(0, 1))


def rmse(a, b):
    return float(np.sqrt(np.mean((np.abs(a) - np.abs(b)) ** 2)))


def ringing(img):
    """Total variation in flat background bands beside the outer edges.

    Ground truth is uniformly zero there, so any variation is ringing.
    """
    m = np.abs(img)
    bands = [m[8:19, :], m[77:88, :], m[:, 6:17], m[:, 79:90]]
    return float(np.mean([np.mean(np.abs(np.diff(b, axis=0))) for b in bands]))


def save(path, arr, dtype):
    nb.save(nb.Nifti1Image(arr.astype(dtype), np.eye(4)), path)


def load(path):
    img = nb.load(path)
    if np.issubdtype(img.get_data_dtype(), np.complexfloating):
        return img.get_fdata(dtype=np.complex64)
    return img.get_fdata()


def run(runner, args):
    cmd = [runner] + args
    print("  $", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"command failed with exit code {proc.returncode}")
    return proc.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True,
                    help="host directory for inputs and outputs")
    ap.add_argument("--container-workdir", required=True,
                    help="the same directory as seen by the binaries")
    ap.add_argument("--runner", required=True,
                    help="command prefix that runs a binary, e.g. trun.sh")
    ap.add_argument("--report", default=None,
                    help="optional path to write the JSON report")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    hw, cw = args.workdir, args.container_workdir

    truth = make_phantom()
    results = {}

    for label, n_missing, factor in (("pf75", NY // 4, 0.75),
                                     ("pf875", NY // 8, 0.875)):
        print(f"\n=== {label} ({factor} partial Fourier, "
              f"{n_missing} of {NY} lines missing) ===", flush=True)

        pf = truncate(truth, n_missing)
        save(f"{hw}/{label}_complex.nii.gz", pf, np.complex64)
        save(f"{hw}/{label}_mag.nii.gz", np.abs(pf), np.float32)

        run(args.runner, ["/opt/bin/GibbsComplex",
                          "-i", f"{cw}/{label}_complex.nii.gz",
                          "-o", f"{cw}/{label}_pocs.nii.gz",
                          "--pe_dir", "1", "--pocs", "1"])
        run(args.runner, ["/opt/bin/GibbsComplex",
                          "-i", f"{cw}/{label}_complex.nii.gz",
                          "-o", f"{cw}/{label}_nopocs.nii.gz",
                          "--pe_dir", "1", "--pocs", "0"])
        # Gibbs takes positional args: in out kspace_coverage pe_dir
        run(args.runner, ["/opt/bin/Gibbs",
                          f"{cw}/{label}_mag.nii.gz",
                          f"{cw}/{label}_rpg.nii.gz",
                          str(factor), "1"])

        arms = {
            "uncorrected": pf,
            "complex+pocs": load(f"{hw}/{label}_pocs.nii.gz"),
            "complex": load(f"{hw}/{label}_nopocs.nii.gz"),
            "rpg": load(f"{hw}/{label}_rpg.nii.gz"),
        }

        scores = {name: {"rmse": rmse(a, truth), "ringing": ringing(a)}
                  for name, a in arms.items()}
        results[label] = scores

        print(f"{'arm':<14} {'RMSE':>10} {'ringing':>10}")
        for name, s in scores.items():
            print(f"{name:<14} {s['rmse']:>10.4f} {s['ringing']:>10.4f}")

        ratio = scores["complex+pocs"]["rmse"] / scores["rpg"]["rmse"]
        results[label]["complex_pocs_vs_rpg_ratio"] = ratio
        print(f"complex+pocs / rpg RMSE ratio: {ratio:.4f}")

    r75 = results["pf75"]["complex_pocs_vs_rpg_ratio"]
    r875 = results["pf875"]["complex_pocs_vs_rpg_ratio"]

    print("\n=== acceptance ===")
    print(f"6/8 ratio {r75:.4f}, 7/8 ratio {r875:.4f}")
    print("The issue's claim is that complex+POCS beats magnitude RPG, and that "
          "the margin is larger at 6/8 than at 7/8 -- so both ratios should be "
          "below 1 and r75 should be below r875.")

    if args.report:
        with open(args.report, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"wrote {args.report}")

    failed = False
    for key, value in (("pf75_complex_pocs_vs_rpg_ratio", r75),
                       ("pf875_complex_pocs_vs_rpg_ratio", r875)):
        limit = THRESHOLDS[key]
        if limit is None:
            print(f"{key}: {value:.4f} (threshold not yet calibrated)")
        elif value > limit:
            print(f"FAIL {key}: {value:.4f} > {limit:.4f}")
            failed = True
        else:
            print(f"PASS {key}: {value:.4f} <= {limit:.4f}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the harness**

```bash
micromamba run -n linc311 python \
    src/tools/UnRing/validation/run_validation.py \
    --workdir <scratchpad>/data/validation \
    --container-workdir /data/validation \
    --runner <scratchpad>/trun.sh \
    --report <scratchpad>/data/validation/report.json
```

Expected: two score tables and two ratios, each reported as "threshold not yet calibrated", exit 0.

- [ ] **Step 3: Calibrate and record the thresholds**

Read the two measured ratios from the run. Set each `THRESHOLDS` entry to the measured value rounded up to the next 0.05, so the gate catches a regression without tripping on run-to-run noise. Replace the `None` values in `run_validation.py` and add a one-line comment recording the date and the measured values.

If either ratio is at or above 1.0, **stop and report rather than adjusting the threshold to fit.** That result would mean complex+POCS does not beat RPG on this phantom, which contradicts the premise of the feature and needs investigation — most likely in the phase estimate or in the ringing metric's band placement, but possibly in the premise itself.

- [ ] **Step 4: Re-run to confirm the gate passes**

```bash
micromamba run -n linc311 python \
    src/tools/UnRing/validation/run_validation.py \
    --workdir <scratchpad>/data/validation \
    --container-workdir /data/validation \
    --runner <scratchpad>/trun.sh
```

Expected: two `PASS` lines, exit 0.

- [ ] **Step 5: Commit**

```bash
git add src/tools/UnRing/validation/run_validation.py
git commit -m "Add synthetic partial-Fourier validation harness for GibbsComplex"
```

---

## Task 7: Documentation

**Files:**
- Modify: `TORTOISEV4/tools/README.md`

**Interfaces:**
- Consumes: the finished CLI from Task 5.
- Produces: user-facing documentation.

- [ ] **Step 1: Add the GibbsComplex entry**

`TORTOISEV4/tools/README.md` groups tools under headings such as `# Input Related Tools`, with one `## ToolName` block each carrying a `Usage:` line and a description. Add this entry, keeping the file's existing heading style and placing it alongside the other DWI-processing tools:

```markdown
## GibbsComplex

Usage: GibbsComplex -i input_complex_nifti -o output_complex_nifti [options]

Partial-Fourier-aware Gibbs ringing correction for complex-valued DWIs. Reads a complex-valued (COMPLEX64 or COMPLEX128) 4D NIFTI, optionally restores the un-acquired partial-Fourier region of k-space with POCS, and applies the Kellner et al. local subvoxel-shift method to the complex data rather than to its magnitude. Lee, Novikov and Fieremans (MRM 2021) show this is more robust than magnitude-domain RPG when phase information is available, particularly at aggressive partial-Fourier factors.

The partial-Fourier factor and the truncated side of k-space are detected automatically from the zero-filled band. Data with no zero band -- full-Fourier acquisitions, or reconstructions where the vendor already applied homodyne or POCS -- is reported as such and the POCS stage is skipped, so the command will not apply a second partial-Fourier reconstruction on top of the scanner's.

For magnitude-only data, use the Gibbs command, which implements the magnitude-domain RPG method.

    -i, --input               Full path to the input complex-valued 4D NIFTI. REQUIRED.
    -o, --output              Full path to the output complex-valued 4D NIFTI. REQUIRED.
        --output_magnitude    Optional additional magnitude-only output, written in the same pass.
        --pe_dir              Phase encoding direction. 0: horizontal, 1: vertical. Default: 1
        --pocs                Run POCS partial-Fourier reconstruction (0/1). Default: 1
        --pocs_iters          Maximum POCS iterations. Default: 10
        --pocs_tol            POCS relative-change stopping tolerance. Default: 1e-4
        --pf_factor           Override the detected partial-Fourier factor, e.g. 0.75 or 0.875.
        --pf_side             Override the detected truncated side: low or high. low means the
                              most-negative-ky lines are missing.
        --force_pf            Proceed when declared and detected geometry disagree (0/1). Default: 0
        --zero_tol            Normalised-energy threshold for an empty k-space line. Default: 1e-6
        --nsh                 Number of subvoxel shifts. Default: 25
        --minW                Minimum window size. Default: 1
        --maxW                Maximum window size. Default: 3
        --ncores              Number of cores to use.
        --disable_itk_threads Pin ITK to a single thread (0/1). Default: 0
```

- [ ] **Step 2: Verify the documented defaults match the code**

```bash
grep -n "return 25;\|return 1;\|return 3;\|return 10;\|return 1e-4;\|return 1e-6;\|return true;" \
     src/tools/UnRing/gibbs_complex_parser.cxx
```

Every default in the README block above must match the corresponding getter's fallback value. A README that drifts from the parser is worse than no README, because it is believed.

- [ ] **Step 3: Run the full test suite one final time**

```bash
<scratchpad>/tbuild.sh ctest
```

Expected: `100% tests passed, 0 tests failed out of 6`.

- [ ] **Step 4: Commit**

```bash
git add TORTOISEV4/tools/README.md
git commit -m "Document the GibbsComplex command"
```

---

## Self-Review

**Spec coverage.** Every spec section maps to a task: complex I/O to Task 1; complex SuShi to Task 2; PF detection, including the not-zero-filled safety check, to Task 3; POCS to Task 4; the CLI contract and the full error-handling table to Task 5 (steps 3 and 9); the Python harness and its acceptance claim to Task 6; docs to Task 7; the Docker build environment to Task 0. The CRLF constraint appears in Global Constraints and in every commit step, which uses explicit paths.

**Two deliberate divergences from the spec**, both flagged inline where they occur:

1. `PFGeometry` gains a `Status` enum. The spec's single `bool is_partial_fourier` cannot distinguish "no zero band" (continue, exit 0) from "factor below 0.5" (error, exit 1), yet its error table requires different behaviour for each.
2. Test 5 asserts a 1e-5 relative bound on acquired k-space lines rather than the spec's "bit-preserved". Exact preservation is not achievable across an inverse FFT and a store to `float` pixels; the substitution itself is exact, but the round trip is not.

**Placeholder scan:** no TBD/TODO markers, and every code step carries complete code. The one intentionally uncalibrated value — `THRESHOLDS` in the harness — is a measurement that does not exist until Task 6 Step 2 runs, and Step 3 both fills it in and states what to do if the measurement contradicts the premise.

**Type consistency:** `ImageType4DComplex`, `PFGeometry`, `PFGeometry::Side`, `PFGeometry::Status`, `ShiftedToUnshifted`, `POCSParams`, `POCSResult`, `BuildPOCSMasks`, `ApplyPOCS`, `UnRingFullComplex`, and `GibbsComplex_PARSER` are each defined in exactly one task and used with identical spelling and signatures thereafter. Buffer indexing is `[nx*y+x]` and plans are `fftw_plan_dft_2d(ny, nx, ...)` everywhere, matching the existing `unring.h` convention.

**Known risk carried into execution:** the ITK enum spellings (`itk::IOPixelEnum::COMPLEX`, `itk::IOComponentEnum::FLOAT`, `itk::ImageIOFactory::FileModeEnum::ReadMode`, `GetPixelTypeAsString`) differ between ITK 5.3 and 5.4+. Tasks 1 and 5 each name the fallback spelling to use if compilation fails, rather than leaving it as a puzzle.
