# POCS partial-Fourier reconstruction: excellent on phantoms, unhelpful on real DWI

Status: **open question, needs investigation**
Date: 2026-08-29
Code state: branch `complex-degibbs`, including the complex-TV fix in
[Resolved: complex SuShi was not phase-rotation invariant](#resolved-complex-sushi-was-not-phase-rotation-invariant)
Related: [PennLINC/qsiprep#1099](https://github.com/PennLINC/qsiprep/issues/1099),
design spec `docs/superpowers/specs/2026-08-29-complex-pf-gibbs-design.md`

## The finding in one paragraph

The `GibbsComplex` command implements POCS partial-Fourier reconstruction followed by
complex-valued subvoxel-shift (SuShi) unringing. On synthetic phantoms POCS works
extremely well — it cuts magnitude RMSE against ground truth by ~99% relative to
zero-filling, and restores acquired k-space lines to within 2.4e-9. On three real
Siemens Prisma diffusion runs that are genuinely zero-filled partial Fourier, POCS did
**not** improve a ringing-per-sharpness proxy in any of six comparisons, and clearly
worsened it at high b-value. The gap between those two results is unexplained and is the
subject of this document. A secondary but blocking defect in the partial-Fourier detector
is described in [Blocking side issue](#blocking-side-issue-the-detector-threshold).

**This document is written to be actionable without access to the original datasets.**
All measured numbers are inlined, the synthetic reproduction is fully in-repo, and
[Reproducing without the original data](#reproducing-without-the-original-data) gives the
acceptance criteria a substitute dataset must meet.

---

## 1. What the implementation does

Files (POCS/detector as at commit `28dc728`; `unring.h` additionally carries the complex-TV fix):

| File | Contents |
|---|---|
| `src/tools/UnRing/pocs.h` | `POCSParams`, `POCSResult`, `BuildPOCSMasks`, `ApplyPOCS` |
| `src/tools/UnRing/pf_geometry.h` | `PFGeometry`, `ShiftedToUnshifted`, `DetectPFGeometry` |
| `src/tools/UnRing/unring.h` | `UnRingFullComplex` (appended; pre-existing functions untouched) |
| `src/tools/UnRing/gibbs_complex_main.cxx` | driver: read → transpose → detect → POCS → SuShi → write |
| `src/tools/UnRing/gibbs_complex_parser.cxx` | CLI |
| `src/tools/UnRing/tests/test_gibbs_complex.cxx` | 7 ctest cases |
| `src/tools/UnRing/validation/run_validation.py` | synthetic phantom harness |

### The POCS algorithm as implemented

1. Estimate image phase from the symmetric, fully-sampled centre band of k-space,
   Hamming-windowed so the estimate does not itself ring.
   Band half-width is `m = min(a, b)` where `a`, `b` are the acquired extents below and
   above DC in fftshifted coordinates.
2. Iterate up to `iters` times:
   - take the current magnitude, impose the estimated phase
   - forward FFT
   - **overwrite the acquired k-space lines with the measured values** (data consistency)
   - inverse FFT, scaled by `nfac = 1/(nx*ny)`
   - stop early if relative change `< tol`
3. Defaults: `iters = 10` (`pocs.h:18`), `tol = 1e-4` (`pocs.h:18`).

Conventions, shared with `unring.h` and `pf_geometry.h`: plans are
`fftw_plan_dft_2d(ny, nx, ...)`, buffers are indexed `buf[nx*y + x]`, and
`ShiftedToUnshifted(i, n) = (i + n - n/2) % n` maps fftshifted to raw FFT indices
(DC at raw 0, shifted `n/2`). Operates per 2D slice, parallelised over volumes.

The implementation has been reviewed and independently verified; see
[What has been ruled out](#4-what-has-been-ruled-out). Treat an implementation bug as
unlikely but not impossible.

---

## 2. Simulated results — POCS works

### 2.1 Unit test (`pocs_accuracy`, `pocs_consistency`)

64×64 complex phantom: hard-edged rectangles, magnitude 100 / 40, with phase
`0.5·cos(2πx/nx) + 0.3·sin(2πy/ny)`. 16 of 64 ky lines zero-filled at the low end (6/8 PF).

| measure | value | bound | note |
|---|---|---|---|
| Worst acquired-line deviation after POCS | **2.3787e-09** | < 1e-5 | data consistency is essentially exact |
| Magnitude RMSE ratio, POCS / zero-filled | **0.0087** | < 0.9 | ~99% error reduction |

**Caveat that matters for interpretation:** this phantom's phase is *by construction*
exactly the smooth, low-order function POCS estimates. Near-perfect recovery is the
expected result, not evidence the method generalises.

### 2.2 Synthetic harness (`run_validation.py`)

96×96 phantom, same construction, PF 0.75 (24 lines) and 0.875 (12 lines) at the low end.
Four arms scored against known ground truth. RMSE (magnitude), lower is better:

| PF | uncorrected | complex+POCS | complex (no POCS) | magnitude RPG |
|---|---|---|---|---|
| 0.75 (6/8) | 2.7857 | **0.3737** | 3.8562 | 3.0142 |
| 0.875 (7/8) | 1.7001 | **0.3756** | 2.5034 | 2.3369 |

Committed acceptance thresholds (calibrated from these runs, rounded up to the next 0.05):
`pf75_complex_pocs_vs_rpg_ratio = 0.15`, `pf875_complex_pocs_vs_rpg_ratio = 0.20`.

Note that on this phantom **both non-POCS arms score worse than no correction at all**.
Reviewed and judged a genuine property of a noiseless, perfectly-sharp-edged phantom under
asymmetric truncation — a harsher distortion than the symmetric-Gibbs model that RPG's
interpolation assumes — rather than a harness defect. This inflates the POCS-vs-RPG ratio,
so the defensible synthetic claim is "POCS is the only arm that improves on doing nothing",
not "POCS is ~8× better than RPG".

---

## 3. Real-data results — POCS does not help

### 3.1 The data

| | NIBS sub-22449 ses-01 | ds006131 sub-20188 ses-1 |
|---|---|---|
| runs used | `dir-AP_run-01`, `dir-PA_run-01` | `dir-AP_run-01` |
| shape | 140×140×87×76 | 136×136×84×104 |
| voxel | 1.7 mm iso | 1.69×1.69×1.7 mm |
| `PartialFourier` | 0.75 | 0.875 |
| `PhaseEncodingDirection` | `j-` | `j-` |
| scanner | Siemens Prisma | Siemens Prisma |
| TE / TR | 0.088 / 4.8 s | 0.09 / 4.3 s |
| MultibandAccelerationFactor | 3 | 4 |
| ParallelReductionFactorInPlane | (absent) | **2** |
| volumes analysed | 0 (b=5), 6 (b=3005) | 0 (b=5), 15 (b=5000) |

Phase stored as Siemens 12-bit integers spanning `[-4096, 4094]`; converted with
`rad = au · π / 4096`, giving `[-3.142, 3.140]`. Complex assembled as `S = M·exp(iφ)` and
`|S| == M` asserted before use.

### 3.2 These runs really are zero-filled

Judged by comparing each un-acquired line to its **conjugate mirror** about k-space centre
(0 = zero-filled, 1 = fully reconstructed). This is the scale-free test; see
[Blocking side issue](#blocking-side-issue-the-detector-threshold) for why a DC-relative
test is not.

| run | band/mirror energy | amplitude | suppression |
|---|---|---|---|
| NIBS AP | 0.0198 | 0.141 | 51× |
| NIBS PA | 0.0046 | 0.068 | 219× |
| ds006131 | 0.0025 | 0.050 | 403× |

The suppressed band sits at very nearly the declared width and **flips ends between the AP
and PA runs**, as reversing the phase-encode direction requires. POCS is therefore
*applicable in principle* to all three.

Indexing note that cost one analysis pass: with `nm` missing lines at the low end
(indices `0..nm-1`) and centre `c = n/2`, the band spans **offsets `c-(nm-1)..c`**, not
`0..nm-1`. Index 0 is the Nyquist line and has no mirror — exclude it.

### 3.3 Detection and POCS engagement

At default `zero_tol = 1e-6`, **all three runs were reported not-partial-Fourier and POCS
was skipped** (see the blocking side issue). To exercise POCS at all:

- `ds006131` with `--zero_tol 1e-4`: detected 22 of 136 lines, factor 0.838 (sidecar says
  0.875), low side, band energy ratio 9.29e-06. POCS ran 10 iterations, final relative
  change 0.005075.
- NIBS AP/PA: even at `1e-4` the suppressed run starts one line in from the edge, so it is
  classified interior, not PF. Required declaring geometry explicitly:
  `--pf_factor 0.75 --pf_side low|high --force_pf 1`.

Sanity check performed: where POCS was skipped, the default / `--pocs 0` / `--pocs 1` arms
produced **bit-identical** magnitude output (max abs diff 0.000e+00), confirming the arms
are what they claim. Where POCS ran, max abs diff was 1.4e4.

### 3.4 Metric definition

No ground truth exists on real data, so results are relative. Any method can lower a raw
ringing score by blurring, so ringing is reported **against edge sharpness measured on the
same images**.

On the 16 central slices of each volume, with `R` = uncorrected magnitude and `A` = the arm
under test, PE axis = axis 1:

- `scale` = mean of `R` over voxels above its 60th percentile
- `edge` = voxels where `|∂R/∂y|` exceeds its 99th percentile
- `band` = voxels 3–8 voxels from an edge along y (dilate 8, minus dilate 2), restricted to
  `R > 0.05·scale`
- **ringing** = `mean(|∂²A/∂y²|)` over `band`, divided by `scale`
- **sharpness** = `mean(|∂A/∂y|)` over `edge`, divided by `scale`

### 3.5 Raw metric values

| run | vol | arm | ringing | sharpness | Δ vs uncorr |
|---|---|---|---|---|---|
| NIBS AP | b≈0 | uncorrected | 0.16822 | 1.12647 | 0.00000 |
| | | RPG | 0.11115 | 0.85538 | 0.04515 |
| | | SuShi only | 0.12665 | 0.91785 | 0.03708 |
| | | POCS+SuShi | 0.18328 | 1.19044 | 0.04194 |
| NIBS AP | high-b | uncorrected | 0.44038 | 1.24133 | 0.00000 |
| | | RPG | 0.26903 | 0.86567 | 0.07207 |
| | | SuShi only | 0.30898 | 0.94395 | 0.06562 |
| | | POCS+SuShi | 0.52743 | 1.30474 | 0.07623 |
| NIBS PA | b≈0 | uncorrected | 0.15112 | 1.00747 | 0.00000 |
| | | RPG | 0.10096 | 0.76009 | 0.04240 |
| | | SuShi only | 0.11333 | 0.81369 | 0.03534 |
| | | POCS+SuShi | 0.16750 | 1.07246 | 0.03966 |
| NIBS PA | high-b | uncorrected | 0.28445 | 0.93856 | 0.00000 |
| | | RPG | 0.17928 | 0.68649 | 0.05619 |
| | | SuShi only | 0.20408 | 0.74554 | 0.04976 |
| | | POCS+SuShi | 0.33582 | 0.98065 | 0.05654 |
| ds006131 | b≈0 | uncorrected | 0.17634 | 1.10851 | 0.00000 |
| | | RPG | 0.10346 | 0.78088 | 0.04863 |
| | | SuShi only | 0.13343 | 0.91613 | 0.03678 |
| | | POCS+SuShi | 0.16729 | 1.03661 | 0.03672 |
| ds006131 | high-b | uncorrected | 0.86756 | 2.03640 | 0.00000 |
| | | RPG | 0.39535 | 1.09589 | 0.14804 |
| | | SuShi only | 0.65035 | 1.48554 | 0.14527 |
| | | POCS+SuShi | 0.83531 | 1.80073 | 0.15001 |

### 3.6 Normalised summary

> **RETRACTED AS A RANKING.** The table below is retained because the numbers are real, but
> §3.7 shows the metric is dominated by blur sensitivity and **cannot rank these methods**.
> Do not cite this table to say RPG beat SuShi, or that either beat POCS.

Change in ringing/sharpness versus uncorrected; negative is an improvement:

| run | vol | RPG | SuShi only | POCS+SuShi |
|---|---|---|---|---|
| NIBS AP | b≈0 | −13% | −8% | **+3%** |
| NIBS AP | high-b | −12% | −8% | **+14%** |
| NIBS PA | b≈0 | −11% | −7% | **+4%** |
| NIBS PA | high-b | −14% | −10% | **+13%** |
| ds006131 | b≈0 | −17% | −8% | **+1%** |
| ds006131 | high-b | −15% | +3% | **+9%** |

### 3.7 The metric is invalid for ranking these arms

Two controls were run after the fact. Both fail.

**Control 1 — does a pure blur "improve" the metric?** A Gaussian blur along the phase-encode
axis corrects no ringing whatsoever. On NIBS AP b≈0 it scores −5% (σ=0.4), −18% (σ=0.6),
−21% (σ=0.8), −19% (σ=1.0); on ds006131 high-b, up to −24%. Compare SuShi −8% and RPG −13%
to −17%. **A method that does nothing but blur outscores both real methods.** Normalising
ringing by sharpness does not neutralise blur — a second derivative is attenuated faster than
a first, so the ratio retains a large systematic preference for smoothing.

**Control 2 — at matched sharpness, does any method beat an equally-blurring filter?**
For each arm, the blur σ giving the same edge sharpness was found by interpolation over a fine
σ grid, and residual ringing compared. Positive would mean the method is genuinely
ringing-specific:

| run | vol | SuShi only | RPG | POCS+SuShi |
|---|---|---|---|---|
| NIBS AP | b≈0 | −7% | −5% | sharper than any blur |
| NIBS AP | high-b | −9% | −9% | sharper than any blur |
| NIBS PA | b≈0 | −8% | −8% | sharper than any blur |
| NIBS PA | high-b | −8% | −11% | sharper than any blur |
| ds006131 | b≈0 | −4% | −5% | −6% |
| ds006131 | high-b | −13% | −1% | −13% |

**Every arm is negative.** None removes more ringing than a blur that costs the same
sharpness. POCS sits off the scale in the opposite direction — it is *sharper* than the
uncorrected image, so no blur matches it, which means the §3.6 ranking was comparing
quantities that do not lie on a common axis.

**Consequence.** The apparent ordering in §3.6 tracks how much each method blurs, not how much
ringing it removes. The real-data arm comparison in this document establishes **nothing** about
the relative merit of RPG, complex SuShi, or POCS. H1 in §5 is therefore no longer a
hypothesis; it is confirmed, and it invalidates the ranking rather than merely qualifying it.

**What survives**, because it does not use this proxy: the zero-filled determination by
conjugate-mirror ratio (§3.2), the AP/PA side flip, the measured-vs-declared factor
disagreement on ds006131 (0.838 vs 0.875), the detector threshold defect, and the complex-TV
defect and its fix.


---

## 4. What has been ruled out

Do not spend effort re-checking these; each was verified explicitly.

**Implementation correctness of POCS.** Independently reviewed against the source:
`nfac` applied on every inverse FFT that feeds back into the iterate; deliberately omitted
in the phase-estimate path because a positive real scalar cancels in `atan2`; shifted↔raw
index mapping consistent between `pf_geometry.h` (producer) and `pocs.h` (consumer);
`m = min(a,b)` traced correctly (n=64, nm=16 → a=16, b=31, m=16, window [16,48]); the
degenerate `m ≤ 0` case falls back to a DC-only window rather than an empty one; data
consistency re-applied every iteration, not once; per-thread FFTW buffers, no leaks, no
race. Unit tests confirm data consistency to 2.4e-9 and a no-op on full k-space.

**FFT convention mismatch between files.** All three of `unring.h`, `pf_geometry.h`,
`pocs.h` verified to use `fftw_plan_dft_2d(ny, nx, ...)` with `buf[nx*y+x]`.

**Side inversion.** `ShiftedToUnshifted(n/2, n) == 0` verified algebraically for even and
odd `n`; confirmed empirically by the AP/PA band flipping ends on real data.

**Phase rescaling.** `|S| == M` asserted; phase spans `[-3.142, 3.140]`, i.e. the full
±π, consistent with Siemens 12-bit encoding.

**The arms being confused with each other.** Bit-identical outputs where POCS was skipped;
large difference where it ran.

**Complex SuShi marshalling.** `UnRingFullComplex` fed zero-imaginary input reproduces the
pre-existing magnitude `UnRingFull` output with max relative difference **exactly 0**. Note
this validates marshalling only — see
[Resolved: complex SuShi was not phase-rotation invariant](#resolved-complex-sushi-was-not-phase-rotation-invariant)
for a defect in the shift-selection objective that this test was structurally unable to detect,
now fixed.

**Interaction between POCS and the complex SuShi objective.** Tested directly by re-running
both affected arms after the TV fix; per-voxel output changes materially, aggregate results do
not. See the same section.

---

## Resolved: complex SuShi was not phase-rotation invariant

Raised by external review (`docs/chatgpt/2026-08-29-complex-pf-gibbs-followup.md`), confirmed
empirically, and **fixed**. Recorded here because it was a genuine defect in the complex path
that the original test suite could not have caught.

### The defect

`unring_1D` selected the subvoxel shift by minimising a total-variation term accumulated as
`fabs(dRe) + fabs(dIm)` — an L1 norm in the (Re, Im) plane. A global phase offset is an
arbitrary receiver convention, so unringing must commute with it:
`U(S·e^{iθ}) == U(S)·e^{iθ}`. An L1 norm in the complex plane does not: it changes under
rotation, so the selected shift depended on an arbitrary phase.

### Empirical confirmation, before the fix

Measured on the built binary with `--pocs 0`, comparing `U(S·e^{iθ})·e^{-iθ}` against `U(S)`:

| θ | max abs diff, rel. to peak | RMS magnitude diff | object voxels changed |
|---|---|---|---|
| π/8 | 1.17e-2 | 2.22e-3 | 16.3% |
| π/4 | 1.17e-2 | 3.00e-3 | 24.9% |
| π/3 | 1.47e-2 | 2.24e-3 | 20.2% |
| **π/2** | **4.4e-14** | **0.00** | **0.0%** |

The π/2 row identifies the mechanism unambiguously. Rotating by 90° swaps Re and Im, and
`|a| + |b|` is exactly invariant under that swap — so π/2 passes to machine precision while
every other angle fails. A generic numerical bug would not produce exact invariance at
precisely the angle where L1 happens to be invariant.

### Why the existing tests could not catch it

The `magnitude_equivalence` regression test feeds input with a zero imaginary channel. Then
`dIm ≡ 0`, so `|dRe| + |dIm| ≡ |dRe| ≡ |dS|` — **the L1 and L2 norms are identical exactly on
the case that test covers.** The strongest test in the suite was structurally blind to this
class of defect. Worth remembering when designing regression anchors: a test that pins the
degenerate case validates the marshalling, not the objective.

### The fix

`unring_1D` and `unring_2d` take a defaulted `bool complex_tv = false`. When true, both the
initial TV block and the sliding-window update use `std::hypot(dRe, dIm)`. When false they
retain the original expressions **verbatim and in their original evaluation order**, so
floating-point association is unchanged and the magnitude path is bit-for-bit identical.
Only `UnRingFullComplex` opts in. `UnRingFull`, `UnRing78`, `UnRing68` and the `Gibbs`
binary are untouched.

Verification after the fix:

- `magnitude_equivalence`: max relative difference **0** — legacy path bit-identical.
- `complex_phase_equivariance` (new, permanent ctest case): deviations fall from ~1.2e-2 to
  **~1e-7** across θ ∈ {π/8, π/4, π/3, π/2, 1.0}. That residual is the `complex<float>`
  storage floor (eps ≈ 1.2e-7), not a remaining defect.
- Suite is 7/7.

The test deliberately checks several angles. **A test that only checked π/2 would pass against
the broken implementation** — do not "simplify" it.

### Did it change the real-data conclusions? No.

Both affected arms were re-run on all three runs. Per-voxel output changed substantially;
the aggregate metric did not.

| arm | run / vol | old | new | max abs Δ magnitude, rel. to peak |
|---|---|---|---|---|
| SuShi only | NIBS AP b≈0 | −8% | −8% | 1.13e-1 |
| SuShi only | NIBS AP high-b | −8% | −8% | 1.13e-1 |
| SuShi only | NIBS PA b≈0 | −7% | −8% | 1.13e-1 |
| SuShi only | NIBS PA high-b | −10% | −11% | 1.13e-1 |
| SuShi only | ds006131 b≈0 | −8% | −9% | 6.58e-2 |
| SuShi only | ds006131 high-b | +3% | +3% | 6.58e-2 |
| POCS+SuShi | NIBS AP b≈0 | +3% | +3% | 1.01e-1 |
| POCS+SuShi | NIBS AP high-b | +14% | +14% | 1.01e-1 |
| POCS+SuShi | NIBS PA b≈0 | +4% | +4% | 9.89e-2 |
| POCS+SuShi | NIBS PA high-b | +13% | +13% | 9.89e-2 |
| POCS+SuShi | ds006131 b≈0 | +1% | +2% | 6.59e-2 |
| POCS+SuShi | ds006131 high-b | +9% | +9% | 6.59e-2 |

Individual voxels move by up to ~11% of peak — the defect was real and materially affected
the images — but no arm ordering, and no conclusion in this document, changes.

**This eliminates one confound.** The review's concern that POCS-attributed differences might
be an interaction between POCS's phase rotation and a non-invariant SuShi objective is
specifically tested here and does not hold. H1 (metric confound) and H2 (phase assumption)
both remain open; the tables in §3 stand as measured.

---

## 5. Candidate explanations, ranked

### H1 — The metric is invalid *(CONFIRMED, see §3.7 — no longer a hypothesis)*

POCS restores high-spatial-frequency content that zero-filling removed. That legitimately
raises **both** `|∂²A/∂y²|` (the ringing proxy) **and** `|∂A/∂y|` (sharpness).

The raw numbers are consistent with this: at NIBS AP b≈0 the POCS arm raised ringing from
0.16822 → 0.18328 (+9.0%) and sharpness from 1.12647 → 1.19044 (+5.7%) — both up, roughly
together. RPG by contrast lowered both (−33.9% and −24.1%). So POCS did not add oscillation
*disproportionately*; it scaled both up. A second-derivative proxy cannot distinguish
"restored edge detail" from "added ringing".

This means the small b≈0 penalties (+1% to +4%) may be **an artefact of the metric, not a
real degradation**. The high-b penalties (+9% to +14%) are large enough that they are
less easily explained away, but the same confound applies in principle.

**Resolved in §3.7: a pure blur outscores every arm, and at matched sharpness no arm beats a
blur. "POCS made it worse" is NOT established, and neither is "RPG is best". The metric cannot
rank these methods at all.** H2 below remains open and now needs ground truth to test.

### H2 — POCS's smooth-phase assumption is violated by real diffusion phase

POCS estimates image phase from the symmetric centre band and treats it as smooth and
low-order. Real DWI phase carries B₀ inhomogeneity, eddy-current phase, and — at high
b-value — motion-induced phase that is neither smooth nor low-order.

Supporting evidence: the penalty roughly triples from b≈0 to high-b in every run
(AP +3%→+14%, PA +4%→+13%, ds006131 +1%→+9%). The synthetic phantom satisfies the
assumption by construction and POCS excels there.

Against: b≈0 volumes also show a small penalty, so motion phase alone is not the whole story
(though see H1 — the b≈0 penalty may not be real).

### H3 — Parallel imaging and multiband break the data-consistency premise

POCS's data-consistency step assumes acquired k-space lines are measured ground truth. With
GRAPPA (`ParallelReductionFactorInPlane: 2` on ds006131) and multiband (3 and 4), a
substantial fraction of the "acquired" lines are themselves *reconstructed*, with correlated
noise and imperfect fidelity. Pinning them as truth every iteration may propagate their error
into the synthesised region.

Note ds006131 has in-plane acceleration and NIBS does not, yet both show the effect — so this
cannot be the sole cause, but multiband applies to both.

### H4 — The residual 5–14% amplitude in the "empty" band is real information

The band is 51–403× suppressed, not exactly zero. POCS overwrites whatever is there with
content synthesised from the phase estimate. If that residual is genuine signal (or a
vendor filter roll-off rather than noise), POCS is discarding information. Magnitude of the
effect is probably small, but it is untested.

### H5 — 2D slice-wise POCS applied to data reconstructed in 3D/multiband

`ApplyPOCS` operates per 2D slice. Multiband data is separated by slice-GRAPPA; residual
inter-slice leakage is not accounted for.

### H6 — Phase quantisation

12-bit phase gives steps of `2π/8192 ≈ 7.7e-4` rad. Order-of-magnitude estimate suggests
this is far too small to explain a k-space floor at the observed level, so this is listed
for completeness and should be cheap to dismiss.

---

## 6. Recommended next steps

Roughly in order of information gained per unit effort.

1. **Settle H1 by changing the metric.** Three options, best first:
   - *Ground-truth comparison on real data.* Acquire or locate a **full-Fourier** run of the
     same protocol/subject. Retrospectively truncate it to 6/8 and 7/8, run all arms, and
     score against the untruncated original. This converts the real-data evaluation into the
     same well-posed problem as the phantom and removes the proxy entirely. **This is the
     single most valuable experiment available.**
   - *Frequency-targeted ringing metric.* Gibbs ringing from a truncation at `k_max` has a
     known spatial period. Measure oscillation energy in that specific band beside edges
     (e.g. via a windowed FFT of intensity profiles perpendicular to edges) rather than a
     generic second derivative, which cannot separate ringing from detail.
   - *Downstream evaluation.* Score DTI/DKI fit residuals, FA map noise in homogeneous white
     matter, or test-retest consistency between the AP and PA runs. Answers "does this help
     the science" rather than "does this move a proxy".
2. **Test H2 directly.** Fit a low-order polynomial (or low-pass filter) to each volume's
   phase and compute the residual. Correlate per-volume POCS benefit against per-volume phase
   residual. If H2 holds, benefit should fall as residual rises, and b≈0 volumes should
   cluster at low residual.
3. **Vary the phase-estimate window.** `BuildPOCSMasks` uses a Hamming window over the full
   symmetric band (`m = min(a,b)`). Sweep the window width; a narrower window gives a
   smoother, more robust phase estimate at the cost of detail. If POCS is sensitive to this,
   H2 is implicated and a tunable `--pocs_phase_width` may be worth adding.
4. **Cross-check against a reference implementation.** Run
   [mchiew/partial-fourier-tutorial](https://github.com/mchiew/partial-fourier-tutorial)'s
   POCS on a single exported slice and compare to `ApplyPOCS` output. Cheap, and definitively
   settles any residual implementation doubt.
5. **Test H3** by evaluating on data acquired without in-plane acceleration, and ideally
   without multiband.
6. **Benchmark complex RPG** (§7). It uses phase, carries no smooth-phase assumption, and is
   mostly already implemented — plausibly a cheaper path to a working method than tuning POCS.
7. **Sweep `--pocs_iters`.** Currently 10, with observed final relative change 0.005075 on
   ds006131 — i.e. it hit the iteration cap without reaching `tol = 1e-4`. Check whether more
   iterations converge and whether that helps or hurts.

---

## 7. Complex RPG — an untested candidate, already half-implemented

### RPG and POCS+SuShi are alternatives, not variants

Worth stating plainly, because the two are easy to conflate and the distinction drives
everything below. Zero-filled partial Fourier produces ringing at **two** periods: the
ordinary Gibbs period from the symmetric part of k-space, plus a shorter-period component
from the asymmetric truncation. The two methods attack that from opposite ends.

- **RPG leaves the data alone and unrings more cleverly.** It never restores k-space. In
  `UnRing78` (`unring.h:676`) and `UnRing68` (`unring.h:862`): nearest-neighbour upsample ×3
  along PE, split rows into 4 interleaved sub-images (`SplitImageRows`, `unring.h:577`),
  ordinary SuShi on each, recombine (`unring.h:629`), downsample, then a final 1D pass. The
  resampling makes the shorter-period ringing land at the *ordinary* period on the resampled
  grid, where plain SuShi can remove it. Magnitude-only by design — built for when phase is
  unavailable.
- **POCS+SuShi fixes the data, then unrings ordinarily.** POCS fills the missing lines from
  the image phase; once k-space is symmetric the second period is gone, so ordinary Kellner
  SuShi suffices. `UnRingFullComplex` (`unring.h:466`) is plain SuShi — it contains none of
  RPG's resampling machinery.

Lee et al. present these as alternatives: POCS+SuShi when phase is available, RPG when it is
not. Neither is a version of the other.

This also explains a result in §2.2 that otherwise looks like a bug: on the zero-filled
synthetic phantom, complex SuShi *alone* scored RMSE 3.86 against 2.79 uncorrected — worse
than no correction. That is the predicted behaviour of applying ordinary SuShi to data that
still carries an asymmetric truncation. **The "SuShi only" arm is not a stand-in for complex
RPG**, and on all three real runs (all zero-filled) it is the theoretically wrong model.

### The candidate

There is a third option in the design space that this branch does not implement and that I
have not found in the literature: **apply RPG's resampling scheme to complex data instead of
magnitude.**

Why it is attractive given the current state of this investigation:

- It uses the phase information, so it does not throw away what magnitude RPG cannot see.
- It carries **no smooth-phase assumption** — which is the leading algorithmic suspect (H2)
  for POCS's behaviour on real diffusion data.
- It targets the actual artifact structure of zero-filled PF, which ordinary complex SuShi
  demonstrably does not.
- Most of it already exists in this file.

### What it would take

The pattern is the one already used for `UnRingFullComplex`, and the hard part is done:

1. A complex 3D overload of `UnRingFull` (mirroring `unring.h:403`) that passes
   `complex_tv = true` — the rotation-invariant TV criterion added in the section above
   already generalises the shift selection correctly, so no new algorithmic work is needed
   there.
2. Complex versions of `SplitImageRows` / `CombineImageRows`. These are pure index
   operations, so they are mechanical.
3. Resampling. `resample_3D_image` takes `ImageType3D` (real). RPG uses **nearest-neighbour**
   for both the ×3 upsample and the ÷3 downsample, and NN is index selection — so resampling
   the real and imaginary parts separately is *exactly* equivalent to resampling the complex
   image, with no interpolation error and no loss of phase equivariance. That makes this step
   easy and safe. Do not silently switch to linear or B-spline interpolation here; that would
   change RPG's behaviour as well as complicate the equivariance argument.
4. `UnRing78Complex` / `UnRing68Complex` assembled from the above, with the final `unring_1D`
   call passing `complex_tv = true`.
5. Leave `UnRing78` / `UnRing68` untouched, exactly as `UnRingFullComplex` left `UnRingFull`
   untouched. They are used by `TORTOISEProcess` and have no test coverage.

### How to test it

- Extend `complex_phase_equivariance` to cover the new functions. Every step above is either
  index selection or the fixed TV criterion, so it should pass at the `complex<float>` floor
  (~1e-7). If it does not, one of the resampling or split/recombine steps is not doing what
  this section assumes.
- Add it as a fifth arm to `run_validation.py`. That harness has ground truth, so unlike the
  real-data comparison it can actually rank arms. The specific prediction to check: on
  zero-filled synthetic data complex RPG should beat complex SuShi alone (which scores worse
  than no correction), and should not depend on phase smoothness the way POCS does.
- Then run it on the real data alongside the others — but note that ranking on real data still
  requires the ground-truth experiment in §6, not the retracted proxy.

### Caveats

This is a research idea, not a known-good method. RPG's resampling argument is derived for
magnitude images in Lee et al.; whether it carries over cleanly to complex data is exactly
what the phantom test above would establish. Treat a negative result there as informative
rather than as an implementation failure.


---

## RESOLVED: the detector threshold (was blocking)

**Fixed.** The description below is retained because it explains the failure mode and the
measurements that exposed it; the resolution is at the end of this section.

`DetectPFGeometry` (`pf_geometry.h:117-123`) normalises the k-space energy profile by its
**maximum**, which is the DC line. `zero_tol` is therefore effectively "fraction of DC
energy". The DC line dwarfs every line at the edge of k-space, so against any real noise
floor this test is far too strict:

- All three real runs are genuinely zero-filled (51–403× suppressed vs their mirrors), yet
  **all three were declined at the default `zero_tol = 1e-6`**.
- Per-run DC-relative profile minima: NIBS AP 1.01e-5, NIBS PA 6.05e-7, ds006131 5.04e-7.
- Lines below threshold, and the longest run (start, length):
  - NIBS AP — `1e-6`: 0 lines. `1e-4`: 35 lines at (1, 35).
  - NIBS PA — `1e-6`: 14 at (118, 14). `1e-4`: 36 at (104, 36).
  - ds006131 — `1e-6`: 8 at (6, 8). `1e-4`: 23 at (0, 23).

The synthetic phantom hid this completely: its band is *exactly* zero, so any threshold works.

**Recommended fix:** replace the DC-relative test with the scale-free conjugate-mirror ratio
— compare each candidate line's energy to its mirror about k-space centre, and classify as
zero-filled below roughly 0.05. This also naturally handles the case the current code gets
wrong in a second way: NIBS AP's suppressed run starts at ky index 1, not 0, so it touches
neither edge and is classified `InteriorBand` even at a loose threshold.

Note the mirror test is itself imperfect — conjugate symmetry holds exactly only for
real-valued images, and these are genuinely complex. It is nonetheless far more robust than
a DC-relative threshold, and the observed suppression is one-sided, matches the declared
width, and flips with phase-encode direction, which noise would not do.

### Resolution

Detection was demoted from decision-maker to diagnostic, and POCS made explicit opt-in.

- **`ComputePFDiagnostics`** (`pf_geometry.h`) takes the band width from the **declared**
  partial-Fourier factor rather than from a threshold, and measures the un-acquired band
  against its **conjugate mirror**. Ratios are formed per slice and per volume and aggregated
  by median with p10/p90, so a single pooled number cannot mislead and high-signal b=0 volumes
  cannot dominate. Both candidate sides are measured and the more suppressed one is inferred,
  with the asymmetry reported as confidence. The Nyquist line, which has no mirror, is
  excluded. Threshold for "compatible with zero filling" is a mirror ratio below 0.05.
- **POCS is now opt-in** (`--pocs` default `0`) and **requires `--pf_factor`**. It is never
  enabled from image-derived detection alone. If the compatibility check fails, POCS is refused
  with the measured ratio in the message; `--force_pf 1` overrides.
- **Every run prints a diagnostics block** — declared factor, expected missing lines, inferred
  side and asymmetry, band/mirror ratio with spread, compatibility verdict, plus the old
  literal-empty-band scan as one informational line.
- `DetectPFGeometry` is retained for that informational line (it still usefully reports the
  symmetric-zero-padding case) but no longer gates anything.

Separation achieved on synthetic data, `pf_diagnostics` test: **2e-15** for a zero-filled band
versus **0.9995** for full k-space. On the real NIBS AP run the median ratio is **0.0139**
(p10 0.0066, p90 0.0269, n=36 samples, side inferred low with 4868x asymmetry), agreeing with
the independent Python analysis in §3.2 — and POCS now correctly *runs* on data the old
DC-relative test wrongly declined.

Also surfaced by this work: the driver now warns when POCS reaches its iteration cap without
meeting `--pocs_tol`, which is what happens on real data (relative change 0.005 against a
requested 1e-4). That is item 7 in §6.

---

## Reproducing without the original data

### The synthetic side is fully in-repo

```bash
# build (Docker; see the plan doc for the dev image recipe)
<scratchpad>/tbuild.sh GibbsComplexTests
<scratchpad>/tbuild.sh ctest          # 6/6 expected

# synthetic harness
micromamba run -n linc311 python src/tools/UnRing/validation/run_validation.py \
    --workdir  <dir> \
    --container-workdir /data/validation \
    --runner   <dir>/trun.sh \
    --report   <dir>/report.json
```

### Requirements for a substitute real dataset

Any dataset meeting all of these will reproduce the real-data condition:

1. BIDS `part-mag` + `part-phase` 4D DWI pair from the same acquisition.
2. `PartialFourier < 1` in the sidecar (0.75 and 0.875 are the cases studied).
3. **Critically: exported without vendor partial-Fourier reconstruction**, i.e. mirror
   ratio below ~0.05 by the test below. This is the property that cannot be read from
   metadata and must be measured. If the candidate data fails this, it is the *other*
   case (already reconstructed) and POCS is legitimately inapplicable — see the artifact
   report for the RPG-vs-mrdegibbs decision rule.
4. Phase in scanner-native units; note the scaling used (Siemens 12-bit → `× π/4096`).

### Diagnostic: is a candidate dataset zero-filled?

```python
import nibabel as nb, numpy as np

def mirror_ratio(mag_path, phase_path, pf, pe_axis=1, vol=0, phase_scale=np.pi/4096.0):
    """Energy of the un-acquired band relative to its conjugate mirror.
    ~0 => zero-filled (POCS applicable);  ~1 => vendor-reconstructed (use mrdegibbs)."""
    m = nb.load(mag_path); p = nb.load(phase_path)
    z = m.shape[2] // 2
    mag = np.asarray(m.dataobj[:, :, z, vol], dtype=np.float64)
    ph  = np.asarray(p.dataobj[:, :, z, vol], dtype=np.float64) * phase_scale
    K = np.fft.fftshift(np.fft.fft2(mag * np.exp(1j * ph)))
    prof = (np.abs(K) ** 2).sum(axis=1 - pe_axis)      # sum over readout
    n = len(prof); c = n // 2; nm = round((1 - pf) * n)
    side = "low" if prof[:nm].mean() < prof[-nm:].mean() else "high"
    idx = np.arange(0, nm) if side == "low" else np.arange(n - nm, n)
    idx = idx[idx != 0]                                # Nyquist line has no mirror
    mir = 2 * c - idx
    keep = (mir > 0) & (mir < n); idx, mir = idx[keep], mir[keep]
    return prof[idx].sum() / prof[mir].sum(), side
```

Expected on the studied runs: `0.0198` (low), `0.0046` (high), `0.0025` (low).

### Assembling complex data

```python
mag = np.stack([np.asarray(m.dataobj[..., v], dtype=np.float64) for v in vols], axis=-1)
ph  = np.stack([np.asarray(p.dataobj[..., v], dtype=np.float64) for v in vols], axis=-1)
S   = mag * np.exp(1j * ph * np.pi / 4096.0)
assert np.allclose(np.abs(S), mag, rtol=1e-5)
nb.save(nb.Nifti1Image(S.astype(np.complex64), m.affine), "complex.nii.gz")
```

Note `nibabel` cannot fancy-index an `ArrayProxy`; select volumes one at a time and
`np.stack`, as above.

### Running the arms

```bash
# POCS + SuShi (auto-detect)
GibbsComplex -i complex.nii.gz -o pocs.nii.gz --output_magnitude pocs_mag.nii.gz --pe_dir 1

# SuShi only
GibbsComplex -i complex.nii.gz -o sushi.nii.gz --output_magnitude sushi_mag.nii.gz --pe_dir 1 --pocs 0

# force POCS when detection declines (needed given the threshold defect above)
GibbsComplex -i complex.nii.gz -o pocs.nii.gz --pe_dir 1 --pocs 1 \
             --pf_factor 0.75 --pf_side low --force_pf 1

# magnitude RPG baseline: Gibbs <in> <out> <kspace_coverage> <pe_dir>
Gibbs mag.nii.gz rpg.nii.gz 0.75 1
```

`--pe_dir 1` corresponds to `PhaseEncodingDirection: j` or `j-` (PE along the second image
axis). The suppressed side differs between `j` and `j-`, which is why `--pf_side` exists and
why auto-detection is preferable once the threshold is fixed.

---

## Limits of this finding

- Three runs, two subjects, one vendor, one scanner model. Says nothing about GE, Philips,
  or other reconstruction configurations.
- Two volumes per run, 16 central slices per volume.
- No ground truth on the real data — every real-data number is relative, and the proxy metric
  is the leading suspect (H1).
- The negative POCS result is a statement about *these already-exported, accelerated
  acquisitions*, not about POCS as a method. On synthetic zero-filled data the same code cuts
  error by ~99%.
- Nothing here has been evaluated downstream of unringing (tensor fitting, tractography).
