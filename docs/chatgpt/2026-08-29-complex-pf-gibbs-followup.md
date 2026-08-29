# Follow-up recommendations for complex-valued partial-Fourier Gibbs removal in TORTOISE

The new writeup changes the direction of the project in an important way. It does **not** convince me that the overall idea in QSIPrep issue #1099 is wrong, but it does convince me that **standard POCS should no longer be treated as the obvious phase-aware solution for zero-filled DWI**.

The writeup also suggests that there are actually **three** technical issues to address before interpreting the present real-data comparison as definitive.

## 1. The zero-fill detector should be demoted from decision-maker to diagnostic

The current detector normalizes the phase-encoding k-space energy profile by its maximum—effectively the DC line—and identifies a contiguous run below a fixed threshold. The real data show why that is brittle: the nominally empty band can be 50–400× suppressed relative to its conjugate-side counterpart and still be nowhere near `1e-6` of DC; one NIBS case also leaves the Nyquist-edge line nonzero, so the suppressed region does not literally touch the array edge.

I would not simply replace `1e-6` with a looser fixed threshold such as `0.05`. Instead, separate three questions that the current code conflates:

1. **What PF geometry was acquired?**  
   Get the PF factor and dimension from acquisition metadata whenever possible. Do not estimate 0.75 versus 0.875 from reconstructed image data if BIDS already tells us.

2. **Which side appears to have been omitted?**  
   Infer this from the expected-width bands on the two sides. This is a reasonable image-derived quantity.

3. **Does the exported complex image look sufficiently like a zero-filled reconstruction that POCS is even worth considering?**  
   This should be a compatibility/confidence score, not the same threshold used to determine geometry.

For the third question, the conjugate-side energy ratio in the writeup is much better than DC normalization, but I would make it more robust:

- use the **known PF width**, rather than a threshold-defined width;
- tolerate ±1–2 boundary lines;
- exclude the Nyquist line;
- calculate the ratio separately for many slices and volumes;
- normalize within each slice/volume before aggregation so b≈0 volumes do not dominate the energy sum;
- report a median/IQR or quantiles rather than only one pooled number;
- test both candidate sides and require a substantial asymmetry.

I would also investigate a **phase-demodulated mirror score**. Estimate the same smooth central phase that POCS would use, multiply the complex image by `exp(-i*phi_low)`, and then assess Hermitian consistency. If that demodulated image still has strong high-spatial-frequency complex phase, two things become true simultaneously: conjugate-mirror detection becomes less trustworthy **and POCS's phase assumption itself is questionable**.

For an initial production implementation, however, I would be more conservative:

> **Never automatically enable POCS from reconstructed-image detection alone.**

Use detection to report that an image is or is not *compatible with* zero filling, but require explicit reconstruction provenance or an explicit `zero-fill` declaration to actually enable POCS. This is consistent with the caution already built into QSIPrep issue #1099.

## 2. The POCS degradation at high b-value is physically plausible

The strongest real-data result is the systematic b-value interaction for POCS+SuShi:

- NIBS AP: +3% → +14%
- NIBS PA: +4% → +13%
- ds006131: +1% → +9%

from b≈0 to high-b.

That is the direction expected if the main limitation is POCS's low-resolution phase constraint.

The current branch computes phase from the symmetric central k-space region with a Hamming window and then, on every POCS iteration, replaces the image phase with that fixed estimate while retaining the current magnitude. That is a conventional POCS approach, but DWI is an unusually difficult case because diffusion encoding makes the signal phase highly sensitive to motion and other sources of non-smooth phase variation.

There is prior DWI-specific literature showing that conventional POCS can suffer increased noise and residual blurring when diffusion-induced phase is complex or rapidly varying, and can become little better—or even worse—than zero filling in unfavorable conditions. More general partial-Fourier literature likewise identifies rapid local phase variation as a fundamental limitation of low-resolution-phase POCS.

I would therefore elevate the writeup's H2 from a generic candidate explanation to:

> **Leading algorithmic explanation, strongly supported by prior DWI literature, but not yet isolated experimentally in these brain datasets.**

The RPG paper's favorable treatment of POCS+SuShi should therefore be interpreted conditionally: it shows the advantage when the missing PF information is recoverable under the POCS phase model, but it does not establish that conventional low-resolution-phase POCS will outperform zero filling for arbitrary diffusion-induced phase.

## 3. The current complex SuShi objective is not phase-rotation invariant

This should be fixed **before doing much more POCS benchmarking**.

The existing TORTOISE `unring_1D` chooses shifts using a criterion equivalent to

\[
|\Delta \operatorname{Re}(S)| + |\Delta \operatorname{Im}(S)|,
\]

because it adds the absolute real and imaginary differences separately.

That was harmless when the imaginary component was always zero. For genuinely complex-valued data it is problematic.

A more appropriate complex-valued TV term is

\[
|\Delta S|
=
\sqrt{(\Delta \operatorname{Re}S)^2 + (\Delta \operatorname{Im}S)^2}.
\]

Why this matters: multiplying an MRI by an arbitrary global phase factor `exp(i*theta)` should not change which subvoxel shift SuShi considers optimal. The current L1 norm in the real/imaginary coordinate system changes under rotation of the complex plane; the Euclidean complex modulus does not.

The writeup's regression test showing that zero-imaginary complex input exactly reproduces ordinary `UnRingFull` is useful, but it does **not** validate the genuinely complex case.

I would immediately add a global-phase-equivariance test:

\[
U(S e^{i\theta}) \approx U(S)e^{i\theta}
\]

for many values of `theta`.

I expect the current implementation to fail this test.

This potentially contaminates the POCS comparison in a subtle way: **POCS changes the phase field, and then SuShi's shift-selection criterion itself depends on the orientation of that phase in the real/imaginary plane.** Some difference currently attributed to POCS may therefore actually be an interaction between POCS and an inappropriate complex-SuShi TV norm.

I would implement a separate complex path using `hypot(dreal, dimag)` while leaving legacy magnitude SuShi/RPG bit-for-bit unchanged.

# Revised implementation and validation priorities

## Phase 1 — Make complex SuShi itself defensible

Before further POCS work:

- replace the real+imaginary L1 criterion with complex-magnitude TV in the complex path;
- add global-phase-equivariance tests;
- test arbitrary spatially varying but smooth phase;
- test rapid phase variation;
- compare magnitude outputs to complex-input `mrdegibbs`;
- rerun the existing real-data **SuShi-only** arm.

This creates a reliable complex baseline independent of PF restoration.

If complex SuShi still trails RPG on zero-filled 6/8 and 7/8 data, that is entirely plausible: ordinary SuShi does not remove the second, PF-specific ringing period that motivated RPG.

## Phase 2 — Separate PF detection from POCS eligibility

Change the conceptual API from:

```text
detect zero filling
       ↓
yes → POCS
```

to:

```text
acquisition metadata
       ↓
PF geometry

complex-k-space diagnostic
       ↓
zero-fill compatibility score

phase diagnostic
       ↓
POCS suitability score
```

Initially make:

```text
POCS = explicit opt-in only
```

The automatic detector can mature independently.

Useful diagnostics would include, for example:

```text
Declared PF:                  0.750
Expected missing lines:       35
Likely missing side:           low
Missing/opposite energy:       0.0198
Zero-fill compatibility:       strong

Central-phase residual:        ...
High-frequency phase energy:   ...
POCS phase compatibility:      poor
```

That is much more informative than a single `DetectedPF=true` flag.

## Phase 3 — Directly test whether phase complexity predicts POCS failure

For **every diffusion volume**:

1. construct the POCS low-resolution phase estimate;
2. within a high-SNR brain mask calculate the circular residual

\[
\Delta\phi = \arg(S e^{-i\phi_{\rm low}});
\]

3. quantify:
   - circular RMS residual;
   - residual phase-gradient magnitude;
   - high-spatial-frequency phase energy;
   - fraction of voxels exceeding thresholds such as 0.25, 0.5, and 1 rad;
4. calculate the POCS effect independently for that volume.

Then examine

\[
\text{POCS benefit}
\quad\text{vs}\quad
\text{high-frequency phase residual}.
\]

If high-b volumes systematically move toward larger phase residuals and poorer POCS performance, that would provide a strong mechanistic result rather than merely a negative benchmark.

Do this per-volume for **diagnosis**, but I would not yet let TORTOISE switch POCS on/off independently across gradient directions. A volume-dependent reconstruction/PSF could itself introduce q-space inconsistency.

## Phase 4 — Resolve the real-data metric problem with actual ground truth

The second-derivative ringing metric is fundamentally confounded by resolution recovery. The b≈0 results in particular could simply show POCS restoring high frequencies: both ringing and sharpness increase by similar amounts.

The decisive experiment is therefore:

```text
real fully sampled complex DWI
             ↓
       reference image
             │
             ├── retrospectively zero-fill 7/8
             ├── retrospectively zero-fill 6/8
             └── retrospectively zero-fill 5/8
```

Then compare:

- zero filling;
- corrected complex SuShi;
- RPG;
- POCS;
- POCS + corrected complex SuShi.

Now magnitude RMSE, phase error, edge recovery, and diffusion-model bias all have a real target.

Crucially, retain the phase of the fully sampled **real DWI**. Do not rely on another synthetic smooth-phase phantom. The current phantom is close to an ideal POCS case because the phase model is deliberately smooth and compatible with the low-resolution phase estimate, which explains the near-perfect recovery.

I would also build synthetic variants with increasing phase complexity:

```text
smooth phase
↓
smooth + realistic B0 phase
↓
motion-like spatial phase
↓
measured real-DWI phase
```

That should reveal the transition where POCS stops helping.

## Phase 5 — Sweep POCS convergence

The current real-data test also shows that POCS was not necessarily converged. In ds006131 it stopped at 10 iterations with a relative change of approximately 0.005 despite a requested tolerance of `1e-4`.

Run at least:

```text
1
2
5
10
20
50
100
```

iterations and store the full convergence trajectory.

Two outcomes would be informative:

- **error improves with convergence** → ten iterations was simply insufficient;
- **error/noise worsens while the numerical iterate converges** → the algorithm is successfully converging to the wrong solution because its phase constraint is inappropriate.

The second outcome would strongly support the phase-model explanation.

# If conventional POCS remains poor, do not abandon complex PF reconstruction

Instead, replace the standard POCS component with a method less dependent on a globally smooth low-resolution phase constraint.

Potential next candidates include:

- **local-phase-recovery iterative PF methods**, which allow higher-frequency/local phase to be recovered rather than freezing the image to a single low-resolution phase estimate;
- **k-space convolution/data-fitting approaches**, which were developed specifically to reduce sensitivity to rapidly varying image phase;
- DWI-specific learned approaches such as **DRPF**;
- the complex-input CNN strategy from **Muckley et al.**, already discussed in QSIPrep issue #1099, which jointly targets PF missing information and Gibbs/noise without imposing conventional POCS's smooth-phase model.

The longer-term algorithm hierarchy could therefore become:

```text
                         complex PF data
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
         scanner-reconstructed          true zero-fill
                 │                           │
          complex SuShi              PF restoration method
                                             │
                                  ┌──────────┴──────────┐
                                  │                     │
                              POCS if valid      robust alternative
                                  │            (local-phase / k-space
                                  │             fitting / learned)
                                  └──────────┬──────────┘
                                             │
                                      complex SuShi
```

# Recommended status for the current branch

Keep the branch, but change its conceptual status from **"implementation of the proposed solution"** to an **experimental framework for evaluating complex PF reconstruction**.

The current architecture is already useful because it cleanly separates detection, POCS, and complex SuShi, and permits POCS to be disabled. That modularity should be preserved and extended so that the PF-restoration stage is pluggable, for example:

```cpp
enum PFRestoreMethod {
    None,
    POCS,
    LocalPhasePOCS,
    KSpaceFit
};
```

with the possibility of adding a learned backend later.

# Recommended immediate order of work

1. **Fix complex SuShi's non-rotation-invariant TV criterion.**
2. **Disable automatic POCS selection as intended production behavior.**
3. **Replace the detector with metadata-guided, robust zero-fill compatibility diagnostics.**
4. **Measure real-DWI phase complexity and correlate it with POCS performance.**
5. **Sweep POCS convergence.**
6. **Acquire or locate fully sampled real complex DWI and retrospectively PF-truncate it.**
7. Only then decide whether standard POCS deserves a supported mode.
8. If not, benchmark local-phase-recovery and/or k-space-fitting PF reconstruction before moving to a neural approach.

The most consequential immediate point is the complex-SuShi objective: **the current experiments compare POCS against a complex SuShi implementation whose shift-selection objective is itself not properly rotation-invariant in the complex plane.** That should be corrected before spending substantial effort tuning the zero-fill detector or POCS. After that, the b-dependent POCS degradation becomes the central scientific question, and existing DWI partial-Fourier literature makes it quite plausible that it reflects a real limitation of conventional POCS rather than an implementation bug.

## Sources used in this assessment

- Attached implementation/validation writeup: `2026-08-29-pocs-on-simulated-and-real-dwi(1).md`
- QSIPrep issue #1099: https://github.com/PennLINC/qsiprep/issues/1099
- TORTOISE complex-degibbs branch: https://github.com/PennLINC/qsiprep_TORTOISE/tree/complex-degibbs
- Lee HH, Novikov DS, Fieremans E. *Removal of Partial Fourier-Induced Gibbs (RPG) Ringing artifacts in MRI*. Magnetic Resonance in Medicine. 2021;86:2733–2750.
- Gadjimuradov F, et al. DWI partial-Fourier reconstruction work examining POCS limitations in the presence of diffusion-related phase.
