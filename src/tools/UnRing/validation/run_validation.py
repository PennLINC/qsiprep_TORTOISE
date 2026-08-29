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
# Calibrated 2026-08-29: measured ratios were 0.1240 (pf75) and 0.1607
# (pf875); thresholds below are each rounded up to the next 0.05.
THRESHOLDS = {
    "pf75_complex_pocs_vs_rpg_ratio": 0.15,
    "pf875_complex_pocs_vs_rpg_ratio": 0.20,
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
