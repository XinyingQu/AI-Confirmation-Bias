"""Check that the simulated data behave as designed, before any model
sees them. Plain regressions only; no model runs.

(i)   With no planted effect, the full-control regression rejects at
      about the 5 percent rate (2,000 fresh draws, so the rate is precise).
(ii)  In the Step 2 data, the regression without founding-team size
      shows a strong association, which disappears once team size is
      added.
(iii) In the Step 3 curved data, the straight-line fit favors the
      direction of the planted shape's more populated side, and the reversal test
      confirms a reversal inside the observed range; in the monotone
      data it does not.

Datasets are never discarded because a realization fails to show the
pattern; the checks concern the simulation as a whole.

Usage: python src/check_data.py   (writes results/data_checks.txt;
exits with an error if any check fails)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import reference  # noqa: E402
import simulate  # noqa: E402


def main() -> None:
    ref = pd.read_csv(ROOT / "results" / "reference.csv")
    lines, ok = [], True

    def check(label, value, lo, hi):
        nonlocal ok
        passed = lo <= value <= hi
        ok &= passed
        lines.append(f"{'PASS' if passed else 'FAIL'}  {label}: {value:.3f} "
                     f"(required between {lo} and {hi})")

    # (i) size of the reference test on fresh null draws (indices not used
    # by any study dataset)
    p = [reference.reference_regression(pd.DataFrame(simulate.make(1, "none", 10_000 + i)))["p"]
         for i in range(2000)]
    check("(i) rejection rate, 2,000 fresh no-effect datasets", np.mean(np.array(p) < 0.05), 0.035, 0.065)
    s1 = ref[ref.step == 1]
    check("(i) rejection rate, Step 1 no-effect datasets", np.mean(s1[s1.cell == "none"].ref_p < 0.05), 0.0, 0.10)
    check("    power, Step 1 positive datasets", np.mean(s1[s1.cell == "positive"].ref_p < 0.05), 0.95, 1.0)
    check("    power, Step 1 negative datasets", np.mean(s1[s1.cell == "negative"].ref_p < 0.05), 0.95, 1.0)

    # (ii) Step 2: strong association without team size, none with it
    for cell, sgn in (("apparent_positive", 1), ("apparent_negative", -1)):
        s2 = ref[ref.cell == cell]
        check(f"(ii) {cell}: significant in the apparent direction without team size",
              np.mean((s2.omit_p < 0.05) & (np.sign(s2.omit_coef) == sgn)), 0.95, 1.0)
        check(f"(ii) {cell}: rejection rate with team size", np.mean(s2.ref_p < 0.05), 0.0, 0.10)

    # (iii) Step 3
    for cell, sgn in (("inverted_u", 1), ("u_shape", -1)):
        s3 = ref[ref.cell == cell]
        check(f"(iii) {cell}: straight-line fit favors the {'+' if sgn > 0 else '-'} direction",
              np.mean(np.sign(s3.ref_coef) == sgn), 0.90, 1.0)
        check(f"(iii) {cell}: reversal confirmed by the reversal test",
              np.mean(s3.curve_reversal_confirmed.astype(str) == "True"), 0.75, 1.0)
    for cell in ("rising", "falling"):
        s3 = ref[ref.cell == cell]
        check(f"(iii) {cell}: reversal confirmed (should be rare)",
              np.mean(s3.curve_reversal_confirmed.astype(str) == "True"), 0.0, 0.05)

    text = "\n".join(lines) + f"\n\nALL CHECKS {'PASS' if ok else 'FAIL'}\n"
    (ROOT / "results" / "data_checks.txt").write_text(text)
    print(text)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
