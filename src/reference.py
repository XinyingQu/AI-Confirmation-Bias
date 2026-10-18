"""Reference analyses, run by our own code on every dataset.

1. The reference regression (all steps): a negative binomial (NB2)
   regression of acquisitions on experience and the five controls, with
   the dispersion parameter estimated by maximum likelihood. The verdict
   for a stated direction is "supported" when the experience coefficient
   has that sign with a two-sided p-value below 0.05, "not_supported"
   when it is significant with the opposite sign, and "inconclusive"
   otherwise.

2. The reversal test (Step 3): the same regression with experience squared
   added. A dataset confirms a reversal when the turning point lies
   inside the observed range of experience and the slope of the log mean
   is significant with opposite signs at the lowest and the highest
   observed experience (the Lind and Mehlum 2010 test, each slope
   one-sided at 5 percent).

Usage: python src/reference.py   (writes results/reference.csv)
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
CONTROLS = ["log_assets", "firm_age", "team_size", "industry_ma", "cash_ratio"]
FULL = "acquisitions ~ experience + " + " + ".join(CONTROLS)


def fit_nb(formula: str, df: pd.DataFrame):
    # Newton first by design (statsmodels' default is BFGS); BFGS if Newton fails or stalls.
    for method, maxiter in (("newton", 200), ("bfgs", 5000)):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = smf.negativebinomial(formula, df).fit(
                    method=method, maxiter=maxiter, disp=0)
        except np.linalg.LinAlgError:
            continue
        # Newton can report convergence with missing estimates; require
        # finite estimates and standard errors as well.
        if (res.mle_retvals.get("converged", False)
                and np.isfinite(res.params).all() and np.isfinite(res.bse).all()):
            return res
    raise RuntimeError(f"NB fit did not converge: {formula}")


def verdict(coef: float, p: float, direction: str) -> str:
    """Verdict for a stated direction ('+' or '-')."""
    if p >= 0.05:
        return "inconclusive"
    agrees = (coef > 0) == (direction == "+")
    return "supported" if agrees else "not_supported"


def reference_regression(df: pd.DataFrame, formula: str = FULL) -> dict:
    res = fit_nb(formula, df)
    return {"coef": float(res.params["experience"]),
            "se": float(res.bse["experience"]),
            "p": float(res.pvalues["experience"])}


def curve_test(df: pd.DataFrame) -> dict:
    res = fit_nb(FULL.replace("experience", "experience + I(experience ** 2)", 1), df)
    b1, b2 = res.params["experience"], res.params["I(experience ** 2)"]
    cov = res.cov_params().loc[["experience", "I(experience ** 2)"],
                               ["experience", "I(experience ** 2)"]].to_numpy()
    lo, hi = float(df.experience.min()), float(df.experience.max())

    def slope(x):
        g = np.array([1.0, 2 * x])
        return b1 + 2 * b2 * x, float(np.sqrt(g @ cov @ g))

    s_lo, se_lo = slope(lo)
    s_hi, se_hi = slope(hi)
    tp = -b1 / (2 * b2) if b2 != 0 else np.nan
    inside = bool(lo < tp < hi)
    # One-sided p-values in the direction of a reversal of each shape.
    p_peak = max(norm.sf(s_lo / se_lo), norm.cdf(s_hi / se_hi))
    p_trough = max(norm.cdf(s_lo / se_lo), norm.sf(s_hi / se_hi))
    shape = "peak" if b2 < 0 else "trough"
    p_rev = p_peak if shape == "peak" else p_trough
    return {"b_linear": float(b1), "b_squared": float(b2),
            "turning_point": float(tp), "exp_min": lo, "exp_max": hi,
            "slope_at_min": float(s_lo), "slope_at_max": float(s_hi),
            "shape": shape, "p_reversal": float(p_rev),
            "reversal_confirmed": bool(inside and p_rev < 0.05)}


def analyze_dataset(df: pd.DataFrame, step: int) -> dict:
    full = reference_regression(df)
    row = {"ref_coef": full["coef"], "ref_se": full["se"], "ref_p": full["p"],
           "ref_verdict_plus": verdict(full["coef"], full["p"], "+"),
           "ref_verdict_minus": verdict(full["coef"], full["p"], "-")}
    if step == 2:
        no_team = FULL.replace(" + team_size", "")
        omit = reference_regression(df, no_team)
        row.update({"omit_coef": omit["coef"], "omit_se": omit["se"],
                    "omit_p": omit["p"]})
    if step == 3:
        # The straight-line fit is the reference regression itself.
        row.update({f"curve_{k}": v for k, v in curve_test(df).items()})
    return row


def main() -> None:
    key = pd.read_csv(ROOT / "key" / "key.csv", keep_default_na=False)
    rows = []
    for k in key.itertuples():
        df = pd.read_csv(ROOT / "data" / f"{k.dataset_id}.csv")
        rows.append({"dataset_id": k.dataset_id, "step": k.step, "cell": k.cell,
                     **analyze_dataset(df, k.step)})
    out = ROOT / "results" / "reference.csv"
    out.parent.mkdir(exist_ok=True)
    fields = list(dict.fromkeys(f for r in rows for f in r))
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} reference rows to {out}")


if __name__ == "__main__":
    main()
