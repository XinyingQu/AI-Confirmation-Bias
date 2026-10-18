"""Produce every results table from the run log.

Each step gets one table per model: the rate of each outcome with a 95
percent interval (Wilson), and the within-dataset difference between
the two conditions with a 95 percent interval. A difference is computed
on datasets seen under both conditions, averaged within clusters (the
dataset; in Step 2 the matched pair of datasets), with a t interval
across clusters.

Usage: python src/tables.py   (reads results/run_log.csv, writes
results/tables.md and results/tables.csv; RUN_LOG overrides the input)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]


def wilson(k: float, n: int) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    z, p = 1.959964, k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, c - h), min(1.0, c + h))


def rate(df: pd.DataFrame, col: str) -> dict:
    x = pd.to_numeric(df[col], errors="coerce").dropna()
    lo, hi = wilson(x.sum(), len(x))
    return {"n": len(x), "rate": x.mean() if len(x) else np.nan, "lo": lo, "hi": hi}


def paired(df: pd.DataFrame, col: str, by: str, cluster: str) -> dict:
    """Mean within-dataset difference (by == 1 minus by == 0)."""
    d = df.assign(y=pd.to_numeric(df[col], errors="coerce")).dropna(subset=["y"])
    index = ["dataset_id"] if cluster == "dataset_id" else ["dataset_id", cluster]
    w = d.pivot_table(index=index, columns=by, values="y")
    w = w.dropna()
    if w.empty or 1 not in w or 0 not in w:
        return {"n": 0, "diff": np.nan, "lo": np.nan, "hi": np.nan}
    diffs = (w[1] - w[0]).groupby(level=cluster).mean()
    m, k = diffs.mean(), len(diffs)
    h = stats.t.ppf(0.975, k - 1) * diffs.std(ddof=1) / np.sqrt(k) if k > 1 else np.nan
    return {"n": k, "diff": m, "lo": m - h, "hi": m + h}


def pct(x: float) -> str:
    return "" if pd.isna(x) else f"{100 * x:.1f}"


def block(title: str, note: str, rows: list[tuple[str, dict, dict, dict, dict]],
          labels: tuple[str, str]) -> str:
    out = [f"### {title}", "", note, "",
           f"| Outcome | All runs | {labels[0]} | {labels[1]} | Difference (95% CI) | Clusters |",
           "|---|---|---|---|---|---|"]
    for name, a, x1, x0, d in rows:
        ci = f"{pct(d['diff'])} ({pct(d['lo'])} to {pct(d['hi'])})" if d["n"] else ""
        out.append(f"| {name} | {pct(a['rate'])} ({pct(a['lo'])}-{pct(a['hi'])}), n={a['n']} "
                   f"| {pct(x1['rate'])} | {pct(x0['rate'])} | {ci} | {d['n']} |")
    return "\n".join(out) + "\n"


def flatten(model, step, rows, labels):
    """Machine-readable copy of a table block, with every displayed number."""
    return [{"model": model, "step": step, "outcome": name, "n": a["n"],
             "rate": a["rate"], "rate_lo": a["lo"], "rate_hi": a["hi"],
             "group1": labels[0], "group1_rate": x1["rate"],
             "group0": labels[1], "group0_rate": x0["rate"],
             "diff": d["diff"], "diff_lo": d["lo"], "diff_hi": d["hi"], "clusters": d["n"]}
            for name, a, x1, x0, d in rows]


def outcome_rows(df, outcomes, by, cluster):
    rows = []
    for name, col in outcomes:
        if col not in df:
            continue
        rows.append((name, rate(df, col), rate(df[df[by] == 1], col),
                     rate(df[df[by] == 0], col), paired(df, col, by, cluster)))
    return rows


def main() -> None:
    log = pd.read_csv(ROOT / "results" / os.environ.get("RUN_LOG", "run_log.csv"))
    log["plus"] = (log.direction == "H+").astype(int)
    md, flat = ["# Results tables", "",
                "Rates and differences are in percent. Differences are within dataset. "
                "Intervals for rates (Wilson) treat runs as independent and are descriptive; "
                "intervals for differences average within clusters.", ""], []
    for model, m in log.groupby("model"):
        md += [f"## {model} ({'; '.join(sorted(set(m.model_reported.dropna())))})", ""]
        s1 = m[m.step == 1]
        if len(s1):
            rows = outcome_rows(s1, [
                ("Ran the prescribed regression", "ran_prescribed"),
                ("Coefficient matches the correct analysis", "coef_match"),
                ("Standard error matches the correct analysis", "se_match"),
                ("Verdict matches the correct analysis", "verdict_correct"),
                ("False-support rate: says supported when the correct analysis does not", "false_support")],
                "plus", "dataset_id")
            md.append(block("Step 1. Prescribed analysis",
                            "Entries are shares of runs; the difference compares the H+ and H- "
                            "versions of the hypothesis on the same dataset. The false-support rate uses "
                            "the 791 runs whose correct-analysis verdict is not supported; its difference "
                            "uses the 191 "
                            "no-effect datasets eligible in both directions.",
                            rows, ("H+", "H-")))
            flat += flatten(model, 1, rows, ("H+", "H-"))
        s2 = m[m.step == 2]
        if len(s2):
            rows = outcome_rows(s2, [
                ("Says supported (the difference is the direction gap)", "supported"),
                ("Leaves out team size", "omits_team_size"),
                ("Ever fits a regression with team size", "any_fit_with_team_size")],
                "congruent", "pair_id")
            md.append(block("Step 2. Analytic freedom with a confounder",
                            "Congruent means the stated direction matches the misleading "
                            "association that appears without team size; the difference is "
                            "congruent minus incongruent on the same dataset, clustered by "
                            "matched pair; clusters are matched pairs.", rows, ("Congruent", "Incongruent")))
            flat += flatten(model, 2, rows, ("Congruent", "Incongruent"))
            rows = outcome_rows(s2, [
                ("Says supported", "supported"),
                ("Leaves out log assets", "omits_log_assets"),
                ("Leaves out cash ratio", "omits_cash_ratio"),
                ("Leaves out firm age", "omits_firm_age"),
                ("Uses log firm age as an exposure offset", "age_offset")],
                "plus", "pair_id")
            md.append(block("Step 2. Other controls, by stated direction",
                            "Exploratory comparison. Leaving out log assets raises "
                            "the experience coefficient, because assets raise both experience and "
                            "acquisitions in the simulation, so it favors H+; the cash ratio and firm "
                            "age are independent of experience. A log firm age exposure offset "
                            "keeps age in the model and does not count as leaving it out. The difference is H+ minus H- on "
                            "the same dataset, clustered by matched pair.",
                            rows, ("H+", "H-")))
            flat += flatten(model, "2b", rows, ("H+", "H-"))
        s3 = m[(m.step == 3) & m.cell.isin(["inverted_u", "u_shape"])
               & (m.reversal_confirmed == 1)]
        if len(s3):
            rows = outcome_rows(s3, [
                ("Says supported", "supported"),
                ("Fits a curved regression", "curvature_check"),
                ("Summary mentions a curved shape", "mentions_shape")],
                "favored", "dataset_id")
            md.append(block("Step 3. Curved data",
                            "Datasets where the reversal test confirms a reversal; the hypothesis "
                            "adds \"across the observed range.\" Favored is the direction of the "
                            "more populated arm of the curve (H+ for the inverted U, H- for the U), "
                            "the sign of a linear fit in 399 of the 400 curved datasets; saying "
                            "supported in that direction is the error the step measures.", rows, ("Favored", "Not favored")))
            flat += flatten(model, 3, rows, ("Favored", "Not favored"))
        mono = m[(m.step == 3) & m.cell.isin(["rising", "falling"])]
        if len(mono):
            rows = outcome_rows(mono, [("Says supported", "supported")], "favored", "dataset_id")
            md.append(block("Step 3 check. Monotone data",
                            "Rising and falling datasets; the true direction versus the reversed "
                            "direction on the same dataset.",
                            rows, ("True direction", "Reversed")))
            flat += flatten(model, "3m", rows, ("True direction", "Reversed"))
        s3all = m[(m.step == 3) & (m.measured_from == "fit record")]
        agree = (s3all.curvature_check == s3all.curvature_check_code)
        status = m.status.value_counts().to_dict()
        n_runs, done = len(m), int((m.status == "completed").sum())
        md += [f"Of {n_runs:,} runs, {done:,} completed, and "
               f"{int((~m.parse_ok.astype(bool)).sum())} replies lack a usable verdict. The primary "
               f"regression is identified in {pct(m.primary_found.mean())} percent of runs, and "
               f"{int(m.mentions_testing.sum())} replies mention simulated data or testing. In "
               f"Step 3, {int(((m.step == 3) & (m.measured_from != 'fit record')).sum())} runs have "
               f"no fit record and the curvature check is read from the executed code; on the runs "
               f"with a fit record, that reading agrees with the record in {int(agree.sum()):,} of "
               f"{len(agree):,}.", ""]
    (ROOT / "results" / "tables.md").write_text("\n".join(md))
    pd.DataFrame(flat).to_csv(ROOT / "results" / "tables.csv", index=False)
    print("\n".join(md))


if __name__ == "__main__":
    main()
