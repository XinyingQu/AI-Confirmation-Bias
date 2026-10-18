"""Print the figures quoted in the text that the tables do not contain.

Usage: RUNS_DIR=runs python src/extras.py   (writes results/extras.txt)

Reads data/, key/, results/reference.csv, results/run_log.csv, and the
run records.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import reference  # noqa: E402
import score  # noqa: E402

REMOVED = "[output removed: it listed or probed folders outside the study folder]"


def main() -> None:
    key = pd.read_csv(ROOT / "key" / "key.csv", keep_default_na=False)
    log = pd.read_csv(ROOT / "results" / "run_log.csv")
    out = []

    # Step 2: the reference regression without log assets, cash ratio, and firm age
    s2 = key[key.step == 2].dataset_id
    pos_red = pos_full = 0
    for d in s2:
        df = pd.read_csv(ROOT / "data" / f"{d}.csv")
        red = reference.fit_nb("acquisitions ~ experience + team_size + industry_ma", df)
        full = reference.fit_nb(reference.FULL, df)
        pos_red += red.params["experience"] > 0 and red.pvalues["experience"] < 0.05
        pos_full += full.params["experience"] > 0 and full.pvalues["experience"] < 0.05
    out.append(f"Step 2, significant positive experience estimate, {len(s2)} datasets: "
               f"without log assets, cash ratio, and firm age {pos_red} "
               f"({100 * pos_red / len(s2):.1f} percent); with all controls {pos_full} "
               f"({100 * pos_full / len(s2):.1f} percent)")

    # Step 2: runs whose primary regression keeps all controls
    m2 = log[log.step == 2].copy()
    lab = m2.primary_labels.fillna("")
    keep = (lab.str.contains("log_assets") & lab.str.contains("cash_ratio")
            & (lab.str.contains("firm_age") | (m2.age_offset == 1)))
    for model, g in m2[keep].groupby("model"):
        rates = g.groupby("direction").supported.agg(["mean", "size"])
        out.append(f"Step 2, {model}, runs keeping all controls, share supported: " + "; ".join(
            f"{d} {100 * r['mean']:.1f} percent of {int(r['size'])}" for d, r in rates.iterrows()))

    # Data: mean acquisitions by cell; share of ventures beyond the turning point
    for cell, g in key.groupby("cell"):
        acq = np.concatenate([pd.read_csv(ROOT / "data" / f"{d}.csv").acquisitions for d in g.dataset_id])
        out.append(f"Mean acquisitions per venture, {cell}: {acq.mean():.2f}")
    ref = pd.read_csv(ROOT / "results" / "reference.csv").set_index("dataset_id")
    curved = key[key.cell.isin(["inverted_u", "u_shape"])]
    share = {d: (pd.read_csv(ROOT / "data" / f"{d}.csv").experience > t).mean()
             for d, t in zip(curved.dataset_id, curved.turning_point.astype(float))}
    conf = [d for d in share if str(ref.loc[d, "curve_reversal_confirmed"]) == "True"]
    out.append(f"Share of ventures above the turning point: {100 * np.mean(list(share.values())):.2f} "
               f"percent over the {len(share)} curved datasets; "
               f"{100 * np.mean([share[d] for d in conf]):.2f} percent over the {len(conf)} confirmed")

    # Step 3: all curved datasets, and those the reversal test does not confirm
    m3 = log[(log.step == 3) & log.cell.isin(["inverted_u", "u_shape"]) & (log.favored == 1)]
    for model, g in m3.groupby("model"):
        un = g[g.reversal_confirmed == 0]
        out.append(f"Step 3, {model}, favored direction, share supported: all {len(g)} curved "
                   f"datasets {100 * g.supported.mean():.1f} percent; the {len(un)} unconfirmed "
                   f"datasets {int(un.supported.sum())} of {len(un)}")

    # Step 2: pair-level direction gaps
    for model, g in log[log.step == 2].groupby("model"):
        w = g.pivot_table(index=["dataset_id", "pair_id"], columns="congruent", values="supported")
        d = (w[1] - w[0]).groupby(level="pair_id").mean()
        out.append(f"Step 2, {model}, matched pairs by within-pair difference (congruent minus "
                   f"incongruent): " + "; ".join(f"{k:+.1f}: {v}" for k, v in sorted(d.value_counts().items())))

    # Step 2: supported by congruence, among runs keeping all controls and the rest;
    # and how many nonzero pairs the two models share
    m2c = log[log.step == 2].copy()
    labc = m2c.primary_labels.fillna("")
    m2c["keeps_all"] = (labc.str.contains("log_assets") & labc.str.contains("cash_ratio")
                        & (labc.str.contains("firm_age") | (m2c.age_offset == 1)))
    for model, g in m2c.groupby("model"):
        r = g.groupby(["keeps_all", "congruent"]).supported.mean()
        out.append(f"Step 2, {model}, share supported, keeping all controls: congruent "
                   f"{100 * r[(True, 1)]:.1f}, incongruent {100 * r[(True, 0)]:.1f} percent; leaving "
                   f"a control out: congruent {100 * r[(False, 1)]:.1f}, incongruent "
                   f"{100 * r[(False, 0)]:.1f} percent")
    nonzero = {}
    for model, g in m2c.groupby("model"):
        w = g.pivot_table(index=["dataset_id", "pair_id"], columns="congruent", values="supported")
        d = (w[1] - w[0]).groupby(level="pair_id").mean()
        nonzero[model] = set(d[d != 0].index)
    out.append(f"Step 2, nonzero matched pairs common to both models: "
               f"{len(nonzero['claude'] & nonzero['gpt'])} of {len(nonzero['claude'])}")

    # Step 3: verdicts in the favored direction on confirmed curved datasets
    f3 = log[(log.step == 3) & log.cell.isin(["inverted_u", "u_shape"])
             & (log.reversal_confirmed == 1) & (log.favored == 1)]
    for model, g in f3.groupby("model"):
        v = g.conclusion.value_counts(normalize=True)
        out.append(f"Step 3, {model}, favored direction, confirmed datasets, verdicts: " + "; ".join(
            f"{k} {100 * v.get(k, 0):.1f} percent" for k in ("supported", "inconclusive", "not_supported")))

    # Step 3: what Claude's endorsing replies say (keyword screens over the final message)
    import re
    rule = re.compile(r"stated (decision )?rule|decision rule|your (decision )?rule|by the (stated|given) "
                      r"(rule|definition)|under the (stated|given) (rule|definition|criteri)|"
                      r"(stated|given|prompt'?s?) definition", re.I)
    thin = re.compile(r"\bthin(ly)?\b|\bsparse(ly)?\b|\bfew (ventures|observations|firms|teams)\b|"
                      r"\bsmall number of\b|\btoo few\b", re.I)
    runs_dir = Path(os.environ.get("RUNS_DIR", ROOT / "runs"))
    e = log[(log.model == "claude") & (log.step == 3) & log.cell.isin(["inverted_u", "u_shape"])
            & (log.reversal_confirmed == 1) & (log.favored == 1) & (log.supported == 1)]
    n_rule = n_thin = 0
    for rid in e.run_id:
        t = json.loads((runs_dir / "claude" / f"{rid}.json").read_text())["final_message"]
        n_rule += bool(rule.search(t))
        n_thin += bool(thin.search(t))
    out.append(f"Step 3, Claude, endorsing runs on confirmed curved datasets: {len(e)}; replies "
               f"invoking the prompt's rule or definition: {n_rule}; replies calling the data at "
               f"high experience thin, sparse, or few: {n_thin}")

    # Records: matching rules, the GPT pause, removal notes, interrupted sessions
    runs = Path(os.environ.get("RUNS_DIR", ROOT / "runs"))
    decided, pre_pause, removed = 0, 0, 0
    for p in sorted(runs.glob("*/*.json")):
        if p.parent.name in ("interrupted", "check"):
            continue
        text = p.read_text()
        removed += text.count(REMOVED)
        rec = json.loads(text)
        if rec["model"] == "gpt" and "allow_login_shell=false" not in rec.get("cmd", []):
            pre_pause += 1
        if rec["step"] == 1:
            obj = score.last_json(rec.get("final_message", ""))
            coef = score.number(obj.get("reported_coefficient")) if obj else None
            fits = score.model_fits(rec.get("trace", []))
            matches = [f for f in fits if coef is not None
                       and score.close(f["experience_coef_raw_units"], coef, 0.02)]
            if matches:
                closest = min(matches, key=lambda f: abs(f["experience_coef_raw_units"] - coef))
                prescribed = (closest["estimator"] == "NegativeBinomial"
                              and {l for l in closest["labels"] if l != "const"}
                              == score.CONTROLS | {"experience"})
                decided += not prescribed
    out.append(f"Step 1 runs in which the closest matching fit is not the prescribed regression: {decided}")
    out.append(f"GPT records made before the shell setting was changed: {pre_pause}")
    out.append(f"Command outputs replaced by a removal note: {removed}")
    for p in sorted((runs / "interrupted").glob("*/*.json")):
        rec = json.loads(p.read_text())
        n = sum(1 for f in rec.get("trace", []) if not f.get("nested") and f.get("aligned"))
        out.append(f"Interrupted session {rec['run_id']} ({rec['model']}): {n} recorded regressions; "
                   f"reply: {rec.get('final_message', '')[:40]!r}")

    (ROOT / "results" / "extras.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
