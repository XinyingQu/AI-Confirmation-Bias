"""Simulate the datasets for all three steps.

Each dataset has 1,000 synthetic ventures. The five controls are drawn
first, then experience (a count that rises with founding-team size), then
acquisitions from a negative binomial (NB2) distribution whose log mean
depends on experience and the controls. Only the planted relationship
between experience and acquisitions changes across steps and cells.

Outputs (paths relative to the repository root):
  data/<dataset_id>.csv   the file a model sees (no planted effect in it)
  data/manifest.sha256    SHA-256 of every dataset
  key/key.csv             one row per dataset with the planted effect,
                          kept apart from the files the models see

Usage: python src/simulate.py
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
from scipy.stats import nbinom

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
KEY = ROOT / "key"

BASE_SEED = 20261012
N_VENTURES = 1000
ALPHA = 0.5  # NB2 dispersion: variance = mu + ALPHA * mu^2

# Log-mean coefficients on the controls (centered so the intercept sets
# the typical count). Team size enters with a step-specific coefficient.
B0 = 0.0
B_LOG_ASSETS = 0.25
B_FIRM_AGE = 0.06
B_INDUSTRY_MA = 0.01
B_CASH_RATIO = -1.0
B_TEAM_BASE = 0.10        # Steps 1 and 3
B_TEAM_CONFOUND = 0.35    # Step 2, with sign +/- by cell

B_EXPERIENCE = 0.15       # Step 1 positive/negative and Step 3 monotone cells
CURVE_PEAK = 4.0          # Step 3 turning point (in prior ventures)
CURVE_B2 = 0.05           # Step 3 curvature magnitude

# (step, cell, number of datasets). Step 2 cells come in matched pairs.
DESIGN = [
    (1, "none", 200), (1, "positive", 200), (1, "negative", 200),
    (2, "apparent_positive", 200), (2, "apparent_negative", 200),
    (3, "inverted_u", 200), (3, "u_shape", 200),
    (3, "rising", 50), (3, "falling", 50),
]
CELL_CODE = {cell: k for k, (_, cell, _) in enumerate(DESIGN)}


def draw_dataset(rng: np.random.Generator, focal, b_team: float) -> dict:
    """Draw one dataset. `focal` maps experience to its log-mean term."""
    n = N_VENTURES
    team_size = 1 + np.minimum(rng.poisson(1.3, n), 5)
    log_assets = rng.normal(1.5, 1.0, n)
    firm_age = np.round(2 + rng.gamma(4.0, 1.5, n), 1)
    industry_ma = rng.negative_binomial(5, 5 / (5 + 40), n)
    cash_ratio = rng.beta(2.0, 6.0, n)
    # Each founder's prior ventures: Poisson, at most three per founder.
    lam = 0.7 * np.exp(0.15 * (log_assets - 1.5))
    per_founder = np.minimum(rng.poisson(lam[:, None], (n, 6)), 3)
    experience = (per_founder * (np.arange(6) < team_size[:, None])).sum(axis=1)
    u = rng.uniform(size=n)  # shared across matched Step 2 pairs
    eta = (B0 + focal(experience)
           + B_LOG_ASSETS * (log_assets - 1.5)
           + B_FIRM_AGE * (firm_age - 8.0)
           + b_team * (team_size - 2.3)
           + B_INDUSTRY_MA * (industry_ma - 40)
           + B_CASH_RATIO * (cash_ratio - 0.25))
    mu = np.exp(eta)
    acquisitions = nbinom.ppf(u, 1 / ALPHA, 1 / (1 + ALPHA * mu)).astype(int)
    return {
        "venture_id": np.arange(1, n + 1),
        "experience": experience,
        "acquisitions": acquisitions,
        "log_assets": np.round(log_assets, 4),
        "firm_age": firm_age,
        "team_size": team_size,
        "industry_ma": industry_ma,
        "cash_ratio": np.round(cash_ratio, 4),
    }


def cell_parameters(cell: str):
    """Planted focal term and team-size coefficient for a cell."""
    b, c = CURVE_B2, CURVE_PEAK
    return {
        "none": (lambda x: 0.0 * x, B_TEAM_BASE),
        "positive": (lambda x: B_EXPERIENCE * x, B_TEAM_BASE),
        "negative": (lambda x: -B_EXPERIENCE * x, B_TEAM_BASE),
        "apparent_positive": (lambda x: 0.0 * x, B_TEAM_CONFOUND),
        "apparent_negative": (lambda x: 0.0 * x, -B_TEAM_CONFOUND),
        "inverted_u": (lambda x: 2 * b * c * x - b * x ** 2, B_TEAM_BASE),
        "u_shape": (lambda x: -2 * b * c * x + b * x ** 2, B_TEAM_BASE),
        "rising": (lambda x: B_EXPERIENCE * x, B_TEAM_BASE),
        "falling": (lambda x: -B_EXPERIENCE * x, B_TEAM_BASE),
    }[cell]


def dataset_seed(step: int, cell: str, i: int) -> list[int]:
    # Matched Step 2 pairs share their seed, so the two members differ
    # only in the sign of the team-size effect.
    code = CELL_CODE["apparent_positive"] if step == 2 else CELL_CODE[cell]
    return [BASE_SEED, step, code, i]


def make(step: int, cell: str, i: int) -> dict:
    focal, b_team = cell_parameters(cell)
    rng = np.random.default_rng(dataset_seed(step, cell, i))
    return draw_dataset(rng, focal, b_team)


def dataset_id(step: int, cell: str, i: int) -> str:
    return f"s{step}_{cell}_{i:03d}"


def write_csv(path: Path, cols: dict) -> None:
    names = list(cols)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(names)
        for row in zip(*(cols[k] for k in names)):
            w.writerow(row)


def main() -> None:
    DATA.mkdir(exist_ok=True)
    key_rows = []
    for step, cell, count in DESIGN:
        _, b_team = cell_parameters(cell)
        for i in range(count):
            did = dataset_id(step, cell, i)
            write_csv(DATA / f"{did}.csv", make(step, cell, i))
            key_rows.append({
                "dataset_id": did, "step": step, "cell": cell, "index": i,
                "seed": "-".join(map(str, dataset_seed(step, cell, i))),
                "pair_id": f"s2_pair_{i:03d}" if step == 2 else did,
                "b_team_size": b_team,
                "b_experience": {"positive": B_EXPERIENCE, "rising": B_EXPERIENCE,
                                 "negative": -B_EXPERIENCE,
                                 "falling": -B_EXPERIENCE}.get(cell, 0.0),
                "curve": {"inverted_u": "peak", "u_shape": "trough"}.get(cell, ""),
                "turning_point": CURVE_PEAK if cell in ("inverted_u", "u_shape") else "",
            })
    KEY.mkdir(exist_ok=True)
    with (KEY / "key.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(key_rows[0]))
        w.writeheader()
        w.writerows(key_rows)
    # SHA-256 of every dataset, so that a regenerated copy can be compared
    # with the files the models saw.
    lines = []
    for r in key_rows:
        name = f"{r['dataset_id']}.csv"
        lines.append(f"{hashlib.sha256((DATA / name).read_bytes()).hexdigest()}  {name}")
    (DATA / "manifest.sha256").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(key_rows)} datasets to {DATA}")


if __name__ == "__main__":
    main()
