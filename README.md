# Testing the Tester

Code, data, and run records for a ground-truth audit of how two AI models (Claude Opus 5.5 and GPT-6 Astra) test a management hypothesis. The audit asks three questions, one per step:

1. Does the model report correctly when the analysis is prescribed?
2. Does the model choose controls opportunistically when it has analytic freedom?
3. Does the model report accurately when the true relationship is curved and the hypothesis is linear?

Every dataset is simulated, so the true relationship is known, and every dataset is analyzed under the hypothesis and under its mirror image (the same sentence with "higher" replaced by "lower"). A model whose choice of analysis, or whose verdict relative to the correct analysis, depends on the stated direction shows a pattern consistent with confirmation bias.

## Contents

| Path | What it holds |
|---|---|
| `specs/` | One page per step: hypothesis text, variables, data, prompt, scoring rule, runs |
| `prompts/` | The two prompt templates (prescribed and free analysis) and the hypothesis texts |
| `src/simulate.py` | Simulates all 1,500 datasets |
| `src/reference.py` | Our own reference regressions on every dataset |
| `src/check_data.py` | Checks that the simulated data behave as designed |
| `src/run.py` | Sends each dataset and hypothesis to each model and saves one record per run |
| `src/fit_trace.py` | Records every regression a model fits during its session |
| `src/score.py` | Scores every run and writes the run log |
| `src/tables.py` | Produces every results table from the run log |
| `src/extras.py` | Prints the figures quoted in the text that the tables do not contain |
| `data/` | The 1,500 datasets exactly as the models saw them |
| `key/key.csv` | The true effect in every dataset (never shown to a model): `b_experience` is the linear effect in the monotone cells (0 in the curved cells), `curve` and `turning_point` describe the curved cells (log mean 0.05 (8x - x^2) for the inverted U and its negative for the U), and `b_team_size` is the team-size coefficient |
| `results/` | Reference regressions, data checks, run log, and tables |
| `runs/` | The archived run records (redacted, see `DECISIONS.md` item 27): `claude/` and `gpt/` hold the 6,000 scored runs, `interrupted/` the five sessions a usage limit cut off, `check/` the last round of check runs |
| `DECISIONS.md` | Implementation choices with reasons, events during the runs, and the redaction of the shared records |

## Reproducing the results

Python 3.9 or later with the packages in `requirements.txt`:

```bash
python -m venv env && env/bin/pip install -r requirements.txt
env/bin/python src/simulate.py      # regenerates data/ and key/ exactly
env/bin/python src/reference.py     # results/reference.csv
env/bin/python src/check_data.py    # results/data_checks.txt
tar -xzf runs/run_records.tar.gz -C runs/
RUNS_DIR=runs env/bin/python src/score.py   # results/run_log.csv
env/bin/python src/tables.py        # results/tables.md
RUNS_DIR=runs env/bin/python src/extras.py   # results/extras.txt, figures quoted outside the tables
```

These steps need no access to any AI model; they rebuild every table from the archived records.

## Repeating the model runs

New runs will differ from the archived ones, because model outputs are not deterministic. They need the Claude Code and Codex command-line tools with active accounts and the following setup, which `src/run.py` expects under the home folder:

```bash
python -m venv ~/pyenvs/stats && ~/pyenvs/stats/bin/pip install -r requirements.txt
mkdir -p ~/pyenvs/site && cp src/fit_trace.py ~/pyenvs/site/sitecustomize.py   # the regression recorder
npm install -g --prefix ~/tools/codex-cli @openai/codex                          # Codex CLI
python src/run.py --model claude --smoke    # check runs on 9 extra datasets, then inspect them
python src/run.py --model claude --workers 6
python src/run.py --model gpt --workers 8
```

Records go to `~/tools/runs/<model>/` (set `RUNS_DIR` to change it), and each model works in a fresh folder under `~/sessions/`. The run is resumable: a record already on disk is not run again.

## How the runs were made

Each run is a fresh session in an empty working folder that holds only the dataset. Both models can run shell commands and Python with pandas, statsmodels, and scipy (see `DECISIONS.md` item 5 for the tools each CLI lists), the same reasoning effort (medium), and a 20-minute limit. Neither sees the true effect, the random seed, or any other run. Runs went in random order. A run was repeated only after a connection failure, at most twice; refusals, failed analyses, and timeouts are recorded and reported as they are.
