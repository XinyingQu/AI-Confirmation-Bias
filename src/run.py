"""Send every (dataset, hypothesis direction) to each model and save one
record per run.

Each run happens in a fresh session in an empty working directory that
holds only data.csv. The prompt goes to the model on stdin. Both models
can run shell commands and Python with pandas, statsmodels, and scipy
(Claude may also read, write, and search files), with the same
reasoning effort and the same time limit. Runs go in
random order. A run is repeated only after a connection failure, at
most twice; a refusal, a failed analysis, or a timeout is recorded as
is. When a subscription usage limit stops the CLI before the model
starts, or cuts off a session before the model replies, the script
waits and starts the run again; neither counts as an attempt.

Records go to $RUNS_DIR/<model>/<run_id>.json (default ~/tools/runs).
The script skips runs that already have a record, so it can be stopped
and restarted at any time.

Usage:
  python src/run.py --model claude --workers 6
  python src/run.py --model gpt --workers 6
  python src/run.py --model claude --smoke   # 3 extra datasets per step
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import simulate  # noqa: E402

HOME = Path.home() / "tools"
# The models' Python environment and working folders sit apart from the
# study files and carry neutral names: a model sees its working folder's
# path and can see where Python lives, so neither may reveal the design.
VENV_BIN = Path.home() / "pyenvs" / "stats" / "bin"
# The fit recorder (src/fit_trace.py) is loaded from PYTHONPATH, so that it
# runs in any Python interpreter a model starts, not only ours.
SITE_DIR = Path.home() / "pyenvs" / "site"
SESSIONS = Path.home() / "sessions"
CODEX = HOME / "codex-cli" / "bin" / "codex"
TIME_LIMIT_S = 1200
EFFORT = "medium"
MODELS = {"claude": "claude-opus-5-5", "gpt": "gpt-6-astra"}
HYPOTHESES = json.loads((ROOT / "prompts" / "hypotheses.json").read_text())
LIMIT_RE = re.compile(r"(usage|session|weekly|daily|rate).?limit|limit reached|"
                      r"too many requests|(?<![\d.])429(?![\d.])|quota", re.I)


def build_prompt(step: int, direction: str) -> tuple[str, str]:
    template = "prescribed.txt" if step == 1 else "free.txt"
    key = direction + ("_range" if step == 3 else "")
    text = (ROOT / "prompts" / template).read_text()
    return template, text.replace("{HYPOTHESIS}", HYPOTHESES[key])


def run_list(smoke: bool) -> list[dict]:
    if smoke:
        picks = [(1, "none"), (2, "apparent_positive"), (3, "inverted_u")]
        rows = [{"dataset_id": f"smoke_{simulate.dataset_id(s, c, 900 + i)}",
                 "step": s, "cell": c, "index": 900 + i}
                for s, c in picks for i in range(3)]
        directions = ["H+"]
    else:
        key = pd.read_csv(ROOT / "key" / "key.csv", keep_default_na=False)
        rows = key[["dataset_id", "step", "cell", "index"]].to_dict("records")
        directions = ["H+", "H-"]
    return [{**r, "direction": d, "run_id": f"{r['dataset_id']}__{d}"}
            for r in rows for d in directions]


def call_claude(prompt: str, workdir: Path, env: dict) -> dict:
    cmd = ["claude", "-p", "--model", MODELS["claude"], "--effort", EFFORT,
           "--setting-sources", "", "--strict-mcp-config",
           "--no-session-persistence", "--output-format", "stream-json",
           "--verbose", "--allowedTools", "Bash", "Read", "Write", "Glob", "Grep"]
    cp = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                        cwd=workdir, env=env, timeout=TIME_LIMIT_S)
    events = [json.loads(l) for l in cp.stdout.splitlines() if l.startswith("{")]
    result = next((e for e in reversed(events) if e.get("type") == "result"), {})
    return {"rc": cp.returncode, "events": events,
            "final_message": result.get("result", ""),
            "model_reported": sorted(result.get("modelUsage", {}) or {}) or None,
            "usage": result.get("usage"), "cost_usd_list": result.get("total_cost_usd"),
            "stderr": cp.stderr[-2000:], "cmd": cmd}


def call_gpt(prompt: str, workdir: Path, env: dict) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tf:
        last = Path(tf.name)
    trace_dir = str(Path(env["FIT_TRACE"]).parent)
    cmd = [str(CODEX), "exec", "--ignore-user-config", "-m", MODELS["gpt"],
           "-c", f'model_reasoning_effort="{EFFORT}"',
           "-c", f'sandbox_workspace_write.writable_roots=["{trace_dir}"]',
           # Without this, Codex runs commands in a login shell, which
           # reorders PATH so that python3 may not be the study's Python.
           "-c", "allow_login_shell=false",
           "--skip-git-repo-check", "--ephemeral", "-s", "workspace-write",
           "-C", str(workdir), "--json", "-o", str(last), "-"]
    try:
        cp = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                            cwd=workdir, env=env, timeout=TIME_LIMIT_S)
        final = last.read_text() if last.exists() else ""
    finally:
        last.unlink(missing_ok=True)
    events = [json.loads(l) for l in cp.stdout.splitlines() if l.startswith("{")]
    usage = next((e.get("usage") for e in reversed(events)
                  if e.get("type") == "turn.completed"), None)
    return {"rc": cp.returncode, "events": events, "final_message": final,
            "model_reported": [MODELS["gpt"]],  # Codex reports no model id: the configured one
            "usage": usage,
            "cost_usd_list": None, "stderr": cp.stderr[-2000:], "cmd": cmd}


def tokens_used(res: dict) -> int:
    u = res.get("usage") or {}
    return int(u.get("output_tokens") or 0) + int(u.get("input_tokens") or 0)


def one_run(model: str, spec: dict, out_dir: Path) -> None:
    out = out_dir / f"{spec['run_id']}.json"
    if out.exists():
        return
    template, prompt = build_prompt(spec["step"], spec["direction"])
    attempts, record = 0, None
    while attempts < 3:
        token = uuid.uuid4().hex[:12]
        workdir = SESSIONS / token
        trace = SESSIONS / f".{token}-log" / "fits.jsonl"
        workdir.mkdir(parents=True)
        trace.parent.mkdir()
        if spec["dataset_id"].startswith("smoke_"):
            df = pd.DataFrame(simulate.make(spec["step"], spec["cell"], spec["index"]))
            df.to_csv(workdir / "data.csv", index=False)
        else:
            shutil.copy(ROOT / "data" / f"{spec['dataset_id']}.csv", workdir / "data.csv")
        env = dict(os.environ, PATH=f"{VENV_BIN}{os.pathsep}{os.environ['PATH']}",
                   PYTHONPATH=str(SITE_DIR),
                   FIT_TRACE=str(trace), FIT_TRACE_DATA=str(workdir / "data.csv"),
                   MPLCONFIGDIR=str(workdir / ".mpl"))
        t0 = time.time()
        try:
            res = (call_claude if model == "claude" else call_gpt)(prompt, workdir, env)
            res["status"] = "completed" if res["rc"] == 0 and res["final_message"].strip() else "failed"
        except subprocess.TimeoutExpired:
            res = {"status": "timeout", "rc": None, "events": [], "final_message": "",
                   "usage": None}
        res["duration_s"] = round(time.time() - t0, 1)
        no_tokens = tokens_used(res) == 0
        text = res.get("final_message", "") + res.get("stderr", "")
        # A usage limit can also stop a session after the model has started;
        # the reply is then only the limit notice. Neither case is an attempt.
        cut_off = res["status"] == "failed" and LIMIT_RE.search(res.get("final_message", "")[:200])
        if (no_tokens and res["status"] != "timeout" and LIMIT_RE.search(text)) or cut_off:
            print(f"[usage limit] {model} {spec['run_id']}: waiting 5 minutes", flush=True)
            if cut_off:  # keep the cut-off session with the records; it is not scored
                kept = out.parent.parent / "interrupted" / model / f"{spec['run_id']}.{token}.json"
                kept.parent.mkdir(parents=True, exist_ok=True)
                kept.write_text(json.dumps({
                    **spec, "model": model, "model_requested": MODELS[model],
                    "session_folder": token, "prompt_template": template, "prompt": prompt,
                    "attempts": attempts + 1, "started": t0, "effort": EFFORT,
                    "time_limit_s": TIME_LIMIT_S, **res,
                    "trace": [json.loads(l) for l in trace.read_text().splitlines()]
                    if trace.exists() else []}))
            shutil.rmtree(workdir, ignore_errors=True)
            shutil.rmtree(trace.parent, ignore_errors=True)
            time.sleep(300)
            continue  # stopped by the vendor before a reply: not an attempt
        attempts += 1
        res["connection_failure"] = res["status"] == "failed" and no_tokens
        record = {**spec, "model": model, "model_requested": MODELS[model],
                  "session_folder": token,
                  "prompt_template": template, "prompt": prompt, "attempts": attempts,
                  "started": t0, "effort": EFFORT, "time_limit_s": TIME_LIMIT_S, **res,
                  "trace": [json.loads(l) for l in trace.read_text().splitlines()]
                  if trace.exists() else [],
                  "files_left": sorted(p.name for p in workdir.iterdir())}
        shutil.rmtree(workdir, ignore_errors=True)
        shutil.rmtree(trace.parent, ignore_errors=True)
        if not res["connection_failure"]:
            break
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(record))
    tmp.rename(out)
    print(f"{model} {spec['run_id']}: {record['status']} in {record['duration_s']}s", flush=True)


def safe_run(model: str, spec: dict, out_dir: Path) -> None:
    try:
        one_run(model, spec, out_dir)
    except Exception as exc:  # report and move on; the run stays to do
        print(f"ERROR {model} {spec['run_id']}: {exc!r}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    base = Path(os.environ.get("RUNS_DIR", HOME / ("runs_smoke" if a.smoke else "runs")))
    out_dir = base / a.model
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = run_list(a.smoke)
    random.Random(f"order-{a.model}").shuffle(runs)
    todo = [r for r in runs if not (out_dir / f"{r['run_id']}.json").exists()]
    print(f"{a.model}: {len(runs)} runs, {len(todo)} to do", flush=True)
    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(lambda r: safe_run(a.model, r, out_dir), todo))
    print(f"{a.model}: work list complete", flush=True)


if __name__ == "__main__":
    main()
