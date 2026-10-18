"""Record every statsmodels regression a model runs during a session.

Installed as `sitecustomize` in the Python environment the models use,
so it loads in every Python process they start. It is active only when
the environment variable FIT_TRACE names a log file. For each fitted
model it appends one JSON line with the estimator, the outcome, and
the variables in the regression, identified by comparing each column
with the columns of data.csv (so the record does not depend on the
names the model chose). Nothing the model sees or prints changes.

For each regressor column the label is one of:
  <var>       an affine copy of a data column (raw or standardized)
  f(<var>)    a nonlinear function of one data column (log, square, dummies)
  const       the intercept
  other       anything else (for example an interaction)

Two limits. An offset or exposure is not part of the design matrix and
is not recorded (src/score.py detects a log(firm_age) offset by
refitting). For negative binomial fits estimated with an optimizer that
works on log(alpha), the last stored parameter is log(alpha) rather
than alpha, because the record is written before statsmodels
transforms it; the experience coefficient is unaffected.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import json
import os
import sys
import time

_DATA = None


def _data():
    global _DATA
    if _DATA is None:
        import pandas as pd
        _DATA = pd.read_csv(os.environ.get("FIT_TRACE_DATA", "data.csv"))
    return _DATA


def _label(col, data):
    import numpy as np
    col = np.asarray(col, dtype=float)
    if np.nanstd(col) == 0:
        return "const"
    for v in data.columns:
        x = data[v].to_numpy(dtype=float)
        if np.std(x) == 0:
            continue
        if abs(np.corrcoef(col, x)[0, 1]) > 0.9999:
            return v
    for v in data.columns:
        x = data[v].to_numpy(dtype=float)
        uniq = np.unique(x)
        if len(uniq) <= 30:
            # A function of a discrete column is constant within its values.
            if all(np.ptp(col[x == u]) == 0 for u in uniq):
                return f"f({v})"
        else:
            order = np.argsort(x)
            d = np.diff(col[order])
            if (d >= 0).all() or (d <= 0).all():
                return f"f({v})"
    return "other"


def _nested() -> bool:
    """True when this fit runs inside another statsmodels fit (for
    example the Poisson start values of a negative binomial fit)."""
    owners = set()
    f = sys._getframe(2)
    while f is not None:
        if f.f_code.co_name == "fit" and "statsmodels" in f.f_code.co_filename:
            owner = f.f_locals.get("self")
            if owner is not None:
                owners.add(id(owner))
        f = f.f_back
    return len(owners) > 1


def _record(results, model, params):
    import numpy as np
    path = os.environ.get("FIT_TRACE")
    if not path:
        return
    data = _data()
    exog = getattr(model, "exog", None)
    names = list(getattr(model, "exog_names", None) or [])
    rec = {"time": time.time(), "python": sys.executable,
           "estimator": type(model).__name__,
           "family": type(getattr(model, "family", None)).__name__,
           "nested": _nested(), "exog_names": names,
           "params": [float(p) for p in np.ravel(params)][:50]}
    endog = getattr(model, "endog", None)
    aligned = exog is not None and endog is not None and len(endog) == len(data)
    rec["aligned"] = bool(aligned)
    if aligned:
        rec["endog"] = _label(np.ravel(endog), data)
        exog = np.asarray(exog, dtype=float).reshape(len(data), -1)
        rec["labels"] = [_label(exog[:, j], data) for j in range(exog.shape[1])]
        if "experience" in rec["labels"]:
            j = rec["labels"].index("experience")
            x = data["experience"].to_numpy(dtype=float)
            # Coefficient per one prior venture, undoing any rescaling.
            scale = np.polyfit(x, exog[:, j], 1)[0]
            rec["experience_coef_raw_units"] = float(np.ravel(params)[j] * scale)
    with open(path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def _patch(module):
    original = module.Results.__init__

    def init(self, model, params, *args, **kwargs):
        original(self, model, params, *args, **kwargs)
        try:
            _record(self, model, params)
        except Exception:  # tracing must never disturb the model's work
            pass

    module.Results.__init__ = init


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != "statsmodels.base.model":
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        run = spec.loader.exec_module

        def exec_module(module):
            run(module)
            _patch(module)

        spec.loader.exec_module = exec_module
        return spec


if os.environ.get("FIT_TRACE"):
    sys.meta_path.insert(0, _Finder())
