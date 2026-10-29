"""
reference_date.py
-----------------
AKL Pipeline — fixed reference date for time-dependent indicators.

Every quantity that depends on "today" (document age, citations per year,
the "recent years" window of the growth regimes, impact quadrants) is computed
relative to ONE reference date, so that re-running a step later does not change
its results.

Resolution order:
  1. `analysis.reference_date` in pipeline_config.yaml (ISO date, e.g. "2026-10-09");
     normally the date on which the bibliographic data were retrieved.
  2. The completion date of Step 1 recorded in pipeline_state.json.
  3. Today's date (with a warning: results then depend on the run date).
"""

import json
import warnings
from datetime import date, datetime
from pathlib import Path


def _parse(value):
    if not value:
        return None
    if isinstance(value, (date, datetime)):
        return value if isinstance(value, date) else value.date()
    try:
        return datetime.fromisoformat(str(value)[:10]).date()
    except ValueError:
        return None


def reference_date(workspace=".", config=None):
    """Return the reference date (datetime.date) used for time-dependent indicators."""
    workspace = Path(workspace)
    cfg = config
    if cfg is None:
        try:
            import yaml
            with open(workspace / "pipeline_config.yaml", "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}
    d = _parse((cfg.get("analysis", {}) or {}).get("reference_date"))
    if d:
        return d
    try:
        with open(workspace / "pipeline_state.json", "r", encoding="utf-8") as f:
            state = json.load(f)
        d = _parse(state.get("steps", {}).get("1", {}).get("completed_at"))
        if d:
            return d
    except Exception:
        pass
    warnings.warn(
        "analysis.reference_date is not set and Step 1 has no completion date; "
        "using today's date. Time-dependent indicators will change with the run date."
    )
    return date.today()


def reference_year(workspace=".", config=None):
    """Year of the reference date."""
    return reference_date(workspace, config).year
