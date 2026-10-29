"""
step1_if_fill_worker.py
-----------------------
AKL Pipeline — Step 1 (extension): Fill missing journal impact factors.

Strategy (in order of preference):
  1. OpenAlex Sources API → summary_stats.2yr_mean_citedness
     (same definition as the JCR Impact Factor: 2-year mean citedness)
  2. impact_factor library (with name normalization variants)

After filling, recomputes `jnif` (normalized IF across the corpus) and
updates the corpus pickle + inspection CSV.

Usage:
    ./run_worker.sh workers/step1_if_fill_worker.py \
        --input data/02_intermediate/step1_corpus_raw.pkl \
        --config pipeline_config.yaml \
        --state pipeline_state.json
"""

import argparse
import difflib
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import yaml

OPENALEX_MAILTO = "akl-pipeline@example.com"


def _setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("step1_if_fill_worker")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [STEP1-IF] %(levelname)s %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


def _norm(name) -> str:
    """Normalize a journal name for comparison (lowercase, alphanumeric only)."""
    if not name or pd.isna(name):
        return ""
    s = re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()
    s = re.sub(r"\b(the|and|of|for|in|on|at|&)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _openalex_if(journal_name: str):
    """Look up 2yr_mean_citedness via the OpenAlex Sources API.

    Accepts an exact normalized-name match first, then a fuzzy match
    (SequenceMatcher ratio >= 0.9) to tolerate typos in source data.
    """
    try:
        r = requests.get(
            "https://api.openalex.org/sources",
            params={"search": journal_name, "per-page": 5,
                    "mailto": OPENALEX_MAILTO},
            timeout=30,
        ).json()
    except Exception:
        return None

    target = _norm(journal_name)
    results = []
    for s in r.get("results", []):
        ss = s.get("summary_stats") or {}
        if ss.get("2yr_mean_citedness") is None:
            continue
        src_name = _norm(s.get("display_name", ""))
        if src_name == target:
            return ss["2yr_mean_citedness"]  # exact match wins immediately
        ratio = difflib.SequenceMatcher(None, src_name, target).ratio()
        results.append((ratio, ss["2yr_mean_citedness"]))

    if results:
        best_ratio, best_if = max(results, key=lambda x: x[0])
        if best_ratio >= 0.9:
            return best_if
    return None


def _library_if(journal_name: str):
    """Fallback: impact_factor library with name normalization variants."""
    try:
        from impact_factor.core import Factor
    except ImportError:
        return None
    try:
        factor = Factor()
    except Exception:
        return None

    variants = [
        journal_name,
        journal_name.upper(),
        journal_name.upper().replace("THE ", ""),
        journal_name.upper().replace(": ", " "),
    ]
    for v in variants:
        try:
            r = factor.search(v)
            if r:
                val = float(r[0].get("factor", 0)) or None
                if val:
                    return val
        except Exception:
            continue
    return None


def main():
    parser = argparse.ArgumentParser(
        description="AKL Step 1 — Fill missing journal impact factors")
    parser.add_argument("--input", "-i", default="data/02_intermediate/step1_corpus_raw.pkl",
                        help="Corpus pickle to enrich (default: step1_corpus_raw.pkl)")
    parser.add_argument("--config", default="pipeline_config.yaml")
    parser.add_argument("--state", default="pipeline_state.json")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with open(args.state, encoding="utf-8") as f:
        state = json.load(f)

    workspace = state.get("project", {}).get("workspace_path", ".")
    logger = _setup_logger(str(Path(workspace) / "execution.log"))

    df = pd.read_pickle(args.input)
    before = int(df["journal_if"].notna().sum())
    logger.info(f"Loaded {len(df)} records — journal_if filled: {before}")

    missing = df[df["journal_if"].isna()]["source_journal"].dropna()
    missing = missing[missing.str.strip() != ""]
    journals = missing.value_counts().index.tolist()
    logger.info(f"{len(journals)} distinct journals missing journal_if")

    fill_map = {}
    n_openalex = 0
    n_library = 0
    n_unresolved = 0

    for i, j in enumerate(journals, 1):
        if j in fill_map:
            continue
        val = _openalex_if(j)
        source = "openalex"
        if val is None:
            val = _library_if(j)
            source = "library"
        if val is not None:
            fill_map[j] = val
            if source == "openalex":
                n_openalex += 1
            else:
                n_library += 1
        else:
            n_unresolved += 1
        if i % 20 == 0 or i == len(journals):
            logger.info(f"  Progress {i}/{len(journals)} — filled: {len(fill_map)}")
        time.sleep(0.12)  # be polite to the OpenAlex API

    # Apply to the corpus
    df["journal_if"] = df.apply(
        lambda r: fill_map.get(r["source_journal"], r["journal_if"])
        if pd.isna(r["journal_if"]) else r["journal_if"],
        axis=1,
    )

    after = int(df["journal_if"].notna().sum())
    logger.info(f"journal_if filled: {before} → {after} "
                f"(+{after - before}; OpenAlex: {n_openalex}, library: {n_library}, "
                f"unresolved journals: {n_unresolved})")

    # Recompute jnif (normalized IF across the corpus)
    if_series = pd.to_numeric(df["journal_if"], errors="coerce")
    mean_if = if_series.mean()
    std_if = if_series.std()
    if std_if and std_if > 0:
        df["jnif"] = ((if_series - mean_if) / std_if).round(4)
    else:
        df["jnif"] = 0.0
    logger.info(f"jnif recomputed (mean={mean_if:.2f}, std={std_if:.2f})")

    # Save (same convention as step1_worker)
    output_dir_pkl = Path(workspace) / "data" / "02_intermediate"
    output_dir_csv = Path(workspace) / "data" / "01_raw"
    output_dir_pkl.mkdir(parents=True, exist_ok=True)
    output_dir_csv.mkdir(parents=True, exist_ok=True)

    pkl_path = output_dir_pkl / "step1_corpus_raw.pkl"
    df.to_pickle(pkl_path)
    csv_path = output_dir_csv / "step1_corpus_raw.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved enriched corpus to {pkl_path}")

    # Update state (note only — step 1 remains completed)
    now = datetime.now().isoformat(timespec="seconds")
    step = state.setdefault("steps", {}).setdefault("1", {})
    step.setdefault("params_used", {}).update({
        "journal_if_filled_before": before,
        "journal_if_filled_after": after,
        "journal_if_filled_via_openalex": n_openalex,
        "journal_if_filled_via_library": n_library,
        "journal_if_unresolved_journals": n_unresolved,
    })
    state["project"]["last_updated"] = now
    with open(args.state, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

    print("\n── Step 1 IF Fill Result ───────────────────────")
    print(f"  journal_if: {before} → {after}  (+{after - before})")
    print(f"  Coverage:   {after/len(df)*100:.1f}%")
    print(f"  Via OpenAlex: {n_openalex} journals | via library: {n_library} | unresolved: {n_unresolved}")
    print(f"  jnif recomputed (mean={mean_if:.2f}, std={std_if:.2f})")
    print(f"  Output:     {pkl_path}")


if __name__ == "__main__":
    main()