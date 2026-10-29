"""
step1_merge_worker.py
---------------------
AKL Pipeline — Step 1 (extension): Merge multiple source corpora.

Combines the canonical DataFrames produced by separate Step 1 acquisitions
(e.g., WOS + OpenAlex) into a single corpus_raw.pkl, deduplicating by DOI
(with normalized-title fallback for records without DOI) and preserving the
`raw_source` column for traceability.

Priority order = the order of --inputs (first file wins on duplicates).

Usage:
    ./run_worker.sh workers/step1_merge_worker.py \
        --inputs data/02_intermediate/step1_corpus_raw_wos.pkl \
                 data/02_intermediate/step1_corpus_raw_openalex.pkl \
        --config pipeline_config.yaml \
        --state pipeline_state.json
"""

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml


def _setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("step1_merge_worker")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [STEP1-MERGE] %(levelname)s %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


def _normalize_title(t) -> str:
    """Normalize a title for fuzzy deduplication (lowercase, alphanumeric only)."""
    if not t or pd.isna(t):
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(t).lower())


def main():
    parser = argparse.ArgumentParser(
        description="AKL Step 1 — Merge multiple source corpora (dedup by DOI)")
    parser.add_argument("--inputs", "-i", nargs="+", required=True,
                        help="Pickle files to merge, in priority order (first = highest priority)")
    parser.add_argument("--config", default="pipeline_config.yaml",
                        help="Path to pipeline_config.yaml")
    parser.add_argument("--state", default="pipeline_state.json",
                        help="Path to pipeline_state.json")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with open(args.state, encoding="utf-8") as f:
        state = json.load(f)

    workspace = state.get("project", {}).get("workspace_path", ".")
    logger = _setup_logger(str(Path(workspace) / "execution.log"))

    # ── Load all source frames ──────────────────────────────────────────────
    frames = []
    for path in args.inputs:
        p = Path(path)
        if not p.exists():
            logger.error(f"Input file not found: {path}")
            sys.exit(1)
        df = pd.read_pickle(p)
        frames.append(df)
        logger.info(f"Loaded {len(df)} records from {path}")

    # Derive the merged source label from the raw_source values actually present
    source_labels = []
    for df in frames:
        if "raw_source" in df.columns:
            vals = [str(v) for v in df["raw_source"].dropna().unique() if str(v).strip()]
            source_labels.extend(vals)
    source_label = " + ".join(dict.fromkeys(source_labels)) if source_labels else "merged"
    logger.info(f"Merged source label: {source_label}")

    df_all = pd.concat(frames, ignore_index=True)
    total_raw = len(df_all)
    logger.info(f"Concatenated: {total_raw} records from {len(frames)} sources")

    # ── Deduplication ───────────────────────────────────────────────────────
    # 1) By DOI (case-insensitive, stripped). First occurrence wins → priority order.
    df_all["_doi_key"] = df_all["doi"].astype(str).str.strip().str.lower()
    df_all.loc[df_all["_doi_key"].isin(["", "nan", "none"]), "_doi_key"] = ""

    df_with_doi = df_all[df_all["_doi_key"] != ""].drop_duplicates(
        subset="_doi_key", keep="first")
    df_no_doi = df_all[df_all["_doi_key"] == ""].drop(columns=["_doi_key"])

    # 2) By normalized title for records without DOI
    df_no_doi["_title_key"] = df_no_doi["title"].apply(_normalize_title)
    df_no_doi = df_no_doi[df_no_doi["_title_key"] != ""].drop_duplicates(
        subset="_title_key", keep="first")
    df_no_doi = df_no_doi.drop(columns=["_title_key"])

    df_final = pd.concat([df_with_doi.drop(columns=["_doi_key"]), df_no_doi],
                         ignore_index=True)
    removed = total_raw - len(df_final)
    logger.info(f"Deduplication removed {removed} duplicate records "
                f"→ {len(df_final)} final records")

    # ── Save (same convention as step1_worker) ──────────────────────────────
    output_dir_pkl = Path(workspace) / "data" / "02_intermediate"
    output_dir_csv = Path(workspace) / "data" / "01_raw"
    output_dir_pkl.mkdir(parents=True, exist_ok=True)
    output_dir_csv.mkdir(parents=True, exist_ok=True)

    pkl_path = output_dir_pkl / "step1_corpus_raw.pkl"
    df_final.to_pickle(pkl_path)
    csv_path = output_dir_csv / "step1_corpus_raw.csv"
    df_final.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved {len(df_final)} records to {pkl_path}")

    # ── Update pipeline_state.json ──────────────────────────────────────────
    missing_abstract = int(
        df_final["abstract"].isna().sum() + (df_final["abstract"] == "").sum()
    ) if "abstract" in df_final.columns else 0

    now = datetime.now().isoformat(timespec="seconds")
    step = state.setdefault("steps", {}).setdefault("1", {})
    step["status"] = "completed"
    step["completed_at"] = now
    step["notes"] = ""  # clear any stale error note from a previous failed run
    step.setdefault("output", {})["raw_data_file"] = str(pkl_path)
    step["output"]["raw_data_file_csv"] = str(csv_path)
    step.setdefault("params_used", {}).update({
        "source_type": f"{source_label} (merged)",
        "records_raw": total_raw,
        "records_output": len(df_final),
        "records_missing_abstract": missing_abstract,
        "records_removed_duplicates": removed,
        "merge_priority": args.inputs,
    })
    state["project"]["last_updated"] = now
    with open(args.state, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

    print("\n── Step 1 Merge Result ─────────────────────────")
    print(f"  Records before dedup: {total_raw}")
    print(f"  Records after dedup:  {len(df_final)}")
    print(f"  Duplicates removed:   {removed}")
    print(f"  Missing abstracts:    {missing_abstract}")
    print(f"  Output file:          {pkl_path}")


if __name__ == "__main__":
    main()