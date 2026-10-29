"""
audit_framework.py
------------------
AKL Pipeline — Generic Audit Framework (Step 8)

Provides reusable, generic audit capabilities for validating classification
outputs produced by the pipeline:

- export_audit_sample(): sample N classified items with their predicted labels
  into data/03_review/step8_audit_<name>.xlsx for specialist review.
- compute_audit_metrics(): compare reviewed gold labels against predictions and
  produce accuracy, precision, recall, F1, and a confusion matrix.
- bootstrap_ci(): percentile bootstrap confidence intervals for any numeric series.

The framework reads its configuration from the `audits:` section of
pipeline_config.yaml and discovers active semantic vectors from Section 7 of
project_config.md (Step 7 vectors).

Usage (via run_worker.sh):
    ./run_worker.sh workers/audit_framework.py export    # generate review samples
    ./run_worker.sh workers/audit_framework.py compute   # compute metrics from reviewed files
"""

import sys
import json
import logging
import re
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import yaml
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


def _setup_logger(workspace_path):
    logger = logging.getLogger("audit_framework")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [AUDIT] %(message)s", datefmt="%H:%M:%S")
        fh = logging.FileHandler(Path(workspace_path) / "execution.log", encoding="utf-8")
        fh.setFormatter(fmt)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


class AuditFramework:
    """Generic audit framework for the AKL pipeline.

    Configuration lives in the `audits:` section of pipeline_config.yaml.
    Each audit entry supports:

        enabled: bool          # run this audit?
        sample_size: int       # number of items to sample for review
        source: str            # "step7_extractions", "step5_corpus" or "step5_full_predictions"
        filter_vector: str     # vector name from Section 7 ("" = all vectors)
        filter_range: dict     # optional numeric range filter, e.g.
                               #   {column: proba_yes, min: 0.40, max: 0.60}
                               # (used by the Step-5 decision-boundary audit)
        predicted_col: str     # column holding the predicted label
        gold_col: str          # column the specialist fills during review
        columns: list          # optional columns to export (defaults to the
                               # Step-7 extraction columns)
        description: str       # human-readable description
    """

    def __init__(self, config_path="pipeline_config.yaml", state_path="pipeline_state.json"):
        self.config_path = Path(config_path)
        self.state_path = Path(state_path)

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        with open(self.state_path, "r", encoding="utf-8") as f:
            self.state = json.load(f)

        self.workspace = Path(self.state.get("project", {}).get("workspace_path", "."))
        self.logger = _setup_logger(self.workspace)

        self.audit_cfg = self.config.get("audits", {})
        self.review_dir = self.workspace / "data" / "03_review"
        self.report_dir = self.workspace / "data" / "04_reports"

    # ── Discovery helpers ─────────────────────────────────────────────────────

    def discover_active_vectors(self):
        """Parse Section 7 of project_config.md and return active vector names."""
        md_path = self.workspace / "project_config.md"
        if not md_path.exists():
            return []
        content = md_path.read_text(encoding="utf-8")
        match = re.search(
            r"## 7\. Semantic extraction vectors.*?(\| Vector \|.*?\n(?:\|.*?\|.*?\n)+)",
            content, re.DOTALL | re.IGNORECASE,
        )
        vectors = []
        if match:
            lines = match.group(1).strip().split("\n")
            for line in lines[2:]:
                parts = [p.strip() for p in line.split("|")[1:-1]]
                if len(parts) >= 4 and parts[3].lower() == "yes":
                    vectors.append(parts[0])
        return vectors

    def _load_source(self, source):
        """Load a configured data source into a DataFrame."""
        if source == "step7_extractions":
            path = self.workspace / "data" / "02_intermediate" / "step7_extractions.pkl"
        elif source == "step5_corpus":
            path = self.workspace / "data" / "02_intermediate" / "step5_corpus_filtered_relevant.pkl"
        elif source == "step5_full_predictions":
            path = self.workspace / "data" / "02_intermediate" / "step5_full_predictions.pkl"
        else:
            raise ValueError(f"Unknown audit source: {source}")

        if not path.exists():
            raise FileNotFoundError(f"Audit source not found: {path}")
        return pd.read_pickle(path)

    # ── Sample export (Step 8 Phase 1) ────────────────────────────────────────

    def export_audit_sample(self, df, n, columns, seed=42):
        """Sample n rows from df, keeping only `columns` plus an empty gold column.

        Returns a DataFrame ready for specialist review. The gold column is
        named `audit_label` and starts empty.
        """
        if df.empty:
            return df
        sample = df.sample(n=min(n, len(df)), random_state=seed).copy()
        keep = [c for c in columns if c in sample.columns]
        sample = sample[keep].reset_index(drop=True)
        sample["audit_label"] = ""
        return sample

    def export_all(self):
        """Export review samples for every enabled audit.

        Returns a summary dict {audit_name: {"file": ..., "rows": ...}}.
        """
        self.review_dir.mkdir(parents=True, exist_ok=True)
        active_vectors = self.discover_active_vectors()
        results = {}

        for name, cfg in self.audit_cfg.items():
            if not cfg.get("enabled", False):
                self.logger.info(f"Audit '{name}' disabled — skipping.")
                continue

            try:
                df = self._load_source(cfg.get("source", "step7_extractions"))
                filter_vector = cfg.get("filter_vector", "")
                if filter_vector:
                    if "vector" not in df.columns:
                        self.logger.warning(f"Audit '{name}': source has no 'vector' column — skipping filter.")
                    else:
                        df = df[df["vector"] == filter_vector]
                        if filter_vector not in active_vectors:
                            self.logger.warning(
                                f"Audit '{name}': vector '{filter_vector}' is not active in "
                                f"project_config.md Section 7 (active: {active_vectors})."
                            )

                predicted_col = cfg.get("predicted_col", "micro_label")
                if predicted_col not in df.columns:
                    self.logger.warning(f"Audit '{name}': predicted column '{predicted_col}' not found — skipping.")
                    continue

                # Optional numeric range filter (e.g. decision-boundary audit)
                filter_range = cfg.get("filter_range")
                if filter_range:
                    col = filter_range.get("column")
                    lo = filter_range.get("min")
                    hi = filter_range.get("max")
                    if not col or col not in df.columns:
                        self.logger.warning(
                            f"Audit '{name}': filter_range column '{col}' not found in source — skipping."
                        )
                        continue
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    if lo is not None:
                        df = df[df[col] >= lo]
                    if hi is not None:
                        df = df[df[col] <= hi]
                    if df.empty:
                        self.logger.warning(
                            f"Audit '{name}': no rows in range {col} [{lo}, {hi}] — skipping."
                        )
                        continue

                sample_size = int(cfg.get("sample_size", 100))
                columns = cfg.get("columns") or ["doi", "vector", "scope", "macro_category", "micro_label", "original_text"]
                sample = self.export_audit_sample(df, sample_size, columns)

                out_path = self.review_dir / f"step8_audit_{name}.xlsx"
                sample.to_excel(out_path, index=False, engine="openpyxl")
                self.logger.info(f"Audit '{name}': exported {len(sample)} rows to {out_path}")
                results[name] = {"file": str(out_path), "rows": int(len(sample))}

            except FileNotFoundError as e:
                # Source not generated yet (e.g. boundary audit before a Step 5
                # rerun) — skip gracefully instead of failing the whole export.
                self.logger.warning(f"Audit '{name}' skipped: {e}")
                results[name] = {"skipped": str(e)}
            except Exception as e:
                self.logger.error(f"Audit '{name}' export failed: {e}", exc_info=True)
                results[name] = {"error": str(e)}

        return results

    # ── Metrics computation (Step 8 Phase 2) ─────────────────────────────────

    def compute_audit_metrics(self, reviewed_df, pred_col, gold_col="audit_label"):
        """Compute accuracy, precision, recall, F1, and confusion matrix.

        Rows where the gold label is empty/NaN are treated as "pending review"
        and excluded from the metrics.
        """
        if reviewed_df.empty:
            return None
        if pred_col not in reviewed_df.columns or gold_col not in reviewed_df.columns:
            raise ValueError(f"Missing columns: need '{pred_col}' and '{gold_col}'.")

        df = reviewed_df.copy()
        df[gold_col] = df[gold_col].astype(str).str.strip()
        df = df[df[gold_col] != ""]
        df = df[df[gold_col].str.lower() != "nan"]
        if df.empty:
            return {"reviewed": 0, "pending": int(len(reviewed_df)), "note": "no gold labels filled yet"}

        y_true = df[gold_col].astype(str).str.strip()
        y_pred = df[pred_col].astype(str).str.strip()

        # Normalize case for comparison (labels are case-insensitive)
        y_true_n = y_true.str.lower()
        y_pred_n = y_pred.str.lower()

        n_classes = y_true_n.nunique()
        if n_classes == 2:
            average = "binary"
            # Binary case: sklearn defaults to pos_label=1, but our labels are
            # strings (e.g., 'yes'/'no'). Prefer 'yes' as the positive class
            # (semantic positive); fall back to the minority class in y_true.
            classes = sorted(set(y_true_n) | set(y_pred_n))
            if "yes" in classes:
                pos_label = "yes"
            else:
                pos_label = y_true_n.value_counts().idxmin()
        else:
            average = "weighted"
            pos_label = None

        metrics = {
            "reviewed": int(len(df)),
            "pending": int(len(reviewed_df) - len(df)),
            "accuracy": round(float(accuracy_score(y_true_n, y_pred_n)), 4),
            "precision": round(float(precision_score(y_true_n, y_pred_n, average=average, pos_label=pos_label, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true_n, y_pred_n, average=average, pos_label=pos_label, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true_n, y_pred_n, average=average, pos_label=pos_label, zero_division=0)), 4),
            "average": average,
            "n_classes": int(n_classes),
        }

        # Confusion matrix (only when the number of classes is manageable)
        labels = sorted(set(y_true_n) | set(y_pred_n))
        if len(labels) <= 20:
            cm = confusion_matrix(y_true_n, y_pred_n, labels=labels)
            metrics["confusion_matrix"] = {
                "labels": labels,
                "matrix": cm.tolist(),
            }

        return metrics

    def bootstrap_ci(self, series, n_resamples=5000, ci=0.95, seed=42):
        """Percentile bootstrap confidence interval for the mean of a numeric series.

        Returns (lower, upper, mean). Falls back to (mean, mean, mean) when the
        series has fewer than 2 valid values.
        """
        arr = pd.Series(series).dropna().to_numpy(dtype=float)
        if len(arr) < 2:
            mean = float(arr[0]) if len(arr) == 1 else float("nan")
            return mean, mean, mean

        rng = np.random.default_rng(seed)
        means = np.array([
            rng.choice(arr, size=len(arr), replace=True).mean()
            for _ in range(n_resamples)
        ])
        alpha = (1.0 - ci) / 2.0
        lower, upper = np.percentile(means, [100 * alpha, 100 * (1 - alpha)])
        return float(lower), float(upper), float(arr.mean())

    def compute_all(self):
        """Compute metrics for every enabled audit from its reviewed file.

        Returns a dict {audit_name: metrics} and writes the aggregated report
        to data/04_reports/step8_audit_report.md.
        """
        self.report_dir.mkdir(parents=True, exist_ok=True)
        results = {}

        for name, cfg in self.audit_cfg.items():
            if not cfg.get("enabled", False):
                continue

            review_file = self.review_dir / f"step8_audit_{name}.xlsx"
            if not review_file.exists():
                self.logger.warning(f"Audit '{name}': review file not found ({review_file}). Run 'export' first.")
                results[name] = {"error": "review file not found — run export first"}
                continue

            try:
                df = pd.read_excel(review_file)
                pred_col = cfg.get("predicted_col", "micro_label")
                gold_col = cfg.get("gold_col", "audit_label")
                metrics = self.compute_audit_metrics(df, pred_col, gold_col)
                metrics["description"] = cfg.get("description", "")
                metrics["sample_size"] = int(cfg.get("sample_size", 100))
                results[name] = metrics
                self.logger.info(f"Audit '{name}': accuracy={metrics.get('accuracy')}, reviewed={metrics.get('reviewed')}")
            except Exception as e:
                self.logger.error(f"Audit '{name}' compute failed: {e}", exc_info=True)
                results[name] = {"error": str(e)}

        report_path = self.report_dir / "step8_audit_report.md"
        report_path.write_text(self._render_report(results), encoding="utf-8")
        self.logger.info(f"Audit report saved to {report_path}")
        return results

    # ── Report rendering ──────────────────────────────────────────────────────

    def _render_report(self, results):
        md = ["# Step 8: Audit Report\n"]
        md.append("> Generated by the AKL generic audit framework.\n")
        md.append("> Each audit samples classified items; the specialist reviews them at the Step 8 breakpoint; metrics are computed against the gold labels.\n")

        if not results:
            md.append("No audits configured or enabled.\n")
            return "\n".join(md)

        for name, res in results.items():
            md.append(f"## {name}\n")
            if "error" in res:
                md.append(f"⚠️ {res['error']}\n")
                continue

            md.append(f"**{res.get('description', '')}**\n")
            md.append(f"- Sample size: {res.get('sample_size', '?')}")
            md.append(f"- Reviewed: {res.get('reviewed', 0)}")
            md.append(f"- Pending review: {res.get('pending', 0)}")
            if "note" in res:
                md.append(f"- Note: {res['note']}\n")
                continue

            md.append(f"- Accuracy: **{res.get('accuracy', '—')}**")
            md.append(f"- Precision ({res.get('average', '?')}): {res.get('precision', '—')}")
            md.append(f"- Recall ({res.get('average', '?')}): {res.get('recall', '—')}")
            md.append(f"- F1 ({res.get('average', '?')}): {res.get('f1', '—')}")

            cm = res.get("confusion_matrix")
            if cm:
                labels = cm["labels"]
                matrix = cm["matrix"]
                md.append("\n### Confusion matrix (rows: gold label; columns: predicted label)\n")
                header = "| gold \\ predicted | " + " | ".join(labels) + " |"
                sep = "|---|" + "---|" * len(labels)
                md.append(header)
                md.append(sep)
                for i, row in enumerate(matrix):
                    md.append(f"| {labels[i]} | " + " | ".join(str(v) for v in row) + " |")
            md.append("")
        return "\n".join(md)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "export"
    framework = AuditFramework()
    if mode == "export":
        print(json.dumps(framework.export_all(), indent=2))
    elif mode == "compute":
        print(json.dumps(framework.compute_all(), indent=2))
    else:
        print(f"Unknown mode: {mode}. Use 'export' or 'compute'.")
        sys.exit(1)