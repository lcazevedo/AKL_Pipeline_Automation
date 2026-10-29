"""
step2_worker.py
---------------
AKL Pipeline — Step 2: Data Cleaning and Structuring

Responsibilities:
- Load corpus_raw.pkl produced by Step 1
- Validate mandatory fields (title, abstract, year, first/last author)
- Validate additional required fields from pipeline_config.yaml
- Infer additional required fields from project_config.md via LLM (optional)
- Remove records that fail mandatory validation
- Deduplicate by DOI
- Normalize abstract (remove near-empty strings)
- Save corpus_clean.pkl + corpus_clean.csv to data/inputs/
- Save quality_report.md to data/inputs/
- Update pipeline_state.json

Usage:
    from workers.step2_worker import Step2Worker
    worker = Step2Worker("pipeline_config.yaml", "pipeline_state.json")
    result = worker.run("data/inputs/corpus_raw.pkl")

CLI:
    ./run_worker.sh workers/step2_worker.py \
        --input data/inputs/corpus_raw.pkl \
        --config pipeline_config.yaml \
        --state pipeline_state.json \
        [--project-config project_config.md]
"""

import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

warnings.filterwarnings("ignore")


# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("step2_worker")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [STEP2] %(levelname)s %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Main worker class
# ─────────────────────────────────────────────────────────────────────────────

class Step2Worker:
    """
    Cleans and validates the raw corpus produced by Step 1.
    Produces a clean, deduplicated corpus ready for topic modeling.
    """

    # Mandatory fields — records missing any of these are removed
    MANDATORY_FIELDS = ["title", "abstract", "year", "first_author", "last_author"]

    # Minimum abstract length in characters to be considered valid
    MIN_ABSTRACT_LENGTH = 50

    def __init__(self, config_path: str, state_path: str,
                 project_config_path: Optional[str] = None):
        self.config_path = Path(config_path)
        self.state_path = Path(state_path)
        self.project_config_path = Path(project_config_path) if project_config_path else None
        self.config = self._load_yaml(config_path)
        self.state = self._load_json(state_path)
        self.schema = self.config.get("schema", {})

        workspace = self.state.get("project", {}).get("workspace_path", ".")
        self.logger = _setup_logger(str(Path(workspace) / "execution.log"))
        self.logger.info("Step2Worker initialized.")

        # Additional required fields from config (beyond the mandatory core)
        self.extra_required = self.config.get("step2_cleaning", {}).get(
            "required_fields", [])

        # Language handling (config-driven)
        # detect_language: fill/overwrite the 'language' column from title+abstract content
        # language_filter: remove records whose final language != this value (e.g. "English")
        self.detect_language = bool(self.config.get("step2_cleaning", {}).get(
            "detect_language", False))
        self.language_filter = str(self.config.get("step2_cleaning", {}).get(
            "language_filter", "") or "").strip()

        # Analysis window (config-driven) — restrict corpus to [year_min, year_max]
        self.year_min = self.config.get("step2_cleaning", {}).get("year_min")
        self.year_max = self.config.get("step2_cleaning", {}).get("year_max")

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, input_file: str) -> dict:
        """
        Full cleaning pipeline.
        Returns result dict with status, output files, and stats.
        """
        self.logger.info(f"Starting Step 2 — input: {input_file}")
        self._update_state_status("running", params={"input_file": input_file})

        try:
            df = self._load_input(input_file)
            n_raw = len(df)
            self.logger.info(f"Loaded {n_raw} records from {input_file}")

            # ── Cleaning steps ────────────────────────────────────────────────
            df, removed_mandatory = self._validate_mandatory_fields(df)
            df, removed_abstract = self._validate_abstract_quality(df)
            df, removed_year = self._apply_year_filter(df)
            df = self._detect_language(df)
            df, removed_language = self._apply_language_filter(df)
            df, removed_duplicates = self._deduplicate(df)
            df = self._normalize_types(df)

            # ── Additional required fields (from config + project_config) ─────
            extra_fields = self._resolve_extra_required_fields()
            df, removed_extra = self._validate_extra_fields(df, extra_fields)

            # ── Save outputs ──────────────────────────────────────────────────
            pkl_path, csv_path = self._save_output(df)
            report_path = self._save_quality_report(
                n_raw, removed_mandatory, removed_abstract,
                removed_year, removed_language, removed_duplicates, removed_extra,
                extra_fields, len(df)
            )

            stats = {
                "records_input": n_raw,
                "records_output": len(df),
                "removed_missing_mandatory": removed_mandatory,
                "removed_abstract_quality": removed_abstract,
                "removed_year_filter": removed_year,
                "year_window": f"{self.year_min or '…'}–{self.year_max or '…'}",
                "removed_language_filter": removed_language,
                "removed_duplicates": removed_duplicates,
                "removed_missing_extra": removed_extra,
                "extra_fields_checked": extra_fields,
                "language_filter": self.language_filter,
                "detect_language": self.detect_language,
            }

            self._update_state_status(
                "completed",
                output_pkl=pkl_path,
                output_csv=csv_path,
                quality_report=report_path,
                params=stats
            )

            self.logger.info(
                f"Step 2 complete. {n_raw} → {len(df)} records. "
                f"Report: {report_path}"
            )
            return {"status": "completed", "output_file": pkl_path,
                    "quality_report": report_path, **stats}

        except Exception as e:
            self.logger.error(f"Step 2 failed: {e}", exc_info=True)
            self._update_state_status("failed", notes=str(e))
            raise

    # ── Cleaning steps ────────────────────────────────────────────────────────

    def _load_input(self, input_file: str) -> pd.DataFrame:
        path = Path(input_file)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        if path.suffix == ".pkl":
            return pd.read_pickle(path)
        elif path.suffix == ".csv":
            self.logger.warning("Loading from CSV — list columns will be strings. "
                                "Prefer .pkl for correct types.")
            return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        else:
            raise ValueError(f"Unsupported input format: {path.suffix}. Use .pkl or .csv")

    def _validate_mandatory_fields(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Remove records missing any mandatory field."""
        n_before = len(df)
        mask_valid = pd.Series(True, index=df.index)

        for field in self.MANDATORY_FIELDS:
            if field not in df.columns:
                self.logger.warning(f"Mandatory field '{field}' not found in corpus — "
                                    f"all records will be flagged as missing it.")
                mask_valid &= False
                continue

            field_empty = (
                df[field].isna() |
                (df[field].astype(str).str.strip() == "") |
                (df[field].astype(str).str.strip().str.lower().isin(
                    ["none", "null", "nan", "n/a", "na"]))
            )
            n_missing = field_empty.sum()
            if n_missing > 0:
                self.logger.info(f"  '{field}': {n_missing} records missing")
            mask_valid &= ~field_empty

        df_clean = df[mask_valid].copy()
        removed = n_before - len(df_clean)
        self.logger.info(f"Mandatory validation: removed {removed} records.")
        return df_clean, removed

    def _validate_abstract_quality(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """
        Remove records whose abstract passed presence check but is too short
        to be useful (e.g. 'See full text', 'N/A', placeholder strings).
        """
        n_before = len(df)
        if "abstract" not in df.columns:
            return df, 0

        too_short = df["abstract"].astype(str).str.strip().str.len() < self.MIN_ABSTRACT_LENGTH
        n_removed = too_short.sum()
        if n_removed > 0:
            self.logger.info(f"Abstract quality: removed {n_removed} records "
                             f"with abstract shorter than {self.MIN_ABSTRACT_LENGTH} chars.")
        return df[~too_short].copy(), int(n_removed)

    def _apply_year_filter(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """
        Restrict the corpus to the configured analysis window [year_min, year_max]
        (inclusive). Records outside the window or with an invalid year are removed.
        """
        if self.year_min is None and self.year_max is None:
            return df, 0
        if "year" not in df.columns:
            self.logger.warning("year_min/year_max configured but 'year' column is "
                                "missing — skipping the year filter.")
            return df, 0

        n_before = len(df)
        years = pd.to_numeric(df["year"], errors="coerce")
        keep = years.notna()
        if self.year_min is not None:
            keep &= years >= self.year_min
        if self.year_max is not None:
            keep &= years <= self.year_max

        df = df[keep].copy()
        removed = n_before - len(df)
        window = f"{self.year_min or '…'}–{self.year_max or '…'}"
        self.logger.info(f"Year filter (window {window}): removed {removed} records.")
        return df, int(removed)

    def _detect_language(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect the language of each document from its title and abstract
        content (langdetect) and fill/overwrite the 'language' column.

        Divergence rule (specialist-defined):
        - If title and abstract agree on a language, use it.
        - If they diverge and one of them is English, use the NON-English one
          (conservative: any non-English content flags the document).
        - If they diverge and neither is English, use the abstract language
          (more text, more reliable).
        - If detection fails for a record, keep the original 'language' value.
        """
        if not self.detect_language:
            return df

        try:
            from langdetect import detect, DetectorFactory
            DetectorFactory.seed = 0  # deterministic results
        except ImportError:
            self.logger.warning("langdetect not installed — skipping language detection. "
                                "Install it with: pip install langdetect")
            return df

        def _detect(text) -> str:
            text = str(text).strip()
            if len(text) < 20:
                return ""
            try:
                return detect(text)
            except Exception:
                return ""

        original = df["language"].astype(str) if "language" in df.columns \
            else pd.Series("", index=df.index)

        detected_codes = []
        for _, row in df.iterrows():
            t = _detect(row.get("title", ""))
            a = _detect(row.get("abstract", ""))
            if t and a and t != a:
                if t == "en":
                    lang = a
                elif a == "en":
                    lang = t
                else:
                    lang = a  # both non-English → abstract (more text)
            else:
                lang = t or a
            detected_codes.append(lang)

        filled = 0
        names = []
        for code, orig in zip(detected_codes, original):
            if code:
                names.append(self._iso_lang_name(code))
                filled += 1
            else:
                # keep original value when detection failed
                names.append(orig if orig.lower() not in ("", "nan", "none") else "")
        df["language"] = names

        self.logger.info(f"Language detection: identified language for {filled} records "
                         f"(title+abstract, divergence → non-English).")
        return df

    @staticmethod
    def _iso_lang_name(code: str) -> str:
        """Convert an ISO 639-1 code (e.g. 'en', 'zh-cn') to a full name."""
        code = code.split("-")[0].lower()
        try:
            import pycountry
            lang = pycountry.languages.get(alpha_2=code)
            return lang.name if lang else code
        except Exception:
            return code

    def _apply_language_filter(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """
        Remove records whose final 'language' differs from the configured
        language_filter (e.g. "English"). Case-insensitive comparison.
        """
        if not self.language_filter:
            return df, 0
        if "language" not in df.columns:
            self.logger.warning("language_filter is set but 'language' column is missing — "
                                "skipping the filter.")
            return df, 0

        n_before = len(df)
        lang_col = df["language"].astype(str).str.strip()
        keep = lang_col.str.lower() == self.language_filter.lower()
        df = df[keep].copy()
        removed = n_before - len(df)
        self.logger.info(f"Language filter ('{self.language_filter}'): "
                         f"removed {removed} records.")
        return df, int(removed)

    def _deduplicate(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """
        Deduplicate by DOI (when available) then by title similarity.
        DOI dedup is exact. Title dedup normalizes case and whitespace.
        """
        n_before = len(df)

        # DOI dedup — only on non-empty DOIs
        if "doi" in df.columns:
            doi_valid = df["doi"].astype(str).str.strip()
            doi_valid = doi_valid[~doi_valid.isin(["", "none", "null", "nan", "n/a"])]
            duplicated_doi = df.index.isin(
                df[df["doi"].isin(doi_valid[doi_valid.duplicated()])].index
            )
            # Keep first occurrence of each DOI
            df = df[~df["doi"].duplicated(keep="first")].copy()
            n_doi_dupes = n_before - len(df)
            if n_doi_dupes > 0:
                self.logger.info(f"Deduplication by DOI: removed {n_doi_dupes} duplicates.")
        else:
            n_doi_dupes = 0

        # Title dedup — normalize then deduplicate
        n_after_doi = len(df)
        if "title" in df.columns:
            df["_title_norm"] = (df["title"].astype(str)
                                 .str.lower()
                                 .str.strip()
                                 .str.replace(r"\s+", " ", regex=True))
            df = df[~df["_title_norm"].duplicated(keep="first")].copy()
            df = df.drop(columns=["_title_norm"])
            n_title_dupes = n_after_doi - len(df)
            if n_title_dupes > 0:
                self.logger.info(f"Deduplication by title: removed {n_title_dupes} duplicates.")

        removed = n_before - len(df)
        self.logger.info(f"Total deduplication: removed {removed} records.")
        return df.reset_index(drop=True), removed

    def _normalize_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column types for downstream consistency."""
        # year → integer
        if "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

        # citation_count → integer
        if "citation_count" in df.columns:
            df["citation_count"] = pd.to_numeric(
                df["citation_count"], errors="coerce").fillna(0).astype(int)

        # citations_per_year → float
        if "citations_per_year" in df.columns:
            df["citations_per_year"] = pd.to_numeric(
                df["citations_per_year"], errors="coerce").fillna(0.0)

        # journal_if → float
        if "journal_if" in df.columns:
            df["journal_if"] = pd.to_numeric(df["journal_if"], errors="coerce")

        # String fields: strip whitespace
        str_cols = ["title", "abstract", "first_author", "last_author",
                    "source_journal", "language", "document_type"]
        for col in str_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        return df

    def _resolve_extra_required_fields(self) -> list:
        """
        Build the list of extra required fields beyond the mandatory core.
        Sources (in priority order):
        1. pipeline_config.yaml — step2_cleaning.required_fields
        2. project_config.md   — inferred from 'Essential data' section if present
        """
        extra = list(self.extra_required)  # from config

        # Try to read project_config.md for explicit extra requirements
        if self.project_config_path and self.project_config_path.exists():
            inferred = self._infer_required_from_project_config()
            for field in inferred:
                if field not in extra and field not in self.MANDATORY_FIELDS:
                    extra.append(field)
                    self.logger.info(f"Extra required field inferred from project_config: '{field}'")

        return extra

    def _infer_required_from_project_config(self) -> list:
        """
        Parse project_config.md for explicit field requirements.
        Looks for a section '## Essential data fields' (or similar header
        containing 'essential/required/mandatory' + 'fields') and maps the
        field names mentioned inside it to canonical schema fields.
        Falls back to a legacy keyword scan if no such section exists.
        Returns a list of canonical schema field names.
        """
        required = []
        try:
            import re
            text = self.project_config_path.read_text(encoding="utf-8")

            # Simple keyword-based inference — covers the most common cases
            # Maps keywords in project_config to canonical field names
            keyword_field_map = {
                "journal": "source_journal",
                "publication name": "source_journal",
                "impact factor": "journal_if",
                "jnif": "jnif",
                "language": "language",
                "document type": "document_type",
                "keyword": "keywords",
                "country": "first_country",
                "institution": "first_institution",
                "doi": "doi",
                "citation": "citation_count",
            }

            # Preferred: dedicated section header, e.g. "## Essential data fields"
            section_match = re.search(
                r"^##\s*(essential|required|mandatory)[^\n]*fields?[^\n]*$",
                text, re.IGNORECASE | re.MULTILINE)
            if section_match:
                section_start = section_match.end()
                next_header = re.search(r"^##\s", text[section_start:], re.MULTILINE)
                section_end = (section_start + next_header.start()
                               if next_header else len(text))
                section_text = text[section_start:section_end].lower()
                for keyword, field in keyword_field_map.items():
                    if keyword in section_text:
                        required.append(field)
                if required:
                    self.logger.info(
                        f"Extra required fields inferred from project_config "
                        f"'Essential data fields' section: {required}")
            else:
                # Legacy fallback: keyword scan near 'essential/required/mandatory'
                essential_section = re.search(
                    r"(essential|required|mandatory).{0,200}", text, re.DOTALL)
                if essential_section:
                    section_text = essential_section.group(0)
                    for keyword, field in keyword_field_map.items():
                        if keyword in section_text:
                            required.append(field)
        except Exception as e:
            self.logger.warning(f"Could not parse project_config.md for extra fields: {e}")

        return required

    def _validate_extra_fields(self, df: pd.DataFrame,
                               extra_fields: list) -> tuple[pd.DataFrame, int]:
        """Remove records missing any extra required field."""
        if not extra_fields:
            return df, 0

        n_before = len(df)
        mask_valid = pd.Series(True, index=df.index)

        for field in extra_fields:
            if field not in df.columns:
                self.logger.warning(f"Extra required field '{field}' not in corpus — skipping.")
                continue
            field_empty = (
                df[field].isna() |
                (df[field].astype(str).str.strip() == "") |
                (df[field].astype(str).str.strip().str.lower().isin(
                    ["none", "null", "nan", "n/a", "na"]))
            )
            n_missing = field_empty.sum()
            if n_missing > 0:
                self.logger.info(f"  Extra field '{field}': {n_missing} records missing")
            mask_valid &= ~field_empty

        df_clean = df[mask_valid].copy()
        removed = n_before - len(df_clean)
        if removed > 0:
            self.logger.info(f"Extra field validation: removed {removed} records.")
        return df_clean, removed

    # ── Output ────────────────────────────────────────────────────────────────

    def _save_output(self, df: pd.DataFrame) -> tuple[str, str]:
        """Save corpus_clean.pkl (primary) and corpus_clean.csv (inspection)."""
        workspace = self.state.get("project", {}).get("workspace_path", ".")
        output_dir_int = Path(workspace) / "data" / "02_intermediate"
        output_dir_exp = Path(workspace) / "data" / "05_exports"
        output_dir_int.mkdir(parents=True, exist_ok=True)
        output_dir_exp.mkdir(parents=True, exist_ok=True)

        pkl_path = str(output_dir_int / "step2_corpus_clean.pkl")
        csv_path = str(output_dir_exp / "step2_corpus_clean.csv")

        df.to_pickle(pkl_path)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        self.logger.info(f"Saved {len(df)} records to {pkl_path}")
        return pkl_path, csv_path

    def _save_quality_report(self, n_raw: int, removed_mandatory: int,
                             removed_abstract: int, removed_year: int,
                             removed_language: int, removed_duplicates: int,
                             removed_extra: int, extra_fields: list,
                             n_final: int) -> str:
        """Generate a markdown quality report for auditing."""
        workspace = self.state.get("project", {}).get("workspace_path", ".")
        output_dir = Path(workspace) / "data" / "04_reports"
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = str(output_dir / "step2_quality_report.md")

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_removed = (removed_mandatory + removed_abstract + removed_year
                         + removed_language + removed_duplicates + removed_extra)
        retention_pct = round(100 * n_final / n_raw, 1) if n_raw > 0 else 0

        lines = [
            "# AKL Step 2 — Data Quality Report",
            f"\nGenerated: {now}",
            "\n## Summary\n",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Records input (Step 1) | {n_raw} |",
            f"| Removed — missing mandatory fields | {removed_mandatory} |",
            f"| Removed — abstract too short (<{self.MIN_ABSTRACT_LENGTH} chars) | {removed_abstract} |",
            f"| Removed — outside analysis window "
            f"({self.year_min or '…'}–{self.year_max or '…'}) | {removed_year} |",
            f"| Removed — language filter ({self.language_filter or 'none'}) | {removed_language} |",
            f"| Removed — duplicates (DOI + title) | {removed_duplicates} |",
            f"| Removed — missing extra required fields | {removed_extra} |",
            f"| **Records output (corpus_clean)** | **{n_final}** |",
            f"| Retention rate | {retention_pct}% |",
            "\n## Mandatory fields checked\n",
        ]
        for f in self.MANDATORY_FIELDS:
            lines.append(f"- `{f}`")

        if self.detect_language:
            lines.append("\n## Language handling\n")
            lines.append("- Language detection: **enabled** (langdetect on title + abstract; "
                         "on divergence, the non-English language is kept).")
            if self.language_filter:
                lines.append(f"- Language filter: keep only records with language = "
                             f"**{self.language_filter}**.")
            else:
                lines.append("- Language filter: not configured (all languages kept).")

        if extra_fields:
            lines.append("\n## Extra required fields checked\n")
            for f in extra_fields:
                lines.append(f"- `{f}`")
        else:
            lines.append("\n## Extra required fields\n")
            lines.append("None configured.")

        if self.year_min is not None or self.year_max is not None:
            lines.append("\n## Analysis window\n")
            lines.append(f"- Corpus restricted to publication years "
                         f"**{self.year_min or '…'}–{self.year_max or '…'}** (inclusive).")
            lines.append("- Records outside this window (e.g. early-access items dated "
                         "after the window end) were removed.")

        if total_removed == 0:
            lines.append("\n## Notes\n")
            lines.append("No records were removed. Corpus passed all quality checks.")
        elif retention_pct < 70:
            lines.append("\n## ⚠️  Warning\n")
            lines.append(
                f"Retention rate is {retention_pct}% — more than 30% of records were removed. "
                "Review the source data or consider relaxing extra field requirements."
            )

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        self.logger.info(f"Quality report saved to {report_path}")
        return report_path

    # ── State management ──────────────────────────────────────────────────────

    def _update_state_status(self, status: str, output_pkl: str = "",
                             output_csv: str = "", quality_report: str = "",
                             params: dict = None, notes: str = "") -> None:
        now = datetime.now().isoformat(timespec="seconds")
        step = self.state.setdefault("steps", {}).setdefault("2", {})
        step["status"] = status

        if status == "running":
            step["started_at"] = now
            self.state["current_step"] = 2
        elif status in ("completed", "failed"):
            step["completed_at"] = now

        if output_pkl:
            step.setdefault("output", {}).update({
                "corpus_file": output_pkl,
                "corpus_file_csv": output_csv,
                "quality_report_file": quality_report,
            })
        if params:
            step.setdefault("params_used", {}).update(params)
        if notes:
            step["notes"] = notes

        self.state["project"]["last_updated"] = now
        self._save_json(self.state, self.state_path)

    # ── File helpers ──────────────────────────────────────────────────────────

    def _load_yaml(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _load_json(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_json(self, data: dict, path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AKL Step 2 — Data Cleaning")
    parser.add_argument("--input", "-i", default="data/02_intermediate/step1_corpus_raw.pkl",
                        help="Path to corpus_raw.pkl (default: data/inputs/corpus_raw.pkl)")
    parser.add_argument("--config", default="pipeline_config.yaml")
    parser.add_argument("--state", default="pipeline_state.json")
    parser.add_argument("--project-config", default="project_config.md",
                        help="Path to project_config.md for extra field inference")
    args = parser.parse_args()

    worker = Step2Worker(args.config, args.state, args.project_config)
    result = worker.run(args.input)

    print("\n── Step 2 Result ──────────────────────────────")
    for k, v in result.items():
        print(f"  {k}: {v}")