"""
step1_language_if_worker.py
----------------------------
AKL Pipeline — Pre-Step-2 enrichment pass on the Step 1 corpus.

Two activities:

1. LANGUAGE VALIDATION — validates/corrects the `language` column by
   detecting the language of title + abstract (langdetect):

   - language says "English" but title/abstract is in another language
     → corrected to that language (only when detection is very confident:
     prob >= 0.9 AND abstract-only detection agrees)
   - language says another language but title/abstract is in English
     → corrected to "English" (prob >= 0.7 AND abstract-only agrees)
   - empty or garbage values (e.g., "Ang") → filled from detection

2. IMPACT FACTOR FILL — fills missing `journal_if` values using:
   - the `impact_factor` library (exact match, then journal abbreviation)
   - OpenAlex Sources `2yr_mean_citedness` via pyalex as fallback
     (the pipeline already treats this metric as journal_if for OpenAlex
     records, so it is consistent methodology)

After filling journal_if, the normalized `jnif` column is recomputed on
the enriched corpus.

Saves the corrected corpus back to step1_corpus_raw.pkl (and the CSV
inspection copy). Does NOT change pipeline_state.json — this is an
enrichment of the Step 1 output, not a pipeline step.

Usage:
    ./run_worker.sh workers/step1_language_if_worker.py \
        --config pipeline_config.yaml \
        --state pipeline_state.json
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

# ── Logging ──────────────────────────────────────────────────────────────────

def _setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("step1_language_if_worker")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [STEP1-ENRICH] %(levelname)s %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


# ── Language helpers ─────────────────────────────────────────────────────────

# langdetect ISO codes that pycountry does not resolve cleanly
LANG_SPECIAL = {
    "zh-cn": "Chinese", "zh-tw": "Chinese", "zh": "Chinese",
    "nb": "Norwegian", "nn": "Norwegian", "no": "Norwegian",
    "pt-br": "Portuguese", "pt-pt": "Portuguese",
    "iw": "Hebrew", "in": "Indonesian", "ji": "Yiddish",
    "tl": "Tagalog", "fil": "Filipino", "msa": "Malay",
    "fa": "Persian", "uk": "Ukrainian", "el": "Greek",
    "he": "Hebrew", "ar": "Arabic", "hi": "Hindi",
    "bn": "Bengali", "th": "Thai", "vi": "Vietnamese",
    "ko": "Korean", "ja": "Japanese", "ru": "Russian",
    "pl": "Polish", "cs": "Czech", "sk": "Slovak",
    "hu": "Hungarian", "ro": "Romanian", "bg": "Bulgarian",
    "da": "Danish", "sv": "Swedish", "fi": "Finnish",
    "nl": "Dutch", "it": "Italian", "fr": "French",
    "de": "German", "es": "Spanish", "pt": "Portuguese",
    "en": "English", "tr": "Turkish", "id": "Indonesian",
    "sr": "Serbian", "hr": "Croatian", "bs": "Bosnian",
    "ms": "Malay", "ca": "Catalan", "eu": "Basque",
    "gl": "Galician", "cy": "Welsh", "ga": "Irish",
    "mt": "Maltese", "is": "Icelandic", "lt": "Lithuanian",
    "lv": "Latvian", "et": "Estonian", "sq": "Albanian",
    "mk": "Macedonian", "sl": "Slovenian", "az": "Azerbaijani",
    "ka": "Georgian", "hy": "Armenian", "kk": "Kazakh",
    "uz": "Uzbek", "mn": "Mongolian", "ne": "Nepali",
    "si": "Sinhala", "ta": "Tamil", "te": "Telugu",
    "kn": "Kannada", "ml": "Malayalam", "gu": "Gujarati",
    "pa": "Punjabi", "ur": "Urdu", "am": "Amharic",
    "sw": "Swahili", "af": "Afrikaans", "zu": "Zulu",
    "xh": "Xhosa", "yo": "Yoruba", "ig": "Igbo",
    "ha": "Hausa", "so": "Somali", "rw": "Kinyarwanda",
    "ny": "Chichewa", "sn": "Shona", "st": "Sesotho",
    "tn": "Setswana", "ts": "Tsonga", "ve": "Venda",
    "ss": "Swati", "nso": "Sepedi", "ceb": "Cebuano",
    "war": "Waray", "ilo": "Iloko", "hmn": "Hmong",
    "mg": "Malagasy", "mi": "Maori", "sm": "Samoan",
    "to": "Tongan", "fj": "Fijian", "ht": "Haitian Creole",
    "jv": "Javanese", "su": "Sundanese", "my": "Burmese",
    "km": "Khmer", "lo": "Lao", "bo": "Tibetan",
    "dz": "Dzongkha", "ps": "Pashto", "sd": "Sindhi",
    "dv": "Divehi", "si": "Sinhala", "tk": "Turkmen",
    "ky": "Kyrgyz", "tg": "Tajik", "be": "Belarusian",
    "mk": "Macedonian", "sq": "Albanian", "hy": "Armenian",
    "ka": "Georgian", "az": "Azerbaijani", "kk": "Kazakh",
    "uz": "Uzbek", "mn": "Mongolian", "ne": "Nepali",
    "ta": "Tamil", "te": "Telugu", "kn": "Kannada",
    "ml": "Malayalam", "gu": "Gujarati", "pa": "Punjabi",
    "ur": "Urdu", "am": "Amharic", "sw": "Swahili",
    "af": "Afrikaans", "zu": "Zulu", "xh": "Xhosa",
    "yo": "Yoruba", "ig": "Igbo", "ha": "Hausa",
    "so": "Somali", "rw": "Kinyarwanda", "ny": "Chichewa",
    "sn": "Shona", "st": "Sesotho", "tn": "Setswana",
    "ts": "Tsonga", "ve": "Venda", "ss": "Swati",
    "nso": "Sepedi", "ceb": "Cebuano", "war": "Waray",
    "ilo": "Iloko", "hmn": "Hmong", "mg": "Malagasy",
    "mi": "Maori", "sm": "Samoan", "to": "Tongan",
    "fj": "Fijian", "ht": "Haitian Creole", "jv": "Javanese",
    "su": "Sundanese", "my": "Burmese", "km": "Khmer",
    "lo": "Lao", "bo": "Tibetan", "dz": "Dzongkha",
    "ps": "Pashto", "sd": "Sindhi", "dv": "Divehi",
    "tk": "Turkmen", "ky": "Kyrgyz", "tg": "Tajik",
    "be": "Belarusian",
}

GARBAGE_LANGUAGES = {"", "ang", "nan", "none", "unknown", "n/a", "na"}


def _iso_to_language_name(code: str) -> str:
    """Convert a langdetect ISO code to a full language name."""
    if not code:
        return ""
    code = code.lower()
    if code in LANG_SPECIAL:
        return LANG_SPECIAL[code]
    try:
        import pycountry
        obj = pycountry.languages.get(alpha_2=code)
        if obj:
            return obj.name
    except Exception:
        pass
    return code.title()


def _detect_language(text: str):
    """Detect the language of a text. Returns (iso_code, prob) or (None, 0.0)."""
    if not text or len(str(text).strip()) < 20:
        return None, 0.0
    try:
        from langdetect import detect_langs
        from langdetect.lang_detect_exception import LangDetectException
        try:
            langs = detect_langs(str(text)[:5000])
            if langs:
                return langs[0].lang, langs[0].prob
        except LangDetectException:
            pass
    except ImportError:
        pass
    return None, 0.0


def _correct_language(current: str, ta_code: str, ta_prob: float,
                      abs_code: str, abs_prob: float) -> str:
    """
    Apply the specialist's correction rules to a single record.

    Rules:
    - current empty/garbage → fill from detection (prob >= 0.5)
    - current "English", detected other → correct only if very confident
      (prob >= 0.9 AND abstract-only agrees)
    - current other, detected "English" → correct to "English" if confident
      (prob >= 0.7 AND abstract-only agrees when an abstract exists)
    - otherwise keep current
    """
    current_clean = str(current or "").strip()
    current_low = current_clean.lower()

    if ta_code is None:
        return current_clean  # could not detect — keep as-is

    ta_name = _iso_to_language_name(ta_code)
    abs_name = _iso_to_language_name(abs_code) if abs_code else None

    # 1) Empty / garbage current value → fill from detection
    if current_low in GARBAGE_LANGUAGES:
        if ta_prob >= 0.5:
            return ta_name
        return current_clean

    # 2) Current "English" but content detected as another language
    if current_low == "english" and ta_code != "en":
        # Only override when very confident AND abstract-only agrees
        if ta_prob >= 0.9 and (abs_code is None or abs_code == ta_code):
            return ta_name
        return current_clean

    # 3) Current another language but content detected as English
    if current_low != "english" and ta_code == "en":
        if ta_prob >= 0.7 and (abs_code is None or abs_code == "en"):
            return "English"
        return current_clean

    return current_clean


# ── Impact factor helpers ────────────────────────────────────────────────────

# Markers that identify non-journal sources (no impact factor by definition)
NON_JOURNAL_MARKERS = [
    "arxiv", "zenodo", "drops", "ascilite", "ssrn", "research square",
    "preprints.org", "osf preprints", "techrxiv", "authorea", "figshare",
    "biorxiv", "medrxiv", "psyarxiv", "socarxiv", "edarxiv", "engrxiv",
    "ieee dataport", "proceedings of the canadian engineering",
    "hal open science", "openreview", "osf.io",
]

_IF_INSTANCE = None
_SOURCES_API = None


def _get_if_instance():
    global _IF_INSTANCE
    if _IF_INSTANCE is None:
        from impact_factor.core import Factor
        _IF_INSTANCE = Factor()
    return _IF_INSTANCE


def _get_sources_api():
    global _SOURCES_API
    if _SOURCES_API is None:
        from pyalex import Sources
        import pyalex
        pyalex.config.email = "akl-pipeline-test@example.com"
        _SOURCES_API = Sources()  # instance — .search() is an instance method
    return _SOURCES_API


def _fill_journal_if(source_journal: str):
    """Fill a missing journal_if for one journal name. Returns float or None."""
    if not source_journal or pd.isna(source_journal):
        return None
    sj = str(source_journal).strip()
    if not sj or sj.lower() in ("nan", "none"):
        return None
    low = sj.lower()
    if any(m in low for m in NON_JOURNAL_MARKERS):
        return None

    # 1) impact_factor — exact journal name
    try:
        res = _get_if_instance().search(sj)
        if res:
            v = float(res[0].get("factor", 0)) or None
            if v:
                return v
    except Exception:
        pass

    # 2) impact_factor — journal abbreviation
    try:
        res = _get_if_instance().search(sj, key="journal_abbr")
        if res:
            v = float(res[0].get("factor", 0)) or None
            if v:
                return v
    except Exception:
        pass

    # 3) OpenAlex Sources — 2yr_mean_citedness (JIF-equivalent used by the pipeline)
    try:
        sources = list(_get_sources_api().search(sj).get())
        if sources:
            ss = sources[0].get("summary_stats") or {}
            v = ss.get("2yr_mean_citedness")
            if v is not None:
                return float(v)
    except Exception:
        pass

    return None


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AKL Step 1 — language validation + impact factor enrichment")
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

    corpus_path = Path(workspace) / "data" / "02_intermediate" / "step1_corpus_raw.pkl"
    if not corpus_path.exists():
        logger.error(f"Corpus not found: {corpus_path}")
        sys.exit(1)

    df = pd.read_pickle(corpus_path)
    logger.info(f"Loaded {len(df)} records from {corpus_path}")

    # ── 1) Language validation ──────────────────────────────────────────────
    logger.info("Activity 1: validating language column (langdetect)...")
    lang_changes = []
    for df_idx, r in df.iterrows():
        ta_text = f"{r.get('title') or ''} {r.get('abstract') or ''}".strip()
        ta_code, ta_prob = _detect_language(ta_text)
        abs_text = (r.get("abstract") or "").strip()
        abs_code, abs_prob = _detect_language(abs_text) if abs_text else (None, 0.0)
        new_lang = _correct_language(r.get("language"), ta_code, ta_prob,
                                     abs_code, abs_prob)
        old_lang = str(r.get("language") or "").strip()
        if new_lang != old_lang:
            lang_changes.append((df_idx, old_lang, new_lang, ta_code, ta_prob))
            df.at[df_idx, "language"] = new_lang

    logger.info(f"Language corrections: {len(lang_changes)} records changed")
    for df_idx, old, new, code, prob in lang_changes:
        logger.info(f"  [{df_idx}] '{old}' -> '{new}' (detected {code}, prob {prob:.3f})")

    # ── 2) Impact factor fill ───────────────────────────────────────────────
    logger.info("Activity 2: filling missing journal_if values...")
    missing_mask = df["journal_if"].isna()
    logger.info(f"  Records with missing journal_if: {missing_mask.sum()}")

    # Cache lookups per unique journal name
    if_cache = {}
    filled = 0
    for idx in df.index[missing_mask]:
        sj = df.at[idx, "source_journal"]
        if not sj or pd.isna(sj):
            continue
        sj = str(sj).strip()
        if sj not in if_cache:
            if_cache[sj] = _fill_journal_if(sj)
        val = if_cache[sj]
        if val is not None:
            df.at[idx, "journal_if"] = val
            filled += 1

    logger.info(f"  Filled journal_if for {filled} records "
                f"({len(if_cache)} unique journals looked up)")
    still_missing = df["journal_if"].isna().sum()
    logger.info(f"  Still missing journal_if: {still_missing}")

    # ── Recompute jnif on the enriched corpus ───────────────────────────────
    if "journal_if" in df.columns:
        if_series = pd.to_numeric(df["journal_if"], errors="coerce")
        mean_if = if_series.mean()
        std_if = if_series.std()
        if std_if and std_if > 0:
            df["jnif"] = ((if_series - mean_if) / std_if).round(4)
        else:
            df["jnif"] = 0.0
        logger.info(f"Recomputed jnif (mean={mean_if:.3f}, std={std_if:.3f})")

    # ── Save ────────────────────────────────────────────────────────────────
    output_dir_pkl = Path(workspace) / "data" / "02_intermediate"
    output_dir_csv = Path(workspace) / "data" / "01_raw"
    pkl_path = output_dir_pkl / "step1_corpus_raw.pkl"
    csv_path = output_dir_csv / "step1_corpus_raw.csv"
    df.to_pickle(pkl_path)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved enriched corpus to {pkl_path}")

    print("\n── Step 1 Enrichment Result ───────────────────────")
    print(f"  Language corrections:      {len(lang_changes)}")
    print(f"  journal_if filled:         {filled}")
    print(f"  journal_if still missing:  {still_missing}")
    print(f"  Output file:               {pkl_path}")


if __name__ == "__main__":
    main()