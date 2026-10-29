"""
step1_worker.py
---------------
AKL Pipeline — Step 1: Raw Data Acquisition

Responsibilities:
- Load data from WOS (CSV/TXT upload), OpenAlex (API query),
  PubMed (API query), or arXiv (API query)
- Parse and derive all author/affiliation fields
- Normalize everything to the internal canonical schema
- Save corpus_raw.csv to data/inputs/
- Update pipeline_state.json

Usage (from a skill or directly):
    from workers.step1_worker import Step1Worker
    worker = Step1Worker("pipeline_config.yaml", "pipeline_state.json")
    result = worker.run_wos("data/inputs/wos_export.txt")
    # or
    result = worker.run_openalex("alkaline electrolysis hydrogen production", max_results=5000)
    # or
    result = worker.run_arxiv('(all:"computer science education") AND (all:ChatGPT)', max_results=5000)
"""

import json
import logging
import os
import re
import sys
import warnings
from datetime import datetime

try:
    from workers.reference_date import reference_year as _reference_year
except ImportError:  # executed from inside workers/
    from reference_date import reference_year as _reference_year
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

warnings.filterwarnings("ignore")

# ── Optional dependencies — imported lazily to give clear error messages ──────

def _require(package: str, install_name: Optional[str] = None):
    """Import a package, raising a clear error if missing."""
    import importlib
    try:
        return importlib.import_module(package)
    except ImportError:
        pip_name = install_name or package
        raise ImportError(
            f"Required package '{pip_name}' is not installed.\n"
            f"Install it with: pip install {pip_name}"
        )


# ── Logging setup ─────────────────────────────────────────────────────────────

def _setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("step1_worker")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [STEP1] %(levelname)s %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        # file handler
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        # console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger

# ── Aligned WoS Address Parsing Utilities ────────────────────────────────────

COUNTRY_CORRECTIONS = {
    'United States of America': 'United States',
    "USA": "United States",
    "U Arab Emirates": "United Arab Emirates",
    "Turkiye": "Turkey",
    "Viet Nam": "Vietnam",
    "BELARUS": "Belarus",
    "DEM REP CONGO": "Democratic Republic of the Congo",
    "IL 6020 USA": "United States",
    "Indian Inst Technol Jodhpur": "India",
    "Peoples R China": "China",
    "UK": "United Kingdom",
    "U.K.": "United Kingdom",
    "U.K": "United Kingdom",
    "England": "United Kingdom",
    "Scotland": "United Kingdom",
    "Wales": "United Kingdom",
    "Northern Ireland": "United Kingdom",
    "UAE": "United Arab Emirates",
    "PR China": "China",
    "P.R. China": "China",
    "ROC": "Taiwan",
    "R.O.C.": "Taiwan",
}

USA_PATTERN = re.compile(r"[A-Z]{2}\s\d{5}\sUSA$", re.IGNORECASE)    
USA_PATTERN_2 = re.compile(r"[A-Z]{2}\sUSA$", re.IGNORECASE)    

def _split_outside_brackets(s, sep=';'):
    if pd.isna(s) or not isinstance(s, str):
        return []
    parts, buf = [], []
    level = 0
    for ch in s:
        if ch == '[':
            level += 1
            buf.append(ch)
        elif ch == ']':
            level = max(level - 1, 0)
            buf.append(ch)
        elif ch == sep and level == 0:
            part = ''.join(buf).strip()
            if part:
                parts.append(part)
            buf = []
        else:
            buf.append(ch)
    last = ''.join(buf).strip()
    if last:
        parts.append(last)
    return parts

def _extract_authors_block(block):
    m = re.search(r'\[(.*?)\]', block)
    if not m:
        return []
    return [a.strip() for a in m.group(1).split(';') if a.strip()]

def _extract_institution_from_block(block):
    after = block.split(']', 1)[1].strip() if ']' in block else block.strip()
    if not after:
        return "Unknown"
    inst = after.split(',', 1)[0].strip().lstrip('.,;:')
    if not inst:
        return "Unknown"
    return "Cent South Univ" if inst == "Cent S Univ" else inst

def _extract_country_from_block(block):
    after = block.split(']', 1)[1].strip() if ']' in block else block.strip()
    if not after:
        return "Unknown"
    parts = [p.strip() for p in after.split(',') if p.strip()]
    country_raw = parts[-1] if parts else "Unknown"
    country = COUNTRY_CORRECTIONS.get(country_raw, country_raw)
    if USA_PATTERN.search(country) or USA_PATTERN_2.search(country):
        return "United States"
    return country

def _normalize_name(name):
    if name is None:
        return name
    return re.sub(r'\s+', ' ', re.sub(r'\.', '', str(name))).strip().lower()

# ── Country → Region lookup ───────────────────────────────────────────────────

COUNTRY_REGION_CACHE: dict[str, str] = {}

def _iso_to_name(code: str, is_language=False) -> str:
    """Convert ISO codes from OpenAlex to full names (Countries or Languages)."""
    if not code or pd.isna(code):
        return "" if is_language else "Unknown"
    try:
        pycountry = _require("pycountry")
        if is_language:
            obj = pycountry.languages.get(alpha_2=code.lower())
            return obj.name if obj else str(code).title()
        else:
            obj = pycountry.countries.get(alpha_2=code.upper())
            return obj.name if obj else str(code)
    except Exception:
        return str(code).title() if is_language else str(code)
    
def _country_to_region(country_name: str) -> str:
    """Convert country name or ISO code to continent/region name."""
    if not country_name or pd.isna(country_name):
        return ""
    key = str(country_name).strip()
    if key in COUNTRY_REGION_CACHE:
        return COUNTRY_REGION_CACHE[key]
    try:
        pycountry = _require("pycountry")
        pc_convert = _require("pycountry_convert")
        # try by name first, then by alpha-2 code
        country = (pycountry.countries.get(name=key)
                   or pycountry.countries.get(alpha_2=key.upper())
                   or pycountry.countries.search_fuzzy(key)[0])
        alpha2 = country.alpha_2
        continent_code = pc_convert.country_alpha2_to_continent_code(alpha2)
        continent_map = {
            "AF": "Africa", "AN": "Antarctica", "AS": "Asia",
            "EU": "Europe", "NA": "North America", "OC": "Oceania",
            "SA": "South America",
        }
        region = continent_map.get(continent_code, "Unknown")
    except Exception:
        region = "Unknown"
    COUNTRY_REGION_CACHE[key] = region
    return region


# ── Impact factor lookup ──────────────────────────────────────────────────────

IF_INSTANCE = None

def _get_impact_factor(journal_name: str) -> Optional[float]:
    """Look up journal impact factor using the impact_factor library."""
    global IF_INSTANCE
    if not journal_name or pd.isna(journal_name):
        return None
    try:
        if IF_INSTANCE is None:
            Factor = _require("impact_factor.core", "impact_factor").Factor
            IF_INSTANCE = Factor()
        result = IF_INSTANCE.search(str(journal_name))
        if result and len(result) > 0:
            return float(result[0].get("factor", 0)) or None
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main worker class
# ─────────────────────────────────────────────────────────────────────────────

class Step1Worker:
    """
    Acquires raw bibliometric data from WOS or OpenAlex and normalizes
    it to the AKL internal canonical schema.
    """

    def __init__(self, config_path: str, state_path: str):
        self.config_path = Path(config_path)
        self.state_path = Path(state_path)
        self.config = self._load_yaml(config_path)
        self.state = self._load_json(state_path)
        self.schema = self.config.get("schema", {})
        self.source_mapping = self.config.get("source_mapping", {})
        log_path = self.state.get("project", {}).get("workspace_path", ".")
        self.logger = _setup_logger(os.path.join(log_path, "execution.log"))
        self.logger.info("Step1Worker initialized.")

    # ── Public entry points ───────────────────────────────────────────────────

    def run_wos(self, input_file: str, output_stem: str = "") -> dict:
        """
        Process a WOS export file (tab-separated .txt or .csv).
        Returns a result dict with status, output_file, and stats.
        """
        self.logger.info(f"Starting WOS acquisition from: {input_file}")
        self._update_state_status("running", params={"source_type": "WOS",
                                                      "upload_filename": input_file})
        try:
            df_raw = self._load_wos_file(input_file)
            self.logger.info(f"Loaded {len(df_raw)} raw records from WOS.")

            df = self._parse_wos(df_raw)
            df = self._compute_derived_fields(df, source="WOS")
            df = self._normalize_to_schema(df, source="wos")

            output_file = self._save_output(df, "corpus_raw.csv", output_stem)
            stats = self._build_stats(df, len(df_raw))
            self._update_state_status("completed", output_file=output_file,
                                      params={"source_type": "WOS",
                                              "upload_filename": input_file,
                                              **stats})
            self.logger.info(f"WOS acquisition complete. Output: {output_file}")
            return {"status": "completed", "output_file": output_file, **stats}

        except Exception as e:
            self.logger.error(f"WOS acquisition failed: {e}", exc_info=True)
            self._update_state_status("failed", notes=str(e))
            raise

    def run_openalex(self, query: str, max_results: int = 10000,
                     email: Optional[str] = None, output_stem: str = "") -> dict:
        """
        Query the OpenAlex API and return the canonical corpus.
        email: optional polite-pool email (10x rate limit).
        """
        self.logger.info(f"Starting OpenAlex acquisition. Query: '{query}', max: {max_results}")
        self._update_state_status("running", params={"source_type": "OpenAlex",
                                                       "openalex_query": query})
        try:
            pyalex = _require("pyalex")
            if email:
                pyalex.config.email = email

            records = self._fetch_openalex(pyalex, query, max_results)
            self.logger.info(f"Fetched {len(records)} records from OpenAlex.")

            df = self._parse_openalex(records)
            df = self._compute_derived_fields(df, source="OpenAlex")
            df = self._normalize_to_schema(df, source="openalex")

            output_file = self._save_output(df, "corpus_raw.csv", output_stem)
            stats = self._build_stats(df, len(records))
            self._update_state_status("completed", output_file=output_file,
                                      params={"source_type": "OpenAlex",
                                              "openalex_query": query,
                                              **stats})
            self.logger.info(f"OpenAlex acquisition complete. Output: {output_file}")
            return {"status": "completed", "output_file": output_file, **stats}

        except Exception as e:
            self.logger.error(f"OpenAlex acquisition failed: {e}", exc_info=True)
            self._update_state_status("failed", notes=str(e))
            raise

    # ── PubMed API fetching and parsing ──────────────────────────────────────

    def run_pubmed(self, query: str, max_results: int = 10000,
                   api_key: Optional[str] = None, email: Optional[str] = None,
                   output_stem: str = "") -> dict:
        """
        Query PubMed via NCBI E-utilities and return the canonical corpus.
        api_key: optional NCBI API key (raises rate limit from 3/s to 10/s).
        email: optional email for NCBI polite pool.
        """
        self.logger.info(f"Starting PubMed acquisition. Query: '{query}', max: {max_results}")
        self._update_state_status("running", params={"source_type": "PubMed",
                                                       "pubmed_query": query})
        try:
            pmids = self._fetch_pubmed_pmids(query, max_results, api_key, email)
            self.logger.info(f"Found {len(pmids)} PMIDs.")

            if not pmids:
                raise ValueError("No records found for the given query.")

            records = self._fetch_pubmed_records(pmids, api_key, email)
            self.logger.info(f"Fetched {len(records)} full records from PubMed.")

            df = self._parse_pubmed_xml(records)
            df = self._compute_derived_fields(df, source="PubMed")
            df = self._normalize_to_schema(df, source="pubmed")

            output_file = self._save_output(df, "corpus_raw.csv", output_stem)
            stats = self._build_stats(df, len(df))
            self._update_state_status("completed", output_file=output_file,
                                      params={"source_type": "PubMed",
                                              "pubmed_query": query,
                                              **stats})
            self.logger.info(f"PubMed acquisition complete. Output: {output_file}")
            return {"status": "completed", "output_file": output_file, **stats}

        except Exception as e:
            self.logger.error(f"PubMed acquisition failed: {e}", exc_info=True)
            self._update_state_status("failed", notes=str(e))
            raise

    # ── arXiv API fetching and parsing ──────────────────────────────────────

    def run_arxiv(self, query: str, max_results: int = 10000,
                  email: Optional[str] = None, output_stem: str = "") -> dict:
        """
        Query the arXiv API (Atom feed) and return the canonical corpus.

        Preprints without a published DOI receive the DataCite fallback DOI
        `10.48550/arXiv.<id>` so the `doi` field is never blank.
        email: optional contact email appended to the User-Agent (arXiv
        recommends including it for the polite pool).
        """
        self.logger.info(f"Starting arXiv acquisition. Query: '{query}', max: {max_results}")
        self._update_state_status("running", params={"source_type": "arXiv",
                                                       "arxiv_query": query})
        try:
            xml_chunks = self._fetch_arxiv(query, max_results, email)
            self.logger.info(f"Fetched {len(xml_chunks)} XML chunks from arXiv.")

            df = self._parse_arxiv(xml_chunks)
            df = self._compute_derived_fields(df, source="arXiv")
            df = self._normalize_to_schema(df, source="arxiv")

            output_file = self._save_output(df, "corpus_raw.csv", output_stem)
            stats = self._build_stats(df, len(df))
            self._update_state_status("completed", output_file=output_file,
                                      params={"source_type": "arXiv",
                                              "arxiv_query": query,
                                              **stats})
            self.logger.info(f"arXiv acquisition complete. Output: {output_file}")
            return {"status": "completed", "output_file": output_file, **stats}

        except Exception as e:
            self.logger.error(f"arXiv acquisition failed: {e}", exc_info=True)
            self._update_state_status("failed", notes=str(e))
            raise

    def _fetch_arxiv(self, query: str, max_results: int,
                     email: Optional[str] = None) -> list:
        """
        Fetch entries from the arXiv API with pagination.

        Uses https://export.arxiv.org/api/query with
        sortBy=submittedDate&sortOrder=descending (newest first).
        Respects the arXiv rate limit (1 request per 3 seconds).
        Returns a list of XML response strings (one per page).
        """
        import time
        import urllib.request
        import urllib.parse

        base = "https://export.arxiv.org/api/query"
        per_page = 200  # arXiv maximum per request
        user_agent = "AKL-Pipeline/1.0 (bibliometric research)"
        if email:
            user_agent += f" (mailto:{email})"

        chunks = []
        start = 0
        total = None

        while True:
            params = {
                "search_query": query,
                "start": str(start),
                "max_results": str(per_page),
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
            url = f"{base}?{urllib.parse.urlencode(params)}"
            self.logger.info(f"arXiv request: start={start}, max_results={per_page}")

            req = urllib.request.Request(url, headers={"User-Agent": user_agent})
            with urllib.request.urlopen(req, timeout=120) as resp:
                xml_data = resp.read().decode("utf-8")

            chunks.append(xml_data)

            # Extract totalResults from the first response
            if total is None:
                m = re.search(r"<opensearch:totalResults>(\d+)</opensearch:totalResults>",
                              xml_data)
                total = int(m.group(1)) if m else 0
                self.logger.info(f"arXiv total results: {total}")

            limit = min(total, max_results) if total else max_results
            start += per_page
            if start >= limit:
                break

            # arXiv rate limit: 1 request per 3 seconds
            time.sleep(3)

        return chunks

    def _parse_arxiv(self, xml_chunks: list) -> pd.DataFrame:
        """Parse arXiv Atom XML responses into a DataFrame."""
        from xml.etree import ElementTree as ET

        ns = {
            "a": "http://www.w3.org/2005/Atom",
            "ar": "http://arxiv.org/schemas/atom",
            "os": "http://a9.com/-/spec/opensearch/1.1/",
        }

        rows = []
        for xml_str in xml_chunks:
            try:
                root = ET.fromstring(xml_str)
            except ET.ParseError as e:
                self.logger.warning(f"XML parse error: {e} — skipping chunk.")
                continue

            for entry in root.findall("a:entry", ns):
                row = self._parse_single_arxiv_entry(entry, ns)
                if row:
                    rows.append(row)

        self.logger.info(f"Parsed {len(rows)} entries from arXiv XML.")
        return pd.DataFrame(rows)

    def _parse_single_arxiv_entry(self, entry, ns) -> Optional[dict]:
        """Extract canonical fields from a single arXiv Atom entry."""
        try:
            row = {}

            # arXiv ID — strip version suffix (e.g., "2609.14572v1" → "2609.14572")
            id_el = entry.find("a:id", ns)
            arxiv_id_full = id_el.text.strip() if id_el is not None and id_el.text else ""
            arxiv_id = re.sub(r"v\d+$", "", arxiv_id_full.rsplit("/", 1)[-1])
            row["arxiv_id"] = arxiv_id

            # Title
            title_el = entry.find("a:title", ns)
            row["title"] = " ".join(title_el.itertext()).strip() if title_el is not None else ""

            # Abstract (summary)
            summary_el = entry.find("a:summary", ns)
            row["abstract"] = " ".join(summary_el.itertext()).strip() if summary_el is not None else ""

            # Publication date (ISO YYYY-MM-DD)
            pub_el = entry.find("a:published", ns)
            row["publication_date"] = (pub_el.text or "")[:10] if pub_el is not None and pub_el.text else ""
            row["year"] = int(row["publication_date"][:4]) if len(row["publication_date"]) >= 4 else None

            # DOI — published DOI when present, else DataCite fallback
            doi_el = entry.find("ar:doi", ns)
            if doi_el is not None and doi_el.text and doi_el.text.strip():
                row["doi"] = doi_el.text.strip()
            else:
                row["doi"] = f"10.48550/arXiv.{arxiv_id}"

            # Journal reference (only present when published in a venue)
            jref_el = entry.find("ar:journal_ref", ns)
            row["source_journal"] = jref_el.text.strip() if jref_el is not None and jref_el.text else "arXiv"

            # Categories → research areas
            cats = [c.get("term", "") for c in entry.findall("a:category", ns) if c.get("term")]
            row["research_areas"] = "; ".join(cats)

            # Primary category
            prim = entry.find("ar:primary_category", ns)
            row["primary_category"] = prim.get("term", "") if prim is not None else (cats[0] if cats else "")

            # Authors (arXiv provides full names only)
            authors = []
            for author_el in entry.findall("a:author", ns):
                name_el = author_el.find("a:name", ns)
                if name_el is not None and name_el.text:
                    authors.append(name_el.text.strip())
            row["authors_list"] = authors
            row["authors_full_name_list"] = authors
            row["authors_institution"] = [""] * len(authors)
            row["authors_country"] = [""] * len(authors)

            row["first_author"] = authors[0] if authors else ""
            row["last_author"] = authors[-1] if authors else ""
            row["first_author_full_name"] = authors[0] if authors else ""
            row["last_author_full_name"] = authors[-1] if authors else ""
            row["first_institution"] = ""
            row["last_institution"] = ""
            row["first_country"] = ""
            row["last_country"] = ""

            # arXiv does not provide citation counts
            row["citation_count"] = 0

            # arXiv is English by default
            row["language"] = "English"

            # Document type
            row["document_type"] = "Preprint"

            # Keywords — not provided by arXiv
            row["keywords"] = ""

            # external_id — arXiv ID (always filled)
            row["external_id"] = arxiv_id

            row["raw_source"] = "arXiv"
            return row

        except Exception as e:
            self.logger.warning(f"Error parsing arXiv entry: {e}")
            return None

    def _fetch_pubmed_pmids(self, query: str, max_results: int,
                            api_key: Optional[str] = None,
                            email: Optional[str] = None) -> list:
        """Use ESearch to get PMIDs matching the query."""
        import urllib.request
        import urllib.parse

        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": str(max_results),
            "retmode": "json",
            "sort": "relevance",
        }
        if api_key:
            params["api_key"] = api_key
        if email:
            params["email"] = email

        url = f"{base}?{urllib.parse.urlencode(params)}"
        self.logger.info(f"ESearch URL: {url[:120]}...")

        req = urllib.request.Request(url, headers={"User-Agent": "AKL-Pipeline/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        result = data.get("esearchresult", {})
        id_list = result.get("idlist", [])
        total = int(result.get("count", 0))
        self.logger.info(f"ESearch: {total} total hits, returning {len(id_list)} PMIDs.")
        return id_list

    def _fetch_pubmed_records(self, pmids: list, api_key: Optional[str] = None,
                               email: Optional[str] = None) -> list:
        """Use EFetch to retrieve MEDLINE XML for a list of PMIDs (batched)."""
        import urllib.request
        import urllib.parse

        batch_size = 200  # NCBI recommends ≤200 per request
        all_records = []

        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i + batch_size]
            self.logger.info(f"  EFetch batch {i // batch_size + 1}: PMIDs {i+1}–{i+len(batch)}...")

            base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            params = {
                "db": "pubmed",
                "id": ",".join(batch),
                "rettype": "medline",
                "retmode": "xml",
            }
            if api_key:
                params["api_key"] = api_key
            if email:
                params["email"] = email

            url = f"{base}?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(url, headers={"User-Agent": "AKL-Pipeline/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                xml_data = resp.read().decode("utf-8")

            all_records.append(xml_data)

        return all_records

    def _parse_pubmed_xml(self, xml_chunks: list) -> pd.DataFrame:
        """Parse MEDLINE XML responses into a DataFrame."""
        from xml.etree import ElementTree as ET

        rows = []
        for xml_str in xml_chunks:
            try:
                root = ET.fromstring(xml_str)
            except ET.ParseError as e:
                self.logger.warning(f"XML parse error: {e} — skipping chunk.")
                continue

            for article in root.findall(".//PubmedArticle"):
                row = self._parse_single_pubmed_article(article)
                if row:
                    rows.append(row)

        self.logger.info(f"Parsed {len(rows)} articles from PubMed XML.")
        return pd.DataFrame(rows)

    def _parse_single_pubmed_article(self, article) -> Optional[dict]:
        """Extract canonical fields from a single PubmedArticle XML element."""
        try:
            row = {}
            medline = article.find(".//MedlineCitation")
            if medline is None:
                return None

            # PMID
            pmid_el = medline.find("PMID")
            row["pmid"] = pmid_el.text if pmid_el is not None else ""
            row["external_id"] = row["pmid"]  # traceability: PMID

            # Article info
            art = medline.find("Article")
            if art is None:
                return None

            # Title
            title_el = art.find("ArticleTitle")
            row["title"] = "".join(title_el.itertext()).strip() if title_el is not None else ""

            # Abstract — may have multiple AbstractText elements (structured)
            abstract_parts = []
            for abs_el in art.findall(".//Abstract/AbstractText"):
                label = abs_el.get("Label", "")
                text = "".join(abs_el.itertext()).strip()
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
            row["abstract"] = " ".join(abstract_parts)

            # Year
            pub_date = art.find(".//ArticleDate")
            if pub_date is None:
                pub_date = art.find(".//Journal/JournalIssue/PubDate")
            year = None
            if pub_date is not None:
                year_el = pub_date.find("Year")
                if year_el is not None:
                    year = int(year_el.text)
                else:
                    medline_date = pub_date.find("MedlineDate")
                    if medline_date is not None and medline_date.text:
                        m = re.search(r"(\d{4})", medline_date.text)
                        if m:
                            year = int(m.group(1))
            row["year"] = year

            # Language
            lang_el = art.find("Language")
            row["language"] = lang_el.text if lang_el is not None else ""

            # Document type
            dt_parts = []
            for dt_el in art.findall(".//PublicationTypeList/PublicationType"):
                dt_parts.append(dt_el.text or "")
            row["document_type"] = "; ".join(dt_parts) if dt_parts else ""

            # Authors
            authors_list = []
            authors_full = []
            institutions = []
            countries = []
            for author_el in art.findall(".//AuthorList/Author"):
                last = author_el.findtext("LastName", "")
                fore = author_el.findtext("ForeName", "")
                initials = author_el.findtext("Initials", "")
                collective = author_el.findtext("CollectiveName", "")

                if collective:
                    # Collective author (e.g., "The Cancer Genome Atlas Network")
                    authors_list.append(collective)
                    authors_full.append(collective)
                    institutions.append("")
                    countries.append("")
                elif last:
                    short_name = f"{last} {initials}".strip() if initials else last
                    full_name = f"{last} {fore}".strip() if fore else last
                    authors_list.append(short_name)
                    authors_full.append(full_name)

                    # Affiliation — first AffiliationInfo
                    aff_el = author_el.find(".//AffiliationInfo/Affiliation")
                    aff_text = aff_el.text if aff_el is not None else ""
                    inst, country = self._parse_pubmed_affiliation(aff_text)
                    institutions.append(inst)
                    countries.append(country)

            row["authors_list"] = authors_list
            row["authors_full_name_list"] = authors_full
            row["authors_institution"] = institutions
            row["authors_country"] = countries

            row["first_author"] = authors_list[0] if authors_list else ""
            row["last_author"] = authors_list[-1] if authors_list else ""
            row["first_author_full_name"] = authors_full[0] if authors_full else ""
            row["last_author_full_name"] = authors_full[-1] if authors_full else ""
            row["first_institution"] = institutions[0] if institutions else ""
            row["last_institution"] = institutions[-1] if institutions else ""
            row["first_country"] = countries[0] if countries else ""
            row["last_country"] = countries[-1] if countries else ""

            # DOI
            row["doi"] = ""
            for id_el in article.findall(".//ArticleIdList/ArticleId"):
                if id_el.get("IdType") == "doi":
                    row["doi"] = (id_el.text or "").strip()
                    break

            # Journal
            journal_el = art.find(".//Journal/Title")
            row["source_journal"] = journal_el.text if journal_el is not None else ""

            # Citation count — PubMed doesn't provide this natively
            row["citation_count"] = 0

            # Keywords — from MeSH terms + author keywords
            mesh_terms = []
            for mesh_el in medline.findall(".//MeshHeadingList/MeshHeading/DescriptorName"):
                if mesh_el.text:
                    mesh_terms.append(mesh_el.text)
            row["research_areas"] = "; ".join(mesh_terms)

            kw_list = []
            for kw_el in art.findall(".//KeywordList/Keyword"):
                if kw_el.text:
                    kw_list.append(kw_el.text)
            row["keywords"] = "; ".join(kw_list)

            row["raw_source"] = "PubMed"
            return row

        except Exception as e:
            self.logger.warning(f"Error parsing PubMed article: {e}")
            return None

    def _parse_pubmed_affiliation(self, aff_text: str) -> tuple:
        """Extract institution and country from a PubMed affiliation string."""
        if not aff_text:
            return "", ""

        # PubMed affiliations often end with country names or ISO codes
        # Example: "Department of Chemistry, University of Oxford, Oxford, UK"
        parts = [p.strip() for p in aff_text.split(",")]

        country = ""
        inst = ""

        if parts:
            # Last part is often the country — clean trailing punctuation
            last_part = parts[-1].strip().rstrip(".,;:").strip()
            if not last_part:
                last_part = parts[-2].strip().rstrip(".,;:").strip() if len(parts) > 1 else ""

            # Remove emails that PubMed sometimes appends to affiliations
            last_part = re.sub(r"\s*\S+@\S+\.\S+", "", last_part).strip()

            # 1) Check corrections first (UK, USA, etc.)
            corrected = COUNTRY_CORRECTIONS.get(last_part, last_part)
            if corrected != last_part:
                country = corrected
            # 2) Then check if it's a 2-letter ISO code
            elif len(last_part) == 2 and last_part.isalpha():
                country = _iso_to_name(last_part)
            # 3) Then try to resolve as a country name
            else:
                resolved = _country_to_region(last_part)
                if resolved != "Unknown":
                    country = last_part
                else:
                    # Last part is not a country — try the second-to-last part
                    if len(parts) > 1:
                        prev_part = parts[-2].strip().rstrip(".,;:").strip()
                        corrected_prev = COUNTRY_CORRECTIONS.get(prev_part, prev_part)
                        if corrected_prev != prev_part:
                            country = corrected_prev
                        else:
                            resolved_prev = _country_to_region(prev_part)
                            if resolved_prev != "Unknown":
                                country = prev_part
                            else:
                                country = last_part  # keep as-is
                    else:
                        country = last_part

            # Institution is typically everything before the country part
            inst_parts = parts[:-1] if country else parts
            inst = ", ".join(inst_parts[:2]).strip()

        return inst, country

    # ── Generic CSV upload ────────────────────────────────────────────────────

    def run_generic_csv(self, input_file: str, mapping_key: str = "generic",
                        output_stem: str = "") -> dict:
        """
        Load a generic CSV file and map it to the canonical schema using
        source_mapping[mapping_key] from pipeline_config.yaml.

        Supports two formats (set 'format' in the mapping section):
        - "wide": one row per document, author lists semicolon-separated
        - "long": one row per author (DOI repeated), automatic aggregation
        """
        self.logger.info(f"Starting generic CSV acquisition from: {input_file} (mapping: {mapping_key})")
        self._update_state_status("running", params={"source_type": "Generic CSV",
                                                       "upload_filename": input_file,
                                                       "mapping_key": mapping_key})
        try:
            mapping = self.source_mapping.get(mapping_key)
            if not mapping:
                raise ValueError(
                    f"Mapping key '{mapping_key}' not found in source_mapping "
                    f"in pipeline_config.yaml. Available: {list(self.source_mapping.keys())}"
                )

            fmt = mapping.get("format", "wide")
            self.logger.info(f"Generic CSV format: {fmt}")

            df_raw = self._load_generic_csv(input_file)
            self.logger.info(f"Loaded {len(df_raw)} rows from CSV.")

            if fmt == "long":
                df = self._parse_generic_long(df_raw, mapping)
            else:
                df = self._parse_generic_wide(df_raw, mapping)

            df = self._compute_derived_fields(df, source="Generic")
            df = self._normalize_to_schema(df, source="generic")

            output_file = self._save_output(df, "corpus_raw.csv", output_stem)
            stats = self._build_stats(df, len(df_raw) if fmt == "long" else len(df))
            self._update_state_status("completed", output_file=output_file,
                                      params={"source_type": "Generic CSV",
                                              "upload_filename": input_file,
                                              "mapping_key": mapping_key,
                                              "format": fmt,
                                              **stats})
            self.logger.info(f"Generic CSV acquisition complete. Output: {output_file}")
            return {"status": "completed", "output_file": output_file, **stats}

        except Exception as e:
            self.logger.error(f"Generic CSV acquisition failed: {e}", exc_info=True)
            self._update_state_status("failed", notes=str(e))
            raise

    def _load_generic_csv(self, input_file: str) -> pd.DataFrame:
        """Load a CSV file with flexible encoding detection."""
        path = Path(input_file)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {input_file}")

        try:
            df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1", low_memory=False)

        self.logger.info(f"CSV columns: {list(df.columns)}")
        return df

    def _parse_generic_wide(self, df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
        """Parse a wide-format CSV (one row per document)."""
        out = pd.DataFrame()

        # Map direct fields: canonical_field → csv_column_name
        for canon_field, csv_col in mapping.items():
            if canon_field == "format":
                continue  # config key, not a column mapping
            if csv_col.startswith("_") or csv_col == "":
                continue  # computed or empty
            if csv_col in df.columns:
                out[canon_field] = df[csv_col]
            else:
                self.logger.warning(f"CSV column '{csv_col}' not found — field '{canon_field}' will be empty.")
                out[canon_field] = None

        # Split list fields by semicolon
        list_fields = ["authors_list", "authors_full_name_list",
                       "authors_institution", "authors_country"]
        for field in list_fields:
            col_name = mapping.get(field, "")
            if col_name and col_name in df.columns and field in out.columns:
                out[field] = out[field].apply(
                    lambda x: [item.strip() for item in str(x).split(";") if item.strip()]
                    if pd.notna(x) else []
                )

        # Ensure list fields are lists even if not in mapping
        for field in list_fields:
            if field not in out.columns:
                out[field] = [[] for _ in range(len(out))]

        out["raw_source"] = "Generic"
        return out

    def _parse_generic_long(self, df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
        """Parse a long-format CSV (one row per author) and aggregate by DOI."""
        # Identify the document key (usually DOI)
        doi_col = mapping.get("doi", "doi")
        if doi_col not in df.columns:
            raise ValueError(
                f"Long format requires a DOI column for grouping. "
                f"Column '{doi_col}' not found in CSV. Available: {list(df.columns)}"
            )

        # Identify aggregation columns (author-level fields)
        agg_map = {}  # csv_col → canonical_field
        direct_map = {}  # csv_col → canonical_field (document-level)

        for canon_field, csv_col in mapping.items():
            if canon_field == "format":
                continue  # config key, not a column mapping
            if csv_col.startswith("_") or csv_col == "":
                continue
            if csv_col not in df.columns:
                continue

            # Fields that need aggregation (per-author in long format)
            if canon_field in ["authors_list", "authors_full_name_list",
                               "authors_institution", "authors_country"]:
                agg_map[csv_col] = canon_field
            else:
                direct_map[csv_col] = canon_field

        self.logger.info(f"Long format: {len(agg_map)} aggregation columns, "
                         f"{len(direct_map)} direct columns.")

        # Aggregate
        grouped = df.groupby(doi_col, sort=False)

        rows = []
        for doi_val, group in grouped:
            row = {}

            # Direct fields — take the first value (same across all rows for a DOI)
            for csv_col, canon_field in direct_map.items():
                vals = group[csv_col].dropna()
                row[canon_field] = vals.iloc[0] if len(vals) > 0 else None

            # Aggregated fields — join unique values with ";"
            for csv_col, canon_field in agg_map.items():
                vals = group[csv_col].dropna().astype(str).unique()
                # Split each value by ";" in case it already contains lists
                all_items = []
                for v in vals:
                    all_items.extend([item.strip() for item in v.split(";") if item.strip()])
                # Deduplicate preserving order
                seen = set()
                unique_items = []
                for item in all_items:
                    if item not in seen:
                        seen.add(item)
                        unique_items.append(item)
                row[canon_field] = unique_items

            row["raw_source"] = "Generic"
            rows.append(row)

        self.logger.info(f"Aggregated {len(df)} rows → {len(rows)} documents.")
        return pd.DataFrame(rows)

    # ── WOS parsing ──────────────────────────────────────────────────────────

    # WOS two-letter code → human-readable column name
    WOS_COLUMN_MAPPING = {
        'FN': 'File Name', 'VR': 'Version Number', 'PT': 'Publication Type',
        'AU': 'Authors', 'AF': 'Author Full Name', 'BA': 'Book Authors',
        'BF': 'Book Authors Full Name', 'CA': 'Group Authors', 'GP': 'Book Group Authors',
        'BE': 'Editors', 'TI': 'Document Title', 'SO': 'Publication Name',
        'SE': 'Book Series Title', 'BS': 'Book Series Subtitle', 'LA': 'Language',
        'DT': 'Document Type', 'CT': 'Conference Title', 'CY': 'Conference Date',
        'CL': 'Conference Location', 'SP': 'Conference Sponsors', 'HO': 'Conference Host',
        'DE': 'Author Keywords', 'ID': 'Keywords Plus®', 'AB': 'Abstract',
        'C1': 'Author Address', 'RP': 'Reprint Address', 'EM': 'E-mail Address',
        'RI': 'ResearcherID Number', 'OI': 'ORCID',
        'FU': 'Funding Agency and Grant Number', 'FP': 'Name Of The Funding Agencies',
        'FX': 'Funding Text', 'CR': 'Cited References', 'NR': 'Cited Reference Count',
        'TC': 'WOS Core Collection Times Cited Count', 'Z9': 'Total Times Cited Count',
        'U1': 'Usage Count Last 180 Days', 'U2': 'Usage Count Since 2013',
        'PU': 'Publisher', 'PI': 'Publisher City', 'PA': 'Publisher Address',
        'SN': 'ISSN', 'EI': 'eISSN', 'BN': 'ISBN',
        'J9': '29-Character Source Abbreviation', 'JI': 'ISO Source Abbreviation',
        'PD': 'Publication Date', 'PY': 'Year Published', 'VL': 'Volume',
        'IS': 'Issue', 'SI': 'Special Issue', 'PN': 'Part Number',
        'SU': 'Supplement', 'MA': 'Meeting Abstract', 'BP': 'Beginning Page',
        'EP': 'Ending Page', 'AR': 'Article Number', 'DI': 'DOI',
        'D2': 'Book-DOI', 'PG': 'Page Count', 'P2': 'Book Citation Index',
        'WC': 'WOS Categories', 'SC': 'Research Areas',
        'GA': 'Document Delivery Number', 'UT': 'Accession Number',
        'PM': 'PubMed ID', 'ER': 'End of Record', 'EF': 'End of File',
        'C3': 'Affiliating Institutions', 'DL': 'Link to full text',
        'EA': 'Publication Stage', 'WE': 'WE', 'OA': 'Open Access',
        'HC': 'HC', 'HP': 'HP', 'DA': 'DA',
    }

    def _load_wos_file(self, input_file: str) -> pd.DataFrame:
        """
        Load WOS export file (tab-separated .txt or .csv) and immediately
        rename two-letter WOS codes to human-readable column names.
        After this method, all downstream code uses readable names only.
        """
        path = Path(input_file)
        if not path.exists():
            raise FileNotFoundError(f"WOS file not found: {input_file}")

        # WOS .txt exports are tab-separated; may have a BOM or a "FN Clarivate" header
        if path.suffix.lower() == ".txt":
            try:
                df = pd.read_csv(path, sep="\t", encoding="utf-8-sig", skiprows=0,
                                 on_bad_lines="skip", low_memory=False)
                # If first column still looks like a WOS file-header line, skip it
                if str(df.columns[0]).startswith("FN "):
                    df = pd.read_csv(path, sep="\t", encoding="utf-8-sig", skiprows=1,
                                     on_bad_lines="skip", low_memory=False)
            except Exception:
                df = pd.read_csv(path, sep="\t", encoding="latin-1", skiprows=1,
                                 on_bad_lines="skip", low_memory=False)
        else:
            df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

        # Rename two-letter codes → readable names (only columns present in the file)
        rename_map = {k: v for k, v in self.WOS_COLUMN_MAPPING.items()
                      if k in df.columns}
        if rename_map:
            df = df.rename(columns=rename_map)
            self.logger.info(f"Renamed {len(rename_map)} WOS column codes to readable names.")
        else:
            # File already has readable names (exported via WOS web interface)
            self.logger.info("WOS file already has readable column names — no renaming needed.")

        return df

    @staticmethod
    def _parse_wos_publication_date(pd_str, py_str) -> str:
        """Parse WOS Publication Date (PD) into ISO YYYY-MM-DD.

        Handles the formats observed in WOS exports:
          - "2026 FEB 11"  (year + month + day)
          - "DEC 1"        (month + day; year taken from PY)
          - "JUN"          (month only; year taken from PY)
          - "SEP-OCT"      (month range; start month used)
          - ""             (empty; falls back to PY-01-01)
        """
        MONTHS = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                  'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
        pd_str = str(pd_str or "").strip()
        py_str = str(py_str or "").strip()

        year = int(py_str) if py_str.isdigit() else None
        month = None
        day = None

        # Format 1: "2026 FEB 11"
        m = re.match(r'^(\d{4})\s+([A-Z]{3})(?:\s+(\d{1,2}))?$', pd_str)
        if m:
            year = int(m.group(1))
            month = MONTHS.get(m.group(2))
            day = int(m.group(3)) if m.group(3) else None
        else:
            # Format 2: "DEC 1" or "JUN"
            m2 = re.match(r'^([A-Z]{3})(?:\s+(\d{1,2}))?$', pd_str)
            if m2:
                month = MONTHS.get(m2.group(1))
                day = int(m2.group(2)) if m2.group(2) else None
            else:
                # Format 3: "SEP-OCT" (month range → start month)
                m3 = re.match(r'^([A-Z]{3})-([A-Z]{3})$', pd_str)
                if m3:
                    month = MONTHS.get(m3.group(1))

        if year is None:
            return ""
        month = month or 1
        day = day or 1
        return f"{year:04d}-{month:02d}-{day:02d}"

    def _parse_wos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse WOS-specific columns into intermediate normalized form."""
        m = self.source_mapping.get("wos", {})
        out = pd.DataFrame()

        # Direct columns
        for canon_field, wos_col in m.items():
            if wos_col.startswith("_"):
                continue  # computed later
            if wos_col in df.columns:
                out[canon_field] = df[wos_col]
            else:
                self.logger.warning(f"WOS column '{wos_col}' not found — field '{canon_field}' will be empty.")
                out[canon_field] = None

        # external_id — WOS Accession Number (UT) for traceability
        ut_col = next((c for c in ["Accession Number", "UT"] if c in df.columns), None)
        if ut_col:
            out["external_id"] = df[ut_col].astype(str).str.strip()
        else:
            self.logger.warning("WOS column 'Accession Number' not found — 'external_id' will be empty.")
            out["external_id"] = ""

        # Publication date — parse from PD + PY (PD formats vary across exports)
        pd_col = next((c for c in ["Publication Date", "PD"] if c in df.columns), None)
        py_col = next((c for c in ["Year Published", "PY"] if c in df.columns), None)
        if pd_col:
            out["publication_date"] = df.apply(
                lambda r: self._parse_wos_publication_date(
                    r.get(pd_col), r.get(py_col) if py_col else None),
                axis=1
            )
        else:
            out["publication_date"] = ""

        # Ensure citations come from the correct column and are integers
        if "Total Times Cited Count" in df.columns:
            out["citation_count"] = pd.to_numeric(df["Total Times Cited Count"], errors="coerce").fillna(0).astype(int)
        else:
            self.logger.warning("WOS column 'Total Times Cited Count' not found — 'citation_count' set to 0.")
            out["citation_count"] = 0

        # Authors — split by semicolon
        if "Authors" in df.columns:
            out["authors_list"] = df["Authors"].apply(
                lambda x: [a.strip() for a in str(x).split(";") if a.strip()]
                if pd.notna(x) else []
            )
        if "Author Full Names" in df.columns:
            out["authors_full_name_list"] = df["Author Full Names"].apply(
                lambda x: [a.strip() for a in str(x).split(";") if a.strip()]
                if pd.notna(x) else []
            )
        elif "Author Full Name" in df.columns:
            out["authors_full_name_list"] = df["Author Full Name"].apply(
                lambda x: [a.strip() for a in str(x).split(";") if a.strip()]
                if pd.notna(x) else []
            )

        # Author address parsing
        addr_col = next((c for c in ["Author Addresses", "Author Address", "Addresses"]
                         if c in df.columns), None)
        if addr_col:
            # Inject address string into the temporary output dataframe for row-level alignment
            out["_temp_addr"] = df[addr_col]
            
            def _map_row(row):
                # Use full names list as master; fallback to short names if unavailable
                authors_master = row.get("authors_full_name_list")
                if not isinstance(authors_master, list) or not authors_master:
                    authors_master = row.get("authors_list") or []
                
                addr_str = row["_temp_addr"]
                if addr_str is None or (isinstance(addr_str, float) and pd.isna(addr_str)):
                    addr_str = ""
                else:
                    addr_str = str(addr_str)
                
                if "[" not in addr_str and "]" not in addr_str:
                    inst = _extract_institution_from_block(addr_str)
                    country = _extract_country_from_block(addr_str)
                    mapped_inst = [inst] * len(authors_master)
                    mapped_country = [country] * len(authors_master)
                else:
                    address_blocks = _split_outside_brackets(addr_str, sep=';')
                    author_to_inst = {}
                    author_to_country = {}
                    
                    for block in address_blocks:
                        authors_in_block = _extract_authors_block(block)
                        inst = _extract_institution_from_block(block)
                        country = _extract_country_from_block(block)
                        for a in authors_in_block:
                            key = _normalize_name(a)
                            if key not in author_to_inst:
                                author_to_inst[key] = inst
                            if key not in author_to_country:
                                author_to_country[key] = country
                    
                    mapped_inst = [author_to_inst.get(_normalize_name(a), "Unknown") for a in authors_master]
                    mapped_country = [author_to_country.get(_normalize_name(a), "Unknown") for a in authors_master]
                
                return pd.Series({
                    "authors_institution": mapped_inst,
                    "authors_country": mapped_country,
                    "first_institution": mapped_inst[0] if mapped_inst else "",
                    "last_institution": mapped_inst[-1] if mapped_inst else "",
                    "first_country": mapped_country[0] if mapped_country else "",
                    "last_country": mapped_country[-1] if mapped_country else ""
                })

            # Apply the row-by-row alignment mapping
            aligned_data = out.apply(_map_row, axis=1)
            for col in aligned_data.columns:
                out[col] = aligned_data[col]
                
            out.drop(columns=["_temp_addr"], inplace=True)
        else:
            self.logger.warning("No author address column found in WOS file.")
            for col in ["authors_institution", "authors_country",
                        "first_institution", "last_institution",
                        "first_country", "last_country"]:
                out[col] = None

        out["raw_source"] = "WOS"
        return out


    # ── OpenAlex fetching and parsing ─────────────────────────────────────────

    def _fetch_openalex(self, pyalex, query: str, max_results: int) -> list:
        """Fetch works from OpenAlex API with pagination.

        Uses `step1_openalex.filters` from pipeline_config.yaml when present
        (e.g., title_and_abstract.search, type, date range). Falls back to the
        full-text `search` parameter when no filters are configured.
        """
        per_page = 200  # OpenAlex maximum per page
        records = []
        total_fetched = 0

        filters = self.config.get("step1_openalex", {}).get("filters", {}) or {}

        self.logger.info("Querying OpenAlex API...")
        works = pyalex.Works()
        if filters:
            self.logger.info(f"Applying OpenAlex filters: {filters}")
            works = works.filter(**filters)
        elif query:
            works = works.search(query)
        else:
            raise ValueError(
                "OpenAlex acquisition requires either a --query or "
                "step1_openalex.filters in pipeline_config.yaml"
            )

        works = works.select([
            "id", "doi", "title", "abstract_inverted_index",
            "publication_year", "publication_date", "language", "type",
            "authorships", "cited_by_count", "keywords",
            "primary_location", "topics", "counts_by_year"
        ])

        # Iterate over pages returned by paginate
        for page in works.paginate(per_page=per_page, n_max=max_results):
            records.extend(page)  # Add page items to master list
            total_fetched += len(page)
            self.logger.info(f"  Fetched {total_fetched} records...")

        # Ensure we don't exceed exact max_results (page may return extras)
        return records[:max_results]

    def _parse_openalex(self, records: list) -> pd.DataFrame:
        """Convert OpenAlex JSON records to intermediate DataFrame."""

        rows = []
        for rec in records:
            row = {}

            # Direct fields
            row["doi"] = str(rec.get("doi") or "").replace("https://doi.org/", "")
            row["title"] = rec.get("title") or ""
            row["abstract"] = self._reconstruct_abstract(
                rec.get("abstract_inverted_index"))
            row["year"] = rec.get("publication_year")
            row["publication_date"] = rec.get("publication_date") or ""
            row["language"] = _iso_to_name(rec.get("language"), is_language=True)
            row["document_type"] = str(rec.get("type") or "").title()
            row["citation_count"] = rec.get("cited_by_count") or 0

            # external_id — OpenAlex work ID (e.g., "https://openalex.org/W123...")
            row["external_id"] = str(rec.get("id") or "").replace("https://openalex.org/", "")

            # Keywords
            kw_list = rec.get("keywords") or []
            row["keywords"] = "; ".join(k.get("display_name", "") for k in kw_list)

            # Journal / source
            primary = rec.get("primary_location") or {}
            source = primary.get("source") or {}
            row["source_journal"] = source.get("display_name") or ""
            summary_stats = source.get("summary_stats") or {}
            # Native OpenAlex 2-year mean citedness — same definition as JIF
            row["journal_if"] = summary_stats.get("2yr_mean_citedness")

            # Topics as research areas
            topics = rec.get("topics") or []
            row["research_areas"] = "; ".join(
                t.get("display_name", "") for t in topics[:5]
            )

            # Authorships
            authorships = rec.get("authorships") or []
            auth_names, auth_full, institutions_per_auth, countries_per_auth = \
                self._parse_openalex_authorships(authorships)

            row["authors_list"] = auth_names
            row["authors_full_name_list"] = auth_full
            row["authors_institution"] = institutions_per_auth
            row["authors_country"] = countries_per_auth

            row["first_author"] = auth_names[0] if auth_names else ""
            row["last_author"] = auth_names[-1] if auth_names else ""
            row["first_author_full_name"] = auth_full[0] if auth_full else ""
            row["last_author_full_name"] = auth_full[-1] if auth_full else ""

            # First/last institution and country
            row["first_institution"] = institutions_per_auth[0] if institutions_per_auth else ""
            row["last_institution"] = institutions_per_auth[-1] if institutions_per_auth else ""
            row["first_country"] = countries_per_auth[0] if countries_per_auth else ""
            row["last_country"] = countries_per_auth[-1] if countries_per_auth else ""

            row["raw_source"] = "OpenAlex"
            rows.append(row)

        return pd.DataFrame(rows)

    def _parse_openalex_authorships(self, authorships: list) -> tuple:
        """Unpack OpenAlex authorships list into parallel author/institution/country lists."""
        names, full_names, institutions, countries = [], [], [], []
        for auth in authorships:
            author = auth.get("author") or {}
            name = author.get("display_name") or ""
            names.append(name)
            full_names.append(name)  # OpenAlex does not separate abbreviated/full

            # Option B: Concatenate multiple institutions into a single string
            inst_list = auth.get("institutions") or []
            inst_names = [i.get("display_name", "") for i in inst_list if i.get("display_name")]
            institutions.append("; ".join(inst_names) if inst_names else "Unknown")

            # Option B: Convert ISO codes to full names and concatenate
            country_list = auth.get("countries") or []
            country_names = [_iso_to_name(c) for c in country_list if c]
            countries.append("; ".join(country_names) if country_names else "Unknown")

        return names, full_names, institutions, countries

    def _reconstruct_abstract(self, inverted_index: Optional[dict]) -> str:
        """Reconstruct plain text abstract from OpenAlex inverted index format."""
        if not inverted_index:
            return ""
        try:
            # inverted_index = {"word": [pos1, pos2, ...], ...}
            position_word = {}
            for word, positions in inverted_index.items():
                for pos in positions:
                    position_word[pos] = word
            return " ".join(position_word[i] for i in sorted(position_word))
        except Exception:
            return ""

    # ── Derived fields (common to both sources) ───────────────────────────────

    def _compute_derived_fields(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """Compute fields that are always derived regardless of source."""
        current_year = _reference_year(config=self.config)

        # Ensure document_type is Title Case uniformly (e.g., "Article", "Review")
        if "document_type" in df.columns:
            df["document_type"] = df["document_type"].astype(str).str.title().replace("Nan", "")

        # first_author / last_author from authors_list (if not already set by parser)
        if "authors_list" in df.columns:
            if "first_author" not in df.columns or df["first_author"].isna().all():
                df["first_author"] = df["authors_list"].apply(
                    lambda x: x[0] if isinstance(x, list) and x else "")
            if "last_author" not in df.columns or df["last_author"].isna().all():
                df["last_author"] = df["authors_list"].apply(
                    lambda x: x[-1] if isinstance(x, list) and x else "")
        
        # first_author_full_name / last_author_full_name from authors_full_name_list (if not already set)
        if "authors_full_name_list" in df.columns:
            if "first_author_full_name" not in df.columns or df["first_author_full_name"].isna().all():
                df["first_author_full_name"] = df["authors_full_name_list"].apply(
                    lambda x: x[0] if isinstance(x, list) and x else "")
            if "last_author_full_name" not in df.columns or df["last_author_full_name"].isna().all():
                df["last_author_full_name"] = df["authors_full_name_list"].apply(
                    lambda x: x[-1] if isinstance(x, list) and x else "")

        # first_institution / last_institution from authors_institution (if not already set)
        if "authors_institution" in df.columns:
            if "first_institution" not in df.columns or df["first_institution"].isna().all():
                df["first_institution"] = df["authors_institution"].apply(
                    lambda x: x[0] if isinstance(x, list) and x else "")
            if "last_institution" not in df.columns or df["last_institution"].isna().all():
                df["last_institution"] = df["authors_institution"].apply(
                    lambda x: x[-1] if isinstance(x, list) and x else "")

        # first_country / last_country from authors_country (if not already set)
        if "authors_country" in df.columns:
            if "first_country" not in df.columns or df["first_country"].isna().all():
                df["first_country"] = df["authors_country"].apply(
                    lambda x: x[0] if isinstance(x, list) and x else "")
            if "last_country" not in df.columns or df["last_country"].isna().all():
                df["last_country"] = df["authors_country"].apply(
                    lambda x: x[-1] if isinstance(x, list) and x else "")
                
        # citations_per_year
        if "citation_count" in df.columns and "year" in df.columns:
            def _cpy(row):
                try:
                    count = float(row["citation_count"] or 0)
                    # Same definition as Steps 8 and 9: age counted inclusively
                    # (a document published in the reference year has age 1).
                    age = max(1, current_year - int(row["year"] or current_year) + 1)
                    return round(count / age, 2)
                except Exception:
                    return 0.0
            df["citations_per_year"] = df.apply(_cpy, axis=1)
        else:
            df["citations_per_year"] = 0.0

        # country region
        for prefix in ["first", "last"]:
            country_col = f"{prefix}_country"
            region_col = f"{prefix}_country_region"
            if country_col in df.columns:
                self.logger.info(f"Computing {region_col}...")
                df[region_col] = df[country_col].apply(_country_to_region)
            else:
                df[region_col] = ""

        # journal_if — fill only the missing ones (native values are preserved)
        if "source_journal" in df.columns:
            if "journal_if" not in df.columns:
                df["journal_if"] = None
            missing_mask = df["journal_if"].isna()
            if missing_mask.any():
                self.logger.info("Looking up journal impact factors (this may take a moment)...")
                df.loc[missing_mask, "journal_if"] = df.loc[missing_mask, "source_journal"].apply(_get_impact_factor)

        # JNIF — normalized IF: (IF - mean) / std across the corpus
        if "journal_if" in df.columns:
            if_series = pd.to_numeric(df["journal_if"], errors="coerce")
            mean_if = if_series.mean()
            std_if = if_series.std()
            if std_if and std_if > 0:
                df["jnif"] = ((if_series - mean_if) / std_if).round(4)
            else:
                df["jnif"] = 0.0
        else:
            df["jnif"] = None

        return df

    # ── Schema normalization ──────────────────────────────────────────────────

    def _normalize_to_schema(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """
        Reorder and rename columns to match the canonical schema exactly.
        Any schema field missing from df is added as None.
        Extra columns not in schema are dropped.
        """
        schema_cols = list(self.schema.values())
        # Build output with only schema columns, in schema order
        out = pd.DataFrame()
        for col in schema_cols:
            if col in df.columns:
                out[col] = df[col]
            else:
                self.logger.warning(f"Schema column '{col}' not produced — filling with None.")
                out[col] = None

        # NOTE: list columns (authors_list, authors_institution, etc.) are kept
        # as native Python lists. Pickle preserves them exactly. The CSV copy
        # will serialize them as repr strings — that is acceptable for inspection
        # only; the CSV is never read back by the pipeline.

        return out

    # ── Output and state ──────────────────────────────────────────────────────

    def _save_output(self, df: pd.DataFrame, filename: str,
                     output_stem: str = "") -> str:
        """
        Save DataFrame in two formats:
        - .pkl  (primary) — preserves all Python types (lists, None, float, etc.)
        - .csv  (inspection copy) — human-readable, NOT read back by the pipeline

        output_stem: optional custom stem for the output files (e.g., "corpus_raw_wos").
        When empty, the stem is derived from `filename` (default "corpus_raw").

        Returns the .pkl path, which is what pipeline_state.json records.
        """
        workspace = self.state.get("project", {}).get("workspace_path", ".")
        output_dir_pkl = Path(workspace) / "data" / "02_intermediate"
        output_dir_csv = Path(workspace) / "data" / "01_raw"
        output_dir_pkl.mkdir(parents=True, exist_ok=True)
        output_dir_csv.mkdir(parents=True, exist_ok=True)

        stem = output_stem or Path(filename).stem  # e.g. "corpus_raw"

        # Primary: pickle
        pkl_path = str(output_dir_pkl / f"step1_{stem}.pkl")
        df.to_pickle(pkl_path)
        self.logger.info(f"Saved {len(df)} records to {pkl_path} (pickle)")

        # Inspection copy: CSV (list columns become their repr — for eyes only)
        csv_path = str(output_dir_csv / f"step1_{stem}.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        self.logger.info(f"Saved inspection CSV to {csv_path}")

        return pkl_path  # pipeline_state.json always points to the pkl

    def _build_stats(self, df: pd.DataFrame, raw_count: int) -> dict:
        """Build a stats dict for state and logging."""
        missing_abstract = int(df["abstract"].isna().sum() +
                               (df["abstract"] == "").sum()) \
            if "abstract" in df.columns else 0
        return {
            "records_raw": raw_count,
            "records_output": len(df),
            "records_missing_abstract": missing_abstract,
        }

    def _update_state_status(self, status: str, output_file: str = "",
                             params: dict = None, notes: str = "") -> None:
        """Update pipeline_state.json for step 1."""
        now = datetime.now().isoformat(timespec="seconds")
        step = self.state.setdefault("steps", {}).setdefault("1", {})
        step["status"] = status

        if status == "running":
            step["started_at"] = now
            self.state["current_step"] = 1
        elif status in ("completed", "failed"):
            step["completed_at"] = now

        if output_file:
            # output_file is always the .pkl path
            step.setdefault("output", {})["raw_data_file"] = output_file
            step["output"]["raw_data_file_csv"] = output_file.replace("02_intermediate", "01_raw").replace(".pkl", ".csv")
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


# ── CLI entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AKL Step 1 — Data Acquisition")
    parser.add_argument("source", choices=["wos", "openalex", "pubmed", "arxiv", "generic"],
                        help="Data source to use")
    parser.add_argument("--input", "-i",
                        help="WOS/Generic: path to export file")
    parser.add_argument("--query", "-q",
                        help="OpenAlex/PubMed/arXiv: search query string")
    parser.add_argument("--max-results", "-n", type=int, default=10000,
                        help="OpenAlex/PubMed/arXiv: maximum records to fetch (default: 10000)")
    parser.add_argument("--api-key", "-k",
                        help="PubMed: NCBI API key (raises rate limit to 10 req/s)")
    parser.add_argument("--email", "-e",
                        help="OpenAlex/PubMed/arXiv: email for polite pool")
    parser.add_argument("--mapping", "-m", default="generic",
                        help="Generic CSV: mapping key in pipeline_config.yaml (default: 'generic')")
    parser.add_argument("--output-stem", "-o", default="",
                        help="Custom output stem (e.g., 'corpus_raw_wos'). Default: 'corpus_raw'")
    parser.add_argument("--config", default="pipeline_config.yaml",
                        help="Path to pipeline_config.yaml")
    parser.add_argument("--state", default="pipeline_state.json",
                        help="Path to pipeline_state.json")
    args = parser.parse_args()

    worker = Step1Worker(args.config, args.state)

    if args.source == "wos":
        if not args.input:
            parser.error("--input is required for WOS source")
        result = worker.run_wos(args.input, args.output_stem)
    elif args.source == "pubmed":
        if not args.query:
            parser.error("--query is required for PubMed source")
        result = worker.run_pubmed(args.query, args.max_results, args.api_key,
                                   args.email, args.output_stem)
    elif args.source == "arxiv":
        if not args.query:
            parser.error("--query is required for arXiv source")
        result = worker.run_arxiv(args.query, args.max_results, args.email,
                                  args.output_stem)
    elif args.source == "generic":
        if not args.input:
            parser.error("--input is required for Generic CSV source")
        result = worker.run_generic_csv(args.input, args.mapping, args.output_stem)
    else:  # openalex
        has_filters = bool(worker.config.get("step1_openalex", {}).get("filters"))
        if not args.query and not has_filters:
            parser.error("--query is required for OpenAlex source (or set step1_openalex.filters in pipeline_config.yaml)")
        result = worker.run_openalex(args.query or "", args.max_results,
                                     args.email, args.output_stem)

    print("\n── Step 1 Result ──────────────────────────────")
    for k, v in result.items():
        print(f"  {k}: {v}")