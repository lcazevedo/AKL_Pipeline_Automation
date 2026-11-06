"""
step1b_openalex_enrichment.py
-----------------------------
Enrich the corpus with OpenAlex metadata, one pass over the DOIs (50 per request):

  citations per year (counts_by_year, 2012 onwards), field-weighted citation impact (FWCI),
  normalised citation percentile, retraction flag, reference lists, funders and awards,
  open-access status, document type and the OpenAlex primary topic (domain, field, subfield).

Run after Step 5 (it reads the filtered corpus; Step 2's clean corpus is the fallback):
    ./run_worker.sh workers/step1b_openalex_enrichment.py

Outputs:
    data/02_intermediate/openalex_enrichment.pkl   one row per DOI (light columns)
    data/02_intermediate/openalex_references.pkl   {doi: [OpenAlex work ids]}  (reference lists, large)
    data/04_reports/openalex_enrichment_report.md  coverage summary

Resumable: results are cached in data/02_intermediate/openalex_enrichment_cache.pkl after every block.
The API key is read from OPENALEX_API_KEY (environment or .env) and is never printed.
"""
import os
import re
import sys
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
import yaml

ROOT = Path(os.environ.get("AKL_WORKSPACE") or Path.cwd())   # the workspace (run folder); defaults to the current directory
API = "https://api.openalex.org/works"
SELECT = ("id,doi,publication_year,type,cited_by_count,counts_by_year,fwci,citation_normalized_percentile,"
          "cited_by_percentile_year,is_retracted,referenced_works_count,referenced_works,funders,awards,"
          "open_access,has_fulltext,primary_topic")


def _logger():
    lg = logging.getLogger("step1b_enrichment")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        fmt = logging.Formatter("%(asctime)s [ENRICH] %(message)s", datefmt="%H:%M:%S")
        for h in (logging.FileHandler(ROOT / "execution.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
            h.setFormatter(fmt)
            lg.addHandler(h)
    return lg


def _api_key():
    key = os.environ.get("OPENALEX_API_KEY")
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*OPENALEX_API_KEY\s*=\s*(.+)", line)
            if m:
                key = m.group(1).strip().strip("\"'")
    if not key:
        raise RuntimeError("OPENALEX_API_KEY not found in the environment or in .env")
    return key


def norm_doi(d):
    return str(d).strip().lower().replace("https://doi.org/", "").replace("http://doi.org/", "")


def parse(w):
    """Light record plus reference list for one OpenAlex work."""
    p = w.get("citation_normalized_percentile") or {}
    py = w.get("cited_by_percentile_year") or {}
    pt = w.get("primary_topic") or {}
    short = lambda v: str(v).rsplit("/", 1)[-1] if v else None
    rec = dict(
        doi=norm_doi(w.get("doi") or ""),
        openalex_id=short(w.get("id")),
        type=w.get("type"),
        year=w.get("publication_year"),
        cited_by_count=w.get("cited_by_count"),
        counts_by_year={int(x["year"]): int(x["cited_by_count"]) for x in (w.get("counts_by_year") or [])},
        fwci=w.get("fwci"),
        cit_norm_percentile=p.get("value"),
        top1pct=p.get("is_in_top_1_percent"),
        top10pct=p.get("is_in_top_10_percent"),
        pct_year_min=py.get("min"), pct_year_max=py.get("max"),
        is_retracted=w.get("is_retracted"),
        n_refs=w.get("referenced_works_count"),
        has_fulltext=w.get("has_fulltext"),
        oa_status=(w.get("open_access") or {}).get("oa_status"),
        funders=[(short(f.get("id")), f.get("display_name")) for f in (w.get("funders") or [])],
        n_awards=len(w.get("awards") or []),
        oa_topic_id=short(pt.get("id")), oa_topic=pt.get("display_name"),
        oa_subfield=(pt.get("subfield") or {}).get("display_name"),
        oa_field=(pt.get("field") or {}).get("display_name"),
        oa_domain=(pt.get("domain") or {}).get("display_name"),
    )
    refs = [short(r) for r in (w.get("referenced_works") or [])]
    return rec, refs


def _get(session, url, params, retries=6):
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=90)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(60, 2 ** attempt * (float(r.headers.get("Retry-After", 1)) if r.status_code == 429 else 1)))
                continue
            raise RuntimeError(f"OpenAlex returned HTTP {r.status_code}")
        except requests.RequestException:
            time.sleep(2 ** attempt)
    raise RuntimeError("OpenAlex request failed after retries")


def fetch_batch(args):
    """DOIs containing ',' or '|' break the filter syntax: they are fetched one by one."""
    dois, key = args
    session = requests.Session()
    found = {}
    easy = [d for d in dois if not re.search(r"[,|]", d)]
    hard = [d for d in dois if d not in easy]
    if easy:
        data = _get(session, API, {"filter": "doi:" + "|".join("https://doi.org/" + d for d in easy),
                                   "select": SELECT, "per-page": 50, "api_key": key})
        for w in data.get("results", []):
            rec, refs = parse(w)
            found.setdefault(rec["doi"], (rec, refs))
    for d in hard:
        try:
            w = _get(session, f"{API}/doi:{quote(d, safe='')}", {"select": SELECT, "api_key": key})
            rec, refs = parse(w)
            found.setdefault(rec["doi"], (rec, refs))
        except Exception:
            pass
    return dois, found


def main():
    log = _logger()
    cfg = (yaml.safe_load(open(ROOT / "pipeline_config.yaml", encoding="utf-8")) or {}).get("enrichment_openalex", {})
    if not cfg.get("enabled", True):
        log.info("enrichment_openalex disabled in pipeline_config.yaml; nothing to do.")
        return
    batch, workers, block = int(cfg.get("batch_size", 50)), int(cfg.get("workers", 4)), int(cfg.get("save_every_batches", 40))
    corpus = ROOT / cfg.get("corpus_file", "data/02_intermediate/step5_corpus_filtered_relevant.pkl")
    if not corpus.exists():
        corpus = ROOT / "data/02_intermediate/step2_corpus_clean.pkl"
    df = pd.read_pickle(corpus)
    dois = sorted({norm_doi(d) for d in df["doi"].dropna()})
    log.info(f"Corpus {corpus.name}: {len(dois)} unique DOIs")

    cache_path = ROOT / "data/02_intermediate/openalex_enrichment_cache.pkl"
    cache = pd.read_pickle(cache_path) if cache_path.exists() else {"rec": {}, "refs": {}, "missing": set()}
    todo = [d for d in dois if d not in cache["rec"] and d not in cache["missing"]]
    log.info(f"{len(cache['rec'])} cached, {len(cache['missing'])} known missing, {len(todo)} to fetch")
    key = _api_key()

    batches = [todo[i:i + batch] for i in range(0, len(todo), batch)]
    t0, done = time.time(), 0
    for g in range(0, len(batches), block):
        group = batches[g:g + block]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for asked, found in ex.map(fetch_batch, [(b, key) for b in group]):
                for d, (rec, refs) in found.items():
                    cache["rec"][d] = rec
                    cache["refs"][d] = refs
                cache["missing"].update(d for d in asked if d not in found)
        pd.to_pickle(cache, cache_path)
        done += len(group)
        el = time.time() - t0
        log.info(f"  {done}/{len(batches)} batches ({100 * done // len(batches)}%), {el:.0f}s elapsed, ~{el / done * (len(batches) - done):.0f}s left")

    table = pd.DataFrame(list(cache["rec"].values())).set_index("doi")
    table.to_pickle(ROOT / "data/02_intermediate/openalex_enrichment.pkl")
    pd.to_pickle({d: cache["refs"][d] for d in table.index}, ROOT / "data/02_intermediate/openalex_references.pkl")

    n = len(dois)
    pct = lambda s: f"{100 * s / n:.1f}%"
    cby = table.counts_by_year.map(bool)
    md = [f"# OpenAlex enrichment report\n", f"Corpus: `{corpus.name}`, {n} unique DOIs.\n",
          "| Item | Count | Share of corpus |", "|---|---|---|",
          f"| Found in OpenAlex | {len(table)} | {pct(len(table))} |",
          f"| Not found | {len(cache['missing'])} | {pct(len(cache['missing']))} |",
          f"| With FWCI | {int(table.fwci.notna().sum())} | {pct(table.fwci.notna().sum())} |",
          f"| With citations by year | {int(cby.sum())} | {pct(cby.sum())} |",
          f"| With reference list | {int((table.n_refs.fillna(0) > 0).sum())} | {pct((table.n_refs.fillna(0) > 0).sum())} |",
          f"| With funder information | {int(table.funders.map(bool).sum())} | {pct(table.funders.map(bool).sum())} |",
          f"| Retracted | {int(table.is_retracted.fillna(False).sum())} | {pct(table.is_retracted.fillna(False).sum())} |",
          f"| With OpenAlex primary topic | {int(table.oa_topic.notna().sum())} | {pct(table.oa_topic.notna().sum())} |", "",
          "Open-access status:\n", table.oa_status.value_counts(dropna=False).to_markdown(), "",
          "Citations by year cover 2012 onwards only; FWCI of papers from the last 0-2 years is preliminary.\n"]
    (ROOT / "data/04_reports/openalex_enrichment_report.md").write_text("\n".join(md), encoding="utf-8")
    log.info(f"Done: {len(table)} found, {len(cache['missing'])} not found. Report: data/04_reports/openalex_enrichment_report.md")


if __name__ == "__main__":
    main()
