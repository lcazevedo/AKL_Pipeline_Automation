"""WP2: candidate list of papers that are relevant to the study independently of our query, to check the recall of the query.

Method: take the reference lists of a few well-known reviews and benchmark papers on ML in chemistry and materials
(seed DOIs below; the script prints the resolved title of each seed so you can verify it), keep the references from
2000-2025 whose title mentions both ML and chemistry/materials terms, and check how many are in the corpus.
The result is a CANDIDATE list: the specialists must curate it (delete what is not relevant) before the share found
is interpreted as a recall estimate.

Run from the workspace root (needs OPENALEX_API_KEY in the environment or in .env):
    python tools/known_relevant_candidates.py
Outputs: data/03_review/wp2/known_relevant_candidates.csv
"""
import os, re, sys
from collections import Counter
from pathlib import Path
import pandas as pd
import requests

SEEDS = [
    "10.1038/s41586-018-0337-2",   # Butler et al., Machine learning for molecular and materials science (Nature 2018)
    "10.1038/s41524-019-0221-0",   # Schmidt et al., ML in solid-state materials science (npj Comput. Mater. 2019)
    "10.1038/s41524-017-0056-5",   # Ramprasad et al., Machine learning in materials informatics (npj Comput. Mater. 2017)
    "10.1021/acs.chemrev.1c00107", # Keith et al., Combining ML and computational chemistry (Chem. Rev. 2021)
    "10.1021/acs.chemrev.0c01111", # Unke et al., Machine learning force fields (Chem. Rev. 2021)
    "10.1038/s41524-020-00406-3",  # Dunn et al., Matbench (npj Comput. Mater. 2020)
    "10.1038/s41524-024-01259-w",  # JARVIS-Leaderboard (npj Comput. Mater. 2024)
]
ML = re.compile(r"machine[- ]learning|neural|deep[- ]learning|data[- ]driven|genetic algorithm|evolutionary|random forest|gaussian process|bayesian|"
                r"interatomic potential|force field|graph network|generative|active learning|high[- ]throughput|qspr|surrogate|artificial intelligence|"
                r"support vector|descriptor|learning", re.I)
CHEM = re.compile(r"material|molecul|chemi|catal|alloy|crystal|polymer|perovskite|batter|electrolyte|reaction|synthesis|solid|energy|"
                  r"density functional|dft|atomistic|band gap|adsorption|zeolite|ceramic|oxide|semiconductor|spectr|compound|organic|inorganic|"
                  r"potential energy|quantum|formation energy|thermoelectric|photovolt", re.I)

def key():
    k = os.environ.get("OPENALEX_API_KEY")
    if not k and Path(".env").exists():
        for line in open(".env"):
            m = re.match(r"\s*OPENALEX_API_KEY\s*=\s*(.+)", line)
            if m: k = m.group(1).strip().strip("\"'")
    if not k: sys.exit("OPENALEX_API_KEY not found")
    return k
K = key()
get = lambda params: requests.get("https://api.openalex.org/works", params={**params, "api_key": K}, timeout=90).json()

refs = Counter(); resolved = []
for d in SEEDS:
    r = get({"filter": f"doi:https://doi.org/{d}", "select": "id,doi,title,publication_year,referenced_works"}).get("results", [])
    if not r:
        print(f"  seed NOT RESOLVED: {d}"); continue
    w = r[0]; resolved.append(d); refs.update(x.rsplit("/", 1)[-1] for x in (w.get("referenced_works") or []))
    print(f"  seed {d}: {w['title'][:90]} ({w['publication_year']}) | {len(w.get('referenced_works') or [])} references")
ids = list(refs); rows = []
for i in range(0, len(ids), 50):
    for w in get({"filter": "openalex:" + "|".join(ids[i:i + 50]), "select": "id,doi,title,publication_year,type,cited_by_count", "per-page": 50}).get("results", []):
        rows.append(dict(openalex_id=w["id"].rsplit("/", 1)[-1], doi=(w.get("doi") or "").replace("https://doi.org/", "").lower(), title=w.get("title") or "",
                         year=w.get("publication_year"), type=w.get("type"), cited_by_count=w.get("cited_by_count"), n_seeds_citing=refs[w["id"].rsplit("/", 1)[-1]]))
c = pd.DataFrame(rows)
c = c[(c.year.between(2000, 2025)) & (c.doi != "") & c.title.apply(lambda t: bool(ML.search(t)) and bool(CHEM.search(t))) & c.type.isin(["article", "review", "preprint"])]
corpus = pd.read_pickle("data/02_intermediate/step2_corpus_clean.pkl"); inc = set(corpus.doi.astype(str).str.lower().str.replace("https://doi.org/", "", regex=False))
pred = pd.read_pickle("data/02_intermediate/step5_full_predictions.pkl"); pred["d"] = pred.doi.astype(str).str.lower().str.replace("https://doi.org/", "", regex=False)
pm = pred.drop_duplicates("d").set_index("d")
c["in_clean_corpus"] = c.doi.isin(inc)
c["filter_decision"] = c.doi.map(pm.predicted_relevance).fillna("")
c["proba_yes"] = c.doi.map(pm.proba_yes)
c = c.sort_values(["n_seeds_citing", "cited_by_count"], ascending=False).reset_index(drop=True)
out = Path("data/03_review/wp2"); out.mkdir(parents=True, exist_ok=True)
c.assign(keep_after_curation="").to_csv(out / "known_relevant_candidates.csv", index=False)
n = len(c); k = int(c.in_clean_corpus.sum()); kk = int((c.filter_decision == "yes").sum())
print(f"\n{len(resolved)}/{len(SEEDS)} seeds resolved | {len(refs)} distinct references | {n} candidates (2000-2025, ML + chemistry/materials words in the title)")
print(f"in the clean corpus: {k} ({100*k/max(1,n):.0f}%) | kept by the filter: {kk} ({100*kk/max(1,n):.0f}%)  [uncurated: only a first look]")
print(c.groupby(c.year // 5 * 5).agg(n=("doi", "size"), in_corpus=("in_clean_corpus", "mean")).round(2).to_string())
