"""
step7_worker.py
---------------
AKL Pipeline — Step 7: Semantic Extraction (LLM + CPU Clustering)

Responsibilities:
- Parse `project_config.md` to find active extraction vectors and their scopes.
- Load corpus (output from Step 5 or 2).
- Apply sentence chunking via spaCy if scope == 'sentence'.
- Use local vLLM asynchronously to extract micro-labels.
- Filter out 'None' responses.
- Embed micro-labels using sentence-transformers (on CPU to save VRAM).
- Cluster embeddings using MiniBatchKMeans.
- Name clusters using vLLM to create Macro-Categories.
- Save relational tidy dataset (`step7_extractions.xlsx` and `.pkl`).
"""

import sys
import json
import logging
import asyncio
import re
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import yaml
from tqdm.asyncio import tqdm as async_tqdm
import openai

# ML/NLP imports
from spacy.lang.en import English
from spacy.language import Language
from sklearn.cluster import MiniBatchKMeans
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

# ── Logging Setup ─────────────────────────────────────────────────────────────

def _setup_logger(workspace_path):
    logger = logging.getLogger("step7_worker")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [STEP7] %(message)s", datefmt="%H:%M:%S")
        fh = logging.FileHandler(Path(workspace_path) / "execution.log", encoding="utf-8")
        fh.setFormatter(fmt)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

# ── spaCy Custom Sentencizer ──────────────────────────────────────────────────

ABBREVS = {"ca", "e.g", "i.e", "fig", "figs", "vs", "et", "al"}

if not Language.has_factory("custom_sentencizer_after"):
    @Language.component("custom_sentencizer_after")
    def custom_sentencizer_after(doc):
        for i, token in enumerate(doc):
            lower = token.text.lower()
            candidate_start_idx = None
            
            if lower in ABBREVS and i + 1 < len(doc) and doc[i + 1].text == ".":
                candidate_start_idx = i + 2
            elif any(lower == f"{a}." for a in ABBREVS):
                candidate_start_idx = i + 1

            if candidate_start_idx is None:
                continue

            j = candidate_start_idx
            while j < len(doc) and doc[j].is_punct:
                j += 1

            if j < len(doc):
                doc[j].is_sent_start = False
        return doc

# ── Main Worker Class ─────────────────────────────────────────────────────────

class Step7Worker:
    def __init__(self, config_path="pipeline_config.yaml", state_path="pipeline_state.json"):
        self.config_path = Path(config_path)
        self.state_path = Path(state_path)
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        with open(self.state_path, "r", encoding="utf-8") as f:
            self.state = json.load(f)
            
        self.workspace = Path(self.state.get("project", {}).get("workspace_path", "."))
        self.logger = _setup_logger(self.workspace)
        
        self.cfg_vllm = self.config.get("step7_vllm", {})
        self.cfg_models = self.config.get("models", {})
        
        # Load spaCy
        self.nlp = English()
        self.nlp.add_pipe("sentencizer")
        self.nlp.add_pipe("custom_sentencizer_after", after="sentencizer")

    def _update_state(self, status, output_dict=None, params=None, notes=""):
        now = datetime.now().isoformat(timespec="seconds")
        step = self.state.setdefault("steps", {}).setdefault("7", {})
        step["status"] = status
        if status == "running":
            step["started_at"] = now
            self.state["current_step"] = 7
        elif status in ["completed", "failed"]:
            step["completed_at"] = now
        
        if output_dict:
            step.setdefault("output", {}).update(output_dict)
        if params:
            step.setdefault("params_used", {}).update(params)
        if notes:
            step["notes"] = notes
            
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)

    def _ensure_vllm_running(self):
        import requests
        import time
        base_url = self.cfg_vllm.get("base_url", "http://localhost:8000/v1")
        health_url = base_url.rstrip("/v1").rstrip("/") + "/health"
        
        try:
            if requests.get(health_url, timeout=3).status_code == 200:
                return
        except:
            pass
            
        self.logger.info("Starting vLLM server via VLLMManager...")
        sys.path.insert(0, str(self.workspace))
        from workers.vllm_manager import VLLMManager
        manager = VLLMManager(str(self.config_path))
        manager.start_server()
        
        elapsed = 0
        while elapsed < 180:
            time.sleep(5)
            elapsed += 5
            try:
                if requests.get(health_url, timeout=3).status_code == 200:
                    self.logger.info("vLLM server ready.")
                    return
            except:
                pass
        raise RuntimeError("vLLM server did not become ready.")

    def _parse_vectors_from_md(self):
        md_path = self.workspace / "project_config.md"
        if not md_path.exists():
            raise FileNotFoundError("project_config.md not found.")
            
        content = md_path.read_text(encoding="utf-8")
        match = re.search(r"## 7\. Semantic extraction vectors.*?(\| Vector \|.*?\n(?:\|.*?\|.*?\n)+)", content, re.DOTALL | re.IGNORECASE)
        
        vectors = []
        if match:
            lines = match.group(1).strip().split('\n')
            for line in lines[2:]:
                parts = [p.strip() for p in line.split('|')[1:-1]]
                if len(parts) >= 4:
                    v_name, v_desc, v_scope, v_active = parts[0], parts[1], parts[2], parts[3]
                    if v_active.lower() == 'yes':
                        vectors.append({"name": v_name, "desc": v_desc, "scope": v_scope.lower()})
        return vectors

    def _load_corpus(self):
        f1 = self.workspace / "data" / "02_intermediate" / "step5_corpus_filtered_relevant.pkl"
        f2 = self.workspace / "data" / "02_intermediate" / "step2_corpus_clean.pkl"
        
        if f1.exists():
            self.logger.info(f"Loaded filtered corpus: {f1.name}")
            return pd.read_pickle(f1)
        elif f2.exists():
            self.logger.info(f"Loaded clean corpus: {f2.name}")
            return pd.read_pickle(f2)
        else:
            raise FileNotFoundError("No valid corpus found in 02_intermediate/.")

    def _prefilter_cues(self):
        """Cue stems for the sentence pre-filter (None = disabled)."""
        cfg = self.config.get("step7_extraction", {}).get("sentence_prefilter", {}) or {}
        if not cfg.get("enabled", False):
            return None
        return [c.lower() for c in cfg.get("cues", [])] or None

    def _chunk_corpus(self, df, scope, cues=None):
        """Split documents into texts to analyse.

        scope == 'abstract': one record per document (abstract truncated to
        `max_abstract_words`, so the prompt fits the server's max_model_len).
        scope == 'sentence': one record per sentence; when `cues` is given,
        sentences without any cue stem are dropped (cheap recall-oriented filter).
        """
        max_words = int(self.config.get("step7_extraction", {}).get("max_abstract_words", 350))
        records = []
        for _, row in df.iterrows():
            doi = row.get("doi", "unknown")
            abstract = str(row.get("abstract", "")).strip()
            title = str(row.get("title", "")).strip()

            if not abstract: continue

            if scope == "abstract":
                abstract = " ".join(abstract.split()[:max_words])
                text = f"Title: {title}\nAbstract: {abstract}"
                records.append({"doi": doi, "original_text": text})
            else:
                doc = self.nlp(abstract)
                for sent in doc.sents:
                    phrase = sent.text.strip()
                    if len(phrase) <= 20:  # skip very short noise
                        continue
                    if cues and not any(c in phrase.lower() for c in cues):
                        continue
                    records.append({"doi": doi, "original_text": phrase})

        return pd.DataFrame(records)

    _NONE_LABELS = {"none", "no challenge", "no direction", "no controversy", "not applicable", "n/a", "no material", ""}

    def _endpoints(self):
        """vLLM endpoints (one or more replicas); unreachable ones are skipped."""
        import requests
        urls = self.cfg_vllm.get("base_urls") or [self.cfg_vllm.get("base_url", "http://localhost:8000/v1")]
        ok = []
        for u in urls:
            try:
                if requests.get(u.rstrip("/") + "/models", timeout=5).status_code == 200:
                    ok.append(u)
                    continue
            except Exception:
                pass
            self.logger.warning(f"vLLM endpoint not reachable, skipping: {u}")
        if not ok:
            raise RuntimeError("No reachable vLLM endpoint.")
        self.logger.info(f"Using {len(ok)} vLLM endpoint(s): {ok}")
        return ok

    def _group_prompt_prefix(self, vectors, scope):
        """Instructions come first and are identical across calls (vLLM prefix caching)."""
        spec = "\n".join(f'- "{v["name"]}": {v["desc"]}' for v in vectors)
        unit = "paper (title and abstract)" if scope == "abstract" else "sentence from a paper abstract"
        example = ", ".join(f'"{v["name"]}": "..."' for v in vectors)
        return f"""You are a research analyst extracting specific information from scientific literature.
For the {unit} given below, fill in each of these fields:
{spec}

Rules:
1. For each field, if the text describes it, write a short, standardized label of 2-5 words. Generalize (e.g., 'degrades after 10 hours' -> 'Poor Stability').
2. If the text does NOT describe a field, its value MUST be exactly "None".
3. Answer ONLY with one JSON object with exactly these keys: {{{example}}}. No other text.

Text to analyze:
"""

    @staticmethod
    def _parse_json_labels(raw, names):
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return None
        if not isinstance(obj, dict):
            return None
        out = {}
        for n in names:
            val = obj.get(n, "None")
            if isinstance(val, list):
                val = val[0] if val else "None"
            val = str(val).strip().strip('"\'')
            out[n] = "None" if val.lower() in ("", "none", "not stated", "unknown", "not specified", "n/a") else val
        return out

    def _extract_with_reuse(self, df_chunked, gvecs, scope):
        """LLM extraction, reusing the labels of an earlier run for texts already processed.

        Config: step7_extraction.reuse_extractions_from = directory with the raw caches of the previous run
        (<scope>.pkl, as written by this worker). A text is reused when the same (doi, text) pair was extracted
        before and the cached labels contain every vector of this pass; only the remaining texts go to the LLM.
        With temperature 0 the outputs are nearly deterministic, but not guaranteed to be identical across batches
        and servers; the reuse counts are logged so the share of reused rows is reported with the results.
        """
        names = [v["name"] for v in gvecs]
        reuse_dir = self.config.get("step7_extraction", {}).get("reuse_extractions_from")
        cache = {}
        if reuse_dir:
            path = self.workspace / reuse_dir / f"{scope}.pkl"
            if path.exists():
                old = pd.read_pickle(path)
                odf, olab = old["df"], old["labels"]
                for doi, text, lab in zip(odf["doi"].astype(str), odf["original_text"].astype(str), olab):
                    if lab and all(n in lab for n in names):
                        cache[(doi, text)] = lab
                self.logger.info(f"Extraction cache for '{scope}': {len(cache)} reusable texts from {path}.")
        keys = list(zip(df_chunked["doi"].astype(str), df_chunked["original_text"].astype(str)))
        labels = [None] * len(keys)
        missing = []
        for i, k in enumerate(keys):
            hit = cache.get(k)
            if hit is not None:
                labels[i] = {n: hit[n] for n in names}
            else:
                missing.append(i)
        self.logger.info(f"Scope '{scope}': {len(keys) - len(missing)} of {len(keys)} texts reused, {len(missing)} to extract.")
        if missing:
            sub = df_chunked.iloc[missing].reset_index(drop=True)
            new = asyncio.run(self._extract_group(sub, gvecs, scope))
            for i, lab in zip(missing, new):
                labels[i] = lab
        self._reuse_stats = getattr(self, "_reuse_stats", {})
        self._reuse_stats[scope] = {"texts": len(keys), "reused": len(keys) - len(missing), "extracted": len(missing)}
        return labels

    async def _extract_group(self, df_chunked, vectors, scope, progress_name=None):
        """One LLM call per text, returning every vector of the group (list of dicts aligned to df_chunked rows)."""
        urls = self._endpoints()
        model_name = self.cfg_models.get("llm_extraction", "gemma-3-4b-instruct")
        clients = [openai.AsyncOpenAI(api_key="not-needed", base_url=u) for u in urls]
        sem = asyncio.Semaphore(self.cfg_vllm.get("batch_size", 32) * len(urls))
        names = [v["name"] for v in vectors]
        prefix = self._group_prompt_prefix(vectors, scope)
        max_tok = min(400, 30 * len(names) + 20)

        async def fetch(pos, text):
            client = clients[pos % len(clients)]
            async with sem:
                for attempt in range(2):
                    try:
                        kwargs = {"response_format": {"type": "json_object"}} if attempt == 0 else {}
                        resp = await client.chat.completions.create(
                            model=model_name,
                            messages=[{"role": "user", "content": prefix + f'"{text}"'}],
                            temperature=0.0, max_tokens=max_tok, **kwargs)
                        parsed = self._parse_json_labels(resp.choices[0].message.content or "", names)
                        if parsed is not None:
                            return pos, parsed
                    except Exception:
                        pass
                return pos, {n: "Extraction Error" for n in names}

        tasks = [fetch(i, t) for i, t in enumerate(df_chunked["original_text"].tolist())]
        results = [None] * len(tasks)
        done = 0
        log_every = max(1, len(tasks) // 10)
        label = progress_name or "+".join(names)
        for f in async_tqdm.as_completed(tasks, total=len(tasks), desc=f"Extracting [{scope}]",
                                         disable=not sys.stderr.isatty()):
            pos, parsed = await f
            results[pos] = parsed
            done += 1
            if done % log_every == 0:
                self.logger.info(f"  [{scope}] {done}/{len(tasks)} ({100 * done // len(tasks)}%) — {label}")
        return results

    def _cluster_and_name_macro(self, df_extracted, vector_name):
        # 1. Filter out invalid/none responses
        invalid_labels = ["none", "no challenge", "no direction", "no controversy", 
                          "not applicable", "n/a", "extraction error", "no material"]
        
        df_valid = df_extracted[
            ~df_extracted["micro_label"].str.lower().isin(invalid_labels) & 
            (df_extracted["micro_label"].str.len() > 3)
        ].copy()
        
        if df_valid.empty:
            self.logger.warning(f"No valid labels found for {vector_name}.")
            return df_valid
            
        # 2. Closed taxonomies (few distinct labels, e.g. validation, data_source, readiness):
        #    the normalized label IS the category; clustering words would only blur it.
        df_valid["_norm"] = df_valid["micro_label"].str.lower().str.strip().str.rstrip(".")
        max_closed = int(self.config.get("step7_extraction", {}).get("closed_taxonomy_max_labels", 25))
        norm_counts = df_valid["_norm"].value_counts()
        # a label only counts when it is a real category (>= 0.5% of rows), the rest is noise
        main = norm_counts[norm_counts >= max(2, 0.005 * len(df_valid))]
        if len(main) <= max_closed and main.sum() >= 0.9 * len(df_valid):
            self.logger.info(f"{vector_name}: closed taxonomy ({len(main)} categories); using labels as categories.")
            df_valid = df_valid[df_valid["_norm"].isin(main.index)].copy()
            df_valid["macro_category"] = df_valid["_norm"].str.title()
            return df_valid.drop(columns=["_norm"])

        # 3. Embeddings (on CPU)
        emb_model_name = self.cfg_models.get("embedding", "sentence-transformers/all-MiniLM-L6-v2")
        self.logger.info(f"Loading {emb_model_name} on CPU for clustering...")
        model = SentenceTransformer(emb_model_name, device="cpu")

        unique_labels = df_valid["_norm"].unique().tolist()
        self.logger.info(f"Encoding {len(unique_labels)} unique labels for {vector_name}...")
        embeddings = model.encode(unique_labels, show_progress_bar=False)
        del model

        # 4. K-Means
        n_clusters = min(15, max(3, len(unique_labels) // 15))
        self.logger.info(f"Clustering into {n_clusters} macro-categories...")
        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=256)
        cluster_ids = km.fit_predict(embeddings)

        label_to_cluster = dict(zip(unique_labels, cluster_ids))
        df_valid["cluster_id"] = df_valid["_norm"].map(label_to_cluster)

        # 5. Name clusters via vLLM, showing the most frequent labels (not word fragments)
        self.logger.info("Naming macro-categories via vLLM...")
        urls = self._endpoints()
        llm_model = self.cfg_models.get("llm_extraction", "gemma-3-4b-instruct")
        clients = [openai.OpenAI(api_key="not-needed", base_url=u, timeout=60, max_retries=2) for u in urls]

        cluster_to_macro = {}
        for c in range(n_clusters):
            cluster_points = df_valid[df_valid["cluster_id"] == c]
            top_labels = cluster_points["_norm"].value_counts().head(15).index.tolist()
            labels_str = "; ".join(top_labels)

            prompt = f"""You are an expert scientific researcher.
Below are the most frequent labels of one cluster of **{vector_name}** extracted from chemistry and materials science papers.
Labels: {labels_str}

Write a short, descriptive category name (2-5 words) that covers them. Respond with ONLY the name and nothing else."""

            macro_name = None
            for client in clients:
                try:
                    resp = client.chat.completions.create(
                        model=llm_model, messages=[{"role": "user", "content": prompt}],
                        temperature=0.1, max_tokens=20)
                    macro_name = resp.choices[0].message.content.strip().replace('"', '').replace("'", "")
                    if macro_name:
                        break
                except Exception as e:
                    self.logger.warning(f"Naming failed for {vector_name} cluster {c} on {client.base_url}: {str(e)[:80]}")
            if not macro_name:
                macro_name = f"Cluster {c} ({top_labels[0]})"
            cluster_to_macro[c] = macro_name

        df_valid["macro_category"] = df_valid["cluster_id"].map(cluster_to_macro)
        df_valid.drop(columns=["cluster_id", "_norm"], inplace=True)

        return df_valid

    def recluster(self):
        """Recompute categories for every existing checkpoint (no LLM extraction), then rebuild the master outputs."""
        self._endpoints()  # fail early (VPN/servers down) instead of saving unnamed clusters
        for f in sorted(self._checkpoint_dir().glob("*.pkl")):
            df = pd.read_pickle(f)
            if "vector" not in df.columns or df.empty:
                continue
            v = df["vector"].iloc[0]; scope = df["scope"].iloc[0]
            self.logger.info(f"--- Re-clustering vector: {v} ---")
            base = df[["doi", "original_text", "micro_label"]].copy()
            out = self._cluster_and_name_macro(base, v)
            out["vector"] = v; out["scope"] = scope
            self._save_checkpoint(v, out)
        return self.run()

    # ── Checkpoint helpers (resume-safe long runs) ─────────────────────────────

    def _checkpoint_dir(self):
        d = self.workspace / "data" / "02_intermediate" / "step7_checkpoints"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _load_checkpoints(self):
        """Load per-vector checkpoints produced by previous (possibly killed) runs.

        Returns {vector_name: DataFrame} for every valid checkpoint found.
        """
        checkpoints = {}
        ckpt_dir = self.workspace / "data" / "02_intermediate" / "step7_checkpoints"
        if ckpt_dir.exists():
            for f in sorted(ckpt_dir.glob("*.pkl")):
                try:
                    df = pd.read_pickle(f)
                    if "vector" in df.columns and len(df) > 0:
                        checkpoints[df["vector"].iloc[0]] = df
                except Exception as e:
                    self.logger.warning(f"Could not load checkpoint {f.name}: {e}")
        if checkpoints:
            self.logger.info(
                f"Loaded {len(checkpoints)} existing checkpoint(s): {sorted(checkpoints.keys())}"
            )
        return checkpoints

    def _save_checkpoint(self, vector_name, df):
        """Persist one vector's extraction and mirror progress in the state file."""
        path = self._checkpoint_dir() / f"{vector_name}.pkl"
        df.to_pickle(str(path))
        self.logger.info(f"Checkpoint saved: {path} (rows: {len(df)})")

        step = self.state.setdefault("steps", {}).setdefault("7", {})
        pu = step.setdefault("params_used", {})
        vectors_done = list(pu.get("vectors_extracted", []) or [])
        if vector_name not in vectors_done:
            vectors_done.append(vector_name)
        pu["vectors_extracted"] = vectors_done
        pu["checkpoint_last"] = vector_name
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)

    def run_pilot(self, n_docs=300, n_compare=100):
        """Measure throughput and quality of the grouped extraction on a small sample.

        No checkpoints and no state changes. Reports: LLM calls and time per scope,
        extrapolation to the full corpus, how many positives the sentence pre-filter
        would lose (recall), and None/non-None agreement between the grouped call and
        the old one-call-per-vector approach on `n_compare` abstracts.
        """
        import time
        vectors = self._parse_vectors_from_md()
        df_corpus = self._load_corpus()
        n_total = len(df_corpus)
        sample = df_corpus.sample(min(n_docs, n_total), random_state=7)
        self._ensure_vllm_running()
        cues = self._prefilter_cues()
        none = self._NONE_LABELS
        is_pos = lambda x: str(x).strip().lower() not in none and len(str(x).strip()) > 3 and str(x) != "Extraction Error"
        lines = [f"# Step 7 pilot\nSample: {len(sample)} documents (of {n_total}).\n"]
        summary = {}

        for scope in ("abstract", "sentence"):
            gvecs = [v for v in vectors if v["scope"] == scope]
            if not gvecs:
                continue
            names = [v["name"] for v in gvecs]
            df_c = self._chunk_corpus(sample, scope)          # no pre-filter: needed to measure its recall
            t0 = time.time()
            labels = asyncio.run(self._extract_group(df_c, gvecs, scope))
            dt = time.time() - t0
            per_doc = dt / len(sample)
            err = sum(1 for r in labels if any(v == "Extraction Error" for v in r.values()))
            lines.append(f"## Scope: {scope}\n- Vectors ({len(names)}): {', '.join(names)}\n- Texts: {len(df_c)} | time: {dt:.0f}s | "
                         f"{per_doc:.2f} s/doc -> full corpus estimate: **{per_doc * n_total / 3600:.1f} h**\n- Extraction errors: {err}\n")
            summary[scope] = {"texts": len(df_c), "seconds": round(dt), "est_hours_full": round(per_doc * n_total / 3600, 1)}
            lines.append("| Vector | Positive rate | Example labels |\n|---|---|---|")
            for n in names:
                pos = [r[n] for r in labels if is_pos(r[n])]
                ex = "; ".join(pd.Series(pos).value_counts().head(4).index) if pos else "-"
                lines.append(f"| {n} | {len(pos) / max(1, len(labels)):.1%} | {ex} |")
                dist = pd.Series([str(x).lower() for x in pos]).value_counts().head(8)
                summary[f"top_{n}"] = {k: int(v) for k, v in dist.items()}
            lines.append("")

            if scope == "sentence" and cues:
                texts = df_c["original_text"].str.lower().tolist()
                passes = [any(c in t for c in cues) for t in texts]
                kept = sum(passes) / max(1, len(passes))
                lines.append(f"### Pre-filter ({len(cues)} cues)\n- Sentences kept: **{kept:.1%}** (LLM work reduced to this fraction)")
                for n in names:
                    posi = [i for i, r in enumerate(labels) if is_pos(r[n])]
                    rec = sum(passes[i] for i in posi) / max(1, len(posi))
                    lines.append(f"- Recall of '{n}' positives retained: **{rec:.1%}** ({len(posi)} positives)")
                    summary[f"prefilter_recall_{n}"] = round(rec, 3)
                lines.append("")
                summary["prefilter_kept"] = round(kept, 3)
                est = per_doc * n_total * kept / 3600
                lines.append(f"- Sentence pass estimate with pre-filter: **{est:.1f} h**\n")

            if scope == "abstract":
                sub = df_c.head(n_compare)
                urls = self._endpoints()
                model = self.cfg_models.get("llm_extraction", "")
                client = openai.AsyncOpenAI(api_key="not-needed", base_url=urls[0])

                async def single(desc, text):
                    prompt = (f"You are a research analyst extracting specific information from scientific literature.\n"
                              f"Your task is to identify: **{desc}**\n\nInstructions:\n1. Read the following text carefully.\n"
                              f"2. If the text describes the target information, summarize it into a short, standardized label of 2-5 words.\n"
                              f'3. If the text does NOT describe the target information, you MUST respond with the exact string "None".\n'
                              f"4. Your response must ONLY be the label or \"None\".\n\nText to analyze:\n\"{text}\"\n\nLabel:")
                    r = await client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                                             temperature=0.0, max_tokens=15)
                    return r.choices[0].message.content.strip()

                async def compare():
                    out = {}
                    for v in gvecs:
                        out[v["name"]] = await asyncio.gather(*[single(v["desc"], t) for t in sub["original_text"]])
                    return out
                single_res = asyncio.run(compare())
                lines.append(f"### Agreement with one-call-per-vector ({len(sub)} abstracts)\n| Vector | Same present/absent decision | Identical label |\n|---|---|---|")
                for i, v in enumerate(gvecs):
                    g = [labels[j][v["name"]] for j in range(len(sub))]
                    sg = single_res[v["name"]]
                    same = sum(is_pos(a) == is_pos(b) for a, b in zip(g, sg)) / len(sub)
                    ident = sum(str(a).lower() == str(b).lower() for a, b in zip(g, sg)) / len(sub)
                    lines.append(f"| {v['name']} | {same:.1%} | {ident:.1%} |")
                    summary[f"agreement_{v['name']}"] = round(same, 3)
                lines.append("")

        rep = self.workspace / "data" / "04_reports" / "step7_pilot.md"
        rep.parent.mkdir(parents=True, exist_ok=True)
        rep.write_text("\n".join(lines), encoding="utf-8")
        self.logger.info(f"Pilot report saved to {rep}")
        return {"pilot_report": str(rep), **summary}

    def run(self):
        self.logger.info("Starting Step 7: Semantic Extraction")
        self._update_state("running")
        
        try:
            vectors = self._parse_vectors_from_md()
            if not vectors:
                self.logger.info("No active vectors found in project_config.md. Exiting.")
                self._update_state("completed", notes="No active vectors configured.")
                return {"status": "completed", "extracted_rows": 0}
                
            self.logger.info(f"Found {len(vectors)} active vectors: {[v['name'] for v in vectors]}")
            
            # Resume: load previously completed vectors (survives shell/kill restarts)
            checkpoints = self._load_checkpoints()
            pending = [v for v in vectors if v["name"] not in checkpoints]
            
            # State hygiene: rebuild progress fields from ACTUAL checkpoints on
            # disk (they are the source of truth; the state file may carry stale
            # values from previous runs).
            step = self.state.setdefault("steps", {}).setdefault("7", {})
            pu = step.setdefault("params_used", {})
            done = sorted(checkpoints.keys())
            ckpt_dir = self.workspace / "data" / "02_intermediate" / "step7_checkpoints"
            last_ckpt = max(
                (f for f in ckpt_dir.glob("*.pkl") if f.suffix == ".pkl"),
                key=lambda f: f.stat().st_mtime, default=None
            )
            pu["vectors_extracted"] = done
            pu["checkpoint_last"] = last_ckpt.stem if last_ckpt else ""
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
            
            df_corpus = self._load_corpus()
            
            if not pending:
                self.logger.info("All vectors already processed in previous checkpoints. Rebuilding master outputs only.")
            else:
                self._ensure_vllm_running()
                self.logger.info(f"Pending vectors: {[v['name'] for v in pending]}")
            
            final_dfs = [checkpoints[v["name"]] for v in vectors if v["name"] in checkpoints]
            
            groups = {}
            for v in pending:
                groups.setdefault(v["scope"], []).append(v)

            for scope, gvecs in groups.items():
                gnames = [v["name"] for v in gvecs]
                self.logger.info(f"--- Extraction pass: scope={scope}, vectors={gnames} (one LLM call per text) ---")
                cues = self._prefilter_cues() if scope == "sentence" else None
                if cues:
                    self.logger.info(f"Sentence pre-filter ON ({len(cues)} cue stems).")

                # Raw extraction cache: expensive LLM pass survives a failure in clustering
                raw_dir = self._checkpoint_dir() / "raw"
                raw_dir.mkdir(exist_ok=True)
                raw_path = raw_dir / f"{scope}.pkl"
                df_chunked, raw_labels = None, None
                if raw_path.exists():
                    try:
                        cached = pd.read_pickle(raw_path)
                        if (set(gnames) <= set(cached["vectors"]) and cached["n_docs"] == len(df_corpus)
                                and cached["prefilter"] == bool(cues)):
                            df_chunked, raw_labels = cached["df"], cached["labels"]
                            self.logger.info(f"Reusing cached raw extraction for scope '{scope}' ({len(df_chunked)} texts).")
                    except Exception as e:
                        self.logger.warning(f"Ignoring unreadable raw cache {raw_path.name}: {e}")

                if df_chunked is None:
                    df_chunked = self._chunk_corpus(df_corpus, scope, cues)
                    self.logger.info(f"Chunked into {len(df_chunked)} texts.")
                    raw_labels = self._extract_with_reuse(df_chunked, gvecs, scope)
                    pd.to_pickle({"df": df_chunked, "labels": raw_labels, "vectors": gnames,
                                  "n_docs": len(df_corpus), "prefilter": bool(cues)}, str(raw_path))

                for v in gvecs:
                    v_name = v["name"]
                    self.logger.info(f"--- Clustering vector: {v_name} (Scope: {scope}) ---")
                    df_extracted = df_chunked[["doi", "original_text"]].copy()
                    df_extracted["micro_label"] = [r.get(v_name, "None") for r in raw_labels]

                    df_final_vector = self._cluster_and_name_macro(df_extracted, v_name)
                    df_final_vector["vector"] = v_name
                    df_final_vector["scope"] = scope

                    self._save_checkpoint(v_name, df_final_vector)
                    final_dfs.append(df_final_vector)
                    self.logger.info(f"Finished {v_name}. Kept {len(df_final_vector)} valid insights.")

            if not final_dfs:
                raise RuntimeError("No valid insights extracted across all vectors.")
                
            # Combine all
            df_master = pd.concat(final_dfs, ignore_index=True)
            
            # Reorder columns for Tidy Data format
            cols = ["doi", "vector", "scope", "macro_category", "micro_label", "original_text"]
            df_master = df_master[cols]
            
            # Save
            out_dir_int = self.workspace / "data" / "02_intermediate"
            out_dir_exp = self.workspace / "data" / "05_exports"
            out_dir_int.mkdir(parents=True, exist_ok=True)
            out_dir_exp.mkdir(parents=True, exist_ok=True)
            
            pkl_path = out_dir_int / "step7_extractions.pkl"
            xlsx_path = out_dir_exp / "step7_extractions.xlsx"
            
            df_master.to_pickle(str(pkl_path))
            df_master.to_excel(str(xlsx_path), index=False, engine="openpyxl")
            
            self.logger.info(f"All extractions complete! Total rows: {len(df_master)}")
            
            # Update state
            summary_counts = df_master.groupby("vector").size().to_dict()
            vectors_done = sorted(summary_counts.keys())
            
            self._update_state("completed", 
                output_dict={
                    "extractions_pkl": str(pkl_path),
                    "extractions_xlsx": str(xlsx_path)
                },
                params={
                    "vector_counts": summary_counts,
                    "records_processed": int(len(df_master)),
                    "vectors_extracted": vectors_done,
                    # checkpoint_last intentionally NOT overwritten here:
                    # it is kept as the last checkpoint actually saved during
                    # this (or the most recent) run.
                    "llm_model": self.cfg_models.get("llm_extraction", ""),
                    "vllm_base_url": self.cfg_vllm.get("base_url", ""),
                    "chunk_unit": ",".join(sorted({v["scope"] for v in vectors})),
                    "batch_size": self.cfg_vllm.get("batch_size", 0),
                    "extraction_reuse": getattr(self, "_reuse_stats", {})
                }
            )
            
            return {"status": "completed", "vector_counts": summary_counts}

        except Exception as e:
            self.logger.error(f"Step 7 failed: {e}", exc_info=True)
            self._update_state("failed", notes=str(e))
            raise

if __name__ == "__main__":
    worker = Step7Worker()
    if len(sys.argv) > 1 and sys.argv[1] == "recluster":
        print(json.dumps(worker.recluster()))
    elif len(sys.argv) > 1 and sys.argv[1] == "pilot":
        print(json.dumps(worker.run_pilot(int(sys.argv[2]) if len(sys.argv) > 2 else 300)))
    else:
        print(json.dumps(worker.run()))
