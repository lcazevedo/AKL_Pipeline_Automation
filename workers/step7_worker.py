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
import os
import random
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
        self.cfg_repro = self.config.get("reproducibility", {}) or {}

        # Reproducibility: step-level seed defaults to the global seed. It is sent
        # to vLLM and used by the MiniBatchKMeans clustering.
        # NOTE: even with a fixed seed, vLLM batched decoding may not be
        # bit-identical across runs, so the LLM text output is not guaranteed to
        # be deterministic; only the local clustering is.
        self.global_seed = self.cfg_repro.get("seed", 42)
        self.seed = self._resolve_seed(self.cfg_vllm.get("seed"))
        self.logger.info(f"Random seed: {self.seed}")
        
        self.discard_counts = {}

        # Load spaCy
        self.nlp = English()
        self.nlp.add_pipe("sentencizer")
        self.nlp.add_pipe("custom_sentencizer_after", after="sentencizer")

    def _resolve_seed(self, value):
        """Resolve a step-level seed. `None` inherits the global seed."""
        if value is None:
            value = self.global_seed
        try:
            return int(value)
        except (TypeError, ValueError):
            return 42

    def _seed_everything(self):
        """Seed Python/NumPy/Torch before local (CPU) stochastic operations."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        try:
            import torch
            torch.manual_seed(self.seed)
        except Exception:
            pass

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

    def _chunk_corpus(self, df, scope):
        records = []
        for _, row in df.iterrows():
            doi = row.get("doi", "unknown")
            abstract = str(row.get("abstract", "")).strip()
            title = str(row.get("title", "")).strip()
            
            if not abstract: continue
                
            if scope == "abstract":
                text = f"Title: {title}\nAbstract: {abstract}"
                records.append({"doi": doi, "original_text": text})
            else:
                doc = self.nlp(abstract)
                for sent in doc.sents:
                    phrase = sent.text.strip()
                    if len(phrase) > 20: # skip very short noise
                        records.append({"doi": doi, "original_text": phrase})
                        
        return pd.DataFrame(records)

    async def _extract_micro_labels(self, df_chunked, vector_name, vector_desc, scope):
        base_url = self.cfg_vllm.get("base_url", "http://localhost:8000/v1")
        model_name = self.cfg_models.get("llm_extraction", "gemma-3-4b-instruct")
        batch_size = self.cfg_vllm.get("batch_size", 32)
        
        client = openai.AsyncOpenAI(api_key="not-needed", base_url=base_url)
        sem = asyncio.Semaphore(batch_size)
        
        prompt_template = f"""You are a research analyst extracting specific information from scientific literature.
Your task is to identify: **{vector_desc}**

Instructions:
1. Read the following text carefully.
2. If the text describes the target information, summarize it into a short, standardized label of 2-5 words.
3. Crucially, try to generalize the label (e.g., instead of 'degrades after 10 hours', prefer 'Poor Stability').
4. If the text does NOT describe the target information, you MUST respond with the exact string "None".
5. Your response must ONLY be the label or "None", with no additional text or quotes.

Text to analyze:
"{{text}}"

Label:"""

        async def fetch(idx, text):
            async with sem:
                prompt = prompt_template.format(text=text)
                try:
                    resp = await client.chat.completions.create(
                        model=model_name,
                        messages=[{'role': 'user', 'content': prompt}],
                        temperature=0.0,
                        seed=self.seed,
                        max_tokens=15
                    )
                    label = resp.choices[0].message.content.strip()
                except Exception as e:
                    label = "Extraction Error"
                return idx, label

        tasks = [fetch(i, row["original_text"]) for i, row in df_chunked.iterrows()]
        results = []
        
        # Progress tracking strategy:
        # - tqdm's in-place bar (\r redraw) only works on a real TTY. When the
        #   worker is launched in the background (orchestrator pattern), stderr
        #   is not a terminal, so the bar would degrade into raw \r noise.
        # - We therefore disable tqdm outside a TTY and emit clean, loggable
        #   progress lines every 10% of the batch instead. These lines are
        #   picked up by the orchestrator's polling loop and relayed to the
        #   specialist in real time.
        processed = 0
        log_every = max(1, len(tasks) // 10)

        for f in async_tqdm.as_completed(tasks, total=len(tasks),
                                         desc=f"Extracting {vector_name}",
                                         disable=not sys.stderr.isatty()):
            results.append(await f)
            processed += 1
            if processed % log_every == 0:
                self.logger.info(f"  {vector_name}: {processed}/{len(tasks)} ({100 * processed // len(tasks)}%)")
            
        df_chunked["micro_label"] = ""
        for idx, label in results:
            df_chunked.at[idx, "micro_label"] = label
            
        return df_chunked

    def _cluster_and_name_macro(self, df_extracted, vector_name):
        # 1. Filter out invalid/none responses
        # Answers meaning "nothing to extract". Labels are normalised (case,
        # surrounding quotes/punctuation, plural) before matching, so variants such
        # as "None.", "\"None\"" or "No challenges" are also removed. Matching is
        # exact on the normalised string: genuine labels that merely start with
        # "No ..." (e.g. "No standardized benchmarks") are kept.
        invalid_labels = {
            "none", "no", "n/a", "na", "null", "not applicable", "not mentioned",
            "not specified", "not described", "not stated", "no information",
            "extraction error", "no material", "no challenge", "no direction",
            "no research direction", "no future direction", "no controversy",
            "no debate", "no tool", "no ai tool", "no phase", "no label",
        }

        def _normalise(label):
            s = str(label).strip().strip('"\'`*').strip().rstrip(".!:;").strip().lower()
            s = re.sub(r"\s+", " ", s)
            if s.endswith("ies") and s[:-3] + "y" in invalid_labels:
                s = s[:-3] + "y"
            elif s.endswith("s") and s[:-1] in invalid_labels:
                s = s[:-1]
            return s

        norm = df_extracted["micro_label"].map(_normalise)
        keep = ~norm.isin(invalid_labels) & (norm.str.len() > 3)
        self.discard_counts[vector_name] = {
            "processed": int(len(df_extracted)),
            "kept": int(keep.sum()),
            "discarded": int((~keep).sum()),
        }
        self.logger.info(
            f"{vector_name}: kept {int(keep.sum())} of {len(df_extracted)} "
            f"({100 * keep.mean():.1f}%); discarded as empty/invalid: {int((~keep).sum())}."
        )
        df_valid = df_extracted[keep].copy()
        
        if df_valid.empty:
            self.logger.warning(f"No valid labels found for {vector_name}.")
            return df_valid
            
        # 2. Embeddings (on CPU)
        emb_model_name = self.cfg_models.get("embedding", "sentence-transformers/all-MiniLM-L6-v2")
        self.logger.info(f"Loading {emb_model_name} on CPU for clustering...")
        model = SentenceTransformer(emb_model_name, device="cpu")
        
        unique_labels = df_valid["micro_label"].unique().tolist()
        self.logger.info(f"Encoding {len(unique_labels)} unique labels for {vector_name}...")
        self._seed_everything()
        embeddings = model.encode(unique_labels, show_progress_bar=False)
        
        # Free memory immediately
        del model
        
        # 3. K-Means (seeded for reproducibility)
        n_clusters = min(15, max(3, len(unique_labels) // 15))
        self.logger.info(f"Clustering into {n_clusters} macro-categories (seed={self.seed})...")
        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=self.seed, batch_size=256)
        cluster_ids = km.fit_predict(embeddings)
        
        label_to_cluster = dict(zip(unique_labels, cluster_ids))
        df_valid["cluster_id"] = df_valid["micro_label"].map(label_to_cluster)
        
        # 4. Name Clusters via vLLM
        self.logger.info("Naming macro-categories via vLLM...")
        base_url = self.cfg_vllm.get("base_url", "http://localhost:8000/v1")
        llm_model = self.cfg_models.get("llm_extraction", "gemma-3-4b-instruct")
        client = openai.OpenAI(api_key="not-needed", base_url=base_url)
        
        cluster_to_macro = {}
        for c in range(n_clusters):
            cluster_points = df_valid[df_valid["cluster_id"] == c]
            top_terms = cluster_points["micro_label"].str.split().explode().value_counts().head(8).index.tolist()
            terms_str = ", ".join(top_terms)
            
            prompt = f"""You are an expert scientific researcher.
Your task is to generate a short, descriptive macro-category name (3-5 words) for a cluster of **{vector_name}** based on a list of keywords.
The keywords are: {terms_str}

Respond with ONLY the class name and nothing else."""
            
            try:
                resp = client.chat.completions.create(
                    model=llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    seed=self.seed,
                    max_tokens=20
                )
                macro_name = resp.choices[0].message.content.strip().replace('"', '').replace("'", "")
            except Exception:
                macro_name = f"Cluster {c} ({top_terms[0]}...)"
                
            cluster_to_macro[c] = macro_name
            
        df_valid["macro_category"] = df_valid["cluster_id"].map(cluster_to_macro)
        df_valid.drop(columns=["cluster_id"], inplace=True)
        
        return df_valid

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
            
            for v in vectors:
                v_name = v["name"]
                if v_name in checkpoints:
                    self.logger.info(f"Skipping '{v_name}' (checkpoint exists).")
                    continue
                    
                v_desc = v["desc"]
                v_scope = v["scope"]
                
                self.logger.info(f"--- Processing Vector: {v_name} (Scope: {v_scope}) ---")
                
                # 1. Chunk
                df_chunked = self._chunk_corpus(df_corpus, v_scope)
                self.logger.info(f"Chunked into {len(df_chunked)} records.")
                
                # 2. Extract
                df_extracted = asyncio.run(self._extract_micro_labels(df_chunked, v_name, v_desc, v_scope))
                
                # 3. Cluster & Name
                df_final_vector = self._cluster_and_name_macro(df_extracted, v_name)
                
                # 4. Add metadata
                df_final_vector["vector"] = v_name
                df_final_vector["scope"] = v_scope
                
                # 5. Persist checkpoint BEFORE appending (resume-safe)
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
                    "seed": self.seed,
                    # processed / kept / discarded per vector (this run only)
                    "extraction_counts": self.discard_counts
                }
            )
            
            return {"status": "completed", "vector_counts": summary_counts}

        except Exception as e:
            self.logger.error(f"Step 7 failed: {e}", exc_info=True)
            self._update_state("failed", notes=str(e))
            raise

if __name__ == "__main__":
    worker = Step7Worker()
    print(json.dumps(worker.run()))
