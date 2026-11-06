"""
step3_worker.py
---------------
AKL Pipeline — Step 3: Initial Topic Modeling (BERTopic + EVoC)

Responsibilities:
- Load corpus_clean.pkl produced by Step 2
- Compute or load cached embeddings (with parameter checksum)
- Run BERTopic with EVoC clustering across all layers
- Select the best layer within the target topic range (default: 10-40 topics)
- Name topics using LLM via vLLM local server (OpenAI-compatible API)
- Save topics_preliminary.xlsx, embeddings, model artifacts, and corpus with topic columns
- Update pipeline_state.json

Usage:
    from workers.step3_worker import Step3Worker
    worker = Step3Worker("pipeline_config.yaml", "pipeline_state.json")
    result = worker.run("data/inputs/corpus_clean.pkl")

CLI:
    ./run_worker.sh workers/step3_worker.py \
        --input data/inputs/corpus_clean.pkl \
        --config pipeline_config.yaml \
        --state pipeline_state.json
"""

import hashlib
import io
import json
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yaml

warnings.filterwarnings("ignore")

np.bool = np.bool_  # bertopic/evoc compatibility


# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("step3_worker")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [STEP3] %(levelname)s %(message)s",
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

class Step3Worker:
    """
    Runs BERTopic + EVoC topic modeling on the clean corpus.
    Selects the best EVoC layer within the configured topic range
    and names topics using the local vLLM server.
    """

    def __init__(self, config_path: str, state_path: str,
                 naming_mode: str = "vllm"):
        self.config_path = Path(config_path)
        self.state_path = Path(state_path)
        self.config = self._load_yaml(config_path)
        self.state = self._load_json(state_path)

        workspace = self.state.get("project", {}).get("workspace_path", ".")
        self.workspace = Path(workspace)
        self.logger = _setup_logger(str(self.workspace / "execution.log"))

        # Config sections
        self.cfg_bertopic = self.config.get("step3_bertopic", {})
        self.cfg_vllm = self.config.get("step7_vllm", {})  # shared vLLM config
        self.cfg_models = self.config.get("models", {})

        # Topic naming mode: "vllm" (default) or "agent" (orchestrator LLM names topics)
        self.naming_mode = naming_mode

        # Topic range for layer selection
        self.topic_range_min = self.cfg_bertopic.get("topic_range_min", 10)
        self.topic_range_max = self.cfg_bertopic.get("topic_range_max", 40)
        self.topic_range_center = (self.topic_range_min + self.topic_range_max) / 2

        self.logger.info("Step3Worker initialized.")
        self.logger.info(f"Target topic range: {self.topic_range_min}–{self.topic_range_max}")
        self.logger.info(f"Topic naming mode: {self.naming_mode}")

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, input_file: str) -> dict:
        self.logger.info(f"Starting Step 3 — input: {input_file}")
        self._update_state_status("running", params={"input_file": input_file})

        try:
            # 1. Load corpus
            df = self._load_input(input_file)
            self.logger.info(f"Loaded {len(df)} records.")

            # 2. Build document strings (Title + Abstract)
            docs = self._build_docs(df)

            # 3. Ensure vLLM is running (only needed for vLLM naming mode)
            if self.naming_mode == "vllm":
                self._ensure_vllm_running()
            else:
                self.logger.info("Agent naming mode — skipping vLLM startup.")

            # 4. Build models
            embedding_model, bertopic_model, representation_model = self._build_models()

            # 5. Compute or load cached embeddings
            embeddings = self._get_embeddings(docs, embedding_model)

            # 6. Fit BERTopic
            self.logger.info("Fitting BERTopic + EVoC...")
            topics, probs = bertopic_model.fit_transform(docs, embeddings=embeddings)
            self.logger.info("BERTopic fit complete.")

            # 7. Save model artifacts
            model_path = self._save_model(bertopic_model, embedding_model)

            # 8. Process EVoC layers — select best layer
            best_layer_idx, topic_infos, all_topics_per_layer = \
                self._process_evoc_layers(
                    bertopic_model, docs, embeddings, df, representation_model
                )

            # 9. Save corpus with topic columns
            corpus_path = self._save_corpus_with_topics(df, all_topics_per_layer)

            # 10. Save topics_preliminary.xlsx (best layer)
            topics_path = self._save_topics_xlsx(
                topic_infos, best_layer_idx, all_topics_per_layer, df
            )

            stats = {
                "n_records": len(df),
                "n_layers_found": len(topic_infos),
                "best_layer": best_layer_idx,
                "n_topics_best_layer": len(topic_infos[best_layer_idx])
                    if topic_infos else 0,
                "embedding_model": self.cfg_models.get("embedding", ""),
                "naming_mode": self.naming_mode,
            }

            self._update_state_status(
                "completed",
                output_topics=str(topics_path),
                output_corpus=str(corpus_path),
                output_model=str(model_path),
                params=stats
            )

            self.logger.info(
                f"Step 3 complete. Best layer: {best_layer_idx} "
                f"({stats['n_topics_best_layer']} topics). "
                f"Output: {topics_path}"
            )
            return {"status": "completed", "output_file": str(topics_path), **stats}

        except Exception as e:
            self.logger.error(f"Step 3 failed: {e}", exc_info=True)
            self._update_state_status("failed", notes=str(e))
            raise

    # ── vLLM management ───────────────────────────────────────────────────────

    def _ensure_vllm_running(self):
        """
        Check if vLLM server is reachable. If not, start it via VLLMManager
        and wait until it becomes available.
        """
        base_url = self.cfg_vllm.get("base_url", "http://localhost:8000/v1")
        health_url = base_url.rstrip("/v1").rstrip("/") + "/health"
        timeout_sec = self.cfg_vllm.get("startup_timeout_sec", 180)
        poll_interval = 5

        if self._vllm_is_ready(health_url):
            self.logger.info("vLLM server is already running.")
            return

        self.logger.info("vLLM server not reachable — attempting to start via VLLMManager...")
        try:
            sys.path.insert(0, str(self.workspace))
            from workers.vllm_manager import VLLMManager
            manager = VLLMManager(str(self.config_path))
            manager.start_server()
            self.logger.info("VLLMManager.start_server() called. Waiting for server...")
        except Exception as e:
            raise RuntimeError(
                f"Could not start vLLM server: {e}\n"
                "Please start it manually or check workers/vllm_manager.py."
            )

        # Poll until ready or timeout
        elapsed = 0
        while elapsed < timeout_sec:
            time.sleep(poll_interval)
            elapsed += poll_interval
            if self._vllm_is_ready(health_url):
                self.logger.info(f"vLLM server ready after {elapsed}s.")
                return
            self.logger.info(f"  Waiting for vLLM... ({elapsed}/{timeout_sec}s)")

        raise RuntimeError(
            f"vLLM server did not become ready within {timeout_sec}s. "
            "Check Docker logs for errors."
        )

    def _vllm_is_ready(self, health_url: str) -> bool:
        try:
            r = requests.get(health_url, timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    # ── Model construction ────────────────────────────────────────────────────

    def _build_models(self):
        """Build embedding model, BERTopic, and representation model."""
        from bertopic import BERTopic
        from bertopic.representation import OpenAI as BERTopicOpenAI
        from bertopic.representation import KeyBERTInspired
        from bertopic.dimensionality import BaseDimensionalityReduction
        from sklearn.feature_extraction.text import CountVectorizer
        import openai
        import evoc

        # Embedding model
        embedding_model_name = self.cfg_models.get(
            "embedding",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        embedding_prompt = self.cfg_bertopic.get("embedding_prompt") or None

        self.logger.info(f"Loading embedding model: {embedding_model_name}")
        if embedding_prompt:
            self.logger.info("Using custom embedding prompt from config.")
        # Remote server (embedding_server.enabled) or local SentenceTransformer
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from remote_embedder import build_embedding_model
        embedding_model = build_embedding_model(
            self.config, self.logger, prompt=embedding_prompt)

        # EVoC clustering model
        noise_level = self.cfg_bertopic.get("evoc_noise_level", 0.12)
        evoc_model = evoc.EVoC(noise_level=noise_level, random_state=self.cfg_bertopic.get("evoc_random_state", 42))   # same seed + same embeddings => same layers

        # Dimensionality reduction — disabled (EVoC handles it internally)
        umap_model = BaseDimensionalityReduction()

        # Vectorizer
        vectorizer_model = CountVectorizer(
            ngram_range=(1, 3),
            stop_words="english",
            min_df=self.cfg_bertopic.get("min_df", 5),
        )

        # Representation model — LLM via vLLM (OpenAI-compatible)
        # (only constructed in "vllm" naming mode; "agent" mode uses keywords only)
        kw_model = KeyBERTInspired()

        if self.naming_mode == "agent":
            # Agent naming mode — keyword-only representation (no LLM needed)
            self.logger.info("Agent naming mode — using keyword-only representation.")
            representation_model = {
                "Main": [kw_model],
                "Aspect1": [kw_model],
            }
        else:
            # Representation model — LLM via vLLM (OpenAI-compatible)
            base_url = self.cfg_vllm.get("base_url", "http://localhost:8000/v1")
            llm_model = self.cfg_models.get("llm_extraction", "gemma-3-4b-instruct")

            client = openai.OpenAI(
                api_key="not-needed",   # vLLM local doesn't require a real key
                base_url=base_url,
            )

            nr_docs = self.cfg_bertopic.get("llm_nr_docs", 5)
            llm_representation = BERTopicOpenAI(
                client,
                model=llm_model,
                delay_in_seconds=1,
                chat=True,
                nr_docs=nr_docs,
                diversity=0.3,
                # A few abstracts have thousands of words and overflow the LLM
                # context window; truncate each document sent to the LLM.
                doc_length=self.cfg_bertopic.get("llm_doc_length", 250),
                tokenizer="whitespace",
                prompt=self._build_topic_naming_prompt(),
            )

            representation_model = {
                "Main": [llm_representation],
                "Aspect1": [kw_model],
            }

        # BERTopic
        bertopic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=evoc_model,
            vectorizer_model=vectorizer_model,
            representation_model=representation_model,
            calculate_probabilities=True,
            verbose=True,
            top_n_words=20,
        )

        return embedding_model, bertopic_model, representation_model

    def _build_topic_naming_prompt(self) -> str:
        """
        Generic topic naming prompt, not domain-specific.
        Works for any scientific corpus.
        """
        return """
You will extract a short topic label from given documents and keywords.

Here are two examples of topics you created before:

# Example 1
Sample texts from this topic:
- Traditional diets in most cultures were primarily plant-based with a little meat on top.
- Meat, but especially beef, is the worst food in terms of emissions.

Keywords: meat beef eat eating emissions steak food health processed chicken
topic: Environmental impacts of eating meat

# Example 2
Sample texts from this topic:
- I have ordered the product weeks ago but it still has not arrived!
- It took a month longer to deliver than was advised.

Keywords: deliver weeks product shipping long delivery received arrived arrive week
topic: Shipping and delivery issues

# Your task
Sample texts from this topic:
[DOCUMENTS]

Keywords: [KEYWORDS]

Based on the information above, extract a short topic label (max 45 chars) in the following format:
topic: <topic_label>"""

    # ── Embeddings ────────────────────────────────────────────────────────────

    def _get_embeddings(self, docs: list, embedding_model) -> np.ndarray:
        """
        Return embeddings for docs. Uses cached file if parameters match.
        Recomputes if embedding model or prompt has changed (checksum check).
        """
        checkpoints_dir = self.workspace / "data" / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        emb_path = checkpoints_dir / "step3_embeddings.pkl"
        meta_path = checkpoints_dir / "step3_embeddings_meta.json"

        current_checksum = self._embedding_checksum()

        # Try to load cached embeddings
        if emb_path.exists() and meta_path.exists():
            meta = self._load_json(str(meta_path))
            if meta.get("checksum") == current_checksum and \
               meta.get("n_docs") == len(docs):
                self.logger.info(
                    f"Loading cached embeddings from {emb_path} "
                    f"(checksum match, {len(docs)} docs)."
                )
                return pd.read_pickle(str(emb_path))
            else:
                self.logger.info(
                    "Cached embeddings found but parameters changed "
                    "(model, prompt, or corpus size) — recomputing."
                )

        # Compute embeddings
        self.logger.info(
            f"Computing embeddings for {len(docs)} documents "
            f"(model: {self.cfg_models.get('embedding', '')})..."
        )
        embeddings = embedding_model.encode(docs, show_progress_bar=True)

        # Save with metadata
        pd.to_pickle(embeddings, str(emb_path))
        self._save_json(
            {"checksum": current_checksum, "n_docs": len(docs),
             "computed_at": datetime.now().isoformat(timespec="seconds")},
            meta_path
        )
        self.logger.info(f"Embeddings saved to {emb_path}")
        return embeddings

    def _embedding_checksum(self) -> str:
        """Checksum of embedding parameters to detect config changes."""
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from remote_embedder import embedding_backend_id
        params = {
            "model": self.cfg_models.get("embedding", ""),
            "prompt": self.cfg_bertopic.get("embedding_prompt") or "",
            "backend": embedding_backend_id(self.config),
        }
        return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()

    # ── EVoC layer processing ─────────────────────────────────────────────────

    def _process_evoc_layers(self, bertopic_model, docs: list,
                             embeddings: np.ndarray, df: pd.DataFrame,
                             representation_model: dict) -> tuple:
        """
        Iterate over EVoC cluster layers, name topics for valid layers,
        and select the best layer within the configured topic range.

        Returns: (best_layer_idx, topic_infos, all_topics_per_layer)
        """
        from sklearn.metrics.pairwise import cosine_similarity

        cluster_layers = bertopic_model.hdbscan_model.cluster_layers_
        self.logger.info(f"EVoC produced {len(cluster_layers)} layers total.")

        topic_infos = []         # topic_info DataFrame per valid layer
        all_topics_per_layer = {}  # layer_idx -> topic assignments array
        valid_layer_indices = [] # original layer indices that were processed

        for layer_idx in range(len(cluster_layers)):
            current_topics = cluster_layers[layer_idx].copy()
            n_topics = len(set(current_topics)) - (1 if -1 in current_topics else 0)

            self.logger.info(f"Layer {layer_idx}: {n_topics} topics.")

            # Skip layers with more than max*3 topics (too granular to name)
            if n_topics > self.topic_range_max * 3:
                self.logger.info(f"  → Skipping (too many topics: {n_topics}).")
                continue

            # Reassign noise points (-1) via cosine similarity to nearest centroid
            if n_topics > 0:
                topics_range = range(len(set(current_topics)) - 1)
                topic_embeddings = np.array([
                    embeddings[current_topics == t].mean(axis=0)
                    for t in topics_range
                    if (current_topics == t).any()
                ])
                if len(topic_embeddings) > 0:
                    similarity = cosine_similarity(embeddings, topic_embeddings)
                    new_topics = similarity.argmax(axis=1)
                    noise_mask = current_topics == -1
                    current_topics[noise_mask] = new_topics[noise_mask]

            # Update BERTopic representation for this layer
            bertopic_model.update_topics(
                docs=docs,
                topics=current_topics,
                representation_model=representation_model,
            )

            topic_info = bertopic_model.get_topic_info()
            pos_idx = len(topic_infos)
            topic_info = self._name_topics_with_llm(
                topic_info, pos_idx, bertopic_model)
            self._save_topic_info_pickle(topic_info, pos_idx)

            topic_infos.append(topic_info)
            all_topics_per_layer[pos_idx] = current_topics
            valid_layer_indices.append(layer_idx)

            self.logger.info(
                f"  → Layer {layer_idx} processed: {len(topic_info)} topics named."
            )

        if not topic_infos:
            raise RuntimeError(
                "No valid EVoC layers found. "
                "Try adjusting topic_range_max in pipeline_config.yaml."
            )

        # Select best layer
        best_idx = self._select_best_layer(topic_infos)
        self.logger.info(
            f"Best layer selected: index {best_idx} "
            f"(original EVoC layer {valid_layer_indices[best_idx]}) "
            f"with {len(topic_infos[best_idx])} topics."
        )

        return best_idx, topic_infos, all_topics_per_layer

    def _select_best_layer(self, topic_infos: list) -> int:
        """
        Select the layer whose topic count is best within [min, max].
        Priority:
          1. Layers within the range → pick the one closest to center
          2. No layer in range → pick the layer closest to center overall
        """
        center = self.topic_range_center

        def distance_to_center(idx):
            return abs(len(topic_infos[idx]) - center)

        # Layers within the configured range
        in_range = [
            i for i, ti in enumerate(topic_infos)
            if self.topic_range_min <= len(ti) <= self.topic_range_max
        ]

        if in_range:
            return min(in_range, key=distance_to_center)

        # Fallback: closest to center regardless of range
        self.logger.warning(
            f"No layer found within [{self.topic_range_min}, {self.topic_range_max}] topics. "
            "Selecting closest layer to center."
        )
        return min(range(len(topic_infos)), key=distance_to_center)

    # ── LLM topic naming ──────────────────────────────────────────────────────

    def _name_topics_with_llm(self, topic_info: pd.DataFrame, layer_idx: int,
                              bertopic_model=None) -> pd.DataFrame:
        """
        Name topics for a layer.
        - "vllm" mode: call the vLLM server (OpenAI-compatible API).
        - "agent" mode: dump keywords + representative docs to a JSON file so the
          orchestrator LLM can name the topics; placeholder names are used for now.
        Adds columns: New_Name, New_Representation.
        """
        if self.naming_mode == "agent":
            return self._dump_naming_input(topic_info, layer_idx, bertopic_model)

        import openai

        base_url = self.cfg_vllm.get("base_url", "http://localhost:8000/v1")
        llm_model = self.cfg_models.get("llm_extraction", "gemma-3-4b-instruct")
        client = openai.OpenAI(api_key="not-needed", base_url=base_url)

        # Prepare input table for LLM
        kw_col = "Aspect1" if "Aspect1" in topic_info.columns else "Representation"
        df_input = topic_info[["Topic", kw_col]].copy()
        df_input[kw_col] = df_input[kw_col].apply(
            lambda x: ", ".join(x[:10]) if isinstance(x, list) else str(x)
        )
        table_str = df_input.to_markdown(index=False)

        prompt = f"""
You are an expert research taxonomist. I have a table of research topics
from a bibliometric analysis. Each row has a topic ID and its top keywords.

GOAL: Create unique, specific topic labels (max 45 characters each).

RULES:
1. Labels must be derived strictly from the provided keywords.
2. Do NOT introduce concepts absent from the keywords.
3. Be technically specific — avoid generic words like "Study", "Analysis", "Research".
4. Each label must be unique and distinguishable from the others.
5. Max 45 characters per label.

DATA:
{table_str}

OUTPUT FORMAT:
Return ONLY a pipe-separated list with two columns: Topic | New_Label
Example:
0 | Electrochemical Impedance Analysis
1 | LiFePO4 Cathode Synthesis
"""

        try:
            response = client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=self.cfg_vllm.get("max_tokens", 512),
            )
            output = response.choices[0].message.content
            refined = pd.read_csv(
                io.StringIO(output), sep="|",
                names=["Topic", "New_Label"], skipinitialspace=True
            )
            refined["Topic"] = pd.to_numeric(refined["Topic"], errors="coerce")
            refined = refined.dropna(subset=["Topic"])
            refined["Topic"] = refined["Topic"].astype(int)
            refined["New_Label"] = refined["New_Label"].str.strip()

            topic_info = topic_info.merge(refined, on="Topic", how="left")
            topic_info["New_Label"] = topic_info["New_Label"].fillna(
                topic_info["Name"])
            topic_info["New_Name"] = (topic_info["Topic"].astype(str)
                                      + "_" + topic_info["New_Label"])
            topic_info["New_Representation"] = topic_info["New_Label"]
            topic_info = topic_info.drop(columns=["New_Label"])

        except Exception as e:
            self.logger.warning(
                f"LLM topic naming failed: {e}. "
                "Using BERTopic default names."
            )
            topic_info["New_Name"] = topic_info["Name"]
            topic_info["New_Representation"] = topic_info["Name"]

        return topic_info

    def _dump_naming_input(self, topic_info: pd.DataFrame, layer_idx: int,
                           bertopic_model=None) -> pd.DataFrame:
        """
        Agent naming mode: dump per-topic keywords and representative document
        titles to data/02_intermediate/step3_naming_input/layer_{idx}.json so the
        orchestrator LLM can generate topic labels. Placeholder names are used
        until the labels are applied (see step3_apply_labels.py).
        """
        out_dir = self.workspace / "data" / "02_intermediate" / "step3_naming_input"
        out_dir.mkdir(parents=True, exist_ok=True)

        kw_col = "Aspect1" if "Aspect1" in topic_info.columns else "Representation"
        records = []
        for _, row in topic_info.iterrows():
            t = int(row["Topic"])
            kws = row[kw_col]
            kws = ", ".join(kws[:10]) if isinstance(kws, list) else str(kws)

            rep_titles = []
            if bertopic_model is not None and t != -1:
                try:
                    for d in bertopic_model.get_representative_docs(t)[:3]:
                        title = str(d).split("\n\nAbstract:")[0]
                        title = title.replace("Title: ", "").strip()
                        rep_titles.append(title[:150])
                except Exception:
                    pass

            records.append({
                "Topic": t,
                "Count": int(row["Count"]),
                "Name": str(row["Name"]),
                "Keywords": kws,
                "RepresentativeTitles": rep_titles,
            })

        path = out_dir / f"layer_{layer_idx}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Naming input dumped to {path} ({len(records)} topics).")

        # Placeholder names — real labels applied later by the orchestrator
        topic_info["New_Name"] = topic_info["Name"]
        topic_info["New_Representation"] = topic_info["Name"]
        return topic_info

    def _save_topic_info_pickle(self, topic_info: pd.DataFrame,
                                layer_idx: int) -> None:
        """Persist the topic_info DataFrame for later label application."""
        out_dir = self.workspace / "data" / "02_intermediate" / "step3_topic_info"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"layer_{layer_idx}.pkl"
        topic_info.to_pickle(str(path))
        self.logger.info(f"Topic info saved to {path}")

    # ── Document preparation ──────────────────────────────────────────────────

    def _build_docs(self, df: pd.DataFrame) -> list:
        """Concatenate Title + Abstract as BERTopic input documents."""
        title_col = "title"
        abstract_col = "abstract"
        if title_col not in df.columns or abstract_col not in df.columns:
            raise ValueError(
                f"Corpus must have '{title_col}' and '{abstract_col}' columns."
            )
        docs = ("Title: " + df[title_col].fillna("").astype(str) +
                "\n\nAbstract: " + df[abstract_col].fillna("").astype(str))
        return docs.tolist()

    # ── Output ────────────────────────────────────────────────────────────────

    def _save_model(self, bertopic_model, embedding_model) -> Path:
        """Save BERTopic model artifacts."""
        model_dir = self.workspace / "data" / "checkpoints" / "step3_bertopic_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from remote_embedder import RemoteEmbedder
        # A remote embedder cannot be serialized: store only the model name
        # (callers pass the embedder again to BERTopic.load).
        save_emb = (self.cfg_models.get("embedding", "")
                    if isinstance(embedding_model, RemoteEmbedder) else embedding_model)
        try:
            bertopic_model.save(
                str(model_dir),
                serialization="safetensors",
                save_ctfidf=True,
                save_embedding_model=save_emb,
            )
            self.logger.info(f"BERTopic model saved to {model_dir}")
        except Exception as e:
            self.logger.warning(f"Model save failed (non-critical): {e}")
        return model_dir

    def _save_corpus_with_topics(self, df: pd.DataFrame,
                                 all_topics_per_layer: dict) -> Path:
        """Save corpus with one topic column per processed layer."""
        out_dir_int = self.workspace / "data" / "02_intermediate"
        out_dir_exp = self.workspace / "data" / "05_exports"
        out_dir_int.mkdir(parents=True, exist_ok=True)
        out_dir_exp.mkdir(parents=True, exist_ok=True)

        df_out = df.copy()
        for layer_idx, topics_array in all_topics_per_layer.items():
            df_out[f"topic_layer_{layer_idx}"] = topics_array

        pkl_path = out_dir_int / "step3_corpus_with_topics.pkl"
        csv_path = out_dir_exp / "step3_corpus_with_topics.csv"
        df_out.to_pickle(str(pkl_path))
        df_out.to_csv(str(csv_path), index=False, encoding="utf-8-sig")
        self.logger.info(f"Corpus with topics saved to {pkl_path}")
        return pkl_path

    def _save_topics_xlsx(self, topic_infos: list, best_layer_idx: int,
                          all_topics_per_layer: dict,
                          df: pd.DataFrame) -> Path:
        """
        Save preliminary topic files:
        - Generates one .xlsx file for each EVoC layer.
        - Names indicate the layer, appending '_selected' to the best layer.
        - Sheet 'Topics': summary of that layer's topics.
        - Sheet 'All_Layers': topic counts per layer for reference.
        - One sheet per topic with its documents, including an empty 'Relevance' column.
        """
        import re
        out_dir = self.workspace / "data" / "05_exports"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Remove old files to avoid confusion on re-runs
        for old_file in out_dir.glob("step3_topics_preliminary_layer_*.xlsx"):
            try:
                old_file.unlink()
            except Exception as e:
                self.logger.warning(f"Could not delete old file {old_file}: {e}")

        # Build the overview sheet once to include in all files
        layers_overview = pd.DataFrame([
            {"Layer": idx, "N_Topics": len(ti),
             "Selected": "★" if idx == best_layer_idx else ""}
            for idx, ti in enumerate(topic_infos)
        ])

        selected_xlsx_path = None

        for layer_idx, topic_info in enumerate(topic_infos):
            topics_array = all_topics_per_layer[layer_idx]
            kw_col = "Aspect1" if "Aspect1" in topic_info.columns else "Representation"

            is_selected = (layer_idx == best_layer_idx)
            suffix = "_selected" if is_selected else ""
            xlsx_path = out_dir / f"step3_topics_preliminary_layer_{layer_idx}{suffix}.xlsx"
            
            if is_selected:
                selected_xlsx_path = xlsx_path

            with pd.ExcelWriter(str(xlsx_path), engine="openpyxl") as writer:
                # Sheet 1: Topics summary
                summary_cols = ["Topic", "Count", "Name", "New_Name", kw_col]
                summary_cols = [c for c in summary_cols if c in topic_info.columns]
                summary = topic_info[summary_cols].copy()
                summary[kw_col] = summary[kw_col].apply(
                    lambda x: ", ".join(x) if isinstance(x, list) else str(x)
                )
                summary.sort_values("Topic", inplace=True)
                summary.to_excel(writer, sheet_name="Topics", index=False)

                # Sheet 2: All layers overview
                layers_overview.to_excel(writer, sheet_name="All_Layers", index=False)

                # Sheets per topic
                df_with_topic = df.copy()
                df_with_topic["_topic"] = topics_array
                name_map = dict(zip(topic_info["Topic"],
                                    topic_info.get("New_Name", topic_info["Name"])))

                for topic_id, grp in df_with_topic.groupby("_topic"):
                    topic_name = name_map.get(topic_id, f"Topic_{topic_id}")
                    clean_name = re.sub(r'[\\/*?:\[\]]', "_", str(topic_name))
                    sheet_label = clean_name[:31]
                    cols_to_save = ["doi", "title", "abstract", "year",
                                    "first_author", "source_journal"]
                    cols_to_save = [c for c in cols_to_save if c in grp.columns]
                    
                    # Prepare the final dataframe for the sheet
                    grp_out = grp[cols_to_save].copy()
                    
                    # Insert the Relevance column at position 0 (first column) with an empty
                    grp_out.insert(0, "Relevance (yes/no)", "")
                    
                    grp_out.to_excel(writer, sheet_name=sheet_label, index=False)

            self.logger.info(f"Layer {layer_idx} topics saved to {xlsx_path}")

        # Return the path to the selected layer so pipeline_state.json tracks it properly
        return selected_xlsx_path

    # ── State management ──────────────────────────────────────────────────────

    def _update_state_status(self, status: str, output_topics: str = "",
                             output_corpus: str = "", output_model: str = "",
                             params: dict = None, notes: str = "") -> None:
        now = datetime.now().isoformat(timespec="seconds")
        step = self.state.setdefault("steps", {}).setdefault("3", {})
        step["status"] = status

        if status == "running":
            step["started_at"] = now
            self.state["current_step"] = 3
        elif status in ("completed", "failed"):
            step["completed_at"] = now

        if output_topics:
            step.setdefault("output", {}).update({
                "topics_file": output_topics,
                "corpus_with_topics": output_corpus,
                "model_file": output_model,
            })
        if params:
            step.setdefault("params_used", {}).update(params)
        if notes:
            step["notes"] = notes

        self.state["project"]["last_updated"] = now
        self._save_json(self.state, self.state_path)

    # ── File helpers ──────────────────────────────────────────────────────────

    def _load_input(self, input_file: str) -> pd.DataFrame:
        path = Path(input_file)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        return pd.read_pickle(str(path)) if path.suffix == ".pkl" \
            else pd.read_csv(str(path), encoding="utf-8-sig", low_memory=False)

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

    parser = argparse.ArgumentParser(description="AKL Step 3 — Topic Modeling")
    parser.add_argument("--input", "-i", default="data/02_intermediate/step2_corpus_clean.pkl")
    parser.add_argument("--config", default="pipeline_config.yaml")
    parser.add_argument("--state", default="pipeline_state.json")
    parser.add_argument("--naming-mode", default="vllm",
                        choices=["vllm", "agent"],
                        help="Topic naming: 'vllm' (server) or 'agent' (orchestrator LLM)")
    args = parser.parse_args()

    worker = Step3Worker(args.config, args.state, args.naming_mode)
    result = worker.run(args.input)

    print("\n── Step 3 Result ──────────────────────────────")
    for k, v in result.items():
        print(f"  {k}: {v}")
