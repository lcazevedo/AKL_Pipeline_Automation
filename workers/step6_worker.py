"""
step6_worker.py
---------------
AKL Pipeline — Step 6: High-fidelity topic mapping (EVOC)

Responsibilities:
- Load corpus_filtered_relevant.pkl (output from Step 5).
- Recompute/Load embeddings and run BERTopic + EVoC.
- Evaluate each layer using Silhouette, Davies-Bouldin, CVI1 (Standard, sqrt), CVI2 (Adaptive alpha) and Kneedle.
  Definitions follow the case-study scripts: Q = S_norm(min-max) * (1 - DB_norm(min-max)) feeds the Adaptive CVI
  and Kneedle; the Standard CVI uses (S+1)/2. Reliability warnings are written to the report.
- Generate cvi_report.md and kneedle_plot.png.
- Generate 2D Cluster Maps (UMAP + datamapplot) for layers with < 50 topics.
- Check for consensus among metrics.
- Name topics via vLLM and save topics_final_layer_X.xlsx files.
- Mode 'run': If consensus -> mark selected layer and complete. If not -> awaiting_validation.
- Mode 'finalize': Takes user choice, marks selected layer and completes step.
"""

import sys
import json
import logging
import hashlib
import os
import random
import warnings
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import yaml
import requests
import io
import re
from collections import Counter

warnings.filterwarnings("ignore")
np.bool = np.bool_  # Compatibility

# EVoC keys that were removed from the configuration. They are ignored (with a
# warning) so that old configuration files keep working unchanged.
DEPRECATED_EVOC_KEYS = ("min_num_clusters", "next_cluster_size_quantile")

def _require(package: str, install_name: str = None):
    import importlib
    try:
        return importlib.import_module(package)
    except ImportError:
        pip_name = install_name or package
        raise ImportError(f"Required package '{pip_name}' is not installed.\nInstall it with: pip install {pip_name}")

def _setup_logger(workspace_path):
    logger = logging.getLogger("step6_worker")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [STEP6] %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(Path(workspace_path) / "execution.log", encoding="utf-8")
        fh.setFormatter(fmt)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

class Step6Worker:
    def __init__(self, config_path="pipeline_config.yaml", state_path="pipeline_state.json",
                 naming_mode: str = "vllm"):
        self.config_path = Path(config_path)
        self.state_path = Path(state_path)
        # Topic naming mode: "vllm" (default) or "agent" (orchestrator LLM names topics)
        self.naming_mode = naming_mode
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        with open(self.state_path, "r", encoding="utf-8") as f:
            self.state = json.load(f)
            
        self.workspace = Path(self.state.get("project", {}).get("workspace_path", "."))
        self.logger = _setup_logger(self.workspace)
        
        self.cfg_bertopic = self.config.get("step6_evoc", {})
        self.cfg_vllm = self.config.get("step7_vllm", {})
        self.cfg_models = self.config.get("models", {})
        self.cfg_repro = self.config.get("reproducibility", {}) or {}

        # Reproducibility: resolve the effective seed and warn on deprecated EVoC keys.
        self.global_seed = self.cfg_repro.get("seed", 42)
        self.seed = self._resolve_seed(self.cfg_bertopic.get("random_state"))
        self._evoc_params = None
        self._warn_deprecated_evoc_keys("step6_evoc", self.cfg_bertopic)
        self.logger.info(f"Random seed: {self.seed}")

    # ── Reproducibility helpers ───────────────────────────────────────────────

    def _resolve_seed(self, value):
        """Resolve a step-level seed. `None` inherits the global seed."""
        if value is None:
            value = self.global_seed
        try:
            return int(value)
        except (TypeError, ValueError):
            return 42

    def _warn_deprecated_evoc_keys(self, section: str, cfg: dict) -> None:
        """Warn if the config still carries EVoC keys removed from the pipeline."""
        for key in DEPRECATED_EVOC_KEYS:
            if key in cfg:
                self.logger.warning(
                    f"[{section}] configuration key '{key}' is deprecated and ignored; "
                    "use 'max_layers' and/or 'min_similarity_threshold' instead."
                )

    def _seed_everything(self) -> None:
        """Seed Python, NumPy and Torch right before a stochastic operation."""
        os.environ.setdefault("PYTHONHASHSEED", str(self.seed))
        random.seed(self.seed)
        np.random.seed(self.seed)
        try:
            import torch
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        except Exception:
            pass

    def _build_evoc_kwargs(self) -> dict:
        """Build EVoC kwargs from config, guarding against unsupported arguments.

        Only the parameters accepted by the installed EVoC version are passed.
        If `random_state` is not supported, a clear warning is logged (without
        trying to work around it) stating the run will not be reproducible.
        """
        import inspect
        import evoc

        kwargs = {"noise_level": self.cfg_bertopic.get("evoc_noise_level", 0.12)}
        try:
            params = inspect.signature(evoc.EVoC.__init__).parameters
        except (TypeError, ValueError):
            params = {}

        if "random_state" in params:
            kwargs["random_state"] = self.seed
        else:
            self.logger.warning(
                "Installed EVoC version does not accept 'random_state': the clustering "
                "will NOT be reproducible. Upgrade to evoc>=0.3.1 to enable deterministic runs."
            )

        for key in ("max_layers", "min_similarity_threshold"):
            if key in self.cfg_bertopic and self.cfg_bertopic.get(key) is not None:
                if key in params:
                    kwargs[key] = self.cfg_bertopic[key]
                else:
                    self.logger.warning(
                        f"Installed EVoC version does not accept '{key}': ignoring the configured value."
                    )
        return kwargs

    def _collect_evoc_params(self, evoc_model) -> dict:
        try:
            return {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                    for k, v in evoc_model.get_params().items()}
        except Exception:
            return {}

    def _reproducibility_info(self) -> dict:
        """Snapshot of the parameters/versions that affect reproducibility."""
        from importlib.metadata import version as _pkg_version

        def _ver(pkg: str) -> str:
            try:
                return _pkg_version(pkg)
            except Exception:
                return "unknown"

        numba_threads = None
        try:
            import numba
            numba_threads = int(numba.get_num_threads())
        except Exception:
            pass

        device = self.cfg_bertopic.get("embedding_device", "cpu")
        return {
            "seed": self.seed,
            "global_seed": self.global_seed,
            "evoc_version": _ver("evoc"),
            "numba_version": _ver("numba"),
            "numba_num_threads_effective": numba_threads,
            "numba_num_threads_env": os.environ.get("NUMBA_NUM_THREADS"),
            "numpy_version": _ver("numpy"),
            "scikit_learn_version": _ver("scikit-learn"),
            "bertopic_version": _ver("bertopic"),
            "sentence_transformers_version": _ver("sentence-transformers"),
            "torch_version": _ver("torch"),
            "embedding_device": device,
            "evoc_params": self._evoc_params,
        }

    def _update_state(self, status, output_dict=None, params=None, validation=None, notes=""):
        now = datetime.now().isoformat(timespec="seconds")
        step = self.state.setdefault("steps", {}).setdefault("6", {})
        step["status"] = status
        if status == "running":
            step["started_at"] = now
            self.state["current_step"] = 6
        elif status in ["completed", "failed"]:
            step["completed_at"] = now
        
        if output_dict:
            step.setdefault("output", {}).update(output_dict)
        if params:
            step.setdefault("params_used", {}).update(params)
        if validation:
            step.setdefault("validation", {}).update(validation)
        if notes:
            step["notes"] = notes
            
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)

    def _vllm_is_ready(self, health_url: str) -> bool:
        try:
            return requests.get(health_url, timeout=3).status_code == 200
        except:
            return False

    def _ensure_vllm_running(self):
        base_url = self.cfg_vllm.get("base_url", "http://localhost:8000/v1")
        health_url = base_url.rstrip("/v1").rstrip("/") + "/health"
        if self._vllm_is_ready(health_url): return
        
        self.logger.info("Starting vLLM server via VLLMManager...")
        import time
        sys.path.insert(0, str(self.workspace))
        from workers.vllm_manager import VLLMManager
        manager = VLLMManager(str(self.config_path))
        manager.start_server()
        
        elapsed = 0
        while elapsed < 180:
            time.sleep(5)
            elapsed += 5
            if self._vllm_is_ready(health_url):
                self.logger.info("vLLM server ready.")
                return
        raise RuntimeError("vLLM server did not become ready.")

    def run_clustering(self):
        self.logger.info("Starting Step 6: EVoC Topic Mapping (Relevant documents only)...")
        self._update_state("running")
        
        try:
            # 1. Load Data
            input_file = self.workspace / "data" / "02_intermediate" / "step5_corpus_filtered_relevant.pkl"
            if not input_file.exists():
                raise FileNotFoundError(f"Input file not found: {input_file}")
            df = pd.read_pickle(input_file)
            docs = ("Title: " + df["title"].fillna("").astype(str) + "\n\nAbstract: " + df["abstract"].fillna("").astype(str)).tolist()
            
            # 2. Setup VLLM & Models
            if self.naming_mode == "vllm":
                self._ensure_vllm_running()
            else:
                self.logger.info(f"Agent naming mode ({self.naming_mode}) — skipping vLLM server.")
            embedding_model, bertopic_model, representation_model = self._build_models()
            
            # 3. Embeddings
            embeddings = self._get_embeddings(docs, embedding_model)
            
            # 4. Fit BERTopic (seeded right before the stochastic fit/clustering)
            self._seed_everything()
            self.logger.info("Fitting BERTopic + EVoC on relevant corpus...")
            topics, _ = bertopic_model.fit_transform(docs, embeddings=embeddings)
            
            # 5. Process Layers, CVIs, Consensus
            best_layer_idx, consensus_reached, suggested_layers, topic_infos, all_topics_per_layer = self._evaluate_and_name_layers(bertopic_model, docs, embeddings, df, representation_model)
            
            # 5b. Persist per-layer document assignments (reproducibility):
            # used by the verify mode and by run_finalize to re-align the saved
            # model to the layer the specialist selects.
            assignments_path = self.workspace / "data" / "checkpoints" / "step6_layer_assignments.pkl"
            pd.to_pickle(all_topics_per_layer, assignments_path)
            self.logger.info(f"Saved per-layer assignments to {assignments_path}")
            
            # 6. Save Excel files
            self._save_topic_excels(topic_infos, best_layer_idx if consensus_reached else None, all_topics_per_layer, df)
            
            # 7. Generate 2D Maps
            self._generate_cluster_maps(embeddings, all_topics_per_layer, topic_infos, df)

            # 8. Save Model — re-aligned to the selected layer so that the saved
            # model reproduces the exported topic assignments (reviewer
            # validation #12). Without this, the model retains the state of the
            # LAST layer processed by _evaluate_and_name_layers.
            model_dir = self.workspace / "data" / "checkpoints" / "step6_bertopic_model"
            if consensus_reached:
                selected_topics = all_topics_per_layer[best_layer_idx]
                self.logger.info(f"Re-aligning saved model to selected Layer {best_layer_idx}...")
                bertopic_model.update_topics(docs=docs, topics=selected_topics)
                # BERTopic 0.17.4 bug: update_topics never recreates topic
                # embeddings for custom assignments (_update_topic_size overwrites
                # topics_ before the comparison). Call explicitly so the saved
                # model is internally consistent with the selected layer.
                bertopic_model._create_topic_vectors()
            bertopic_model.save(str(model_dir), serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)
            
            # 9. State update
            repro = self._reproducibility_info()
            self.logger.info(f"Step 6 reproducibility: {repro}")
            if consensus_reached:
                self.logger.info(f"Consensus reached! Automatically selecting Layer {best_layer_idx}.")
                final_file = self.workspace / "data" / "05_exports" / f"step6_topics_final_layer_{best_layer_idx}_selected.xlsx"
                self._update_state("completed", 
                    output_dict={"topics_final_file": str(final_file), "model_file": str(model_dir)},
                    params={"best_layer": int(best_layer_idx), "naming_mode": self.naming_mode,
                            "naming_llm": (self.cfg_models.get("llm_agent") or "not recorded") if self.naming_mode == "agent" else self.cfg_models.get("llm_extraction", ""),
                            "reproducibility": repro},
                    validation={"status": "completed", "consensus_reached": True, "suggested_layers": suggested_layers, "validated_at": datetime.now().isoformat()}
                )
                return {"status": "completed", "layer": best_layer_idx}
            else:
                self.logger.info("No consensus among CVIs. Awaiting human validation.")
                self._update_state("awaiting_validation",
                    output_dict={"model_file": str(model_dir)},
                    params={"best_layer": None, "naming_mode": self.naming_mode,
                            "naming_llm": (self.cfg_models.get("llm_agent") or "not recorded") if self.naming_mode == "agent" else self.cfg_models.get("llm_extraction", ""),
                            "reproducibility": repro},
                    validation={"status": "awaiting_validation", "consensus_reached": False, "suggested_layers": suggested_layers}
                )
                return {"status": "awaiting_validation", "suggested": suggested_layers}

        except Exception as e:
            self.logger.error(f"Step 6 failed: {e}", exc_info=True)
            self._update_state("failed", notes=str(e))
            raise

    def run_finalize(self, selected_layer: int):
        self.logger.info(f"Finalizing Step 6 with user-selected Layer: {selected_layer}")
        out_dir = self.workspace / "data" / "05_exports"
        
        target_file = out_dir / f"step6_topics_final_layer_{selected_layer}.xlsx"
        selected_file = out_dir / f"step6_topics_final_layer_{selected_layer}_selected.xlsx"
        
        if target_file.exists():
            target_file.rename(selected_file)
        elif not selected_file.exists():
            raise FileNotFoundError(f"Could not find excel file for layer {selected_layer} in {out_dir}")

        # Re-align the saved model to the user-selected layer so the model
        # reproduces the exported topic assignments (reviewer validation #12).
        assignments_path = self.workspace / "data" / "checkpoints" / "step6_layer_assignments.pkl"
        if assignments_path.exists():
            assignments = pd.read_pickle(assignments_path)
            if selected_layer < len(assignments):
                self.logger.info(f"Re-aligning saved model to selected Layer {selected_layer}...")
                input_file = self.workspace / "data" / "02_intermediate" / "step5_corpus_filtered_relevant.pkl"
                df = pd.read_pickle(input_file)
                docs = ("Title: " + df["title"].fillna("").astype(str) + "\n\nAbstract: " + df["abstract"].fillna("").astype(str)).tolist()

                from sentence_transformers import SentenceTransformer
                from bertopic import BERTopic
                emb_name = self.cfg_models.get("embedding", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
                self._seed_everything()
                embedding_model = SentenceTransformer(emb_name, device=self.cfg_bertopic.get("embedding_device", "cpu"))
                model_dir = self.workspace / "data" / "checkpoints" / "step6_bertopic_model"
                bertopic_model = BERTopic.load(str(model_dir), embedding_model=embedding_model)
                bertopic_model.update_topics(docs=docs, topics=assignments[selected_layer])
                # BERTopic 0.17.4 bug: update_topics never recreates topic
                # embeddings for custom assignments — call explicitly.
                bertopic_model._create_topic_vectors()
                bertopic_model.save(str(model_dir), serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)
                self.logger.info(f"Model re-aligned to Layer {selected_layer} and re-saved.")
            else:
                self.logger.warning(f"Selected layer {selected_layer} out of range of saved assignments ({len(assignments)} layers). Model not re-aligned.")
        else:
            self.logger.warning("No step6_layer_assignments.pkl found — model not re-aligned (rerun Step 6 to generate it).")
            
        self._update_state("completed",
            output_dict={"topics_final_file": str(selected_file)},
            validation={"status": "completed", "validated_at": datetime.now().isoformat()}
        )
        self.logger.info(f"Step 6 finalized. Target output: {selected_file}")
        return {"status": "completed", "selected_layer": selected_layer}

    def run_verify(self):
        """Verify mode (reviewer validation #12 — topic sizes): verify that the
        topic sizes reported in the exported Excel match the actual per-document
        assignments.

        Two checks are performed:
        1. PRIMARY — saved per-layer assignments (step6_layer_assignments.pkl)
           vs. the Excel 'Topics' sheet counts. Must match exactly.
        2. SECONDARY (informational) — re-predict with the saved model and
           compare. EVoC assignments come from hierarchical clustering, while
           BERTopic.transform() uses cosine similarity to topic embeddings, so
           boundary documents may differ; reported for transparency only.
        """
        self.logger.info("Starting Step 6 verify mode (topic size verification)...")

        topics_file = Path(self.state["steps"]["6"].get("output", {}).get("topics_final_file", ""))
        if not topics_file.exists():
            candidates = sorted((self.workspace / "data" / "05_exports").glob("step6_topics_final_layer_*_selected.xlsx"))
            if not candidates:
                raise FileNotFoundError("No selected topics Excel file found in data/05_exports.")
            topics_file = candidates[0]

        # Determine which layer the selected file corresponds to
        m = re.search(r"layer_(\d+)_selected", topics_file.name)
        selected_layer = int(m.group(1)) if m else None

        # 1. Load exported Excel counts
        xlsx = pd.read_excel(topics_file, sheet_name="Topics")
        excel_counts = dict(zip(xlsx["Topic"].astype(int), xlsx["Count"].astype(int)))

        # 2. PRIMARY: compare with saved per-layer assignments
        assignments_path = self.workspace / "data" / "checkpoints" / "step6_layer_assignments.pkl"
        primary_verified = False
        primary_discrepancies = []
        n_docs = 0
        if assignments_path.exists() and selected_layer is not None:
            assignments = pd.read_pickle(assignments_path)
            if selected_layer < len(assignments):
                arr = assignments[selected_layer]
                n_docs = int(len(arr))
                pred_counts = Counter(arr.tolist())
                all_tids = sorted(set(pred_counts.keys()) | set(excel_counts.keys()))
                primary_discrepancies = [
                    {"topic": int(t), "assignments_count": int(pred_counts.get(t, 0)), "excel_count": int(excel_counts.get(t, 0))}
                    for t in all_tids if pred_counts.get(t, 0) != excel_counts.get(t, 0)
                ]
                primary_verified = len(primary_discrepancies) == 0
                self.logger.info(
                    f"Primary check (saved assignments vs Excel): {len(all_tids)} topics, "
                    f"{len(primary_discrepancies)} discrepancies."
                )
            else:
                self.logger.warning(f"Selected layer {selected_layer} out of range of saved assignments ({len(assignments)}).")
        else:
            self.logger.warning("No step6_layer_assignments.pkl found — primary check skipped (rerun Step 6 to generate it).")

        # 3. SECONDARY (informational): re-predict with the saved model
        model_dir = self.workspace / "data" / "checkpoints" / "step6_bertopic_model"
        transform_counts = None
        if model_dir.exists():
            try:
                from sentence_transformers import SentenceTransformer
                from bertopic import BERTopic
                emb_name = self.cfg_models.get("embedding", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
                self._seed_everything()
                embedding_model = SentenceTransformer(emb_name, device=self.cfg_bertopic.get("embedding_device", "cpu"))
                bertopic_model = BERTopic.load(str(model_dir), embedding_model=embedding_model)
                input_file = self.workspace / "data" / "02_intermediate" / "step5_corpus_filtered_relevant.pkl"
                df = pd.read_pickle(input_file)
                docs = ("Title: " + df["title"].fillna("").astype(str) + "\n\nAbstract: " + df["abstract"].fillna("").astype(str)).tolist()
                topics, _ = bertopic_model.transform(docs)
                transform_counts = dict(sorted(Counter(topics).items()))
                self.logger.info("Secondary check (model re-prediction) completed.")
            except Exception as e:
                self.logger.warning(f"Secondary check failed (informational): {e}")

        # 4. Report
        report_path = self.workspace / "data" / "04_reports" / "step6_verify_report.md"
        md = [
            "# Step 6: Topic Size Verification Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Selected layer: **{selected_layer}** — file: `{topics_file.name}`",
            "",
            "## Primary check — saved assignments vs. exported Excel",
            "",
            f"- Documents: **{n_docs}**",
            f"- Topics checked: **{len(excel_counts)}**",
            f"- Discrepancies: **{len(primary_discrepancies)}**",
            "",
        ]
        if primary_verified:
            md.append("✅ **VERIFIED:** every topic size in the exported Excel matches the saved per-document assignments.")
        else:
            md.append("⚠️ **DISCREPANCIES FOUND:** the following topics differ between the saved assignments and the export:")
            md.append("")
            md.append("| Topic | Assignments count | Excel count |")
            md.append("|---|---|---|")
            for d in primary_discrepancies:
                md.append(f"| {d['topic']} | {d['assignments_count']} | {d['excel_count']} |")
        md.append("")
        md.append("## Secondary check — model re-prediction (informational)")
        md.append("")
        md.append("EVoC topic assignments come from hierarchical clustering, while `BERTopic.transform()`")
        md.append("re-assigns documents by cosine similarity to topic embeddings. Boundary documents may")
        md.append("therefore differ; this check is reported for transparency only.")
        md.append("")
        if transform_counts:
            md.append("| Topic | Model re-prediction | Excel count |")
            md.append("|---|---|---|")
            for t in sorted(set(transform_counts) | set(excel_counts)):
                md.append(f"| {t} | {transform_counts.get(t, 0)} | {excel_counts.get(t, 0)} |")
        else:
            md.append("Model re-prediction not available.")
        md.append("")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(md), encoding="utf-8")
        self.logger.info(f"Verify report saved to {report_path}")

        # 5. State update (preserve current status, add verify info)
        current_status = self.state["steps"]["6"].get("status", "completed")
        self._update_state(
            current_status,
            output_dict={"verify_report_file": str(report_path)},
            params={
                "verify_primary_verified": primary_verified,
                "verify_topics_checked": len(excel_counts),
                "verify_discrepancies": len(primary_discrepancies),
            },
        )

        return {
            "verified": primary_verified,
            "selected_layer": selected_layer,
            "topics_checked": len(excel_counts),
            "discrepancies": primary_discrepancies,
            "report_file": str(report_path),
        }

    # ------------------------------------------------------------------
    # Granularity indicators (AKL framework, Section 2.2 of the paper)
    # ------------------------------------------------------------------
    # Definitions (identical to the scripts that produced the case-study tables):
    #   DB_norm(k)  = min-max normalization of Davies-Bouldin across candidate layers
    #   S_norm(k)   = min-max normalization of Silhouette across candidate layers
    #   Q(k)        = S_norm(k) * (1 - DB_norm(k))            -> Adaptive CVI and Kneedle
    #   CVI1(k)     = ((S(k)+1)/2) * (1 - DB_norm(k)) / sqrt(k)  -> Standard CVI
    #   CVI2(k)     = Q(k) / k^alpha, alpha fitted on log(Q+eps) ~ -alpha*log(k)+c
    # Note: Q(k) is a RELATIVE score, zero by construction for the layer with the
    # lowest Silhouette and for the layer with the highest Davies-Bouldin.
    CVI_EPS = 1e-9

    def _reassign_noise(self, embeddings, labels):
        """Reassign EVoC noise points (-1) to the topic with the nearest centroid
        (cosine similarity), computed on the non-noise documents only.
        Returns (labels, n_noise). No-op when there is no noise. This is the same
        procedure used for the battery case study; the hydrogen case study kept
        the -1 group as a separate (catch-all) cluster instead."""
        from sklearn.metrics.pairwise import cosine_similarity
        labels = labels.copy()
        noise_mask = labels == -1
        n_noise = int(noise_mask.sum())
        if n_noise == 0:
            return labels, 0
        topic_ids = np.array(sorted(t for t in set(labels.tolist()) if t != -1))
        topic_embeddings = np.array([embeddings[labels == t].mean(axis=0) for t in topic_ids])
        sim = cosine_similarity(embeddings[noise_mask], topic_embeddings)
        labels[noise_mask] = topic_ids[sim.argmax(axis=1)]
        return labels, n_noise

    def _compute_cvi(self, df_metrics: pd.DataFrame):
        """Compute the three granularity indicators from per-layer metrics.

        Input columns: Layer, N_Topics, Silhouette, Davies_Bouldin.
        Returns (df_metrics_with_indices, info). Pure function of its input
        (apart from logging), so it can be tested against published tables.
        """
        from sklearn.linear_model import LinearRegression
        eps = self.CVI_EPS
        df = df_metrics.copy()
        n_layers = len(df)

        db_min, db_max = df['Davies_Bouldin'].min(), df['Davies_Bouldin'].max()
        s_min, s_max = df['Silhouette'].min(), df['Silhouette'].max()
        df['DB_norm'] = (df['Davies_Bouldin'] - db_min) / (db_max - db_min + eps)
        df['S_norm'] = (df['Silhouette'] - s_min) / (s_max - s_min + eps)       # min-max (Q, Adaptive, Kneedle)
        df['S_norm_v2'] = (df['Silhouette'] + 1) / 2                             # absolute [-1,1] -> [0,1] (Standard CVI)

        df['Q_score'] = df['S_norm'] * (1 - df['DB_norm'])

        # Decay factor alpha (eps keeps log defined for the layers with Q = 0)
        X = np.log(df['N_Topics'] + eps).values.reshape(-1, 1)
        y = np.log(df['Q_score'] + eps).values
        alpha_hat = float(-LinearRegression().fit(X, y).coef_[0])

        df['CVI1_Sqrt'] = df['S_norm_v2'] * (1 - df['DB_norm']) / np.sqrt(df['N_Topics'])
        df['CVI2_Adaptive'] = df['Q_score'] / (df['N_Topics'] ** alpha_hat)

        # Diagnostic only (does not take part in the vote): alpha estimated
        # excluding the layers with Q ~ 0, which are zero by construction.
        alpha_clean = None
        df['CVI2_Adaptive_clean'] = np.nan
        pos = df[df['Q_score'] > 1e-6]
        if len(pos) >= 3:
            m = LinearRegression().fit(np.log(pos['N_Topics']).values.reshape(-1, 1), np.log(pos['Q_score']).values)
            alpha_clean = float(-m.coef_[0])
            df['CVI2_Adaptive_clean'] = df['Q_score'] / (df['N_Topics'] ** alpha_clean)

        best_cvi1 = int(df.loc[df['CVI1_Sqrt'].idxmax(), 'Layer'])
        best_cvi2 = int(df.loc[df['CVI2_Adaptive'].idxmax(), 'Layer'])
        best_cvi2_clean = None
        if alpha_clean is not None:
            best_cvi2_clean = int(df.loc[df['CVI2_Adaptive_clean'].idxmax(), 'Layer'])

        # Kneedle (applied to Q(k) in layer order, as in the case-study scripts)
        kneed = _require("kneed")
        kneedle = kneed.KneeLocator(df['N_Topics'].values, df['Q_score'].values, curve='concave', direction='increasing')
        knee_k = kneedle.knee
        best_kneedle = None
        if knee_k is not None:
            best_kneedle = int(df.loc[df['N_Topics'] == knee_k, 'Layer'].iloc[0])

        # Reliability warnings
        warn = []
        adaptive_reliable = True
        if n_layers < 4:
            adaptive_reliable = False
            warn.append(f"Adaptive CVI not informative: only {n_layers} candidate layers (alpha is fitted on {n_layers} points, "
                        f"including the layer(s) with Q=0 by construction).")
        if alpha_hat < 0:
            adaptive_reliable = False
            warn.append(f"Adaptive CVI unreliable: negative decay factor (alpha={alpha_hat:.3f}), which rewards larger k; "
                        f"driven by the layer(s) with Q=0 in the log-regression.")
        if n_layers < 5:
            warn.append(f"Kneedle may be unreliable: only {n_layers} candidate layers.")
        if alpha_clean is not None and best_cvi2_clean != best_cvi2:
            warn.append(f"Adaptive CVI maximum changes from Layer {best_cvi2} to Layer {best_cvi2_clean} when alpha is estimated "
                        f"only from layers with Q>0 (alpha={alpha_clean:.3f}); treat the Adaptive vote with caution.")

        info = {
            "alpha_hat": alpha_hat, "alpha_clean": alpha_clean,
            "best_cvi1": best_cvi1, "best_cvi2": best_cvi2, "best_cvi2_clean": best_cvi2_clean,
            "knee_k": (int(knee_k) if knee_k is not None else None), "best_kneedle": best_kneedle,
            "adaptive_reliable": adaptive_reliable, "warnings": warn,
        }
        return df, info

    def _evaluate_and_name_layers(self, bertopic_model, docs, embeddings, df, representation_model):
        from sklearn.metrics import silhouette_score, davies_bouldin_score

        cluster_layers = bertopic_model.hdbscan_model.cluster_layers_
        metrics_data = []
        topic_infos = []
        all_topics_per_layer = {}
        valid_layer_indices = []

        self.logger.info(f"EVoC generated {len(cluster_layers)} layers. Evaluating CVIs...")

        for layer_idx in range(len(cluster_layers)):
            current_topics = cluster_layers[layer_idx].copy()
            n_noise_raw = int((current_topics == -1).sum())
            n_topics = len(set(current_topics.tolist())) - (1 if n_noise_raw > 0 else 0)

            if n_topics > 1:
                # Reassign noise to the nearest topic centroid (no-op when there is no noise)
                current_topics, n_noise = self._reassign_noise(embeddings, current_topics)
                self.logger.info(
                    f"Layer {layer_idx}: {n_topics} topics; {n_noise} noise documents reassigned "
                    f"({100 * n_noise / len(current_topics):.2f}%)."
                )

                # Metrics (Silhouette: Euclidean, sampled when n > 65,000; Davies-Bouldin: full corpus)
                sil_sample = min(65000, len(embeddings))
                sil = silhouette_score(embeddings, current_topics, metric="euclidean", sample_size=sil_sample, random_state=self.seed)
                db = davies_bouldin_score(embeddings, current_topics)

                metrics_data.append({
                    "Layer": layer_idx,
                    "N_Topics": n_topics,
                    "Silhouette": sil,
                    "Davies_Bouldin": db,
                    "N_Noise_Reassigned": n_noise,
                    "Pct_Noise_Reassigned": round(100 * n_noise / len(current_topics), 2),
                    "Silhouette_Sample_Size": sil_sample,
                })

                # Name topics using LLM
                bertopic_model.update_topics(docs=docs, topics=current_topics, representation_model=representation_model)
                topic_info = self._name_topics_with_llm(bertopic_model.get_topic_info(), layer_idx, bertopic_model)
                self._save_topic_info_pickle(topic_info, layer_idx)
                topic_infos.append(topic_info)
                all_topics_per_layer[layer_idx] = current_topics
                valid_layer_indices.append(layer_idx)

        # ── CVI Calculations ──
        df_metrics = pd.DataFrame(metrics_data)
        if df_metrics.empty:
            raise RuntimeError("No valid layers with >1 clusters found.")

        df_metrics, info = self._compute_cvi(df_metrics)
        alpha_hat = info["alpha_hat"]
        best_cvi1, best_cvi2, best_kneedle, knee_k = info["best_cvi1"], info["best_cvi2"], info["best_kneedle"], info["knee_k"]
        for w in info["warnings"]:
            self.logger.warning(w)

        suggested_layers = {
            "cvi1_sqrt": int(best_cvi1),
            "cvi2_adaptive": int(best_cvi2),
            "kneedle": int(best_kneedle) if best_kneedle is not None else None
        }

        consensus_reached = False
        final_best_idx = None

        if best_cvi1 == best_cvi2:
            if best_kneedle is None or best_kneedle == best_cvi1:
                # An automatic choice is only taken when the Adaptive CVI is informative;
                # otherwise the decision is delegated to the expert (set
                # step6_evoc.require_reliable_indicators: false to disable this guard).
                if info["adaptive_reliable"] or not self.cfg_bertopic.get("require_reliable_indicators", True):
                    consensus_reached = True
                    final_best_idx = int(best_cvi1)
                else:
                    self.logger.warning("Indicators agree, but the Adaptive CVI is not reliable: human validation required.")

        # Generate Reports
        cvi_md_path = self.workspace / "data" / "04_reports" / "step6_cvi_report.md"
        plot_path = self.workspace / "data" / "04_reports" / "step6_kneedle_plot.png"

        info["evoc_params"] = getattr(self, "_evoc_params", None)
        self._generate_cvi_report(df_metrics, alpha_hat, suggested_layers, consensus_reached, cvi_md_path, info)
        self._generate_kneedle_plot(df_metrics, knee_k, plot_path)

        first_layer_topics = next(iter(all_topics_per_layer.values()), None)
        self._update_state("running",
                           output_dict={"cvi_report_file": str(cvi_md_path), "kneedle_plot_file": str(plot_path)},
                           params={"n_topics_found": len(set(first_layer_topics.tolist())) if first_layer_topics is not None else 0,
                                   "cvi_alpha": float(alpha_hat),
                                   "cvi_alpha_clean": info["alpha_clean"],
                                   "cvi_adaptive_reliable": bool(info["adaptive_reliable"]),
                                   "cvi_warnings": info["warnings"],
                                   "evoc_params": getattr(self, "_evoc_params", None)})

        idx_mapping = {layer_num: i for i, layer_num in enumerate(valid_layer_indices)}
        mapped_best = idx_mapping[final_best_idx] if final_best_idx is not None else None

        return mapped_best, consensus_reached, suggested_layers, topic_infos, list(all_topics_per_layer.values())

    def _generate_cvi_report(self, df_metrics, alpha_hat, suggested_layers, consensus_reached, report_path, info=None):
        info = info or {}
        md = f"# EVoC CVI Analysis Report\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        md += "## Definitions\n"
        md += "- `DB_norm`: min-max normalization of the Davies-Bouldin index across the candidate layers.\n"
        md += "- `Q(k) = S_norm * (1 - DB_norm)`, with `S_norm` the min-max normalized Silhouette across the candidate layers. "
        md += "Q is a relative score: it is zero by construction for the layer with the lowest Silhouette and for the layer with the highest Davies-Bouldin. "
        md += "It feeds the Adaptive CVI and Kneedle.\n"
        md += "- **CVI 1 (Standard)** `= ((S + 1) / 2) * (1 - DB_norm) / sqrt(k)`.\n"
        md += "- **CVI 2 (Adaptive)** `= Q(k) / k^alpha`, with alpha fitted on `log(Q + 1e-9)` versus `log(k)`.\n"
        md += "- Silhouette: Euclidean distance on the embeddings (sampled if the corpus exceeds 65,000 documents). Davies-Bouldin: full corpus.\n"
        if info.get("evoc_params"):
            md += f"- EVoC parameters used: `{info['evoc_params']}`\n"
        md += "\n"

        md += f"**Alpha estimated via Linear Regression:** {alpha_hat:.4f}\n\n"
        if info.get("alpha_clean") is not None:
            md += (f"**Alpha excluding layers with Q=0 (diagnostic only):** {info['alpha_clean']:.4f} "
                   f"(Adaptive CVI maximum at Layer {info['best_cvi2_clean']})\n\n")

        md += "## Metrics per Layer\n"
        cols = ['Layer', 'N_Topics', 'Silhouette', 'Davies_Bouldin', 'Q_score', 'CVI1_Sqrt', 'CVI2_Adaptive']
        for extra in ['CVI2_Adaptive_clean', 'N_Noise_Reassigned', 'Pct_Noise_Reassigned']:
            if extra in df_metrics.columns:
                cols.append(extra)
        md += df_metrics[cols].to_markdown(index=False)

        md += "\n\n## Recommendations\n"
        md += f"- **CVI 1 (Standard)** suggests Layer: **{suggested_layers['cvi1_sqrt']}**\n"
        md += f"- **CVI 2 (Adaptive)** suggests Layer: **{suggested_layers['cvi2_adaptive']}**\n"
        kneedle_txt = suggested_layers['kneedle'] if suggested_layers['kneedle'] is not None else "none (no knee returned)"
        md += f"- **Kneedle Algorithm** suggests Layer: **{kneedle_txt}**\n\n"

        if info.get("warnings"):
            md += "## Reliability warnings\n"
            for w in info["warnings"]:
                md += f"- ⚠️ {w}\n"
            md += "\n"

        if consensus_reached:
            md += "✅ **CONSENSUS REACHED.** The pipeline has automatically selected this optimal layer.\n"
        else:
            md += "⚠️ **NO CONSENSUS.** The metrics diverged or were not reliable. Please check the `step6_kneedle_plot.png` and 2D maps to decide which layer to select based on your research goals.\n"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(md)

    def _generate_kneedle_plot(self, df_metrics, knee_k, plot_path):
        plt = _require("matplotlib.pyplot", "matplotlib")
        plt.figure(figsize=(10, 6))
        plt.plot(df_metrics['N_Topics'], df_metrics['Q_score'], marker='o', linestyle='-', linewidth=2, label='Quality Score Q(k)')
        
        if knee_k is not None:
            plt.axvline(x=knee_k, color='red', linestyle='--', linewidth=2, label=f'Detected Knee (k={knee_k})')
            
        plt.xlabel('Number of Topics (k)', fontsize=12)
        plt.ylabel('Quality Score Q(k)', fontsize=12)
        plt.title('EVoC: Quality Score Decay and Kneedle Detection', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

    def _generate_cluster_maps(self, embeddings, all_topics_per_layer, topic_infos, df):
        umap = _require("umap", "umap-learn")
        datamapplot = _require("datamapplot")
        
        out_dir = self.workspace / "data" / "04_reports" / "step6_cluster_maps"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Clear old maps
        for old_file in out_dir.glob("step6_cluster_map_layer_*.*"):
            try: old_file.unlink()
            except: pass
            
        self._update_state("running", output_dict={"cluster_maps_folder": str(out_dir)})

        # Check if we have layers with < 50 topics
        layers_to_plot = [i for i, info in enumerate(topic_infos) if len(info) < 50]
        
        if not layers_to_plot:
            self.logger.info("No layers with < 50 topics found. Skipping 2D maps.")
            return

        self.logger.info("Computing global UMAP 2D reduction for cluster maps...")
        reduced_embeddings = umap.UMAP(n_components=2, random_state=self.seed).fit_transform(embeddings)

        for i in layers_to_plot:
            self.logger.info(f"Generating 2D map for Layer {i}...")
            topic_info = topic_infos[i]
            topic_ids = all_topics_per_layer[i]
            counts = Counter(topic_ids)

            name_mapping = {}
            for _, row in topic_info.iterrows():
                tid = row['Topic']
                name = row.get('New_Name', row.get('Name', str(tid)))
                # Extract clean name after 'ID_' if present
                if "_" in str(name):
                    name = str(name).split("_", 1)[1]
                count = counts[tid]
                name_mapping[tid] = f"{name} ({count})"

            formatted_labels = np.array([name_mapping.get(tid, str(tid)) for tid in topic_ids], dtype=object)

            fig, ax = datamapplot.create_plot(
                reduced_embeddings,
                formatted_labels,
                figsize=(11, 11),
                title=f"Knowledge Landscape - Layer {i}",
                label_over_points=True,
                label_wrap_width=20,
                label_font_size=14,
                color_label_text=False,
                min_font_weight=500,
                max_font_weight=501,
                font_family="sans-serif"
            )

            png_path = out_dir / f"step6_cluster_map_layer_{i}.png"
            pdf_path = out_dir / f"step6_cluster_map_layer_{i}.pdf"
            
            fig.savefig(str(png_path), bbox_inches="tight", dpi=150)
            fig.savefig(str(pdf_path), bbox_inches="tight")
            
            # Close figure to free memory
            import matplotlib.pyplot as plt
            plt.close(fig)

    def _save_topic_excels(self, topic_infos, best_layer_idx, all_topics_per_layer, df):
        import re
        out_dir = self.workspace / "data" / "05_exports"
        
        for old_file in out_dir.glob("step6_topics_final_layer_*.xlsx"):
            try: old_file.unlink()
            except: pass

        layers_overview = pd.DataFrame([{"Layer": idx, "N_Topics": len(ti), "Selected": "★" if idx == best_layer_idx else ""} for idx, ti in enumerate(topic_infos)])

        for layer_idx, topic_info in enumerate(topic_infos):
            topics_array = all_topics_per_layer[layer_idx]
            kw_col = "Aspect1" if "Aspect1" in topic_info.columns else "Representation"

            suffix = "_selected" if layer_idx == best_layer_idx else ""
            xlsx_path = out_dir / f"step6_topics_final_layer_{layer_idx}{suffix}.xlsx"

            with pd.ExcelWriter(str(xlsx_path), engine="openpyxl") as writer:
                # Sheet 1
                summary = topic_info.copy()
                summary[kw_col] = summary[kw_col].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
                cols = [c for c in ["Topic", "Count", "Name", "New_Name", kw_col] if c in summary.columns]
                summary[cols].sort_values("Topic").to_excel(writer, sheet_name="Topics", index=False)
                # Sheet 2
                layers_overview.to_excel(writer, sheet_name="All_Layers", index=False)
                # Data sheets
                df_temp = df.copy()
                df_temp["_topic"] = topics_array
                name_map = dict(zip(topic_info["Topic"], topic_info.get("New_Name", topic_info["Name"])))
                for topic_id, grp in df_temp.groupby("_topic"):
                    topic_name = name_map.get(topic_id, f"Topic_{topic_id}")
                    sheet_label = re.sub(r'[\\/*?:\[\]]', "_", str(topic_name))[:31]
                    cols_to_save = [c for c in ["doi", "title", "abstract", "year", "first_author", "source_journal"] if c in grp.columns]
                    grp[cols_to_save].to_excel(writer, sheet_name=sheet_label, index=False)

    def _build_models(self):
        from sentence_transformers import SentenceTransformer
        from bertopic import BERTopic
        from bertopic.representation import OpenAI as BERTopicOpenAI, KeyBERTInspired
        from bertopic.dimensionality import BaseDimensionalityReduction
        from sklearn.feature_extraction.text import CountVectorizer
        import openai
        import evoc

        emb_name = self.cfg_models.get("embedding", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        device = self.cfg_bertopic.get("embedding_device", "cpu")
        self.logger.info(f"Loading embedding model: {emb_name}")
        if device != "cpu":
            self.logger.warning(
                f"embedding_device='{device}' is not CPU: embedding vectors may vary between "
                "runs/hardware (non-deterministic GPU kernels), so outputs may not be identical "
                "even with a fixed seed."
            )
        embedding_model = SentenceTransformer(emb_name, device=device)

        # EVoC clustering model (random_state guarded by inspect.signature inside
        # _build_evoc_kwargs — warns instead of working around an unsupported arg).
        self._seed_everything()
        evoc_kwargs = self._build_evoc_kwargs()
        self.logger.info(f"Building EVoC with kwargs: {evoc_kwargs}")
        evoc_model = evoc.EVoC(**evoc_kwargs)
        self._evoc_params = self._collect_evoc_params(evoc_model)
        self.logger.info(f"EVoC effective parameters: {self._evoc_params}")
        umap_model = BaseDimensionalityReduction()
        vectorizer_model = CountVectorizer(ngram_range=(1, 3), stop_words="english", min_df=self.cfg_bertopic.get("min_df", 5))

        kw_model = KeyBERTInspired()

        if self.naming_mode == "agent":
            # Agent naming mode — keyword-only representation (no LLM needed)
            self.logger.info("Agent naming mode — using keyword-only representation.")
            representation_model = {
                "Main": [kw_model],
                "Aspect1": [kw_model],
            }
        else:
            client = openai.OpenAI(api_key="not-needed", base_url=self.cfg_vllm.get("base_url", "http://localhost:8000/v1"))
            llm_model = self.cfg_models.get("llm_extraction", "gemma-3-4b-instruct")

            llm_rep = BERTopicOpenAI(client, model=llm_model, chat=True, nr_docs=5, diversity=0.3,
                                     generator_kwargs={"temperature": 0.0, "seed": self.seed}, prompt="""
You will extract a short topic label from given documents and keywords.
[DOCUMENTS]
Keywords: [KEYWORDS]
Based on the information above, extract a short topic label (max 45 chars) in the following format:
topic: <topic_label>""")

            representation_model = {"Main": [llm_rep], "Aspect1": [kw_model]}

        bertopic_model = BERTopic(
            embedding_model=embedding_model, umap_model=umap_model, hdbscan_model=evoc_model,
            vectorizer_model=vectorizer_model, representation_model=representation_model,
            calculate_probabilities=True, verbose=True, top_n_words=20
        )
        return embedding_model, bertopic_model, representation_model

    def _get_embeddings(self, docs: list, embedding_model) -> np.ndarray:
        checkpoints_dir = self.workspace / "data" / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        emb_path = checkpoints_dir / "step6_embeddings.pkl"
        meta_path = checkpoints_dir / "step6_embeddings_meta.json"
        
        params = {"model": self.cfg_models.get("embedding", "")}
        current_checksum = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
        
        if emb_path.exists() and meta_path.exists():
            meta = json.load(open(meta_path, "r"))
            if meta.get("checksum") == current_checksum and meta.get("n_docs") == len(docs):
                self.logger.info("Loading cached embeddings.")
                return pd.read_pickle(str(emb_path))
                
        self.logger.info(f"Computing embeddings for {len(docs)} documents...")
        self._seed_everything()
        embeddings = embedding_model.encode(docs, show_progress_bar=True)
        pd.to_pickle(embeddings, str(emb_path))
        with open(meta_path, "w") as f:
            json.dump({"checksum": current_checksum, "n_docs": len(docs)}, f)
        return embeddings

    def _name_topics_with_llm(self, topic_info: pd.DataFrame, layer_idx: int = 0,
                              bertopic_model=None) -> pd.DataFrame:
        """
        Name topics for a layer.
        - "vllm" mode: call the vLLM server (OpenAI-compatible API).
        - "agent" mode: dump keywords + representative doc titles to a JSON file
          so the orchestrator LLM can name the topics; placeholder names are used
          until the labels are applied (see step6_apply_labels.py).
        Adds columns: New_Name, New_Representation.
        """
        if self.naming_mode == "agent":
            return self._dump_naming_input(topic_info, layer_idx, bertopic_model)

        import openai
        client = openai.OpenAI(api_key="not-needed", base_url=self.cfg_vllm.get("base_url", "http://localhost:8000/v1"))
        llm_model = self.cfg_models.get("llm_extraction", "gemma-3-4b-instruct")

        kw_col = "Aspect1" if "Aspect1" in topic_info.columns else "Representation"
        df_input = topic_info[["Topic", kw_col]].copy()
        df_input[kw_col] = df_input[kw_col].apply(lambda x: ", ".join(x[:10]) if isinstance(x, list) else str(x))
        table_str = df_input.to_markdown(index=False)

        prompt = f"""You are an expert research taxonomist. Extract unique specific topic labels (max 45 chars).
DATA:
{table_str}
OUTPUT FORMAT:
Return ONLY a pipe-separated list with two columns: Topic | New_Label"""
        try:
            resp = client.chat.completions.create(model=llm_model, messages=[{"role": "user", "content": prompt}], temperature=0.0, seed=self.seed, max_tokens=512)
            refined = pd.read_csv(io.StringIO(resp.choices[0].message.content), sep="|", names=["Topic", "New_Label"], skipinitialspace=True)
            refined["Topic"] = pd.to_numeric(refined["Topic"], errors="coerce")
            refined = refined.dropna(subset=["Topic"])
            topic_info = topic_info.merge(refined, on="Topic", how="left")
            topic_info["New_Name"] = topic_info["Topic"].astype(str) + "_" + topic_info["New_Label"].fillna(topic_info["Name"]).str.strip()
            topic_info["New_Representation"] = topic_info["New_Label"]
        except Exception as e:
            self.logger.warning(f"LLM renaming failed: {e}")
            topic_info["New_Name"] = topic_info["Name"]
        return topic_info

    def _dump_naming_input(self, topic_info: pd.DataFrame, layer_idx: int,
                           bertopic_model=None) -> pd.DataFrame:
        """
        Agent naming mode: dump per-topic keywords and representative document
        titles to data/02_intermediate/step6_naming_input/layer_{idx}.json so the
        orchestrator LLM can generate topic labels. Placeholder names are used
        until the labels are applied (see step6_apply_labels.py).
        """
        out_dir = self.workspace / "data" / "02_intermediate" / "step6_naming_input"
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
        out_dir = self.workspace / "data" / "02_intermediate" / "step6_topic_info"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"layer_{layer_idx}.pkl"
        topic_info.to_pickle(str(path))
        self.logger.info(f"Topic info saved to {path}")

if __name__ == "__main__":
    args = sys.argv[1:]
    naming_mode = "vllm"
    if "--naming-mode" in args:
        i = args.index("--naming-mode")
        naming_mode = args[i + 1]
        del args[i:i + 2]
    mode = args[0] if args else "run"
    worker = Step6Worker(naming_mode=naming_mode)
    if mode == "run":
        print(json.dumps(worker.run_clustering()))
    elif mode == "verify":
        print(json.dumps(worker.run_verify()))
    elif mode == "finalize" and len(args) > 1:
        print(json.dumps(worker.run_finalize(int(args[1]))))
    else:
        print(json.dumps({"error": "Invalid arguments. Use 'run', 'verify', or 'finalize <layer_idx>'"}))