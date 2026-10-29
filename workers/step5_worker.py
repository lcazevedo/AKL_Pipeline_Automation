"""
step5_worker.py
---------------
AKL Pipeline — Step 5: Corpus Filtering Model (SetFit)

Responsibilities:
- Load labeled data from data/review/relevance_review.xlsx
- Train a SetFit model using few-shot learning
- Evaluate the model and generate setfit_report.md
- Apply the trained model to the entire corpus_clean.pkl
- Save relevant/not relevant documents in .pkl (inputs) and .xlsx (outputs)
- Compute AI_Confidence for each prediction
- Save the trained model to data/checkpoints/setfit_model
- Update pipeline_state.json
"""

import sys
import json
import logging
import os
import random
import warnings
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import yaml

warnings.filterwarnings("ignore")

def _setup_logger(workspace_path):
    logger = logging.getLogger("step5_worker")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [STEP5] %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(Path(workspace_path) / "execution.log", encoding="utf-8")
        fh.setFormatter(fmt)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

class Step5Worker:
    def __init__(self, config_path="pipeline_config.yaml", state_path="pipeline_state.json"):
        self.config_path = Path(config_path)
        self.state_path = Path(state_path)
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        with open(self.state_path, "r", encoding="utf-8") as f:
            self.state = json.load(f)
            
        self.workspace = Path(self.state.get("project", {}).get("workspace_path", "."))
        self.logger = _setup_logger(self.workspace)
        
        # Hyperparameters
        self.setfit_cfg = self.config.get("step5_setfit", {})
        self.base_model = self.config.get("models", {}).get("setfit_base", "sentence-transformers/paraphrase-mpnet-base-v2")
        self.num_epochs = self.setfit_cfg.get("num_epochs", 1)
        self.batch_size = self.setfit_cfg.get("batch_size", 16)
        self.min_samples = self.setfit_cfg.get("min_samples_per_class", 8)
        # Audit-driven options (Phase 2)
        self.group_by_topic = self.setfit_cfg.get("group_by_topic", True)   # GroupShuffleSplit keeps Step-3 topics together
        self.retention_check = self.setfit_cfg.get("retention_check", True) # verify the model retains the expert's Step-4 labels
        self.save_proba_yes = self.setfit_cfg.get("save_proba_yes", True)   # save P(yes) for the Step-8 boundary audit

        # Reproducibility: step-level seed defaults to the global seed.
        self.cfg_repro = self.config.get("reproducibility", {}) or {}
        self.global_seed = self.cfg_repro.get("seed", 42)
        self.seed = self._resolve_seed(self.setfit_cfg.get("seed"))
        self.logger.info(f"Random seed: {self.seed}")

    def _resolve_seed(self, value):
        """Resolve a step-level seed. `None` inherits the global seed."""
        if value is None:
            value = self.global_seed
        try:
            return int(value)
        except (TypeError, ValueError):
            return 42

    def _apply_seed(self):
        """Seed Python, NumPy, Torch, HuggingFace and SetFit before training."""
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
        try:
            from transformers import set_seed as _hf_set_seed
            _hf_set_seed(self.seed)
        except Exception:
            pass
        try:
            import setfit
            if hasattr(setfit, "set_seed"):
                setfit.set_seed(self.seed)
        except Exception:
            pass

    @staticmethod
    def _build_text(title, abstract):
        """Single canonical text builder — used for BOTH training and corpus
        application. Guarantees input consistency between train and deploy
        (reviewer validation #3)."""
        t = title.fillna("") if hasattr(title, "fillna") else (title if title is not None else "")
        a = abstract.fillna("") if hasattr(abstract, "fillna") else (abstract if abstract is not None else "")
        return t + ".\\n" + a

    def _update_state(self, status, output_dict=None, params=None, notes=""):
        now = datetime.now().isoformat(timespec="seconds")
        step = self.state.setdefault("steps", {}).setdefault("5", {})
        step["status"] = status
        if status == "running":
            step["started_at"] = now
            self.state["current_step"] = 5
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

    def _load_labeled_data(self):
        review_file = Path(self.state["steps"]["4"]["output"]["review_file"])
        if not review_file.exists():
            raise FileNotFoundError(f"Review file not found: {review_file}")
            
        sheets = pd.read_excel(review_file, sheet_name=None)
        labeled_rows = []
        
        for s_name, df in sheets.items():
            if s_name in ["Topics", "All_Layers"]: continue
            col = "Relevance (yes/no)"
            if col in df.columns:
                valid_df = df[df[col].notna() & (df[col].astype(str).str.strip() != "")].copy()
                valid_df["label"] = valid_df[col].astype(str).str.strip().str.lower()
                valid_df = valid_df[valid_df["label"].isin(["yes", "no"])]
                # The sheet name is the Step-3 topic (cluster) — used as the
                # group for GroupShuffleSplit so topics never leak across splits.
                valid_df["group"] = s_name
                labeled_rows.append(valid_df)
                
        if not labeled_rows:
            raise ValueError("No valid labeled data ('yes' or 'no') found in the review file.")
            
        df_labeled = pd.concat(labeled_rows, ignore_index=True)
        df_labeled["text"] = self._build_text(df_labeled["title"], df_labeled["abstract"])
        
        counts = df_labeled["label"].value_counts()
        yes_c = counts.get("yes", 0)
        no_c = counts.get("no", 0)
        
        self.logger.info(f"Loaded {len(df_labeled)} labeled examples. Yes: {yes_c}, No: {no_c}")
        if yes_c < self.min_samples or no_c < self.min_samples:
            self.logger.warning(f"Class count below minimum ({self.min_samples}). Training may be unstable.")
            
        return df_labeled[["text", "label", "group"]]

    def _generate_report(self, true_labels, pred_labels, report_path):
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        acc = accuracy_score(true_labels, pred_labels)
        cls_report = classification_report(true_labels, pred_labels)
        cm = confusion_matrix(true_labels, pred_labels, labels=["yes", "no"])
        
        report_dict = classification_report(true_labels, pred_labels, output_dict=True)
        macro_f1 = report_dict["macro avg"]["f1-score"]
        
        md_content = f"""# SetFit Training Quality Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary Metrics
- **Global Accuracy**: {acc:.4f}
- **Macro F1-Score**: {macro_f1:.4f}
- **Random seed**: {self.seed}

## Classification Report
```text
{cls_report}
```

## Confusion Matrix
| | Predicted Yes | Predicted No |
|---|---|---|
| **Actual Yes** | {cm[0][0]} | {cm[0][1]} |
| **Actual No** | {cm[1][0]} | {cm[1][1]} |

## Interpretation of AI_Confidence
The `AI_Confidence` column added to the `.xlsx` inspection files represents the SetFit model's certainty in its prediction (ranging from roughly 0.5 to 1.0). 
- Values close to **1.0** indicate high certainty.
- Values closer to **0.5** indicate the model was unsure if the document was relevant or not. 
You can sort the Excel files by this column to review the documents the AI struggled with the most.
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(md_content)
            
        return acc, macro_f1

    def run(self):
        self.logger.info("Starting Step 5: SetFit Filtering...")
        self._update_state("running")
        
        try:
            # 1. Imports
            from datasets import Dataset, DatasetDict
            from setfit import SetFitModel, Trainer, TrainingArguments
            from sklearn.model_selection import GroupShuffleSplit
            
            # 2. Load and prep data
            df_labeled = self._load_labeled_data()
            
            # GroupShuffleSplit (reviewer validation #1): documents from the same
            # Step-3 topic (group) never appear in both train and test, preventing
            # topic-level leakage and giving a more honest generalization estimate.
            if self.group_by_topic and "group" in df_labeled.columns:
                gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=self.seed)
                train_idx, test_idx = next(gss.split(df_labeled, df_labeled["label"], groups=df_labeled["group"]))
                df_train = df_labeled.iloc[train_idx].reset_index(drop=True)
                df_test = df_labeled.iloc[test_idx].reset_index(drop=True)
                n_groups_train = df_train["group"].nunique()
                n_groups_test = df_test["group"].nunique()
                self.logger.info(
                    f"GroupShuffleSplit: {len(df_train)} train / {len(df_test)} test "
                    f"({n_groups_train} topics train, {n_groups_test} topics test)."
                )
            else:
                from sklearn.model_selection import train_test_split
                df_train, df_test = train_test_split(df_labeled, test_size=0.2, random_state=self.seed, stratify=df_labeled["label"])
                self.logger.info("Plain train_test_split used (group_by_topic disabled).")
            
            ds = DatasetDict({
                "train": Dataset.from_pandas(df_train[["text", "label"]]),
                "eval": Dataset.from_pandas(df_test[["text", "label"]])
            })
            
            # 3. Train Model
            self.logger.info(f"Loading base model: {self.base_model}")
            self._apply_seed()
            model = SetFitModel.from_pretrained(self.base_model)

            import inspect
            args_kwargs = {"batch_size": self.batch_size, "num_epochs": self.num_epochs}
            if "seed" in inspect.signature(TrainingArguments.__init__).parameters:
                args_kwargs["seed"] = self.seed
            else:
                self.logger.warning(
                    "Installed setfit TrainingArguments does not accept 'seed': "
                    "model training may not be fully reproducible."
                )
            args = TrainingArguments(**args_kwargs)
            self.logger.info(f"TrainingArguments: {args_kwargs}")

            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=ds['train'],
                eval_dataset=ds['eval']
            )
            
            self.logger.info("Training SetFit model...")
            trainer.train()
            
            # 4. Evaluate
            self.logger.info("Evaluating model...")
            pred_labels = model.predict(ds["eval"]["text"])
            
            report_path = self.workspace / "data" / "04_reports" / "step5_setfit_report.md"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            acc, f1 = self._generate_report(ds["eval"]["label"], pred_labels, report_path)
            self.logger.info(f"Model evaluated. Accuracy: {acc:.2f}, F1: {f1:.2f}")
            
            # 4b. Retention check (reviewer validation #4): apply the model to ALL
            # Step-4 labeled documents and verify it retains the expert's labels.
            retention_acc = None
            if self.retention_check:
                self.logger.info(f"Retention check: applying model to all {len(df_labeled)} Step-4 labeled documents...")
                ret_preds = model.predict(df_labeled["text"].tolist())
                from sklearn.metrics import accuracy_score as _acc
                retention_acc = float(_acc(df_labeled["label"], ret_preds))
                self.logger.info(f"Retention accuracy on {len(df_labeled)} labeled docs: {retention_acc:.4f}")
                # Append retention section to the report
                n_yes = int((df_labeled["label"] == "yes").sum())
                n_no = int((df_labeled["label"] == "no").sum())
                with open(report_path, "a", encoding="utf-8") as f:
                    f.write(f"""
## Retention Check (Reviewer Validation #4)
The trained model was re-applied to all {len(df_labeled)} documents labeled by the
specialist in Step 4 ({n_yes} relevant, {n_no} not relevant) to verify it retains the
expert's labels.

- **Retention accuracy**: {retention_acc:.4f}
- **Split method**: GroupShuffleSplit (grouped by Step-3 topic) — no topic-level leakage
- **Train / Test**: {len(df_train)} / {len(df_test)} documents

A retention accuracy well below the eval accuracy may indicate overfitting to the
test split or label noise; investigate before trusting the corpus filter.
""")
            
            # 5. Save Model
            model_dir = self.workspace / "data" / "checkpoints" / "setfit_model"
            model.save_pretrained(str(model_dir))
            
            # 6. Apply to full corpus & Extract Confidence
            corpus_path = self.workspace / "data" / "02_intermediate" / "step2_corpus_clean.pkl"
            df_corpus = pd.read_pickle(corpus_path)
            n_before = len(df_corpus)
            
            self.logger.info(f"Applying model to full corpus ({n_before} records)...")
            pred_texts = self._build_text(df_corpus["title"], df_corpus["abstract"])
            
            # Predictions
            full_preds = model.predict(pred_texts.tolist())
            
            # Probabilities to extract Confidence
            probas = model.predict_proba(pred_texts.tolist())
            if hasattr(probas, "cpu"):
                probas = probas.cpu().numpy()
            else:
                probas = np.array(probas)
            confidences = np.max(probas, axis=1)
            
            df_corpus["predicted_relevance"] = full_preds
            df_corpus["AI_Confidence"] = confidences
            
            # Probability of the "yes" class (reviewer validation #2 prerequisite):
            # needed by the Step-8 boundary audit to find documents the model is
            # unsure about (confidence near 0.5). Robust to class ordering.
            if self.save_proba_yes and probas.shape[1] == 2:
                df_corpus["proba_yes"] = np.where(
                    np.array(full_preds) == "yes", confidences, 1.0 - confidences
                )
                self.logger.info("Saved 'proba_yes' column for the Step-8 boundary audit.")
            else:
                df_corpus["proba_yes"] = np.nan
            
            df_rel = df_corpus[df_corpus["predicted_relevance"] == "yes"].copy()
            df_nrel = df_corpus[df_corpus["predicted_relevance"] == "no"].copy()
            
            n_after = len(df_rel)
            n_discarded = len(df_nrel)
            
            # 7. Save outputs (.pkl in inputs, .xlsx in outputs)
            pkl_rel = self.workspace / "data" / "02_intermediate" / "step5_corpus_filtered_relevant.pkl"
            pkl_nrel = self.workspace / "data" / "02_intermediate" / "step5_corpus_filtered_not_relevant.pkl"
            
            xlsx_rel = self.workspace / "data" / "05_exports" / "step5_corpus_filtered_relevant.xlsx"
            xlsx_nrel = self.workspace / "data" / "05_exports" / "step5_corpus_filtered_not_relevant.xlsx"
            
            df_rel.to_pickle(pkl_rel)
            df_nrel.to_pickle(pkl_nrel)
            df_rel.to_excel(xlsx_rel, index=False)
            df_nrel.to_excel(xlsx_nrel, index=False)
            
            # Full prediction table for the Step-8 boundary audit (reviewer
            # validation #2): ALL documents with their predicted label and
            # P(yes), so the audit can sample documents near the 0.5 decision
            # boundary from BOTH sides (not only the kept "yes" side).
            pkl_full = self.workspace / "data" / "02_intermediate" / "step5_full_predictions.pkl"
            if self.save_proba_yes:
                df_corpus.to_pickle(pkl_full)
                self.logger.info(f"Saved full prediction table for the boundary audit: {pkl_full}")
            
            self.logger.info(f"Filtering complete. Kept {n_after} relevant, separated {n_discarded} not relevant.")
            
            notes = ""
            if n_after < 100:
                notes = "Filtered corpus is very small (< 100 records). Consider reviewing relevance criteria."
                
            self._update_state("completed", 
                output_dict={
                    "corpus_relevant_pkl": str(pkl_rel),
                    "corpus_not_relevant_pkl": str(pkl_nrel),
                    "corpus_relevant_xlsx": str(xlsx_rel),
                    "corpus_not_relevant_xlsx": str(xlsx_nrel),
                    "full_predictions_pkl": str(pkl_full) if self.save_proba_yes else "",
                    "model_file": str(model_dir),
                    "setfit_report_file": str(report_path)
                },
                params={
                    "setfit_base_model": self.base_model,
                    "seed": self.seed,
                    "accuracy": acc,
                    "f1_score": f1,
                    "retention_accuracy": retention_acc,
                    "split_method": "GroupShuffleSplit (grouped by Step-3 topic)" if self.group_by_topic else "train_test_split (stratified)",
                    "records_before": n_before,
                    "records_after": n_after,
                    "records_discarded": n_discarded
                },
                notes=notes
            )
            
            return {"status": "completed", "records_after": n_after, "records_discarded": n_discarded, "accuracy": acc}
            
        except Exception as e:
            self.logger.error(f"Step 5 failed: {e}", exc_info=True)
            self._update_state("failed", notes=str(e))
            raise

if __name__ == "__main__":
    worker = Step5Worker()
    print(json.dumps(worker.run()))
