"""Grouped-CV sweep of SetFit settings on the Step 4 labels (standalone, run on pc2)."""
import json, sys, time, gc, warnings
import numpy as np, pandas as pd, torch
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments

warnings.filterwarnings("ignore")
REVIEW = "data/03_review/step4_relevance_review.xlsx"
OUT = "experiments/sweep_results.jsonl"

# --- labeled data, same logic as Step5Worker._load_labeled_data ---
rows = []
for name, df in pd.read_excel(REVIEW, sheet_name=None).items():
    if name in ("Topics", "All_Layers") or "Relevance (yes/no)" not in df.columns:
        continue
    lab = df["Relevance (yes/no)"].astype(str).str.strip().str.lower()
    d = df[lab.isin(["yes", "no"])].copy()
    d["label"] = lab[lab.isin(["yes", "no"])]
    d["group"] = name
    rows.append(d)
data = pd.concat(rows, ignore_index=True)
data["text"] = data["title"].fillna("") + ".\\n" + data["abstract"].fillna("")
data = data[["text", "label", "group"]].reset_index(drop=True)
print(f"labeled: {len(data)} | {data.label.value_counts().to_dict()} | topics: {data.group.nunique()}", flush=True)

CONFIGS = []
for iters in (10, 20, 40):
    for cw in (None, "balanced"):
        CONFIGS.append(("all-MiniLM-L6-v2", iters, 1, cw))
for base in ("all-mpnet-base-v2", "BAAI/bge-base-en-v1.5"):
    for cw in (None, "balanced"):
        CONFIGS.append((base, 20, 1, cw))
CONFIGS.append(("all-MiniLM-L6-v2", 20, 3, "balanced"))

import os
if os.environ.get("QUICK"):   # small re-check after the label expansion
    CONFIGS = [("all-MiniLM-L6-v2", 10, 1, None), ("all-MiniLM-L6-v2", 10, 1, "balanced"),
               ("all-MiniLM-L6-v2", 20, 1, "balanced")]
    OUT = "experiments/sweep_results_449.jsonl"
_only = os.environ.get("ONLY_BASE")
if _only:   # run only the configs of one base model (fresh process => fresh GPU memory)
    CONFIGS = [c for c in CONFIGS if c[0] == _only]

gkf = GroupKFold(n_splits=4)
for base, iters, epochs, cw in CONFIGS:
    t0 = time.time()
    oof_p = np.zeros(len(data)); ok = True
    try:
        for tr, te in gkf.split(data, data.label, groups=data.group):
            kw = {"head_params": {"class_weight": cw}} if cw else {}
            model = SetFitModel.from_pretrained(base, **kw)
            args = TrainingArguments(batch_size=16, num_epochs=epochs, num_iterations=iters,
                                     sampling_strategy="oversampling", show_progress_bar=False,
                                     logging_strategy="no", report_to="none")
            ds = Dataset.from_pandas(data.iloc[tr][["text", "label"]].reset_index(drop=True))
            Trainer(model=model, args=args, train_dataset=ds).train()
            p = model.predict_proba(data.iloc[te]["text"].tolist())
            p = torch.as_tensor(p).cpu().numpy()
            classes = list(model.model_head.classes_)
            oof_p[te] = p[:, classes.index("yes")]
            del model; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        ok = False; err = f"{type(e).__name__}: {str(e)[:150]}"
    if not ok:
        rec = {"base": base, "iters": iters, "epochs": epochs, "cw": cw, "error": err}
    else:
        y = (data.label == "yes").to_numpy().astype(int)
        pred = (oof_p >= 0.5).astype(int)
        ths = np.linspace(0.05, 0.99, 95)
        best = max(((f1_score(y, (oof_p >= t).astype(int), average="macro"), t) for t in ths))
        rec = {"base": base, "iters": iters, "epochs": epochs, "cw": cw,
               "macroF1@0.5": round(f1_score(y, pred, average="macro"), 3),
               "recall_no": round(recall_score(y, pred, pos_label=0), 3),
               "precision_no": round(precision_score(y, pred, pos_label=0, zero_division=0), 3),
               "recall_yes": round(recall_score(y, pred, pos_label=1), 3),
               "AUC": round(roc_auc_score(y, oof_p), 3),
               "best_macroF1": round(best[0], 3), "best_threshold": round(float(best[1]), 2),
               "pred_yes_share": round(float(pred.mean()), 3), "seconds": round(time.time() - t0)}
    print(json.dumps(rec), flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
print("DONE", flush=True)
