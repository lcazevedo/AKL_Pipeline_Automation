"""
step7_sentence_filter.py
------------------------
Precision filter for the sentence-level vectors of Step 7 (challenges, directions).

Gemma extracts a label for every sentence, but the Step-8 audits showed many labels
are attached to sentences that are not challenges / future directions (audit accuracy
0.70 and 0.38). A small supervised sentence classifier (DanL/scientific-challenges-and-directions,
PubMedBERT fine-tuned on CORD-19, multi-label Challenge/Direction) is used as a second
opinion: a Gemma row is kept only if the classifier also marks the sentence as positive.

Usage (after Step 7, no LLM server needed):
    ./run_worker.sh workers/step7_sentence_filter.py
    ./run_worker.sh workers/step7_worker.py          # rebuilds step7_extractions.* from the filtered checkpoints

Originals are preserved in data/02_intermediate/step7_checkpoints/unfiltered/ and the
script always filters from them, so it can be rerun with another threshold.
"""
import sys, json, logging, shutil, time
import os
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT = Path(os.environ.get("AKL_WORKSPACE") or Path.cwd())   # the workspace (run folder); defaults to the current directory
sys.path.insert(0, str(ROOT))


def _logger():
    lg = logging.getLogger("step7_filter")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        fmt = logging.Formatter("%(asctime)s [STEP7-FILTER] %(message)s", datefmt="%H:%M:%S")
        for h in (logging.FileHandler(ROOT / "execution.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
            h.setFormatter(fmt); lg.addHandler(h)
    return lg


def score_texts(texts, model_dir, batch_size, max_length, log):
    import torch
    from transformers import AutoConfig, AutoTokenizer, BertForSequenceClassification
    cfg = AutoConfig.from_pretrained(model_dir)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification(cfg)
    # weights_only=True: the checkpoint is a pickle from a third party; load tensors only
    sd = torch.load(Path(model_dir) / "pytorch_model.bin", weights_only=True, map_location="cpu")
    model.load_state_dict(sd, strict=False)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    log.info(f"Scoring {len(texts)} sentences on {device} (batch {batch_size}, max_length {max_length})")
    order = np.argsort([len(t) for t in texts])          # length-sorted batches: far less padding
    out = np.zeros((len(texts), 2), dtype=np.float32)
    t0, step = time.time(), max(1, len(texts) // 10 // batch_size) * batch_size
    for n, i in enumerate(range(0, len(texts), batch_size)):
        idx = order[i:i + batch_size]
        b = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        with torch.no_grad():
            out[idx] = torch.sigmoid(model(**b).logits).cpu().numpy()
        if i % step < batch_size and i > 0:
            el = time.time() - t0
            log.info(f"  scored {i}/{len(texts)} ({100 * i // len(texts)}%) — {el:.0f}s elapsed, ~{el / i * (len(texts) - i):.0f}s left")
    return out


def main():
    log = _logger()
    cfg = yaml.safe_load(open(ROOT / "pipeline_config.yaml"))
    sc = cfg.get("step7_extraction", {}).get("sentence_classifier", {})
    if not sc.get("enabled", False):
        log.info("sentence_classifier disabled in pipeline_config.yaml; nothing to do.")
        return {"status": "skipped"}
    thr = float(sc.get("threshold", 0.5))
    label_idx = {"Challenge": 0, "Direction": 1}
    targets = sc.get("vectors", {"challenges": "Challenge", "directions": "Direction"})
    ck = ROOT / "data" / "02_intermediate" / "step7_checkpoints"
    bk = ck / "unfiltered"; bk.mkdir(exist_ok=True)

    frames = {}
    for v in targets:
        orig, cur = bk / f"{v}.pkl", ck / f"{v}.pkl"
        if not orig.exists():
            shutil.copy2(cur, orig)
            log.info(f"Backed up {cur.name} -> unfiltered/")
        frames[v] = pd.read_pickle(orig)

    cache_path = ROOT / "data" / "02_intermediate" / "step7_sentence_scores.pkl"
    cache = pd.read_pickle(cache_path) if cache_path.exists() else {}
    texts = sorted({t for f in frames.values() for t in f["original_text"].astype(str)} - set(cache))
    if texts:
        s = score_texts(texts, ROOT / sc.get("model_dir", "data/checkpoints/sentence_classifier"),
                        int(sc.get("batch_size", 64)), int(sc.get("max_length", 128)), log)
        cache.update({t: tuple(map(float, p)) for t, p in zip(texts, s)})
        pd.to_pickle(cache, cache_path)

    summary = {}
    for v, label in targets.items():
        df = frames[v].copy()
        df["classifier_prob"] = df["original_text"].astype(str).map(lambda t: cache[t][label_idx[label]])
        kept = df[df["classifier_prob"] >= thr].copy()
        kept.to_pickle(ck / f"{v}.pkl")
        summary[v] = {"before": int(len(df)), "kept": int(len(kept)), "kept_pct": round(100 * len(kept) / len(df), 1)}
        log.info(f"{v}: kept {len(kept)} of {len(df)} rows ({summary[v]['kept_pct']}%) with P({label}) >= {thr}")
    return {"status": "completed", "threshold": thr, **summary}


if __name__ == "__main__":
    print(json.dumps(main()))
