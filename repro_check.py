"""
repro_check.py — AKL reproducibility evidence collector (read-only; does not change any worker).

Usage (from the workspace root):
    python repro_check.py snapshot run1      # after finishing a step (or the whole pipeline)
    python repro_check.py snapshot run2      # after rerunning the same step(s) with the same seed
    python repro_check.py compare run1 run2  # writes data/04_reports/repro_report.md and .json
    python repro_check.py funnel             # document counts per step (from pipeline_state.json)

A snapshot copies whatever artifacts exist at that moment, so it can be taken after Step 3 only,
after Steps 3+5, or after the full pipeline. `compare` reports only on artifacts present in both.
"""
import sys, json, shutil, hashlib, platform, subprocess, datetime
from pathlib import Path
import numpy as np
import pandas as pd

WS = Path(".")
SNAP_ROOT = WS / "data" / "08_repro"
ARTIFACTS = {
    "step3_corpus_with_topics.pkl": "data/02_intermediate/step3_corpus_with_topics.pkl",
    "step5_full_predictions.pkl": "data/02_intermediate/step5_full_predictions.pkl",
    "step5_corpus_filtered_relevant.pkl": "data/02_intermediate/step5_corpus_filtered_relevant.pkl",
    "step6_layer_assignments.pkl": "data/checkpoints/step6_layer_assignments.pkl",
    "step7_extractions.pkl": "data/02_intermediate/step7_extractions.pkl",
    "pipeline_state.json": "pipeline_state.json",
    "pipeline_config.yaml": "pipeline_config.yaml",
    "project_config.md": "project_config.md",                       # research design, step-7 vectors/prompts
    "step1_corpus_raw.pkl": "data/02_intermediate/step1_corpus_raw.pkl",
    "step2_corpus_clean.pkl": "data/02_intermediate/step2_corpus_clean.pkl",
    "step4_relevance_review.xlsx": "data/03_review/step4_relevance_review.xlsx",   # human-validated labels
    "step8_insights_review.md": "data/04_reports/step8_insights_review.md",
    "execution.log": "execution.log",
    "step3_embeddings.pkl": "data/checkpoints/step3_embeddings.pkl",
    "step6_embeddings.pkl": "data/checkpoints/step6_embeddings.pkl",
}
# Human/agent-produced artifacts whose file name contains the layer number
ARTIFACT_GLOBS = ["data/05_exports/step3_topics_preliminary_layer_*_selected.xlsx",   # step-3 topic names
                  "data/05_exports/step6_topics_final_layer_*_selected.xlsx",         # step-6 topic names
                  "data/01_raw/*"]                                                    # raw inputs (WOS export, etc.)

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def env_info():
    info = {"timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "python": platform.python_version(), "platform": platform.platform(),
            "cpu_count": __import__("os").cpu_count()}
    try:
        from importlib.metadata import version
        for pkg in ["evoc", "numba", "numpy", "scikit-learn", "bertopic", "sentence-transformers",
                    "torch", "setfit", "umap-learn", "pandas"]:
            try: info[pkg] = version(pkg)
            except Exception: pass
    except Exception:
        pass
    try:
        info["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                     capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        info["gpu"] = None
    return info

def snapshot(tag):
    d = SNAP_ROOT / tag
    d.mkdir(parents=True, exist_ok=True)
    manifest = {"env": env_info(), "files": {}}
    for name, rel in ARTIFACTS.items():
        src = WS / rel
        if src.exists():
            shutil.copy2(src, d / name)
            manifest["files"][name] = {"sha256": sha256(src), "bytes": src.stat().st_size}
    for pattern in ARTIFACT_GLOBS:
        for src in sorted(WS.glob(pattern)):
            if src.is_file():
                dst = d / "extra"; dst.mkdir(exist_ok=True)
                shutil.copy2(src, dst / src.name)
                manifest["files"]["extra/" + src.name] = {"sha256": sha256(src), "bytes": src.stat().st_size}
    try:   # exact package versions of the interpreter running this script
        fr = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=60).stdout
        (d / "pip_freeze.txt").write_text(fr)
    except Exception:
        pass
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Snapshot '{tag}' saved in {d} ({len(manifest['files'])} artifacts).")

def load_manifest(tag):
    return json.loads((SNAP_ROOT / tag / "manifest.json").read_text())

def agreement(a, b):
    """ARI / NMI / identical (partition-equal) / exact-equal between two label vectors."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    a, b = np.asarray(a), np.asarray(b)
    if len(a) != len(b):
        return {"comparable": False, "reason": f"different length ({len(a)} vs {len(b)})"}
    return {"comparable": True, "exact_equal": bool((a == b).all()),
            "ARI": float(adjusted_rand_score(a, b)), "NMI": float(normalized_mutual_info_score(a, b)),
            "n_topics_A": int(len(set(a.tolist()) - {-1})), "n_topics_B": int(len(set(b.tolist()) - {-1})),
            "noise_A": float((a == -1).mean()), "noise_B": float((b == -1).mean())}

def cmp_step3(A, B):
    fa, fb = A / "step3_corpus_with_topics.pkl", B / "step3_corpus_with_topics.pkl"
    if not (fa.exists() and fb.exists()): return None
    da, db = pd.read_pickle(fa), pd.read_pickle(fb)
    cols = sorted([c for c in da.columns if c.startswith("topic_layer_") and c in db.columns],
                  key=lambda c: int(c.split("_")[-1]))
    return {"n_docs": [len(da), len(db)], "layers": {c: agreement(da[c].values, db[c].values) for c in cols}}

def cmp_emb(A, B, name):
    fa, fb = A / name, B / name
    if not (fa.exists() and fb.exists()): return None
    try:
        ea, eb = np.asarray(pd.read_pickle(fa), dtype=float), np.asarray(pd.read_pickle(fb), dtype=float)
    except Exception as e:
        return {"comparable": False, "reason": str(e)}
    if ea.shape != eb.shape: return {"comparable": False, "reason": f"shape {ea.shape} vs {eb.shape}"}
    d = np.abs(ea - eb)
    return {"comparable": True, "shape": list(ea.shape), "exact_equal": bool((ea == eb).all()),
            "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean())}

def cmp_step5(A, B):
    fa, fb = A / "step5_full_predictions.pkl", B / "step5_full_predictions.pkl"
    if not (fa.exists() and fb.exists()): return None
    da, db = pd.read_pickle(fa), pd.read_pickle(fb)
    out = {"n_docs": [len(da), len(db)]}
    if len(da) != len(db): return out
    if "proba_yes" in da.columns:
        diff = np.abs(da["proba_yes"].astype(float).values - db["proba_yes"].astype(float).values)
        out["proba_yes_max_abs_diff"] = float(diff.max()); out["proba_yes_mean_abs_diff"] = float(diff.mean())
    if "predicted_relevance" in da.columns:
        out["label_agreement"] = float((da["predicted_relevance"].values == db["predicted_relevance"].values).mean())
    return out

def cmp_step6(A, B):
    fa, fb = A / "step6_layer_assignments.pkl", B / "step6_layer_assignments.pkl"
    if not (fa.exists() and fb.exists()): return None
    pa, pb = pd.read_pickle(fa), pd.read_pickle(fb)
    # step6_layer_assignments.pkl may be a list (one array per layer) or a dict {layer: array}
    if not isinstance(pa, dict): pa = dict(enumerate(pa))
    if not isinstance(pb, dict): pb = dict(enumerate(pb))
    ka, kb = sorted(pa.keys()), sorted(pb.keys())
    out = {"layers_A": len(ka), "layers_B": len(kb), "layers": {}}
    for k in ka:
        if k in pb: out["layers"][str(k)] = agreement(pa[k], pb[k])
    return out

def cmp_step7(A, B):
    fa, fb = A / "step7_extractions.pkl", B / "step7_extractions.pkl"
    if not (fa.exists() and fb.exists()): return None
    da, db = pd.read_pickle(fa), pd.read_pickle(fb)
    key = ["doi", "vector", "original_text"]
    for d in (da, db): d["_k"] = d.groupby(key).cumcount()
    m = da.merge(db, on=key + ["_k"], suffixes=("_A", "_B"), how="outer", indicator=True)
    both = m[m["_merge"] == "both"]
    out = {"rows_A": len(da), "rows_B": len(db), "rows_matched": int(len(both))}
    if len(both):
        out["micro_label_exact_agreement"] = float((both["micro_label_A"].str.strip().str.lower() ==
                                                     both["micro_label_B"].str.strip().str.lower()).mean())
        from sklearn.metrics import adjusted_rand_score
        out["macro_category_ARI"] = float(adjusted_rand_score(both["macro_category_A"], both["macro_category_B"]))
        out["by_vector"] = {v: float((g["micro_label_A"].str.strip().str.lower() ==
                                      g["micro_label_B"].str.strip().str.lower()).mean())
                            for v, g in both.groupby("vector")}
    return out

def compare(ta, tb):
    A, B = SNAP_ROOT / ta, SNAP_ROOT / tb
    ma, mb = load_manifest(ta), load_manifest(tb)
    res = {"run_A": ta, "run_B": tb, "env_A": ma["env"], "env_B": mb["env"],
           "hash_identical": {n: (ma["files"][n]["sha256"] == mb["files"][n]["sha256"])
                              for n in ma["files"] if n in mb["files"] and n.endswith(".pkl")},
           "emb3": cmp_emb(A, B, "step3_embeddings.pkl"), "emb6": cmp_emb(A, B, "step6_embeddings.pkl"),
           "step3": cmp_step3(A, B), "step5": cmp_step5(A, B), "step6": cmp_step6(A, B), "step7": cmp_step7(A, B)}
    rep = WS / "data" / "04_reports"; rep.mkdir(parents=True, exist_ok=True)
    (rep / "repro_report.json").write_text(json.dumps(res, indent=2, default=str))
    L = [f"# Reproducibility report: {ta} vs {tb}", "",
         f"- Run A: {ma['env']['timestamp']} | Run B: {mb['env']['timestamp']}",
         f"- evoc {ma['env'].get('evoc')} / {mb['env'].get('evoc')} | numba {ma['env'].get('numba')} / {mb['env'].get('numba')}",
         f"- GPU: {ma['env'].get('gpu')} / {mb['env'].get('gpu')}", ""]
    L.append("## Byte-identical artifacts (.pkl SHA-256)")
    for n, ok in res["hash_identical"].items(): L.append(f"- {n}: {'identical' if ok else 'different'}")
    def layer_table(title, d):
        L.extend(["", f"## {title}", "", "| Layer | Topics A/B | Noise A/B | ARI | NMI | Exact equal |", "|---|---|---|---|---|---|"])
        for k, v in d["layers"].items():
            if v.get("comparable"):
                L.append(f"| {k} | {v['n_topics_A']}/{v['n_topics_B']} | {v['noise_A']:.1%}/{v['noise_B']:.1%} | {v['ARI']:.3f} | {v['NMI']:.3f} | {v['exact_equal']} |")
            else: L.append(f"| {k} | not comparable: {v['reason']} | | | | |")
    for k in ("emb3", "emb6"):
        if res[k]: L.extend(["", f"## Embeddings ({k})", ""] + [f"- {a}: {b}" for a, b in res[k].items()])
    if res["step3"]: layer_table("Step 3 — layer assignments", res["step3"])
    if res["step5"]:
        L.extend(["", "## Step 5 — SetFit predictions", ""] + [f"- {k}: {v}" for k, v in res["step5"].items()])
    if res["step6"]:
        L.append(f"\n(Step 6: layers A={res['step6']['layers_A']}, B={res['step6']['layers_B']})")
        layer_table("Step 6 — layer assignments", res["step6"])
    if res["step7"]:
        L.extend(["", "## Step 7 — LLM extraction agreement", ""] + [f"- {k}: {v}" for k, v in res["step7"].items()])
    (rep / "repro_report.md").write_text("\n".join(L))
    print("\n".join(L)); print(f"\nWritten: {rep/'repro_report.md'} and repro_report.json")

def funnel():
    st = json.loads((WS / "pipeline_state.json").read_text())
    steps = st.get("steps", st)
    rows = []
    for k, v in (steps.items() if isinstance(steps, dict) else enumerate(steps, 1)):
        if isinstance(v, dict):
            p = v.get("params_used", {}) or {}
            keep = {x: p[x] for x in p if any(t in x for t in ("record", "n_", "count", "best_layer", "accuracy", "f1", "retention"))}
            rows.append({"step": k, "status": v.get("status"), **keep})
    out = WS / "data" / "04_reports" / "repro_funnel.json"
    out.write_text(json.dumps(rows, indent=2, default=str)); print(json.dumps(rows, indent=2, default=str)); print(f"Written: {out}")

if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 2 and a[0] == "snapshot": snapshot(a[1])
    elif len(a) == 3 and a[0] == "compare": compare(a[1], a[2])
    elif a == ["funnel"]: funnel()
    else: print(__doc__)
