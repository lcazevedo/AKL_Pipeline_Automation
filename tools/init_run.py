"""Create a new run workspace that reuses the frozen inputs of a previous run.

    python tools/init_run.py --name new_run --source . [--reuse-steps 1,2]

Creates runs/<name>/ with:
  - symlinks to the shared code (workers, tools, skills, run scripts, AGENTS.md, .env);
  - its own pipeline_config.yaml, project_config.md and pipeline_state.json (a copy of the source configuration,
    Steps listed in --reuse-steps marked completed with a note saying where they come from);
  - its own data/ with ONLY the reusable inputs (Step 2 corpus, Step 3 embeddings, imported labels, Step 7 caches,
    OpenAlex enrichment): everything else is recomputed, so the run is independent of the source run;
  - RUN_INFO.json with the SHA-256 of every reused file and the code version.
Run the workers from inside runs/<name>/ (the workspace is the current directory).
"""
import argparse, hashlib, json, os, shutil, subprocess, sys
from datetime import datetime
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--source", default=".")
ap.add_argument("--reuse-steps", default="1,2")
ap.add_argument("--manifest-out", default=None, help="also write the reused-input hashes to this JSON (versioned copy)")
a = ap.parse_args()

src = Path(a.source).resolve(); dst = src / "runs" / a.name
if dst.exists():
    sys.exit(f"{dst} already exists; remove it or pick another name.")
dst.mkdir(parents=True)
code_root = Path(__file__).resolve().parent.parent

for item in ("workers", "tools", ".opencode", "run_worker.sh", "run_worker_setfit.sh", "AGENTS.md", ".env", "base_pipeline_config.yaml", "base_project_config.md"):
    p = code_root / item
    if p.exists():
        os.symlink(p, dst / item)

for f in ("pipeline_config.yaml", "project_config.md"):
    shutil.copy2(src / f, dst / f)

for d in ("01_raw", "02_intermediate", "03_review", "04_reports", "05_exports"):
    (dst / "data" / d).mkdir(parents=True, exist_ok=True)
(dst / "data" / "checkpoints").mkdir(parents=True, exist_ok=True)

REUSE = [  # (source relative path, destination relative path)
    ("data/02_intermediate/step2_corpus_clean.pkl", "data/02_intermediate/step2_corpus_clean.pkl"),
    ("data/04_reports/step2_quality_report.md", "data/04_reports/step2_quality_report.md"),
    ("data/checkpoints/step3_embeddings.pkl", "data/checkpoints/step3_embeddings.pkl"),
    ("data/checkpoints/step3_embeddings_meta.json", "data/checkpoints/step3_embeddings_meta.json"),
    ("data/03_review/labels_relevance.csv", "data/03_review/labels_relevance.csv"),
    ("data/02_intermediate/step7_checkpoints/raw/abstract.pkl", "data/02_intermediate/step7_cache_previous/abstract.pkl"),
    ("data/02_intermediate/step7_checkpoints/raw/sentence.pkl", "data/02_intermediate/step7_cache_previous/sentence.pkl"),
    ("data/02_intermediate/step7_verifier_scores.pkl", "data/02_intermediate/step7_verifier_scores.pkl"),
    ("data/02_intermediate/openalex_enrichment.pkl", "data/02_intermediate/openalex_enrichment.pkl"),
    ("data/02_intermediate/openalex_references.pkl", "data/02_intermediate/openalex_references.pkl"),
]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
hashes = {}
for s_rel, d_rel in REUSE:
    s_p, d_p = src / s_rel, dst / d_rel
    if not s_p.exists():
        print("  (missing, skipped)", s_rel); continue
    d_p.parent.mkdir(parents=True, exist_ok=True)
    if subprocess.run(["cp", "-c", str(s_p), str(d_p)], capture_output=True).returncode != 0:   # APFS clone when possible
        shutil.copy2(s_p, d_p)
    hashes[d_rel] = {"sha256": sha(s_p), "bytes": s_p.stat().st_size, "from": s_rel}
    print("  reused", d_rel)

# state: base template + the reused steps of the source run
state = json.load(open(code_root / "base_pipeline_state.json", encoding="utf-8"))
old = json.load(open(src / "pipeline_state.json", encoding="utf-8"))
now = datetime.now().isoformat(timespec="seconds")
state["project"].update({"name": f"{old.get('project', {}).get('name') or 'AKL'} ({a.name})", "created_at": now, "last_updated": now, "workspace_path": "."})
reuse = [x.strip() for x in a.reuse_steps.split(",") if x.strip()]
for k in reuse:
    st = json.loads(json.dumps(old["steps"][k])); st["status"] = "completed"
    st["notes"] = f"Reused from the source run ({src.name}); inputs hashed in RUN_INFO.json."
    state["steps"][k] = st
state["current_step"] = max(int(k) for k in reuse) if reuse else 0
json.dump(state, open(dst / "pipeline_state.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

try:
    commit = subprocess.run(["git", "-C", str(code_root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(code_root), "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
except Exception:
    commit, dirty = "", None
info = {"run": a.name, "created_at": now, "source_run": str(src), "reused_steps": reuse, "code_commit": commit, "code_has_uncommitted_changes": dirty,
        "reused_files": hashes}
json.dump(info, open(dst / "RUN_INFO.json", "w", encoding="utf-8"), indent=2)
if a.manifest_out:
    Path(a.manifest_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({k: {"sha256": v["sha256"], "bytes": v["bytes"]} for k, v in hashes.items()}, open(a.manifest_out, "w", encoding="utf-8"), indent=2)
print(f"\nrun workspace ready: {dst}\nreused steps: {reuse} | code commit {commit[:7]} (uncommitted changes: {dirty})")
