"""
step7_sentence_verifier.py
--------------------------
Precision filter for the sentence-level vectors of Step 7 (challenges, directions), using the LLM
as a strict yes/no verifier instead of the extraction label.

Why: the Step-8 audits found the extraction labels had 0.70 (challenges) and 0.38 (directions) accuracy.
On the audited sentences (specialist gold), a yes/no verifier with explicit definitions reached AUC 0.97
(challenges) and 0.91 (directions), better than the DanL sentence classifier (0.90 / 0.77).

Usage (needs the vLLM server(s) listed in step7_extraction.sentence_verifier.base_urls):
    ./run_worker.sh workers/step7_sentence_verifier.py
    ./run_worker.sh workers/step7_worker.py           # rebuilds step7_extractions.* from the filtered checkpoints

Resumable: scores are cached in data/02_intermediate/step7_verifier_scores.pkl after every chunk.
The unfiltered checkpoints in step7_checkpoints/unfiltered/ are always the source; the script can be
rerun with other thresholds without calling the LLM again.
"""
import sys, json, logging, shutil, time, asyncio
import os
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import openai

ROOT = Path(os.environ.get("AKL_WORKSPACE") or Path.cwd())   # the workspace (run folder); defaults to the current directory

DEFS = {
    "challenges": ("CHALLENGE",
        "a sentence that states a problem, difficulty, flaw, limitation, failure, lack of clarity or knowledge gap "
        "(in the field, in existing methods or in the approach). A sentence that only reports a result, an advantage "
        "or the method itself is NOT a challenge."),
    "directions": ("FUTURE DIRECTION",
        "a sentence that states a suggestion or need for further research, future work, a hypothesis or an opportunity "
        "worth exploring. A sentence that only describes the present work, its results, background, or a current "
        "problem is NOT a future direction."),
}


def _logger():
    lg = logging.getLogger("step7_verifier")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        fmt = logging.Formatter("%(asctime)s [STEP7-VERIFY] %(message)s", datefmt="%H:%M:%S")
        for h in (logging.FileHandler(ROOT / "execution.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
            h.setFormatter(fmt); lg.addHandler(h)
    return lg


def _prompt(vector, text):
    name, defn = DEFS[vector]
    return (f"Decide whether the sentence below, taken from a scientific abstract in chemistry or materials science, is a {name}.\n"
            f"A {name} is {defn}\n\nSentence: \"{text}\"\n\nAnswer with only \"yes\" or \"no\".")


async def _score_chunk(pairs, clients, model, concurrency):
    sem = asyncio.Semaphore(concurrency)

    async def one(i, vec, txt):
        async with sem:
            for _ in range(3):
                try:
                    r = await clients[i % len(clients)].chat.completions.create(
                        model=model, messages=[{"role": "user", "content": _prompt(vec, txt)}],
                        temperature=0, max_tokens=1, logprobs=True, top_logprobs=10)
                    tl = r.choices[0].logprobs.content[0].top_logprobs
                    py = sum(np.exp(x.logprob) for x in tl if x.token.strip().lower().startswith("y"))
                    pn = sum(np.exp(x.logprob) for x in tl if x.token.strip().lower().startswith("n"))
                    return float(py / (py + pn + 1e-9))
                except Exception:
                    await asyncio.sleep(1)
            return None
    return await asyncio.gather(*[one(i, v, t) for i, (v, t) in enumerate(pairs)])


def main():
    log = _logger()
    cfg = yaml.safe_load(open(ROOT / "pipeline_config.yaml"))
    sv = cfg.get("step7_extraction", {}).get("sentence_verifier", {})
    if not sv.get("enabled", False):
        log.info("sentence_verifier disabled in pipeline_config.yaml; nothing to do.")
        return {"status": "skipped"}
    thr = {k: float(v) for k, v in sv.get("thresholds", {"challenges": 0.2, "directions": 0.5}).items()}
    ck = ROOT / "data" / "02_intermediate" / "step7_checkpoints"
    bk = ck / "unfiltered"; bk.mkdir(exist_ok=True)
    frames = {}
    for v in thr:
        orig, cur = bk / f"{v}.pkl", ck / f"{v}.pkl"
        if not orig.exists():
            shutil.copy2(cur, orig); log.info(f"Backed up {cur.name} -> unfiltered/")
        frames[v] = pd.read_pickle(orig)

    cache_path = ROOT / "data" / "02_intermediate" / "step7_verifier_scores.pkl"
    cache = pd.read_pickle(cache_path) if cache_path.exists() else {}
    todo = sorted({(v, t) for v, f in frames.items() for t in f["original_text"].astype(str)} - set(cache))

    if todo:
        urls = sv.get("base_urls") or [cfg["step7_vllm"]["base_url"]]
        import requests
        ok = []
        for u in urls:
            try:
                if requests.get(u.rstrip("/") + "/models", timeout=5).status_code == 200: ok.append(u)
            except Exception:
                log.warning(f"vLLM endpoint not reachable, skipping: {u}")
        if not ok:
            raise RuntimeError("No reachable vLLM endpoint for the sentence verifier.")
        log.info(f"Verifying {len(todo)} sentences on {ok}")
        clients = [openai.AsyncOpenAI(api_key="not-needed", base_url=u, timeout=120, max_retries=1) for u in ok]
        model = cfg["models"]["llm_extraction"]
        conc = int(sv.get("concurrency", 24)) * len(ok)
        chunk = int(sv.get("checkpoint_every", 4000))
        t0 = time.time(); done = 0
        for i in range(0, len(todo), chunk):
            part = todo[i:i + chunk]
            res = asyncio.run(_score_chunk(part, clients, model, conc))
            for p, r in zip(part, res):
                if r is not None: cache[p] = r
            pd.to_pickle(cache, cache_path)
            done += len(part); el = time.time() - t0
            log.info(f"  verified {done}/{len(todo)} ({100 * done // len(todo)}%) — {el:.0f}s elapsed, ~{el / done * (len(todo) - done):.0f}s left"
                     f" | failed in chunk: {sum(r is None for r in res)}")
        missing = [p for p in todo if p not in cache]
        if missing:
            raise RuntimeError(f"{len(missing)} sentences could not be scored; rerun to retry them.")

    summary = {}
    for v, f in frames.items():
        df = f.copy()
        df["verifier_prob"] = [cache[(v, t)] for t in df["original_text"].astype(str)]
        kept = df[df["verifier_prob"] >= thr[v]].copy()
        kept.to_pickle(ck / f"{v}.pkl")
        summary[v] = {"before": int(len(df)), "kept": int(len(kept)), "kept_pct": round(100 * len(kept) / len(df), 1), "threshold": thr[v]}
        log.info(f"{v}: kept {len(kept)} of {len(df)} rows ({summary[v]['kept_pct']}%) with P(yes) >= {thr[v]}")
    return {"status": "completed", **summary}


if __name__ == "__main__":
    print(json.dumps(main()))
