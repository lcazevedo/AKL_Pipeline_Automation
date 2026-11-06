"""Select extra candidates for labeling (mostly likely-'no' + boundary + random in doubtful topics)
and pre-label them with the vLLM LLM. Output: data/03_review/step4b_extra_labels.xlsx (for specialist validation)."""
import re, sys, json, random
import numpy as np, pandas as pd, yaml, openai
from sklearn.linear_model import LogisticRegression

random.seed(7); np.random.seed(7)
COL = "Relevance (yes/no)"
REVIEW = "data/03_review/step4_relevance_review.xlsx"
import os
OUT = os.environ.get("OUT_PATH", "data/03_review/step4b_extra_labels.xlsx")
N_LOW, CAP_LOW = 90, 10        # lowest P(yes), max per topic
N_BND, CAP_BND = 60, 6         # closest to P(yes)=0.5, max per topic
N_RND_PER_TOPIC = 5            # random docs per doubtful topic

cfg = yaml.safe_load(open("pipeline_config.yaml"))
emb = np.asarray(pd.read_pickle("data/checkpoints/step3_embeddings.pkl"))
corp = pd.read_pickle("data/02_intermediate/step2_corpus_clean.pkl")
key = corp["doi"].astype(str).str.lower().str.strip().to_numpy()
pos = {k: i for i, k in enumerate(key)}

# review sheets: doi -> topic sheet, labeled rows
topic_of, labeled = {}, []
for name, df in pd.read_excel(REVIEW, sheet_name=None).items():
    if name in ("Topics", "All_Layers") or COL not in df.columns:
        continue
    d = df.copy(); d["topic"] = name
    d["k"] = d["doi"].astype(str).str.lower().str.strip()
    for k in d["k"]: topic_of[k] = name
    lab = d[COL].astype(str).str.strip().str.lower()
    labeled.append(d[lab.isin(["yes", "no"])].assign(label=lab[lab.isin(["yes", "no"])]))
lab = pd.concat(labeled, ignore_index=True)
lab["idx"] = lab["k"].map(pos)
lab = lab.dropna(subset=["idx"])
print(f"labeled {len(lab)} (yes {sum(lab.label=='yes')}, no {sum(lab.label=='no')})", flush=True)

X = emb / np.linalg.norm(emb, axis=1, keepdims=True)
clf = LogisticRegression(C=1, class_weight="balanced", max_iter=3000).fit(
    X[lab["idx"].astype(int)], (lab.label == "yes").astype(int))
p = clf.predict_proba(X)[:, 1]

pool = pd.DataFrame({"idx": np.arange(len(corp)), "k": key, "p": p})
pool["topic"] = pool["k"].map(topic_of)
pool = pool[pool["topic"].notna() & ~pool["k"].isin(set(lab["k"]))]

def pick(df, n, cap, ascending=True, by="p"):
    out, cnt = [], {}
    for _, r in df.sort_values(by, ascending=ascending).iterrows():
        if cnt.get(r.topic, 0) >= cap: continue
        out.append(r.name); cnt[r.topic] = cnt.get(r.topic, 0) + 1
        if len(out) >= n: break
    return out

sel = {}
for i in pick(pool, N_LOW, CAP_LOW): sel[i] = "low P(yes)"
bnd = pool.assign(d=(pool.p - 0.5).abs())
for i in pick(bnd[~bnd.index.isin(sel)], N_BND, CAP_BND, by="d"): sel[i] = "boundary"
# doubtful topics: >=25% 'no' among labeled
rate = lab.groupby("topic").label.apply(lambda s: (s == "no").mean())
doubt = rate[rate >= 0.25].index.tolist()
print("doubtful topics:", doubt, flush=True)
for t in doubt:
    cand = pool[(pool.topic == t) & ~pool.index.isin(sel)]
    for i in cand.sample(min(N_RND_PER_TOPIC, len(cand)), random_state=7).index: sel[i] = "random (doubtful topic)"

cand = pool.loc[list(sel)].copy(); cand["reason"] = pd.Series(sel)
rows = corp.iloc[cand["idx"].to_numpy()][["doi", "title", "abstract", "year"]].reset_index(drop=True)
cand = pd.concat([cand.reset_index(drop=True), rows], axis=1)
print("candidates:", len(cand), cand.reason.value_counts().to_dict(), flush=True)

# LLM pre-labeling with the Section 6 criteria (same prompt as Step 4)
text = open("project_config.md", encoding="utf-8").read()
m = re.search(r"## 6\. Relevance criteria.*?## 7\.", text, re.DOTALL | re.IGNORECASE)
criteria = m.group(0).strip() if m else text
vcfg = cfg["step7_vllm"]
client = openai.OpenAI(api_key="not-needed", base_url=vcfg["base_url"])
model = cfg["models"]["llm_extraction"]
labels = []
for n, r in enumerate(cand.itertuples(), 1):
    abstract = " ".join(str(r.abstract).split()[:350])
    prompt = f"""You are an expert classifying academic papers for a systematic review.
Based strictly on these project criteria:
{criteria}

Paper Title: {r.title}
Abstract: {abstract}

Is this paper RELEVANT to the study?
Reply ONLY with the word "yes" or the word "no". Do not include any other text."""
    try:
        a = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                           temperature=0.0, max_tokens=5).choices[0].message.content.strip().lower()
        labels.append("yes" if "yes" in a else ("no" if "no" in a else ""))
    except Exception as e:
        labels.append("")
        print("LLM error:", str(e)[:100], flush=True)
    if n % 40 == 0: print(f"  labeled {n}/{len(cand)}", flush=True)
cand["llm_label"] = labels
cand[COL] = cand["llm_label"]   # suggestion; the specialist validates/corrects this column

out = cand[[COL, "llm_label", "reason", "topic", "p", "doi", "title", "abstract", "year"]].rename(columns={"p": "model_P(yes)"})
out["_o"] = out["llm_label"].map({"no": 0, "": 1, "yes": 2}); out = out.sort_values(["_o", "model_P(yes)"]).drop(columns="_o")
out.to_excel(OUT, index=False)
print("LLM:", out.llm_label.value_counts().to_dict(), "->", OUT, flush=True)
