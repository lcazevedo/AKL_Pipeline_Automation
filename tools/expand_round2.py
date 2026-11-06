"""Round 2 of label expansion: documents near the SetFit decision boundary + a random sample of topic 23.
Output: data/03_review/step4c_extra_labels.xlsx (the specialist edits ONLY the 'Relevance (yes/no)' column)."""
import re, os
import numpy as np, pandas as pd, yaml, openai

COL = "Relevance (yes/no)"
REVIEW = "data/03_review/step4_relevance_review.xlsx"
OUT = os.environ.get("OUT_PATH", "data/03_review/step4c_extra_labels.xlsx")
N_BND, CAP_BND, LO, HI = 90, 8, 0.40, 0.60
TOPIC23_PREFIX, N_T23 = "23_", 30

cfg = yaml.safe_load(open("pipeline_config.yaml"))
pred = pd.read_pickle("data/02_intermediate/step5_full_predictions.pkl")
pred["k"] = pred["doi"].astype(str).str.lower().str.strip()

topic_of, labeled = {}, set()
for name, df in pd.read_excel(REVIEW, sheet_name=None).items():
    if name in ("Topics", "All_Layers") or COL not in df.columns:
        continue
    k = df["doi"].astype(str).str.lower().str.strip()
    for x in k: topic_of[x] = name
    lab = df[COL].astype(str).str.strip().str.lower().isin(["yes", "no"])
    labeled |= set(k[lab])
print("already labeled:", len(labeled), flush=True)

pool = pred[~pred.k.isin(labeled)].copy()
pool["topic"] = pool.k.map(topic_of)
pool = pool[pool.topic.notna()]

sel = {}
b = pool[(pool.proba_yes >= LO) & (pool.proba_yes <= HI)].assign(d=lambda x: (x.proba_yes - 0.5).abs()).sort_values("d")
cnt = {}
for i, r in b.iterrows():
    if cnt.get(r.topic, 0) >= CAP_BND: continue
    sel[i] = "boundary (0.4-0.6)"; cnt[r.topic] = cnt.get(r.topic, 0) + 1
    if len(sel) >= N_BND: break
t23 = pool[pool.topic.str.startswith(TOPIC23_PREFIX) & ~pool.index.isin(sel)]
for i in t23.sample(min(N_T23, len(t23)), random_state=11).index: sel[i] = "random, topic 23"

cand = pool.loc[list(sel)].copy(); cand["reason"] = pd.Series(sel)
print("candidates:", len(cand), cand.reason.value_counts().to_dict(), flush=True)

text = open("project_config.md", encoding="utf-8").read()
m = re.search(r"## 6\. Relevance criteria.*?## 7\.", text, re.DOTALL | re.IGNORECASE)
criteria = m.group(0).strip() if m else text
client = openai.OpenAI(api_key="not-needed", base_url=cfg["step7_vllm"]["base_url"])
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
        labels.append(""); print("LLM error:", str(e)[:100], flush=True)
    if n % 40 == 0: print(f"  labeled {n}/{len(cand)}", flush=True)
cand["LLM_suggestion (do not edit)"] = labels
cand[COL] = labels   # prefilled suggestion; the specialist corrects ONLY this column

out = cand[[COL, "LLM_suggestion (do not edit)", "reason", "topic", "proba_yes", "predicted_relevance", "doi", "title", "abstract", "year"]]
out = out.rename(columns={"proba_yes": "SetFit_P(yes)", "predicted_relevance": "SetFit_prediction"})
out.to_excel(OUT, index=False)
print("LLM suggestions:", pd.Series(labels).value_counts().to_dict(), "| SetFit vs LLM disagreement:",
      int((out["SetFit_prediction"] != out[COL]).sum()), "->", OUT, flush=True)
