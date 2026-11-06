"""Export the specialist's document-level relevance labels to one table (doi, label, source, round).

Sources (current workspace):
  - Step 4 review workbook (data/03_review/step4_relevance_review.xlsx): source "step4"
  - Step 8 boundary audit (data/03_review/step8_audit_boundary.xlsx):    source "step8_boundary_audit"
    (hard cases for the filter of that run: keep them as a held-out set, not as training data)

Output: data/03_review/labels_relevance.csv
Run from the workspace root:  python tools/export_labels.py
"""
import pandas as pd
from pathlib import Path

COL = "Relevance (yes/no)"
norm = lambda s: s.astype(str).str.lower().str.strip().str.replace("https://doi.org/", "", regex=False)

rows = []
wb = Path("data/03_review/step4_relevance_review.xlsx")
if wb.exists():
    for name, d in pd.read_excel(wb, sheet_name=None).items():
        if name in ("Topics", "All_Layers") or COL not in d.columns:
            continue
        x = d[d[COL].astype(str).str.strip().str.lower().isin(["yes", "no"])]
        rows.append(pd.DataFrame({"doi": norm(x["doi"]), "label": x[COL].astype(str).str.strip().str.lower(),
                                  "source": "step4", "round": "step4"}))
bd = Path("data/03_review/step8_audit_boundary.xlsx")
if bd.exists():
    d = pd.read_excel(bd, keep_default_na=False, na_values=[""])
    x = d[d["audit_label"].astype(str).str.lower().isin(["yes", "no"])]
    rows.append(pd.DataFrame({"doi": norm(x["doi"]), "label": x["audit_label"].astype(str).str.lower(),
                              "source": "step8_boundary_audit", "round": "step8"}))
out = pd.concat(rows, ignore_index=True).drop_duplicates("doi", keep="first")
out.to_csv("data/03_review/labels_relevance.csv", index=False)
print(out.groupby(["source", "label"]).size().to_string())
print("written data/03_review/labels_relevance.csv:", len(out), "labels")
