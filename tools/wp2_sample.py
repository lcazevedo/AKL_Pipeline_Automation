"""WP2: blind annotation sample for estimating the precision of the relevance filter and the share of relevant papers among the discarded.

Design (WP2 of the project protocol):
  - frame: all cleaned papers (Step 5 predictions) minus the documents that already have a specialist label (labels_relevance.csv);
  - strata from the P(relevant) of the relevance filter (kept: 0.5-0.8, 0.8-0.95, 0.95-1; discarded: 0-0.05, 0.05-0.2, 0.2-0.4, 0.4-0.5);
  - 250 papers: 150 kept + 100 discarded, shuffled; the first 100 are also annotated by the second annotator.
Outputs (data/03_review/wp2/):
  wp2_annotator_A.xlsx   250 items for the main annotator (no filter output, no stratum)
  wp2_annotator_B.xlsx   the first 100 items, for the second annotator
  wp2_design.csv         DO NOT SHARE with annotators: doi, id, stratum, stratum size, weight, P(relevant)
Run from the workspace root:  python tools/wp2_sample.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from openpyxl import load_workbook
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter

SEED = 20261104
OUT = Path("data/03_review/wp2"); OUT.mkdir(parents=True, exist_ok=True)
norm = lambda s: s.astype(str).str.lower().str.strip().str.replace("https://doi.org/", "", regex=False)

pred = pd.read_pickle("data/02_intermediate/step5_full_predictions.pkl")
pred["doi_n"] = norm(pred["doi"])
labeled = set(pd.read_csv("data/03_review/labels_relevance.csv")["doi"])
frame = pred[~pred["doi_n"].isin(labeled)].drop_duplicates("doi_n").copy()
kept = frame["predicted_relevance"].astype(str).str.lower() == "yes"
bins = [(True, 0.5, 0.8, 20), (True, 0.8, 0.95, 30), (True, 0.95, 1.0001, 100),
        (False, 0.0, 0.05, 30), (False, 0.05, 0.2, 30), (False, 0.2, 0.4, 20), (False, 0.4, 0.5, 20)]
rng = np.random.RandomState(SEED); parts = []
for is_kept, lo, hi, n in bins:
    m = (kept == is_kept) & (frame["proba_yes"] >= lo) & (frame["proba_yes"] < hi)
    pool = frame[m]; pick = pool.sample(n, random_state=rng)
    pick = pick.assign(stratum=f"{'kept' if is_kept else 'discarded'} P[{lo},{min(hi,1.0)})", N_h=len(pool), n_h=n, weight=len(pool) / n)
    parts.append(pick)
s = pd.concat(parts).sample(frac=1, random_state=rng).reset_index(drop=True)
s["id"] = np.arange(1, len(s) + 1); s["double_annotation"] = s["id"] <= 100
s[["id", "doi_n", "stratum", "N_h", "n_h", "weight", "proba_yes", "predicted_relevance", "double_annotation"]].rename(columns={"doi_n": "doi"}).to_csv(OUT / "wp2_design.csv", index=False)

INSTR = [
 ["WP2: blind relevance annotation (DRAFT of the rule book: ratify before annotating)"],
 [""],
 ["Judge each paper ONLY by its title and abstract. Do not consult model predictions or LLM suggestions."],
 ["Answer the three questions (yes or no) and mark 'em_duvida' (unsure) with 1 if, even with the rule book, the case is truly borderline."],
 ["Q1 does the paper use ML/AI as a method (or is it a review, benchmark, dataset or best-practice paper about ML in this domain)?"],
 ["Q2 is the object of study a substance, material, molecule or reaction (properties, structure, synthesis, discovery, design, characterization, catalysis, energy materials, ML potentials and force fields, QSPR)?"],
 ["Q3 does it fall under an exclusion: drugs and biological QSAR; a general ML method with chemistry only as one benchmark; control or monitoring of an industrial process or civil engineering without chemical insight; ML only mentioned in passing?"],
 ["RELEVANT = Q1 yes, Q2 yes and Q3 no. (The final column is computed later; you do not need to fill it in.)"],
 [""],
 ["Provisional decisions for recurring categories (write the number in the 'regra' (rule) column when it applies):"],
 ["1 industrial process: Q2 no | 2 battery state (SoC, SoH): Q2 no if only cell/BMS time series, yes if it involves electrode or electrolyte chemistry | 3 sensors with ML: Q2 yes if the ML analyzes signals to identify or quantify substances, no if it is device engineering"],
 ["4 synaptic/neuromorphic devices: Q1 no, unless ML is used to design or predict materials | 5 ML force fields for biomolecules: Q2 yes if the focus is molecular modeling methodology, Q3 yes if clinical | 6 analytical QSPR: Q2 yes"],
 ["7 QSAR of biological activity, drug-likeness, drug screening: Q3 yes | 8 civil and structural engineering: Q3 yes | 9 damage detection and non-destructive testing: Q3 yes | 10 optimization of manufacturing processes: Q2 yes if the target is material composition, microstructure or property, no if only machine parameters"],
 ["11 ML method with chemistry as one benchmark among several: Q3 yes | 12 review: relevant only if it is about ML in chemistry or materials | 13 bioprocess or protein engineering with ML: Q2 no (to be confirmed) | 14 thermophysical properties by QSPR: Q2 yes"],
 [""],
 ["Do not change the columns id, titulo (title), resumo (abstract), ano (year)."],
]
def build(df, name):
    x = pd.DataFrame({"id": df["id"], "titulo": df["title"], "resumo": df["abstract"], "ano": df["year"],
                      "q1_ML_metodo": "", "q2_objeto_quimica_materiais": "", "q3_exclusao": "", "em_duvida": "", "regra": "", "observacao": ""})
    path = OUT / name
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        pd.DataFrame(INSTR).to_excel(w, sheet_name="Instrucoes", index=False, header=False)
        x.to_excel(w, sheet_name="Itens", index=False)
    wb = load_workbook(path); ws = wb["Itens"]; wi = wb["Instrucoes"]
    wi.column_dimensions["A"].width = 160; wi["A1"].font = Font(bold=True, size=13)
    for r in wi.iter_rows():
        for c in r: c.alignment = Alignment(wrap_text=True, vertical="top")
    widths = {"A": 6, "B": 55, "C": 110, "D": 7, "E": 14, "F": 16, "G": 13, "H": 11, "I": 8, "J": 40}
    for k, v in widths.items(): ws.column_dimensions[k].width = v
    for row in ws.iter_rows(min_row=2):
        for c in row[:3]: c.alignment = Alignment(wrap_text=True, vertical="top")
    yn = DataValidation(type="list", formula1='"yes,no"', allow_blank=True); ws.add_data_validation(yn)
    for col in ("E", "F", "G"): yn.add(f"{col}2:{col}{ws.max_row}")
    dv = DataValidation(type="list", formula1='"0,1"', allow_blank=True); ws.add_data_validation(dv); dv.add(f"H2:H{ws.max_row}")
    ws.freeze_panes = "D2"; wb.save(path)
    print("wrote", path, len(x), "items")
build(s, "wp2_annotator_A.xlsx"); build(s[s["double_annotation"]], "wp2_annotator_B.xlsx")
print(s.groupby("stratum").agg(n=("id", "size"), N_h=("N_h", "first"), weight=("weight", "first")).round(1).to_string())
print("overlap with labeled docs:", int(s["doi_n"].isin(labeled).sum()), "| duplicate DOIs:", int(s["doi_n"].duplicated().sum()))
