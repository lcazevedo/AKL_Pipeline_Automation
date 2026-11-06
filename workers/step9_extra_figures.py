"""
step9_extra_figures.py
----------------------
Ad-hoc figures requested in data/04_reports/step8_visualizations_requested.md (priority A and B,
plus the Section-9 items not produced by the standard worker).

Run (after Step 8 and `workers/step8_extra_analyses.py`):
    ./run_worker.sh workers/step9_extra_figures.py            # all figures
    ./run_worker.sh workers/step9_extra_figures.py A1 A4      # selected ones

Outputs: data/04_reports/figures/ad_hoc/<id>_<name>.png (and .svg for A1/A2).
All time series stop at the last complete year (2025); 2026 is partial.
"""
import re, sys, textwrap, warnings
import os
from pathlib import Path
from collections import Counter
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import seaborn as sns

warnings.filterwarnings("ignore")
ROOT = Path(os.environ.get("AKL_WORKSPACE") or Path.cwd())   # the workspace (run folder); defaults to the current directory
OUT = ROOT / "data" / "04_reports" / "figures" / "ad_hoc"
OUT.mkdir(parents=True, exist_ok=True)
LAST = 2025

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200, "savefig.bbox": "tight", "font.size": 13,
                     "axes.labelsize": 14, "axes.titlesize": 14, "axes.titleweight": "bold",
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12, "legend.title_fontsize": 13})

# Okabe-Ito colour-blind-safe palette, one colour per regime across all figures
REG = {"rising share": "#009E73", "rising share (recent)": "#009E73", "stable share": "#0072B2",
       "peaked, share falling": "#E69F00", "declining share": "#D55E00"}
REG_LABEL = {"rising share": "Rising share", "stable share": "Stable share",
             "peaked, share falling": "Peaked, share falling", "declining share": "Declining share"}


def pretty(s):
    return str(s).replace("Dft Or", "DFT or").replace(" Or ", " or ").replace(" And ", " and ")


# ── data ──────────────────────────────────────────────────────────────────────
tabs = pd.read_pickle(ROOT / "data/02_intermediate/step8_extra_tables.pkl")
T, EV, X, EXP, AM, TOT, DD, CIT = (tabs[k] for k in ("T", "ev", "X", "exp", "am", "tot", "dd", "cit"))
corpus_all = pd.read_pickle(ROOT / "data/02_intermediate/step5_corpus_filtered_relevant.pkl")
corpus_all["k"] = corpus_all.doi.astype(str).str.lower().str.strip()
ext = pd.read_pickle(ROOT / "data/02_intermediate/step7_extractions.pkl")
ext["k"] = ext.doi.astype(str).str.lower().str.strip()

topic_file = next((ROOT / "data/05_exports").glob("step6_topics_final_layer_*_selected.xlsx"))
sheets = pd.read_excel(topic_file, sheet_name=None)
TOPICS = sheets["Topics"]
FULL = {str(r.Topic): str(r.New_Name).split("_", 1)[-1] for r in TOPICS.itertuples()}   # sheet names are cut at 31 chars
TERMS = {str(r.Topic): str(r.Aspect1) for r in TOPICS.itertuples()}


def short(sheet):
    return FULL.get(str(sheet).split("_", 1)[0], str(sheet).split("_", 1)[-1])


tm = []
for n, d in sheets.items():
    if n in ("Topics", "All_Layers") or "doi" not in d.columns:
        continue
    tm.append(pd.DataFrame({"k": d.doi.astype(str).str.lower().str.strip(), "topic": short(n), "tid": n.split("_", 1)[0]}))
tm = pd.concat(tm).drop_duplicates("k")
C = corpus_all.merge(tm, on="k", how="left")
C_t = C[C.topic.notna() & C.year.between(2000, LAST)].copy()
YEARS = list(range(2000, LAST + 1))
TOTY = C_t.groupby("year").size().reindex(YEARS, fill_value=0)
T = T.assign(name=T.topic.map(short)).set_index("name")
EV = EV.assign(name=EV.index.map(short)).set_index("name")
FIELD_GROWTH = 100 * (np.exp(np.polyfit(np.arange(2015, LAST + 1), np.log(TOTY.loc[2015:LAST].values), 1)[0]) - 1)


def save(fig, name, svg=False, keep_panel_titles=False):
    """Figures carry no titles (title and description go in the caption, as in a LaTeX figure);
    multi-panel figures get panel letters instead of panel titles."""
    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.set_text("")
    panels = [a for a in fig.axes if a.get_label() != "<colorbar>" and a.axison]
    if not keep_panel_titles:
        for a in fig.axes:
            a.set_title("")
    if len(panels) > 1 and not keep_panel_titles:
        for i, a in enumerate(panels):
            a.text(-0.02, 1.03, f"({chr(97 + i)})", transform=a.transAxes, fontsize=15, fontweight="bold", ha="right", va="bottom")
    fig.savefig(OUT / f"{name}.png")
    if svg:
        fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)
    print("saved", name)


def repel(xs, ys, texts, ax, iters=200, step=0.012):
    """Tiny label de-overlap in axes-fraction space."""
    inv = ax.transAxes.inverted()
    pts = inv.transform(ax.transData.transform(np.c_[xs, ys]))
    pos = pts.copy() + np.array([0.008, 0.008])
    for _ in range(iters):
        moved = False
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                dx, dy = pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1]
                if abs(dx) < 0.16 and abs(dy) < 0.035:
                    sy = step if dy >= 0 else -step
                    pos[i, 1] += sy; pos[j, 1] -= sy; moved = True
        if not moved:
            break
    for (px, py), (x0, y0), t in zip(pos, pts, texts):
        right = px > 0.68      # labels near the right edge are drawn to the left of their anchor so they stay inside the axes
        ax.annotate(t, xy=(x0, y0), xycoords="axes fraction", xytext=(min(px, 0.985) if right else px, py), textcoords="axes fraction", ha="right" if right else "left", fontsize=10.9,
                    arrowprops=dict(arrowstyle="-", color="#888", lw=0.5), zorder=5)


# ── A0 topic map (replaces the Step 6 image: no title, de-overlapped labels, larger font) ─────
def A0():
    cache = ROOT / "data/02_intermediate/step9_umap2d.npy"
    assign = pd.read_pickle(ROOT / "data/checkpoints/step6_layer_assignments.pkl")
    layer = int(topic_file.stem.split("layer_")[1].split("_")[0])
    tids = np.asarray(assign[layer])
    if cache.exists():
        xy = np.load(cache)
    else:
        import umap
        emb = pd.read_pickle(ROOT / "data/checkpoints/step6_embeddings.pkl")
        xy = umap.UMAP(n_components=2, random_state=42).fit_transform(emb)      # same call as Step 6
        np.save(cache, xy)
    assert len(xy) == len(tids), "embeddings and assignments are not aligned"
    names = {int(r.Topic): str(r.New_Name).split("_", 1)[-1] for r in TOPICS.itertuples()}
    ids = sorted(set(tids))
    pal = sns.color_palette("husl", len(ids)); rng = np.random.RandomState(1); order = rng.permutation(len(ids))
    col = {t: pal[order[i]] for i, t in enumerate(ids)}
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.scatter(xy[:, 0], xy[:, 1], s=3, c=[col[t] for t in tids], alpha=0.55, linewidths=0, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values(): sp.set_visible(False)
    xl, yl = ax.get_xlim(), ax.get_ylim()
    cen = {t: np.median(xy[tids == t], axis=0) for t in ids}
    lab = [t for t in ids if t in names]
    pos = {t: np.array([(cen[t][0] - xl[0]) / (xl[1] - xl[0]), (cen[t][1] - yl[0]) / (yl[1] - yl[0])]) for t in lab}
    anchor = {t: pos[t].copy() for t in lab}
    wid = {t: 0.0105 * len(names[t]) for t in lab}; hgt = 0.026
    for _ in range(600):
        moved = False
        for i, a in enumerate(lab):
            for b in lab[i + 1:]:
                dx, dy = pos[a][0] - pos[b][0], pos[a][1] - pos[b][1]
                ox, oy = (wid[a] + wid[b]) / 2 - abs(dx), hgt * 1.15 - abs(dy)
                if ox > 0 and oy > 0:
                    if oy < ox * 0.35:
                        sy = 0.5 * oy * (1 if dy >= 0 else -1); pos[a][1] += sy; pos[b][1] -= sy
                    else:
                        sx = 0.5 * ox * (1 if dx >= 0 else -1) * 0.6; pos[a][0] += sx; pos[b][0] -= sx
                    moved = True
        for t in lab:
            pos[t] = np.clip(pos[t], [wid[t] / 2 + 0.005, 0.02], [1 - wid[t] / 2 - 0.005, 0.98])
        if not moved: break
    for t in lab:
        far = np.linalg.norm(pos[t] - anchor[t]) > 0.03
        ax.annotate(names[t], xy=anchor[t], xycoords="axes fraction", xytext=pos[t], textcoords="axes fraction", ha="center", va="center",
                    fontsize=12, color="#111", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.72),
                    arrowprops=dict(arrowstyle="-", color="#777", lw=0.6) if far else None)
    save(fig, "A0_topic_map")


# ── A1 ripe-or-hype map ───────────────────────────────────────────────────────
def A1():
    d = T.join(EV[["exp_validation_%", "named_benchmark_%", "dft_sim_data_%"]])
    d["x"] = d["growth_yr_pct"] - FIELD_GROWTH
    d["y"] = d["exp_validation_%"]
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.axvline(0, color="#555", lw=0.8); ax.axhline(d.y.median(), color="#555", lw=0.8, ls="--")
    for reg, g in d.groupby(d.share_trend.replace({"rising share (recent)": "rising share"})):
        ax.scatter(g.x, g.y, s=g.n / 6, c=REG[reg], alpha=0.8, edgecolor="white", lw=0.6)
    hype = ["Electrocatalysis Hydrogen", "Lithium Dendrite Interface", "Generative Metamaterials", "DFT Molecular Theory",
            "Perovskite Lattice Prediction", "Metamaterial Plasmonics"]
    promising = ["Alloy Solidification", "Piezoelectric Nanogenerators", "Titanium Alloy Microstructure", "Lithium Electrode Materials",
                 "Battery Voltage Estimation", "Alloy Ductility Optimization", "MOF Adsorption"]
    mature = ["Nanocluster Alloys", "Molecular Descriptor Prediction", "Superconductivity Pressure", "Machining Optimization",
              "Superalloy Recrystallization", "Methane Hydrogenation", "Nonadiabatic Molecular Reactions"]
    lab = [n for n in hype + promising + mature if n in d.index]
    repel(d.loc[lab, "x"].values, d.loc[lab, "y"].values, lab, ax)
    ax.set_xlim(ax.get_xlim()[0] - 2, ax.get_xlim()[1] + 14)
    xl, yl = ax.get_xlim(), ax.get_ylim(); ym = d.y.median()
    ax.text(xl[1], yl[1], "PROMISING\n(gaining share, well validated)", ha="right", va="top", color="#009E73", fontsize=13.0, fontweight="bold")
    ax.text(xl[1], yl[0], "PROBABLE HYPE\n(gaining share, thin validation)", ha="right", va="bottom", color="#D55E00", fontsize=13.0, fontweight="bold")
    ax.text(xl[0], yl[1], "MATURE, EVIDENCE-RICH", ha="left", va="top", color="#0072B2", fontsize=13.0, fontweight="bold")
    ax.text(xl[0], yl[0], "MATURE, MOSTLY COMPUTATIONAL", ha="left", va="bottom", color="#666", fontsize=13.0, fontweight="bold")
    ax.set_xlabel(f"Yearly growth of paper counts (2015-{LAST}) minus whole-field growth ({FIELD_GROWTH:.0f}% per year), percentage points")
    ax.set_ylabel("Papers with experimental validation (% of the topic)")
    ax.set_title("Ripe or hype: growth relative to the field versus experimental validation, 46 topics")
    leg = ax.legend(handles=[Line2D([], [], marker="o", ls="", color=REG[k], markersize=9, label=REG_LABEL[k]) for k in REG_LABEL],
                    title="Share trend", loc="upper left", bbox_to_anchor=(1.02, 0.95), frameon=False)
    ax.add_artist(leg)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color="#999", markersize=np.sqrt(n / 6), label=f"{n} papers") for n in (300, 1000, 1800)],
              title="Bubble size", loc="upper left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    pass
    save(fig, "A1_ripe_or_hype_map", svg=True)


# ── A2 topic share small multiples ────────────────────────────────────────────
def A2():
    cnt = C_t.groupby(["topic", "year"]).size().unstack(fill_value=0).reindex(columns=YEARS, fill_value=0)
    share = 100 * cnt.div(TOTY, axis=1)
    order = T.assign(r=T.share_trend.map({"rising share": 0, "rising share (recent)": 0, "stable share": 1, "peaked, share falling": 2, "declining share": 3})) \
             .sort_values(["r", "n"], ascending=[True, False]).index.tolist()
    fig, axes = plt.subplots(8, 6, figsize=(22, 26), sharex=True)
    for ax, t in zip(axes.flat, order):
        reg = T.loc[t, "share_trend"]
        ax.set_facecolor(matplotlib.colors.to_rgba(REG[reg], 0.10))
        ax.plot(YEARS, share.loc[t].values, color=REG[reg], lw=1.6)
        ax.set_title(t, fontsize=11.6)
        ax.set_xlim(2005, LAST); ax.tick_params(labelsize=10.2)
        ax.set_ylim(0, max(1, share.loc[t].loc[2005:].max() * 1.15))
    for ax in list(axes.flat)[len(order):]:
        ax.axis("off")
    fig.legend(handles=[Line2D([0], [0], color=c, lw=3, label=l) for l, c in ((REG_LABEL[k], REG[k]) for k in REG_LABEL)],
               loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("Share of the yearly corpus by topic, 2005-2025 (% of papers; 2026 excluded: partial year)", y=1.0, fontsize=18.8, fontweight="bold")
    fig.supylabel("% of the yearly corpus"); fig.tight_layout(rect=(0, 0, 1, 0.975))
    save(fig, "A2_topic_share_over_time", svg=True, keep_panel_titles=True)


# ── A3 lifecycle map ──────────────────────────────────────────────────────────
def A3():
    d = T.copy(); d["delta"] = d.share23_25 - d.share14_16
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.axhline(0, color="#555", lw=0.8)
    for reg, g in d.groupby("share_trend"):
        ax.scatter(g.share23_25, g.delta, s=g.n / 5, c=REG[reg], alpha=0.8, edgecolor="white", lw=0.6)
    big = d[(d.delta.abs() > 1.2) | (d.share23_25 > 3.6)].index.tolist()
    repel(d.loc[big, "share23_25"].values, d.loc[big, "delta"].values, big, ax)
    ax.set_xlabel("Share of the yearly corpus, 2023-25 (%)"); ax.set_ylabel("Change in share since 2014-16 (percentage points)")
    ax.set_title("Topic lifecycle map: where each topic stands and where it is heading")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=REG[k], markersize=10, label=REG_LABEL[k]) for k in REG_LABEL], frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), title="Share trend")
    save(fig, "A3_lifecycle_map")


# ── A4 application x method heatmap ───────────────────────────────────────────
def A4():
    ratio = np.log2((X.values + 0.5) / (EXP + 0.5))
    cols = [pretty(c) for c in X.columns]
    fig, ax = plt.subplots(figsize=(18, 10))
    sns.heatmap(pd.DataFrame(ratio, index=[pretty(i) for i in X.index], columns=cols), cmap="RdBu_r", center=0, vmin=-2, vmax=2,
                annot=X.values, fmt="d", annot_kws={"fontsize": 10.5}, cbar_kws={"label": "log2(observed / expected papers)"}, ax=ax, linewidths=0.4, linecolor="white")
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if EXP[i, j] >= 30 and X.values[i, j] < 0.5 * EXP[i, j]:
                ax.add_patch(Rectangle((j, i), 1, 1, fill=False, ec="black", lw=1.6))
    ax.set_title("Application x method: paper counts (numbers) and deviation from independence (colour); boxes = under 50% of expected, expected >= 30")
    ax.set_xlabel("ML method category"); ax.set_ylabel("Application task")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    pass
    save(fig, "A4_application_method_heatmap")


# ── A5 method mix over time ───────────────────────────────────────────────────
def A5():
    am = AM[AM.year.between(2005, LAST)]
    top = am.method.value_counts().head(8).index.tolist()
    m = am.assign(method=am.method.where(am.method.isin(top), "Other methods"))
    mix = pd.crosstab(m.year, m.method, normalize="index") * 100
    mix = mix[top + ["Other methods"]]
    pal = sns.color_palette("colorblind", len(mix.columns))
    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.stackplot(mix.index, mix.T.values, labels=[pretty(c) for c in mix.columns], colors=pal, alpha=0.95)
    ax.set_xlim(2005, LAST); ax.set_ylim(0, 100); ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_ylabel("% of papers"); ax.set_xlabel("Year")
    ax.set_title("Method mix over time: genetic algorithms and small networks give way to generic ML and deep learning")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, title="ML method")
    save(fig, "A5_method_mix_over_time")


# ── A6 future directions ──────────────────────────────────────────────────────
THEMES = {"Physics-informed / hybrid": r"physics-informed|physics-based|hybrid", "Transfer learning / foundation models": r"transfer learning|foundation|pre-?train",
          "Language models": r"\b(?:llm|large language|language model|chatgpt|gpt)", "Autonomous / self-driving labs": r"autonom|self-driving|closed-loop|robot",
          "Inverse design / generative": r"inverse design|generative|diffusion", "Interpretability": r"interpret|explainab",
          "Active learning / Bayesian opt.": r"active learning|bayesian optim", "Data sharing / benchmarks": r"benchmark|standard|database|dataset|data sharing"}


def A6():
    d = ext[ext.vector == "directions"][["k", "micro_label", "macro_category"]].merge(C_t[["k", "year"]], on="k")
    ml = d.micro_label.astype(str).str.lower(); docs = d.groupby("year").k.nunique()   # papers with >= 1 filtered direction sentence (as in the synthesis)
    yrs = list(range(2012, LAST + 1))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(19, 8), gridspec_kw={"width_ratios": [1.15, 1]})
    pal = sns.color_palette("colorblind", len(THEMES))
    for (nme, rx), col in zip(THEMES.items(), pal):
        s = d[ml.str.contains(rx, regex=True)].groupby("year").k.nunique().reindex(yrs, fill_value=0)
        ser = (100 * s / docs.reindex(yrs)).rolling(2, min_periods=1).mean()
        a1.plot(yrs, ser.values, label=nme, color=col, lw=2)
    a1.set_ylabel("% of papers with a direction sentence (2-year moving average)"); a1.set_xlabel("Year")
    a1.set_title("Themes named in future-direction sentences"); a1.legend(frameon=False, fontsize=11.6, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    dd = DD.copy(); dd.index = [pretty(i) for i in dd.index]
    dd = dd[dd.total >= 400].sort_values("ratio")
    r = np.log2(dd.ratio)
    a2.barh(dd.index, r, color=["#009E73" if v > 0 else "#D55E00" for v in r])
    a2.axvline(0, color="#333", lw=0.8); a2.set_xlabel("log2(share 2023-25 / share 2015-20)")
    a2.set_title("Direction categories gaining or losing share")
    fig.subplots_adjust(wspace=0.75)
    fig.suptitle("Future directions after sentence filtering: read trends, not absolute levels (2026 excluded)", fontweight="bold")
    save(fig, "A6_future_directions")


# ── A7 bottleneck heatmaps ────────────────────────────────────────────────────
def A7():
    rows_per_topic = ext[ext.vector == "challenges"][["k"]].merge(C[["k", "topic"]].dropna(), on="k").topic.value_counts()
    top = rows_per_topic.head(10).index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(21, 8))
    for ax, vec, ncat, title in ((axes[0], "challenges", 8, "ML and data challenges"), (axes[1], "application_challenges", 8, "Domain challenges")):
        d = ext[ext.vector == vec][["k", "macro_category"]].merge(C[["k", "topic"]].dropna(), on="k")
        d = d[d.topic.isin(top)]
        cats = d.macro_category.value_counts().head(ncat).index
        ct = pd.crosstab(d.topic, d.macro_category, normalize="index").reindex(top)[cats] * 100
        ct.columns = [pretty(c) for c in ct.columns]
        sns.heatmap(ct, annot=True, fmt=".0f", annot_kws={"size": 12}, cmap="YlOrRd", ax=ax, cbar_kws={"label": "% of the topic's mentions", "pad": 0.02}, linewidths=0.4, linecolor="white")
        for i, t in enumerate(ct.index):
            j = int(np.argmax(ct.loc[t].values)); ax.add_patch(Rectangle((j, i), 1, 1, fill=False, ec="black", lw=1.8))
        ax.set_title(title); ax.set_ylabel(""); ax.set_xlabel("")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.subplots_adjust(wspace=0.5)
    fig.suptitle("Bottlenecks by topic (10 topics with most challenge sentences, all years, row-normalised; box = leading category)", fontweight="bold")
    save(fig, "A7_bottleneck_heatmaps")


# ── B1 data source and validation by topic ────────────────────────────────────
def B1():
    def comp(vec, cats):
        d = ext[ext.vector == vec][["k", "macro_category"]].drop_duplicates("k").merge(C_t[["k", "topic"]], on="k")
        return (pd.crosstab(d.topic, d.macro_category, normalize="index") * 100).reindex(columns=cats, fill_value=0)
    ds = comp("data_source", ["Dft Or Simulation", "Experiments", "Mixed", "Literature Text Mining", "Public Database", "Industrial Data"])
    va = comp("validation", ["Experimental Validation", "Computational Benchmark", "Cross-Validation Only", "No Validation"])
    order = ds.sort_values("Dft Or Simulation").index.tolist()
    bm = C_t.assign(b=C_t.k.isin(ext[ext.vector == "benchmarks_datasets"].k)).groupby("topic").b.mean() * 100
    fig, axes = plt.subplots(1, 3, figsize=(21, 15), sharey=True, gridspec_kw={"width_ratios": [1.3, 1.3, 0.6]})
    pal = sns.color_palette("colorblind")
    ds.loc[order].rename(columns=pretty).plot.barh(stacked=True, ax=axes[0], color=pal[:6], width=0.85, legend=True)
    va.loc[order].rename(columns=pretty).plot.barh(stacked=True, ax=axes[1], color=[pal[2], pal[0], pal[1], pal[3]], width=0.85, legend=True)
    axes[2].barh(order, bm.reindex(order).values, color="#555", height=0.6)
    axes[0].set_title("Where the training data comes from"); axes[1].set_title("How results are validated"); axes[2].set_title("Papers naming a benchmark (%)")
    for a in axes[:2]:
        a.set_xlabel("% of the topic's papers"); a.legend(fontsize=10.2, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    axes[2].set_xlabel("%")
    fig.suptitle("Data source, validation and benchmark coverage by topic (sorted by share of simulation data)", fontweight="bold", y=0.93)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "B1_data_source_validation_by_topic")


# ── B2 pioneers and geography ─────────────────────────────────────────────────
def B2():
    c = C_t.assign(country=C_t.first_country.fillna("Unknown"), early=C_t.year <= 2012)
    cs = c.groupby(["country", "early"]).size().unstack(fill_value=0)
    cs = 100 * cs / cs.sum()
    tot = c.country.value_counts()
    show = [x for x in tot.index if x != "Unknown"][:10]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(20, 12), gridspec_kw={"width_ratios": [1, 1.25]})
    fig.subplots_adjust(wspace=0.6)
    def spread(vals, gap=0.85):
        """vertical label positions with a minimum spacing"""
        order = sorted(vals, key=vals.get); pos, last = {}, -1e9
        for k in order:
            pos[k] = max(vals[k], last + gap); last = pos[k]
        return pos
    pe, pl = spread({c_: cs.loc[c_, True] for c_ in show}), spread({c_: cs.loc[c_, False] for c_ in show})
    for ctry in show:
        e, l = cs.loc[ctry, True], cs.loc[ctry, False]
        col = "#009E73" if l > e else "#D55E00"
        a1.plot([0, 1], [e, l], marker="o", color=col, lw=2)
        a1.text(-0.04, pe[ctry], f"{ctry} {e:.0f}%", ha="right", va="center", fontsize=11.6); a1.text(1.04, pl[ctry], f"{ctry} {l:.0f}%", ha="left", va="center", fontsize=11.6)
    a1.set_xlim(-1.0, 2.0); a1.set_xticks([0, 1]); a1.set_xticklabels(["Papers up to 2012", f"Papers 2013-{LAST}"]); a1.set_ylabel("% of papers (first-author country)")
    a1.set_title("Country share, early versus later"); a1.grid(False)
    fy = T.sort_values("first_year_5p")
    cm = {k: REG[k] for k in REG}
    a2.scatter(fy.first_year_5p, range(len(fy)), c=[cm[r] for r in fy.share_trend], s=fy.n / 8 + 15)
    a2.set_yticks(range(len(fy))); a2.set_yticklabels(fy.index, fontsize=11); a2.set_xlabel("First year with at least 5 papers"); a2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    a2.set_title("When each topic took off"); a2.invert_yaxis()
    unk = 100 * (c[c.early].country == "Unknown").mean()
    pass
    save(fig, "B2_pioneers_and_geography")


# ── B3 growth-rate forest plot ────────────────────────────────────────────────
def B3():
    d = T.sort_values("growth_yr_pct")
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.hlines(range(len(d)), d.growth_lo, d.growth_hi, color="#999", lw=1.2)
    ax.scatter(d.growth_yr_pct, range(len(d)), c=[REG[r] for r in d.share_trend], s=36, zorder=3)
    ax.axvline(FIELD_GROWTH, color="#333", ls="--", lw=1); ax.text(FIELD_GROWTH + 0.5, -1.2, f"whole field {FIELD_GROWTH:.0f}%", fontsize=11.6)
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d.index, fontsize=10.9)
    ax.set_xlabel(f"Yearly growth of paper counts, 2015-{LAST} (%, 95% interval of the fitted slope)")
    ax.set_title("Growth rate by topic (a description of the past trend, not a forecast)")
    save(fig, "B3_growth_rate_forest")


# ── B4 citation effects ───────────────────────────────────────────────────────
def B4():
    md = (ROOT / "data/04_reports/step8_data_crunching.md").read_text(encoding="utf-8")
    rows = re.findall(r"\| topic_(\d+)_([^|]+?) \| (-?[\d.]+) \| [\d.]+ \| \[(-?[\d.]+), (-?[\d.]+)\] \|", md)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(19, 8))
    if rows:
        r = pd.DataFrame(rows, columns=["id", "name", "coef", "lo", "hi"]); r[["coef", "lo", "hi"]] = r[["coef", "lo", "hi"]].astype(float)
        r["name"] = r.id.map(lambda i: FULL.get(i, i)); r = r.sort_values("coef")
        a1.hlines(range(len(r)), r.lo, r.hi, color="#999"); a1.scatter(r.coef, range(len(r)), c=["#009E73" if v > 0 else "#D55E00" for v in r.coef], zorder=3)
        a1.axvline(0, color="#333", lw=0.8); a1.set_yticks(range(len(r))); a1.set_yticklabels(r.name)
        a1.set_xlabel("Effect on log(1 + citations), controlling for year and journal IF (bootstrap 95% CI)")
        a1.set_title("Citation effect of topic (regression audit, 10 largest topics)")
    cit = C_t.groupby("topic").citations_per_year.median().sort_values()
    a2.barh(cit.index, cit.values, color="#0072B2"); a2.tick_params(axis="y", labelsize=9.5)
    a2.set_xlabel("Median citations per year"); a2.set_title("Median citations per year by topic (recent papers are underexposed)")
    save(fig, "B4_citation_effects")


# ── B5 audit and filter transparency ──────────────────────────────────────────
def B5():
    rep = (ROOT / "data/04_reports/step8_audit_report.md").read_text(encoding="utf-8")
    acc = {m.group(1): float(m.group(2)) for m in re.finditer(r"## (\w+)\n.*?Accuracy: \*\*([\d.]+)\*\*", rep, re.S)}
    bk = pd.read_csv(ROOT / "data/04_reports/step8_filter_bakeoff.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(19, 7.5), gridspec_kw={"width_ratios": [0.8, 1.4]})
    names = {"challenges_micro": "Challenge labels\n(exact match)", "directions_micro": "Direction labels\n(exact match)", "boundary": "Relevance filter\non boundary papers"}
    vals = [acc.get(k, np.nan) for k in names]
    a1.bar(list(names.values()), vals, color=["#0072B2", "#D55E00", "#E69F00"])
    for i, v in enumerate(vals):
        a1.text(i, v + 0.015, f"{v:.2f}", ha="center")
    a1.set_ylim(0, 1); a1.set_ylabel("Agreement with the specialist (100 items each)"); a1.set_title("Audit results")
    b = bk[bk.model.isin(["DanL (PubMedBERT)", "Gemma verifier"])].copy()
    b["set"] = b.set.map({"challenges_micro": "Challenges", "directions_micro": "Directions"})
    b = b.melt(id_vars=["set", "model"], value_vars=["AUC", "P", "R"], var_name="metric", value_name="v")
    b["metric"] = b.metric.map({"AUC": "AUC", "P": "Precision @0.5", "R": "Recall @0.5"})
    sns.barplot(data=b, x="metric", y="v", hue=b.set + " | " + b.model, ax=a2, palette="colorblind")
    a2.set_ylim(0, 1.05); a2.set_xlabel(""); a2.set_ylabel(""); a2.set_title("Sentence filters on the 200 audited sentences"); a2.legend(fontsize=11.6, frameon=False)
    pass
    save(fig, "B5_audit_and_filter_transparency")


# ── S1 publications per year (Section 9 standard list) ────────────────────────
def S1():
    y = corpus_all[corpus_all.year.between(2000, 2026)].groupby("year").size()
    fig, ax = plt.subplots(figsize=(12, 6.5))
    bars = ax.bar(y.index, y.values, color=["#0072B2" if i <= LAST else "#BBBBBB" for i in y.index])
    bars[-1].set_hatch("//"); ax.set_ylabel("Papers"); ax.set_xlabel("Year")
    ax.text(2026, y.loc[2026], "2026\n(partial)", ha="center", va="bottom", fontsize=11.6)
    ax.set_title(f"Publications per year in the filtered corpus (n = {int(y.sum())})")
    save(fig, "S1_publications_per_year")


# ── S2 term panels per topic (stand-in for word clouds) ───────────────────────
def S2():
    top = T.sort_values("n", ascending=False).head(12)
    inv = {v: k for k, v in FULL.items()}
    fig, axes = plt.subplots(3, 4, figsize=(20, 12.5))
    for ax, (name, r) in zip(axes.flat, top.iterrows()):
        terms = [t.strip() for t in TERMS.get(inv.get(name, ""), "").split(",") if t.strip() not in ("abstract",)][:10]
        for i, t in enumerate(terms):
            ax.text(0.5, 0.93 - i * 0.092, t, ha="center", va="center", fontsize=15 - i * 0.9, color=sns.color_palette("viridis", 12)[i], transform=ax.transAxes)
        ax.set_title(f"{name} (n={int(r.n)})", fontsize=13.0); ax.axis("off")
    fig.suptitle("Top terms of the 12 largest topics (c-TF-IDF rank; larger = more characteristic)", fontweight="bold")
    save(fig, "S2_topic_terms", keep_panel_titles=True)


# ── S3 keyword co-occurrence network ──────────────────────────────────────────
def S3():
    import networkx as nx
    kw = C_t.keywords.dropna().astype(str).str.lower().str.split(";")
    generic = {"machine learning", "deep learning", "artificial neural networks", "neural networks", "artificial neural network", "neural network", "artificial intelligence"}
    docs = [sorted({k.strip() for k in ks if k.strip() and k.strip() not in generic}) for ks in kw]
    freq = Counter(k for ks in docs for k in ks)
    keep = {k for k, _ in freq.most_common(45)}
    co = Counter(p for ks in docs for p in combinations([k for k in ks if k in keep], 2))
    G = nx.Graph()
    for k in keep: G.add_node(k, n=freq[k])
    for (a, b), w in co.items():
        if w >= 10: G.add_edge(a, b, weight=w)
    G.remove_nodes_from([n for n in list(G) if G.degree(n) == 0])
    comm = list(nx.algorithms.community.greedy_modularity_communities(G, weight="weight"))
    pal = sns.color_palette("colorblind", len(comm)); col = {n: pal[i] for i, c in enumerate(comm) for n in c}
    pos = nx.spring_layout(G, k=2.2, seed=3, weight="weight", iterations=300)
    pxs = np.array([p_[0] for p_ in pos.values()]); pys = np.array([p_[1] for p_ in pos.values()])
    pos = {n: np.array([0.06 + 0.88 * (p_[0] - pxs.min()) / (pxs.max() - pxs.min()), 0.06 + 0.88 * (p_[1] - pys.min()) / (pys.max() - pys.min())]) for n, p_ in pos.items()}
    fig, ax = plt.subplots(figsize=(20, 15)); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    nx.draw_networkx_edges(G, pos, alpha=0.25, width=[G[u][v]["weight"] / 12 for u, v in G.edges()], ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=[G.nodes[n]["n"] * 5 for n in G], node_color=[col[n] for n in G], alpha=0.9, ax=ax)
    # labels: de-overlap in normalised coordinates, white halo, leader line when moved
    lp = {n: pos[n].copy() for n in G}
    anchor = {n: lp[n].copy() for n in G}; wid = {n: 0.0085 * len(n) for n in G}; hgt = 0.024
    nodes = list(G)
    for _ in range(500):
        moved = False
        for a_i, a in enumerate(nodes):
            for b in nodes[a_i + 1:]:
                dx, dy = lp[a][0] - lp[b][0], lp[a][1] - lp[b][1]
                ox, oy = (wid[a] + wid[b]) / 2 - abs(dx), hgt * 1.1 - abs(dy)
                if ox > 0 and oy > 0:
                    if oy < ox * 0.4:
                        sy = 0.5 * oy * (1 if dy >= 0 else -1); lp[a][1] += sy; lp[b][1] -= sy
                    else:
                        sx = 0.3 * ox * (1 if dx >= 0 else -1); lp[a][0] += sx; lp[b][0] -= sx
                    moved = True
        if not moved: break
    for n in G:
        far = np.linalg.norm(lp[n] - anchor[n]) > 0.035
        ax.annotate(n, xy=(pos[n][0], pos[n][1]), xycoords="axes fraction", xytext=(lp[n][0], lp[n][1]), textcoords="axes fraction", ha="center", va="center",
                    fontsize=11, color="#111", zorder=6, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7),
                    arrowprops=dict(arrowstyle="-", color="#888", lw=0.5) if far else None)
    ax.axis("off")
    save(fig, "S3_keyword_cooccurrence_network")


ALL = {"A0": A0, "A1": A1, "A2": A2, "A3": A3, "A4": A4, "A5": A5, "A6": A6, "A7": A7, "B1": B1, "B2": B2, "B3": B3, "B4": B4, "B5": B5, "S1": S1, "S2": S2, "S3": S3}

if __name__ == "__main__":
    want = [a.upper() for a in sys.argv[1:]] or list(ALL)
    for k in want:
        try:
            ALL[k]()
        except Exception as e:
            print(f"FAILED {k}: {type(e).__name__}: {e}")
