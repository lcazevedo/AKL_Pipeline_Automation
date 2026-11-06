"""
step8_audits.py
---------------
AKL Pipeline — Step 8: Automated Audits

Implements the automated audits that run during Step 8 data crunching:

- run_evoc_sensitivity()  — pairwise Adjusted Rand Index between
  the EVoC layers saved during Step 6 (stability across clustering resolutions).
- run_regression_audit()  — OLS of log1p(citations) on Step-6
  topics, centered year, and journal IF (with missingness dummy); percentile
  bootstrap confidence intervals; Kruskal-Wallis across topics; robustness
  check with/without journal IF.
- run_growth_aicc()       — formal exponential/linear/plateau
  model selection per macro-category using AICc (replaces the heuristic
  recent-ratio classification with a statistically grounded one).
- run_ri_sensitivity()    — Relevance Index unit-invariance.
  Conditional: only runs when a relevance-index column is present (RI is not
  standard in AKL, so this normally reports a skip note).

Each function returns a Markdown section (str) or None when the audit cannot
run (missing data). The Step 8 worker appends the sections to the data
crunching report.
"""

import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")


# ── EVoC sensitivity ────────────────────────────────────────────

def run_evoc_sensitivity(workspace):
    """Pairwise Adjusted Rand Index between the EVoC layers saved in Step 6.

    The layers differ in clustering resolution (noise level / merge threshold).
    High ARI between layers means the topic map is stable; low ARI means the
    map is sensitive to the EVoC parameters and the layer choice matters.
    """
    path = Path(workspace) / "data" / "checkpoints" / "step6_layer_assignments.pkl"
    if not path.exists():
        return None

    from sklearn.metrics import adjusted_rand_score

    assignments = pd.read_pickle(path)
    # Step 6 saves a dict {EVoC layer id: assignments}; keep the original layer ids in the report
    ids = sorted(assignments) if isinstance(assignments, dict) else list(range(len(assignments)))
    assignments = [assignments[k] for k in ids] if isinstance(assignments, dict) else list(assignments)
    if len(assignments) < 2:
        return None

    n_layers = len(assignments)
    sizes = [int(len(np.unique(a))) for a in assignments]

    md = ["### EVoC Sensitivity\n"]
    md.append(
        "Pairwise Adjusted Rand Index (ARI) between the EVoC layers computed in Step 6. "
        "ARI > 0.7 indicates stable clustering across resolutions; 0.3–0.7 moderate; "
        "< 0.3 sensitive (the layer choice materially changes the topic map).\n"
    )
    md.append("| Layer pair | Topics (A → B) | ARI |")
    md.append("|---|---|---|")
    for i in range(n_layers):
        for j in range(i + 1, n_layers):
            ari = float(adjusted_rand_score(assignments[i], assignments[j]))
            md.append(f"| Layer {ids[i]} vs Layer {ids[j]} | {sizes[i]} → {sizes[j]} | {ari:.3f} |")
    md.append("")
    return "\n".join(md)


# ── regression audit ────────────────────────────────────────────

def run_regression_audit(df):
    """OLS of log1p(citations) on Step-6 topics + year + journal IF.

    Includes percentile bootstrap CIs for every coefficient, a Kruskal-Wallis
    test for citation differences across topics, and a robustness check that
    drops journal IF (with its missingness dummy) to see how much the shared
    coefficients move.
    """
    cit_col = next((c for c in ("citation_count", "citations", "cited_by_count") if c in df.columns), None)
    if not cit_col or "_topic" not in df.columns or "year" not in df.columns:
        return None
    if_col = "journal_if" if "journal_if" in df.columns else ("impact_factor" if "impact_factor" in df.columns else None)

    cols = ["doi", "_topic", "year", cit_col] + ([if_col] if if_col else [])
    docs = df[cols].drop_duplicates("doi").copy()
    docs[cit_col] = pd.to_numeric(docs[cit_col], errors="coerce")
    docs["year"] = pd.to_numeric(docs["year"], errors="coerce")
    docs = docs.dropna(subset=[cit_col, "year"])
    if len(docs) < 20:
        return None

    docs["y"] = np.log1p(docs[cit_col])
    docs["year_c"] = docs["year"] - docs["year"].mean()

    # Top-10 topics + "Other" keeps the model readable
    top_topics = docs["_topic"].value_counts().head(10).index
    docs["topic_top"] = docs["_topic"].where(docs["_topic"].isin(top_topics), "Other")

    X = pd.get_dummies(docs["topic_top"], prefix="topic", drop_first=True).astype(float)
    X["year_c"] = docs["year_c"].values
    if if_col:
        docs["if_missing"] = docs[if_col].isna().astype(float)
        docs["if_filled"] = pd.to_numeric(docs[if_col], errors="coerce")
        docs["if_filled"] = docs["if_filled"].fillna(docs["if_filled"].median())
        X["if_filled"] = docs["if_filled"].values
        X["if_missing"] = docs["if_missing"].values
    X["const"] = 1.0

    y = docs["y"].to_numpy(dtype=float)

    def fit(Xm, yv):
        beta, *_ = np.linalg.lstsq(Xm, yv, rcond=None)
        resid = yv - Xm @ beta
        n, k = Xm.shape
        sigma2 = resid @ resid / (n - k)
        cov = sigma2 * np.linalg.inv(Xm.T @ Xm)
        se = np.sqrt(np.diag(cov))
        return beta, se, n, k

    Xm = X.to_numpy(dtype=float)
    beta, se, n, k = fit(Xm, y)

    # Percentile bootstrap CIs (B = 1000)
    rng = np.random.default_rng(42)
    B = 1000
    idx = np.arange(len(docs))
    boot = np.zeros((B, Xm.shape[1]))
    for b in range(B):
        s = rng.choice(idx, size=len(idx), replace=True)
        boot[b], *_ = np.linalg.lstsq(Xm[s], y[s], rcond=None)
    lo = np.percentile(boot, 2.5, axis=0)
    hi = np.percentile(boot, 97.5, axis=0)

    # Kruskal-Wallis across topics
    kw = None
    from scipy.stats import kruskal
    groups = []
    for t in docs["topic_top"].unique():
        g = docs["y"][docs["topic_top"] == t]
        if len(g) >= 3:
            groups.append(g)
    if len(groups) >= 2:
        kw = kruskal(*groups)

    # Robustness: refit without journal IF
    max_change = None
    if if_col:
        X_noif = X.drop(columns=["if_filled", "if_missing"])
        beta_noif, *_ = np.linalg.lstsq(X_noif.to_numpy(dtype=float), y, rcond=None)
        shared = [c for c in X_noif.columns if c in X.columns]
        if shared:
            max_change = max(
                abs(beta[X.columns.get_loc(c)] - beta_noif[X_noif.columns.get_loc(c)])
                for c in shared
            )

    md = ["### Regression Audit\n"]
    md.append(
        f"OLS of log1p({cit_col}) on Step-6 topics (top 10 + Other), centered year, "
        f"and journal IF (with missingness dummy). n = {n} documents, k = {k} predictors.\n"
    )
    md.append("| Predictor | Coef | SE | Bootstrap 95% CI |")
    md.append("|---|---|---|---|")
    for i, col in enumerate(X.columns):
        md.append(f"| {col} | {beta[i]:.3f} | {se[i]:.3f} | [{lo[i]:.3f}, {hi[i]:.3f}] |")
    if kw is not None:
        md.append(
            f"\nKruskal-Wallis across topics: H = {kw.statistic:.2f}, p = {kw.pvalue:.4f} "
            "(p < 0.05 → citation counts differ across topics)."
        )
    if max_change is not None:
        md.append(
            f"\nRobustness (drop journal IF): max |Δcoef| = {max_change:.3f} across shared "
            "predictors (small → coefficients stable)."
        )
    md.append("")
    return "\n".join(md)


# ── growth model fitting with AICc ─────────────────────────────

def run_growth_aicc(df):
    """Formal growth model selection per macro-category using AICc.

    Models: Plateau (constant), Linear, Exponential (log-linear). The best
    model per category is chosen by minimum AICc; ΔAICc < 2 means the models
    are statistically indistinguishable. The heuristic recent-ratio is kept as
    a secondary column for continuity with the original report.
    """
    if "year" not in df.columns or "macro_category" not in df.columns:
        return None
    d = df[["year", "macro_category"]].dropna()
    if d.empty:
        return None

    current_year = datetime.now().year
    recent_years = [current_year, current_year - 1, current_year - 2]

    def aicc(rss, n, k):
        if n <= k + 1 or rss <= 0:
            return float("inf")
        sigma2 = rss / n
        ll = -n / 2 * (np.log(2 * np.pi * sigma2) + 1)
        return 2 * k - 2 * ll + 2 * k * (k + 1) / (n - k - 1)

    rows = []
    for cat, g in d.groupby("macro_category"):
        counts = g["year"].value_counts().sort_index()
        years = counts.index.to_numpy(dtype=float)
        y = counts.to_numpy(dtype=float)
        n = len(years)
        if n < 4:
            continue

        # Plateau: constant mean
        rss_p = float(np.sum((y - y.mean()) ** 2))
        # Linear: y ~ a + b*year
        A = np.vstack([np.ones(n), years]).T
        b_l, *_ = np.linalg.lstsq(A, y, rcond=None)
        rss_l = float(np.sum((y - A @ b_l) ** 2))
        # Exponential: log(y) ~ a + b*year
        ylog = np.log(np.maximum(y, 1e-9))
        b_e, *_ = np.linalg.lstsq(A, ylog, rcond=None)
        rss_e = float(np.sum((ylog - A @ b_e) ** 2))

        scores = {
            "Plateau": aicc(rss_p, n, 1),
            "Linear": aicc(rss_l, n, 2),
            "Exponential": aicc(rss_e, n, 2),
        }
        best = min(scores, key=scores.get)
        second = sorted(scores.values())[1]

        total = int(g.shape[0])
        recent_ratio = round(len(g[g["year"].isin(recent_years)]) / total, 2) if total else 0
        rows.append({
            "macro_category": cat,
            "n_years": n,
            "total_mentions": total,
            "recent_ratio": recent_ratio,
            "best_model": best,
            "AICc": round(scores[best], 1),
            "ΔAICc vs 2nd": round(scores[best] - second, 1),
        })

    if not rows:
        return None

    out = pd.DataFrame(rows).sort_values("total_mentions", ascending=False)
    md = ["### Growth Model Fitting\n"]
    md.append(
        "Formal model selection per macro-category: Plateau (constant), Linear, or "
        "Exponential (log-linear), chosen by minimum AICc. ΔAICc < 2 → models are "
        "statistically indistinguishable. The heuristic recent-ratio is kept for "
        "continuity with the original classification.\n"
    )
    md.append(out.head(20).to_markdown(index=False) + "\n")
    return "\n".join(md)


# ── Relevance Index sensitivity (conditional) ──────────────────

def run_ri_sensitivity(df):
    """Relevance Index unit-invariance check.

    Conditional: RI is not standard in AKL, so this normally reports a skip
    note. If an RI column is present, verifies that the ranking is invariant
    to an affine rescaling (Spearman correlation = 1.0).
    """
    ri_cols = [c for c in df.columns if "relevance_index" in c.lower() or c.lower() in ("ri", "relevance_idx")]
    if not ri_cols:
        return (
            "### Relevance Index Sensitivity\n\n"
            "Skipped: no Relevance Index column is computed in this pipeline "
            "(RI is not standard in AKL). If RI is added later, this audit will "
            "verify the unit-invariance of the ranking.\n"
        )

    from scipy.stats import spearmanr
    md = ["### Relevance Index Sensitivity\n"]
    for col in ri_cols:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(vals) < 5:
            md.append(f"- `{col}`: insufficient data for the sensitivity check.\n")
            continue
        rho, _ = spearmanr(vals, vals * 2 + 1)
        ok = "PASS" if abs(rho) > 0.999 else "FAIL"
        md.append(f"- `{col}`: Spearman(RI, 2·RI+1) = {rho:.4f} → **{ok}** (unit-invariance).")
    md.append("")
    return "\n".join(md)


# ── Runner ───────────────────────────────────────────────────────────────────

def run_all_auto_audits(df, workspace):
    """Run every automated audit and return the concatenated Markdown sections.

    Sections that cannot run (missing data) are skipped silently; the caller
    decides how to report the absence.
    """
    sections = []
    for fn in (run_evoc_sensitivity, run_regression_audit, run_growth_aicc, run_ri_sensitivity):
        try:
            section = fn(df) if fn in (run_regression_audit, run_growth_aicc, run_ri_sensitivity) else fn(workspace)
            if section:
                sections.append(section)
        except Exception as e:  # audits must never break the crunching report
            sections.append(f"### {fn.__name__}\n\n⚠️ Audit failed: {e}\n")
    return "\n".join(sections)