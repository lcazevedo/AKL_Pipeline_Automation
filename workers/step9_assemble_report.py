"""
step9_assemble_report.py
------------------------
Builds data/04_reports/AKL_Final_Report.md from the specialist-validated synthesis
(step8_insights_review.md), embeds the standard and ad-hoc figures with captions in the
matching sections, and appends the validation and audit summary.

Run: ./run_worker.sh workers/step9_assemble_report.py   (then: step9_worker.py compile_html)
"""
import re
import os
from pathlib import Path

ROOT = Path(os.environ.get("AKL_WORKSPACE") or Path.cwd())   # the workspace (run folder); defaults to the current directory
REP = ROOT / "data" / "04_reports"

# section number -> list of (image path relative to 04_reports, title, description). Captions are LaTeX style:
# the figures carry no titles, the title and the panel descriptions live here.
FIG = {
    1: [("figures/ad_hoc/S1_publications_per_year.png", "Publications per year in the filtered corpus",
         "Number of papers per year, 2000-2026. The 2026 bar (hatched) is a partial year and is excluded from every trend analysis."),
        ("figures/growth_curves.png", "Yearly counts of the nine most frequent extracted categories",
         "Mentions per year, 2000-2025. All categories grow steeply because the whole field does; share-based views follow in the next sections. 2026 is excluded (partial year).")],
    2: [("figures/ad_hoc/A0_topic_map.png", "Map of the 46 topics",
         "Two-dimensional projection (UMAP) of the document embeddings, coloured by topic (Step 6, layer 3). Labels are placed at the topic medians and moved apart where they would overlap."),
        ("figures/ad_hoc/A3_lifecycle_map.png", "Topic lifecycle map",
         "Share of the yearly corpus in 2023-25 (x) against its change since 2014-16 (y, percentage points). Bubble area is proportional to the number of papers; colours give the share trend."),
        ("figures/ad_hoc/A2_topic_share_over_time.png", "Share of the yearly corpus by topic",
         "Percentage of the yearly corpus for each of the 46 topics, 2005-2025, grouped by share trend (rising, stable, peaked, declining). 2026 is excluded (partial year); the vertical scale differs between panels.")],
    3: [("figures/ad_hoc/B2_pioneers_and_geography.png", "Pioneers and early adopters",
         "(a) Share of papers by first-author country, up to 2012 versus 2013-2025; green lines gain share, orange lines lose share. About 21% of the papers up to 2012 have no country (older Web of Science records), so early shares are partial. (b) First year in which each topic reaches five papers; colours as in the lifecycle map."),
        ("figures/impact_quadrants_likert.png", "Research impact categories by country",
         "Share of each country's papers in four impact categories defined by the medians of journal impact factor and citations per year (Star: both above; Hidden Gem: low impact factor, high citations; Coattail: high impact factor, low citations; Long Tail: both below)."),
        ("figures/clustermap_countries.png", "Countries by research category",
         "Hierarchical clustermap of the 15 most productive countries by the 15 most frequent extracted categories (rows normalised to proportions).")],
    4: [("figures/ad_hoc/A5_method_mix_over_time.png", "Method mix over time",
         "Share of papers by ML method category, 2005-2025 (stacked areas sum to 100%)."),
        ("figures/ad_hoc/A4_application_method_heatmap.png", "Application task by ML method",
         "Numbers are paper counts; colour is log2(observed/expected) under independence. Boxed cells are below half of the expected count (expected at least 30). Most boxed cells are structural, for example graph networks need atomic structures that are rare in process-level problems, so they are not necessarily opportunities.")],
    5: [("figures/ad_hoc/B1_data_source_validation_by_topic.png", "Data source, validation and benchmarks by topic",
         "(a) Where the training data comes from. (b) How results are validated. (c) Share of papers naming a benchmark. Topics are sorted by the share of simulation data. All labels were extracted from abstracts by an LLM and are not audited."),
        ("figures/ad_hoc/B4_citation_effects.png", "Citation effects by topic",
         "(a) Topic coefficients of the regression audit on log(1 + citations), controlling for year and journal impact factor, with bootstrap 95% intervals (10 largest topics). (b) Median citations per year by topic; recent papers are under-exposed.")],
    6: [("figures/ad_hoc/A7_bottleneck_heatmaps.png", "Bottlenecks by topic",
         "Share of each challenge category among the sentences of the ten topics with most challenge sentences (all years, rows normalised). (a) ML and data challenges. (b) Domain challenges. Boxes mark the leading category of each topic."),
        ("figures/clustermap_topics.png", "Topics by research category",
         "Hierarchical clustermap of the 20 largest topics by the 15 most frequent extracted categories (rows normalised to proportions).")],
    8: [("figures/ad_hoc/A6_future_directions.png", "Future directions",
         "(a) Themes named in future-direction sentences, as a share of the papers with at least one direction sentence (2-year moving average, 2026 excluded). (b) Change in share of direction categories between 2015-20 and 2023-25. The sentences were filtered by an LLM verifier; read trends, not absolute levels.")],
    10: [("figures/ad_hoc/B3_growth_rate_forest.png", "Growth rate by topic",
          "Yearly growth rate of paper counts per topic (2015-2025 log-linear fit) with the 95% interval of the slope; the dashed line is the growth of the whole field. This describes the past trend and is not a forecast.")],
    11: [("figures/ad_hoc/A1_ripe_or_hype_map.png", "Ripe-or-hype map",
          "Growth of each topic relative to the whole field (x, percentage points) against the share of its papers with experimental validation (y). Bubble area is proportional to the number of papers; colours give the share trend; the dashed line is the median across topics. Experimental validation is read from abstracts by an LLM and is not audited.")],
    12: [("figures/ad_hoc/B5_audit_and_filter_transparency.png", "Audit results and sentence filters",
          "(a) Agreement with the specialist in the three audits (100 items each). (b) AUC, precision and recall at threshold 0.5 of the two sentence filters on the 200 audited sentences. The verifier prompt was written after reading errors in this same sample, and the sample only contains sentences the extractor had marked as positive, so these figures are optimistic.")],
}
APPENDIX = [("figures/ad_hoc/S2_topic_terms.png", "Top terms of the largest topics",
             "The ten most characteristic terms (c-TF-IDF rank; larger type means more characteristic) of the 12 largest topics."),
            ("figures/ad_hoc/S3_keyword_cooccurrence_network.png", "Author-keyword co-occurrence network",
             "Co-occurrence of the 45 most frequent author keywords (generic ML terms removed). Only papers with author keywords are included (about 13% of the corpus). Colours mark communities found by modularity.")]

text = (REP / "step8_insights_review.md").read_text(encoding="utf-8")
parts = re.split(r"(?m)^(?=## )", text)
n = 0


def block(figs):
    global n
    out = []
    for path, title, desc in figs:
        if not (REP / path).exists():
            continue
        n += 1
        out.append(f"![Figure {n}]({path})\n\n**Figure {n}. {title}.** {desc}\n")
    return "\n".join(out)


res = []
for part in parts:
    m = re.match(r"## (\d+)\.", part)
    if m and int(m.group(1)) in FIG:
        figs = block(FIG[int(m.group(1))])
        # insert before the closing horizontal rule of the section
        idx = part.rstrip().rfind("\n---")
        part = (part[:idx].rstrip() + "\n\n" + figs + "\n" + part[idx:]) if idx != -1 else part.rstrip() + "\n\n" + figs + "\n"
    res.append(part)
body = "".join(res).rstrip() + "\n\n---\n\n## Appendix A. Topic terms and keyword network\n\n" + block(APPENDIX)
audit = REP / "step9_audit_summary.md"
if audit.exists():
    body += "\n---\n\n## Appendix B. Validation and audit summary\n\n" + audit.read_text(encoding="utf-8")
(REP / "AKL_Final_Report.md").write_text(body, encoding="utf-8")
print(f"AKL_Final_Report.md written with {n} figures")
