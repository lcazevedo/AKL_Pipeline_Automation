"""
step9_worker.py
---------------
AKL Pipeline — Step 9: Advanced Visualizations & Final HTML Report

Responsibilities:
- Rescues the 2D Topographic map from Step 6.
- Generates Likert-Scale Impact Quadrant charts (Volume vs Efficiency).
- Generates Hierarchical Clustermaps (with dendrograms) for Countries vs Challenges and Topics vs Challenges.
- Generates Growth Regime multi-panel line charts.
- Compiles the final Markdown into a polished HTML report.
"""

import sys
import json
import logging
import warnings
import shutil
from pathlib import Path
from datetime import datetime

try:
    from workers.reference_date import reference_year as _reference_year
except ImportError:  # executed from inside workers/
    from reference_date import reference_year as _reference_year
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import markdown

warnings.filterwarnings("ignore")

def _setup_logger(workspace_path):
    logger = logging.getLogger("step9_worker")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [STEP9] %(message)s", datefmt="%H:%M:%S")
        fh = logging.FileHandler(Path(workspace_path) / "execution.log", encoding="utf-8")
        fh.setFormatter(fmt)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

class Step9Worker:
    def __init__(self, state_path="pipeline_state.json"):
        self.state_path = Path(state_path)
        with open(self.state_path, "r", encoding="utf-8") as f:
            self.state = json.load(f)
            
        self.workspace = Path(self.state.get("project", {}).get("workspace_path", "."))
        self.logger = _setup_logger(self.workspace)
        
        self.reports_dir = self.workspace / "data" / "04_reports"
        self.figures_dir = self.reports_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        (self.figures_dir / "ad_hoc").mkdir(exist_ok=True)
        
        # Plotting style
        sns.set_theme(style="whitegrid", context="paper")
        plt.rcParams.update({'figure.dpi': 300, 'savefig.bbox': 'tight'})

    def _update_state(self, status):
        now = datetime.now().isoformat(timespec="seconds")
        step = self.state.setdefault("steps", {}).setdefault("9", {})
        step["status"] = status
        if status == "running":
            step["started_at"] = now
            self.state["current_step"] = 9
        elif status == "completed":
            step["completed_at"] = now
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)

    def _geo_column(self, df):
        for col in ('first_author_country', 'first_country', 'last_country', 'country'):
            if col in df.columns: return col
        return None

    def load_data(self):
        ext_path = self.workspace / "data" / "02_intermediate" / "step7_extractions.pkl"
        if not ext_path.exists():
            return None
        
        df_ext = pd.read_pickle(ext_path)
        
        corpus_path = self.workspace / "data" / "02_intermediate" / "step5_corpus_filtered_relevant.pkl"
        if not corpus_path.exists():
            corpus_path = self.workspace / "data" / "02_intermediate" / "step2_corpus_clean.pkl"
            
        if corpus_path.exists():
            df_corpus = pd.read_pickle(corpus_path)
            cols = [c for c in df_corpus.columns if c not in df_ext.columns or c == 'doi']
            df_master = pd.merge(df_ext, df_corpus[cols], on="doi", how="left")
            
            topic_files = list((self.workspace / "data" / "05_exports").glob("step6_topics_final_layer_*_selected.xlsx"))
            if topic_files:
                try:
                    frames = []
                    for sheet, d in pd.read_excel(topic_files[0], sheet_name=None).items():
                        d['_topic'] = sheet
                        frames.append(d)
                    df_topics = pd.concat(frames, ignore_index=True)
                    if 'doi' in df_topics.columns and '_topic' in df_topics.columns:
                        df_master = pd.merge(df_master, df_topics[['doi', '_topic']].drop_duplicates(), on='doi', how='left')
                except:
                    pass
            return df_master
        return df_ext

    def import_step6_map(self):
        source = self.workspace / "data" / "04_reports" / "topic_map.png"
        dest = self.figures_dir / "topic_map_2d.png"
        if source.exists():
            shutil.copy(source, dest)
            self.logger.info("Step 6 Topographic 2D map imported successfully.")

    def plot_impact_likert(self, df):
        geo_col = self._geo_column(df)
        cit_col = 'citations' if 'citations' in df.columns else 'cited_by_count' if 'cited_by_count' in df.columns else 'citation_count'
        if_col = 'impact_factor' if 'impact_factor' in df.columns else 'journal_if'
        
        if not (geo_col and cit_col in df.columns and if_col in df.columns and 'year' in df.columns):
            self.logger.warning("Missing data (Citations, IF, Year or Country) for Impact Quadrants Likert plot.")
            return

        df_docs = df[['doi', 'year', geo_col, cit_col, if_col]].drop_duplicates().dropna()
        if df_docs.empty: return

        df_docs[cit_col] = pd.to_numeric(df_docs[cit_col], errors='coerce')
        df_docs[if_col] = pd.to_numeric(df_docs[if_col], errors='coerce')
        df_docs = df_docs.dropna()
        
        current_year = _reference_year(getattr(self, 'workspace', '.'))
        df_docs['age'] = current_year - df_docs['year'] + 1
        df_docs['cpy'] = df_docs[cit_col] / df_docs['age']
        
        t_cit = df_docs['cpy'].median()
        t_if = df_docs[if_col].median()
        
        def assign_quadrant(row):
            if row[if_col] >= t_if and row['cpy'] >= t_cit: return "Star"
            if row[if_col] < t_if and row['cpy'] >= t_cit: return "Hidden Gem"
            if row[if_col] >= t_if and row['cpy'] < t_cit: return "Coattail"
            return "Long Tail"
            
        df_docs['Quadrant'] = df_docs.apply(assign_quadrant, axis=1)
        
        top_countries = df_docs[geo_col].value_counts().head(15).index
        df_top = df_docs[df_docs[geo_col].isin(top_countries)]
        
        ct = pd.crosstab(df_top[geo_col], df_top['Quadrant'], normalize='index') * 100
        # Ensure all columns exist
        for col in ["Long Tail", "Coattail", "Hidden Gem", "Star"]:
            if col not in ct.columns: ct[col] = 0.0
            
        ct['Utility'] = ct['Star'] + ct['Hidden Gem']
        ct = ct.sort_values('Utility', ascending=True)
        plot_data = ct[["Long Tail", "Coattail", "Hidden Gem", "Star"]]

        colors = ["#b2182b", "#fd8d3c", "#4393c3", "#053061"]
        ax = plot_data.plot(kind='barh', stacked=True, figsize=(10, 7), color=colors)
        
        plt.title("Distribution of Research Impact Categories by Country")
        plt.xlabel("Percentage of papers in each category (%)")
        plt.ylabel("")
        plt.legend(title="Impact Quadrant", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(self.figures_dir / "impact_quadrants_likert.png", bbox_inches='tight')
        plt.close()

    def plot_clustermaps(self, df):
        # 1. Countries vs Challenges
        geo_col = self._geo_column(df)
        if geo_col and 'macro_category' in df.columns:
            try:
                top_countries = df[geo_col].value_counts().head(15).index
                df_geo = df[df[geo_col].isin(top_countries)]
                
                # Filter to top 15 macro categories
                top_cats = df_geo['macro_category'].value_counts().head(15).index
                df_geo = df_geo[df_geo['macro_category'].isin(top_cats)]
                
                ct_geo = pd.crosstab(df_geo[geo_col], df_geo['macro_category'], normalize='index')
                
                sns.clustermap(ct_geo, method='ward', metric='euclidean', cmap='YlGnBu', figsize=(12, 10),
                               cbar_kws={'label': 'Proportion of Research Effort'})
                plt.savefig(self.figures_dir / "clustermap_countries.png", bbox_inches='tight')
                plt.close()
            except Exception as e:
                self.logger.warning(f"Failed to plot Country Clustermap: {e}")

        # 2. Topics vs Challenges
        if '_topic' in df.columns and 'macro_category' in df.columns:
            try:
                top_topics = df['_topic'].value_counts().head(20).index
                df_topic = df[df['_topic'].isin(top_topics)]
                
                top_cats_t = df_topic['macro_category'].value_counts().head(15).index
                df_topic = df_topic[df_topic['macro_category'].isin(top_cats_t)]
                
                ct_topic = pd.crosstab(df_topic['_topic'], df_topic['macro_category'], normalize='index')
                
                sns.clustermap(ct_topic, method='ward', metric='euclidean', cmap='YlGnBu', figsize=(14, 12),
                               cbar_kws={'label': 'Proportion of Topic Effort'})
                plt.savefig(self.figures_dir / "clustermap_topics.png", bbox_inches='tight')
                plt.close()
            except Exception as e:
                self.logger.warning(f"Failed to plot Topic Clustermap: {e}")

    def plot_growth_curves(self, df):
        if 'year' not in df.columns or 'macro_category' not in df.columns:
            return
            
        df_y = df[df['year'].notnull()]
        
        # Get top 9 categories to plot in a 3x3 grid
        top_cats = df_y['macro_category'].value_counts().head(9).index
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True)
        axes = axes.flatten()
        
        for i, cat in enumerate(top_cats):
            df_cat = df_y[df_y['macro_category'] == cat]
            yearly_counts = df_cat['year'].value_counts().sort_index()
            
            axes[i].plot(yearly_counts.index, yearly_counts.values, marker='o', color='#2b8cbe')
            axes[i].set_title(cat[:40] + '...' if len(cat) > 40 else cat, fontsize=10)
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, linestyle='--', alpha=0.6)
            
        plt.suptitle("Temporal Evolution of Top Research Themes", fontsize=16)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "growth_curves.png", bbox_inches='tight')
        plt.close()

    def plot_standard(self):
        self.logger.info("Generating advanced scientific plots...")
        self._update_state("running")
        df = self.load_data()
        if df is None:
            self.logger.error("No data available for plotting.")
            return {"status": "failed"}

        self.import_step6_map()
        self.plot_impact_likert(df)
        self.plot_clustermaps(df)
        self.plot_growth_curves(df)
                
        return {"status": "awaiting_agent_assembly"}

    def generate_audit_summary(self):
        """Generate a consolidated validation & audit summary for the final report.

        Reads the Step 8 audit report (classification audits) and the Step 6
        verify report (topic-size verification) and condenses them into
        `data/04_reports/step9_audit_summary.md`, which the agent includes in
        the final report (reviewer-requested transparency).
        """
        self.logger.info("Generating validation & audit summary...")
        audit_report = self.reports_dir / "step8_audit_report.md"
        verify_report = self.reports_dir / "step6_verify_report.md"

        md = ["## Validation & Audit Summary\n"]
        md.append(
            "This section summarizes the reviewer-requested validations performed "
            "during the pipeline run.\n"
        )

        # Step 6 topic-size verification
        if verify_report.exists():
            text = verify_report.read_text(encoding="utf-8")
            verified = "VERIFIED" in text and "DISCREPANCIES FOUND" not in text
            md.append("### Topic-Size Verification (Step 6)\n")
            md.append(
                f"- **Status:** {'Passed' if verified else 'Discrepancies found'} — "
                "the topic sizes in the exported Excel match the saved per-document assignments.\n"
            )
            md.append(f"- Full report: `data/04_reports/step6_verify_report.md`\n")
        else:
            md.append("### Topic-Size Verification (Step 6)\n- Not performed.\n")

        # Step 8 classification audits
        md.append("\n### Classification Audits (Step 8)\n")
        if audit_report.exists():
            text = audit_report.read_text(encoding="utf-8")
            # Extract per-audit blocks: "## <name>" ... metrics lines
            import re as _re
            blocks = _re.split(r"\n## ", text)
            rows = []
            for block in blocks[1:]:
                name = block.split("\n", 1)[0].strip()
                acc = _re.search(r"Accuracy: \*\*([\d.]+|—)\*\*", block)
                f1 = _re.search(r"F1 \([\w]+\): ([\d.]+|—)", block)
                reviewed = _re.search(r"Reviewed: (\d+)", block)
                pending = _re.search(r"Pending review: (\d+)", block)
                if acc or f1:
                    rows.append({
                        "Audit": name,
                        "Accuracy": acc.group(1) if acc else "—",
                        "F1": f1.group(1) if f1 else "—",
                        "Reviewed": reviewed.group(1) if reviewed else "0",
                        "Pending": pending.group(1) if pending else "0",
                    })
            if rows:
                md.append("| Audit | Accuracy | F1 | Reviewed | Pending |")
                md.append("|---|---|---|---|---|")
                for r in rows:
                    md.append(f"| {r['Audit']} | {r['Accuracy']} | {r['F1']} | {r['Reviewed']} | {r['Pending']} |")
                md.append("")
                md.append("Full report: `data/04_reports/step8_audit_report.md`\n")
            else:
                md.append("No audit metrics computed (no audits enabled or no labels reviewed).\n")
        else:
            md.append("No audit report found.\n")

        # Step 8 automated audits (reviewer validations #5, #9, #10, #11)
        # Extracted from Section 5 of the data-crunching report so that EVERY
        # audit performed during the run appears in the final report.
        crunch_report = self.reports_dir / "step8_data_crunching.md"
        if crunch_report.exists():
            text = crunch_report.read_text(encoding="utf-8")
            marker = "## 5. Automated Audits"
            idx = text.find(marker)
            if idx != -1:
                section = text[idx:].strip()
                # Demote sub-headings so they nest under the audit summary,
                # then rename the section header itself (kept at ### level).
                section = _re.sub(r"^### ", "#### ", section, flags=_re.MULTILINE)
                section = section.replace(marker, "### Automated Audits (Step 8)")
                md.append("\n" + section + "\n")
                md.append("Full report: `data/04_reports/step8_data_crunching.md`\n")
            else:
                md.append("\n### Automated Audits (Step 8)\n- Section not found in the data-crunching report.\n")
        else:
            md.append("\n### Automated Audits (Step 8)\n- Data-crunching report not found.\n")

        out_path = self.reports_dir / "step9_audit_summary.md"
        out_path.write_text("\n".join(md), encoding="utf-8")
        self.logger.info(f"Audit summary saved to {out_path}")
        return {"file": str(out_path)}

    def compile_html(self):
        self.logger.info("Compiling final HTML report...")
        md_file = self.reports_dir / "AKL_Final_Report.md"
        html_file = self.reports_dir / "AKL_Final_Report.html"
        
        if not md_file.exists():
            self.logger.error("AKL_Final_Report.md not found. Agent must generate it first.")
            return {"status": "failed"}
            
        md_text = md_file.read_text(encoding="utf-8")
        html_body = markdown.markdown(md_text, extensions=['tables', 'fenced_code'])
        
        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AKL Final Report</title>
    <style>
        body {{ font-family: "Times New Roman", Times, serif; line-height: 1.6; color: #333; max-width: 900px; margin: 0 auto; padding: 2rem; }}
        h1, h2, h3, h4 {{ color: #1a365d; margin-top: 1.5em; }}
        h1 {{ border-bottom: 2px solid #1a365d; padding-bottom: 0.5em; }}
        img {{ max-width: 100%; height: auto; display: block; margin: 2rem auto; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        table {{ width: 100%; border-collapse: collapse; margin: 1.5rem 0; font-size: 0.9em; }}
        th, td {{ border: 1px solid #e2e8f0; padding: 12px; text-align: left; }}
        th {{ background-color: #f7fafc; font-weight: bold; color: #2d3748; }}
        blockquote {{ border-left: 4px solid #4299e1; padding-left: 1rem; color: #4a5568; background-color: #ebf8ff; margin: 1.5rem 0; padding: 1rem; font-style: italic; }}
        @media print {{
            body {{ padding: 0; max-width: 100%; }}
            .page-break {{ page-break-before: always; }}
            img {{ page-break-inside: avoid; }}
        }}
    </style>
</head>
<body>
    {html_body}
</body>
</html>"""

        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_template)
            
        self._update_state("completed")
        self.logger.info(f"HTML Report successfully compiled at {html_file}")
        return {"status": "completed"}

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "plot_standard"
    worker = Step9Worker()
    if mode == "plot_standard":
        print(json.dumps(worker.plot_standard()))
    elif mode == "audit_summary":
        print(json.dumps(worker.generate_audit_summary()))
    elif mode == "compile_html":
        print(json.dumps(worker.compile_html()))