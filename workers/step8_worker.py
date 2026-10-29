"""
step8_worker.py
---------------
AKL Pipeline — Step 8: Data Crunching for Expert Synthesis

Responsibilities:
- Load the semantic extractions (Step 7) and the original corpus metadata.
- Calculate Thematic Lifecycles (Exponential, Linear, Plateau).
- Calculate Geographic Focus (Pioneers vs Traditionalists).
- Perform Row-Normalized Cross-tabulations for Bottleneck Regimes (Topics vs Vectors).
- Calculate Impact Quadrants (Star, Hidden Gem, Coattail, Long Tail) if citation data exists.
- Save the statistical report to data/04_reports/step8_data_crunching.md.
"""

import sys
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime

try:
    from workers.reference_date import reference_year as _reference_year
except ImportError:  # executed from inside workers/
    from reference_date import reference_year as _reference_year
import pandas as pd
import numpy as np

# Allow importing sibling workers (audit_framework, step8_audits)
sys.path.insert(0, str(Path(__file__).resolve().parent))

warnings.filterwarnings("ignore")

def _setup_logger(workspace_path):
    logger = logging.getLogger("step8_worker")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [STEP8] %(message)s", datefmt="%H:%M:%S")
        fh = logging.FileHandler(Path(workspace_path) / "execution.log", encoding="utf-8")
        fh.setFormatter(fmt)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

class Step8Worker:
    def __init__(self, state_path="pipeline_state.json"):
        self.state_path = Path(state_path)
        with open(self.state_path, "r", encoding="utf-8") as f:
            self.state = json.load(f)
            
        self.workspace = Path(self.state.get("project", {}).get("workspace_path", "."))
        self.logger = _setup_logger(self.workspace)

    def _agent_llm(self):
        """Name of the model acting as the agent (models.llm_agent in pipeline_config.yaml)."""
        try:
            import yaml
            with open(self.workspace / "pipeline_config.yaml", "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            return (cfg.get("models", {}) or {}).get("llm_agent") or "not recorded"
        except Exception:
            return "not recorded"

    def _update_state(self, status):
        now = datetime.now().isoformat(timespec="seconds")
        step = self.state.setdefault("steps", {}).setdefault("8", {})
        step["status"] = status
        step.setdefault("params", {})["synthesis_llm"] = self._agent_llm()
        if status == "running":
            step["started_at"] = now
            self.state["current_step"] = 8
        elif status in ["awaiting_agent_synthesis", "completed"]:
            step["completed_at"] = now
            
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)

    def load_data(self):
        ext_path = self.workspace / "data" / "02_intermediate" / "step7_extractions.pkl"
        if not ext_path.exists():
            raise FileNotFoundError("step7_extractions.pkl not found.")
        
        df_ext = pd.read_pickle(ext_path)
        
        # Merge with Corpus for Metadata (year, country, citations, IF)
        corpus_path = self.workspace / "data" / "02_intermediate" / "step5_corpus_filtered_relevant.pkl"
        if not corpus_path.exists():
            corpus_path = self.workspace / "data" / "02_intermediate" / "step2_corpus_clean.pkl"
            
        if corpus_path.exists():
            df_corpus = pd.read_pickle(corpus_path)
            cols_to_use = [c for c in df_corpus.columns if c not in df_ext.columns or c == 'doi']
            df_master = pd.merge(df_ext, df_corpus[cols_to_use], on="doi", how="left")
            
            # Merge with Step 6 Topic Assignment if available
            topic_files = list((self.workspace / "data" / "05_exports").glob("step6_topics_final_layer_*_selected.xlsx"))
            if topic_files:
                try:
                    # Each sheet of the Step 6 workbook holds the documents of one topic;
                    # the topic label is the sheet name (e.g. "15_ChatGPT Applications").
                    sheet_frames = []
                    for sheet_name, sheet_df in pd.read_excel(topic_files[0], sheet_name=None).items():
                        sheet_df = sheet_df.copy()
                        sheet_df['_topic'] = sheet_name
                        sheet_frames.append(sheet_df)
                    df_topics = pd.concat(sheet_frames, ignore_index=True)
                    if 'doi' in df_topics.columns and '_topic' in df_topics.columns:
                        topic_map = df_topics[['doi', '_topic']].drop_duplicates()
                        df_master = pd.merge(df_master, topic_map, on='doi', how='left')
                except Exception as e:
                    self.logger.warning(f"Could not load topic assignments: {e}")
                    
            return df_master
        return df_ext

    def _geo_column(self, df):
        """Pick the geographic column present in the DataFrame."""
        for col in ('first_author_country', 'first_country', 'last_country', 'country'):
            if col in df.columns:
                return col
        return None

    def classify_growth_regimes(self, df):
        if 'year' not in df.columns:
            return None
        
        df = df[df['year'].notnull()]
        current_year = _reference_year(self.workspace)
        recent_years = [current_year, current_year-1, current_year-2]
        
        regimes = []
        for cat, group in df.groupby('macro_category'):
            total = len(group)
            recent_count = len(group[group['year'].isin(recent_years)])
            recent_ratio = recent_count / total if total > 0 else 0
            
            if recent_ratio >= 0.5:
                regime = "Exponential/Emerging"
            elif 0.25 <= recent_ratio < 0.5:
                regime = "Linear/Mainstream"
            else:
                regime = "Plateau/Declining"
                
            regimes.append({'macro_category': cat, 'total_mentions': total, 'recent_ratio': round(recent_ratio, 2), 'growth_regime': regime})
            
        return pd.DataFrame(regimes).sort_values('total_mentions', ascending=False)

    def classify_impact_quadrants(self, df):
        has_citations = any(c in df.columns for c in ('citations', 'cited_by_count', 'citation_count'))
        has_if = any(c in df.columns for c in ('impact_factor', 'journal_if'))
        if not (has_citations and has_if and 'year' in df.columns):
            return None
            
        cit_col = 'citations' if 'citations' in df.columns else 'cited_by_count' if 'cited_by_count' in df.columns else 'citation_count'
        if_col = 'impact_factor' if 'impact_factor' in df.columns else 'journal_if'
        
        # Unique documents for impact calculation
        df_docs = df[['doi', 'year', cit_col, if_col]].drop_duplicates().dropna()
        if df_docs.empty: return None
        
        df_docs[cit_col] = pd.to_numeric(df_docs[cit_col], errors='coerce')
        df_docs[if_col] = pd.to_numeric(df_docs[if_col], errors='coerce')
        df_docs = df_docs.dropna()
        
        current_year = _reference_year(self.workspace)
        df_docs['age'] = current_year - df_docs['year'] + 1
        df_docs['cpy'] = df_docs[cit_col] / df_docs['age']
        
        t_cit = df_docs['cpy'].median()
        t_if = df_docs[if_col].median()
        
        def assign_quadrant(row):
            if row[if_col] >= t_if and row['cpy'] >= t_cit: return "Star (High IF, High CPY)"
            if row[if_col] < t_if and row['cpy'] >= t_cit: return "Hidden Gem (Low IF, High CPY)"
            if row[if_col] >= t_if and row['cpy'] < t_cit: return "Coattail (High IF, Low CPY)"
            return "Long Tail (Low IF, Low CPY)"
            
        df_docs['impact_quadrant'] = df_docs.apply(assign_quadrant, axis=1)
        
        # Merge back to calculate percentages per country/institution
        geo_col = self._geo_column(df)
        if not geo_col: return df_docs['impact_quadrant'].value_counts().to_frame("Count").reset_index()
        
        df_merged = pd.merge(df_docs, df[['doi', geo_col]].drop_duplicates(), on='doi', how='left')
        top_countries = df_merged[geo_col].value_counts().head(15).index
        df_top = df_merged[df_merged[geo_col].isin(top_countries)]
        
        ct = pd.crosstab(df_top[geo_col], df_top['impact_quadrant'], normalize='index') * 100
        ct['High-Utility Output (Star + Hidden Gem) %'] = ct.get("Star (High IF, High CPY)", 0) + ct.get("Hidden Gem (Low IF, High CPY)", 0)
        return ct.round(1).sort_values('High-Utility Output (Star + Hidden Gem) %', ascending=False).reset_index()

    def generate_bottleneck_matrix(self, df):
        if '_topic' not in df.columns:
            return None
            
        md_tables = []
        for vector in df['vector'].unique():
            df_v = df[df['vector'] == vector]
            top_cats = df_v['macro_category'].value_counts().head(8).index
            df_v = df_v[df_v['macro_category'].isin(top_cats)]
            
            top_topics = df_v['_topic'].value_counts().head(10).index
            df_v = df_v[df_v['_topic'].isin(top_topics)]
            
            if df_v.empty: continue
                
            ct = pd.crosstab(df_v['_topic'], df_v['macro_category'], normalize='index') * 100
            ct = ct.round(1).reset_index()
            ct.rename(columns={'_topic': 'Topic ID'}, inplace=True)
            md_tables.append(f"#### Bottleneck Regime: {vector.upper()} vs Topics (Row-Normalized %)\n" + ct.to_markdown(index=False) + "\n")
            
        return "\n".join(md_tables) if md_tables else None

    def crunch_data(self, df):
        md = ["# Step 8: Statistical Data Crunching Report\n"]
        md.append("> This report contains computed scientometric matrices. The LLM agent will use this to generate the final qualitative synthesis.\n")

        # 1. Thematic Lifecycle
        md.append("## 1. Thematic Lifecycle (Growth Regimes)")
        df_growth = self.classify_growth_regimes(df)
        if df_growth is not None:
            md.append("Classifies research categories based on their volume concentration in the last 3 years.\n")
            md.append(df_growth.head(20).to_markdown(index=False) + "\n")
            
            # 2. Strategic Geopolitics (Pioneers vs Traditionalists)
            geo_col = self._geo_column(df)
            if geo_col:
                md.append("## 2. Strategic Geopolitics (Pioneers vs Traditionalists)")
                md.append(f"Row-normalized proportion of a country's effort dedicated to Exponential/Emerging vs Plateau/Mainstream themes.\n")
                
                df_geo = pd.merge(df, df_growth[['macro_category', 'growth_regime']], on='macro_category', how='left')
                top_countries = df_geo[geo_col].value_counts().head(15).index
                df_geo_top = df_geo[df_geo[geo_col].isin(top_countries)]
                
                ct_geo = pd.crosstab(df_geo_top[geo_col], df_geo_top['growth_regime'], normalize='index') * 100
                md.append(ct_geo.round(1).reset_index().to_markdown(index=False) + "\n")

        # 3. Bottleneck Regimes
        md.append("## 3. Bottleneck Regimes & Semantic Clustering")
        bottlenecks = self.generate_bottleneck_matrix(df)
        if bottlenecks:
            md.append(bottlenecks)
        else:
            md.append("Topic assignment not found or insufficient data for cross-tabulation.\n")

        # 4. Impact Quadrants
        md.append("## 4. Decoupling Volume and Efficiency (Impact Quadrants)")
        df_impact = self.classify_impact_quadrants(df)
        if df_impact is not None:
            md.append("Classifies documents using median Journal IF and Citations Per Year (CPY). Shows which countries lead in high-utility efficiency vs pure volume.\n")
            md.append(df_impact.to_markdown(index=False) + "\n")
        else:
            md.append("*Impact Factor or Citation data not available in the corpus metadata to calculate quadrants.*\n")

        return "\n".join(md)

    def run(self):
        self.logger.info("Starting Step 8 Data Crunching...")
        self._update_state("running")
        
        try:
            df = self.load_data()
            md_report = self.crunch_data(df)

            # Automated audits (reviewer validations #5, #9, #10, #11)
            from step8_audits import run_all_auto_audits
            auto_audits = run_all_auto_audits(df, self.workspace)
            if auto_audits:
                md_report += "\n---\n\n## 5. Automated Audits\n\n" + auto_audits

            # Audit sample export (Phase 1.5 — reviewer validations #2, #6, #7, #8)
            from audit_framework import AuditFramework
            audit_results = AuditFramework().export_all()
            exported = {k: v for k, v in audit_results.items() if "file" in v}
            if exported:
                self.logger.info(f"Audit samples exported: {list(exported.keys())}")
            
            out_dir = self.workspace / "data" / "04_reports"
            out_dir.mkdir(parents=True, exist_ok=True)
            report_path = out_dir / "step8_data_crunching.md"
            
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(md_report)
                
            self.logger.info(f"Data crunching completed. Report saved to {report_path}")
            self.logger.info(f"Synthesis (agent) model recorded: {self._agent_llm()}")
            self._update_state("awaiting_agent_synthesis")
            
            return {
                "status": "awaiting_agent_synthesis",
                "file": str(report_path),
                "audit_samples_exported": list(exported.keys()),
            }
            
        except Exception as e:
            self.logger.error(f"Step 8 failed: {e}", exc_info=True)
            self._update_state("failed")
            raise

    def compute_audits(self):
        """Compute audit metrics from the reviewed files (Step 8 Phase 3)."""
        from audit_framework import AuditFramework
        self.logger.info("Computing audit metrics from reviewed files...")
        results = AuditFramework().compute_all()
        return results

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    worker = Step8Worker()
    if mode == "run":
        print(json.dumps(worker.run()))
    elif mode == "compute_audits":
        print(json.dumps(worker.compute_audits(), indent=2))
    else:
        print(f"Unknown mode: {mode}. Use 'run' or 'compute_audits'.")
        sys.exit(1)