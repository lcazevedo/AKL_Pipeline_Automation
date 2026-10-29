# Automated Knowledge Landscapes (AKL) — Pipeline Automation

This repository contains the complete multi-agent automation code (Python workers, OpenWork/OpenCode orchestration skills, and configuration templates) for the scientific article **"Automated Knowledge Landscapes: A Human-in-the-Loop Semantic Bibliometric Roadmap for Strategic Discovery"**.

The **Automated Knowledge Landscapes (AKL)** framework is a nine-step, *human-in-the-loop* semantic bibliometric workflow designed to transform large bibliographic corpora into navigable, auditable Knowledge Landscapes. It combines dense transformer-based sentence embeddings, full-dimensional hierarchical clustering via Embedding Vector Oriented Clustering (**EVōC**) and **BERTopic**, few-shot supervised relevance filtering (**SetFit**), sentence-level semantic intent extraction via locally deployed quantized Large Language Models (**vLLM**), and built-in validation audits at strategic expert breakpoints.

> **Looking for the Proof-of-Concept (PoC) dataset, audit samples, and reproducibility verification?**  
> See the companion repository **`AKL_PoC`** (and its `REPRODUCIBILITY.md`), which packages this automation alongside the text-free corpus identifiers, topic partitions, extracted semantic labels, and audit artifacts from the Generative AI in computing education proof-of-concept study.

---

## 📂 Repository Content

```
.opencode/skills/   12 OpenWork/OpenCode agent skills orchestrating Steps 1–9, status, and config sync
workers/            18 Python worker scripts implementing data ingestion, modeling, extraction, and audits
data/               Workspace directory skeleton for raw inputs, intermediate states, reviews, and reports
AGENTS.md           System prompt and operational instructions for the orchestration agent
project_config.md   Research design template (domain context, queries, relevance criteria, vectors)
pipeline_config.yaml Technical hyperparameter configuration and canonical schema mappings
pipeline_state.json Execution state tracker (single source of truth for step preconditions and provenance)
```

### Root Configuration & Execution Files
* `AGENTS.md` — Orchestrator agent instructions defining session initialization, step precondition checks, human-in-the-loop breakpoint rules (Steps 4 and 8), and language consistency policies.
* `project_config.md` (and template `base_project_config.md`) — Human-readable research design document where the domain expert specifies research questions, data sources, inclusion/exclusion criteria for Step 4, active semantic extraction vectors for Step 7, and natural-language configuration overrides.
* `pipeline_config.yaml` (and template `base_pipeline_config.yaml`) — Technical configuration controlling the global reproducibility seed (`reproducibility.seed: 42`), analysis reference date, embedding/LLM/SetFit model identifiers, EVōC clustering parameters, vLLM server endpoints, Step 8 audit definitions, and canonical schema mappings.
* `pipeline_state.json` (and template `base_pipeline_state.json`) — JSON state tracker recording execution timestamps, input/output file paths, document counts, validation status, and exact software/hyperparameter provenance (`params_used.reproducibility`).
* `run_worker.sh` & `run_worker_setfit.sh` — Shell wrappers that activate the isolated Conda environments (`akl_automation` and `akl_setfit`) before executing Python workers.
* `run_step7_detached.sh` — Detached background launcher (`start_new_session=True`) for long-running Step 7 LLM extractions, preventing shell tool timeouts on large corpora and writing progress to `data/07_logs/step7_run.log`.
* `repro_check.py` — Read-only reproducibility utility (`snapshot`, `compare`, `funnel`) that hashes artifacts, verifies embedding matrix equivalence, and computes Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI) across repeated executions.
* `akl_automation_requirements.txt` & `akl_setfit_requirements.txt` — Pinned Python dependency specifications for the main pipeline and SetFit environments.
* `akl_manual.md` & `generic_csv_import_guide.md` — Detailed user manual and guide for mapping custom CSV exports (e.g., Scopus, Dimensions) to the canonical AKL schema.

### `workers/` — Pipeline Execution & Audit Modules
* **Step 1 (Acquisition & Enrichment):**
  * `step1_worker.py` — Ingests bibliographic records from Web of Science (`.txt`/`.csv`), OpenAlex API, PubMed E-utilities API, arXiv Atom API, or generic CSV files (`wide` or `long` format) and normalizes them into the 30-column canonical AKL schema.
  * `step1_merge_worker.py` — Concatenates and deduplicates multi-source corpora by DOI and normalized title according to a configurable source priority.
  * `step1_language_if_worker.py` & `step1_if_fill_worker.py` — Validates document language via `langdetect` and enriches missing journal Impact Factors (`journal_if`) and normalized impact scores (`jnif`).
* **Step 2 (Sanitization):**
  * `step2_worker.py` — Enforces mandatory metadata completeness, filters by publication year window and language, removes residual duplicates, and generates `data/04_reports/step2_quality_report.md`.
* **Step 3 (Preliminary Topic Mapping):**
  * `step3_worker.py` — Computes dense document embeddings (`sentence-transformers`) and runs full-dimensional hierarchical clustering via `BERTopic` + `EVōC` (`random_state=42`), reassigning residual noise (`-1`) to the nearest topic centroid by cosine similarity.
  * `step3_apply_labels.py` — Applies human/agent-generated topic descriptors to the preliminary hierarchy and exports multi-tab Excel workbooks (`data/05_exports/step3_topics_preliminary_layer_*.xlsx`).
* **Step 4 (Human-in-the-Loop Curation — Breakpoint 1):**
  * `step4_worker.py` — Stratifies a representative sample across preliminary topics, coordinates agent pre-labeling against `project_config.md` criteria, and prepares `data/03_review/step4_relevance_review.xlsx` for expert validation.
* **Step 5 (Supervised Relevance Filtering):**
  * `step5_worker.py` — Trains a few-shot `SetFit` classifier using topic-held-out `GroupShuffleSplit` validation (preventing topic-level data leakage), verifies input consistency (`title + abstract`), runs an in-sample retention check on expert labels, and exports class probabilities (`proba_yes`) for decision-boundary auditing.
* **Step 6 (High-Fidelity Topic Mapping):**
  * `step6_worker.py` — Executes definitive `BERTopic` + `EVōC` clustering on the filtered relevant corpus, computes multi-criteria structural validation indices (**Standard CVI**, **Adaptive CVI**, and **Kneedle** inflection detection) in `data/04_reports/step6_cvi_report.md`, and generates 2D UMAP topographic maps per candidate layer.
  * `step6_apply_labels.py` — Applies validated topic names to the selected hierarchical layer and runs an automated topic-size verification routine (`data/04_reports/step6_verify_report.md`).
* **Step 7 (Sentence-Level Semantic Extraction):**
  * `step7_worker.py` & `vllm_manager.py` — Segments abstracts into sentences (or full abstracts, per vector scope), queries a local/remote `vLLM` OpenAI-compatible server (`temperature=0.0`, seeded) to extract micro-labels (e.g., *challenges*, *directions*, *controversies*), and clusters extracted micro-labels into macro-categories.
* **Step 8 (Scientometric Synthesis & Audits — Breakpoint 2):**
  * `step8_worker.py` — Computes cross-dimensional scientometric matrices (Thematic Lifecycles, Strategic Geopolitics, Bottleneck Regimes, and Impact Quadrants) in `data/04_reports/step8_data_crunching.md` and drafts `data/04_reports/step8_insights_review.md` for expert revision.
  * `audit_framework.py` — Generic human-in-the-loop audit engine (`export` and `compute` modes) that samples classified records (e.g., sentence-level challenge extractions and Step 5 decision-boundary records with $0.40 \le P(\text{yes}) \le 0.60$) into `data/03_review/step8_audit_*.xlsx` and calculates precision, recall, F1-score, accuracy, and confusion matrices (`data/04_reports/step8_audit_report.md`).
  * `step8_audits.py` — Automated statistical and sensitivity audits: pairwise ARI across EVōC layers, OLS impact regressions with percentile bootstrap confidence intervals ($B=1{,}000$) and Kruskal–Wallis tests, AICc growth-model selection (Plateau, Linear, Exponential), and Relevance Index unit-invariance checks.
  * `reference_date.py` — Resolves the fixed `analysis.reference_date` so time-dependent indicators (document age, citations per year, recent-year windows) remain invariant across re-executions.
* **Step 9 (Final Reporting & Visualization):**
  * `step9_worker.py` — Generates publication-grade figures (`data/04_reports/figures/`), compiles the validation & audit summary (`data/04_reports/step9_audit_summary.md`), and renders the standalone academic report (`AKL_Final_Report.md` and `AKL_Final_Report.html`).

### `.opencode/skills/` — Agent Orchestration Skills
Each subfolder contains a `SKILL.md` specification invoked by the OpenWork/OpenCode agent:
* `status` & `help` — Session diagnostics, pipeline progress reporting, and user guidance.
* `config_sync` — Translates natural-language overrides from Section 11 of `project_config.md` into `pipeline_config.yaml`.
* `step1_acquisition` through `step9_reporting` — Step-specific orchestration protocols enforcing preconditions, worker invocation, state verification, and human-in-the-loop validation gates.

---

## 🛠️ Installation & Environment Setup

The pipeline uses two isolated Python 3.11 Conda environments to prevent dependency conflicts between `BERTopic`/`EVōC` and `SetFit`, plus an OpenAI-compatible `vLLM` server for Step 7 semantic extraction.

### 1. Create the Conda environments
```bash
# Main automation environment (Steps 1–4, 6–9)
conda create -n akl_automation python=3.11 -y
conda activate akl_automation
pip install -r akl_automation_requirements.txt

# SetFit environment (Step 5 relevance filter)
conda create -n akl_setfit python=3.11 -y
conda activate akl_setfit
pip install -r akl_setfit_requirements.txt
```

### 2. Configure the shell wrappers
Ensure `CONDA_BASE` in `run_worker.sh` and `run_worker_setfit.sh` points to your local Miniconda/Anaconda installation (defaults to `${HOME}/miniconda3`) and make the scripts executable:
```bash
chmod +x run_worker.sh run_worker_setfit.sh run_step7_detached.sh
```

### 3. Start the vLLM Server (Required for Step 7, optional for Steps 3/4/6 if using agent mode)
On your GPU workstation, launch the OpenAI-compatible `vLLM` server:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model RedHatAI/gemma-3-4b-it-quantized.w8a8 \
    --max-model-len 102944 \
    --port 8000
```
Update `step7_vllm.base_url` in `pipeline_config.yaml` if the server runs on a remote host.

---

## 🚀 Running the Pipeline

### Option A: Conversational Orchestration via OpenWork / OpenCode
1. Open this repository folder as a workspace in **OpenWork** (`https://openworklabs.com/`) or **OpenCode**.
2. Fill in your research design in `project_config.md`.
3. Start a chat session with the agent (e.g., *"What is the current pipeline status?"* or *"Run Step 1"*).
4. Review and validate the human-in-the-loop artifacts at **Step 4** (`data/03_review/step4_relevance_review.xlsx`) and **Step 8** (`data/04_reports/step8_insights_review.md` and `data/03_review/step8_audit_*.xlsx`).

### Option B: Direct Command-Line Execution
All workers can also be invoked directly via the environment wrappers:
```bash
# Step 1: Data ingestion (wos | openalex | pubmed | arxiv | generic)
./run_worker.sh workers/step1_worker.py openalex --query "your search query"

# Step 2: Cleaning and standardization
./run_worker.sh workers/step2_worker.py

# Step 3: Preliminary topic modeling
./run_worker.sh workers/step3_worker.py --skip-llm
./run_worker.sh workers/step3_apply_labels.py --labels data/02_intermediate/step3_naming_output.json --best-layer 0

# Step 4: Sample export and curation application (Breakpoint 1)
./run_worker.sh workers/step4_worker.py --prepare-only
# (After reviewing data/03_review/step4_relevance_review.xlsx)
./run_worker.sh workers/step4_worker.py --finalize

# Step 5: Supervised relevance filtering (SetFit)
./run_worker_setfit.sh workers/step5_worker.py

# Step 6: High-fidelity topic mapping (EVoC + CVI + Kneedle)
./run_worker.sh workers/step6_worker.py --skip-llm
./run_worker.sh workers/step6_apply_labels.py --labels data/02_intermediate/step6_naming_output.json --best-layer 1

# Step 7: Sentence-level LLM semantic extraction (detached)
./run_step7_detached.sh

# Step 8: Scientometric synthesis & human/automated audits (Breakpoint 2)
./run_worker.sh workers/step8_worker.py
./run_worker.sh workers/audit_framework.py export
# (After reviewing data/03_review/step8_audit_*.xlsx)
./run_worker.sh workers/audit_framework.py compute

# Step 9: Final visualizations and report compilation
./run_worker.sh workers/step9_worker.py
```

---

## 📄 License and Citation

The custom code developed for this project is released for academic and research use. Feel free to use and adapt it, provided the original article is properly cited.

**Article Citation:**
> Luis C. Azevedo and Ronaldo C. Prati (2026). *Automated Knowledge Landscapes: A Human-in-the-Loop Semantic Bibliometric Roadmap for Strategic Discovery*. DOI: [To be added upon publication]

