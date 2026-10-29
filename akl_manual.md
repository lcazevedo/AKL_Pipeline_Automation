# AKL (Automated Knowledge Landscapes) - Official Manual v0.3.0-beta.1

## 1. Overview
The AKL pipeline is a 9-step scientometric workflow designed to automate semantic mapping and systematic literature reviews. Its key differentiator is its **Human-in-the-Loop** architecture: the pipeline automates the heavy computational workload (extraction, clustering, statistical calculations), while pausing at critical validation points (*Breakpoints* — Steps 4 and 8) to ensure scientific rigor through expert curation.

**Current version:** v0.3.0-beta.1 — Full functional pipeline (Steps 1–9), 5 data sources (WOS, OpenAlex, PubMed, arXiv, generic CSV), skills/workers in English, generated artifacts in English, and agent chat in the user's preferred language.

## 2. Core Configuration Files
Before running the pipeline, the user interacts with the following files in the workspace root:

1. **`project_config.md`**: The research design document. Must be completed by the domain expert before starting. This is where you define the *research intent*, research questions, and review boundaries.
2. **`pipeline_config.yaml`**: Contains technical parameters (global seed, models, remote vLLM settings, EVoC parameters, Step 8 audits). Used to tailor AI models to your hardware capabilities. Also contains column mappings for generic CSV import.
3. **`pipeline_state.json`**: The execution "brain". It tracks which steps have been completed. **Never edit this file manually**, except if you need to change the `current_step` key to force re-execution of a step.

## 2.1 Reproducibility
The global seed is stored in `pipeline_config.yaml` → `reproducibility.seed` (default 42) and is inherited by Steps 3, 4, 5, 6, and 7 (each step can override it via `random_state`/`seed`). For a new execution to reproduce the cluster partitions:
- Use the pinned versions in the requirements file (especially `evoc>=0.3.1`, which accepts `random_state`; earlier versions are **not** reproducible).
- Prefer `step3_bertopic.embedding_device: "cpu"` and `step7_vllm.temperature: 0.0`.
- Preserve the human-in-the-loop artifacts (Step 4 and Step 8 spreadsheets): they anchor the execution.
- LLM outputs (topic labels, Step 7 extractions) may not be bit-for-bit identical even with a fixed seed; reproducibility of these stages should be evaluated as cross-run agreement.
- Each step logs package versions and seeds in `pipeline_state.json` (`params_used.reproducibility`).

## 3. Supported Data Sources (Step 1)

Step 1 accepts **5 data sources** (WOS, OpenAlex, PubMed, arXiv, and generic CSV):

| Source | Type | How it works | Required configuration |
|---|---|---|---|
| **WOS** | Upload | Export file (.txt or .csv) from Web of Science | `upload_filename` in `project_config.md` |
| **OpenAlex** | API | Query against the OpenAlex open API | `query` in `project_config.md` |
| **PubMed** | API | Query against the NCBI E-utilities API | `query` in `project_config.md`; optional API key |
| **arXiv** | API | Query against the arXiv Atom API (no citation count; fallback DOI `10.48550/arXiv.<id>`) | `query` in `project_config.md` |
| **Generic CSV** | Upload | Any CSV (Scopus, Dimensions, manual...) | `upload_filename` + mapping in `pipeline_config.yaml` |

### 3.1 PubMed (API)
- Requires only the **query string** (search terms), similar to OpenAlex.
- **API key (optional, recommended):** free at https://www.ncbi.nlm.nih.gov/account/ — increases the rate limit from 3 to 10 requests/second.
- **Email (optional):** enables NCBI's "polite pool".
- Automatically extracted fields: title, abstract (structured or plain), authors, affiliations, country, year, DOI, journal, MeSH terms, keywords.
- **Limitation:** PubMed does not provide citation counts — the `citation_count` field is set to 0.

### 3.2 Generic CSV (Upload)
- Accepts any CSV with columns mappable to the AKL canonical schema.
- Supports two formats:
  - **Wide:** 1 row per document, `;`-separated author lists
  - **Long:** 1 row per author (repeated DOI) — the worker aggregates rows automatically
- The user maps columns in the `source_mapping.generic` section of `pipeline_config.yaml`.
- **Dedicated guide:** see `generic_csv_import_guide.md` for detailed instructions.

## 4. Summary of the 9 Pipeline Steps
1. **Data Acquisition:** Connection to bibliographic databases (WOS via upload, OpenAlex via API, PubMed via API, arXiv via API, or generic CSV via upload).
2. **Cleaning and Standardization:** Duplicate removal and corpus structuring.
3. **Preliminary Semantic Clustering:** Embedding extraction and generation of multi-layer topic hierarchies using BERTopic/EVŌC.
4. **Expert Curation (Breakpoint 1):** The human specialist evaluates a stratified sample to teach the AI what is relevant to the research scope.
5. **Relevance Filtering:** Training a SetFit (Few-Shot) model to classify and filter the remainder of the corpus.
6. **Topographic Mapping:** Generation of the final 2D map and consolidation of definitive topics.
7. **Semantic Extraction (LLM):** Sentence-by-sentence (or abstract-level) extraction of configured semantic vectors (e.g., *Challenges*, *Directions*, *Controversies*) using the vLLM server.
8. **Expert Synthesis (Breakpoint 2):** Computation of scientometric matrices (Growth Lifecycles, Geopolitics, Impact Quadrants) and generation of an interpretive draft report by the Agent, to be validated and enriched by the specialist.
9. **Final Report and Visualizations:** Generation of advanced scientific charts (Dendrograms, Likert charts, Growth Curves) and compilation of the final HTML/PDF report.

---

## 5. Breakpoints (Human-in-the-Loop)
The pipeline pauses at two vital moments for human validation:

### Step 4: Expert Curation and Sampling
**Objective**: Validate document relevance to train the AI filter (Step 5).
**Procedure**:
1. In Step 3, the pipeline generates workbooks such as `step3_topics_preliminary_layer_X_selected.xlsx` in `data/05_exports/`.
2. The specialist reviews the curation spreadsheet (`data/03_review/step4_relevance_review.xlsx`).
3. In the **"Relevance (yes/no)"** column, the specialist classifies a sample of documents strictly with **"yes"** (relevant) or **"no"** (not relevant).
4. **Guidelines for effective AI training**:
   - Classify 50 to 100 documents in total.
   - Distribute labels across **all topics** (3–5 examples per topic).
   - Aim for an approximately **50/50** balance between "yes" and "no".
5. Save the spreadsheet and notify the agent in chat to finalize Step 4.

### Step 8: Synthesis and Validation
**Objective**: Validate the insights, cross-tabulations (e.g., Pioneers vs. Traditionalists), and bottlenecks generated by the LLM.
**Procedure**:
1. The Agent cross-references Step 7 outputs with the guidelines in `project_config.md` and generates an in-depth executive draft.
2. The specialist reviews `step8_insights_review.md` in `data/04_reports/` (and any exported audit spreadsheets in `data/03_review/step8_audit_*.xlsx`).
3. Edit the file, refining the text, adding domain intuition, and removing anything inaccurate.
4. **Step 9 Ad-Hoc Requests**: If you add a paragraph requesting a new data cross-tabulation or chart (e.g., *"Generate a line chart showing controversies over the years"*), leave that note in the text. The Agent can generate that chart *ad-hoc* in Step 9!
5. Notify the agent in chat (e.g., "validated") to complete the step.

---

## 6. Final Output (Step 9)
After completing Step 9, all visual assets are saved in `data/04_reports/figures/`.
The final standalone document is generated at **`data/04_reports/AKL_Final_Report.html`**.
Open this file in your web browser and use the system print dialog (`Ctrl+P`) to **Save as PDF**. The CSS stylesheet is optimized for academic typographic quality.

---

## 7. Versioning and Release Strategy

This project follows **Semantic Versioning (SemVer)** with pre-release identifiers during the beta phase:

| Format | Meaning | Example |
|---------|-------------|---------|
| `v0.y.z-beta.N` | Public beta / advisor testing | `v0.2.0-beta.1` |
| `v0.y.z-rc.N` | Release Candidate (feature-complete) | `v0.9.0-rc.1` |
| `v1.0.0` | First stable/production release | `v1.0.0` |
| `v1.y.z` | Post-release patches / minor improvements | `v1.1.0` |
| `vX.0.0` | Breaking changes | `v2.0.0` |

### GitHub Branches
- **`main`** — Latest stable version (`vX.Y.Z` tags). Receives merges only from `release/*` or hotfixes.
- **`develop`** — Continuous integration of validated features. Base for upcoming betas.
- **`feature/*`** — One branch per improvement (e.g., `feature/M3-data-reorg`, `feature/M6-llm-flex`).
- **`release/v0.x.y-beta.N`** — Temporary branch to stabilize a beta (testing, docs, targeted fixes). When closed, merges into `develop` and `main` + tag.
- **`hotfix/*`** — Urgent production fixes.

### Suggested Workflow for the Beta Phase
1. Work on `feature/M*` branches off `develop`.
2. When a set of features is ready for testing: create `release/v0.x.y-beta.N` from `develop`.
3. Test on `release/*`. Fixes are committed directly to this branch.
4. Once approved: merge `release/*` → `develop` and `release/*` → `main` + `git tag v0.x.y-beta.N`.
5. Delete `release/*`.

### Release Tags
- `v0.1.0` — Original MVP (`main`, incomplete Steps 1–7)
- `v0.2.0-beta.1` — Phase 1 improvements (complete Steps 1–9, M1/M2/M4/M5)
- `v0.3.0-beta.1` — **Current version** (new data sources: PubMed API + generic CSV, arXiv, reproducibility & audit framework)

### Planned Milestones
- `v0.4.0-beta.1` — M7.x (automated tests, robust checkpoints, documentation)
- `v0.9.0-rc.1` — Release Candidate (feature-complete, final polish)
- `v1.0.0` — Stable 1.0 release for publication

---

## 8. Execution Flow
All interaction takes place through the Agent Chat in OpenWork/OpenCode:
- Start the session by asking: *"What is the pipeline status?"*
- Advance through the steps sequentially: *"Run step 1"* or *"Proceed to step 9"*.
- Ensure that the vLLM server (for Steps 7 and 8) is active on your infrastructure before requesting semantic extraction tasks.

