# AKL (Automated Knowledge Landscapes) - Official Manual

## 1. Overview
The AKL pipeline is a 9-step scientometric workflow designed to automate the semantic mapping and the systematic review of the scientific literature. Its distinguishing feature is the **Human-in-the-Loop** architecture: the pipeline automates the heavy computational load (extraction, clustering, statistical calculations), but pauses at critical validation points (*Breakpoints* — Steps 4 and 8) so that the results are checked through expert curation.

**Current version:** [TODO: release tag] — Complete pipeline 1–9 working, 4 data sources (WOS, OpenAlex, PubMed, generic CSV), skills/workers in English, artifacts generated in English, agent chat in the user's language. The pipeline is orchestrated by an LLM coding agent that reads `AGENTS.md` and the skill files in `.opencode/skills/`.

## 2. Main Configuration Files
Before running the pipeline, the user must work with the following files in the workspace root:

1. **`project_config.md`**: The research design document. It must be filled in by the expert before starting. This is where you define the *intent*, the research questions, and the scope of the review.
2. **`pipeline_config.yaml`**: Contains the technical parameters (models, remote vLLM settings, similarity thresholds). Used to adjust the AI to the capabilities of your hardware. It also contains the column mappings for the generic CSV import.
3. **`pipeline_state.json`**: The "brain" of the execution. It tracks which steps were completed. **Never edit this file manually**, except to change the `current_step` key to force a step to run again.

## 3. Supported Data Sources (Step 1)

Step 1 accepts **4 data sources**:

| Source | Type | How it works | Required configuration |
|---|---|---|---|
| **WOS** | Upload | Export file (.txt or .csv) from the Web of Science | `upload_filename` in `project_config.md` |
| **OpenAlex** | API | Query to the open OpenAlex API | `query` in `project_config.md` |
| **PubMed** | API | Query to the NCBI E-utilities API | `query` in `project_config.md`; optional API key |
| **Generic CSV** | Upload | Any CSV (Scopus, Dimensions, manual...) | `upload_filename` + mapping in `pipeline_config.yaml` |

### 3.1 PubMed (API)
- Requires only the **query string** (search terms), as for OpenAlex.
- **API key (optional, recommended):** free at https://www.ncbi.nlm.nih.gov/account/ — raises the limit from 3 to 10 requests per second.
- **Email (optional):** enables the NCBI "polite pool".
- Fields extracted automatically: title, abstract (structured or plain), authors, affiliations, country, year, DOI, journal, MeSH terms, keywords.
- **Limitation:** PubMed does not provide citation counts, so the field `citation_count` is set to 0.

### 3.2 Generic CSV (Upload)
- Accepts any CSV with columns that can be mapped to the canonical AKL schema.
- Supports two formats:
  - **Wide:** 1 row per document, author lists separated by `;`
  - **Long:** 1 row per author (repeated DOI); the worker aggregates them automatically
- The user must map the columns in the `source_mapping.generic` section of `pipeline_config.yaml`.
- **Dedicated guide:** see `generic_csv_import_guide.md` for detailed instructions.

## 4. Summary of the 9 Pipeline Steps
1. **Data Acquisition:** Connection to databases (WOS via upload, OpenAlex via API, PubMed via API, or generic CSV via upload).
2. **Cleaning and Standardization:** Removal of duplicates and structuring of the corpus.
3. **Preliminary Semantic Clustering:** Extraction of *embeddings* and generation of several layers of topic granularity with BERTopic/EVŌC.
4. **Expert Curation (Breakpoint 1):** The human evaluates a sample to teach the AI what is relevant to the scope.
5. **Relevance Filtering:** Training of a SetFit (few-shot) model to classify and filter the rest of the corpus.
6. **Topographic Mapping:** Generation of the final 2D map and consolidation of the final topics.
7. **Semantic Extraction (LLM):** Extraction of the configured vectors (e.g., *Challenges*, *Directions*, *Methods*) from sentences or abstracts, with one call per text and scope, using the vLLM server. An optional pilot, keyword pre-filter, and sentence verifier are described in `README.md`.
8. **Expert Synthesis (Breakpoint 2):** Processing of the scientometric matrices (Growth Cycles, Geopolitics, Impact Quadrants) and generation of an interpretive report by the Agent, which must be validated and complemented by the expert.
9. **Final Report and Visualizations:** Generation of charts (Dendrograms, Likert, Growth Curves) and compilation of the final HTML/PDF report.

---

## 4. Breakpoints (Human-in-the-Loop)
The pipeline stops at two points for human validation:

### Step 4: Expert Curation and Sampling
**Goal**: Validate the relevance of the documents to train the AI filter (Step 5).
**Procedure**:
1. In Step 3, the pipeline generates spreadsheets such as `step3_topics_preliminary_layer_X_selected.xlsx` in the folder `data/05_exports/`.
2. The expert opens the spreadsheet of the desired granularity layer.
3. In the column **"Relevance (yes/no)"**, the expert labels a sample of documents, filling in strictly **"yes"** (relevant) or **"no"** (not relevant).
4. **Guidelines for training the AI well**:
   - Label 50 to 100 documents in total.
   - Spread the labels across **all topics** (3-5 examples per topic).
   - Keep a proportion close to **50/50** between "yes" and "no".
5. Save the spreadsheet and tell the agent in the chat to proceed with Step 4.

### Step 8: Synthesis and Validation
**Goal**: Validate the insights, cross-analyses (e.g., Pioneers vs. Traditionalists), and bottlenecks generated by the LLM.
**Procedure**:
1. The Agent crosses the Step 7 data with the guidelines of your `project_config.md` and writes an executive draft.
2. The expert reviews the file `step8_insights_review.md` in the folder `data/04_reports/`.
3. The expert edits the file, correcting the text, adding their own judgment, and deleting what they consider incorrect.
4. **Requests for Step 9**: If you add a paragraph asking for a new cross-analysis (e.g., *"Generate a line chart showing the controversies over the years"*), leave it written in the text. The Agent will try to generate this *ad-hoc* chart in the next step.
5. Tell the agent in the chat (e.g., "validated") so that it closes the step.

---

## 5. Final Output (Step 9)
After Step 9 is completed, all visual assets will be in the folder `data/04_reports/figures/`.
The final document will be generated as **`data/04_reports/AKL_Final_Report.html`**.
Open this file in your web browser and use the system print function (`Ctrl+P`) to **Save as PDF**. The CSS formatting was set up for academic typography.

---

## 7. Versioning and Release Strategy

This project follows **Semantic Versioning (SemVer)** with pre-release identifiers during the beta phase:

| Format | Meaning | Example |
|---------|-------------|---------|
| `v0.y.z-beta.N` | Public beta / test with the advisor | `v0.2.0-beta.1` |
| `v0.y.z-rc.N` | Release Candidate (almost final) | `v0.9.0-rc.1` |
| `v1.0.0` | First stable/production version | `v1.0.0` |
| `v1.y.z` | Fixes/minor changes after release | `v1.1.0` |
| `vX.0.0` | Changes that break compatibility | `v2.0.0` |

### Branches on GitHub
- **`main`** — Latest stable version (tags `vX.Y.Z`). Receives merges only from `release/*` or hotfixes.
- **`develop`** — Continuous integration of validated features. Base for the next betas.
- **`feature/*`** — One branch per improvement (e.g., `feature/M3-data-reorg`, `feature/M6-llm-flex`).
- **`release/v0.x.y-beta.N`** — Temporary branch to stabilize a beta (tests, docs, small fixes). When closed, it is merged into `develop` and `main` + tag.
- **`hotfix/*`** — Urgent fixes in production.

### Suggested flow for this phase (beta with the advisor)
1. Work on the `feature/M*` branches from `develop`.
2. When a set of features is ready for testing, create `release/v0.x.y-beta.N` from `develop`.
3. The advisor tests on `release/*`. Adjustments go in as commits on this branch.
4. Once approved: merge `release/*` → `develop` and `release/*` → `main` + `git tag v0.x.y-beta.N`.
5. Delete the `release/*` branch.

### Current tags
- `v0.1.0` — Original MVP (main, Steps 1–7 incomplete)
- `v0.2.0-beta.1` — (melhorias_fase1, Steps 1–9 complete, M1/M2/M4/M5)
- `v0.3.0-beta.1` — **This version** (new data sources: PubMed API + generic CSV)

### Next planned milestones
- `v0.4.0-beta.1` — M7.x (tests, robust checkpoints, docs)
- `v0.9.0-rc.1` — Release Candidate (feature-complete, polishing only)
- `v1.0.0` — Stable version 1.0 for the thesis/publication

---

## 8. Execution Flow
All interaction happens through the chat of your coding agent (for example, Claude Code or OpenWork/OpenCode):
- Start the session by asking: *"What is the status of the pipeline?"*
- Move through the steps in sequence: *"Run step 1"* or *"Go to step 9"*.
- Make sure that the vLLM server (for Steps 7 and 8) is running on your infrastructure before requesting the extraction tasks.
