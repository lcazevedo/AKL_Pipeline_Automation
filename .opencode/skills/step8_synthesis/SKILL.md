---
name: step8_synthesis
description: Executes Step 8 of the AKL pipeline using Agent-Driven Analysis. The agent runs the local worker to crunch advanced statistical matrices (Growth Regimes, Pioneers vs Traditionalists, Bottlenecks, Impact Quadrants) and independently drafts the deep qualitative synthesis.
---

# AKL Pipeline — Step 8: Expert Synthesis & Validation (Agent-Driven)

## Preconditions

1. Read `pipeline_state.json`.
   - If Step 7 is not `completed`, stop: > Step 7 must be completed before starting the Specialized Synthesis (Step 8).
   - If Step 8 `status` is `completed`, inform the user and ask if they want to revise the synthesis.

---

## Phase 1: Data Crunching (Local Worker)

If Step 8 `status` is `pending`:

Inform the user:
> "Starting Step 8 using Agent-Driven Analysis. I will trigger the local worker to compute the Advanced Matrices (Growth Regimes, Impact Quadrants, Bottlenecks, and Geopolitical Profiles) using Pandas."

Use the `cmd_exec` tool to run the data crunching worker:
`./run_worker.sh workers/step8_worker.py run`

The worker now also:
1. Runs the **automated audits** (reviewer validations #5, #9, #10, #11) and appends them as Section 5 of `step8_data_crunching.md`:
   - EVoC sensitivity (pairwise ARI between Step-6 layers)
   - Regression audit (OLS + bootstrap CIs + Kruskal-Wallis + robustness)
   - Growth model fitting (AICc model selection per macro-category)
   - Relevance Index sensitivity (conditional — skipped when RI is absent)
2. Exports the **audit review samples** (Phase 1.5 below).

Once it succeeds, the state will be `awaiting_agent_synthesis`. Proceed immediately to Phase 2.

---

## Phase 1.5: Audit Sample Export (Generic Audit Framework)

The Step 8 "study" phase also covers **auditing** the pipeline's classification outputs. The generic audit framework samples classified items into review files that the specialist validates at this breakpoint.

1. Read the `audits:` section of `pipeline_config.yaml` to see which audits are enabled.
2. The audit export already ran inside `step8_worker.py run` (Phase 1). If it needs to be re-run standalone:
   `./run_worker.sh workers/audit_framework.py export`
3. Report to the user which audit files were generated:
   > "Audit samples exported to `data/03_review/step8_audit_<name>.xlsx`. Please review them during this breakpoint and fill the `audit_label` column with the correct labels."

### How to fill the `audit_label` column (MUST relay to the user)

When reporting the audit files, the agent MUST explain exactly how to fill
`audit_label`, because the comparison target depends on the audit type:

1. **Micro-label audits** (e.g., `challenges_micro`, `directions_micro`,
   `controversies_micro` — configured with `predicted_col: micro_label`):
   - The framework compares `audit_label` (expert gold) against **`micro_label`**
     (the AI's sentence-level prediction) — **NOT** against `macro_category`.
   - `macro_category` is the coarse aggregated grouping used for the Step-8
     matrices; comparing against it would be too permissive and would mask real
     extraction errors.
   - Fill each row with the **correct micro-label** for the sentence in
     `original_text`:
     - If the AI got it right → copy the value from `micro_label`.
     - If the AI got it wrong → write a short, descriptive label of the actual
       phenomenon in the sentence (keep the same style/vocabulary as the
       existing micro-labels).
     - Leave blank to skip a row (excluded from metrics as "pending").
   - ⚠️ The comparison is **exact string match** (case-insensitive, stripped):
     any rewording (even synonyms) counts as a mismatch, so the accuracy metric
     is conservative.

2. **Boundary audit** (`boundary` — configured with `predicted_col:
   predicted_relevance`): fill `audit_label` with **`yes`** (relevant) or
   **`no`** (not relevant) for each document sampled from the SetFit decision
   boundary (P(relevant) in [0.40, 0.60]).

3. **Macro-category audit** (if enabled — `predicted_col: macro_category`):
   fill `audit_label` with the correct **macro-category name** for the row.

The agent should relay this explanation in the user's language whenever audit
files are exported, and point out which `predicted_col` each audit compares
against (read it from the `audits:` section of `pipeline_config.yaml`).

Notes:
- The `boundary` audit (reviewer validation #2) samples documents with P(relevant) in [0.40, 0.60] from `step5_full_predictions.pkl`. If that file does not exist yet (Step 5 was not re-run with `save_proba_yes: true`), the audit is skipped gracefully — mention this to the user.
- If no audits are enabled (`enabled: false` for all), skip this phase silently.

---

## Phase 2: Agent Cognitive Synthesis

Read the contents of the following files using your file reading tools:  
1. `project_config.md` (Focus on Section 8: Analytical Questions & Focus Areas).  
2. `data/04_reports/step8_data_crunching.md` (The pre-calculated matrices).  
3. `data/04_reports/step8_audit_report.md` (Audit metrics, if any audits were computed).  

**AGENT INSTRUCTIONS FOR SYNTHESIS:**
You are a Senior Scientist. You must write the final executive synthesis document extracting profound insights from the data crunching report.  
- **Thematic Lifecycle:** Detail what is Exponential/Emerging (the new frontier) vs what is Plateau/Mainstream (mature technologies).  
- **Strategic Geopolitics:** Analyze the "Pioneers vs Traditionalists" matrix. Who is shifting their portfolio to emerging topics vs who relies on high-volume legacy tech?  
- **Bottleneck Regimes:** Analyze the row-normalized Topic vs Challenges matrix. Identify structural patterns (e.g., do mature topics suffer from analytical bottlenecks while emerging ones suffer from fabrication constraints?).  
- **Volume vs Efficiency (Impact Quadrants):** If available, contrast the "Volume Giants" with the "Efficiency Leaders" (Star/Hidden Gem ratios).  
- **Answer the Config:** Directly address any specific analytical questions posed in `project_config.md`.  

**CRITICAL MARKDOWN FORMATTING RULES:**
1. **Metadata Header:** Use a bulleted list for the initial document info to prevent PDF rendering issues.  
   * **Document:** Specialized Synthesis (Human Breakpoint)  
   * **Empirical Basis:** Semantic Extractions and Scientometric Matrices (Steps 1-7)  
   * **Generated on:** [Current Date]  
2. **NO DENSE TABLES:** DO NOT use Markdown tables for text-heavy interpretations. Tables render poorly in PDFs when they contain paragraphs. Instead, use structured lists (`-`, `*`) and subheadings.  
   *Example:*  
   ### Pioneer Fronts  
   * **[Name of Vector]**: Strategic reading goes here in a normal paragraph...  
3. Use clear hierarchies (`##`, `###`) and blockquotes (`>`) for emphasis.  

**Output:** Use your file editing tools to create and write the full synthesis into `data/04_reports/step8_insights_review.md`. Use highly professional academic language **in English**.

Inform the user:
> "I have generated the deep executive synthesis crossing all methodological dimensions (Growth Regimes, Strategic Matrices, and Impact Quadrants). The optimized file is available at `data/04_reports/step8_insights_review.md`. Since this is a **Human Breakpoint**, please open the file, add your expert insights, and let me know with 'validated' or 'completed' to finish this step."

---

## Phase 3: Finalization

If the user confirms they have validated/edited the Markdown file (e.g., "validated"):

1. If any audit review files were exported (Phase 1.5), run the audit metrics computation:
   `./run_worker.sh workers/step8_worker.py compute_audits`
   This reads the reviewed `audit_label` columns and writes `data/04_reports/step8_audit_report.md`.
2. Use your python or file tools to update `pipeline_state.json` setting Step 8 status to `completed` and registering the `completed_at` timestamp.

Inform the user:
> "✓ Step 8 Completed Successfully! Your specialized synthesis has been saved and the audit finalized. We are ready for **Step 9**, where we will generate the Dashboards and Topographic Visualizations for the final report."
