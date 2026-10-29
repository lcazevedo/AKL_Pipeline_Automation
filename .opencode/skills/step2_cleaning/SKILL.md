---
name: step2_cleaning
description: Executes Step 2 of the AKL pipeline — Data Cleaning and Structuring. Validates mandatory fields, removes low-quality abstracts, deduplicates by DOI and title, and saves corpus_clean.pkl to data/inputs/. Generates a quality report. Updates pipeline_state.json on completion.
---

# AKL Pipeline — Step 2: Data Cleaning and Structuring

## Preconditions — check before doing anything else

1. Read `pipeline_state.json`.
   - If step 1 `status` is not `completed`, stop and inform:
     > Step 1 must be completed before running Step 2. Please run the `step1_acquisition` skill first.
   - If step 2 `status` is `completed`, inform the specialist and ask explicitly if they want to **rerun** it (this will overwrite `corpus_clean.pkl`). Warn that rerunning will also require rerunning all subsequent steps. Only proceed if they confirm.
   - If step 2 `status` is `running`, warn that a previous execution may have been interrupted and ask if they want to restart.

2. Confirm that `data/02_intermediate/step1_corpus_raw.pkl` exists. If not found, stop and inform:
   > `corpus_raw.pkl` not found in `data/inputs/`. Please rerun Step 1 to regenerate it.

3. Confirm that `run_worker.sh` exists in the workspace root. If not found, stop and inform:
   > `run_worker.sh` not found. This file is required to run workers in the correct Python environment.

---

## Execution

No branching in this step — run directly:

```bash
cd <workspace_root>
./run_worker.sh workers/step2_worker.py \
    --input data/02_intermediate/step1_corpus_raw.pkl \
    --config pipeline_config.yaml \
    --state pipeline_state.json \
    --project-config project_config.md
```

Inform the specialist that execution has started and may take a moment depending on corpus size.

---

## Report result

After the worker completes, read `pipeline_state.json` to get the final stats and present:

```
✓ Step 2 completed — Data Cleaning

Records input:                  [records_input]
Records output (corpus_clean):  [records_output]
Removed — missing mandatory:    [removed_missing_mandatory]
Removed — abstract too short:   [removed_abstract_quality]
Removed — duplicates:           [removed_duplicates]
Removed — missing extra fields: [removed_missing_extra]

Quality report: data/04_reports/step2_quality_report.md
Output file:    data/02_intermediate/step2_corpus_clean.pkl
```

Then read `data/04_reports/step2_quality_report.md` and surface any warnings found there:
- If retention rate < 70%: show the warning and suggest the specialist review the quality report before continuing.
- If any `notes` field in step 2 of `pipeline_state.json` is not empty: show it prefixed with ⚠️.

---

## Error handling

If the worker exits with a non-zero return code or prints to stderr:

1. Show the error output clearly.
2. Read `pipeline_state.json` — step 2 status should be `failed` with a note.
3. Report the error and suggest checking the execution log for details:
   > Check `execution.log` for the full error trace. Common issues include: missing input files from previous steps, schema mismatches, configuration errors, or dependency issues.
4. Ask the specialist how they want to proceed.

---

## After successful completion

- Confirm that `pipeline_state.json` shows step 2 as `completed`.
- Confirm that `data/02_intermediate/step2_corpus_clean.pkl` and `data/05_exports/step2_corpus_clean.csv` exist.
- Suggest the specialist open `data/04_reports/step2_quality_report.md` for a detailed breakdown.
- Suggest running the `status` skill to see the full pipeline overview.
- Do NOT automatically start Step 3 — wait for the specialist's instruction.
