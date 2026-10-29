---
name: status
description: Reports the current state of the AKL pipeline. Always run this skill at the start of a session or whenever you need to know where the pipeline stands. Reads pipeline_state.json and project_config.md and presents a clear summary of progress, pending actions, and next steps.
---

# AKL Pipeline — Status Skill

## Your task

Report the current state of the AKL pipeline to the specialist. This skill is the recommended entry point for every session.

## Steps to follow

### 0. Run config_sync first

Before reading the state, run the `config_sync` skill to apply any natural-language configuration overrides from Section 11 of `project_config.md` into `pipeline_config.yaml`. This ensures the status report reflects the specialist's latest requested adjustments.

- If Section 11 is empty, `config_sync` reports "no overrides" and finishes — proceed normally.
- If overrides exist, `config_sync` confirms with the specialist and applies them — then proceed.

### 1. Read the state file
Read `pipeline_state.json` from the workspace root.
If the file does not exist, report:
> pipeline_state.json not found. Please ensure you are in the correct workspace and that the file has been created. You can create it from the template provided in the project documentation.

### 2. Read the project config
Read `project_config.md` from the workspace root.
If not found, note it as missing in the report but do not stop execution.

### 3. Build the status report

Present the report in this exact format:

---

**AKL Pipeline — Status Report**
Project: [project.name from pipeline_state.json, or "not configured" if empty]
Last updated: [project.last_updated, or "never"]

**Progress overview:**

| Step | Name | Status |
|------|------|--------|
| 1 | Raw data acquisition | [status icon + status] |
| 2 | Data cleaning and structuring | [status icon + status] |
| 3 | Initial topic modeling (BERTopic) | [status icon + status] |
| 4 | Expert curation — BREAKPOINT | [status icon + status] |
| 5 | Corpus filtering (SetFit) | [status icon + status] |
| 6 | High-fidelity topic mapping (EVOC) | [status icon + status] |
| 7 | Semantic enrichment (LLM) | [status icon + status] |
| 8 | Expert synthesis — BREAKPOINT | [status icon + status] |
| 9 | Final output | [status icon + status] |

Use these icons for status:
- ✓ completed
- → running
- ★ awaiting_validation  ← requires specialist action
- ✗ failed  ← requires attention
- ○ pending

**Current focus:**
[One sentence describing where the pipeline is right now.]

**Required action:**
[What the specialist needs to do next. Be specific:
- If a step is awaiting_validation: name the exact file to review and where it is.
- If the next step is pending: name the step and ask for confirmation to proceed.
- If a step failed: describe what went wrong using the notes field and suggest corrective action.
- If all steps are completed: confirm completion and offer options.]

---

### 4. Check for warnings

After the report, scan all steps and surface any warnings:
- Any step where `notes` field is not empty → show the note prefixed with ⚠️
- Step 5: if `params_used.records_after` is less than 100, warn that the filtered corpus is very small.
- Step 7: if `params_used.checkpoint_last` is not empty and status is not `completed`, warn that an interrupted extraction was detected and offer to resume.

### 5. Suggest the next action

Close with a single clear suggestion:
> **Suggested next action:** [specific instruction]

Examples:
- "Run Step 1 to begin data acquisition. Shall I start?"
- "Step 4 is waiting for your validation. Please open `data/review/relevance_review.xlsx`, review the suggestions, and confirm when done."
- "Step 7 was interrupted at checkpoint [value]. Shall I resume from where it stopped?"
- "All steps are complete. Would you like to review any output or run a new project?"

## Important rules

- Never modify any file in this skill — this is a read-only operation.
- Always reflect exactly what is in pipeline_state.json. Do not infer or assume status from file existence alone.
- If pipeline_state.json has current_step as 0 and all steps are pending, this is a fresh pipeline — say so explicitly and guide the specialist to fill in project_config.md first if it is incomplete.
- Respond in the same language the specialist is using.
