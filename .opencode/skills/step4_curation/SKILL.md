---
name: step4_curation
description: Executes Step 4 of the AKL pipeline (Human-in-the-Loop Breakpoint). Handles AI suggestion of relevance labels based on project criteria, requests expert validation, and finalizes the curation step.
---

# AKL Pipeline — Step 4: Expert Curation and Sampling

## Preconditions

1. Read `pipeline_state.json`.
   - If Step 3 is not `completed`, stop: > Step 3 must be completed before running Step 4.
   - If Step 4 is `completed`, inform the user. Ask if they want to reopen the validation step.
   - Proceed based on the current `status` of Step 4 (`pending` or `awaiting_validation`).

---

## State: Pending (Starting Step 4)

If Step 4 `status` is `pending`:

1. Explain the goal: The pipeline needs a sample of documents labeled as "yes" (relevant) or "no" (not relevant) to train the AI filter (Step 5).
2. Offer the choice explicitly. Ask the user:
   > "We have reached the curation phase! I can analyze the criteria in Section 6 of your `project_config.md` and pre-label a sample of ~60 documents using AI for you to review, or you can prefer to do the labeling 100% manually.
   > **How would you like to proceed? Reply with 'suggest' or 'manual'.**"
3. STOP. Wait for the user's answer.

### User answers "suggest" (AI Suggestion)
Run the worker in suggest mode (Note: ensure vLLM is running):
```bash
cd <workspace_root>
./run_worker.sh workers/step4_worker.py suggest
```
Once it finishes, tell the user the file is ready at `data/03_review/step4_relevance_review.xlsx`. Explain that they must open it, review the "Relevance (yes/no)" column, correct any AI mistakes, and save. Tell them to keep this file: it is the record of their decisions and must not be regenerated after validation. Instruct them to reply with "validated" or "done" when finished.

### User answers "manual"
Run the worker in manual mode:
```bash
cd <workspace_root>
./run_worker.sh workers/step4_worker.py manual
```
Once it finishes, tell the user the file is ready at `data/03_review/step4_relevance_review.xlsx`. Remind them of the labeling rules (50-100 total, spread across topics, 50/50 balance). Instruct them to reply with "validated" or "done" when finished.

---

## State: Awaiting Validation (Finalizing Step 4)

If Step 4 `status` is `awaiting_validation` AND the user confirms they have finished reviewing (e.g., "validated", "done", "ready"):

1. Run the finalize mode to ingest their labels:
```bash
cd <workspace_root>
./run_worker.sh workers/step4_worker.py finalize
```
2. Read the JSON output or `pipeline_state.json` to get the final counts (`relevant_count` and `not_relevant_count`).
3. Report the result:
   ```text
   ✓ Step 4 completed — Expert Curation
   
   Relevant documents (Yes): [count]
   Not relevant documents (No): [count]
   ```
4. **CRITICAL CHECK**: If `not_relevant_count` is exactly `0`:
   > ⚠️ **Attention:** You did not mark any document as "no" (not relevant). Step 5 (SetFit) requires examples of both classes to train the filter. Since there are no negative examples, **Step 5 is not necessary and will be skipped.**
5. Inform the user they can now proceed to the next applicable step (Step 5, or Step 6 if Step 5 is skipped).
