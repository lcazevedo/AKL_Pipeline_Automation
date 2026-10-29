---
name: step5_filtering
description: Executes Step 5 of the AKL pipeline. Trains a SetFit model using the expert's labels, evaluates its performance, filters the corpus to keep only relevant documents, and saves outputs.
---

# AKL Pipeline — Step 5: Corpus Filtering Model (SetFit)

## Preconditions

1. Read `pipeline_state.json`.
   - If Step 4 is not `completed`, stop: > Step 4 curation must be completed before running Step 5.
   - If Step 4 `not_relevant_count` is 0, stop: > The curation in Step 4 did not identify any non-relevant documents. The SetFit filter requires both classes to train. Step 5 is unnecessary and should be skipped. Proceed directly to Step 6.
   - If Step 5 is `completed`, warn the user that rerunning will overwrite the model and the filtered corpora.

---

## Execution

Inform the user that the SetFit model is being trained based on their labels and applied to the entire dataset. This may take a few minutes.

Run the worker:
```bash
cd <workspace_root>
./run_worker_setfit.sh workers/step5_worker.py
```

### Reviewer-driven behavior (audit framework)

Since the audit implementation, Step 5 includes three reviewer-driven safeguards (configurable in `pipeline_config.yaml` → `step5_setfit`):

1. **GroupShuffleSplit** (`group_by_topic: true`) — the train/test split groups documents by their Step-3 topic, so documents from the same topic never appear in both splits. This prevents topic-level leakage and gives an honest generalization estimate.
2. **Retention check** (`retention_check: true`) — after training, the model is re-applied to ALL Step-4 labeled documents and the retention accuracy is reported in `step5_setfit_report.md`. A retention accuracy well below eval accuracy indicates overfitting or label noise.
3. **`proba_yes` column** (`save_proba_yes: true`) — the probability of the "yes" class is saved in the filtered corpora, enabling the Step-8 boundary audit (documents the model is unsure about, confidence near 0.5).

---

## Report result

Read `pipeline_state.json` to present the final stats:

```text
✓ Step 5 completed — SetFit Filtering

Model Evaluation:
- Accuracy:  [accuracy]
- F1-Score:  [f1_score]
- Retention Accuracy: [retention_accuracy]
- Split Method: [split_method]
- Detailed Report: data/04_reports/step5_setfit_report.md

Corpus Filtering:
- Original size: [records_before]
- Relevant (Kept): [records_after]
- Not Relevant (Discarded): [records_discarded]

Outputs generated:
- Pipeline input (Relevant): data/02_intermediate/step5_corpus_filtered_relevant.pkl
- Alternative use (Not relevant): data/02_intermediate/step5_corpus_filtered_not_relevant.pkl
- Human inspection (Relevant): data/05_exports/step5_corpus_filtered_relevant.xlsx
- Human inspection (Not relevant): data/05_exports/step5_corpus_filtered_not_relevant.xlsx
- Full prediction table (boundary audit): data/02_intermediate/step5_full_predictions.pkl
```

Inform the user about the Excel files:
> 💡 **Tip:** You can open the `.xlsx` files in the `data/05_exports/` folder to view the filtered results. These files include an `AI_Confidence` column showing how certain the model was for each prediction, and a `proba_yes` column with the probability of the "yes" class (used by the Step-8 boundary audit).

If `records_after` < 100, add a warning:
> ⚠️ **Attention:** The resulting filtered corpus is very small (fewer than 100 documents). This may hinder topic extraction in subsequent steps.

Suggest the user run the `status` skill, or ask if they want to proceed to Step 6 (EVoC High-fidelity topic mapping).
