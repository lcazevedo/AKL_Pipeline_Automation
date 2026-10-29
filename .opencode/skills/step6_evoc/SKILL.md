---
name: step6_evoc
description: Executes Step 6 of the AKL pipeline. Maps final topics using EVoC on the filtered corpus. Evaluates cluster validity indices (CVI) and Kneedle to find the optimal layer. Requests user validation if there is no mathematical consensus.
---

# AKL Pipeline — Step 6: High-fidelity topic mapping (EVOC)

## Preconditions

1. Read `pipeline_state.json`.
   - If Step 5 is not `completed` (or if it wasn't skipped due to zero negatives in Step 4), stop: > Step 5 must be completed before running Step 6.
   - Proceed based on the current `status` of Step 6 (`pending` or `awaiting_validation`).

---

## State: Pending (Starting Step 6)

If Step 6 `status` is `pending`:

1. Inform the user that the system will run EVoC topic mapping on the relevant documents and evaluate the optimal layer using CVI metrics and the Kneedle algorithm. 
2. Mention that it will also generate 2D maps for layers with less than 50 topics.
3. Run the worker in cluster mode:
```bash
cd <workspace_root>
./run_worker.sh workers/step6_worker.py run
```
4. Check the JSON output or `pipeline_state.json`:
   - If `status` is `completed`: The math reached a consensus! Go to the **Report Result** section.
   - If `status` is `awaiting_validation`: No consensus was reached. Proceed to the Breakpoint instructions below.

---

## State: Awaiting Validation (Breakpoint)

If Step 6 returns `awaiting_validation` or was already in this state:

1. Inform the user:
   > "⚠️ **The mathematical metrics did not reach a consensus on the optimal layer.**
   > - CVI 1 (Square Root) suggested Layer: [cvi1_sqrt]
   > - CVI 2 (Adaptive) suggested Layer: [cvi2_adaptive]
   > - Kneedle (Knee of curve) suggested Layer: [kneedle]
   > 
   > To make the decision, please open and analyze:
   > 1. The report at `data/04_reports/step6_cvi_report.md`
   > 2. The decay plot at `data/04_reports/step6_kneedle_plot.png`
   > 3. The 2D maps at `data/05_exports/cluster_maps/`
   > 4. The spreadsheets generated at `data/05_exports/topics_final_layer_*.xlsx`
   >
   > **Which layer do you decide to use for the next steps? (Reply with the layer number only)**"
2. STOP and wait for the user to provide a layer number.

### When the user replies with a layer number:
Run the worker in finalize mode, passing the chosen layer:
```bash
cd <workspace_root>
./run_worker.sh workers/step6_worker.py finalize [NUMBER]
```
The finalize mode now also **re-aligns the saved model to the chosen layer**
(reviewer validation #12): the model's topic assignments and embeddings are
updated to match the exported Excel, so the saved model is internally consistent
with the selected layer.

Then proceed to the **Report result** section.

---

## Verify mode (reviewer validation #12 — topic sizes)

After Step 6 completes (consensus or finalize), run the verify mode to confirm
that the topic sizes in the exported Excel match the actual per-document
assignments:

```bash
cd <workspace_root>
./run_worker.sh workers/step6_worker.py verify
```

The verify mode performs two checks:
1. **Primary** — compares the saved per-layer assignments
   (`data/checkpoints/step6_layer_assignments.pkl`) with the Excel `Topics`
   sheet counts. Must match exactly.
2. **Secondary (informational)** — re-predicts with the saved model. EVoC
   assignments come from hierarchical clustering while `BERTopic.transform()`
   uses cosine similarity, so boundary documents may differ; reported for
   transparency only.

The report is written to `data/04_reports/step6_verify_report.md` and the
result is recorded in `pipeline_state.json` (`verify_primary_verified`).

---

## Report result

Read `pipeline_state.json` to present the final stats:

```text
✓ Step 6 completed — High-fidelity Topic Mapping (EVoC)

Topics mapped in the final corpus.
Selected Optimal Layer: [validation.suggested_layers if consensus else user choice]

Outputs generated:
- Model: data/checkpoints/step6_bertopic_model/
- Per-layer assignments (verify): data/checkpoints/step6_layer_assignments.pkl
- Metrics Report: data/04_reports/step6_cvi_report.md
- Decay Plot: data/04_reports/step6_kneedle_plot.png
- 2D Cluster Maps: data/05_exports/cluster_maps/
- Final Topic Spreadsheet: data/05_exports/topics_final_layer_[selected]_selected.xlsx
```

Inform the user they can proceed to Step 7 (Semantic Enrichment via LLM).
