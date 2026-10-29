---
name: step3_topic_modeling
description: Executes Step 3 of the AKL pipeline — Initial Topic Modeling. Runs BERTopic with EVoC clustering across multiple layers, selects the optimal layer, and names topics using the local vLLM server. Saves the model, topics_preliminary.xlsx, and updates pipeline_state.json.
---

# AKL Pipeline — Step 3: Initial Topic Modeling

## Preconditions — check before doing anything else

1. Read `pipeline_state.json`.
   - If step 2 `status` is not `completed`, stop and inform:
     > Step 2 must be completed before running Step 3. Please run the `step2_cleaning` skill first.
   - If step 3 `status` is `completed`, inform the specialist and ask explicitly if they want to **rerun** it (this will overwrite model checkpoints and topic spreadsheets). Warn that rerunning will also require rerunning all subsequent steps. Only proceed if they confirm.
   - If step 3 `status` is `running`, warn that a previous execution may have been interrupted and ask if they want to restart.

2. Confirm that `data/02_intermediate/step2_corpus_clean.pkl` exists. If not found, stop and inform:
   > `step2_corpus_clean.pkl` not found in `data/02_intermediate/`. Please rerun Step 2 to regenerate it.

3. Confirm that `run_worker.sh` exists in the workspace root. If not found, stop and inform:
   > `run_worker.sh` not found. This file is required to run workers in the correct Python environment.

---

## Execution

Inform the specialist that this step is computationally intensive and may take several minutes. Explain that:
1. It will compute embeddings on the device set in `step3_bertopic.embedding_device`.
2. It will automatically start the vLLM server if it is not already running.
3. It will cluster the data and prompt the LLM to name the topics.

Run the worker:

```bash
cd <workspace_root>
./run_worker.sh workers/step3_worker.py \
    --input data/02_intermediate/step2_corpus_clean.pkl \
    --config pipeline_config.yaml \
    --state pipeline_state.json
```

---

## Report result

After the worker completes, read `pipeline_state.json` to get the final stats and present:

```text
✓ Step 3 completed — Initial Topic Modeling

Records processed:      [n_records]
EVoC layers processed:  [n_layers_found]
Best layer selected:    [best_layer]
Topics in best layer:   [n_topics_best_layer]

Model artifacts:        data/checkpoints/step3_bertopic_model/
Topics files generated: data/05_exports/step3_topics_preliminary_layer_*.xlsx
Selected layer file:    [output_topics]
Corpus with topics:     data/02_intermediate/step3_corpus_with_topics.pkl
```

If any `notes` field in step 3 of `pipeline_state.json` is not empty, show it prefixed with ⚠️.

---

## Error handling

If the worker exits with a non-zero return code or prints to stderr:

1. Show the error output clearly.
2. Read `pipeline_state.json` — step 3 status should be `failed` with a note.
3. Report the error and suggest checking the execution log for details:
   > Check `execution.log` for the full error trace. Common issues include: vLLM server connection failures, GPU memory exhaustion, corpus too small for topic modeling, or configuration parameter issues.
4. Ask the specialist how they want to proceed.

---

## After successful completion

- Confirm that `pipeline_state.json` shows step 3 as `completed`.
- Confirm that the `step3_topics_preliminary_layer_*.xlsx` files exist in `data/05_exports/`.
- Inform the specialist that **Step 3 is complete and the optimal clusters have been formed.**
- Inform the specialist that **Step 4 is a Human-in-the-Loop Breakpoint**.
- Suggest they open the `_selected` layer file (or explore the other layers) in `data/05_exports/` to review the generated topics.
- Explain their options clearly: they can either open the Excel files now to perform the labeling **100% manually**, OR they can invoke the Step 4 skill (e.g., "Run step 4") to have the AI **suggest classifications** based on the project criteria.  
- Explain how to perform the manual curation clearly. Instruct the specialist to:
  - Look at the first column `"Relevance (yes/no)"` in the topic sheets and type **"yes"** for relevant documents and **"no"** for non-relevant ones.
  - **Quantity:** Aim to label between 50 and 100 documents in total (or about 5-10% of the data if the corpus is small).
  - **Distribution:** It is crucial to label documents across **ALL topics** to ensure semantic diversity for the AI training. Aim for at least 3 to 5 labeled items per topic sheet.
  - **Bulk labeling:** If an entire topic is clearly aligned or misaligned with the research, they can confidently label all items in that sheet as "yes" or "no".
  - **Balance:** Try to maintain a roughly **50/50 proportion** between "yes" and "no" labels across the whole dataset to prevent model bias.
  - Leave unreviewed rows completely blank.
- Remind them to run the `status` skill to verify the pipeline state.
- Warn the specialist not to rerun Step 3 after starting the labeling, because the labels refer to this run's topics.
- Do NOT automatically start Step 4.
