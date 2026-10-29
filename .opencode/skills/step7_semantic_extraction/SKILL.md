---
name: step7_semantic_extraction
description: Executes Step 7 of the AKL pipeline. Reads active semantic vectors from project_config.md, asynchronously extracts micro-labels using vLLM, clusters them semantically using CPU embeddings, and names macro-categories to generate a relational Tidy Data format output.
---

# AKL Pipeline — Step 7: Semantic Extraction

## Preconditions

1. Read `pipeline_state.json`.
   - If Step 6 is not `completed`, stop: > Step 6 must be completed before running Step 7.
   - If Step 7 is `completed`, warn the user that rerunning will overwrite the extractions and take considerable time.
   
2. Ensure vLLM configurations are set properly in `pipeline_config.yaml` as this step makes thousands of API requests.

---

## Execution

Inform the user:
> "Starting Semantic Extraction (Step 7). This step reads the active vectors from Table 7 of your `project_config.md`. It will make thousands of requests to vLLM to extract the data and then group everything into Macro-Categories. This process is time-consuming. The worker will run in the background and the agent will report progress in the chat every ~20 seconds (current stage + percentage per vector)."

### Launch the worker fully detached

⚠️ **Never use plain `nohup ... &`**: the shell tool waits for the call to finish
and kills the whole process group at its timeout (default 120s), destroying the
worker mid-run. Use the dedicated helper `run_step7_detached.sh`, which launches
the worker via `subprocess.Popen(start_new_session=True)` with no inherited file
descriptors — the call returns **immediately** and the worker survives any tool
timeout (verified: a worker killed this way resumes from its last checkpoint):

```bash
cd <workspace_root>
./run_step7_detached.sh        # prints PID; worker log → data/07_logs/step7_run.log
```

- The launch call must return **immediately** with the PID. If it does not return
  within a few seconds, the detach failed — stop and fix before monitoring.
- Confirm the process is alive with `ps -p $(cat data/07_logs/step7.pid)`.

### Monitoring loop (relay progress to the chat)

⚠️ **Never run a `while ... sleep 20` loop inside a single shell call**: the call
returns its output only when it ends (and it will be killed at the tool timeout),
so nothing is streamed to the user in real time. Instead, use **short, separate
poll calls** and relay **only the new lines** since the previous poll:

1. Record the current byte size of the log right after launch:
   ```bash
   stat -c %s data/07_logs/step7_run.log   # save as OFFSET
   ```
2. Every ~20 seconds, run one short poll call (keep it well under the tool timeout):
   ```bash
   sleep 20
   tail -c +<OFFSET> data/07_logs/step7_run.log | sed -n '/\[STEP7\]/p'
   stat -c %s data/07_logs/step7_run.log   # update OFFSET with this value
   ```
3. **Relay every new `[STEP7]` line to the specialist in the chat**, e.g.:
   - Vector start (CRITICAL — always relay immediately):
     `--- Processing Vector: challenges (Scope: sentence) ---`
   - Progress per vector: `🔄 challenges: 348/3487 (10%)`
   - Vector completion: `✓ Finished challenges. Kept 3166 valid insights.`
4. Repeat until the process exits. Then verify the PID is gone:
   ```bash
   ps -p $(cat data/07_logs/step7.pid) || echo "process exited"
   ```

When the process exits:
- Exit code 0 (check `pipeline_state.json` step 7 `status == "completed"`) → proceed
  to the Report result section.
- Non-zero → read `pipeline_state.json` step 7 `notes` and report the failure.

### Resume after interruption (long runs)

If the worker is killed mid-run (tool timeout, crash, machine reboot), **rerun the
exact same launch command**: the worker loads the per-vector checkpoints in
`data/02_intermediate/step7_checkpoints/`, skips the vectors already extracted and
only processes the remaining ones, then rebuilds the master outputs.

- To **force a full re-extraction** (e.g., changed prompts or vectors), delete the
  checkpoint files first:
  ```bash
  rm -rf data/02_intermediate/step7_checkpoints
  ```

---

## Report result

Read `pipeline_state.json` to present the final stats:

```text
✓ Step 7 completed — Semantic Extraction

Extraction Summary:
[Iterate over params_used.vector_counts dynamically to show counts per vector. e.g., "Challenges: 3200", "Directions: 1500"]

Outputs generated:
- Relational Tidy Data Spreadsheet: data/05_exports/step7_extractions.xlsx
- Intermediate Pickle: data/02_intermediate/step7_extractions.pkl
- Per-vector checkpoints (for resume): data/02_intermediate/step7_checkpoints/
```

Inform the user:
> "The information was consolidated successfully. You can open the `step7_extractions.xlsx` spreadsheet to audit the exact relationship between each original sentence, the generated micro-label, and the assigned Macro-Category."
> "We can proceed to **Step 8 (Specialized Synthesis)** whenever you wish!"
