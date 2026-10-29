
---
name: config_sync
description: Synchronizes natural-language configuration overrides from Section 11 of project_config.md into pipeline_config.yaml. Run this skill at the start of every session, before any pipeline step, to apply the specialist's requested adjustments (e.g., "audit 100 challenge sentences", "use the gemma-3-4b model via vLLM", "increase batch_size to 32"). Only modifies keys that already exist in pipeline_config.yaml; ambiguous requests are confirmed with the specialist before applying.
---

# AKL Pipeline — Config Sync Skill

## Your task

Read Section 11 ("Pipeline configuration overrides") of `project_config.md`, interpret each natural-language request, map it to an existing key in `pipeline_config.yaml`, and apply the change — always confirming with the specialist before writing. This skill is the bridge between the specialist's plain-language intent and the technical configuration file.

## Preconditions

1. `project_config.md` exists in the workspace root. If not, stop and inform the specialist.
2. `pipeline_config.yaml` exists in the workspace root. If not, stop and inform the specialist.

---

## Steps to follow

### 1. Read Section 11 of project_config.md

Read `project_config.md` and locate the section titled `## 11. Pipeline configuration overrides`.

- If the section is missing, inform the specialist that Section 11 was not found and suggest updating `project_config.md` (or `base_project_config.md` for new projects).
- If the section contains only `[OPTIONAL — no overrides requested yet]` or is effectively empty, report:
  > No configuration overrides found in Section 11. Using `pipeline_config.yaml` defaults as-is.
  Then finish — do not modify anything.

### 2. Read the current pipeline_config.yaml

Read `pipeline_config.yaml` in full. You need the exact current values and the exact set of existing keys. **Never invent keys** — every override must map to a key that already exists in this file.

### 3. Interpret each override request

For each line/item in Section 11, interpret the natural-language request and map it to a target key path in `pipeline_config.yaml`.

Use this reference of the main editable keys (always verify against the actual file — the file is the source of truth):

| Key path | Meaning | Typical natural-language trigger |
|---|---|---|
| `models.embedding` | Embedding model name | "use model X for embeddings" |
| `models.llm_extraction` | LLM used for semantic extraction (vLLM) | "use model X via vLLM", "switch extraction model" |
| `models.llm_agent` | Name of the model that acts as the agent (naming topics, pre-labeling, synthesis). Ask the user which model is selected in OpenWork when the project is created, and record it here exactly as shown | "the agent model is X" |
| `models.setfit_base` | Base model for SetFit (Step 5) | "use model X for SetFit" |
| `step1_openalex.filters` | OpenAlex query filters (Step 1) | "restrict OpenAlex to 2020+", "add filter X" |
| `step2_cleaning.detect_language` | Auto-detect language (Step 2) | "enable/disable language detection" |
| `step2_cleaning.language_filter` | Keep only this language (Step 2) | "keep only English", "remove language filter" |
| `step2_cleaning.year_min` / `year_max` | Analysis window (Step 2) | "analyze 2020–2025", "window from 2018" |
| `step3_bertopic.min_topic_size` | Min docs per topic (Step 3) | "smaller topics", "min topic size 5" |
| `step3_bertopic.nr_topics` | Number of topics (Step 3) | "force 10 topics" |
| `step3_bertopic.embedding_device` | CPU vs CUDA for embeddings (Step 3) | "use CPU for embeddings" |
| `step3_bertopic.min_df` | CountVectorizer min_df (Step 3) | "min_df 1" |
| `step3_bertopic.llm_nr_docs` | Docs per topic sent to LLM (Step 3) | "send 3 docs per topic to the LLM" |
| `reproducibility.seed` | Global random seed used by Steps 3-7 | "use seed 123", "change the random seed" |
| `step3_bertopic.max_layers` / `min_similarity_threshold` | Number/merging of topic-granularity layers (Step 3) | "fewer granularity layers" |
| `step5_setfit.num_epochs` | SetFit epochs (Step 5) | "train for 3 epochs" |
| `step5_setfit.min_samples_per_class` | Min samples per class warning (Step 5) | "warn below 10 samples per class" |
| `step5_setfit.batch_size` | SetFit batch size (Step 5) | "batch size 8" |
| `step5_setfit.group_by_topic` | GroupShuffleSplit by Step-3 topic (Step 5) | "disable topic grouping" |
| `step5_setfit.retention_check` | Retention check vs Step-4 labels (Step 5) | "skip the retention check" |
| `step5_setfit.save_proba_yes` | Save P(yes) column (Step 5) | "don't save probabilities" |
| `step6_evoc.min_cluster_size` | EVoC min cluster size (Step 6) | "min cluster size 3" |
| `step6_evoc.metric` | EVoC distance metric (Step 6) | "use euclidean metric" |
| `step6_evoc.min_df` | EVoC min_df (Step 6) | "min_df 1" |
| `step6_evoc.max_layers` / `min_similarity_threshold` | Number/merging of topic-granularity layers (Step 6) | "fewer granularity layers" |
| `step7_extraction.chunk_unit` | Sentence vs abstract extraction (Step 7) | "extract per abstract", "sentence level" |
| `step7_extraction.vectors` | Semantic vectors list (Step 7) | "add vector X", "remove controversies" |
| `step7_vllm.base_url` | vLLM server URL (Step 7) | "point to localhost", "use PC1 server" |
| `step7_vllm.max_tokens` | Max tokens per LLM call (Step 7) | "max tokens 1024" |
| `step7_vllm.temperature` | LLM temperature (Step 7) | "temperature 0.2" |
| `step7_vllm.batch_size` | Extraction batch size (Step 7) | "batch size 64" |
| `step7_vllm.server.*` | vLLM server launch params (Step 7) | "port 8001", "max model len 4096" |
| `step9_output.figures_format` | Figure format (Step 9) | "SVG figures" |
| `step9_output.figures_dpi` | Figure DPI (Step 9) | "300 dpi" |
| `step9_output.include_standard` | Always generate standard figures (Step 9) | "skip standard figures" |
| `audits.<name>.enabled` | Enable/disable an audit (Step 8) | "enable the directions audit" |
| `audits.<name>.sample_size` | Number of items to sample for an audit (Step 8) | "audit 100 challenge sentences", "sample 50 controversies" |
| `audits.<name>.filter_vector` | Which semantic vector an audit targets (Step 8) | "audit the challenges vector" |
| `audits.<name>.filter_range` | Numeric range filter for an audit (Step 8) | "audit documents with P(yes) between 0.40 and 0.60" |
| `audits.<name>.predicted_col` | Column holding the predicted label (Step 8) | "audit macro categories instead of micro labels" |
| `audits.<name>.columns` | Columns exported to the review file (Step 8) | "include title and abstract in the boundary audit" |
| `audits.<name>.source` | Data source for an audit (Step 8) | "sample from the full predictions table" |

If the specialist changes the seed, a model, the embedding device or the temperature after some steps have already run, tell them those steps must be rerun for the change to apply.

Future sections (e.g., additional `audits.*` entries) will be added to this table as they are introduced into `pipeline_config.yaml`. Always re-read the file to discover new keys.

### 4. Present the proposed changes and confirm

Before writing anything, present a clear summary to the specialist:

```
Configuration overrides detected in Section 11:

1. "audit 100 challenge sentences"
   → audits.challenges.sample_size: (not present yet) → 100
   ⚠️ This key does not exist in pipeline_config.yaml yet.
      [If the audit framework is not implemented, say so and ask how to proceed.]

2. "use the gemma-3-4b model via vLLM"
   → models.llm_extraction: "RedHatAI/gemma-3-4b-it-quantized.w8a8" → "gemma-3-4b"
   ⚠️ Ambiguous: which exact model identifier? Confirm the full HuggingFace ID.

Shall I apply these changes?
```

Rules for this step:
- **Ambiguous request** → ask a targeted question before applying. Never guess a model ID, a number, or a filter value.
- **Key does not exist** → do NOT create it. Inform the specialist the key is not available in this version and ask how to proceed (e.g., "the audit framework is not implemented yet — it will be available in a future phase").
- **Conflicting requests** (two lines contradict each other) → surface the conflict and ask which one wins.
- **Out-of-range values** → warn (e.g., temperature must be 0–1, batch_size must be a positive integer).

### 5. Apply the changes

Only after the specialist confirms:

1. Make a timestamped backup of the current file:
   ```bash
   cp pipeline_config.yaml pipeline_config.yaml.bak.<YYYYMMDD-HHMMSS>
   ```
2. Edit `pipeline_config.yaml` — change **only** the confirmed keys. Preserve all comments and the rest of the file exactly as they are.
3. Do NOT touch `pipeline_state.json` — configuration changes do not alter pipeline state.

### 6. Report the result

Report what changed, with before → after values:

```
✓ Configuration applied to pipeline_config.yaml

| Key | Before | After |
|---|---|---|
| models.llm_extraction | RedHatAI/gemma-3-4b-it-quantized.w8a8 | gemma-3-4b-it |

Backup saved as pipeline_config.yaml.bak.<timestamp>.

Note: changes take effect the next time the affected step runs.
```

---

## Important rules

- **Never invent keys.** Every override must map to a key that already exists in `pipeline_config.yaml`. If the specialist asks for something that has no key, explain that the feature is not available and suggest the closest existing option.
- **Never guess values.** Ambiguous requests are always confirmed with the specialist first.
- **Preserve the file.** Only the confirmed keys change; comments, ordering, and unrelated sections stay untouched.
- **Backup before writing.** Always create `pipeline_config.yaml.bak.<timestamp>` first.
- **Do not modify `project_config.md`** during this skill — the specialist owns that file. (The only exception: if the specialist explicitly asks you to record the applied override in the Change history table.)
- **Do not modify `pipeline_state.json`.**
- Respond in the same language the specialist is using.

---

## When to run

- At the **start of every session**, before any pipeline step (the `status` skill references this skill).
- Whenever the specialist says something like "change the config", "use model X", "audit N items", or edits Section 11 mid-session.


