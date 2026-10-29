---
name: step1_acquisition
description: Executes Step 1 of the AKL pipeline — Raw Data Acquisition. Loads bibliometric data from a WOS export file, from the OpenAlex API, from the PubMed API, from the arXiv API, or from a generic CSV upload, normalizes it to the canonical schema, and saves corpus_raw.pkl to data/inputs/. Updates pipeline_state.json on completion.
---

# AKL Pipeline — Step 1: Raw Data Acquisition

## Preconditions — check before doing anything else

1. Read `pipeline_state.json`.
   - If step 1 `status` is `completed`, inform the specialist and ask explicitly if they want to **rerun** it (this will overwrite `corpus_raw.pkl`). Only proceed if they confirm.
   - If any step beyond step 1 is `completed`, warn that rerunning step 1 will require rerunning all subsequent steps as well.
   - If status is `running`, warn that a previous execution may have been interrupted and ask if they want to restart.

2. Read `project_config.md`.
   - Identify the `source_type` field (section 5): `"WOS (upload)"`, `"OpenAlex (query)"`, `"PubMed (query)"`, `"arXiv (query)"`, or `"Generic CSV (upload)"`.
   - If the field is empty or not filled, ask the specialist to fill in section 5 of `project_config.md` before continuing.

3. Confirm that `run_worker.sh` exists in the workspace root. If not found, stop and inform:
   > `run_worker.sh` not found in the workspace root. This file is required to run workers in the correct Python environment. Please restore it from the project repository.

---

## Branch A — WOS upload

Trigger: `source_type` contains "WOS" or the specialist provides a filename.

### Steps

1. Identify the input file:
   - Check `project_config.md` section 5 for `upload_filename`.
   - If not specified, ask: *"What is the name of the WOS export file? It should be in the `data/inputs/` folder."*
   - Confirm the file exists at `data/inputs/<filename>`. If not found, inform the specialist and stop.

2. Run the worker via the shell wrapper:
   ```bash
   cd <workspace_root>
   ./run_worker.sh workers/step1_worker.py wos \
       --input data/01_raw/<filename> \
       --config pipeline_config.yaml \
       --state pipeline_state.json
   ```

3. Capture stdout and stderr. Report the result to the specialist:
   ```
   ✓ Step 1 completed — WOS acquisition

   Records loaded:     [records_raw]
   Records output:     [records_output]
   Missing abstracts:  [records_missing_abstract]
   Output file:        data/02_intermediate/step1_corpus_raw.pkl

   [If records_missing_abstract > 0:]
   ⚠️  [N] records have no abstract. These may be filtered out in Step 5.
   Consider reviewing corpus_raw.csv before proceeding.

   Ready for Step 2 — Data Cleaning.
   ```

---

## Branch B — OpenAlex API

Trigger: `source_type` contains "OpenAlex" or the specialist requests an API query.

### Steps

1. Identify the query:
   - Check `project_config.md` section 5 for `query`.
   - If empty, ask: *"What is the search query for OpenAlex?"*

2. Identify optional parameters:
   - `max_results`: check project_config.md section 5 for the field "Max results to fetch". Use that value if present. If the field is absent or empty, default to 10000 and inform the specialist: "No max_results defined in project_config.md — defaulting to 10000." Do NOT silently use 10000 if the specialist has set a value.
   - `email`: check `project_config.md`. If not provided, inform that an email enables the OpenAlex polite pool (higher rate limit) but is optional.

3. Warn before starting and ask for confirmation:
   ```
   Ready to query OpenAlex:
     Query:       "<query>"
     Max results: <max_results>
     Email:       <email or "not provided">

   This may take several minutes. Shall I proceed?
   ```

4. Only proceed after explicit confirmation. Then run:
   ```bash
   cd <workspace_root>
   ./run_worker.sh workers/step1_worker.py openalex \
       --query "<query>" \
       --max-results <max_results> \
       --email "<email>" \
       --config pipeline_config.yaml \
       --state pipeline_state.json
   ```
   Omit `--email` if not provided.

5. Report the result:
   ```
   ✓ Step 1 completed — OpenAlex acquisition

   Records fetched:    [records_raw]
   Records output:     [records_output]
   Missing abstracts:  [records_missing_abstract]
   Output file:        data/02_intermediate/step1_corpus_raw.pkl

   [If records_missing_abstract > 0:]
   ⚠️  [N] records have no abstract. These may be filtered out in Step 5.

   Ready for Step 2 — Data Cleaning.
   ```

---

## Branch C — PubMed API

Trigger: `source_type` contains "PubMed" or the specialist requests a PubMed query.

### Steps

1. Identify the query:
   - Check `project_config.md` section 5 for `query`.
   - If empty, ask: *"What is the search query for PubMed?"*

2. Identify optional parameters:
   - `max_results`: check project_config.md section 5 for "Max results to fetch". Default to 10000 if not set.
   - `api_key`: check project_config.md section 5 for "PubMed API key". If not provided, inform that without an API key the rate limit is 3 requests/second (vs 10 with key). The key is free at https://www.ncbi.nlm.nih.gov/account/.
   - `email`: check project_config.md. Optional, enables NCBI polite pool.

3. Warn before starting and ask for confirmation:
   ```
   Ready to query PubMed:
     Query:       "<query>"
     Max results: <max_results>
     API key:     <provided or "not provided — rate limit: 3 req/s">
     Email:       <email or "not provided">

   This may take several minutes. Shall I proceed?
   ```

4. Only proceed after explicit confirmation. Then run:
   ```bash
   cd <workspace_root>
   ./run_worker.sh workers/step1_worker.py pubmed \
       --query "<query>" \
       --max-results <max_results> \
       --api-key "<api_key>" \
       --email "<email>" \
       --config pipeline_config.yaml \
       --state pipeline_state.json
   ```
   Omit `--api-key` and `--email` if not provided.

5. Report the result:
   ```
   ✓ Step 1 completed — PubMed acquisition

   Records fetched:    [records_raw]
   Records output:     [records_output]
   Missing abstracts:  [records_missing_abstract]
   Output file:        data/02_intermediate/step1_corpus_raw.pkl

   [If records_missing_abstract > 0:]
   ⚠️  [N] records have no abstract. These may be filtered out in Step 5.

   Ready for Step 2 — Data Cleaning.
   ```

---

## Branch D — Generic CSV upload

Trigger: `source_type` contains "Generic CSV" or the specialist provides a non-WOS CSV file.

### Preconditions

1. The specialist must have placed the CSV file in `data/01_raw/`.
2. The `source_mapping.generic` section in `pipeline_config.yaml` must be configured with the correct column mappings. If not, stop and refer the specialist to `generic_csv_import_guide.md`.

### Steps

1. Identify the input file:
   - Check `project_config.md` section 5 for `upload_filename`.
   - If not specified, ask: *"What is the name of the CSV file? It should be in the `data/01_raw/` folder."*
   - Confirm the file exists at `data/01_raw/<filename>`. If not found, inform the specialist and stop.

2. Verify the mapping configuration:
   - Read `pipeline_config.yaml` → `source_mapping.generic`.
   - Check that `format` is set to either `"wide"` or `"long"`.
   - Check that the essential fields (`doi`, `title`, `abstract`, `year`) have valid column names mapped.
   - If the mapping looks unconfigured (all defaults), warn the specialist and suggest reviewing `generic_csv_import_guide.md`.

3. Run the worker via the shell wrapper:
   ```bash
   cd <workspace_root>
   ./run_worker.sh workers/step1_worker.py generic \
       --input data/01_raw/<filename> \
       --mapping generic \
       --config pipeline_config.yaml \
       --state pipeline_state.json
   ```

4. Capture stdout and stderr. Report the result to the specialist:
   ```
   ✓ Step 1 completed — Generic CSV acquisition

   Source file:        data/01_raw/<filename>
   Format:             <wide or long>
   Records loaded:     [records_raw]
   Records output:     [records_output]
   Missing abstracts:  [records_missing_abstract]
   Output file:        data/02_intermediate/step1_corpus_raw.pkl

   [If records_missing_abstract > 0:]
   ⚠️  [N] records have no abstract. These may be filtered out in Step 5.

   Ready for Step 2 — Data Cleaning.
   ```

---

## Branch E — arXiv API

Trigger: `source_type` contains "arXiv" or the specialist requests an arXiv query.

### Steps

1. Identify the query:
   - Check `project_config.md` section 5 for `query`.
   - If empty, ask: *"What is the search query for arXiv?"*
   - arXiv queries use the Atom API syntax, e.g.:
     `(all:"computer science education" OR all:"software engineering education" OR all:"programming education") AND (all:"generative AI" OR all:"large language models" OR all:ChatGPT OR all:Copilot) AND (cat:cs.SE OR cat:cs.CY OR cat:cs.HC OR cat:cs.PL OR cat:cs.AI)`

2. Identify optional parameters:
   - `max_results`: check project_config.md section 5 for "Max results to fetch". Default to 10000 if not set.
   - `email`: check `project_config.md`. Optional; appended to the User-Agent (arXiv recommends including it).

3. Warn before starting and ask for confirmation:
   ```
   Ready to query arXiv:
     Query:       "<query>"
     Max results: <max_results>
     Email:       <email or "not provided">

   This may take several minutes (rate limit: 1 request per 3 s).
   Shall I proceed?
   ```

4. Only proceed after explicit confirmation. Then run:
   ```bash
   cd <workspace_root>
   ./run_worker.sh workers/step1_worker.py arxiv \
       --query "<query>" \
       --max-results <max_results> \
       --email "<email>" \
       --output-stem corpus_raw_arxiv \
       --config pipeline_config.yaml \
       --state pipeline_state.json
   ```
   Omit `--email` if not provided. `--output-stem corpus_raw_arxiv` keeps the
   per-source pickle separate so it can be merged later.

5. Report the result:
   ```
   ✓ Step 1 completed — arXiv acquisition

   Records fetched:    [records_raw]
   Records output:     [records_output]
   Missing abstracts:  [records_missing_abstract]
   Output file:        data/02_intermediate/step1_corpus_raw_arxiv.pkl

   [If records_missing_abstract > 0:]
   ⚠️  [N] records have no abstract. These may be filtered out in Step 5.

   Ready for Step 2 — Data Cleaning.
   ```

### Notes on arXiv data

- Preprints without a published DOI receive the DataCite fallback DOI
  `10.48550/arXiv.<id>` so the `doi` field is never blank.
- Records that carry a published DOI (`ar:doi`) or a journal reference
  (`ar:journal_ref`) will deduplicate against WOS/OpenAlex during the merge.
- arXiv does not provide citation counts — `citation_count` is set to 0.
- `external_id` is filled with the arXiv ID (version suffix stripped).

---

## Error handling

If the worker exits with a non-zero return code or prints to stderr:

1. Show the error output clearly.
2. Read `pipeline_state.json` — step 1 status should be `failed` with a note.
3. Report the error and suggest checking the execution log for details:
   > Check `execution.log` for the full error trace. Common issues include: missing input files, configuration errors, dependency issues, or connection problems.
4. Ask the specialist how they want to proceed.

---

## After successful completion

- Confirm that `pipeline_state.json` shows step 1 as `completed`.
- Confirm that both data/02_intermediate/step1_corpus_raw.pkl and `data/01_raw/step1_corpus_raw.csv` exist.
- Suggest running the `status` skill to see the full pipeline overview.
- Do NOT automatically start Step 2 — wait for the specialist's instruction.
