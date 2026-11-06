# AKL (Automated Knowledge Landscapes)

AKL is a nine-step, human-in-the-loop pipeline that turns a large bibliographic corpus into a navigable Knowledge Landscape: a relevance-filtered corpus, a validated topic map, and structured extractions (for example, research challenges and directions) from sentences and abstracts.

The pipeline is operated through an LLM coding agent. The workflow is described in plain files (`AGENTS.md` and one skill file per step in `.opencode/skills/`), so any coding agent that can read these files and run shell commands can act as orchestrator, with a cloud model or a locally served model as backend. The proof of concept reported in the paper used Claude Code with Claude Sonnet 5.5, and the skills are also read natively by OpenWork/OpenCode.

The pipeline pauses at two expert breakpoints (Steps 4 and 8) and at Step 6 when the validity indices do not agree on a layer.

Version: [TODO: release tag]

---

## Pipeline

| Step | Name | Type |
|---|---|---|
| 1 | Data acquisition (Web of Science upload, OpenAlex, PubMed, generic CSV) | Automated |
| 2 | Cleaning and structuring | Automated |
| 3 | Preliminary topic map (BERTopic with EVoC) | Automated |
| 4 | Expert curation of relevance labels | **Breakpoint** |
| 5 | Relevance filter (SetFit) | Automated |
| 6 | Final topic map, layer selection with two Combined Validation Indices and Kneedle | Automated, **breakpoint** without consensus |
| 7 | Semantic extraction with a local LLM, optional sentence verifier | Automated |
| 8 | Expert study and synthesis | **Breakpoint** |
| 9 | Report and figures | Automated |

See `manual_akl.md` for the operation of each step.

---

## Prerequisites

1. **Miniconda or Anaconda** (Windows, Mac, or Linux).
2. **An OpenAI-compatible LLM server** for Steps 3, 4, 6, 7, and 8 (topic naming, pre-labeling, extraction, verification). The pipeline was developed with [vLLM](https://docs.vllm.ai/) serving quantized Gemma 3 models on consumer GPUs. The server can run on the same machine or on a remote one reachable over the network.
3. **Optionally, an OpenAI-compatible embedding server** for Steps 3 and 6. Without it, the embeddings are computed locally with sentence-transformers.
4. **An LLM coding agent** to orchestrate the pipeline (see above).

---

## Installation

```bash
git clone https://github.com/lcazevedo/AKL_Pipeline_Automation.git
cd AKL_Pipeline_Automation
```

The pipeline uses two isolated environments, to avoid dependency conflicts with the SetFit training of Step 5.

Main environment:

```bash
conda create -n akl_automation python=3.11 -y
conda activate akl_automation
pip install -r akl_automation_requirements.txt
```

SetFit environment (Step 5):

```bash
conda create -n akl_setfit python=3.11 -y
conda activate akl_setfit
pip install -r akl_setfit_requirements.txt
```

Edit `CONDA_ENV` and `CONDA_BASE` at the top of `run_worker.sh` and `run_worker_setfit.sh` if your environment names or conda path differ.

---

## Configuration

1. Copy the templates to the workspace root:
   ```bash
   cp base_project_config.md project_config.md
   cp base_pipeline_config.yaml pipeline_config.yaml
   cp base_pipeline_state.json pipeline_state.json
   ```
2. Fill in `project_config.md` (sections marked `[FILL]`): research objective, data source and query (Section 5), relevance criteria (Section 6), extraction vectors (Section 7), and questions for the synthesis (Section 8).
3. In `pipeline_config.yaml`, set at least:
   - `models.llm_extraction` and `step7_vllm.base_url` (or `step7_vllm.base_urls` to split the load across several servers);
   - `embedding_server.enabled` and `embedding_server.base_url`, if you use an embedding server;
   - `models.llm_agent`, to record the model used as agent (no worker reads this field).

For a Web of Science source, place the export file in `data/01_raw/`.

---

## Starting the LLM server

Example with vLLM, on the machine with the GPU:

```bash
python -m vllm.entrypoints.openai.api_server --model RedHatAI/gemma-3-12b-it-quantized.w4a16 --max-model-len 4096 --enforce-eager
```

The proof of concept used `RedHatAI/gemma-3-12b-it-quantized.w4a16` with a maximum model length of 4,096 tokens on GPUs with 12 GB of VRAM. Smaller GPUs can use `RedHatAI/gemma-3-4b-it-quantized.w8a8`. Set `models.llm_extraction` to the served model, and keep `step7_extraction.max_abstract_words` compatible with the maximum model length.

---

## Running

Open your coding agent in the workspace root and ask it to read `AGENTS.md`. Then interact in natural language:

1. "What is the current status of the pipeline?"
2. "Run step 1."
3. At the breakpoints:
   - **Step 4:** label a sample of documents as "yes" or "no" in the review spreadsheet. If the sample has few not-relevant documents, the tools `tools/expand_negatives.py` and `tools/expand_round2.py` propose further candidates for labeling.
   - **Step 6:** when the validity indices do not agree, choose the layer and record the reason.
   - **Step 8:** review and correct the synthesis draft `step8_insights_review.md`.
4. Step 9 compiles the figures and the report `AKL_Final_Report.html`.

Long steps write their progress to `execution.log` in the workspace root.

### Step 7 options

- **Pilot:** `./run_worker.sh workers/step7_worker.py pilot 300` runs the extraction on 300 documents and reports the share of positive labels per vector and an estimate of the time for the whole corpus. Run it before the full extraction.
- **Sentence verifier:** `step7_extraction.sentence_verifier` asks the LLM a yes or no question for each sentence-level label and keeps the label when P(yes) reaches the threshold of the vector. It raises precision at the cost of recall, and its thresholds should be set on an audited sample. The scores are stored, so the thresholds can be changed without new calls to the model.
- **Keyword pre-filter:** `step7_extraction.sentence_prefilter` drops sentences without cue words before the LLM. It is cheaper but loses recall, so measure it with the pilot.
- **Extraction cache:** raw answers are cached by document and text. `step7_extraction.reuse_extractions_from` points to an existing cache, so that only new texts are sent to the model.
- **Re-clustering:** `./run_worker.sh workers/step7_worker.py recluster` regroups and renames the categories without extracting again.

### Reproducibility

Steps 3 and 6 use EVoC with a fixed seed (`evoc_random_state`). With the same embeddings and seed, the partitions are identical. A different seed changes the number and size of the layers, so the seed is part of the configuration that must be reported. The tool `tools/seed_test.py` refits EVoC on the Step 6 embeddings with other seeds and compares the saved layers with a refit using the configured seed.

---

## Current limitations

- The pipeline is under development and changes often.
- The LLM steps require an OpenAI-compatible endpoint. Authenticated commercial APIs are not supported for the extraction.
- Some steps can look frozen in the agent chat. Check `execution.log` to see whether a step is still running.

## License

MIT License (see `LICENSE`).

## Citation

> Azevedo, L. C.; Prati, R. C. *Automated Knowledge Landscapes: A Human-in-the-Loop Semantic Bibliometric Roadmap for Strategic Discovery.* [TODO: journal, year, DOI]

The data and results of the proof of concept are in [AKL_PoC](https://github.com/lcazevedo/AKL_PoC).
