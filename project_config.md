# Project Configuration — AKL Pipeline

> This file is read by the orchestrator agent at the beginning of each session.
> Fill in the sections marked with `[FILL]` before starting the pipeline.
> Sections marked with `[OPTIONAL]` can be left blank — the pipeline will use default values.
> Do not remove empty sections: they may be necessary in future steps.

---

## 1. Project identification

- **Project name:** [FILL — e.g.: "Green Hydrogen 2024"]
- **Creation date:** [FILL — e.g.: "2025-07-01"]
- **Responsible:** [FILL — expert's name]
- **Advisor:** [OPTIONAL]
- **Institution:** [OPTIONAL]

---

## 2. Research objective

> Describe in free text the central objective of the bibliometric research.
> This description will be used by the agent in Steps 4 and 8 to generate contextualized suggestions.

[FILL — e.g.: "Map the state of the art in alkaline electrolysis for green hydrogen production, identifying technological challenges, emerging research directions, and knowledge gaps in the 2015-2024 period."]

---

## 3. Domain and thematic area

- **Main area:** [FILL — e.g.: "Electrochemistry / Clean energy"]
- **Relevant subareas:** [FILL — e.g.: "HER, OER, alkaline electrolysis, PEM, fuel cells"]
- **Period of interest:** [FILL — e.g.: "2015–2024"]
- **Corpus languages:** [FILL — e.g.: "English"]

---

## 4. Research questions

> List the questions that the bibliometric analysis must answer.
> They will be used by the agent in Step 8 to guide the synthesis of insights.

1. [FILL]
2. [FILL]
3. [FILL]

---

## 5. Data source (Step 1)

- **Chosen source:** [FILL — "WOS (upload)", "OpenAlex (query)", "PubMed (query)", or "Generic CSV (upload)"]
- **Upload filename:** [FILL if WOS or Generic CSV — e.g.: "wos_export.txt" or "scopus_export.csv"]
- **Search query:** [FILL if OpenAlex or PubMed — e.g.: "alkaline electrolysis hydrogen production"]
- **PubMed API key:** [OPTIONAL — obtain at https://www.ncbi.nlm.nih.gov/account/ — raises rate limit from 3 to 10 req/s]
- **Filters applied at the source:** [OPTIONAL — e.g.: "document type: Article; language: English"]
- **Max results to fetch:** [OPTIONAL — default: 10000]
- **Estimated total records:** [OPTIONAL]

> **Note for Generic CSV users:** After placing your CSV in `data/01_raw/`, edit the `source_mapping.generic` section in `pipeline_config.yaml` to map your CSV column names to the AKL canonical schema. See `generic_csv_import_guide.md` for detailed instructions.

---

## 6. Relevance criteria (Step 4)

> Describe what makes a document relevant to this study.
> The agent will use these criteria to suggest the classification of the sample.

### RELEVANT document when:
- [FILL — e.g.: "Directly addresses alkaline or PEM electrolysis for H₂ production"]
- [FILL — e.g.: "Addresses electrode materials, catalysts, or process energy efficiency"]
- [OPTIONAL]

### NOT RELEVANT document when:
- [FILL — e.g.: "Only addresses fuel cells as an H₂ consumer, without covering production"]
- [FILL — e.g.: "H₂ appears only as a byproduct or secondary reactant"]
- [OPTIONAL]

### Examples of relevant titles (optional but recommended):
- [OPTIONAL]
- [OPTIONAL]
- [OPTIONAL]

### Examples of not relevant titles (optional but recommended):
- [OPTIONAL]
- [OPTIONAL]

---

## 7. Semantic extraction vectors (Step 7)

> Define which semantic dimensions should be extracted from the texts.
> The 'Scope' indicates if the extraction happens sentence by sentence (more granular) or scanning the entire abstract (for unique overall identifiers).
> Leave the "Active" field as "yes" or "no" to control which ones will be executed.

| Vector | Description | Scope | Active |
|---|---|---|---|
| challenges | Challenges, bottlenecks, and problems identified | sentence | yes |
| directions | Future research directions or proposed solutions | sentence | yes |
| controversies | Controversies or disagreements between authors | sentence | yes |
| main_material | The primary material, compound, or system studied | abstract | no |

**Specific domain instruction for materials extraction:** [OPTIONAL — e.g.: "Focus only on electrodes and catalysts for electrolysis"]

---

## 8. Questions and Instructions for Step 8 (Specialized Synthesis)

> The Agent will automatically generate a Deep Scientometric Analysis containing:  
> 1. Thematic Lifecycle (Exponential vs Plateau topics)  
> 2. Strategic Geopolitics (Pioneers vs Traditionalists)  
> 3. Bottleneck Regimes (Row-normalized Challenges by Topic)  
> 4. Impact Quadrants (Volume vs Efficiency)  
>
> List below any additional specific questions the agent should answer based on the data.

**Analytical Questions & Focus Areas:**
1. [FILL — e.g.: "Discuss how the Exponential topics relate to the Pioneers."]
2. [FILL — e.g.: "What are the most frequent challenges for the highest-impact topics?"]
3. [OPTIONAL]

---

## 9. Expected visualizations (Step 9)

> Indicate which visualizations are a priority for the final report.
> The standard set will always be generated. List additional desired visualizations here.

**Standard set (always generated):**
- Topographic topic map
- Temporal evolution of clusters
- Distribution of publications per year
- Word cloud per cluster
- Term co-occurrence network

**Additional requested visualizations:**
- [OPTIONAL]
- [OPTIONAL]

---

## 10. Notes and free observations

> Space for the expert to record decisions, additional context, or relevant observations
> that do not fit in the sections above. The agent reads this section
> but does not process it in a structured way — it serves as context memory.

[OPTIONAL]

---

## Essential data fields

The following fields are required for every record: doi.

---

## 11. Pipeline configuration overrides

> Describe in natural language any pipeline configuration adjustments you want to apply.
> The automation will translate these requests into `pipeline_config.yaml` values.
> To see all available options, open `pipeline_config.yaml` — it contains the technical
> defaults for every step. You only need to write here what you want to change.
> Examples: "audit 100 challenge sentences", "use the gemma-3-4b model via vLLM",
> "increase batch_size to 32".

[FILL — e.g.: "Audit 100 challenge sentences in Step 8."]
[OPTIONAL — e.g.: "Use the gemma-3-4b-it model served by the local vLLM server."]
[OPTIONAL — e.g.: "Increase the Step 7 batch size to 32."]

---

## Change history

| Date | Changed section | Description |
|---|---|---|
| [date] | [section] | Initial file creation |