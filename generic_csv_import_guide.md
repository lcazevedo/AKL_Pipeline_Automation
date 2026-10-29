# AKL Pipeline — Generic CSV Import Guide

> This guide explains how to import bibliometric data from **any CSV file** into the AKL pipeline.
> It is intended for users who have data from sources other than WOS, OpenAlex, or PubMed
> (e.g., Scopus, Dimensions, Lens.org, manual spreadsheets, institutional databases).

---

## 1. Overview

The AKL pipeline works with an internal **canonical schema** — a fixed set of column names
used by all pipeline steps (2–9). Step 1 is the only step that converts external data into
this schema.

For generic CSV files, you provide a **column mapping** in `pipeline_config.yaml` that tells
the pipeline which column in your CSV corresponds to each canonical field. The pipeline then
handles everything else automatically: author list splitting, first/last author derivation,
country → region conversion, impact factor lookup, and citation-per-year calculation.

---

## 2. Two Supported CSV Formats

Your CSV can be in one of two formats. Set the `format` field in the mapping accordingly.

### 2.1 Wide format (one row per document)

Each row is one document. Author lists are **semicolon-separated** in a single cell.

```csv
DOI,Title,Abstract,Year,Authors,Affiliations
10.1001/abc,Title X,"Abstract text...",2023,"Smith J; Lee K","MIT; Oxford"
10.1001/def,Title Y,"Abstract text...",2024,"Garcia L","USP"
```

### 2.2 Long format (one row per author)

Each row is one author of a document. The document-level fields (DOI, title, abstract, year)
are **repeated** on every row. The pipeline **automatically aggregates** rows by DOI.

```csv
DOI,Title,Abstract,Year,AuthorName,Institution,Country
10.1001/abc,Title X,"Abstract text...",2023,Smith J,MIT,United States
10.1001/abc,Title X,"Abstract text...",2023,Lee K,Oxford,United Kingdom
10.1001/abc,Title X,"Abstract text...",2023,Wang M,Tsinghua,China
10.1001/def,Title Y,"Abstract text...",2024,Garcia L,USP,Brazil
```

> **Which format should I use?**
> - **Long format** is the natural export format of Scopus and several other databases.
>   Use it when your source gives you one row per author.
> - **Wide format** is simpler when you already have the data consolidated
>   (e.g., a manual spreadsheet with authors joined by `;`).

---

## 3. Step-by-Step Instructions

### Step 1 — Place your CSV file

Copy your CSV file into the `data/01_raw/` folder of the AKL workspace:

```
data/01_raw/my_source_export.csv
```

### Step 2 — Edit the mapping in `pipeline_config.yaml`

Open `pipeline_config.yaml` and locate the `source_mapping.generic` section.
It looks like this:

```yaml
source_mapping:
  generic:
    format: "wide"          # "wide" or "long"

    # Required fields — map these to your CSV column names:
    doi: "DOI"
    title: "Title"
    abstract: "Abstract"
    year: "Year"

    # Recommended fields:
    language: "Language"
    document_type: "Document Type"
    citation_count: "Citation Count"
    source_journal: "Journal"
    keywords: "Keywords"
    research_areas: "Research Areas"

    # Author fields:
    authors_list: "Authors"
    authors_full_name_list: ""              # empty = use authors_list
    authors_institution: "Affiliations"
    authors_country: ""                     # empty = leave blank

    # Derived fields — do NOT change these:
    first_author: "_derived_"
    last_author: "_derived_"
    ...
```

**What to change:**

1. Set `format` to `"wide"` or `"long"` depending on your CSV structure.
2. Replace the values on the right side with the **exact column names** of your CSV.
   - Example: if your CSV has a column called `Publication Year`, set `year: "Publication Year"`.
3. Leave `""` (empty) for fields your CSV does not have — the pipeline will fill them with
   empty values or derive them automatically.
4. **Do not change** fields marked `_derived_` or `_computed_` — the pipeline calculates them.

### Step 3 — Fill in `project_config.md`

In `project_config.md`, section 5:

```markdown
- **Chosen source:** Generic CSV (upload)
- **Upload filename:** my_source_export.csv
```

### Step 4 — Run Step 1

Tell the agent to run Step 1, or run the worker directly:

```bash
./run_worker.sh workers/step1_worker.py generic \
    --input data/01_raw/my_source_export.csv \
    --mapping generic \
    --config pipeline_config.yaml \
    --state pipeline_state.json
```

The pipeline will:
1. Read your CSV
2. Map the columns according to your configuration
3. Aggregate rows (if `format: "long"`)
4. Derive first/last author, institutions, countries, regions
5. Calculate citations-per-year and normalized impact factor
6. Save the canonical corpus and update the pipeline state

---

## 4. Field Reference

### 4.1 Required fields

| Canonical field | Description | Notes |
|---|---|---|
| `doi` | Document DOI | Used as the grouping key in long format |
| `title` | Document title | |
| `abstract` | Document abstract | Documents without abstracts may be filtered in Step 2 |
| `year` | Publication year | Numeric, e.g., `2023` |

### 4.2 Recommended fields

| Canonical field | Description | Notes |
|---|---|---|
| `language` | Document language | e.g., `English`, `en` |
| `document_type` | e.g., `Article`, `Review` | |
| `citation_count` | Number of citations | If absent, set to 0 |
| `source_journal` | Journal name | Used for impact factor lookup |
| `keywords` | Author keywords | Semicolon-separated in wide format |
| `research_areas` | Subject categories / topics | e.g., WOS Categories, Scopus ASJC |

### 4.3 Author fields

| Canonical field | Wide format | Long format |
|---|---|---|
| `authors_list` | `"Smith J; Lee K"` | One name per row |
| `authors_full_name_list` | `"Smith John; Lee Kim"` | One full name per row |
| `authors_institution` | `"MIT; Oxford"` | One institution per row |
| `authors_country` | `"United States; United Kingdom"` | One country per row |

> **Country names:** the pipeline accepts full names (`United States`), ISO codes (`US`),
> and common abbreviations. It converts them to full names and derives the continent/region
> automatically.

### 4.4 Derived fields (do not map)

These are calculated automatically by the pipeline:

- `first_author`, `last_author`, `first_author_full_name`, `last_author_full_name`
- `first_institution`, `last_institution`, `first_country`, `last_country`
- `first_country_region`, `last_country_region`
- `journal_if` (impact factor lookup by journal name)
- `jnif` (normalized impact factor)
- `citations_per_year`

---

## 5. Common Source Templates

### 5.1 Scopus CSV export (long format)

Scopus exports come in long format. Typical columns:

```yaml
source_mapping:
  generic:
    format: "long"
    doi: "DOI"
    title: "Title"
    abstract: "Abstract"
    year: "Year"
    language: "Language of Original Document"
    document_type: "Document Type"
    citation_count: "Cited by"
    source_journal: "Source title"
    keywords: "Author Keywords"
    research_areas: "Subject area"
    authors_list: "Authors"
    authors_full_name_list: ""
    authors_institution: "Affiliations"
    authors_country: ""
```

### 5.2 Dimensions CSV export

```yaml
source_mapping:
  generic:
    format: "wide"
    doi: "DOI"
    title: "Title"
    abstract: "Abstract"
    year: "Year"
    document_type: "Type"
    citation_count: "Times Cited"
    source_journal: "Journal"
    keywords: "Keywords"
    research_areas: "Fields of Research"
    authors_list: "Authors"
    authors_full_name_list: ""
    authors_institution: ""
    authors_country: ""
```

### 5.3 Manual spreadsheet

```yaml
source_mapping:
  generic:
    format: "wide"
    doi: "DOI"
    title: "Article Title"
    abstract: "Summary"
    year: "Publication Year"
    language: ""
    document_type: ""
    citation_count: ""
    source_journal: "Journal Name"
    keywords: ""
    research_areas: ""
    authors_list: "Author Names"
    authors_full_name_list: ""
    authors_institution: ""
    authors_country: ""
```

---

## 6. Troubleshooting

| Problem | Cause | Solution |
|---|---|---|
| `CSV column 'X' not found` | Column name in mapping does not match the CSV header | Check the exact spelling in your CSV header (including spaces and case) |
| `Long format requires a DOI column` | `doi` field not mapped or column missing | Map `doi` to the correct column; ensure every row has a DOI |
| `Mapping key 'generic' not found` | `source_mapping.generic` missing from config | Restore the section from the template |
| All authors empty after import | `authors_list` column name wrong | Check the mapping; in wide format ensure values are `;`-separated |
| `UnicodeDecodeError` | CSV not in UTF-8 | The pipeline tries `latin-1` automatically; if it still fails, re-save the CSV as UTF-8 |
| Duplicate documents after import | Same DOI appears in multiple rows in wide format | Use long format, or deduplicate your CSV first |

---

## 7. What the Pipeline Does NOT Do

- It does **not** fetch missing data from external APIs (e.g., it will not look up citation
  counts for you).
- It does **not** parse author affiliations from free-text strings (e.g., a single
  `"MIT, Cambridge, USA"` cell). If your source provides separate institution and country
  columns, map them directly.
- It does **not** translate abstracts or titles.

If your source provides rich structured data (like Scopus or Dimensions), map as many
recommended fields as possible — the quality of the downstream analysis (impact factor,
geopolitics, topic mapping) depends on it.