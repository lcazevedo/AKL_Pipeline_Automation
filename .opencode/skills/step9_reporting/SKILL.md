---
name: step9_reporting
description: Executes Step 9 of the AKL pipeline. Generates Likert-scale impact charts, Hierarchical Clustermaps, and Growth curves. The agent then weaves standard and ad-hoc images into a final Markdown report, compiling it into a beautiful HTML/PDF document.
---

# AKL Pipeline — Step 9: Final Reporting & Visual Assembly

## Preconditions
1. Read `pipeline_state.json`. If Step 8 is not `completed`, stop and warn the user.

---

## Phase 1: Generate Standard Plots
Inform the user:
> "Starting Step 9. The pipeline is using Seaborn and Matplotlib to draw Hierarchical Clustermaps (Dendrograms), Likert-scale Efficiency Charts, and Growth Curves."

Use `cmd_exec` to run the worker:
`./run_worker.sh workers/step9_worker.py plot_standard`

---

## Phase 2: Agent Assembly & Ad-Hoc Plotting
1. Read the expert-validated text: `data/04_reports/step8_insights_review.md`.
2. Analyze the text to identify if the expert asked for any ad-hoc visualization. If YES, generate it via python script.
3. Generate the validation & audit summary (reviewer-requested transparency):
   `./run_worker.sh workers/step9_worker.py audit_summary`
   This reads the Step 6 verify report and the Step 8 audit report and writes
   `data/04_reports/step9_audit_summary.md`.
4. **Assembly:** Create a new file called `data/04_reports/AKL_Final_Report.md`. 
   - Copy the text from `step8_insights_review.md`.
   - Append the contents of `step9_audit_summary.md` as the final section
     ("Validation & Audit Summary").
   - Embed the standard images into the appropriate sections of the text using Markdown image tags: 
     - `![Topographic Map](figures/topic_map_2d.png)`
     - `![Impact Quadrants](figures/impact_quadrants_likert.png)`
     - `![Countries Clustermap](figures/clustermap_countries.png)`
     - `![Topics Clustermap](figures/clustermap_topics.png)`
     - `![Temporal Evolution](figures/growth_curves.png)`

---

## Phase 3: HTML Compilation
Once `AKL_Final_Report.md` is ready and saved, use `cmd_exec` to compile it:
`./run_worker.sh workers/step9_worker.py compile_html`

## Phase 4: Final Message
Inform the user:
> "🎉 **AKL Pipeline Completed Successfully!**
> 
> The final report with all advanced scientometric visualizations (Dendrograms, Likert, Curves) has been assembled. 
> You will find it at: `data/04_reports/AKL_Final_Report.html`.
> Tip: Open this file in your browser and use `Ctrl+P` to **Save as PDF**.
> 
> It has been a fantastic journey mapping this knowledge landscape with you!"
