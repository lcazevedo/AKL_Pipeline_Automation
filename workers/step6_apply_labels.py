"""
step6_apply_labels.py
---------------------
AKL Pipeline — Step 6 (agent naming mode): apply orchestrator-LLM topic labels.

Reads:
- data/02_intermediate/step6_naming_output.json   (labels: {layer: {topic: label}})
- data/02_intermediate/step6_topic_info/layer_*.pkl (topic_info per layer)
- data/checkpoints/step6_layer_assignments.pkl     (per-layer document assignments)
- data/02_intermediate/step5_corpus_filtered_relevant.pkl (relevant corpus)

Regenerates the step6_topics_final_layer_*.xlsx files with the real labels and
updates pipeline_state.json notes.

Usage:
    ./run_worker.sh workers/step6_apply_labels.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from workers.step6_worker import Step6Worker


def main():
    workspace = Path(__file__).resolve().parent.parent
    labels_path = workspace / "data" / "02_intermediate" / "step6_naming_output.json"
    topic_info_dir = workspace / "data" / "02_intermediate" / "step6_topic_info"
    assignments_path = workspace / "data" / "checkpoints" / "step6_layer_assignments.pkl"
    corpus_path = workspace / "data" / "02_intermediate" / "step5_corpus_filtered_relevant.pkl"

    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    if not assignments_path.exists():
        raise FileNotFoundError(f"Assignments file not found: {assignments_path}")
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    df = pd.read_pickle(str(corpus_path))
    all_topics_per_layer = pd.read_pickle(str(assignments_path))

    worker = Step6Worker("pipeline_config.yaml", "pipeline_state.json")

    # Reconstruct topic_infos with real labels
    topic_infos = []
    n_layers = len(labels)
    for i in range(n_layers):
        ti_path = topic_info_dir / f"layer_{i}.pkl"
        if not ti_path.exists():
            raise FileNotFoundError(f"Topic info not found: {ti_path}")
        ti = pd.read_pickle(str(ti_path))

        layer_labels = labels.get(str(i), {})
        new_labels = []
        for _, row in ti.iterrows():
            t = int(row["Topic"])
            label = layer_labels.get(str(t))
            if not label:
                label = str(row["Name"])
            new_labels.append(label)
        ti["New_Label"] = new_labels
        ti["New_Name"] = ti["Topic"].astype(str) + "_" + ti["New_Label"]
        ti["New_Representation"] = ti["New_Label"]
        ti = ti.drop(columns=["New_Label"])
        topic_infos.append(ti)

    # Best layer from pipeline state (None if awaiting validation)
    best_layer_idx = worker.state["steps"]["6"].get("params_used", {}).get("best_layer")

    # Regenerate xlsx files with real labels
    worker._save_topic_excels(topic_infos, best_layer_idx, all_topics_per_layer, df)

    # Update state notes
    step = worker.state["steps"]["6"]
    step["notes"] = (
        "Topics named by the orchestrator LLM (agent naming mode). "
        f"Labels applied to {n_layers} layers; best layer "
        f"{best_layer_idx if best_layer_idx is not None else 'pending validation'}."
    )
    worker.state["project"]["last_updated"] = (
        __import__("datetime").datetime.now().isoformat(timespec="seconds")
    )
    with open(worker.state_path, "w", encoding="utf-8") as f:
        json.dump(worker.state, f, indent=2, ensure_ascii=False)

    print("\n-- Step 6 Labels Applied ------------------------")
    print(f"  Layers processed: {n_layers}")
    if best_layer_idx is not None:
        print(f"  Best layer: {best_layer_idx} ({len(topic_infos[best_layer_idx])} topics)")
    else:
        print("  Best layer: pending validation (run 'finalize <N>' after review)")
    print("  Labels applied to all layer xlsx files.")


if __name__ == "__main__":
    main()