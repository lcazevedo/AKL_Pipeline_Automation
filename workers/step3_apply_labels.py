"""
step3_apply_labels.py
---------------------
AKL Pipeline — Step 3 (agent naming mode): apply orchestrator-LLM topic labels.

Reads:
- data/02_intermediate/step3_naming_output.json   (labels: {layer: {topic: label}})
- data/02_intermediate/step3_topic_info/layer_*.pkl (topic_info per layer)
- data/02_intermediate/step3_corpus_with_topics.pkl (corpus with topic_layer_N columns)

Regenerates the step3_topics_preliminary_layer_*.xlsx files with the real
labels and updates pipeline_state.json notes.

Usage:
    ./run_worker.sh workers/step3_apply_labels.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from workers.step3_worker import Step3Worker


def main():
    workspace = Path(__file__).resolve().parent.parent
    labels_path = workspace / "data" / "02_intermediate" / "step3_naming_output.json"
    topic_info_dir = workspace / "data" / "02_intermediate" / "step3_topic_info"
    corpus_path = workspace / "data" / "02_intermediate" / "step3_corpus_with_topics.pkl"

    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    df = pd.read_pickle(str(corpus_path))

    worker = Step3Worker("pipeline_config.yaml", "pipeline_state.json")

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

    # Reconstruct per-layer topic assignments from the corpus columns
    all_topics_per_layer = {}
    for i in range(n_layers):
        col = f"topic_layer_{i}"
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in corpus_with_topics.pkl")
        all_topics_per_layer[i] = df[col].values

    # Best layer from pipeline state
    best_layer_idx = worker.state["steps"]["3"]["params_used"]["best_layer"]

    # Regenerate xlsx files with real labels
    selected_path = worker._save_topics_xlsx(
        topic_infos, best_layer_idx, all_topics_per_layer, df
    )

    # Update state notes
    step = worker.state["steps"]["3"]
    step["notes"] = (
        "Topics named by the orchestrator LLM (agent naming mode). "
        f"Labels applied to {n_layers} layers; best layer {best_layer_idx} "
        f"({len(topic_infos[best_layer_idx])} topics)."
    )
    worker.state["project"]["last_updated"] = (
        __import__("datetime").datetime.now().isoformat(timespec="seconds")
    )
    worker._save_json(worker.state, worker.state_path)

    print("\n── Step 3 Labels Applied ────────────────────────")
    print(f"  Layers processed: {n_layers}")
    print(f"  Best layer: {best_layer_idx} ({len(topic_infos[best_layer_idx])} topics)")
    print(f"  Selected file: {selected_path}")
    print("  Labels applied to all layer xlsx files.")


if __name__ == "__main__":
    main()