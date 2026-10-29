"""
step4_worker.py
---------------
AKL Pipeline — Step 4: Expert Curation and Sampling (Breakpoint)

Usage:
    ./run_worker.sh workers/step4_worker.py suggest
    ./run_worker.sh workers/step4_worker.py manual
    ./run_worker.sh workers/step4_worker.py finalize
"""

import sys
import json
import logging
import random
from pathlib import Path
from datetime import datetime
import pandas as pd
import yaml
import openai
import re

def _setup_logger():
    logger = logging.getLogger("step4_worker")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s [STEP4] %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S"))
    if not logger.handlers:
        logger.addHandler(ch)
    return logger

class Step4Worker:
    def __init__(self, config_path="pipeline_config.yaml", state_path="pipeline_state.json"):
        self.config = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
        self.state = json.load(open(state_path, "r", encoding="utf-8"))
        self.state_path = state_path
        self.workspace = Path(self.state.get("project", {}).get("workspace_path", "."))
        self.logger = _setup_logger()

    def _update_state(self, status, output_file=None, params=None, validation=None):
        step = self.state.setdefault("steps", {}).setdefault("4", {})
        step["status"] = status
        if output_file:
            step.setdefault("output", {})["review_file"] = str(output_file)
        if params:
            step.setdefault("params_used", {}).update(params)
        if validation:
            step.setdefault("validation", {}).update(validation)
        json.dump(self.state, open(self.state_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    def _get_criteria_from_project_config(self):
        config_file = self.workspace / "project_config.md"
        if not config_file.exists(): 
            return "No criteria found."
        text = config_file.read_text(encoding="utf-8")
        # Extract Section 6
        match = re.search(r"## 6\. Relevance criteria.*?## 7\.", text, re.DOTALL | re.IGNORECASE)
        if match: 
            return match.group(0).strip()
        return text

    def run_suggest(self, llm_mode="vllm"):
        self.logger.info(f"Starting AI suggestion mode ({llm_mode})...")
        # Reading Excel generated in Step 3 (Updated for new folder)
        topics_file = Path(self.state["steps"]["3"]["output"]["topics_file"])
        review_dir = self.workspace / "data" / "03_review"
        review_dir.mkdir(parents=True, exist_ok=True)
        review_file = review_dir / "step4_relevance_review.xlsx"

        criteria = self._get_criteria_from_project_config()
        sheets = pd.read_excel(topics_file, sheet_name=None)
        
        topic_sheets = [s for s in sheets.keys() if s not in ["Topics", "All_Layers"]]
        if not topic_sheets: 
            raise ValueError("No topic sheets found in Step 3 output.")

        samples_per_sheet = max(3, 60 // len(topic_sheets))
        
        # Setup vLLM client
        vllm_cfg = self.config.get("step7_vllm", {})
        cfg_repro = self.config.get("reproducibility", {}) or {}
        global_seed = cfg_repro.get("seed", 42)
        seed = vllm_cfg.get("seed", global_seed)
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            seed = 42
        base_url = vllm_cfg.get("base_url", "http://localhost:8000/v1")
        llm_model = self.config.get("models", {}).get("llm_extraction", "gemma-3-4b-instruct")
        client = openai.OpenAI(api_key="not-needed", base_url=base_url)

        total_suggested = 0
        yes_count = 0
        no_count = 0

        # Agent mode: collect samples for the orchestrator LLM to classify
        agent_samples = []

        for sheet_name in topic_sheets:
            df = sheets[sheet_name]
            col_rel = "Relevance (yes/no)"
            if col_rel not in df.columns: 
                df.insert(0, col_rel, "")

            # Forces the column to accept text before inserting the LLM results.
            df[col_rel] = df[col_rel].astype(object)

            # Sample indices that are empty
            empty_idx = df[df[col_rel].isna() | (df[col_rel] == "")].index.tolist()
            # Seeded local RNG: the sampled records are reproducible for a given seed
            rng = random.Random(seed)
            sample_idx = rng.sample(sorted(empty_idx), min(len(empty_idx), samples_per_sheet))

            for idx in sample_idx:
                title = df.at[idx, "title"]
                abstract = df.at[idx, "abstract"]

                if llm_mode == "agent":
                    agent_samples.append({
                        "sheet": sheet_name,
                        "row": int(idx),
                        "title": str(title),
                        "abstract": str(abstract),
                    })
                    continue

                prompt = f"""You are an expert classifying academic papers for a systematic review.
Based strictly on these project criteria:
{criteria}

Paper Title: {title}
Abstract: {abstract}

Is this paper RELEVANT to the study? 
Reply ONLY with the word "yes" or the word "no". Do not include any other text."""

                try:
                    resp = client.chat.completions.create(
                        model=llm_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        seed=seed,
                        max_tokens=5
                    )
                    ans = resp.choices[0].message.content.strip().lower()
                    ans_clean = "yes" if "yes" in ans else ("no" if "no" in ans else "")
                    
                    if ans_clean:
                        df.at[idx, col_rel] = ans_clean
                        total_suggested += 1
                        if ans_clean == "yes": yes_count += 1
                        if ans_clean == "no": no_count += 1
                except Exception as e:
                    self.logger.warning(f"LLM call failed for index {idx}: {e}")

        # Save to review folder
        with pd.ExcelWriter(review_file, engine="openpyxl") as writer:
            for s_name, s_df in sheets.items():
                s_df.to_excel(writer, sheet_name=s_name, index=False)

        if llm_mode == "agent":
            # Dump the sample for the orchestrator LLM to classify
            suggest_dir = self.workspace / "data" / "02_intermediate"
            suggest_dir.mkdir(parents=True, exist_ok=True)
            input_path = suggest_dir / "step4_suggest_input.json"
            payload = {"criteria": criteria, "samples": agent_samples}
            with open(input_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            self.logger.info(
                f"Agent mode: dumped {len(agent_samples)} samples to {input_path}. "
                "Run 'apply-suggestions' after the orchestrator labels them."
            )
            self._update_state(
                "awaiting_validation", 
                review_file, 
                {"suggested_yes": 0, "suggested_no": 0, "llm_mode": "agent", "agent_llm": self.config.get("models", {}).get("llm_agent") or "not recorded"},
                validation={"status": "awaiting_validation", "suggested_at": datetime.now().isoformat()}
            )
            return {"review_file": str(review_file), "agent_samples": len(agent_samples),
                    "input_file": str(input_path)}

        self.logger.info(f"Generated {total_suggested} suggestions (Yes: {yes_count}, No: {no_count}).")
        self._update_state(
            "awaiting_validation", 
            review_file, 
            {"suggested_yes": yes_count, "suggested_no": no_count, "llm_mode": "vllm"},
            validation={"status": "awaiting_validation", "suggested_at": datetime.now().isoformat()}
        )
        return {"review_file": str(review_file), "yes": yes_count, "no": no_count}

    def apply_suggestions(self):
        """
        Agent mode: apply the orchestrator LLM's classifications
        (data/02_intermediate/step4_suggest_output.json) to the review file.
        """
        self.logger.info("Applying orchestrator suggestions...")
        suggest_dir = self.workspace / "data" / "02_intermediate"
        output_path = suggest_dir / "step4_suggest_output.json"
        if not output_path.exists():
            raise FileNotFoundError(f"Suggestions output not found: {output_path}")

        with open(output_path, "r", encoding="utf-8") as f:
            suggestions = json.load(f)

        review_file = Path(self.state["steps"]["4"]["output"]["review_file"])
        sheets = pd.read_excel(review_file, sheet_name=None)

        yes_count = 0
        no_count = 0
        for s in suggestions:
            sheet_name = s["sheet"]
            row = s["row"]
            label = str(s["label"]).strip().lower()
            if label not in ("yes", "no"):
                self.logger.warning(f"Skipping invalid label '{label}' for {sheet_name} row {row}.")
                continue
            if sheet_name not in sheets:
                self.logger.warning(f"Sheet '{sheet_name}' not found — skipping.")
                continue
            df = sheets[sheet_name]
            col_rel = "Relevance (yes/no)"
            if col_rel not in df.columns:
                df.insert(0, col_rel, "")
            df.at[row, col_rel] = label
            if label == "yes":
                yes_count += 1
            else:
                no_count += 1

        with pd.ExcelWriter(review_file, engine="openpyxl") as writer:
            for s_name, s_df in sheets.items():
                s_df.to_excel(writer, sheet_name=s_name, index=False)

        self.logger.info(f"Applied {yes_count + no_count} suggestions (Yes: {yes_count}, No: {no_count}).")
        self._update_state(
            "awaiting_validation",
            review_file,
            {"suggested_yes": yes_count, "suggested_no": no_count, "llm_mode": "agent", "agent_llm": self.config.get("models", {}).get("llm_agent") or "not recorded"},
            validation={"status": "awaiting_validation", "suggested_at": datetime.now().isoformat()}
        )
        return {"review_file": str(review_file), "yes": yes_count, "no": no_count}

    def run_manual(self):
        self.logger.info("Preparing manual review file...")
        topics_file = Path(self.state["steps"]["3"]["output"]["topics_file"])
        review_dir = self.workspace / "data" / "03_review"
        review_dir.mkdir(parents=True, exist_ok=True)
        review_file = review_dir / "step4_relevance_review.xlsx"

        # Just copy the file
        import shutil
        shutil.copy(topics_file, review_file)
        
        self._update_state(
            "awaiting_validation", 
            review_file,
            validation={"status": "awaiting_validation", "suggested_at": datetime.now().isoformat()}
        )
        self.logger.info(f"File copied to {review_file}. Awaiting human validation.")
        return {"review_file": str(review_file)}

    def run_finalize(self):
        self.logger.info("Finalizing validation...")
        review_file = Path(self.state["steps"]["4"]["output"]["review_file"])
        sheets = pd.read_excel(review_file, sheet_name=None)
        
        yes_total = 0
        no_total = 0
        
        for s_name, df in sheets.items():
            if s_name in ["Topics", "All_Layers"]: continue
            col = "Relevance (yes/no)"
            if col in df.columns:
                vals = df[col].astype(str).str.strip().str.lower()
                yes_total += (vals == "yes").sum()
                no_total += (vals == "no").sum()

        self._update_state(
            "completed", 
            params={"relevant_count": int(yes_total), "not_relevant_count": int(no_total)},
            validation={"status": "completed", "validated_at": datetime.now().isoformat(), "corrections_made": True}
        )
        self.logger.info(f"Validation complete. Final counts -> Yes: {yes_total}, No: {no_total}")
        return {"yes": int(yes_total), "no": int(no_total)}

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "suggest"
    llm_mode = sys.argv[2] if len(sys.argv) > 2 else "vllm"
    worker = Step4Worker()
    if mode == "suggest": 
        print(json.dumps(worker.run_suggest(llm_mode)))
    elif mode == "manual": 
        print(json.dumps(worker.run_manual()))
    elif mode == "apply-suggestions":
        print(json.dumps(worker.apply_suggestions()))
    elif mode == "finalize": 
        print(json.dumps(worker.run_finalize()))