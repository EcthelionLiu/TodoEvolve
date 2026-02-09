<div align="center">

# TodoEvolve: Task-Adaptive Meta-Planning for LLM Agents 🧭

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-red.svg)](https://arxiv.org)

<!-- Optional: add your framework figure -->

<!-- ![](assets/framework.png) -->

</div>

## 👋 Introduction

This repo is the official implementation of **TodoEvolve**, a framework for **task-adaptive agentic planning**.

Planning is a core capability of LLM-powered agents, enabling coherent long-horizon execution, global-state maintenance, and coordinated action. However, existing planning systems vary substantially across **planning targets** (single-agent vs multi-agent), **representational forms** (linear to-do, DAG, tree, hierarchical notes), and **domains** (web search, software engineering, embodied tasks). We posit that a universal, one-size-fits-all planner is unrealistic.

TodoEvolve addresses this by learning a **meta-planner** that *customizes* planning systems to each task. Concretely, we optimize the meta-planner with **Impedance-Guided Preference Optimization (IGPO)**, a multi-objective preference learning objective that jointly promotes **performance**, **stability**, and **token efficiency**. The output is a task-specific to-do structure characterized by its **topology**, **initialization**, **adaptation**, and **navigation** strategy.

---

## 🌎 Setup

### 1. Environment Setup 🛠️

```bash
conda create -n TodoEvolve python=3.10
conda activate TodoEvolve
pip install -r requirements.txt
```

### 2. Training Dependencies (Optional) 🧪

If you plan to run training (SFT or IGPO), this repo relies on the **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** ecosystem. Please follow LLaMA-Factory’s official installation instructions for CUDA/PyTorch/vLLM and efficient fine-tuning dependencies.

### 3. Configure Environment Variables 🔑

Following **[FlashSearcher](https://github.com/OPPO-PersonalAI/Flash-Searcher)**, we use `SearchTool` and `CrawlTool` for web search and page crawling.

Set API keys for the selected provider:

* `SERPER_API_KEY` for [Serper](https://serper.dev/)
* `JINA_API_KEY` for [JinaAI](https://jina.ai/)

Also set model-related environment variables:

* `OPENAI_BASE_URL`
* `OPENAI_API_KEY`
* `PLANNING_MODEL`
* `EXECUTE_MODEL`
* `JUDGE_MODEL`

> **Tip**: Use a strong model for `JUDGE_MODEL` to improve preference quality for IGPO.

---

## 🧩 PlanFactory (Planning System Decomposition)

**PlanFactory** is a standardized codebase for decomposing and reproducing heterogeneous planning systems under a unified design space:

* **Topology**: how the plan is structured (linear / DAG / tree / hierarchical)
* **Initialization**: how the structure is instantiated from the task input
* **Adaptation**: when/how the structure is revised during execution
* **Navigation**: how executable directives are issued to the acting agent

### Supported Frameworks 📚

Implemented under `PlanFactory/planning/` and `PlanFactory/prompts/`:

| Planning System    | Source |
| ------------------ | ------ |
| **FlashSearcher**  |  https://github.com/OPPO-PersonalAI/Flash-Searcher      |
| **OAgent**         |    https://github.com/OPPO-PersonalAI/OAgents    |
| **Owl**            |    https://github.com/camel-ai/owl    |
| **CoSight**        |   https://github.com/ZTE-AICloud/Co-Sight     |
| **FlowSearch**     |  https://github.com/InternScience/InternAgent      |
| **AgentOrchestra** |  https://github.com/SkyworkAI/DeepResearchAgent      |
| **JoyAgent**       |  https://github.com/jd-opensource/joyagent-jdgenie      |

### Tools 🧰

Core tool abstractions:

* `tools.py`
* `cosight_tool.py`

### Inference: Running Decomposed Systems ▶️

To reproduce the planning behaviors of the supported frameworks, run the unified entry script and specify the target system. The configuration for these systems are mapped in `EvolveCore/FlashOAgent/agents.py` (lines 598-607).

```bash
python run_flash_searcher.py \
  --infile /path/to/benchmark_file.json \
  --outfile ./output/result.jsonl \
  --max_steps 40 \
  --sample_num 20 \
  --planning_system <PLAN_SYSTEM_NAME>
```

---

## 🧪 TodoEvolve (Data Synthesis & Training)

**TodoEvolve** provides a full pipeline for:

1. collecting planning trajectories from decomposed planning systems, and
2. training a meta-planner via **SFT + IGPO**.

### Collect Raw Training Data 📦

```bash
python PlanFactory/collect_train_data.py \
  --infile /path/to/benchmark_file.json \
  --outfile ./output/train_data.jsonl \
  --concurrency 10 \
  --sample_num 20
```

---

## 🚀 Inference

### 1. Launch vLLM Server (Optional) 🖥️

If you want to load a checkpoint for generation and inference, launch an OpenAI-compatible API server with **vLLM**:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/your/saved/checkpoint \
  --served-model-name your-model-name \
  --port 8001 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --max-model-len 30000 \
  --dtype bfloat16
```

This local vLLM instance replaces **only** the `PLANNING_MODEL`.
`EXECUTE_MODEL` and `JUDGE_MODEL` remain unchanged.

### 2. Generate Plans Only 🧱

```bash
python generator_only.py \
  --infile /path/to/benchmark_file \
  --model /path/to/model \
  --concurrency 50
```

### 3. Inference Only (on pre-generated plans) 🔍

```bash
python run_inference_only.py \
  --infile /path/to/benchmark_file \
  --outfile ./output/result.jsonl \
  --concurrency 1 \
  --sample_num 20
```

### 4. End-to-End Inference 🔁

```bash
python model_infer.py \
  --infile /path/to/benchmark_file \
  --outfile ./output/result.jsonl \
  --concurrency 1 \
  --sample_num 20
```

---

## 🧰 Trainer Utilities

This repo contains lightweight scripts for preparing, analyzing, and validating datasets for LLM training.

### 1. Construct SFT Dataset 🧾

```bash
python construct_SFTdataset.py \
  --input_path ./data/raw_data \
  --output_file ./data/sft_data_final.json
```

### 2. Construct IGPO Dataset 🧠

Processes pairwise data for **Impedance-Guided Preference Optimization (IGPO)**, including filtering based on correctness and **cognitive impedance**.

```bash
python construct_DPOdataset.py \
  --input_path ./data/igpo_raw \
  --output_file ./data/igpo_data_final.json
```

### 3. Training Execution 🏋️

Use scripts in `examples/train_full/` (inherited from **LLaMA-Factory**) to launch SFT/IGPO training.

### 4. Checkpoints 📌

Our fine-tuned **Todo-14B** model weights are available on Hugging Face at [https://huggingface.co/EcthelionLiu/Todo-14B](https://huggingface.co/EcthelionLiu/Todo-14B).

---

## 🫡 Citation

If you find TodoEvolve helpful in your research, please kindly consider citing:

```bibtex
@misc{todoxxxx,
  title={TodoEvolve: Task-Adaptive Meta-Planning via Impedance-Guided Preference Optimization},
  author={...},
  year={2026},
  eprint={XXXX.XXXXX},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

---

## 🙏 Acknowledgments

This repo builds upon and adapts code from:

* **[Flash-Searcher](https://github.com/OPPO-PersonalAI/Flash-Searcher)** — planning-oriented web execution and trajectory collection
* **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** — training ecosystem for SFT and preference optimization

---

## 📄 License

This project is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for details.
