#  TodoEvolve
This repository contains a comprehensive framework for decomposing planning algorithms, collecting data, training evolved models using LLaMA-Factory, and performing scalable inference. The project is divided into two core modules: EvolveCore and PlanFactory.

1. Environment Installation
To get started, we recommend using Conda to manage the environment.

```bash
conda create -n TodoEvolve python=3.10
conda activate TodoEvolve
pip install -r requirements.txt
```
If you intend to run the training pipeline (SFT or DPO), this project relies on the **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** ecosystem. So please refer to the official LLaMA-Factory Repository for detailed installation instructions regarding CUDA, PyTorch, and specific dependencies for efficient fine-tuning.

2. Set up environment variables

Following **[FlashSearcher](https://github.com/OPPO-PersonalAI/Flash-Searcher)**, our framework uses `SearchTool` and `CrawlTool` for web search and page crawling. These tools require setting environment variables with the corresponding API keys, depending on the selected provider:"

* `SERPER_API_KEY` for SerpApi: [Serper](https://serper.dev/)
* `JINA_API_KEY` for JinaApi: [JinaAI](https://jina.ai/)

Depending on the model you want to use, you may need to set environment variables. You need to set the `OPENAI_BASE_URL` ,`OPENAI_API_KEY` , `PLANNING_MODEL`, `EXECUTE_MODEL` and `JUDGE_MODEL` environment variable.

# EvolveCore

EvolveCore serves as the central engine for the project, organized into two primary directories: **FlashOAgents** for planning framework implementations and **PlanFactory** for high-throughput generation and inference.

## FlashOAgents (Planning Frameworks)

This module implements the decomposition of seven state-of-the-art planning frameworks. These implementations serve as the foundation for our data collection and agent exploration.

### Supported Frameworks
The following planning systems are implemented under `FlashOAgents/planning/` and `FlashOAgents/prompts/`:

*   **[FlashSearcher](https://arxiv.org/abs/2509.25301)**
*   **[OAgent](https://arxiv.org/abs/2506.15741)**
*   **[Owl](https://arxiv.org/abs/2505.23885)**
*   **[CoSight](https://arxiv.org/abs/2510.21557)**
*   **[FlowSearch](https://arxiv.org/abs/2510.08521)**
*   **[AgentOrchestra](https://arxiv.org/abs/2506.12508)**
*   **[JoyAgent](https://arxiv.org/abs/2510.00510)**

### Tools
The underlying tools required to reproduce the functionality of these planning systems are located in:
*   `tools.py`
*   `cosight_tool.py`

### collect raw train data by `PlanFactory/collect_train_data.py`
```bash
python collect_train_data.py --infile /path/to/benchmark_file.json --outfile ./output/train_data.jsonl --concurrency 10 --sample_num 20
```

## PlanFactory (Generation & Inference)

The PlanFactory directory handles data generation, plan simulation, and model inference. 

1. Launch vLLM Server(Optional): If you need to load a checkpoint for generation and inference, you must use the **vLLM** framework to launch an OpenAI-compatible API server.

**Note:** This local vLLM instance specifically replaces the `PLANNING_MODEL` in the API calls. The `EXECUTE_MODEL` and `JUDGE_MODEL` configurations remain unchanged.

```bash
python -m vllm.entrypoints.openai.api_server --model /path/to/your/saved/checkpoint --served-model-name your-model-name --port 8001 --trust-remote-code --gpu-memory-utilization 0.9 --max-model-len 30000 --dtype bfloat16
```

2. Generate Plan System Only: Run the generator to create plans without full execution.

```bash
python generator_only.py --infile /path/to/benchmark_file --model /path/to/model --concurrency 50 
```

C. Run Inference Only: Execute inference on pre-generated data.

```bash
python run_inference_only.py --infile /path/to/benchmark_file --outfile ./output/result.jsonl --concurrency 1 --sample_num 20
```

D. End-to-End Inference: Perform the complete generation and inference loop in a single step.

```bash
python model_infer.py --infile /path/to/benchmark_file --outfile ./output/result.jsonl --concurrency 1 --sample_num 20
```

# Trainer

This repository contains a collection of lightweight, efficient utility scripts designed to prepare, analyze, and validate datasets for Large Language Model (LLM) training. 

1. Construct SFT Dataset

Converts raw data into the standard Supervised Fine-Tuning (SFT) format.

```bash
python construct_SFTdataset.py --input_path ./data/raw_data --output_file ./data/sft_data_final.json
```

2. Construct DPO Dataset

Processes pairwise data for Direct Preference Optimization (DPO). It includes logic to filter samples based on correctness and "Cognitive Impedance" .

```bash
python construct_DPOdataset.py --input_path ./data/dpo_raw --output_file ./data/dpo_data_final.json
```

3. Analyze Dataset Statistics

Generates token length statistics for Input, Reasoning, and Output. Useful for paper writing and understanding data distribution.

```bash
python analyze_dataset.py --sft_file ./data/sft_data_final.json --dpo_file ./data/dpo_data_final.json
```

4. Check Training Context Length

Use this script to determine the appropriate cutoff_len for your yaml configuration in LLaMA-Factory. It uses the actual tokenizer to account for special tokens and chat templates.

For SFT:
```bash
python check_training_context_length.py --data ./data/sft_data_final.json --model /path/to/your/model_directory --mode sft
```
For DPO:
```bash
python check_training_context_length.py --data ./data/dpo_data_final.json --model /path/to/your/model_directory --mode dpo
```
5. Training Execution
To launch training, utilize the scripts provided in the examples/train_full/ directory (inherited from **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)**).