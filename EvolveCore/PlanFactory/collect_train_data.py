#!/usr/bin/env python
# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import json
import uuid
import shutil
import logging
import asyncio
import argparse
import threading
import importlib
from collections import Counter
from typing import Dict, Any, Optional

from tqdm.asyncio import tqdm
from dotenv import load_dotenv

# --- Project Imports ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Fail fast if dependencies are missing
from FlashOAgents import OpenAIServerModel
from Planner_factory.base_agent import SearchAgent
from Planner_factory.generator import generate_framework
from utils import read_jsonl, write_jsonl
from lasj import judge_equivalence

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("DataCollector")


def adapt_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Standardize dataset formats."""
    question = item.get("question") or item.get("query") or item.get("input", "")
    
    answer = ""
    for field in ["golden_answer", "answer", "validated_answer", "ground_truth"]:
        if field in item:
            answer = item[field]
            break
    
    item["question"] = question
    item["answer"] = answer
    
    if "instance_id" not in item:
        item["instance_id"] = str(uuid.uuid4())[:8]
        
    return item


async def process_item_async(
    item: Dict[str, Any],
    planning_model: OpenAIServerModel,
    execute_model: OpenAIServerModel,
    summary_interval: int,
    prompts_type: str,
    max_steps: int,
    auto_gen: bool,
    semaphore: asyncio.Semaphore,
    file_lock: threading.Lock,
    outfile: str
) -> Optional[Dict[str, Any]]:
    """
    Executes a single Agent task: Auto-Gen Code -> Run Agent -> Judge.
    """
    async with semaphore:
        item = adapt_item(item)
        question = item["question"]
        golden_answer = item["answer"]
        instance_id = item["instance_id"]
        
        safe_id = instance_id.replace('-', '_')
        planning_system_name = f"planner_{safe_id}"
        
        # Paths for dynamic code generation
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        temp_py_path = os.path.join(base_path, "FlashOAgents", "planning", f"{planning_system_name}.py")
        temp_prompt_dir = os.path.join(base_path, "FlashOAgents", "prompts", planning_system_name)
        
        py_code, yaml_code, gen_think = "", "", ""

        try:
            # --- Step 1: Dynamic Planner Generation ---
            if auto_gen:
                gen_model = os.environ["PLANNING_MODEL"]
                api_key = os.environ.get("OPENAI_API_KEY")
                base_url = os.environ.get("OPENAI_BASE_URL")

                loop = asyncio.get_event_loop()
                success, py_code, yaml_code, _, _, gen_think = await loop.run_in_executor(
                    None, 
                    generate_framework, 
                    question, gen_model, api_key, base_url, planning_system_name
                )
                
                if not success:
                    logger.warning(f"[{instance_id}] Generation failed. Skipping.")
                    return None

                # Import the generated module
                module_path = f"FlashOAgents.planning.{planning_system_name}"
                if module_path in sys.modules:
                    importlib.reload(sys.modules[module_path])
                else:
                    importlib.import_module(module_path)

            # --- Step 2: Agent Execution ---
            agent_prompt = planning_system_name if auto_gen else (prompts_type if prompts_type != "default" else None)
            agent_sys = planning_system_name if auto_gen else "planner"

            search_agent = SearchAgent(
                planning_model,
                summary_interval=summary_interval,
                prompts_type=agent_prompt,
                max_steps=max_steps,
                planning_system=agent_sys,
                execute_model=execute_model
            )

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, search_agent, question)

            if not result or "error" in result:
                return None

            # --- Step 3: Standard Evaluation ---
            # Using standard judge_equivalence for all tasks
            judgment_info = judge_equivalence(
                question=question,
                gt_answer=str(golden_answer),
                pred_answer=result.get("agent_result", ""),
                model=os.environ.get("JUDGE_MODEL")
            )
            
            result["judgement"] = judgment_info.get("judgement", "error")
            logger.info(f"[{instance_id}] Result: {result['judgement']}")

            # --- Step 4: Save Data ---
            output_item = {
                "instance_id": instance_id,
                "question": question,
                "golden_answer": golden_answer,
                "generated_planner_code": py_code,
                "generated_planner_config": yaml_code,
                "generator_thought_process": gen_think,
                **result,
            }

            with file_lock:
                write_jsonl(outfile, [output_item], "a")
            
            return output_item

        except Exception as e:
            # Catch-all is still needed here to prevent one bad sample from crashing the entire batch
            logger.error(f"[{instance_id}] Failed: {e}")
            return None
            
        finally:
            # Simple cleanup
            if auto_gen:
                if os.path.exists(temp_py_path):
                    os.remove(temp_py_path)
                if os.path.exists(temp_prompt_dir):
                    shutil.rmtree(temp_prompt_dir, ignore_errors=True)


async def process_item_with_retry(
    item: Dict[str, Any],
    planning_model: OpenAIServerModel,
    execute_model: OpenAIServerModel,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
    file_lock: threading.Lock
):
    """Retries until target_success count is met."""
    item = adapt_item(item)
    current_success = 0
    max_retries = 5

    for i in range(max_retries):
        if current_success >= args.target_success:
            break
            
        result = await process_item_async(
            item, planning_model, execute_model, 
            args.summary_interval, args.prompts_type, args.max_steps, args.auto_gen,
            semaphore, file_lock, args.outfile
        )
        
        if result:
            current_success += 1

    if current_success < args.target_success:
        # Log failure record
        error_record = {
            "question": item["question"],
            "judgement": "data_collection_failure",
            "metadata": {"success_count": current_success, "target": args.target_success}
        }
        with file_lock:
            write_jsonl(args.outfile, [error_record], "a")


def create_model(model_config: str):
    if not model_config:
        return None
        
    if model_config.startswith("local:") or os.path.isdir(model_config):
        from FlashOAgents.models import LocalTransformersModel
        return LocalTransformersModel(model_config.replace("local:", ""))
    
    return OpenAIServerModel(
        model_config,
        custom_role_conversions={"tool-call": "assistant", "tool-response": "user"},
        max_completion_tokens=32768,
        api_key=os.environ.get("OPENAI_API_KEY"),
        api_base=os.environ.get("OPENAI_API_BASE"),
    )


async def main(args):
    # Path Setup
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    infile = args.infile if os.path.isabs(args.infile) else os.path.join(root, args.infile)
    outfile = args.outfile if os.path.isabs(args.outfile) else os.path.join(root, args.outfile)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    # Load Data
    if infile.endswith('.json'):
        with open(infile, 'r') as f:
            raw = json.load(f)
            data = list(raw.values()) if isinstance(raw, dict) else raw
    else:
        data = read_jsonl(infile)
    
    if args.sample_num:
        data = data[:args.sample_num]

    # Deduplicate
    unique_tasks = {adapt_item(x)["question"]: adapt_item(x) for x in data}.values()
    logger.info(f"Tasks to process: {len(unique_tasks)}")

    # Init Models
    planning_model = create_model(os.environ["PLANNING_MODEL"])
    execute_model = create_model(os.environ["EXECUTE_MODEL"])

    # Resume Logic
    completed_counts = Counter()
    if os.path.exists(outfile):
        completed_counts = Counter([
            x.get("question") for x in read_jsonl(outfile) 
            if x.get("judgement") != "data_collection_failure"
        ])

    # Task Queue
    semaphore = asyncio.Semaphore(args.concurrency)
    file_lock = threading.Lock()
    tasks = []

    for item in unique_tasks:
        if completed_counts[item["question"]] >= args.target_success:
            continue
            
        tasks.append(process_item_with_retry(
            item, planning_model, execute_model, args, semaphore, file_lock
        ))

    if tasks:
        logger.info(f"Starting {len(tasks)} tasks...")
        await tqdm.gather(*tasks, desc="Collecting Data")
    else:
        logger.info("All tasks completed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect Training Data')
    parser.add_argument('--infile', type=str, required=True)
    parser.add_argument('--outfile', type=str, required=True)
    parser.add_argument('--concurrency', type=int, default=5)
    parser.add_argument('--target_success', type=int, default=2)
    parser.add_argument('--sample_num', type=int, default=None)
    parser.add_argument('--auto_gen', action='store_true', default=True)
    parser.add_argument('--max_steps', type=int, default=40)
    parser.add_argument('--summary_interval', type=int, default=8)
    parser.add_argument('--prompts_type', type=str, default="default")

    args = parser.parse_args()
    
    # Ensure critical env vars exist
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is missing in .env")

    asyncio.run(main(args))