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
import asyncio
import argparse
import logging
from typing import Dict, Any

from tqdm.asyncio import tqdm
from dotenv import load_dotenv

# --- Project Imports ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from Planner_factory.generator import generate_framework
    from utils import read_jsonl
except ImportError as e:
    print(f"Error: Missing required modules. Details: {e}")
    sys.exit(1)

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("OfflineGenerator")


async def generate_versions(
    item: Dict[str, Any], 
    model_id: str, 
    api_key: str, 
    base_url: str, 
    semaphore: asyncio.Semaphore,
    versions_per_task: int = 3
) -> int:
    """
    Generates multiple versions of the planning system for a specific question.
    """
    question = item.get("question", item.get("query", ""))
    question_id = str(item.get("id", item.get("instance_id", "unknown")))
    
    success_count = 0
    
    for i in range(1, versions_per_task + 1):
        output_name = f"planner_{question_id}_{i}"
        
        async with semaphore:
            logger.info(f"[ID: {question_id}] Generating version {i}/{versions_per_task}: {output_name}")
            
            try:
                success, _, _, _, _, _ = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    generate_framework, 
                    question, model_id, api_key, base_url, output_name
                )
                
                if success:
                    success_count += 1
                else:
                    logger.warning(f"[Error] Failed to generate {output_name}")
            except Exception as e:
                logger.error(f"[Exception] Error generating {output_name}: {e}")
                
    return success_count


async def main(args):
    load_dotenv()
    
    # 1. Path Setup
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    infile_path = args.infile if os.path.isabs(args.infile) else os.path.join(project_root, args.infile)
    
    # 2. Load Data
    if not os.path.exists(infile_path):
        logger.error(f"Input file not found: {infile_path}")
        return

    data = read_jsonl(infile_path)
    if args.sample_num:
        data = data[:args.sample_num]
        
    logger.info(f"Loaded {len(data)} items. Starting offline generation...")
    
    # 3. Model Configuration
    model_id = args.model or os.getenv("PLANNING_MODEL")
    if not model_id:
        logger.error("Error: Model ID not provided. Use --model or set PLANNING_MODEL env var.")
        return

    api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
    base_url = os.getenv("OPENAI_API_BASE", "None")
    
    # 4. Execution
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = []
    
    for item in data:
        tasks.append(
            generate_versions(item, model_id, api_key, base_url, semaphore, args.versions)
        )
    
    await tqdm.gather(*tasks, desc="Generation Progress")
    
    # 5. Summary
    print("\n" + "="*60)
    print("Generation Task Completed!")
    print(f"Generated Python files: FlashOAgents/planning/planner_{{id}}_{{1-{args.versions}}}.py")
    print(f"Generated YAML files:   FlashOAgents/prompts/planner_{{id}}_{{1-{args.versions}}}/toolcalling_agent.yaml")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline Batch Generator for Planning Systems")
    
    parser.add_argument(
        "--infile", 
        type=str, 
        default="Planner_factory/data/example.jsonl",
        help="Path to input dataset (JSONL)"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        help="Model ID or Local Path (e.g., /path/to/vllm/model)"
    )
    
    parser.add_argument(
        "--concurrency", 
        type=int, 
        default=10,
        help="Number of concurrent generation tasks"
    )
    
    parser.add_argument(
        "--sample_num", 
        type=int, 
        default=None,
        help="Limit number of samples for testing"
    )

    parser.add_argument(
        "--versions", 
        type=int, 
        default=3,
        help="Number of variations to generate per question"
    )
    
    args = parser.parse_args()
    asyncio.run(main(args))