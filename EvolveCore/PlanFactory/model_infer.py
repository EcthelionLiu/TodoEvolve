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
import importlib
import threading
from collections import defaultdict
from typing import Dict, Any, Tuple

from tqdm.asyncio import tqdm
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Imports ---
try:
    from FlashOAgents import (
        OpenAIServerModel, 
        VisualInspectorTool, 
        TextInspectorTool, 
        AudioInspectorTool, 
        get_zip_description, 
        get_single_file_description
    )
    from FlashOAgents.planning import * # Pre-load planning package
    from Planner_factory.base_agent import MMSearchAgent 
    from Planner_factory.generator import generate_framework
    from utils import read_jsonl, write_jsonl
    from lasj import judge_equivalence
except ImportError as e:
    print(f"[Critical Error] Missing required modules: {e}")
    sys.exit(1)

# --- Logging ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("UnifiedPipeline")


def preprocess_question(
    item: Dict[str, Any], 
    data_dir: str, 
    visual_tool: VisualInspectorTool, 
    text_tool: TextInspectorTool, 
    audio_tool: AudioInspectorTool
) -> str:
    """Pre-processes the question to include attachment descriptions."""
    question = item.get("question", item.get("query", ""))
    file_name = item.get("file_name", "")
    
    if file_name:
        basename = os.path.basename(file_name)
        full_path = os.path.join(data_dir, "attachments", basename)
        full_path = full_path.replace("/", os.sep).replace("\\", os.sep)

        if os.path.exists(full_path):
            try:
                description = ""
                lower_path = full_path.lower()
                if ".zip" in lower_path:
                    description = get_zip_description(full_path, question, visual_tool, text_tool, audio_tool)
                    question += "\n\n[System] Attached files content:\n" + description
                else:
                    description = get_single_file_description(full_path, question, visual_tool, text_tool, audio_tool)
                    question += "\n\n[System] Attached file content:\n" + description
                logger.info(f"Processed attachment: {basename}")
            except Exception as e:
                logger.warning(f"Attachment processing failed: {e}")
    return question


async def process_single_item(
    item: Dict[str, Any],
    planning_model_id: str,
    execute_model: OpenAIServerModel,
    api_key: str,
    base_url: str,
    mm_tools: Tuple,
    semaphore: asyncio.Semaphore,
    file_lock: threading.Lock,
    outfile: str,
    stats: defaultdict
):
    """
    Executes the full pipeline for a single item:
    1. Generate Planner (Code & Config)
    2. Run Inference (Agent Execution)
    3. Evaluate Result (Judge)
    """
    visual_tool, text_tool, audio_tool, data_dir = mm_tools
    
    # 1. Identify Task
    question = item.get("question", item.get("query", ""))
    question_id = str(item.get("id", item.get("instance_id", "unknown")))
    golden_answer = item.get("golden_answer", item.get("answer", ""))
    
    # Unique identifier for this specific planner version
    # Using 'v1' as default since we are generating on the fly
    planner_name = f"planner_{question_id}_v1"

    async with semaphore:
        try:
            # --- Phase 1: Generation ---
            
            # call the generator function from Planner_factory
            gen_success, _, _, _, _, _ = await asyncio.get_event_loop().run_in_executor(
                None, 
                generate_framework, 
                question, planning_model_id, api_key, base_url, planner_name
            )
            
            if not gen_success:
                logger.error(f"[Task {question_id}] Generation Failed.")
                return

            # --- Phase 2: Inference ---
            
            # Preprocess question (attachments)
            processed_question = await asyncio.get_event_loop().run_in_executor(
                None, preprocess_question, item, data_dir, visual_tool, text_tool, audio_tool
            )
            
            # Dynamic Import of the newly generated module
            module_path = f"FlashOAgents.planning.{planner_name}"
            
            try:
                importlib.invalidate_caches()
                if module_path in sys.modules:
                    importlib.reload(sys.modules[module_path])
                else:
                    importlib.import_module(module_path)
            except ImportError as e:
                logger.error(f"[Task {question_id}] Failed to import generated module: {e}")
                return

            # Initialize Agent
            search_agent = MMSearchAgent(
                model=execute_model,           
                execute_model=execute_model,   
                summary_interval=8,
                prompts_type=planner_name,
                max_steps=40,
                planning_system=planner_name
            )
            
            # Execute Agent
            result = await asyncio.get_event_loop().run_in_executor(
                None, search_agent, processed_question
            )
            
            if not result or "error" in result:
                logger.warning(f"[Task {question_id}] Agent Execution Failed or returned Error.")
                return

            # --- Phase 3: Evaluation ---
            
            judgment_info = judge_equivalence(
                question=question, 
                gt_answer=str(golden_answer),
                pred_answer=result.get("agent_result", ""),
                model=os.environ.get("JUDGE_MODEL")
            )
            
            is_correct_val = judgment_info.get("judgement", False)
            if isinstance(is_correct_val, str):
                is_correct = is_correct_val.lower() == 'correct'
            else:
                is_correct = bool(is_correct_val)

            output_item = {
                "id": question_id,
                "question": question,
                "golden_answer": golden_answer,
                "planner_name": planner_name,
                "processed_input": processed_question,
                "agent_result": result.get("agent_result"),
                "agent_trajectory": result.get("agent_trajectory"),
                "judgement": is_correct,
                "judge_reasoning": judgment_info.get("reasoning", "")
            }
            
            with file_lock:
                write_jsonl(outfile, [output_item], "a")
                stats["total_processed"] += 1
                if is_correct:
                    stats["total_correct"] += 1
                    
            logger.info(f"[Task {question_id}] Completed. Correct: {is_correct}")

        except Exception as e:
            logger.error(f"[Task {question_id}] Unhandled Exception: {e}")
            import traceback
            traceback.print_exc()


async def main(args):
    load_dotenv()
    
    # 1. Configuration Validation
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")
    planning_model_id = args.planning_model or os.getenv("PLANNING_MODEL")
    execute_model_id = args.execute_model or os.getenv("EXECUTE_MODEL")
    judge_model_id = os.getenv("JUDGE_MODEL")

    if not all([api_key, planning_model_id, execute_model_id, judge_model_id]):
        logger.error("Missing required environment variables or arguments.")
        logger.error("Ensure OPENAI_API_KEY, PLANNING_MODEL, EXECUTE_MODEL, JUDGE_MODEL are set.")
        return

    # 2. Setup Models
    # Execution Model (System 1)
    execute_model = OpenAIServerModel(
        model_id=execute_model_id,
        api_key=api_key,
        api_base=base_url,
        max_completion_tokens=32768,
        custom_role_conversions={"tool-call": "assistant", "tool-response": "user"}
    )
    
    # Multimodal Model (for tools)
    mm_model = OpenAIServerModel(
        model_id=judge_model_id, 
        api_key=api_key, 
        api_base=base_url
    )
    
    # 3. Setup Tools
    visual_tool = VisualInspectorTool(mm_model, 100000)
    text_tool = TextInspectorTool(mm_model, 100000)
    audio_tool = AudioInspectorTool(mm_model, 100000)
    
    # 4. Prepare Paths
    infile_path = os.path.abspath(args.infile)
    outfile_path = os.path.abspath(args.outfile)
    data_dir = os.path.dirname(infile_path)
    mm_tools_pack = (visual_tool, text_tool, audio_tool, data_dir)
    
    os.makedirs(os.path.dirname(outfile_path), exist_ok=True)
    
    # 5. Load Data
    if not os.path.exists(infile_path):
        logger.error(f"Input file not found: {infile_path}")
        return
    data = read_jsonl(infile_path)
    if args.sample_num:
        data = data[:args.sample_num]
        
    logger.info(f"Loaded {len(data)} items.")
    logger.info(f"Planning Model: {planning_model_id}")
    logger.info(f"Execute Model:  {execute_model_id}")

    # 6. Execution Loop
    # Concurrency defaults to 1 to prevent planner file conflicts
    semaphore = asyncio.Semaphore(args.concurrency) 
    file_lock = threading.Lock()
    stats = defaultdict(int)
    
    tasks = []
    for item in data:
        tasks.append(
            process_single_item(
                item, 
                planning_model_id, 
                execute_model, 
                api_key, 
                base_url, 
                mm_tools_pack, 
                semaphore, 
                file_lock, 
                outfile_path, 
                stats
            )
        )
    
    await tqdm.gather(*tasks, desc="Pipeline Progress")
    
    # 7. Final Report
    processed = stats["total_processed"]
    correct = stats["total_correct"]
    acc = (correct / processed * 100) if processed > 0 else 0
    
    print("\n" + "="*60)
    print("="*60)
    print(f"Total Processed: {processed}/{len(data)}")
    print(f"Correct Answers: {correct}")
    print(f"Accuracy:        {acc:.2f}%")
    print(f"Results saved to: {outfile_path}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Unified Flash-Planner Pipeline (Generate -> Execute -> Judge)")
    
    parser.add_argument(
        "--infile", 
        type=str, 
        required=True, 
        help="Path to input JSONL dataset"
    )
    
    parser.add_argument(
        "--outfile", 
        type=str, 
        default="pipeline_results.jsonl", 
        help="Path to save results"
    )
    
    parser.add_argument(
        "--concurrency", 
        type=int, 
        default=1, 
        help="Concurrency level (Default 1 recommended to avoid planner conflicts)"
    )
    
    parser.add_argument(
        "--sample_num", 
        type=int, 
        default=None, 
        help="Limit number of samples for testing"
    )
    
    parser.add_argument(
        "--planning_model", 
        type=str, 
        help="Model ID for generating planners (overrides env var)"
    )
    
    parser.add_argument(
        "--execute_model", 
        type=str, 
        help="Model ID for executing agents (overrides env var)"
    )

    args = parser.parse_args()
    asyncio.run(main(args))