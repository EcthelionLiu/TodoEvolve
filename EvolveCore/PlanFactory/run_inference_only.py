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
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Tuple

from tqdm.asyncio import tqdm
from dotenv import load_dotenv

# --- Project Imports ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from FlashOAgents import (
        OpenAIServerModel, 
        VisualInspectorTool, 
        TextInspectorTool, 
        AudioInspectorTool, 
        get_zip_description, 
        get_single_file_description
    )
    from Planner_factory.base_agent import MMSearchAgent 
    from utils import read_jsonl, write_jsonl
    from lasj import judge_equivalence
except ImportError as e:
    print(f"Error: Critical modules missing. Detail: {e}")
    sys.exit(1)

# --- Logging Config ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("InferenceRunner")


def preprocess_question(
    item: Dict[str, Any], 
    data_dir: str, 
    visual_tool: VisualInspectorTool, 
    text_tool: TextInspectorTool, 
    audio_tool: AudioInspectorTool
) -> str:
    """
    Analyzes attachments (if any) and appends their description to the question.
    Useful for GAIA or other multimodal benchmarks.
    """
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
                    description = get_zip_description(
                        full_path, question, visual_tool, text_tool, audio_tool
                    )
                    question += "\n\nTo solve the task above, you will have to use these attached files:\n"
                else:
                    description = get_single_file_description(
                        full_path, question, visual_tool, text_tool, audio_tool
                    )
                    question += "\n\nTo solve the task above, you will have to use this attached file:"
                
                question += description
                logger.info(f"Attachment processed: {basename}")
            except Exception as e:
                logger.warning(f"Failed to process attachment {basename}: {e}")
        else:
            logger.warning(f"Attachment not found: {full_path}")
            
    return question


async def process_question_with_retries(
    item: Dict[str, Any], 
    execute_model: OpenAIServerModel, 
    semaphore: asyncio.Semaphore, 
    file_lock: threading.Lock, 
    outfile: str, 
    stats: defaultdict, 
    mm_tools: Tuple
) -> bool:
    """
    Tries to solve a single question using pre-generated planners (v1, v2, v3).
    Stops at the first successful execution that returns a valid result.
    """
    visual_tool, text_tool, audio_tool, data_dir = mm_tools
    
    loop = asyncio.get_event_loop()
    processed_question = await loop.run_in_executor(
        None, 
        preprocess_question, 
        item, data_dir, visual_tool, text_tool, audio_tool
    )
    
    original_question = item.get("question", item.get("query", ""))
    question_id = str(item.get("id", item.get("instance_id", "unknown")))
    
    golden_answer = item.get("golden_answer", item.get("answer", ""))

    async with semaphore:
        for version in range(1, 4):
            planning_system_name = f"planner_{question_id}_{version}"
            
            try:
                module_name = f"FlashOAgents.planning.{planning_system_name}"
                py_file = os.path.join(project_root, "FlashOAgents", "planning", f"{planning_system_name}.py")
                
                if not os.path.exists(py_file):
                    if version == 1:
                        logger.debug(f"[Task {question_id}] Planner file not found: {py_file}")
                    continue
                
                importlib.invalidate_caches()
                if module_name in sys.modules:
                    importlib.reload(sys.modules[module_name])
                else:
                    importlib.import_module(module_name)
                
                # Initialize Agent with the specific planner
                search_agent = MMSearchAgent(
                    model=execute_model,           
                    execute_model=execute_model,   
                    summary_interval=8,
                    prompts_type=planning_system_name,
                    max_steps=40,
                    planning_system=planning_system_name
                )

                # Run Inference
                result = await loop.run_in_executor(None, search_agent, processed_question)

                # Check if execution was valid (produced a result, no error)
                if result and "error" not in result and result.get("agent_result"):
                    logger.info(f"[Task {question_id}] Success with version {version}. Saving result.")
                    
                    # Evaluate correctness
                    judgment_info = judge_equivalence(
                        question=original_question, 
                        gt_answer=str(golden_answer),
                        pred_answer=result.get("agent_result", ""),
                        model=os.environ.get("JUDGE_MODEL")
                    )
                    
                    is_correct_val = judgment_info.get("judgement", False)
                    # Handle string "correct"/"incorrect" responses
                    if isinstance(is_correct_val, str):
                        is_correct_bool = is_correct_val.lower() == 'correct'
                    else:
                        is_correct_bool = bool(is_correct_val)

                    # Update Result Object
                    result["judgement"] = is_correct_bool
                    result["version"] = version
                    result["question_id"] = question_id

                    output_item = {
                        "question": original_question, 
                        "processed_input": processed_question, 
                        "golden_answer": golden_answer,
                        "planning_system": planning_system_name,
                        "successful_version": version,
                        **result
                    }
                    
                    # Write to file immediately
                    with file_lock:
                        write_jsonl(outfile, [output_item], "a")
                        
                        # Update Stats
                        stats["total_solved"] += 1  
                        if is_correct_bool:
                            stats["total_correct"] += 1 
                        
                        if version == 1:
                            stats["v1_exec_success"] += 1
                            if is_correct_bool:
                                stats["v1_correct"] += 1
                    
                    return True # Stop trying other versions if this one worked
                
            except Exception as e:
                logger.error(f"[Task {question_id}] Error in version {version}: {e}")
                continue 

        logger.warning(f"[Task {question_id}] Failed all versions.")
        return False


async def main(args):
    load_dotenv()
    
    # 1. Setup Models
    # EXECUTE_MODEL: The "System 1" model that calls tools
    execute_model = OpenAIServerModel(
        model_id=os.environ.get("EXECUTE_MODEL"),
        api_key=os.environ.get("OPENAI_API_KEY"),
        api_base=os.environ.get("OPENAI_API_BASE"),
        max_completion_tokens=32768,
        custom_role_conversions={"tool-call": "assistant", "tool-response": "user"}
    )
    
    # JUDGE_MODEL: Used for checking attachments (visual/text) and final answer evaluation
    mm_model = OpenAIServerModel(
        model_id=os.environ.get("JUDGE_MODEL"), 
        api_key=os.environ.get("OPENAI_API_KEY"),
        api_base=os.environ.get("OPENAI_API_BASE"),
        max_completion_tokens=32768
    )
    
    # 2. Setup Multimodal Tools
    visual_tool = VisualInspectorTool(mm_model, 100000)
    text_tool = TextInspectorTool(mm_model, 100000)
    audio_tool = AudioInspectorTool(mm_model, 100000)
    
    # 3. Path Handling
    project_root_abs = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    infile_path = args.infile if os.path.isabs(args.infile) else os.path.join(project_root_abs, args.infile)
    outfile_path = args.outfile if os.path.isabs(args.outfile) else os.path.join(project_root_abs, args.outfile)
    
    data_dir = os.path.dirname(infile_path)
    mm_tools_pack = (visual_tool, text_tool, audio_tool, data_dir)
    
    os.makedirs(os.path.dirname(outfile_path), exist_ok=True)
    
    # 4. Load Data
    if not os.path.exists(infile_path):
        logger.error(f"Input file not found: {infile_path}")
        return

    data = read_jsonl(infile_path)
    if args.sample_num:
        data = data[:args.sample_num]
        
    logger.info(f"Loaded {len(data)} tasks. Output will be saved to: {outfile_path}")
    
    # 5. Execution Loop
    semaphore = asyncio.Semaphore(args.concurrency)
    file_lock = threading.Lock()
    stats = defaultdict(int)
    
    tasks = [
        process_question_with_retries(
            item, execute_model, semaphore, file_lock, outfile_path, stats, mm_tools_pack
        ) 
        for item in data
    ]
            
    await tqdm.gather(*tasks, desc="Inference Progress")
    
    # 6. Generate Report
    total_tasks = len(data)
    
    # Avoid division by zero
    def safe_div(n, d): return (n / d * 100) if d > 0 else 0.0

    total_exec_rate = safe_div(stats["total_solved"], total_tasks)
    total_acc_absolute = safe_div(stats["total_correct"], total_tasks)
    
    report_item = {
        "type": "summary_report",  
        "total_tasks": total_tasks,
        "overall": {
            "execution_count": stats["total_solved"],
            "execution_rate": f"{total_exec_rate:.2f}%",
            "correct_count": stats["total_correct"],
            "accuracy": f"{total_acc_absolute:.2f}%",
        }
    }

    with file_lock:
        write_jsonl(outfile_path, [report_item], "a")
    
    print(f"  Accuracy: {stats['total_correct']}/{total_tasks} ({total_acc_absolute:.2f}%)")
    
    print("="*60)
    print(f"Detailed results saved to: {outfile_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference using pre-generated planning systems.")
    
    parser.add_argument(
        "--infile", 
        type=str, 
        required=True, 
        help="Path to the input JSONL dataset."
    )
    
    parser.add_argument(
        "--outfile", 
        type=str, 
        default="inference_results.jsonl", 
        help="Path to save the inference results."
    )
    
    parser.add_argument(
        "--concurrency", 
        type=int, 
        default=5, 
        help="Number of concurrent API requests."
    )
    
    parser.add_argument(
        "--sample_num", 
        type=int, 
        default=None, 
        help="Limit number of samples (for testing)."
    )
    
    args = parser.parse_args()
    
    # Check for critical env vars
    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY is not set. API calls may fail.")

    asyncio.run(main(args))