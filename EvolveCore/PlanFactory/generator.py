import os
import re
import argparse
import sys
import traceback
import yaml
import random
from typing import List, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from FlashOAgents.models import OpenAIServerModel, LocalTransformersModel, ChatMessage, MessageRole
except ImportError:
    OpenAIServerModel = None
    LocalTransformersModel = None
    ChatMessage = None
    MessageRole = None
    print("Warning: Could not import FlashOAgents.models. Please ensure you are in the project root.")


def load_prompt_templates() -> Dict[str, Any]:
    """
    Load prompt templates from prompts.yaml file.
    
    Returns:
        Dict containing all prompt templates
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, "prompts.yaml")
    
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            templates = yaml.safe_load(f)
        return templates
    except FileNotFoundError:
        print(f"Error: Could not find prompts.yaml at {yaml_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading prompts.yaml: {e}")
        traceback.print_exc()
        sys.exit(1)


def load_random_examples(num_examples: int = 3) -> str:
    """
    Load a random selection of planner examples to inject into the prompt.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    examples_dir = os.path.join(current_dir, "examples_plansys")

    if not os.path.exists(examples_dir):
        print(f"Warning: Examples directory {examples_dir} not found.")
        return ""

    example_files = [f for f in os.listdir(examples_dir) if f.endswith(".yaml")]
    if not example_files:
        print(f"Warning: No example YAML files found in {examples_dir}.")
        return ""

    selected_files = random.sample(example_files, min(num_examples, len(example_files)))

    examples_parts = []
    for i, file_name in enumerate(selected_files, 1):
        file_path = os.path.join(examples_dir, file_name)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                ex = yaml.safe_load(f)
                if not ex:
                    continue

                py_code = ex.get("python_code", "")
                yaml_cfg = ex.get("yaml_config", "")

                part = f"Example {i}\n"
                part += f"Python Implementation:\n{py_code}\n\n"
                part += f"YAML Configuration:\n{yaml_cfg}\n"
                part += "=" * 40 + "\n"
                examples_parts.append(part)
        except Exception as e:
            print(f"Warning: Failed to load example {file_path}: {e}")

    return "\n".join(examples_parts)


def get_tools_info_str() -> str:
    """
    Dynamically inspects FlashOAgents.tools to generate a detailed tool definition string.
    This ensures the LLM knows EXACTLY what inputs each tool expects.
    """
    try:
        from FlashOAgents import tools
        from FlashOAgents import cosight_tool
        from FlashOAgents import mm_tools

        info_parts = ["Available Tools in TodoRL:\n"]

        tool_classes = [
            tools.FinalAnswerTool,
            tools.VectorSimilarityRetrieve,
            tools.Reasoning,
            tools.Process,
            tools.EndProcess,
            tools.DeleteMemory,
            tools.VoteTool,
            tools.EnsembleTool,
            tools.Executor,
            tools.Refine,
            tools.UpdatePlanStatus,
            tools.CheckPlanProgress,
            mm_tools.VisualInspectorTool,
            mm_tools.AudioInspectorTool,
            mm_tools.TextInspectorTool,
            cosight_tool.ExpertParallelTool,
            cosight_tool.CAMVTool
        ]

        for i, cls in enumerate(tool_classes, 1):
            name = getattr(cls, "name", cls.__name__)
            desc = getattr(cls, "description", "No description provided.")
            inputs = getattr(cls, "inputs", {})

            info_parts.append(f"{i}. {name}")
            info_parts.append(f"   Description: {desc}")
            info_parts.append(f"   Inputs: {inputs}") 
            info_parts.append("")

        return "\n".join(info_parts)
    except Exception as e:
        print(f"Warning: Could not dynamically load tools: {e}")
        return """
        Available Tools in Flash-Searcher:
        (Dynamic load failed, using fallback list)
        1. FinalAnswerTool
        2. VectorSimilarityRetrieve
        3. Reasoning
        4. Process
        5. EndProcess
        6. DeleteMemory
        7. VoteTool
        8. EnsembleTool
        9. Executor
        10. Refine
        11. UpdatePlanStatus
        12. CheckPlanProgress
        13. ExpertParallelTool
        14. CAMVTool
        15. VisualInspectorTool
        16. AudioInspectorTool
        17. TextInspectorTool
        """


def render_system_prompt(template_str: str, *, tools_info: str, examples_str: str) -> str:
    """
    Render system prompt by replacing placeholders with tools info and examples.
    
    Args:
        template_str: The system prompt template string
        tools_info: The dynamically generated tools information
        examples_str: The randomly selected examples
        
    Returns:
        Rendered system prompt
    """
    rendered = template_str.replace("{TOOLS_INFO_PLACEHOLDER}", tools_info)
    rendered = rendered.replace("{EXAMPLES_PLACEHOLDER}", examples_str)
    return rendered


def generate_framework(
    task_description: str, 
    model_id: str, 
    api_key: str, 
    base_url: str, 
    output_name: str = "planner",
    planning_dir_name: str = "planning", 
    prompts_dir_name: str = "prompts"
):
    """
    Generate specialized planning module and YAML config for the given task.
    """
    # Use absolute paths relative to this file to find templates and examples
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    print(f"[{task_description[:30]}...] Loading prompt templates...")
    templates = load_prompt_templates()
    
    print(f"[{task_description[:30]}...] Loading tools information...")
    tools_info = get_tools_info_str()
    
    print(f"[{task_description[:30]}...] Loading random examples...")
    examples_str = load_random_examples(3)
    
    system_prompt = render_system_prompt(
        templates["system_prompt_template"],
        tools_info=tools_info,
        examples_str=examples_str,
    )

    user_message = templates["user_message_template"].format(
        task_description=task_description
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    try:
        if OpenAIServerModel is None:
            raise ImportError("Model classes are not available.")

        if model_id.startswith("local:") or os.path.isdir(model_id):
            print(f"[{task_description[:30]}...] 手动选择：加载本地模型进行生成 {model_id}")
            model = LocalTransformersModel(model_id=model_id)
        else:
            print(f"[{task_description[:30]}...] 手动选择：调用 API 模型进行生成 {model_id}")
            model = OpenAIServerModel(
                model_id=model_id,
                api_key=api_key,
                api_base=base_url,
                max_tokens=4000
            )
        response = model(messages)
        content = response.content
    except Exception as e:
        print(f"Model call failed: {e}")
        return False, "", "", [], "", ""

    python_match = re.search(r"<<<PYTHON>>>([\s\S]*?)<<<END_PYTHON>>>", content)
    yaml_match = re.search(r"<<<YAML>>>([\s\S]*?)<<<END_YAML>>>", content)

    think_match = re.search(r"^([\s\S]*?)(?=<<<PYTHON>>>|<<<YAML>>>)", content)
    generator_think = think_match.group(1).strip() if think_match else ""

    if not python_match or not yaml_match:
        print("Error: Could not find strictly formatted Python or YAML in response.")
        return False, "", "", messages, content, generator_think

    python_code = python_match.group(1).strip()
    yaml_code = yaml_match.group(1).strip()

    # Basic YAML validation before saving
    try:
        yaml.safe_load(yaml_code)
    except Exception as e:
        print(f"Error: Generated YAML is invalid: {e}")
        return False, "", "", messages, content, generator_think

    plan_dir = os.path.join(current_dir, planning_dir_name)
    
    prompt_dir = os.path.join(current_dir, prompts_dir_name, output_name)

    try:
        os.makedirs(plan_dir, exist_ok=True)
        os.makedirs(prompt_dir, exist_ok=True)

        py_path = os.path.join(plan_dir, f"{output_name}.py")
        yaml_path = os.path.join(prompt_dir, "toolcalling_agent.yaml")

        with open(py_path, "w", encoding="utf-8") as f:
            f.write(python_code)

        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(yaml_code)

        print(f"[{task_description[:30]}...] [SUCCESS] Generated {output_name}.py and YAML config.")
        return True, python_code, yaml_code, messages, content, generator_think

    except Exception as e:
        print(f"Error saving files: {e}")
        return False, "", "", messages, content, generator_think


def main():
    parser = argparse.ArgumentParser(
        description="Generate Agent Framework Code from task description"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("PLANNING_MODEL", "gemini-3-flash-preview"),
        help="Model ID to use for generation"
    )

    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    if not base_url:
        print("Error: OPENAI_BASE_URL environment variable not set.")
        sys.exit(1)


    task_input = args.task

    generate_framework(task_input, args.model, api_key, base_url)


if __name__ == "__main__":
    main()