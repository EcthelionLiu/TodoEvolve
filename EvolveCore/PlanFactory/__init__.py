"""
Planner Factory

根据特定的任务描述自动生成、测试并运行定制化的 Agent 规划逻辑。

主要功能：
---------
1. Generator: 基于元提示词(prompts.yaml) 和动态示例库(examples/)
   通过 LLM 自动生成 `planner.py` 代码和 `toolcalling_agent.yaml` 配置文件。
2. Dynamic Few-shot: 每次生成时从范例库中随机抽取 3 个成熟系统作为范例
3. Runner: 提供基础 Agent 类和批量运行脚本，可立即验证生成的规划器效果。

主要文件：
------------
- `generator.py`: 调用 LLM 进行代码和提示词生成。
- `prompts.yaml`: 规划器生成的“元指令”模板。
- `examples/`: 存放 7 个成熟 Planning System 范例的目录。
- `base_agent.py`: 生成系统的基础包装类，已配置好所有核心工具。
- `run_factory.py`: 批量测试运行入口，支持 JSONL 格式输入和并发执行。

使用流程：
---------
  python Planner_factory/run_factory.py --infile data/example.jsonl --outfile output/factory_output.jsonl
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from .generator import generate_framework
from .base_agent import SearchAgent, PlannerBaseAgent

__all__ = [
    "generate_framework",
    "SearchAgent",
    "PlannerBaseAgent",
]
