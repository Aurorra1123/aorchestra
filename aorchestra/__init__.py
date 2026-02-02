"""
aorchestra - 统一的 Orchestra 框架

支持 GAIA、TerminalBench 和 SWE-bench 三种 benchmark。
"""
from aorchestra.subagents import ReActAgent, SWEBenchSubAgent
from aorchestra.sub_agent import OrchestraSubAgent  # 向后兼容
from aorchestra.config import GAIAOrchestraConfig, TerminalBenchOrchestraConfig, SWEBenchOrchestraConfig

__all__ = [
    # SubAgents
    "ReActAgent",
    "SWEBenchSubAgent",
    "OrchestraSubAgent",  # 向后兼容别名
    # Configs
    "GAIAOrchestraConfig",
    "TerminalBenchOrchestraConfig",
    "SWEBenchOrchestraConfig",
]
