"""
向后兼容的导入代理

OrchestraSubAgent 已移动到 aorchestra.subagents.react_agent.ReActAgent
此文件保持向后兼容性。
"""
from aorchestra.subagents.react_agent import ReActAgent

# 向后兼容别名
OrchestraSubAgent = ReActAgent

__all__ = ["OrchestraSubAgent", "ReActAgent"]
