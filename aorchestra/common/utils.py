"""公共工具函数"""
from __future__ import annotations

import json
from typing import Any, Dict, List


def parse_json_response(resp: str) -> Dict[str, Any]:
    """解析可能包含 markdown 代码块的 JSON 响应
    
    Args:
        resp: LLM 响应字符串，可能被 ```json 包裹
        
    Returns:
        解析后的 dict
    """
    s = resp.strip()
    if s.startswith("```"):
        lines = s.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines)
    return json.loads(s)


def indent_text(text: str, indent: str = "   ") -> str:
    """给文本的每一行添加缩进
    
    Args:
        text: 原始文本
        indent: 缩进字符串
        
    Returns:
        缩进后的文本
    """
    return "\n".join(indent + line for line in text.strip().split("\n"))


def format_tools_description(tools: List[Any], verbose: bool = False) -> str:
    """生成工具描述文本供 prompt 使用
    
    Args:
        tools: 工具列表，每个工具需要有 name, description, parameters 属性
        verbose: 是否使用详细格式
        
    Returns:
        格式化的工具描述字符串
    """
    if not tools:
        return "No tools available."
    
    if verbose:
        descriptions = []
        for tool in tools:
            desc = f"""Tool Name: {tool.name}
Description: {tool.description}
Parameters: {json.dumps(tool.parameters, indent=2)}"""
            descriptions.append(desc)
        return "\n\n".join(descriptions)
    else:
        return "\n\n".join([
            f"{t.name}: {t.description}\nParams: {json.dumps(t.parameters, indent=2)}"
            for t in tools
        ])
