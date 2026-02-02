"""Parallel orchestration extensions for MainAgent."""

from aorchestra.parallel.parallel_main_agent import ParallelMainAgent
from aorchestra.parallel.parallel_prompt import append_parallel_delegate_instructions
from aorchestra.parallel.parallel_utils import (
    build_delegate_error_result,
    execute_parallel_delegates,
    parse_multi_action_response,
    summarize_parallel_delegate_results,
)

__all__ = [
    "ParallelMainAgent",
    "append_parallel_delegate_instructions",
    "parse_multi_action_response",
    "execute_parallel_delegates",
    "summarize_parallel_delegate_results",
    "build_delegate_error_result",
]
