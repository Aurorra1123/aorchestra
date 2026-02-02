"""
SWE-Agent for FoundationAgent.

This module implements a SWE-Agent with proper ACI (Agent-Computer Interface) tools,
structured prompts, and few-shot demonstrations following the real SWE-Agent design.
"""
import os
import re
from typing import Any, Dict, Optional

from pydantic import Field

from base.agent.base_agent import BaseAgent
from base.agent.memory import Memory
from base.engine.async_llm import AsyncLLM
from base.engine.logs import logger, LogLevel
from base.engine.utils import parse_llm_action_response
from benchmark.common.env import BasicInfo, Observation, Action

# Suppress mini-swe-agent startup message
os.environ["MSWEA_SILENT_STARTUP"] = "1"


# =============================================================================
# SYSTEM TEMPLATE - Core instructions for the agent
# =============================================================================
SYSTEM_TEMPLATE = """SETTING: You are an autonomous software engineering agent tasked with solving GitHub issues.
You have access to a specialized command interface (ACI) for navigating, viewing, editing, and testing code.
You will work in a Docker container with the repository already cloned and checked out to the correct commit.

COMMANDS:
{command_docs}

RESPONSE FORMAT:
Your response must contain exactly two sections:
1. DISCUSSION: Your reasoning about what to do next
2. COMMAND: A single command to execute

Example response format:
DISCUSSION
I need to find where the bug is located. Let me search for the relevant function.
COMMAND
find_file error_handler

IMPORTANT RULES:
- Execute only ONE command per response
- When a command produces no output, you'll see: "Command ran successfully. No output."
- File contents are displayed with line numbers: 1| def foo():
- The edit command performs automatic syntax checking for Python files
- Always verify your changes work by running relevant tests
- Use the submit action when you're confident the fix is complete

ENVIRONMENT:
- Repository location: /testbed
- To run Python/pytest, activate conda first: source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed
- Basic shell commands (find, grep, cat, sed) do NOT require conda activation
"""


# =============================================================================
# FEW-SHOT DEMONSTRATION - Shows the complete workflow
# =============================================================================
DEMONSTRATION_TEMPLATE = """
--- DEMONSTRATION ---
Below is a demonstration of how to approach a typical bug fix task.
This shows the workflow: Explore -> Locate -> Edit -> Test -> Debug

[Issue: The `calculate_discount()` function returns incorrect values for percentages over 100%]

DISCUSSION
First, I need to find where the calculate_discount function is defined. Let me search for files containing this function.
COMMAND
find_file calculate

OUTPUT
Found 3 files:
src/pricing/calculator.py
src/utils/discount_utils.py
tests/test_calculator.py

DISCUSSION
The discount function is likely in discount_utils.py. Let me open it to examine the implementation.
COMMAND
open src/utils/discount_utils.py

OUTPUT
[File: src/utils/discount_utils.py (45 lines total)]
1| \"\"\"Discount calculation utilities.\"\"\"
2| 
3| def calculate_discount(price: float, percentage: float) -> float:
4|     \"\"\"
5|     Calculate discounted price.
6|     
7|     Args:
8|         price: Original price
9|         percentage: Discount percentage (0-100)
10|     
11|     Returns:
12|         Discounted price
13|     \"\"\"
14|     # BUG: No validation for percentage > 100
15|     discount = price * (percentage / 100)
16|     return price - discount
17| 
18| 
19| def apply_bulk_discount(items: list, percentage: float) -> list:
20|     \"\"\"Apply discount to multiple items.\"\"\"
(showing lines 1-20)

DISCUSSION
I found the bug! Line 14-16 shows there's no validation for percentages over 100, which would result in negative prices. I'll use str_replace to fix this - it's more reliable than edit for precise changes.
COMMAND
str_replace src/utils/discount_utils.py
<<<<<<< SEARCH
    # BUG: No validation for percentage > 100
    discount = price * (percentage / 100)
    return price - discount
=======
    # Clamp percentage to valid range [0, 100]
    percentage = max(0, min(100, percentage))
    discount = price * (percentage / 100)
    return price - discount
>>>>>>> REPLACE

OUTPUT
Successfully replaced content in src/utils/discount_utils.py.
[Syntax check passed]

[File: src/utils/discount_utils.py (46 lines total)]
14|     # Clamp percentage to valid range [0, 100]
15|     percentage = max(0, min(100, percentage))
16|     discount = price * (percentage / 100)
17|     return price - discount
(showing lines 14-17)

DISCUSSION
The edit was successful and syntax check passed. Now I need to run the tests to verify my fix works correctly.
COMMAND
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed && cd /testbed && python -m pytest tests/test_calculator.py -v

OUTPUT
============================= test session starts =============================
collected 5 items

tests/test_calculator.py::test_normal_discount PASSED
tests/test_calculator.py::test_zero_discount PASSED
tests/test_calculator.py::test_full_discount PASSED
tests/test_calculator.py::test_over_100_discount PASSED
tests/test_calculator.py::test_negative_discount PASSED

============================= 5 passed in 0.12s ==============================

DISCUSSION
All tests pass, including the test for over 100% discount. My fix correctly handles edge cases. The issue is resolved.
COMMAND
submit

--- END OF DEMONSTRATION ---
"""


# =============================================================================
# STRATEGY TEMPLATE - Workflow guidance
# =============================================================================
STRATEGY_TEMPLATE = """
RECOMMENDED WORKFLOW:
1. EXPLORE: Use find_file and search_dir to understand the codebase structure
2. LOCATE: Open relevant files, use search_file to find specific code
3. UNDERSTAND: Read the code carefully, scroll through if needed
4. EDIT: Make targeted changes using str_replace (PREFERRED) or edit command
5. TEST: Run tests to verify your changes work
6. DEBUG: If tests fail, analyze errors and iterate

EDITING TIPS:
- PREFER str_replace over edit - it's more reliable for precise changes
- str_replace requires the SEARCH text to be unique and match exactly (including whitespace)
- If str_replace fails because text isn't unique, add more surrounding context
- For large multi-line edits, str_replace avoids command length limits
- Make small, incremental changes rather than large rewrites
- Always verify edits with syntax checking (automatic for Python)

GENERAL TIPS:
- Start by understanding the issue thoroughly before making changes
- Look for existing tests related to the issue
- Make minimal, focused changes
- Always run tests after editing
- If your edit causes syntax errors, the system will show them - fix before proceeding
"""


# =============================================================================
# INSTANCE TEMPLATE - Per-task context
# =============================================================================
INSTANCE_TEMPLATE = """
==== TASK ====
{instruction}

==== CURRENT STATE ====
{state_info}

==== RECENT HISTORY ====
{memory}

==== LAST OBSERVATION ====
{observation}

Now, analyze the situation and provide your next action.
Remember: DISCUSSION first, then COMMAND.
"""


def build_prompt(
    instruction: str,
    command_docs: str,
    state_info: str,
    memory: str,
    observation: str,
    include_demo: bool = True
) -> str:
    """
    Build the complete prompt for the agent.
    
    Args:
        instruction: The task description
        command_docs: Documentation for available commands
        state_info: Current state (open file, working directory)
        memory: Recent history/memory
        observation: Last observation from environment
        include_demo: Whether to include the demonstration
        
    Returns:
        Complete formatted prompt
    """
    parts = []
    
    # System template with command docs
    parts.append(SYSTEM_TEMPLATE.format(command_docs=command_docs))
    
    # Include demonstration for first few steps
    if include_demo:
        parts.append(DEMONSTRATION_TEMPLATE)
    
    # Strategy guidance
    parts.append(STRATEGY_TEMPLATE)
    
    # Instance-specific content
    parts.append(INSTANCE_TEMPLATE.format(
        instruction=instruction,
        state_info=state_info,
        memory=memory,
        observation=observation
    ))
    
    return '\n'.join(parts)


def parse_agent_response(response: str) -> Dict[str, Any]:
    """
    Parse agent response to extract DISCUSSION and COMMAND.
    
    Args:
        response: The raw LLM response
        
    Returns:
        Action dict with command or fallback action
    """
    # Try to find DISCUSSION and COMMAND sections
    discussion = ""
    command = ""
    
    # Pattern for DISCUSSION section
    discussion_match = re.search(
        r'DISCUSSION\s*\n(.*?)(?=\nCOMMAND|\Z)', 
        response, 
        re.DOTALL | re.IGNORECASE
    )
    if discussion_match:
        discussion = discussion_match.group(1).strip()
    
    # Pattern for COMMAND section - capture everything after COMMAND until end or next section
    command_match = re.search(
        r'COMMAND\s*\n(.*?)(?=\n(?:DISCUSSION|OUTPUT)|\Z)',
        response,
        re.DOTALL | re.IGNORECASE
    )
    if command_match:
        command = command_match.group(1).strip()
    
    # Handle submit command
    if command.lower() == 'submit' or 'submit' in command.lower():
        return {
    "action": "submit",
            "params": {},
            "reasoning": discussion
        }
    
    # Handle edit command with multi-line content
    if command.startswith('edit '):
        # The edit command includes content until end_of_edit
        return {
            "action": "aci_command",
            "params": {"command": command},
            "reasoning": discussion
        }
    
    # Regular ACI or bash command
    if command:
        return {
            "action": "aci_command",
            "params": {"command": command},
            "reasoning": discussion
        }
    
    # Fallback: try to parse as JSON (backward compatibility)
    try:
        action = parse_llm_action_response(response)
        if isinstance(action, dict) and action.get("action"):
            return action
    except Exception:
        pass
    
    # Last resort: extract any command-like content
    lines = response.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('DISCUSSION'):
            # Assume it's a command
            return {
                "action": "aci_command",
                "params": {"command": line},
                "reasoning": response
            }
    
    return {
        "action": "error",
        "params": {"message": "Could not parse response"},
        "reasoning": response
    }


class SWEAgent(BaseAgent):
    """
    SWE-Agent for solving GitHub issues.
    
    This agent uses the ACI (Agent-Computer Interface) with specialized commands
    for file navigation, editing, and searching. It follows the real SWE-Agent
    design with structured prompts and few-shot demonstrations.
    """
    name: str = Field(default="SWEAgent")
    description: str = Field(default="Software Engineering Agent with ACI tools")
    current_instruction: str = Field(default="")
    current_action_space: str = Field(default="")
    command_docs: str = Field(default="")
    state_info: str = Field(default="(Open file: n/a) (Current directory: /testbed)")
    memory: Optional[Memory] = Field(default=None)
    step_count: int = Field(default=0)
    include_demo_until_step: int = Field(default=3)  # Include demo for first N steps
    
    class Config:
        arbitrary_types_allowed = True

    def reset(self, env_info: BasicInfo) -> None:
        """Reset agent state for new task."""
        self.memory = Memory(llm=self.llm, max_memory=20)
        self.current_instruction = env_info.instruction
        self.current_action_space = env_info.action_space
        self.command_docs = env_info.meta_data.get("command_docs", self.current_action_space)
        self.state_info = "(Open file: n/a) (Current directory: /testbed)"
        self.step_count = 0
        self.memory.clear()
        logger.info(f"SWEAgent reset for task: {env_info.env_id}")

    def _get_memory(self) -> str:
        """Get formatted memory context."""
        if self.memory:
            return self.memory.as_text()
        return "No previous observations."

    def update_state(self, state_info: str) -> None:
        """Update the current state information."""
        self.state_info = state_info

    async def step(self, observation: Observation, history: Any) -> tuple[Action, str, str]:
        """Execute one step of the agent loop."""
        self.step_count += 1
        
        # Format observation
        if isinstance(observation, dict):
            obs_str = observation.get("output", str(observation))
            # Update state if provided
            if "state_info" in observation:
                self.state_info = observation["state_info"]
        else:
            obs_str = str(observation)
        
        # Build prompt
        include_demo = self.step_count <= self.include_demo_until_step
        prompt = build_prompt(
            instruction=self.current_instruction,
            command_docs=self.command_docs,
            state_info=self.state_info,
            memory=self._get_memory(),
            observation=obs_str,
            include_demo=include_demo
        )
        
        logger.log_to_file(LogLevel.INFO, f"Agent Input (step {self.step_count}):\n{prompt}\n")
        
        # Query LLM
        try:
            resp = await self.llm(prompt)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            resp = "DISCUSSION\nLLM call failed, trying a basic exploration.\nCOMMAND\nls -la /testbed"
        
        # Parse response
        action = parse_agent_response(resp)
        logger.agent_action(f"Agent Action: {action}")
        
        # Extract reasoning for memory
        reasoning = ""
        if isinstance(action, dict):
            reasoning = action.get("reasoning", "")
        
        # Update memory
        if self.memory:
            await self.memory.add_memory(
                obs=obs_str,
                action=action,
                thinking=reasoning,
                raw_response=resp
            )
        
        return action, resp, prompt

    async def run(self, request: Optional[str] = None) -> str:
        """Main run method (not used in benchmark loop)."""
        return ""


class SWEAgentFactory:
    """Factory for creating SWEAgent instances."""
    
    def __init__(self, llm: AsyncLLM):
        self.llm = llm
    
    def create(self) -> SWEAgent:
        """Create a new agent instance."""
        return SWEAgent(
            llm=self.llm,
            memory=Memory(llm=self.llm, max_memory=20)
        )


# Aliases for compatibility
MiniSWEAgent = SWEAgent
MiniSWEAgentFactory = SWEAgentFactory
ReAcTAgent = SWEAgent
