"""
E2B Claude Code Agent - runs Claude CLI inside E2B sandbox
"""
import base64
import json
import shlex
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import Field

from base.agent.base_agent import BaseAgent
from base.agent.memory import Memory
from base.engine.logs import logger
from benchmark.common.env import BasicInfo, Observation, Action

# Load model config
MODEL_CONFIG_PATH = Path(__file__).parent.parent / "config" / "model_config.yaml"

def _load_model_config(model_name: str) -> Dict[str, Any]:
    """Load model configuration from model_config.yaml"""
    if not MODEL_CONFIG_PATH.exists():
        return {}
    try:
        with open(MODEL_CONFIG_PATH) as f:
            config = yaml.safe_load(f)
        return config.get("models", {}).get(model_name, {})
    except Exception as e:
        logger.warning(f"Failed to load model config: {e}")
        return {}

# Default allowed tools for Claude CLI
ALLOWED_TOOLS = [
    "Bash", "Edit", "Write", "Read", "Glob", "Grep", "LS",
    "WebFetch", "NotebookEdit", "NotebookRead", "TodoRead",
    "TodoWrite", "Agent", "Task", "WebSearch",
]


class E2BClaudeCodeAgent(BaseAgent):
    """
    Agent that runs Claude CLI inside E2B sandbox.
    
    Unlike ClaudeCodeAgent (which runs locally), this agent executes
    claude commands inside the E2B container via executor.execute_command().
    """

    name: str = Field(default="E2BClaudeCodeAgent")
    description: str = Field(default="Claude Code agent running in E2B sandbox")

    # Configuration
    model_name: Optional[str] = Field(default=None)
    max_thinking_tokens: Optional[int] = Field(default=None)
    allowed_tools: List[str] = Field(default_factory=lambda: ALLOWED_TOOLS.copy())

    # Execution state
    current_env_instruction: str = Field(default="")
    executor: Any = Field(default=None, exclude=True)  # E2B executor instance
    
    # Logs
    logs_dir: Path = Field(default_factory=lambda: Path("./workspace/logs/e2b_claude_code"))
    
    # Memory and metrics
    memory: Optional[Memory] = Field(default=None)
    usage_metrics: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, executor=None, **data):
        # If allowed_tools is None, use default ALLOWED_TOOLS
        if "allowed_tools" not in data or data.get("allowed_tools") is None:
            data["allowed_tools"] = ALLOWED_TOOLS.copy()
        super().__init__(**data)
        self.executor = executor
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def reset(self, env_info: BasicInfo) -> None:
        """Reset agent state for new task."""
        # E2BClaudeCodeAgent 不需要 Memory（Claude CLI 自己管理状态）
        self.memory = None
        self.current_env_instruction = env_info.instruction
        self.usage_metrics = {}

    def _build_claude_command(self, instruction: str) -> str:
        """Build claude CLI command to run in E2B."""
        import os
        import uuid
        
        # Use a temporary file to pass the prompt (avoids all shell quoting issues)
        prompt_file = f"/tmp/claude_prompt_{uuid.uuid4().hex[:8]}.txt"
        
        # Build environment variable exports
        env_vars = []
        
        # Load model config from model_config.yaml (priority over env vars)
        model_config = _load_model_config(self.model_name) if self.model_name else {}
        
        # Get API credentials from model config or env vars
        api_key = model_config.get("api_key") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")
        base_url = model_config.get("base_url") or os.getenv("ANTHROPIC_BASE_URL")
        
        if base_url:
            # Custom base URL (e.g., OpenRouter, DeepWisdom) - use AUTH_TOKEN
            # OpenRouter integration: remove /v1 suffix for claude CLI
            if base_url.endswith("/v1"):
                base_url = base_url[:-3]  # Remove /v1 suffix
            
            if api_key:
                env_vars.append(f'export ANTHROPIC_AUTH_TOKEN="{api_key}"')
                env_vars.append('export ANTHROPIC_API_KEY=""')  # Blank to avoid conflicts
            
            env_vars.append(f'export ANTHROPIC_BASE_URL="{base_url}"')
            logger.info(f"[E2BClaudeCode] Using custom API base: {base_url}")
        else:
            # Official Anthropic API - use API_KEY
            if api_key:
                env_vars.append(f'export ANTHROPIC_API_KEY="{api_key}"')
        
        # Model configuration
        if self.model_name:
            env_vars.append(f'export ANTHROPIC_MODEL="{self.model_name}"')
            # Model aliases for Claude CLI
            env_vars.append(f'export ANTHROPIC_DEFAULT_SONNET_MODEL="{self.model_name}"')
            env_vars.append(f'export ANTHROPIC_DEFAULT_OPUS_MODEL="{self.model_name}"')
            env_vars.append(f'export ANTHROPIC_DEFAULT_HAIKU_MODEL="{self.model_name}"')
        
        # Other settings
        if self.max_thinking_tokens is not None:
            env_vars.append(f'export MAX_THINKING_TOKENS="{self.max_thinking_tokens}"')
        
        env_vars.append('export FORCE_AUTO_BACKGROUND_TASKS="1"')
        env_vars.append('export ENABLE_BACKGROUND_TASKS="1"')
        env_vars.append('export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1"')
        
        # Ensure PATH includes Claude CLI location
        env_vars.append('export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"')
        
        # Create session directory in E2B
        env_vars.append('export CLAUDE_CONFIG_DIR="/tmp/claude_sessions"')
        env_vars.append('mkdir -p $CLAUDE_CONFIG_DIR/debug $CLAUDE_CONFIG_DIR/projects/-app')
        
        # Build full command
        env_setup = " && ".join(env_vars)
        
        # Build claude command with -p parameter
        # We still use temp file to avoid shell escaping issues, but pass via -p
        write_prompt = f"printf %s {shlex.quote(instruction)} > {prompt_file}"
        
        # Use -p with command substitution to read from file
        # --dangerously-skip-permissions is required for non-interactive tool execution
        claude_cmd = "claude --dangerously-skip-permissions --verbose --output-format stream-json"
        if self.allowed_tools:
            allowed_tools_str = " ".join(self.allowed_tools)
            claude_cmd += f" --allowedTools {allowed_tools_str}"
        claude_cmd += f" -p \"$(cat {prompt_file})\""
        
        cleanup = f"rm -f {prompt_file}"
        
        # Combine: setup env -> write prompt -> run claude -> cleanup
        full_payload = f"{env_setup} && {write_prompt} && {claude_cmd}; {cleanup}"
        full_command = "bash -lc " + shlex.quote(full_payload)
        return full_command

    async def step(self, observation: Observation, history: Any = None, **kwargs) -> Tuple[Action, str, str]:
        """Execute Claude CLI inside E2B sandbox."""
        if not self.executor:
            raise RuntimeError("E2B executor not set. Pass executor to constructor.")
        
        instruction = str(observation.get("instruction", self.current_env_instruction))
        logger.info(f"[E2BClaudeCode] Executing task in E2B sandbox...")
        
        # Build and execute command in E2B
        command = self._build_claude_command(instruction)
        
        try:
            output, exit_code = await self.executor.execute_command(command, timeout=600)
            
            logger.info(f"[E2BClaudeCode] Command completed with exit_code={exit_code}")
            logger.info(f"[E2BClaudeCode] Output length: {len(output) if output else 0} chars")
            if output:
                logger.info(f"[E2BClaudeCode] Full output:\n{output}")
            
            # Parse session if available (session is in E2B at /tmp/claude_sessions)
            # For now, just return based on exit code
            if exit_code == 0:
                action = {
                    "action": "finish",
                    "params": {
                        "result": output[:1000] if output else "",
                        "status": "done",
                        "summary": "Claude Code completed in E2B"
                    }
                }
            else:
                action = {
                    "action": "finish",
                    "params": {
                        "result": "",
                        "status": "error",
                        "summary": f"Claude Code failed with exit_code={exit_code}"
                    }
                }
            
            resp = json.dumps({
                "command": command[:200],
                "exit_code": exit_code,
                "output": output[:1000] if output else "",
            }, ensure_ascii=False)
            
            # Update usage metrics (simplified - would need to parse session logs for accurate data)
            self.usage_metrics = {
                "model": self.model_name or "unknown",
                "total_steps": 1,
                "total_cost": 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
            }
            
            return action, resp, instruction
            
        except Exception as e:
            logger.error(f"[E2BClaudeCode] Error: {e}")
            action = {
                "action": "finish",
                "params": {
                    "result": "",
                    "status": "error",
                    "summary": f"Error: {str(e)}"
                }
            }
            return action, str(e), instruction

    async def run(self, request: Optional[str] = None) -> str:
        """Execute agent's main loop."""
        instruction = request or self.current_env_instruction
        if not instruction:
            return json.dumps({"error": "No instruction provided"})

        observation = {"instruction": instruction}
        action, resp, _ = await self.step(observation)
        
        return json.dumps({
            "action": action,
            "response": resp,
            "usage": self.usage_metrics,
        }, ensure_ascii=False)
