import os
import json
import shlex
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from pydantic import Field

from base.agent.base_agent import BaseAgent
from base.agent.memory import Memory
from base.engine.logs import logger, LogLevel
from benchmark.common.env import BasicInfo, Observation, Action

# Default allowed tools for Claude CLI
ALLOWED_TOOLS = [
    "Bash", "Edit", "Write", "Read", "Glob", "Grep", "LS",
    "WebFetch", "NotebookEdit", "NotebookRead", "TodoRead",
    "TodoWrite", "Agent", "Task", "WebSearch",
]


class ClaudeCodeAgent(BaseAgent):
    """
    An agent that uses the Claude CLI for task execution.

    A single step() call runs the complete CLI execution.
    Internal toolcall -> observation loops are handled by Claude CLI itself,
    and tracked via Memory by parsing session events.
    """

    name: str = Field(default="ClaudeCodeAgent")
    description: str = Field(default="An agent that leverages Claude CLI for task execution.")

    # Configuration
    model_name: Optional[str] = Field(default=None, description="Model to use with Claude CLI")
    max_thinking_tokens: Optional[int] = Field(default=None, description="Max thinking tokens")
    allowed_tools: List[str] = Field(default_factory=lambda: ALLOWED_TOOLS.copy())

    # Execution state
    current_env_instruction: str = Field(default="")
    current_action_space: str = Field(default="")
    logs_dir: Path = Field(default_factory=lambda: Path("./workspace/logs/claude_code"))
    sessions_dir: Optional[Path] = Field(default=None)

    # Memory for trajectory tracking
    memory: Optional[Memory] = Field(default=None)

    # Usage metrics (parsed from session)
    usage_metrics: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        if self.sessions_dir is None:
            self.sessions_dir = self.logs_dir / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def reset(self, env_info: BasicInfo) -> None:
        """Reset the agent state for a new task."""
        self.memory = Memory(llm=self.llm, max_memory=100)
        self.current_env_instruction = env_info.instruction
        self.current_action_space = env_info.action_space
        self.usage_metrics = {}
        self.memory.clear()

    def _build_environment_vars(self) -> Dict[str, str]:
        """Build environment variables for Claude CLI execution."""
        env = {
            "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", ""),
            "ANTHROPIC_BASE_URL": os.environ.get("ANTHROPIC_BASE_URL", ""),
            "CLAUDE_CODE_OAUTH_TOKEN": os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", ""),
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": os.environ.get("CLAUDE_CODE_MAX_OUTPUT_TOKENS", ""),
            "FORCE_AUTO_BACKGROUND_TASKS": "1",
            "ENABLE_BACKGROUND_TASKS": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }

        # Remove empty values
        env = {k: v for k, v in env.items() if v}

        # Handle model name
        if self.model_name:
            if "ANTHROPIC_BASE_URL" in env:
                env["ANTHROPIC_MODEL"] = self.model_name
            else:
                env["ANTHROPIC_MODEL"] = self.model_name.split("/")[-1]
        elif "ANTHROPIC_MODEL" in os.environ:
            env["ANTHROPIC_MODEL"] = os.environ["ANTHROPIC_MODEL"]

        # When using custom base URL, set all model aliases
        if "ANTHROPIC_BASE_URL" in env and "ANTHROPIC_MODEL" in env:
            env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = env["ANTHROPIC_MODEL"]
            env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = env["ANTHROPIC_MODEL"]
            env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = env["ANTHROPIC_MODEL"]
            env["CLAUDE_CODE_SUBAGENT_MODEL"] = env["ANTHROPIC_MODEL"]

        # Set max thinking tokens
        if self.max_thinking_tokens is not None:
            env["MAX_THINKING_TOKENS"] = str(self.max_thinking_tokens)
        elif "MAX_THINKING_TOKENS" in os.environ:
            env["MAX_THINKING_TOKENS"] = os.environ["MAX_THINKING_TOKENS"]

        # Set config directory
        env["CLAUDE_CONFIG_DIR"] = str(self.sessions_dir)

        return env

    def _create_run_commands(self, instruction: str) -> List[Dict[str, Any]]:
        """Create the commands to run Claude CLI."""
        escaped_instruction = shlex.quote(instruction)
        env = self._build_environment_vars()

        output_file = self.logs_dir / "claude-code.txt"

        # Find Claude CLI executable - check multiple possible locations
        possible_paths = [
            os.path.expanduser("~/.local/bin/claude"),  # npm global install location
            os.path.expanduser("~/.claude/local/node_modules/.bin/claude"),  # legacy location
        ]
        claude_bin = "claude"  # fallback to PATH
        for path in possible_paths:
            if os.path.exists(path):
                claude_bin = path
                break

        # Setup command: create necessary directories
        setup_cmd = (
            "mkdir -p $CLAUDE_CONFIG_DIR/debug $CLAUDE_CONFIG_DIR/projects/-app "
            "$CLAUDE_CONFIG_DIR/shell-snapshots $CLAUDE_CONFIG_DIR/statsig "
            "$CLAUDE_CONFIG_DIR/todos && "
            "if [ -d ~/.claude/skills ]; then "
            "cp -r ~/.claude/skills $CLAUDE_CONFIG_DIR/skills 2>/dev/null || true; "
            "fi"
        )

        # Run command: execute Claude CLI
        allowed_tools_str = " ".join(self.allowed_tools)
        run_cmd = (
            f"{claude_bin} --dangerously-skip-permissions "
            f"--verbose --output-format stream-json "
            f"-p {escaped_instruction} --allowedTools "
            f"{allowed_tools_str} 2>&1 </dev/null | tee "
            f"{output_file}"
        )

        return [
            {"command": setup_cmd, "env": env},
            {"command": run_cmd, "env": env},
        ]

    async def _execute_command(self, command: str, env: Dict[str, str],
                               timeout_sec: Optional[int] = None) -> Tuple[int, str, str]:
        """Execute a shell command asynchronously."""
        full_env = os.environ.copy()
        full_env.update(env)

        process = None
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=full_env,
            )

            if timeout_sec:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout_sec
                )
            else:
                stdout, stderr = await process.communicate()

            return (
                process.returncode or 0,
                stdout.decode("utf-8", errors="replace"),
                stderr.decode("utf-8", errors="replace"),
            )
        except asyncio.TimeoutError:
            if process is not None:
                try:
                    process.kill()
                    await process.wait()
                except ProcessLookupError:
                    pass
            return -1, "", "Command timed out"
        except Exception as e:
            if process is not None:
                try:
                    process.kill()
                    await process.wait()
                except (ProcessLookupError, OSError):
                    pass
            return -1, "", str(e)

    def _get_session_dir(self) -> Optional[Path]:
        """Identify the Claude session directory."""
        if not self.sessions_dir or not self.sessions_dir.exists():
            return None

        project_root = self.sessions_dir / "projects"
        if not project_root.exists():
            return None

        candidate_files = list(project_root.glob("**/*.jsonl"))
        if not candidate_files:
            return None

        candidate_dirs = sorted({f.parent for f in candidate_files})
        if not candidate_dirs:
            return None

        if len(candidate_dirs) > 1:
            logger.warning("Multiple Claude Code session directories found")
        return candidate_dirs[0]

    async def _parse_session_to_memory(self, session_dir: Path) -> Dict[str, Any]:
        """
        Parse Claude session events and record them to Memory.
        Returns usage metrics.
        """
        session_files = list(session_dir.glob("*.jsonl"))

        if not session_files:
            logger.warning(f"No Claude Code session files found in {session_dir}")
            return {}

        total_input_tokens = 0
        total_output_tokens = 0
        total_cached_tokens = 0
        model_name = self.model_name
        step_count = 0

        # Log initial user instruction as Step 1
        if self.current_env_instruction:
            step_count += 1
            logger.info(f"[Step {step_count}] User: {self.current_env_instruction}")

        for session_file in session_files:
            try:
                with open(session_file, "r") as handle:
                    for line in handle:
                        stripped = line.strip()
                        if not stripped:
                            continue
                        try:
                            event = json.loads(stripped)
                            message = event.get("message")
                            if not isinstance(message, dict):
                                continue

                            event_type = event.get("type")

                            # Extract model name
                            if not model_name and isinstance(message.get("model"), str):
                                model_name = message.get("model")

                            # Extract usage (handle None and ensure we get actual numbers)
                            usage = message.get("usage")
                            if isinstance(usage, dict):
                                input_tokens = usage.get("input_tokens")
                                output_tokens = usage.get("output_tokens")
                                cache_tokens = usage.get("cache_read_input_tokens")
                                
                                if isinstance(input_tokens, (int, float)):
                                    total_input_tokens += int(input_tokens)
                                if isinstance(output_tokens, (int, float)):
                                    total_output_tokens += int(output_tokens)
                                if isinstance(cache_tokens, (int, float)):
                                    total_cached_tokens += int(cache_tokens)

                            # Record to memory: track toolcalls and observations
                            if self.memory and event_type in ("assistant", "user"):
                                content = message.get("content")

                                # Extract tool calls from assistant messages
                                if event_type == "assistant":
                                    tool_calls = []
                                    text_parts = []
                                    thinking = None

                                    # Handle both string and list content
                                    if isinstance(content, str):
                                        text_parts.append(content)
                                    elif isinstance(content, list):
                                        for block in content:
                                            if isinstance(block, dict):
                                                block_type = block.get("type")
                                                if block_type == "tool_use":
                                                    tool_calls.append({
                                                        "id": block.get("id"),
                                                        "name": block.get("name"),
                                                        "input": block.get("input"),
                                                    })
                                                elif block_type in ("thinking", "reasoning"):
                                                    thinking = block.get("text", "")
                                                elif block_type == "text":
                                                    text_parts.append(block.get("text", ""))

                                    # Only record if there's meaningful content
                                    if tool_calls or text_parts or thinking:
                                        step_count += 1
                                        action_info = {
                                            "action": "assistant",
                                            "step": step_count,
                                        }

                                        if text_parts:
                                            action_info["text"] = "\n".join(text_parts)
                                            preview = text_parts[0].replace('\n', ' ').strip()[:100]
                                            logger.info(f"[Step {step_count}] Text: {preview}...")

                                        if tool_calls:
                                            action_info["tool_calls"] = tool_calls
                                            for tc in tool_calls:
                                                logger.info(f"[Step {step_count}] Tool: {tc.get('name')} | Input: {tc.get('input')}")

                                        if thinking and not text_parts and not tool_calls:
                                            preview = thinking.replace('\n', ' ').strip()[:100]
                                            logger.info(f"[Step {step_count}] Thinking: {preview}...")

                                        obs = {"type": "assistant_response"}

                                        await self.memory.add_memory(
                                            obs=obs,
                                            action=action_info,
                                            thinking=thinking,
                                        )

                                # Extract tool results from user messages
                                elif event_type == "user":
                                    tool_results = []
                                    text_parts = []

                                    if isinstance(content, str):
                                        text_parts.append(content)
                                    elif isinstance(content, list):
                                        for block in content:
                                            if isinstance(block, dict):
                                                if block.get("type") == "tool_result":
                                                    tool_results.append({
                                                        "tool_use_id": block.get("tool_use_id"),
                                                        "content": block.get("content"),
                                                        "is_error": block.get("is_error", False),
                                                    })
                                                elif block.get("type") == "text":
                                                    text_parts.append(block.get("text", ""))

                                    # Only record if there's meaningful content
                                    if tool_results or text_parts:
                                        step_count += 1
                                        action_info = {
                                            "action": "user",
                                            "step": step_count,
                                        }

                                        if tool_results:
                                            action_info["tool_results"] = tool_results
                                            # Log tool results
                                            for tr in tool_results:
                                                result_preview = str(tr.get('content', ''))[:100]
                                                logger.info(f"[Step {step_count}] Tool Result: {result_preview}...")
                                        if text_parts:
                                            action_info["text"] = "\n".join(text_parts)

                                        obs = {"type": "tool_observation" if tool_results else "user_message"}
                                        await self.memory.add_memory(
                                            obs=obs,
                                            action=action_info,
                                        )

                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                logger.warning(f"Failed to read session file {session_file}: {e}")

        logger.info(f"Parsed {step_count} events from Claude session to memory")

        return {
            "model": model_name or "",
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_cached_tokens": total_cached_tokens,
            "total_cost": 0.0,
            "total_steps": step_count,
        }

    def _get_memory(self) -> str:
        """Get memory as text for prompting."""
        if self.memory:
            return self.memory.as_text()
        return "None"

    def _extract_final_result(self, session_dir: Path) -> Optional[str]:
        """
        从 Claude session 中提取最终结果。
        查找 "finish" 工具调用或最后的文本输出。
        """
        session_files = list(session_dir.glob("*.jsonl")) if session_dir else []
        
        final_result = None
        last_text = None
        
        for session_file in session_files:
            try:
                with open(session_file, "r") as handle:
                    for line in handle:
                        stripped = line.strip()
                        if not stripped:
                            continue
                        try:
                            event = json.loads(stripped)
                            message = event.get("message")
                            if not isinstance(message, dict):
                                continue
                            
                            event_type = event.get("type")
                            content = message.get("content")
                            
                            if event_type == "assistant" and isinstance(content, list):
                                for block in content:
                                    if isinstance(block, dict):
                                        # 检查是否调用了 finish 工具
                                        if block.get("type") == "tool_use" and block.get("name") == "finish":
                                            input_data = block.get("input", {})
                                            if isinstance(input_data, dict):
                                                final_result = input_data.get("result", "")
                                        # 记录最后的文本输出
                                        elif block.get("type") == "text":
                                            last_text = block.get("text", "")
                                            
                        except json.JSONDecodeError:
                            continue
            except Exception:
                continue
        
        # 优先返回 finish 工具的结果，否则返回最后的文本
        return final_result if final_result is not None else last_text

    def parse_action(self, resp: str, session_dir: Path = None) -> Action:
        """Parse response to extract action data."""
        # 尝试从 session 中提取最终结果
        final_result = self._extract_final_result(session_dir) if session_dir else None
        
        if final_result is not None:
            # 返回 finish 动作，让环境知道任务完成
            return {
                "action": "finish",
                "params": {
                    "result": final_result,
                    "status": "done",
                    "summary": "Claude CLI completed the task"
                }
            }
        
        # 如果没有找到结果，也返回 finish（避免无限循环）
        return {
            "action": "finish",
            "params": {
                "result": "",
                "status": "partial",
                "summary": "Claude CLI session completed without explicit result"
            }
        }

    async def step(self, observation: Observation, history: Any = None, **kwargs) -> Tuple[Action, str, str]:
        """
        Execute a single step using Claude CLI.

        For ClaudeCodeAgent, a single step runs the complete CLI execution.
        Internal toolcall -> observation loops are handled by Claude CLI itself
        and tracked via Memory.
        """
        instruction = str(observation.get("instruction", self.current_env_instruction))

        # Create and execute commands
        commands = self._create_run_commands(instruction)

        all_output = []
        for cmd_info in commands:
            command = cmd_info["command"]
            env = cmd_info.get("env", {})

            logger.log_to_file(LogLevel.INFO, f"Executing: {command[:100]}...")

            return_code, stdout, stderr = await self._execute_command(command, env)

            output = {
                "command": command,
                "return_code": return_code,
                "stdout": stdout,
                "stderr": stderr,
            }
            all_output.append(output)

            if return_code != 0:
                logger.warning(f"Command failed with return code {return_code}: {stderr}")

        # Parse session events to memory and get usage metrics
        session_dir = self._get_session_dir()
        if session_dir:
            self.usage_metrics = await self._parse_session_to_memory(session_dir)
            logger.info(f"Parsed usage: {self.usage_metrics.get('total_input_tokens', 0)} input, "
                        f"{self.usage_metrics.get('total_output_tokens', 0)} output tokens, "
                        f"{self.usage_metrics.get('total_steps', 0)} steps")

        # Build response
        resp = json.dumps(all_output, ensure_ascii=False, indent=2)
        action = self.parse_action(resp, session_dir)

        return action, resp, instruction

    async def run(self, request: Optional[str] = None) -> str:
        """Execute the agent's main loop."""
        instruction = request or self.current_env_instruction

        if not instruction:
            return json.dumps({"error": "No instruction provided"})

        # Set current instruction for logging
        self.current_env_instruction = instruction

        # Initialize memory for trajectory tracking
        if not self.memory:
            self.memory = Memory(llm=self.llm, max_memory=100)

        observation: Observation = {"instruction": instruction}
        action, resp, _ = await self.step(observation)

        return resp
