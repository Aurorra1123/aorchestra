"""SWE-bench prompts for MainAgent."""
from typing import Any, Dict, List

from base.engine.async_llm import ModelPricing


class SWEBenchMainAgentPrompt:
    """Build prompts for SWE-bench tasks (GitHub issue fixing)."""
    
    @staticmethod
    def _build_model_pricing_table(
        sub_models: List[str], 
        model_to_alias: Dict[str, str] = None
    ) -> str:
        """Generate a pricing table for available sub-models."""
        lines = ["| Model | Input $/1K | Output $/1K |"]
        lines.append("|-------|-----------|------------|")
        
        alias_to_model = {v: k for k, v in model_to_alias.items()} if model_to_alias else {}
        
        for model_display in sub_models:
            real_model = alias_to_model.get(model_display, model_display)
            input_price = ModelPricing.get_price(real_model, "input")
            output_price = ModelPricing.get_price(real_model, "output")
            lines.append(f"| {model_display} | ${input_price:.5f} | ${output_price:.5f} |")
        
        return "\n".join(lines)
    
    @staticmethod
    def build_prompt(
        instruction: str,
        meta: Dict[str, Any],
        tools_description: str,
        prior_context: str,
        attempt_index: int,
        max_attempts: int,
        sub_models: List[str],
        subtask_history: str = "",
        model_to_alias: Dict[str, str] = None,
    ) -> str:
        remaining_attempts = max_attempts - attempt_index + 1
        model_pricing_table = SWEBenchMainAgentPrompt._build_model_pricing_table(sub_models, model_to_alias)
        
        # Extract repository info from metadata
        repo = meta.get("repo", "unknown")
        instance_id = meta.get("instance_id", "unknown")
        
        # Budget warning
        if remaining_attempts <= 2:
            budget_warning = f"🚨 CRITICAL: Only {remaining_attempts} attempt(s) left! Submit now if fix is verified, or make final attempt."
        elif remaining_attempts <= 4:
            budget_warning = f"⚠️ Warning: {remaining_attempts} attempts remaining. Plan carefully."
        else:
            budget_warning = ""
        
        return f"""You are the MainAgent (Orchestrator) for a SWE-bench task. Your goal is to fix a GitHub issue by delegating work to SubAgents.


==== TASK ====
{instruction}

REPOSITORY: {repo}
INSTANCE: {instance_id}

==== DECISION PROCESS ====
1. READ the TASK carefully - understand the GitHub issue and what needs to be fixed
2. REVIEW SUBTASK HISTORY - check SubAgent's progress, completed steps, and test results
3. VERIFY against TASK requirements:
   - Did SubAgent locate the buggy code?
   - Did SubAgent make appropriate code changes?
   - Did SubAgent run tests and confirm the fix works?
4. DECIDE:
   - ✅ status="done" AND tests pass → Use 'submit'
   - ⚠️ status="done" BUT tests fail or incomplete → Use 'delegate_task' to fix remaining issues
   - ⚠️ status="partial" → Use 'delegate_task' with guidance on next steps

CRITICAL: SWE-BENCH CONTAINER BEHAVIOR
- When SubAgent reports status="done" with passing tests, use 'submit' to trigger final evaluation
- 'submit' runs the official test suite (FAIL_TO_PASS + PASS_TO_PASS tests) to determine success


==== MODEL SELECTION ====
{model_pricing_table}

==== Progress ====
[Attempt {attempt_index}/{max_attempts}] Remaining {remaining_attempts} attempts
{budget_warning}


==== SUBTASK HISTORY ====
{subtask_history if subtask_history else "No subtasks completed yet."}

==== AVAILABLE TOOLS ====
{tools_description}

==== OUTPUT ====
Return JSON:

If SubAgent status="done" AND tests pass:
{{
  "action": "submit",
  "reasoning": "Verified: [what was fixed, which tests passed]. Submitting for evaluation.",
  "params": {{ "reason": "Fix verified: [specific fix description]" }}
}}

If SubAgent status="done" BUT tests fail or incomplete:
{{
  "action": "delegate_task",
  "reasoning": "SubAgent reported done but [specific issue]: tests show [failure details]",
  "params": {{
    "task_instruction": "CRITICAL: Previous fix incomplete. [specific next steps needed]",
    "context": "⚠️ ISSUE: [what failed]\\n- ✅ DONE: [completed work]\\n- ❌ TODO: [remaining work]",
    "model": "one of {sub_models}"
  }}
}}

If SubAgent status="partial":
{{
  "action": "delegate_task",
  "reasoning": "SubAgent made partial progress: [summary]. Need to [next steps]",
  "params": {{
    "task_instruction": "Continue: [specific next steps based on SUBTASK HISTORY]",
    "context": "From previous attempt:\\n- ✅ WORKED: [keep these]\\n- ❌ FAILED: [avoid these]",
    "model": "one of {sub_models}"
  }}
}}
""".strip()
