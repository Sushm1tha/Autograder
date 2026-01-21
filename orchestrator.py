"""
ReAct-style Orchestrator for the Autograder Agent.
Dynamically selects and uses tools to gather evidence for grading.
"""
import re
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from llm_client import LLMClient
from tools import Tool, ToolResult, get_all_tools, get_tools_description
from file_processor import SubmissionContent


@dataclass
class AgentAction:
    """Represents an action the agent wants to take."""
    action_type: str  # "tool" or "finish"
    tool_name: Optional[str] = None
    tool_input: Optional[str] = None
    thought: str = ""
    final_answer: Optional[str] = None


@dataclass
class OrchestrationTrace:
    """Records the orchestration process for debugging."""
    steps: List[Dict[str, Any]] = field(default_factory=list)

    def add_step(self, thought: str, action: str, result: str):
        self.steps.append({
            "thought": thought,
            "action": action,
            "result": result
        })

    def to_string(self) -> str:
        lines = []
        for i, step in enumerate(self.steps, 1):
            lines.append(f"Step {i}:")
            lines.append(f"  Thought: {step['thought']}")
            lines.append(f"  Action: {step['action']}")
            lines.append(f"  Result: {step['result'][:500]}...")
        return "\n".join(lines)


class GradingOrchestrator:
    """
    ReAct-style orchestrator that uses tools to gather evidence for grading.
    """

    SYSTEM_PROMPT = """You are an expert grading agent that evaluates student code submissions.
You have access to tools that help you gather evidence for grading.

{tools_description}

Your task is to evaluate a student submission against a rubric. Use the tools to:
1. Check if the code runs without errors (execute_code) - the dataset file is available in the execution directory
2. Analyze code quality (analyze_code)

IMPORTANT: Follow this exact format for your responses:

Thought: <your reasoning about what information you need>
Action: <tool_name>
Action Input: <input for the tool, or "none" if no input needed>

OR when you have enough information:

Thought: <your final reasoning>
Action: finish
Final Grade: <JSON object with scores and feedback>

Rules:
- Always start by executing the code to see if it runs
- Use analyze_code to check code quality
- Base your scores on EVIDENCE from tools, not just reading the code
- Be fair but thorough
- Maximum {max_iterations} tool calls allowed"""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_iterations: int = 5
    ):
        self.llm = LLMClient(provider=provider, model=model, api_key=api_key)
        self.tools: Dict[str, Tool] = {t.name: t for t in get_all_tools()}
        self.max_iterations = max_iterations
        self.trace: Optional[OrchestrationTrace] = None

    def grade(
        self,
        submission: SubmissionContent,
        problem_statement: str,
        rubric_text: str,
        dataset_info: str = "",
        dataset_path: str = "",
        criterion_names: List[str] = None
    ) -> Dict[str, Any]:
        """
        Grade a submission using the ReAct loop.

        Args:
            submission: The student's submission
            problem_statement: What students were asked to do
            rubric_text: Grading criteria
            dataset_info: Optional dataset information
            dataset_path: Optional path to dataset file for code execution
            criterion_names: List of exact criterion names to use

        Returns:
            Dict with grades, feedback, and orchestration trace
        """
        self.trace = OrchestrationTrace()
        self.dataset_path = dataset_path  # Store for tool execution
        self.criterion_names = criterion_names or []  # Store for synthesis

        # Build the initial prompt
        system_prompt = self.SYSTEM_PROMPT.format(
            tools_description=get_tools_description(),
            max_iterations=self.max_iterations
        )

        # Get the code from submission
        code = submission.code or ""

        user_prompt = self._build_initial_prompt(
            code=code,
            problem_statement=problem_statement,
            rubric_text=rubric_text,
            dataset_info=dataset_info,
            filename=submission.filename,
            criterion_names=criterion_names
        )

        # ReAct loop
        conversation = [{"role": "user", "content": user_prompt}]
        observations = []

        for iteration in range(self.max_iterations):
            # Get the next action from the LLM
            response = self.llm.chat(
                messages=conversation,
                system_prompt=system_prompt,
                max_tokens=2048,
                temperature=0.1
            )

            # Parse the response
            action = self._parse_action(response)

            if action.action_type == "finish":
                # Agent is done, extract final grade
                self.trace.add_step(
                    thought=action.thought,
                    action="finish",
                    result=action.final_answer or "No final answer"
                )
                return self._parse_final_grade(action.final_answer, observations)

            elif action.action_type == "tool":
                # Execute the tool
                tool_result = self._execute_tool(
                    action.tool_name,
                    code=code
                )

                observations.append(tool_result)
                self.trace.add_step(
                    thought=action.thought,
                    action=f"{action.tool_name}",
                    result=tool_result.to_prompt_string()
                )

                # Add the observation to the conversation
                observation_text = f"Observation: {tool_result.to_prompt_string()}"
                conversation.append({"role": "assistant", "content": response})
                conversation.append({"role": "user", "content": observation_text})

            else:
                # Unknown action, ask for clarification
                conversation.append({"role": "assistant", "content": response})
                conversation.append({
                    "role": "user",
                    "content": "I couldn't understand your action. Please use the exact format: Action: <tool_name> or Action: finish"
                })

        # Max iterations reached, synthesize from what we have
        return self._synthesize_from_observations(observations, rubric_text)

    def _build_initial_prompt(
        self,
        code: str,
        problem_statement: str,
        rubric_text: str,
        dataset_info: str,
        filename: str,
        criterion_names: List[str] = None
    ) -> str:
        """Build the initial prompt for the grading task."""
        prompt_parts = [
            f"Grade this student submission: {filename}",
            f"\n## Problem Statement\n{problem_statement[:2000]}",
        ]

        if dataset_info:
            prompt_parts.append(f"\n## Dataset Info\n{dataset_info[:1000]}")

        prompt_parts.append(f"\n## Rubric\n{rubric_text}")

        prompt_parts.append(f"\n## Submitted Code\n```python\n{code[:10000]}\n```")

        # Add explicit JSON format with criterion names
        if criterion_names:
            json_template = "{\n"
            for name in criterion_names:
                json_template += f'  "{name}": {{"score": <number>, "max": <max_points>, "feedback": "<feedback>"}},\n'
            json_template += '  "overall_feedback": "<overall comments>",\n'
            json_template += '  "tool_evidence": "<summary of what tools revealed>"\n}'

            prompt_parts.append(f"\n## Required JSON Format for Final Grade\nYou MUST use these EXACT criterion names:\n```json\n{json_template}\n```")

        prompt_parts.append("\nBegin your evaluation. Start by checking if the code runs.")

        return "\n".join(prompt_parts)

    def _parse_action(self, response: str) -> AgentAction:
        """Parse the LLM response to extract the action."""
        # Extract thought
        thought_match = re.search(r'Thought:\s*(.+?)(?=Action:|$)', response, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""

        # Check for finish action
        if re.search(r'Action:\s*finish', response, re.IGNORECASE):
            # Extract final grade JSON
            final_grade_match = re.search(
                r'Final Grade:\s*(\{.+\})',
                response,
                re.DOTALL
            )
            final_answer = final_grade_match.group(1) if final_grade_match else response

            return AgentAction(
                action_type="finish",
                thought=thought,
                final_answer=final_answer
            )

        # Check for tool action
        action_match = re.search(r'Action:\s*(\w+)', response)
        if action_match:
            tool_name = action_match.group(1).lower()

            # Skip "finish" if it was somehow captured differently
            if tool_name == "finish":
                return AgentAction(
                    action_type="finish",
                    thought=thought,
                    final_answer=response
                )

            # Extract action input
            input_match = re.search(r'Action Input:\s*(.+?)(?=Thought:|Action:|$)', response, re.DOTALL)
            tool_input = input_match.group(1).strip() if input_match else ""

            return AgentAction(
                action_type="tool",
                tool_name=tool_name,
                tool_input=tool_input,
                thought=thought
            )

        # Check if the response contains JSON (LLM went straight to grading)
        if re.search(r'\{[\s\S]*"score"[\s\S]*\}', response):
            return AgentAction(
                action_type="finish",
                thought=thought,
                final_answer=response
            )

        # Couldn't parse - return unknown
        return AgentAction(action_type="unknown", thought=thought)

    def _execute_tool(
        self,
        tool_name: str,
        code: str
    ) -> ToolResult:
        """Execute a tool and return the result."""
        tool_name = tool_name.lower()

        if tool_name not in self.tools:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output="",
                error=f"Unknown tool: {tool_name}. Available: {list(self.tools.keys())}"
            )

        tool = self.tools[tool_name]

        if tool_name == "execute_code":
            # Pass dataset_path so code can access the dataset
            return tool.execute(code=code, dataset_path=getattr(self, 'dataset_path', ''))
        elif tool_name == "analyze_code":
            return tool.execute(code=code)
        elif tool_name == "run_tests":
            return tool.execute(code=code, test_code="")
        else:
            return tool.execute(code=code)

    def _parse_final_grade(
        self,
        final_answer: Optional[str],
        observations: List[ToolResult]
    ) -> Dict[str, Any]:
        """Parse the final grade from the LLM response."""
        result = {
            "success": True,
            "scores": {},
            "overall_feedback": "",
            "tool_evidence": "",
            "trace": self.trace.to_string() if self.trace else "",
            "raw_response": final_answer or ""  # Always store for debugging
        }

        if not final_answer:
            result["success"] = False
            result["error"] = "No final grade provided"
            return result

        # Try to extract JSON from the response
        try:
            # Clean up the response - remove markdown code blocks
            cleaned = final_answer
            cleaned = re.sub(r'```json\s*', '', cleaned)
            cleaned = re.sub(r'```\s*', '', cleaned)

            # Find the outermost JSON object
            brace_count = 0
            start_idx = -1
            end_idx = -1

            for i, char in enumerate(cleaned):
                if char == '{':
                    if brace_count == 0:
                        start_idx = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and start_idx != -1:
                        end_idx = i + 1
                        break

            if start_idx != -1 and end_idx != -1:
                json_str = cleaned[start_idx:end_idx]

                # Fix common JSON issues
                # Remove trailing commas before } or ]
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)

                grade_data = json.loads(json_str)

                # Extract scores
                for key, value in grade_data.items():
                    if key in ["overall_feedback", "tool_evidence"]:
                        result[key] = value
                    elif isinstance(value, dict) and "score" in value:
                        result["scores"][key] = value
                    elif isinstance(value, (int, float)):
                        result["scores"][key] = {"score": value, "feedback": ""}

                # Add tool evidence summary if not provided
                if not result["tool_evidence"] and observations:
                    evidence_parts = []
                    for obs in observations:
                        if obs.success:
                            evidence_parts.append(f"{obs.tool_name}: Success")
                        else:
                            evidence_parts.append(f"{obs.tool_name}: {obs.error[:100]}")
                    result["tool_evidence"] = "; ".join(evidence_parts)

            else:
                result["success"] = False
                result["error"] = "Could not find JSON in response"
                result["raw_response"] = final_answer

        except json.JSONDecodeError as e:
            result["success"] = False
            result["error"] = f"JSON parse error: {str(e)}"
            result["raw_response"] = final_answer

        return result

    def _synthesize_from_observations(
        self,
        observations: List[ToolResult],
        rubric_text: str
    ) -> Dict[str, Any]:
        """Synthesize a grade from observations when max iterations reached."""
        # Build a summary of observations
        obs_summary = []
        for obs in observations:
            obs_summary.append(obs.to_prompt_string())

        # Build JSON template with criterion names
        criterion_names = getattr(self, 'criterion_names', [])
        if criterion_names:
            json_template = "{\n"
            for name in criterion_names:
                json_template += f'  "{name}": {{"score": <number>, "feedback": "<feedback>"}},\n'
            json_template += '  "overall_feedback": "<overall comments>"\n}'
        else:
            json_template = '{\n  "criterion_name": {"score": <number>, "feedback": "<feedback>"},\n  "overall_feedback": "<overall comments>"\n}'

        # Ask LLM to synthesize
        prompt = f"""Based on these tool results, provide a final grade.

Tool Results:
{chr(10).join(obs_summary)}

Rubric:
{rubric_text}

Provide your grade as JSON using these EXACT criterion names:
{json_template}"""

        response = self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.1
        )

        return self._parse_final_grade(response, observations)

    def get_trace(self) -> Optional[OrchestrationTrace]:
        """Get the orchestration trace for debugging."""
        return self.trace
