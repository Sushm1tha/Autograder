"""
Tools for the Autograder Agent.
Provides CodeExecutor, StaticAnalyzer, and TestRunner for evidence-based grading.
"""
import subprocess
import tempfile
import os
import sys
import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod


@dataclass
class ToolResult:
    """Result from a tool execution."""
    tool_name: str
    success: bool
    output: str
    error: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    def to_prompt_string(self) -> str:
        """Format result for inclusion in LLM prompt."""
        if self.success:
            return f"[{self.tool_name}] Success:\n{self.output}"
        else:
            return f"[{self.tool_name}] Failed:\n{self.error}"


class Tool(ABC):
    """Base class for all tools."""

    name: str
    description: str

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool and return results."""
        pass

    def get_schema(self) -> Dict[str, Any]:
        """Return the tool schema for the LLM."""
        return {
            "name": self.name,
            "description": self.description
        }


class CodeExecutor(Tool):
    """
    Executes Python code in a sandboxed subprocess.
    Captures stdout, stderr, and handles timeouts.
    Can provide a dataset file for the code to use.
    """

    name = "execute_code"
    description = "Run Python code and capture the output. Use this to check if code runs without errors and see what it produces."

    def __init__(self, timeout: int = 30, max_output_length: int = 5000):
        self.timeout = timeout
        self.max_output_length = max_output_length

    def execute(self, code: str, dataset_path: str = "", **kwargs) -> ToolResult:
        """
        Execute Python code in a subprocess.

        Args:
            code: Python code to execute
            dataset_path: Optional path to dataset file to make available

        Returns:
            ToolResult with execution output
        """
        if not code or not code.strip():
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error="No code provided"
            )

        # Create a temporary directory for execution
        exec_dir = tempfile.mkdtemp()

        # Copy dataset to execution directory if provided
        dataset_filename = ""
        if dataset_path and os.path.exists(dataset_path):
            import shutil
            dataset_filename = os.path.basename(dataset_path)
            dest_path = os.path.join(exec_dir, dataset_filename)
            shutil.copy2(dataset_path, dest_path)

        # Create the code file in execution directory
        temp_file = os.path.join(exec_dir, "submission.py")
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(code)

        try:
            # Run the code in a subprocess from the execution directory
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=exec_dir,
                env={
                    **os.environ,
                    'PYTHONDONTWRITEBYTECODE': '1',
                }
            )

            stdout = result.stdout[:self.max_output_length] if result.stdout else ""
            stderr = result.stderr[:self.max_output_length] if result.stderr else ""

            if result.returncode == 0:
                output = stdout if stdout else "(Code executed successfully with no output)"
                return ToolResult(
                    tool_name=self.name,
                    success=True,
                    output=output,
                    data={
                        "return_code": result.returncode,
                        "has_output": bool(stdout)
                    }
                )
            else:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    output=stdout,
                    error=stderr or f"Process exited with code {result.returncode}",
                    data={"return_code": result.returncode}
                )

        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error=f"Execution timed out after {self.timeout} seconds"
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error=f"Execution error: {str(e)}"
            )
        finally:
            # Clean up execution directory
            try:
                import shutil
                shutil.rmtree(exec_dir)
            except Exception:
                pass


class StaticAnalyzer(Tool):
    """
    Runs static analysis on Python code using pylint and flake8.
    Returns code quality metrics and issues.
    """

    name = "analyze_code"
    description = "Run static analysis (pylint/flake8) on Python code to check code quality, find bugs, and identify style issues."

    def __init__(self, max_issues: int = 20):
        self.max_issues = max_issues

    def execute(self, code: str, **kwargs) -> ToolResult:
        """
        Analyze Python code with static analysis tools.

        Args:
            code: Python code to analyze

        Returns:
            ToolResult with analysis findings
        """
        if not code or not code.strip():
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error="No code provided"
            )

        # Create a temporary file with the code
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write(code)
            temp_file = f.name

        results = {
            "pylint": {"score": None, "issues": []},
            "flake8": {"issues": []}
        }

        try:
            # Run pylint
            pylint_result = self._run_pylint(temp_file)
            results["pylint"] = pylint_result

            # Run flake8
            flake8_result = self._run_flake8(temp_file)
            results["flake8"] = flake8_result

            # Format output
            output_lines = []

            if pylint_result["score"] is not None:
                output_lines.append(f"Pylint Score: {pylint_result['score']}/10")

            total_issues = len(pylint_result["issues"]) + len(flake8_result["issues"])
            output_lines.append(f"Total Issues Found: {total_issues}")

            if pylint_result["issues"]:
                output_lines.append("\nPylint Issues:")
                for issue in pylint_result["issues"][:self.max_issues]:
                    output_lines.append(f"  - {issue}")

            if flake8_result["issues"]:
                output_lines.append("\nFlake8 Issues:")
                for issue in flake8_result["issues"][:self.max_issues]:
                    output_lines.append(f"  - {issue}")

            if total_issues == 0:
                output_lines.append("\nNo issues found. Code looks clean!")

            return ToolResult(
                tool_name=self.name,
                success=True,
                output="\n".join(output_lines),
                data=results
            )

        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error=f"Analysis error: {str(e)}"
            )
        finally:
            try:
                os.unlink(temp_file)
            except Exception:
                pass

    def _run_pylint(self, file_path: str) -> Dict[str, Any]:
        """Run pylint on a file."""
        try:
            result = subprocess.run(
                [
                    sys.executable, '-m', 'pylint',
                    '--output-format=json',
                    '--disable=C0114,C0115,C0116',  # Disable missing docstring warnings
                    file_path
                ],
                capture_output=True,
                text=True,
                timeout=30
            )

            issues = []
            score = None

            # Parse JSON output
            if result.stdout:
                try:
                    messages = json.loads(result.stdout)
                    for msg in messages[:self.max_issues]:
                        issues.append(
                            f"Line {msg.get('line', '?')}: [{msg.get('symbol', 'unknown')}] {msg.get('message', '')}"
                        )
                except json.JSONDecodeError:
                    pass

            # Extract score from stderr (pylint outputs score there)
            if result.stderr:
                score_match = re.search(r'rated at ([\d.]+)/10', result.stderr)
                if score_match:
                    score = float(score_match.group(1))

            return {"score": score, "issues": issues}

        except subprocess.TimeoutExpired:
            return {"score": None, "issues": ["Pylint timed out"]}
        except FileNotFoundError:
            return {"score": None, "issues": ["Pylint not installed"]}
        except Exception as e:
            return {"score": None, "issues": [f"Pylint error: {str(e)}"]}

    def _run_flake8(self, file_path: str) -> Dict[str, Any]:
        """Run flake8 on a file."""
        try:
            result = subprocess.run(
                [
                    sys.executable, '-m', 'flake8',
                    '--max-line-length=120',
                    file_path
                ],
                capture_output=True,
                text=True,
                timeout=30
            )

            issues = []
            if result.stdout:
                for line in result.stdout.strip().split('\n')[:self.max_issues]:
                    if line.strip():
                        # Remove file path prefix, keep just the issue
                        parts = line.split(':', 3)
                        if len(parts) >= 4:
                            issues.append(f"Line {parts[1]}: {parts[3].strip()}")
                        else:
                            issues.append(line)

            return {"issues": issues}

        except subprocess.TimeoutExpired:
            return {"issues": ["Flake8 timed out"]}
        except FileNotFoundError:
            return {"issues": ["Flake8 not installed"]}
        except Exception as e:
            return {"issues": [f"Flake8 error: {str(e)}"]}


@dataclass
class TestCase:
    """Represents a single test case."""
    name: str
    code: str  # Test code that should pass/fail
    expected_behavior: str  # Description of what should happen


class TestRunner(Tool):
    """
    Runs test cases against submitted code.
    Executes predefined tests and reports pass/fail status.
    """

    name = "run_tests"
    description = "Run test cases against the submitted code to verify correctness. Returns which tests passed and which failed."

    def __init__(self, timeout: int = 60):
        self.timeout = timeout

    def execute(self, code: str, test_code: str = "", **kwargs) -> ToolResult:
        """
        Run tests against the submitted code.

        Args:
            code: The submitted code to test
            test_code: Test code to run (pytest-style or simple assertions)

        Returns:
            ToolResult with test results
        """
        if not code or not code.strip():
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error="No code provided"
            )

        if not test_code or not test_code.strip():
            return ToolResult(
                tool_name=self.name,
                success=True,
                output="No test cases provided. Skipping test execution.",
                data={"skipped": True}
            )

        # Combine submission code with test code
        combined_code = f"""
# === Submitted Code ===
{code}

# === Test Code ===
{test_code}
"""

        # Create a temporary file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write(combined_code)
            temp_file = f.name

        try:
            # Try running with pytest first
            result = self._run_with_pytest(temp_file)
            if result is not None:
                return result

            # Fallback to direct execution
            return self._run_directly(temp_file)

        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error=f"Test execution error: {str(e)}"
            )
        finally:
            try:
                os.unlink(temp_file)
            except Exception:
                pass

    def _run_with_pytest(self, file_path: str) -> Optional[ToolResult]:
        """Try to run tests with pytest."""
        try:
            result = subprocess.run(
                [
                    sys.executable, '-m', 'pytest',
                    file_path,
                    '-v',
                    '--tb=short',
                    '--no-header'
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            output = result.stdout + result.stderr

            # Parse pytest output
            passed = len(re.findall(r' PASSED', output))
            failed = len(re.findall(r' FAILED', output))
            errors = len(re.findall(r' ERROR', output))

            total = passed + failed + errors

            if total == 0:
                # No tests found by pytest, return None to try direct execution
                return None

            summary = f"Tests Run: {total}\nPassed: {passed}\nFailed: {failed}\nErrors: {errors}"

            # Extract failure details
            if failed > 0 or errors > 0:
                summary += f"\n\nDetails:\n{output[-2000:]}"  # Last 2000 chars

            return ToolResult(
                tool_name=self.name,
                success=(failed == 0 and errors == 0),
                output=summary,
                data={
                    "total": total,
                    "passed": passed,
                    "failed": failed,
                    "errors": errors
                }
            )

        except FileNotFoundError:
            return None  # pytest not installed, try direct execution
        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error=f"Tests timed out after {self.timeout} seconds"
            )
        except Exception:
            return None

    def _run_directly(self, file_path: str) -> ToolResult:
        """Run the file directly (for simple assertion-based tests)."""
        try:
            result = subprocess.run(
                [sys.executable, file_path],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            if result.returncode == 0:
                output = result.stdout if result.stdout else "All assertions passed."
                return ToolResult(
                    tool_name=self.name,
                    success=True,
                    output=output,
                    data={"method": "direct_execution"}
                )
            else:
                error_output = result.stderr or result.stdout
                # Check for assertion errors
                if "AssertionError" in error_output:
                    return ToolResult(
                        tool_name=self.name,
                        success=False,
                        output="",
                        error=f"Test assertion failed:\n{error_output[-1500:]}",
                        data={"assertion_failed": True}
                    )
                else:
                    return ToolResult(
                        tool_name=self.name,
                        success=False,
                        output="",
                        error=f"Test execution failed:\n{error_output[-1500:]}"
                    )

        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error=f"Tests timed out after {self.timeout} seconds"
            )


def get_all_tools() -> List[Tool]:
    """Return instances of all available tools."""
    return [
        CodeExecutor(),
        StaticAnalyzer(),
        TestRunner()
    ]


def get_tools_description() -> str:
    """Get a formatted description of all tools for the LLM prompt."""
    tools = get_all_tools()
    lines = ["Available tools:"]
    for tool in tools:
        lines.append(f"- {tool.name}: {tool.description}")
    return "\n".join(lines)
