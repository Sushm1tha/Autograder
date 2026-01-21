"""
Grading Engine for evaluating student submissions against rubrics.
Supports both direct LLM grading and tool-assisted grading via orchestrator.
"""
import json
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from llm_client import LLMClient
from file_processor import SubmissionContent
from orchestrator import GradingOrchestrator


@dataclass
class RubricCriteria:
    """Represents a single rubric criterion."""
    name: str
    description: str
    max_points: float
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "max_points": self.max_points,
            "weight": self.weight
        }


@dataclass
class CriterionScore:
    """Score for a single criterion."""
    criterion_name: str
    score: float
    max_points: float
    feedback: str
    percentage: float = 0.0

    def __post_init__(self):
        if self.max_points > 0:
            self.percentage = (self.score / self.max_points) * 100


@dataclass
class GradingResult:
    """Result of grading a submission."""
    filename: str
    scores: Dict[str, CriterionScore] = field(default_factory=dict)
    total_score: float = 0.0
    max_total: float = 0.0
    percentage: float = 0.0
    overall_feedback: str = ""
    errors: List[str] = field(default_factory=list)
    raw_response: str = ""
    tool_evidence: str = ""  # Evidence gathered from tools
    orchestration_trace: str = ""  # ReAct loop trace for debugging
    used_tools: bool = False  # Whether tools were used for grading

    def calculate_totals(self):
        """Calculate total scores."""
        self.total_score = sum(s.score for s in self.scores.values())
        self.max_total = sum(s.max_points for s in self.scores.values())
        if self.max_total > 0:
            self.percentage = (self.total_score / self.max_total) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame."""
        result = {"Filename": self.filename}
        for name, score in self.scores.items():
            result[name] = score.score
        result["Total"] = self.total_score
        result["Max Points"] = self.max_total
        result["Percentage"] = f"{self.percentage:.1f}%"
        return result

    def get_detailed_feedback(self) -> str:
        """Get detailed feedback string."""
        lines = [f"=== Grading Results for {self.filename} ===\n"]

        for name, score in self.scores.items():
            lines.append(f"\n{name}: {score.score}/{score.max_points} ({score.percentage:.1f}%)")
            lines.append(f"  Feedback: {score.feedback}")

        lines.append(f"\n{'='*50}")
        lines.append(f"Total: {self.total_score}/{self.max_total} ({self.percentage:.1f}%)")

        if self.overall_feedback:
            lines.append(f"\nOverall Feedback: {self.overall_feedback}")

        return "\n".join(lines)


class GradingEngine:
    """Engine for grading submissions using LLM."""

    GRADING_SYSTEM_PROMPT = """You are an expert academic grader. Your task is to evaluate student submissions fairly and return scores in a specific format.

CRITICAL RULES:
1. You MUST return valid JSON only - no other text before or after
2. You MUST use the EXACT criterion names provided
3. Scores must be numbers (not strings)
4. Be fair but thorough in your evaluation
5. Give partial credit where appropriate"""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize the grading engine.

        Args:
            provider: LLM provider (ollama, openai, anthropic)
            model: Model name to use
            api_key: API key for the provider
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.llm = LLMClient(provider=provider, model=model, api_key=api_key)
        self.rubric: List[RubricCriteria] = []
        # Always use the ReAct orchestrator with tools
        self.orchestrator = GradingOrchestrator(
            provider=provider,
            model=model,
            api_key=api_key
        )

    def set_rubric(self, rubric: List[RubricCriteria]):
        """Set the rubric for grading."""
        self.rubric = rubric

    def parse_rubric_text(self, rubric_text: str) -> List[RubricCriteria]:
        """Parse rubric from text format."""
        criteria = []
        lines = rubric_text.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Pattern 1: "Name (X points): Description" or "Name (X): Description"
            match = re.match(r'^[•\-\*]?\s*(.+?)\s*\((\d+(?:\.\d+)?)\s*(?:points?)?\)\s*[:\-]?\s*(.*)$', line, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                points = float(match.group(2))
                description = match.group(3).strip() or name
                criteria.append(RubricCriteria(name=name, description=description, max_points=points))
                continue

            # Pattern 2: "Name: Description (X points)"
            match = re.match(r'^[•\-\*]?\s*(.+?)\s*[:\-]\s*(.+?)\s*\((\d+(?:\.\d+)?)\s*(?:points?)?\)\s*$', line, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                description = match.group(2).strip()
                points = float(match.group(3))
                criteria.append(RubricCriteria(name=name, description=description, max_points=points))
                continue

            # Pattern 3: "Name - X points: Description"
            match = re.match(r'^[•\-\*]?\s*(.+?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*(?:points?)?\s*[:\-]?\s*(.*)$', line, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                points = float(match.group(2))
                description = match.group(3).strip() or name
                criteria.append(RubricCriteria(name=name, description=description, max_points=points))
                continue

            # Pattern 4: Simple "Name: Description" (assume 10 points)
            match = re.match(r'^[•\-\*]?\s*(.+?)\s*[:\-]\s*(.+)$', line)
            if match:
                name = match.group(1).strip()
                description = match.group(2).strip()
                points_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:points?|pts?)', description, re.IGNORECASE)
                points = float(points_match.group(1)) if points_match else 10.0
                criteria.append(RubricCriteria(name=name, description=description, max_points=points))

        return criteria

    def grade_submission(
        self,
        submission: SubmissionContent,
        problem_statement: str,
        dataset_info: str = "",
        rubric: Optional[List[RubricCriteria]] = None,
        dataset_path: str = ""
    ) -> GradingResult:
        """
        Grade a single submission using the ReAct orchestrator.

        Args:
            submission: The student's submission
            problem_statement: What students were asked to do
            dataset_info: Optional dataset information
            rubric: Grading criteria
            dataset_path: Path to dataset file for code execution

        Returns:
            GradingResult with scores and feedback
        """
        rubric = rubric or self.rubric

        if not rubric:
            return GradingResult(
                filename=submission.filename,
                errors=["No rubric provided for grading"]
            )

        if submission.errors:
            return GradingResult(
                filename=submission.filename,
                errors=submission.errors
            )

        submission_content = submission.get_full_content()

        if not submission_content.strip():
            return GradingResult(
                filename=submission.filename,
                errors=["Submission appears to be empty or could not be read"]
            )

        # Always use tool-assisted grading with the ReAct orchestrator
        return self._grade_with_tools(
            submission=submission,
            problem_statement=problem_statement,
            dataset_info=dataset_info,
            rubric=rubric,
            dataset_path=dataset_path
        )

    def _grade_with_tools(
        self,
        submission: SubmissionContent,
        problem_statement: str,
        dataset_info: str,
        rubric: List[RubricCriteria],
        dataset_path: str = ""
    ) -> GradingResult:
        """
        Grade a submission using the ReAct orchestrator with tools.

        This method uses CodeExecutor and StaticAnalyzer
        to gather evidence before making grading decisions.
        """
        # Build rubric text for the orchestrator
        rubric_text = self._format_rubric_for_prompt(rubric)

        # Extract criterion names for exact matching
        criterion_names = [c.name for c in rubric]

        try:
            # Run the orchestrator
            orch_result = self.orchestrator.grade(
                submission=submission,
                problem_statement=problem_statement,
                rubric_text=rubric_text,
                dataset_info=dataset_info,
                dataset_path=dataset_path,
                criterion_names=criterion_names
            )

            # Convert orchestrator result to GradingResult
            result = GradingResult(
                filename=submission.filename,
                used_tools=True,
                tool_evidence=orch_result.get("tool_evidence", ""),
                orchestration_trace=orch_result.get("trace", ""),
                raw_response=orch_result.get("raw_response", "")  # Always store for debugging
            )

            if not orch_result.get("success", False):
                result.errors.append(orch_result.get("error", "Orchestration failed"))

            # Map scores from orchestrator to GradingResult
            orch_scores = orch_result.get("scores", {})

            for criterion in rubric:
                score_data = None

                # Try exact match first
                if criterion.name in orch_scores:
                    score_data = orch_scores[criterion.name]
                else:
                    # Try case-insensitive and partial matching
                    criterion_lower = criterion.name.lower()
                    for key, value in orch_scores.items():
                        key_lower = key.lower()
                        # Exact match (case insensitive)
                        if key_lower == criterion_lower:
                            score_data = value
                            break
                        # Partial match (one contains the other)
                        if criterion_lower in key_lower or key_lower in criterion_lower:
                            score_data = value
                            break
                        # Word-based match (key words appear in criterion)
                        key_words = set(key_lower.replace('_', ' ').replace('-', ' ').split())
                        criterion_words = set(criterion_lower.replace('_', ' ').replace('-', ' ').split())
                        if key_words & criterion_words:  # Intersection not empty
                            score_data = value
                            break

                if score_data is not None:
                    if isinstance(score_data, dict):
                        score_value = float(score_data.get("score", 0))
                        feedback = score_data.get("feedback", "")
                    else:
                        try:
                            score_value = float(score_data)
                            feedback = ""
                        except (ValueError, TypeError):
                            score_value = 0
                            feedback = "Could not parse score"

                    # Clamp score to valid range
                    score_value = max(0, min(score_value, criterion.max_points))

                    result.scores[criterion.name] = CriterionScore(
                        criterion_name=criterion.name,
                        score=score_value,
                        max_points=criterion.max_points,
                        feedback=feedback
                    )
                else:
                    # Criterion not found in orchestrator result
                    result.scores[criterion.name] = CriterionScore(
                        criterion_name=criterion.name,
                        score=0,
                        max_points=criterion.max_points,
                        feedback="Score not found in orchestrator response"
                    )

            result.overall_feedback = orch_result.get("overall_feedback", "")
            result.calculate_totals()

            return result

        except Exception as e:
            return GradingResult(
                filename=submission.filename,
                errors=[f"Tool-assisted grading error: {str(e)}"],
                used_tools=True
            )

    def _format_rubric_for_prompt(self, rubric: List[RubricCriteria]) -> str:
        """Format rubric for the grading prompt."""
        lines = []
        for criterion in rubric:
            lines.append(f"- {criterion.name} (max {int(criterion.max_points)} pts): {criterion.description}")
        return "\n".join(lines)

    def _clean_json_response(self, response: str) -> str:
        """Clean up LLM response to extract valid JSON."""
        # Remove markdown code blocks
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)

        # Remove any text before the first {
        first_brace = response.find('{')
        if first_brace > 0:
            response = response[first_brace:]

        # Remove any text after the last }
        last_brace = response.rfind('}')
        if last_brace >= 0:
            response = response[:last_brace + 1]

        # Remove JavaScript-style comments
        response = re.sub(r'//.*?$', '', response, flags=re.MULTILINE)
        response = re.sub(r'/\*.*?\*/', '', response, flags=re.DOTALL)

        # Fix common JSON issues
        # Remove trailing commas before } or ]
        response = re.sub(r',\s*}', '}', response)
        response = re.sub(r',\s*]', ']', response)

        return response.strip()

    def _extract_scores_fallback(self, response: str, rubric: List[RubricCriteria]) -> Dict[str, Dict]:
        """Fallback method to extract scores using regex patterns."""
        scores = {}

        for criterion in rubric:
            # Try to find score for this criterion
            # Pattern: "criterion_name": {"score": X, or "criterion_name": X, or criterion_name: X/max
            patterns = [
                # JSON-like: "Name": {"score": 8
                rf'"{re.escape(criterion.name)}"\s*:\s*\{{\s*"score"\s*:\s*(\d+(?:\.\d+)?)',
                # Simple: "Name": 8
                rf'"{re.escape(criterion.name)}"\s*:\s*(\d+(?:\.\d+)?)',
                # Text-like: Name: 8/10 or Name: 8 points
                rf'{re.escape(criterion.name)}\s*[:\-]\s*(\d+(?:\.\d+)?)\s*(?:/\d+|points?|pts?)?',
                # Score followed by criterion name
                rf'(\d+(?:\.\d+)?)\s*(?:/\d+)?\s*(?:for|on|-)?\s*{re.escape(criterion.name)}',
            ]

            score_found = None
            for pattern in patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    try:
                        score_found = float(match.group(1))
                        break
                    except (ValueError, IndexError):
                        continue

            if score_found is not None:
                # Ensure score is within bounds
                score_found = min(score_found, criterion.max_points)
                score_found = max(score_found, 0)

                # Try to find feedback
                feedback_pattern = rf'{re.escape(criterion.name)}[^"]*"feedback"\s*:\s*"([^"]+)"'
                feedback_match = re.search(feedback_pattern, response, re.IGNORECASE | re.DOTALL)
                feedback = feedback_match.group(1) if feedback_match else "Score extracted from response"

                scores[criterion.name] = {
                    "score": score_found,
                    "feedback": feedback
                }

        return scores

    def _parse_grading_response(
        self,
        response: str,
        filename: str,
        rubric: List[RubricCriteria]
    ) -> GradingResult:
        """Parse the LLM grading response with multiple fallback methods."""
        result = GradingResult(filename=filename)

        # Try to clean and parse JSON
        cleaned_response = self._clean_json_response(response)
        scores_data = {}
        overall_feedback = ""

        # Method 1: Try direct JSON parsing
        try:
            data = json.loads(cleaned_response)
            # Check if it has nested "scores" key or flat structure
            if "scores" in data:
                scores_data = data.get("scores", {})
            else:
                # Flat structure - each criterion is a top-level key
                scores_data = {k: v for k, v in data.items() if k != "overall_feedback"}
            overall_feedback = data.get("overall_feedback", "")
        except json.JSONDecodeError:
            # Method 2: Try to fix JSON and parse again
            try:
                # Try adding missing quotes around keys
                fixed = re.sub(r'(\w+)(?=\s*:)', r'"\1"', cleaned_response)
                data = json.loads(fixed)
                if "scores" in data:
                    scores_data = data.get("scores", {})
                else:
                    scores_data = {k: v for k, v in data.items() if k != "overall_feedback"}
                overall_feedback = data.get("overall_feedback", "")
            except json.JSONDecodeError:
                # Method 3: Fallback to regex extraction
                scores_data = self._extract_scores_fallback(response, rubric)
                # Try to extract overall feedback
                feedback_match = re.search(r'"overall_feedback"\s*:\s*"([^"]+)"', response, re.IGNORECASE)
                if feedback_match:
                    overall_feedback = feedback_match.group(1)

        # Map scores to criteria
        for criterion in rubric:
            criterion_score = None

            # Try exact match
            if criterion.name in scores_data:
                criterion_score = scores_data[criterion.name]
            else:
                # Try case-insensitive and partial match
                for key, value in scores_data.items():
                    if key.lower() == criterion.name.lower():
                        criterion_score = value
                        break
                    # Partial match
                    if criterion.name.lower() in key.lower() or key.lower() in criterion.name.lower():
                        criterion_score = value
                        break

            if criterion_score is not None:
                # Handle different formats
                if isinstance(criterion_score, dict):
                    score_value = criterion_score.get("score", 0)
                    feedback = criterion_score.get("feedback", "No detailed feedback")
                elif isinstance(criterion_score, (int, float)):
                    score_value = criterion_score
                    feedback = "Score assigned"
                else:
                    try:
                        score_value = float(criterion_score)
                        feedback = "Score assigned"
                    except (ValueError, TypeError):
                        score_value = 0
                        feedback = "Could not parse score"

                # Ensure score is valid
                try:
                    score_value = float(score_value)
                except (ValueError, TypeError):
                    score_value = 0

                score_value = min(score_value, criterion.max_points)
                score_value = max(score_value, 0)

                result.scores[criterion.name] = CriterionScore(
                    criterion_name=criterion.name,
                    score=score_value,
                    max_points=criterion.max_points,
                    feedback=str(feedback)
                )
            else:
                # Criterion not found - try one more extraction attempt
                fallback_scores = self._extract_scores_fallback(response, [criterion])
                if criterion.name in fallback_scores:
                    fs = fallback_scores[criterion.name]
                    result.scores[criterion.name] = CriterionScore(
                        criterion_name=criterion.name,
                        score=float(fs.get("score", 0)),
                        max_points=criterion.max_points,
                        feedback=fs.get("feedback", "Extracted from response")
                    )
                else:
                    result.scores[criterion.name] = CriterionScore(
                        criterion_name=criterion.name,
                        score=0,
                        max_points=criterion.max_points,
                        feedback="Could not find score in response"
                    )

        result.overall_feedback = overall_feedback
        result.calculate_totals()

        return result

    def grade_multiple_submissions(
        self,
        submissions: List[SubmissionContent],
        problem_statement: str,
        dataset_info: str = "",
        rubric: Optional[List[RubricCriteria]] = None,
        progress_callback=None
    ) -> List[GradingResult]:
        """Grade multiple submissions."""
        results = []

        for i, submission in enumerate(submissions):
            if progress_callback:
                progress_callback(i, len(submissions), submission.filename)

            result = self.grade_submission(
                submission=submission,
                problem_statement=problem_statement,
                dataset_info=dataset_info,
                rubric=rubric
            )
            results.append(result)

        return results

    def results_to_dataframe(self, results: List[GradingResult]):
        """Convert grading results to a pandas DataFrame."""
        import pandas as pd

        if not results:
            return pd.DataFrame()

        data = [r.to_dict() for r in results]
        df = pd.DataFrame(data)

        # Reorder columns
        if len(df.columns) > 0:
            cols = list(df.columns)
            if 'Filename' in cols:
                cols.remove('Filename')
                cols = ['Filename'] + cols
            for col in ['Total', 'Max Points', 'Percentage']:
                if col in cols:
                    cols.remove(col)
                    cols.append(col)
            df = df[cols]

        return df
