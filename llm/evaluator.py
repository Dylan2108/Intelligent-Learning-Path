from __future__ import annotations

import logging

from pydantic import BaseModel, Field, ValidationError

from llm.client import OllamaClient

logger = logging.getLogger(__name__)


class EvaluationResult(BaseModel):
    """
    Structured evaluation of a learning path returned by the LLM.
    """

    score: float = Field(..., ge=0.0, le=10.0)
    rationale: str
    weaknesses: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class PathEvaluator:
    """
    Evaluates a generated learning trajectory using an LLM (Ollama).
    Returns a structured `EvaluationResult` instead of free text so that
    downstream code (e.g. metaheuristic fitness) can use the numeric score.
    """

    def __init__(self, client: OllamaClient | None = None):
        self.client = client or OllamaClient()

    def _build_prompt(self, career: str, path: list[str]) -> str:
        path_str = " -> ".join(path) if path else "(empty)"
        return f"""
You are a senior career coach and curriculum designer. Evaluate the proposed
learning trajectory for the role of "{career}".

Trajectory:
{path_str}

Return ONLY a JSON object with these fields (no markdown, no comments):
- "score": float between 0 and 10. Be strict. A perfect path scores 9-10.
- "rationale": 2-3 sentence justification of the score.
- "weaknesses": list of short strings naming concrete gaps or weak steps.
- "suggestions": list of short strings with concrete improvements
  (e.g. add a course, reorder, remove a redundant one).

Evaluation criteria:
- Coherence and logical progression of topics.
- Coverage of the skills typically required for the target role.
- Employability: are the steps recognizable and valuable in the job market?
- Practical usefulness and absence of redundancy.
""".strip()

    def evaluate(self, career: str, path: list[str]) -> EvaluationResult:
        """
        Evaluates the given trajectory.
        Args:
            career (str): Target career name.
            path (list[str]): Ordered list of course names.
        Returns:
            EvaluationResult: Structured evaluation.
        Raises:
            ValueError: If the LLM output cannot be parsed.
        """
        prompt = self._build_prompt(career, path)
        raw = self.client.chat_json(prompt=prompt, temperature=0.1)

        try:
            return EvaluationResult(**raw)
        except ValidationError as exc:
            raise ValueError(
                f"LLM output did not match EvaluationResult: {exc}\nRaw: {raw}"
            ) from exc
