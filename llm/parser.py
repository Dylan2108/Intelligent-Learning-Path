from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from llm.client import OllamaClient

logger = logging.getLogger(__name__)


class GoalSchema(BaseModel):
    """
    Structured representation of a user career goal.
    """

    target_career: str = Field(..., description="Exact career name as in careers.json")
    initial_skills: list[str] = Field(
        default_factory=list,
        description="Skills the user already has (must match course names)",
    )
    max_budget: int | None = Field(
        default=None, description="Maximum monetary budget, or null if none"
    )
    max_weeks: int | None = Field(
        default=None, description="Maximum total time in weeks, or null if none"
    )
    notes: str = Field(default="", description="Free-form additional context")


class GoalParser:
    """
    Uses an LLM (Ollama) to parse natural-language career goals
    into a structured `GoalSchema`.
    """

    _CAREER_HINT_PATH = Path("data/careers.json")

    def __init__(self, client: OllamaClient | None = None):
        self.client = client or OllamaClient()
        self._career_catalog: list[str] = self._load_career_catalog()

    def _load_career_catalog(self) -> list[str]:
        if not self._CAREER_HINT_PATH.exists():
            return []
        try:
            data = json.loads(self._CAREER_HINT_PATH.read_text())
            return [c.get("career", "") for c in data if c.get("career")]
        except Exception as exc:
            logger.warning("Could not load career catalog: %s", exc)
            return []

    def _build_prompt(self, text: str) -> str:
        catalog_str = (
            "AVAILABLE CAREERS: " + ", ".join(self._career_catalog)
            if self._career_catalog
            else ""
        )
        return f"""
You are an assistant that extracts structured information from a user's career goal.

The user wants to become a professional and tells you what they already know
and any constraints (time, money). Convert their message into JSON.

{catalog_str}

Return ONLY a JSON object with these fields (no markdown, no comments):
- "target_career": string. Must be one of the available careers if listed.
- "initial_skills": list of strings. Course names the user already knows.
- "max_budget": integer or null. Maximum total cost the user can pay.
- "max_weeks": integer or null. Maximum total weeks the user can study.
- "notes": string. Any other relevant context in one short sentence.

If the user does not mention a constraint, use null. If a value is missing,
infer a reasonable default from the text. Do not invent skills the user did
not mention.

User text:
\"\"\"{text}\"\"\"
""".strip()

    def _coerce(self, raw: dict[str, Any]) -> dict[str, Any]:
        """
        Best-effort coercion so the model can be a little sloppy.
        """
        if "initial_skills" in raw and raw["initial_skills"] is None:
            raw["initial_skills"] = []
        if isinstance(raw.get("initial_skills"), str):
            raw["initial_skills"] = [
                s.strip() for s in re.split(r"[,;]", raw["initial_skills"]) if s.strip()
            ]
        for key in ("max_budget", "max_weeks"):
            if key in raw and raw[key] in ("", "null", "None"):
                raw[key] = None
        return raw

    def parse(self, text: str) -> GoalSchema:
        """
        Parses a natural-language career goal into a `GoalSchema`.
        Args:
            text (str): Raw user text.
        Returns:
            GoalSchema: Validated structured goal.
        Raises:
            ValueError: If the LLM output cannot be coerced or validated.
        """
        prompt = self._build_prompt(text)
        raw = self.client.chat_json(prompt=prompt, temperature=0.0)
        coerced = self._coerce(raw)
        try:
            return GoalSchema(**coerced)
        except ValidationError as exc:
            raise ValueError(
                f"LLM output did not match GoalSchema: {exc}\nRaw: {raw}"
            ) from exc
