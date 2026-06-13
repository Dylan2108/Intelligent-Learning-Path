"""
Greedy solver for the career-path planning problem.

Strategy: at each step, pick the shortest available course that
teaches a missing target skill (or a transitive prerequisite of one)
and whose prerequisite skills are already covered.

This serves as a fast baseline.  With single-skill courses it coincides
with optimal, but combo courses (teaching >1 skill) can expose greedy's
suboptimality.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from planning.constraints import ConstraintManager

logger = logging.getLogger(__name__)


@dataclass
class GreedyResult:
    path: list[str] = field(default_factory=list)
    total_cost: int = 0
    total_time: int = 0
    courses_taken: int = 0


class GreedySolver:
    """
    Greedy solver: always picks the shortest available course that
    contributes to covering the career's required skills.
    """

    def __init__(
        self,
        courses_path: str = "data/courses.json",
        prerequisites_path: str = "data/prerequisites.json",
        careers_path: str = "data/careers.json",
    ):
        self.courses = json.loads(Path(courses_path).read_text())
        self.careers = json.loads(Path(careers_path).read_text())
        self.courses_by_name = {c["name"]: c for c in self.courses}
        self.duration_by_name = {c["name"]: c["duration"] for c in self.courses}
        self.cost_by_name = {c["name"]: c["cost"] for c in self.courses}
        self.constraints = ConstraintManager(prerequisites_path, courses_path)

    def _target_skills(self, career_name: str) -> set[str]:
        for c in self.careers:
            if c["career"] == career_name:
                return set(c["skills"])
        return set()

    def _skills_needed(self, target: set[str]) -> set[str]:
        needed = set(target)
        stack = list(target)
        while stack:
            skill = stack.pop()
            for prereq in self.constraints.skill_prereqs.get(skill, set()):
                if prereq not in needed:
                    needed.add(prereq)
                    stack.append(prereq)
        return needed

    def _covered_skills(
        self, completed: set[str], initial: set[str]
    ) -> set[str]:
        skills = set(initial)
        for name in completed:
            course = self.courses_by_name.get(name, {})
            skills.update(course.get("teaches", []))
        return skills

    def _candidate_courses(
        self, completed: set[str], covered: set[str], target: set[str]
    ) -> list[str]:
        needed = self._skills_needed(target)
        candidates = []
        for course in self.courses:
            name = course["name"]
            if name in completed:
                continue
            if not self.constraints.can_take(covered, course):
                continue
            if set(course["teaches"]) & needed:
                candidates.append(name)
        return candidates

    def solve(
        self,
        initial_skills: list[str],
        target_career: str,
        max_budget: int | None = None,
        max_weeks: int | None = None,
    ) -> GreedyResult | None:
        logger.info(
            "Starting greedy search: career=%s, initial_skills=%s, max_budget=%s, max_weeks=%s",
            target_career,
            initial_skills,
            max_budget,
            max_weeks,
        )

        target = self._target_skills(target_career)
        if not target:
            logger.warning("No target skills found for career: %s", target_career)
            return None

        initial = set(initial_skills)
        completed: set[str] = set()
        covered = set(initial)
        path: list[str] = []
        total_cost = 0
        total_time = 0

        if target.issubset(covered):
            logger.info("Greedy: all target skills already covered")
            return GreedyResult(
                path=[], total_cost=0, total_time=0, courses_taken=0
            )

        max_iterations = len(self.courses) + 1
        for _ in range(max_iterations):
            candidates = self._candidate_courses(completed, covered, target)
            if not candidates:
                logger.warning(
                    "Greedy: no candidates available, target may be unreachable"
                )
                return None

            # Filter by budget/time constraints
            feasible = [
                c for c in candidates
                if (max_budget is None or total_cost + self.cost_by_name.get(c, 0) <= max_budget)
                and (max_weeks is None or total_time + self.duration_by_name.get(c, 0) <= max_weeks)
            ]
            if not feasible:
                logger.warning(
                    "Greedy: no feasible candidates within budget/time constraints"
                )
                return None

            best = min(
                feasible,
                key=lambda c: self.duration_by_name.get(c, float("inf")),
            )
            course = self.courses_by_name[best]
            completed.add(best)
            covered.update(course["teaches"])
            path.append(best)
            total_cost += course["cost"]
            total_time += course["duration"]

            logger.debug(
                "Greedy picked: %s (duration=%d, cost=%d)",
                best,
                course["duration"],
                course["cost"],
            )

            if target.issubset(covered):
                logger.info(
                    "Greedy found solution: %d courses, cost=%d, time=%d weeks",
                    len(path),
                    total_cost,
                    total_time,
                )
                return GreedyResult(
                    path=path,
                    total_cost=total_cost,
                    total_time=total_time,
                    courses_taken=len(path),
                )

        logger.warning("Greedy: max iterations reached without covering target")
        return None
