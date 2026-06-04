"""
Synthetic dataset generator for the career-path planning problem.

Reads `data/skills_seed.json` (hand-curated catalog) and produces:
  - data/courses.json       (one entry per skill)
  - data/careers.json       (one entry per career)
  - data/prerequisites.json (edges of the prerequisite DAG)

Design rationale (for the technical report):
  * One course per skill: a course named after the skill it teaches.
    This matches the existing modeling where careers list course names
    as required "skills".
  * Each course gets difficulty (1-5), duration (weeks, 2-12) and cost
    (dollars, 0-100) drawn deterministically from the seed.
  * Prerequisites are declared in the seed and are guaranteed to form
    an acyclic graph (Kahn's algorithm validates this).
  * The resulting dataset is reproducible: same seed -> same files.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)

SEED_PATH = Path("data/skills_seed.json")
COURSES_PATH = Path("data/courses.json")
CAREERS_PATH = Path("data/careers.json")
PREREQ_PATH = Path("data/prerequisites.json")

DEFAULT_SEED = 20251115


def _flatten_skills(seed: dict) -> list[str]:
    """
    Returns the alphabetical list of unique skill names from the seed.
    """
    seen: set[str] = set()
    out: list[str] = []
    for category in seed["categories"].values():
        for skill in category:
            if skill not in seen:
                seen.add(skill)
                out.append(skill)
    return sorted(out)


def _validate_acyclic(pairs: list[list[str]]) -> None:
    """
    Validates that the prerequisite graph is a DAG. Raises if not.
    """
    adj: dict[str, list[str]] = {}
    indeg: dict[str, int] = {}
    nodes: set[str] = set()
    for course, pre in pairs:
        nodes.add(course)
        nodes.add(pre)
        adj.setdefault(pre, []).append(course)
        adj.setdefault(course, [])
        indeg.setdefault(course, 0)
        indeg[pre] = indeg.get(pre, 0)
    for course, pre in pairs:
        indeg[course] = indeg.get(course, 0) + 1

    queue = sorted([n for n, d in indeg.items() if d == 0])
    visited = 0
    while queue:
        n = queue.pop(0)
        visited += 1
        for m in adj.get(n, []):
            indeg[m] -= 1
            if indeg[m] == 0:
                queue.append(m)
    if visited != len(nodes):
        raise ValueError("Prerequisite graph contains a cycle")


def generate(seed: int = DEFAULT_SEED) -> None:
    """
    Generates the three JSON files in `data/`. Idempotent: overwrite.
    """
    rng = random.Random(seed)
    seed_data = json.loads(SEED_PATH.read_text())

    skills = _flatten_skills(seed_data)
    _validate_acyclic(seed_data["prerequisite_rules"])

    # Courses
    courses: list[dict] = []
    for index, skill in enumerate(skills, start=1):
        difficulty = rng.randint(1, 5)
        duration = rng.randint(2, 12)
        cost = rng.randint(0, 30) * (difficulty + 1)
        courses.append(
            {
                "id": index,
                "name": skill,
                "difficulty": difficulty,
                "duration": duration,
                "cost": cost,
            }
        )

    # Prerequisites
    prerequisites: list[dict] = []
    for course, pre in seed_data["prerequisite_rules"]:
        if course in skills and pre in skills:
            prerequisites.append({"course": course, "prerequisite": pre})

    # Careers (verbatim from seed)
    careers = seed_data["careers"]
    # Sanity check: every skill referenced by a career must exist
    skill_set = set(skills)
    for career in careers:
        missing = [s for s in career["skills"] if s not in skill_set]
        if missing:
            raise ValueError(
                f"Career '{career['career']}' references unknown skills: {missing}"
            )

    COURSES_PATH.write_text(json.dumps(courses, indent=2))
    CAREERS_PATH.write_text(json.dumps(careers, indent=2))
    PREREQ_PATH.write_text(json.dumps(prerequisites, indent=2))

    logger.info(
        "Generated %d courses, %d careers, %d prerequisite edges (seed=%d)",
        len(courses),
        len(careers),
        len(prerequisites),
        seed,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    generate()
