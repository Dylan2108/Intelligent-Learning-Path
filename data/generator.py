"""
Synthetic dataset generator for the career-path planning problem.

Reads `data/skills_seed.json` (hand-curated catalog) and produces:
  - data/courses.json       (2+ courses per skill + combo courses)
  - data/careers.json       (one entry per career, skills are skill names)
  - data/prerequisites.json (skill-to-skill prerequisite edges)

Design rationale:
  * Each skill has two courses (Intro, Deep) with different duration/cost.
  * Some skills are also covered by combo courses that teach two related
    skills at once, creating scenarios where the greedy heuristic (shortest
    course first) can yield suboptimal total time, while A* finds the optimum.
  * Prerequisites are defined between skills (not courses), matching the
    real-world constraint: you need certain knowledge before you can learn
    a more advanced topic.
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

# Pairs of related skills that will have a combo course teaching both.
# Combos create "traps" for the greedy shortest-first heuristic.
_COMBO_PAIRS: list[tuple[str, str, str]] = [
    ("Python", "Pandas", "Python for Data Analysis"),
    ("Python", "NumPy", "Scientific Computing with Python"),
    ("Python", "Machine Learning", "Machine Learning with Python"),
    ("SQL", "ETL", "Data Pipelines with SQL"),
    ("Docker", "Kubernetes", "Container Orchestration"),
    ("Linux", "Security", "Secure Linux Administration"),
    ("HTML/CSS", "React", "Frontend Web Development"),
    ("Statistics", "Probability", "Statistical Methods"),
    ("Deep Learning", "Computer Vision", "Computer Vision with DL"),
    ("Deep Learning", "NLP", "NLP with Deep Learning"),
]


def _flatten_skills(seed: dict) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for category in seed["categories"].values():
        for skill in category:
            if skill not in seen:
                seen.add(skill)
                out.append(skill)
    return sorted(out)


def _validate_acyclic(pairs: list[list[str]]) -> None:
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
    rng = random.Random(seed)
    seed_data = json.loads(SEED_PATH.read_text())

    skills = _flatten_skills(seed_data)
    _validate_acyclic(seed_data["prerequisite_rules"])
    skill_set = set(skills)

    courses: list[dict] = []
    course_id = 1

    # --- 1 course per skill (Intro) ---
    for skill in skills:
        intro_dur = rng.randint(3, 6)
        intro_cost = rng.randint(20, 50)
        courses.append({
            "id": course_id,
            "name": f"{skill} (Intro)",
            "duration": intro_dur,
            "cost": intro_cost,
            "difficulty": rng.randint(1, 3),
            "teaches": [skill],
        })
        course_id += 1

    # --- combo courses (teach 2 skills, duration < sum of intros) ---
    for skill_a, skill_b, combo_name in _COMBO_PAIRS:
        if skill_a not in skill_set or skill_b not in skill_set:
            continue
        intro_a = next(c for c in courses if c["name"] == f"{skill_a} (Intro)")
        intro_b = next(c for c in courses if c["name"] == f"{skill_b} (Intro)")
        combo_dur = max(int((intro_a["duration"] + intro_b["duration"]) * 0.7), 4)
        combo_cost = combo_dur * rng.randint(5, 9)
        courses.append({
            "id": course_id,
            "name": combo_name,
            "duration": combo_dur,
            "cost": combo_cost,
            "difficulty": rng.randint(2, 4),
            "teaches": [skill_a, skill_b],
        })
        course_id += 1

    # --- Prerequisites (skill-based, not course-based) ---
    prerequisites: list[dict] = []
    for course_skill, prereq_skill in seed_data["prerequisite_rules"]:
        if course_skill in skill_set and prereq_skill in skill_set:
            prerequisites.append({"skill": course_skill, "prerequisite": prereq_skill})

    # --- Careers (unchanged: skills are skill names) ---
    careers = seed_data["careers"]
    for career in careers:
        missing = [s for s in career["skills"] if s not in skill_set]
        if missing:
            raise ValueError(
                f"Career '{career['career']}' references unknown skills: {missing}"
            )

    COURSES_PATH.write_text(json.dumps(courses, indent=2))
    CAREERS_PATH.write_text(json.dumps(careers, indent=2))
    PREREQ_PATH.write_text(json.dumps(prerequisites, indent=2))

    combo_count = course_id - 1 - len(skills)
    logger.info(
        "Generated %d courses (%d per-skill + %d combo), %d careers, %d prereq edges (seed=%d)",
        len(courses), len(skills), combo_count, len(careers), len(prerequisites), seed,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    generate()
