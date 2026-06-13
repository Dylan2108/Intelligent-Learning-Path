import json
import logging

logger = logging.getLogger(__name__)


class ConstraintManager:
    """
    Handles prerequisite validation between skills.

    Prerequisites are defined at the skill level:
      to take a course that teaches skill X, you must have already
      covered all prerequisite skills of X.
    """

    def __init__(self, prerequisites_path: str, courses_path: str = ""):
        logger.debug("Loading prerequisites from %s", prerequisites_path)
        with open(prerequisites_path) as f:
            raw = json.load(f)
        self.skill_prereqs: dict[str, set[str]] = {}
        for entry in raw:
            skill = entry.get("skill", entry.get("course", ""))
            prereq = entry.get("prerequisite", "")
            self.skill_prereqs.setdefault(skill, set()).add(prereq)
        logger.info(
            "ConstraintManager loaded %d prerequisite rules for %d skills",
            len(raw),
            len(self.skill_prereqs),
        )

        if courses_path:
            with open(courses_path) as f:
                self.courses_by_name = {c["name"]: c for c in json.load(f)}
        else:
            self.courses_by_name = {}

    def prerequisites_of_skill(self, skill: str) -> set[str]:
        return self.skill_prereqs.get(skill, set())

    def skills_taught_by(self, course_name: str) -> list[str]:
        return self.courses_by_name.get(course_name, {}).get("teaches", [])

    def can_take(self, covered_skills: set[str], course: dict) -> bool:
        taught = set(course.get("teaches", []))
        for skill in taught:
            for prereq in self.skill_prereqs.get(skill, set()):
                if prereq not in taught and prereq not in covered_skills:
                    return False
        return True
