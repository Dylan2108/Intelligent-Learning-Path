import heapq
import json
import logging

from planning.state import State
from planning.constraints import ConstraintManager

logger = logging.getLogger(__name__)


class CareerPlanner:
    """
    A* planner for career trajectories.

    State tracks covered_skills (not just completed courses).  A course is
    available if all prerequisite SKILLS of the skills it teaches are already
    covered.
    """

    def __init__(self):
        logger.debug("Loading course and career datasets")
        with open("data/courses.json") as f:
            self.courses = json.load(f)
        with open("data/careers.json") as f:
            self.careers = json.load(f)
        self.courses_by_name = {c["name"]: c for c in self.courses}
        self.constraints = ConstraintManager(
            "data/prerequisites.json", "data/courses.json"
        )
        self._frac_min_per_skill: dict[str, int] = {}
        for skill_name in {s for c in self.courses for s in c["teaches"]}:
            self._frac_min_per_skill[skill_name] = int(
                min(
                    c["duration"] / len(c["teaches"])
                    for c in self.courses
                    if skill_name in c["teaches"]
                )
            )
        logger.info(
            "CareerPlanner initialized with %d courses, %d careers",
            len(self.courses),
            len(self.careers),
        )

    def get_course(self, name: str) -> dict:
        return self.courses_by_name.get(name, {})

    def target_skills(self, career_name: str) -> set[str]:
        for career in self.careers:
            if career["career"] == career_name:
                return set(career["skills"])
        return set()

    def heuristic(self, covered_skills: set[str], target: set[str]) -> int:
        missing = target - covered_skills
        if not missing:
            return 0
        return sum(self._frac_min_per_skill.get(s, 0) for s in missing)

    def plan(
        self,
        initial_skills: list[str],
        target_career: str,
        max_budget: int | None = None,
        max_weeks: int | None = None,
    ) -> State | None:
        logger.info(
            "Starting A* search: career=%s, initial_skills=%s, max_budget=%s, max_weeks=%s",
            target_career,
            initial_skills,
            max_budget,
            max_weeks,
        )

        target = self.target_skills(target_career)
        if not target:
            logger.warning("No target skills found for career: %s", target_career)
            return None

        start = State(
            completed_courses=set(),
            covered_skills=set(initial_skills),
            path=[],
            total_cost=0,
            total_time=0,
        )

        frontier: list[tuple[int, State]] = []
        heapq.heappush(frontier, (self.heuristic(start.covered_skills, target), start))

        visited: set[tuple[str, ...]] = set()
        nodes_expanded = 0

        while frontier:
            _, current = heapq.heappop(frontier)

            visited_key = tuple(sorted(current.completed_courses))
            if visited_key in visited:
                continue
            visited.add(visited_key)
            nodes_expanded += 1

            if target.issubset(current.covered_skills):
                logger.info(
                    "A* found solution: %d courses, cost=%d, time=%d weeks, nodes_expanded=%d",
                    len(current.path),
                    current.total_cost,
                    current.total_time,
                    nodes_expanded,
                )
                return current

            for course in self.courses:
                name = course["name"]
                if name in current.completed_courses:
                    continue
                if not self.constraints.can_take(current.covered_skills, course):
                    continue

                new_cost = current.total_cost + course["cost"]
                new_time = current.total_time + course["duration"]

                if max_budget is not None and new_cost > max_budget:
                    continue
                if max_weeks is not None and new_time > max_weeks:
                    continue

                new_completed = set(current.completed_courses)
                new_completed.add(name)

                new_covered = set(current.covered_skills)
                new_covered.update(course["teaches"])

                new_state = State(
                    completed_courses=new_completed,
                    covered_skills=new_covered,
                    path=current.path + [name],
                    total_cost=new_cost,
                    total_time=new_time,
                )

                priority = new_state.total_time + self.heuristic(
                    new_covered, target
                )
                heapq.heappush(frontier, (priority, new_state))

        logger.warning(
            "A* search failed: no feasible path found after expanding %d nodes",
            nodes_expanded,
        )
        return None
