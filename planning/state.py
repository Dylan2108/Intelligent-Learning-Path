from __future__ import annotations


class State:
    """
    Represents a search state for the A* algorithm.

    Tracks both the set of completed courses (for path reconstruction)
    and the set of covered skills (for goal checking / heuristic).
    """

    def __init__(
        self,
        completed_courses: set[str],
        covered_skills: set[str],
        path: list[str],
        total_cost: int,
        total_time: int,
    ):
        self.completed_courses = set(completed_courses)
        self.covered_skills = set(covered_skills)
        self.path = path
        self.total_cost = total_cost
        self.total_time = total_time

    def __lt__(self, other: State) -> bool:
        return self.total_time < other.total_time
