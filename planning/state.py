from __future__ import annotations

class State:
    """
    Represents a search state for the A* algorithm.
    A state stores:
    - completed courses
    - trajectory path
    - accumulated cost
    - accumulated time
    """

    def __init__(self, completed_courses: set[str], path: list[str], total_cost: int, total_time: int):
        """
        Initializes a search state.
        Args:
            completed_courses (set[str]): Courses already completed.
            path (list[str]): Ordered trajectory of courses.
            total_cost (int): Total accumulated monetary cost.
            total_time (int): Total accumulated duration.
        """

        self.completed_courses = set(completed_courses)
        self.path = path
        self.total_cost = total_cost
        self.total_time = total_time
    
    def __lt__(self, other: State) -> bool:
        """
        Comparison operator used by heapq.
        Args:
            other (State): Another state.
        Returns:
            bool: True if current state has lower total time.
        """
        return self.total_time < other.total_time