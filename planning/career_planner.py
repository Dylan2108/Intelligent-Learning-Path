import heapq
import json
import logging

from planning.state import State
from planning.constraints import ConstraintManager

logger = logging.getLogger(__name__)

class CareerPlanner:
    """
    A* planner for career trajectories.
    """

    def __init__(self):
        """
        Initializes planner and loads datasets.
        """
        logger.debug("Loading course and career datasets")

        with open('data/courses.json','r') as file:
            self.courses = json.load(file)
        
        with open('data/careers.json','r') as file:
            self.careers = json.load(file)
        
        self.constraints = ConstraintManager(
            'data/prerequisites.json'
        )
        logger.info("CareerPlanner initialized with %d courses, %d careers",
                     len(self.courses), len(self.careers))
    
    def get_course(self, name: str) -> dict:
        """
        Returns course information.
        Args:
            name (str): Course name.
        Returns:
            dict: Course information.
        """

        for course in self.courses:
            if course['name'] == name:
                return course
        
        return {}
    
    def target_skills(self, career_name: str) -> set[str]:
        """
        Returns required skills for a career.
        Args:
            career_name (str): Career objective.
        Returns:
            set[str]: Required skills.
        """

        for career in self.careers:
            if career['career'] == career_name:
                return set(career['skills'])
        
        return set()
    
    def heuristic(self, completed: set[str], target: set[str]) -> int:
        """
        Admissible and consistent heuristic for A*.
        Estimates the remaining time (in weeks) needed to acquire the
        missing target skills. Since the model treats a target "skill"
        as a course name, the lower bound is the sum of the durations
        of the missing target courses that are still available.
        Ignoring transitive prerequisites keeps h admissible (the real
        path may need extra time to satisfy them).
        Args:
            completed (set[str]): Completed skills / courses.
            target (set[str]): Goal skill set.
        Returns:
            int: Estimated remaining time in weeks.
        """

        missing = target - completed
        if not missing:
            return 0
        duration_by_name = {c['name']: c['duration'] for c in self.courses}
        return sum(duration_by_name.get(skill, 0) for skill in missing)
    
    def plan(self, initial_skills: list[str], target_career: str) -> State | None:
        """
        Generates optimal career path.
        Args:
            initial_skills (list[str]): User skills.
            target_career (str): Desired career.
        Returns:
            State | None: Final state or None.
        """
        logger.info("Starting A* search: career=%s, initial_skills=%s",
                     target_career, initial_skills)

        target = self.target_skills(target_career)
        if not target:
            logger.warning("No target skills found for career: %s", target_career)
            return None

        start = State(
            completed_courses = set(initial_skills),
            path = [],
            total_cost = 0,
            total_time = 0
        )

        frontier : list[tuple[int,State]] = []

        heapq.heappush(
            frontier,
            (
                self.heuristic(start.completed_courses,target),
                start
            )
        )

        visited = set()
        nodes_expanded = 0

        while frontier:
            _,current = heapq.heappop(frontier)

            completed_tuple = tuple(sorted(current.completed_courses))

            if completed_tuple in visited:
                continue

            visited.add(completed_tuple)
            nodes_expanded += 1

            if target.issubset(current.completed_courses):
                logger.info("A* found solution: %d courses, cost=%d, time=%d weeks, nodes_expanded=%d",
                            len(current.path), current.total_cost, current.total_time, nodes_expanded)
                return current
            
            for course in self.courses:

                course_name = course['name']

                if course_name in current.completed_courses:
                    continue

                if not self.constraints.can_take(
                    current.completed_courses,
                    course_name
                ):
                    continue

                new_completed = set(current.completed_courses)
                new_completed.add(course_name)

                new_path = current.path + [course_name]
                new_state = State(
                    completed_courses=new_completed,
                    path=new_path,
                    total_cost=current.total_cost + course['cost'],
                    total_time=current.total_time + course['duration']
                )

                priority = (
                    new_state.total_time + self.heuristic(new_completed,target)
                )

                heapq.heappush(
                    frontier,
                    (priority,new_state)
                )
        
        logger.warning("A* search failed: no feasible path found after expanding %d nodes", nodes_expanded)
        return None