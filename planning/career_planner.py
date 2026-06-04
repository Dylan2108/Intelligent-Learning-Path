import heapq
import json

from planning.state import State
from planning.constraints import ConstraintManager

class CareerPlanner:
    """
    A* planner for career trajectories.
    """

    def __init__(self):
        """
        Initializes planner and loads datasets.
        """

        with open('data/courses.json','r') as file:
            self.courses = json.load(file)
        
        with open('data/careers.json','r') as file:
            self.careers = json.load(file)
        
        self.constraints = ConstraintManager(
            'data/prerequisites.json'
        )
    
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
        Heuristic function for A*.
        Args:
            completed (set[str]): Completed skills.
            target (set[str]): Goal skills.
        Returns:
            int: Estimated remaining cost.
        """
        
        return len(target) - len(completed)
    
    def plan(self, initial_skills: list[str], target_career: str) -> State | None:
        """
        Generates optimal career path.
        Args:
            initial_skills (list[str]): User skills.
            target_career (str): Desired career.
        Returns:
            State | None: Final state or None.
        """

        target = self.target_skills(target_career)

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

        while frontier:
            _,current = heapq.heappop(frontier)

            completed_tuple = tuple(sorted(current.completed_courses))

            if completed_tuple in visited:
                continue

            visited.add(completed_tuple)

            if target.issubset(current.completed_courses):
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
        
        return None