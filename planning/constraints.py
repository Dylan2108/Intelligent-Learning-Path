import json
import logging

logger = logging.getLogger(__name__)

class ConstraintManager:
    """
    Handles prerequisite validation.
    """
    def __init__(self, prerequisites_path: str):
        """
        Loads prerequisites dataset.
        Args:
            prerequisites_path (str): Path to prerequisites JSON.
        """
        logger.debug("Loading prerequisites from %s", prerequisites_path)

        with open(prerequisites_path, 'r') as file:
            self.prerequisites = json.load(file)
        
        logger.info("ConstraintManager loaded %d prerequisite rules", len(self.prerequisites))
    
    def prerequisites_of(self, course_name : str) -> list[str]:
        """
        Returns prerequisites for a course.
        Args:
            course_name (str): Course name.
        Returns:
            list[str]: List of prerequisite courses.
        """

        result = []

        for relation in self.prerequisites:
            if relation['course'] == course_name:
                result.append(relation['prerequisite'])
        
        return result
    
    def can_take(self, completed_courses: set[str], course_name: str) -> bool:
        """
        Checks if a course can be taken.
        Args:
            completed_courses (set[str]): Completed courses.
            course_name (str): Desired course.
        Returns:
            bool: True if prerequisites are satisfied.
        """

        prerequisites = self.prerequisites_of(course_name)

        for prereq in prerequisites:
            if prereq not in completed_courses:
                return False
            
        return True