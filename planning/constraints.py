import pandas as pd

class ConstraintManager:
    def __init__(self, prerequisites_path):
        #Obtener los prerequisitos del dataset
        return 1
    
    def prerequisites_of(self, course_name):
        #Obtener prerequisitos de course_name
        return 1
    
    def can_take(self, completed_courses, course_name):
        prerequisites = self.prerequisites_of(course_name)

        for prereq in prerequisites:
            if prereq not in completed_courses:
                return False
            
            return True