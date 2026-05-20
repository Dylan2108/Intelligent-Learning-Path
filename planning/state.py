class State:
    def __init__(self, completed_courses, path, total_cost, total_time):
        self.completed_courses = set(completed_courses)
        self.path = path
        self.total_cost = total_cost
        self.total_time = total_time
    
    def __lt__(self, other):
        return self.total_time < other.total_time