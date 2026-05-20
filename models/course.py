class Course:
    def __init__(self, name, difficulty, duration, cost):
        self.name = name
        self.difficulty = difficulty
        self.duration = duration
        self.cost = cost
    
    def __repr__(self):
        return self.name