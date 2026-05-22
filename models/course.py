class Course:
    """
    Represents a course inside the learning system.
    """
    def __init__(self, name: str, difficulty: int, duration: int, cost: int):
        """
        Initializes a course.
        Args:
            name (str): Name of the course.
            difficulty (int): Difficulty level.
            duration (int): Estimated duration in weeks.
            cost (int): Monetary cost.
        """
        self.name = name
        self.difficulty = difficulty
        self.duration = duration
        self.cost = cost
    
    def __repr__(self) -> str:
        """
        Returns string representation.
        Returns:
            str: Course name.
        """
        return self.name