import logging
import random

logger = logging.getLogger(__name__)


class LearningSimulator:

    def __init__(self, courses: dict | None = None):
        self._courses = courses or {}

    def _get_duration(self, course_id: str) -> int:
        if self._courses and course_id in self._courses:
            return self._courses[course_id].get("duration", 4)
        return 4

    def _get_difficulty(self, course_id: str) -> float:
        if self._courses and course_id in self._courses:
            return self._courses[course_id].get("difficulty", 0.5)
        return 0.5

    def simulate(self, path: list[str], n_runs: int = 1000) -> dict:
        if not path:
            logger.warning("Empty path provided to simulator")
            return {"total_weeks": 0, "abandonment_probability": 0}

        logger.info("Monte Carlo simulation: %d runs, %d courses", n_runs, len(path))

        weekly_results = []
        survival_results = []

        for _ in range(n_runs):
            total_weeks = 0
            survival_prob = 1.0

            for course_id in path:
                duration = self._get_duration(course_id)
                difficulty = self._get_difficulty(course_id)

                random_factor = random.uniform(0.8, 1.3)
                estimated_time = round(random_factor * duration)
                total_weeks += estimated_time

                fail_prob = min(difficulty * random.uniform(0.05, 0.15), 1.0)
                survival_prob *= (1 - fail_prob)

            weekly_results.append(total_weeks)
            survival_results.append(survival_prob)

        avg_weeks = sum(weekly_results) / n_runs
        avg_abandonment = 1 - (sum(survival_results) / n_runs)

        result = {
            "total_weeks": round(avg_weeks),
            "abandonment_probability": round(avg_abandonment, 2),
        }

        logger.info("Simulation result: weeks=%d, abandon=%.2f", result["total_weeks"], result["abandonment_probability"])

        return result