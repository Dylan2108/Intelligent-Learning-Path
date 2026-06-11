import logging
import random

logger = logging.getLogger(__name__)

class LearningSimulator:

    def __init__(self):
        pass

    def simulate(self, path):
        if not path:
            logger.warning("Empty path provided to simulator")
            return {'total_weeks': 0, 'abandonment_probability': 0}

        logger.info("Starting stochastic simulation for %d courses", len(path))
        total_weeks = 0
        abandonment_probability = 0

        for course in path:
            random_factor = random.uniform(0.8,1.3)
            estimated_time = int(random_factor * 4)

            total_weeks += estimated_time

            fatigue = random.uniform(0,0.15)
            abandonment_probability += fatigue

            logger.debug("Course: %s, estimated_duration: %d weeks", course, estimated_time)

        abandonment_probability = min(abandonment_probability,1)

        logger.info("Simulation completed: total_weeks=%d, abandonment_probability=%.2f",
                     total_weeks, abandonment_probability)

        return {
            'total_weeks': total_weeks,
            'abandonment_probability': abandonment_probability
        }