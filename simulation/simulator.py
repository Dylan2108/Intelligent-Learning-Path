import random

class LearningSimulator:

    def __init__(self):
        pass

    def simulate(self, path):
        total_weeks = 0
        abandonment_probability = 0

        print('\n===== SIMULATION =====')

        for course in path:
            random_factor = random.uniform(0.8,1.3)
            estimated_time = int(random_factor * 4)

            total_weeks += estimated_time

            fatigue = random.uniform(0,0.15)
            abandonment_probability += fatigue

            print(f'Course: {course}')
            print(f'Estimated duration: {estimated_time} weeks')
            print('----------------------')

        abandonment_probability = min(abandonment_probability,1)

        print(f'Total estimated time: {total_weeks} weeks')
        print(f'Probability of abandonment: {abandonment_probability:.2f}')

        return {
            'total_weeks': total_weeks,
            'abandonment_probability': abandonment_probability
        }