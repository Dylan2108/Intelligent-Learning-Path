import time

from planning.career_planner import CareerPlanner

planner = CareerPlanner()

initial_skills = ['Python']
career = 'ML Engineer'

start = time.time()

result = planner.plan(initial_skills,career)

end = time.time()

print('\n===== EXPERIMENT =====')
print(f'Execution time: {end-start:.4f} seconds')
print(f'Path found: {result.path}')
print(f'Total time: {result.total_time}')
print(f'Total cost: {result.total_cost}')