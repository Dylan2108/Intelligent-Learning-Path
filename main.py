import logging
import sys

from llm.evaluator import PathEvaluator
from llm.parser import GoalParser
from planning.career_planner import CareerPlanner
from simulation.simulator import LearningSimulator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_GOAL = (
    "I want to become an ML Engineer. I already know Python, "
    "I have a budget of 200 and at most 20 weeks."
)


def _prompt_goal() -> str:
    print("Describe your career goal in natural language.")
    print("Press Enter to use the default goal.\n")
    user_text = input("> ").strip()
    return user_text or DEFAULT_GOAL


def main() -> int:
    goal_text = _prompt_goal()
    logger.info("Parsing user goal with LLM...")
    goal = GoalParser().parse(goal_text)
    print("\n===== PARSED GOAL =====")
    print(goal.model_dump_json(indent=2))

    logger.info("Planning trajectory with A*...")
    planner = CareerPlanner()
    result = planner.plan(
        initial_skills=goal.initial_skills,
        target_career=goal.target_career,
    )

    if result is None:
        print(
            f"\nNo feasible trajectory found for career '{goal.target_career}' "
            "with the current dataset."
        )
        return 1

    print("\n===== OPTIMAL PATH =====")
    for index, course in enumerate(result.path, start=1):
        print(f"{index}. {course}")
    print(f"\nTotal cost: {result.total_cost}")
    print(f"Total duration: {result.total_time} weeks")

    simulator = LearningSimulator()
    simulation_result = simulator.simulate(result.path)
    print("\n===== SIMULATION =====")
    print(f"Estimated total weeks: {simulation_result['total_weeks']}")
    print(f"Abandonment probability: {simulation_result['abandonment_probability']:.2f}")

    logger.info("Asking LLM to evaluate the trajectory...")
    evaluation = PathEvaluator().evaluate(
        career=goal.target_career,
        path=result.path,
    )

    print("\n===== LLM EVALUATION =====")
    print(f"Score: {evaluation.score}/10")
    print(f"Rationale: {evaluation.rationale}")
    if evaluation.weaknesses:
        print("Weaknesses:")
        for w in evaluation.weaknesses:
            print(f"  - {w}")
    if evaluation.suggestions:
        print("Suggestions:")
        for s in evaluation.suggestions:
            print(f"  - {s}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
