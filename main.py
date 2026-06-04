import json
import logging
import sys
from pathlib import Path

from llm.evaluator import PathEvaluator
from llm.parser import GoalParser
from planning.career_planner import CareerPlanner
from planning.metaheuristic import GeneticAlgorithmSolver
from simulation.simulator import LearningSimulator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_GOAL = (
    "I want to become an ML Engineer. I already know Python, "
    "I have a budget of 200 and at most 30 weeks."
)


def _prompt_goal() -> str:
    print("Describe your career goal in natural language.")
    print("Press Enter to use the default goal.\n")
    user_text = input("> ").strip()
    return user_text or DEFAULT_GOAL


def _load_target_skills(career: str) -> set[str]:
    careers = json.loads(Path("data/careers.json").read_text())
    for c in careers:
        if c["career"] == career:
            return set(c["skills"])
    return set()


def _print_section(title: str) -> None:
    print(f"\n===== {title} =====")


def main() -> int:
    goal_text = _prompt_goal()
    logger.info("Parsing user goal with LLM...")
    goal = GoalParser().parse(goal_text)
    print("\n===== PARSED GOAL =====")
    print(goal.model_dump_json(indent=2))

    target_skills = _load_target_skills(goal.target_career)
    if not target_skills:
        print(f"Unknown career '{goal.target_career}'")
        return 1

    # --- 1. A* search
    _print_section("A* SEARCH")
    astar = CareerPlanner().plan(
        initial_skills=goal.initial_skills,
        target_career=goal.target_career,
    )
    if astar is None:
        print("No feasible trajectory found by A*.")
    else:
        for i, c in enumerate(astar.path, start=1):
            print(f"{i}. {c}")
        print(f"Total cost: {astar.total_cost} | Total time: {astar.total_time} weeks")

    # --- 2. Genetic Algorithm
    _print_section("GENETIC ALGORITHM")
    ga = GeneticAlgorithmSolver().solve(
        initial_skills=goal.initial_skills,
        target_career=goal.target_career,
        max_budget=goal.max_budget,
        max_weeks=goal.max_weeks,
    )
    if ga is None:
        print("GA did not find a feasible solution.")
    else:
        for i, c in enumerate(ga.path, start=1):
            print(f"{i}. {c}")
        print(f"Total cost: {ga.total_cost} | Total time: {ga.total_time} weeks")
        print(f"Fitness: {ga.fitness:.2f} | Generations: {ga.generations}")

    # --- 3. Stochastic simulation on the best (A*) path
    if astar is not None:
        _print_section("SIMULATION (A* PATH)")
        sim = LearningSimulator().simulate(astar.path)
        print(f"Estimated weeks: {sim['total_weeks']}")
        print(f"Abandonment probability: {sim['abandonment_probability']:.2f}")

    # --- 4. LLM evaluation
    if astar is not None:
        logger.info("Asking LLM to evaluate the A* trajectory...")
        try:
            ev = PathEvaluator().evaluate(
                career=goal.target_career,
                path=astar.path,
            )
            _print_section("LLM EVALUATION")
            print(f"Score: {ev.score}/10")
            print(f"Rationale: {ev.rationale}")
            if ev.weaknesses:
                print("Weaknesses:")
                for w in ev.weaknesses:
                    print(f"  - {w}")
            if ev.suggestions:
                print("Suggestions:")
                for s in ev.suggestions:
                    print(f"  - {s}")
        except Exception as exc:
            logger.warning("LLM evaluation failed: %s", exc)
            print("(LLM evaluation skipped — is Ollama running?)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
