import json
import logging
import sys
from pathlib import Path

from llm.evaluator import PathEvaluator
from llm.parser import GoalParser
from planning.career_planner import CareerPlanner
from planning.greedy import GreedySolver
from planning.metaheuristic import GeneticAlgorithmSolver
from simulation.simulator import LearningSimulator


def setup_logging(level: int = logging.INFO) -> None:
    """Configure centralized logging for the application."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Suppress overly noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("ollama").setLevel(logging.WARNING)


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
    setup_logging()

    try:
        goal_text = _prompt_goal()
        logger.info("Parsing user goal with LLM...")
        goal = GoalParser().parse(goal_text)
        logger.info("Parsed goal: career=%s, skills=%s, budget=%s, weeks=%s",
                     goal.target_career, goal.initial_skills, goal.max_budget, goal.max_weeks)
        print("\n===== PARSED GOAL =====")
        print(goal.model_dump_json(indent=2))

        target_skills = _load_target_skills(goal.target_career)
        if not target_skills:
            logger.error("Unknown career: %s", goal.target_career)
            print(f"Unknown career '{goal.target_career}'")
            return 1

        # --- 1. A* search
        _print_section("A* SEARCH")
        astar = CareerPlanner().plan(
            initial_skills=goal.initial_skills,
            target_career=goal.target_career,
            max_budget=goal.max_budget,
            max_weeks=goal.max_weeks,
        )
        if astar is None:
            logger.warning("A* found no feasible trajectory")
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
            logger.warning("GA found no feasible solution")
            print("GA did not find a feasible solution.")
        else:
            for i, c in enumerate(ga.path, start=1):
                print(f"{i}. {c}")
            print(f"Total cost: {ga.total_cost} | Total time: {ga.total_time} weeks")
            print(f"Fitness: {ga.fitness:.2f} | Generations: {ga.generations}")

        # --- 3. Greedy search (baseline)
        _print_section("GREEDY SEARCH")
        greedy = GreedySolver().solve(
            initial_skills=goal.initial_skills,
            target_career=goal.target_career,
            max_budget=goal.max_budget,
            max_weeks=goal.max_weeks,
        )
        if greedy is None:
            logger.warning("Greedy found no feasible solution")
            print("Greedy did not find a feasible solution.")
        else:
            for i, c in enumerate(greedy.path, start=1):
                print(f"{i}. {c}")
            print(f"Total cost: {greedy.total_cost} | Total time: {greedy.total_time} weeks")

        # --- 4. Stochastic simulation on the best (A*) path
        if astar is not None:
            _print_section("SIMULATION (A* PATH)")
            sim = LearningSimulator().simulate(astar.path)
            print(f"Estimated weeks: {sim['total_weeks']}")
            print(f"Abandonment probability: {sim['abandonment_probability']:.2f}")

        # --- 5. LLM evaluation
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

        logger.info("Career planning completed successfully")
        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        print("\nInterrupted by user.")
        return 130
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
