"""
Comparison harness for the A* planner, Genetic Algorithm, and Greedy baseline.

For each (initial_skills, career) instance, runs all algorithms N times and
records:
  - execution time (ms)
  - path length (number of courses)
  - total time (weeks)
  - total cost (dollars)
  - feasibility flag

Writes the raw measurements to `experiments/results.csv`.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from planning.career_planner import CareerPlanner
from planning.greedy import GreedySolver
from planning.metaheuristic import GeneticAlgorithmSolver

logger = logging.getLogger(__name__)

DEFAULT_CSV = Path("experiments/results.csv")

CAREER_INSTANCES: list[tuple[list[str], str]] = [
    (["Python"], "ML Engineer"),
    (["Python"], "Data Scientist"),
    (["Python"], "Data Engineer"),
    (["Python"], "MLOps Engineer"),
    (["Python"], "NLP Engineer"),
    (["Python"], "Computer Vision Engineer"),
    (["Python"], "Backend Developer"),
    ([], "Cloud Architect"),
    ([], "DevOps Engineer"),
    ([], "Data Analyst"),
    ([], "Backend Developer"),
    (["Python", "Statistics"], "Data Analyst"),
    (["Python", "SQL"], "Data Engineer"),
    (["Python", "Linux", "Docker"], "MLOps Engineer"),
    (["Python", "Statistics", "Linear Algebra", "Machine Learning"], "NLP Engineer"),
]


@dataclass
class RunResult:
    instance: str
    algorithm: str
    trial: int
    success: bool
    exec_time_ms: float
    path_length: int
    total_time: int
    total_cost: int


def _run_single(
    trial_label: str,
    algorithm: str,
    runner,
    initial: list[str],
    career: str,
    trial: int,
    ga_seed: int,
) -> RunResult:
    start = time.perf_counter()
    try:
        if algorithm == "A*":
            result = runner.plan(initial, career)
        elif algorithm == "GA":
            result = runner.solve(
                initial,
                career,
                max_budget=10_000,
                max_weeks=10_000,
                seed=ga_seed,
            )
        else:  # Greedy
            result = runner.solve(initial, career)
    except Exception as exc:  # noqa: BLE001
        logger.error("Trial %s failed: %s", trial_label, exc)
        return RunResult(trial_label, algorithm, trial, False, 0.0, 0, 0, 0)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if result is None:
        return RunResult(trial_label, algorithm, trial, False, elapsed_ms, 0, 0, 0)
    return RunResult(
        instance=trial_label,
        algorithm=algorithm,
        trial=trial,
        success=True,
        exec_time_ms=elapsed_ms,
        path_length=len(result.path),
        total_time=result.total_time,
        total_cost=result.total_cost,
    )


def _summarize(rows: list[RunResult]) -> dict[str, dict[str, dict[str, float]]]:
    """
    Aggregates raw trials by (instance, algorithm).
    """
    summary: dict[str, dict[str, dict[str, list[float] | float]]] = {}
    for row in rows:
        summary.setdefault(row.instance, {}).setdefault(row.algorithm, []).append(row)
    out: dict[str, dict[str, dict[str, float]]] = {}
    for inst, by_algo in summary.items():
        out[inst] = {}
        for algo, trials in by_algo.items():
            successes = [t for t in trials if t.success]
            if not successes:
                out[inst][algo] = {
                    "n_trials": len(trials),
                    "n_success": 0,
                    "success_rate": 0.0,
                    "mean_exec_ms": float("nan"),
                    "stdev_exec_ms": float("nan"),
                    "mean_path_len": float("nan"),
                    "mean_total_time": float("nan"),
                    "mean_total_cost": float("nan"),
                }
                continue
            exec_times = [t.exec_time_ms for t in successes]
            out[inst][algo] = {
                "n_trials": len(trials),
                "n_success": len(successes),
                "success_rate": len(successes) / len(trials),
                "mean_exec_ms": statistics.fmean(exec_times),
                "stdev_exec_ms": statistics.pstdev(exec_times) if len(exec_times) > 1 else 0.0,
                "mean_path_len": statistics.fmean(t.path_length for t in successes),
                "mean_total_time": statistics.fmean(t.total_time for t in successes),
                "mean_total_cost": statistics.fmean(t.total_cost for t in successes),
            }
    return out


def run(
    n_trials: int = 5,
    csv_path: Path = DEFAULT_CSV,
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Runs the comparison and writes a CSV. Returns the aggregated summary.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    astar = CareerPlanner()
    ga = GeneticAlgorithmSolver()
    greedy = GreedySolver()
    rows: list[RunResult] = []
    for initial, career in CAREER_INSTANCES:
        label = f"{career} | init={','.join(initial) or '∅'}"
        logger.info("Running %s", label)
        for trial in range(n_trials):
            rows.append(
                _run_single(
                    label, "A*", astar, initial, career, trial, ga_seed=trial,
                )
            )
            rows.append(
                _run_single(
                    label,
                    "GA",
                    ga,
                    initial,
                    career,
                    trial,
                    ga_seed=trial,
                )
            )
            rows.append(
                _run_single(
                    label,
                    "Greedy",
                    greedy,
                    initial,
                    career,
                    trial,
                    ga_seed=trial,
                )
            )

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    logger.info("Wrote %d rows to %s", len(rows), csv_path)

    return _summarize(rows)


def _print_table(summary: dict) -> None:
    print(f"\n{'INSTANCE':50s} | {'ALGO':3s} | {'succ':>5s} | {'ms':>8s} | {'len':>3s} | {'time':>4s} | {'cost':>5s}")
    print("-" * 100)
    for inst, by_algo in summary.items():
        for algo, m in by_algo.items():
            print(
                f"{inst:50s} | {algo:3s} | {int(m['n_success'])}/{int(m['n_trials'])} | "
                f"{m['mean_exec_ms']:8.2f} | {m['mean_path_len']:3.1f} | "
                f"{m['mean_total_time']:4.1f} | {m['mean_total_cost']:5.1f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    summary = run(n_trials=args.trials, csv_path=args.csv)
    _print_table(summary)


if __name__ == "__main__":
    main()
