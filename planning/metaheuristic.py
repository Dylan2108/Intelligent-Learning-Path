"""
Genetic Algorithm metaheuristic for the career-path planning problem.

Representation: permutation of candidate course names.
Fitness: weighted sum of (time, cost, uncovered_skills).
Operators: Order Crossover (OX), swap mutation.
Repair: after crossover/mutation, reorder courses so that skill
prerequisites are satisfied, dropping irrelevant courses.
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

from planning.constraints import ConstraintManager

logger = logging.getLogger(__name__)


@dataclass
class GAResult:
    path: list[str] = field(default_factory=list)
    total_cost: int = 0
    total_time: int = 0
    fitness: float = 0.0
    generations: int = 0
    feasible: bool = False


class GeneticAlgorithmSolver:
    """
    GA-based metaheuristic that searches over course permutations.
    """

    def __init__(
        self,
        courses_path: str = "data/courses.json",
        prerequisites_path: str = "data/prerequisites.json",
        careers_path: str = "data/careers.json",
    ):
        self.courses = json.loads(Path(courses_path).read_text())
        self.careers = json.loads(Path(careers_path).read_text())
        self.courses_by_name = {c["name"]: c for c in self.courses}
        self.duration_by_name = {c["name"]: c["duration"] for c in self.courses}
        self.cost_by_name = {c["name"]: c["cost"] for c in self.courses}
        self.constraints = ConstraintManager(prerequisites_path, courses_path)

    def _target_skills(self, career_name: str) -> set[str]:
        for c in self.careers:
            if c["career"] == career_name:
                return set(c["skills"])
        return set()

    def _skills_needed(self, target: set[str]) -> set[str]:
        needed = set(target)
        stack = list(target)
        while stack:
            skill = stack.pop()
            for prereq in self.constraints.skill_prereqs.get(skill, set()):
                if prereq not in needed:
                    needed.add(prereq)
                    stack.append(prereq)
        return needed

    def _candidate_courses(
        self, initial: set[str], target: set[str]
    ) -> list[str]:
        needed = self._skills_needed(target)
        return [
            c["name"]
            for c in self.courses
            if c["name"] not in initial and set(c["teaches"]) & needed
        ]

    def _topological_repair(
        self, sequence: list[str], initial_skills: set[str]
    ) -> list[str]:
        covered = set(initial_skills)
        ordered: list[str] = []
        remaining = list(sequence)
        while True:
            changed = False
            still_remaining: list[str] = []
            for name in remaining:
                course = self.courses_by_name.get(name, {})
                if self.constraints.can_take(covered, course):
                    ordered.append(name)
                    covered.update(course.get("teaches", []))
                    changed = True
                else:
                    still_remaining.append(name)
            remaining = still_remaining
            if not changed:
                break
        # Courses that remain cannot satisfy prerequisites — drop them
        return ordered

    def _decode(
        self, chromosome: list[str], initial: set[str], target: set[str]
    ) -> list[str]:
        needed = self._skills_needed(target)
        covered: set[str] = set(initial) & needed
        kept: list[str] = []
        for name in chromosome:
            course = self.courses_by_name.get(name, {})
            new_skills = set(course.get("teaches", [])) & needed - covered
            if new_skills:
                kept.append(name)
                covered.update(new_skills)
        return self._topological_repair(kept, initial)

    def _fitness(
        self,
        chromosome: list[str],
        initial: set[str],
        target: set[str],
        max_budget: int | None,
        max_weeks: int | None,
        weights: dict[str, float],
    ) -> tuple[float, bool]:
        path = self._decode(chromosome, initial, target)
        needed = self._skills_needed(target)
        covered = set(initial) & needed
        for name in path:
            covered.update(
                set(self.courses_by_name.get(name, {}).get("teaches", [])) & needed
            )
        total_cost = sum(self.cost_by_name.get(c, 0) for c in path)
        total_time = sum(self.duration_by_name.get(c, 0) for c in path)
        missing = target - covered
        feasible = (
            (max_budget is None or total_cost <= max_budget)
            and (max_weeks is None or total_time <= max_weeks)
            and not missing
        )

        fit = (
            weights.get("time", 1.0) * total_time
            + weights.get("cost", 0.5) * total_cost
            + weights.get("missing", 100.0) * len(missing)
            + weights.get("length", 0.0) * len(path)
        )
        if not feasible:
            fit += 1000.0
        return fit, feasible

    def _random_chromosome(
        self, candidates: list[str], rng: random.Random
    ) -> list[str]:
        chrom = list(candidates)
        rng.shuffle(chrom)
        return chrom

    def _crossover_ox(
        self,
        parent_a: list[str],
        parent_b: list[str],
        rng: random.Random,
    ) -> tuple[list[str], list[str]]:
        n = len(parent_a)
        if n < 2:
            return list(parent_a), list(parent_b)
        i, j = sorted(rng.sample(range(n), 2))

        def ox(p1: list[str], p2: list[str]) -> list[str]:
            middle = p1[i:j]
            remaining = [x for x in p2 if x not in middle]
            return remaining[:i] + middle + remaining[i:]

        return ox(parent_a, parent_b), ox(parent_b, parent_a)

    def _mutate_swap(
        self, chrom: list[str], rate: float, rng: random.Random
    ) -> list[str]:
        out = list(chrom)
        for i in range(len(out)):
            if rng.random() < rate:
                j = rng.randrange(len(out))
                out[i], out[j] = out[j], out[i]
        return out

    def solve(
        self,
        initial_skills: list[str],
        target_career: str,
        max_budget: int | None = None,
        max_weeks: int | None = None,
        weights: dict[str, float] | None = None,
        population_size: int = 50,
        generations: int = 100,
        mutation_rate: float = 0.1,
        elite_size: int = 2,
        seed: int | None = 42,
    ) -> GAResult | None:
        rng = random.Random(seed)
        initial = set(initial_skills)
        target = self._target_skills(target_career)
        if not target:
            return None
        weights = weights or {
            "time": 1.0,
            "cost": 0.5,
            "missing": 100.0,
            "length": 0.0,
        }
        candidates = self._candidate_courses(initial, target)
        if not candidates:
            return None

        seed_chrom = self._topological_repair(candidates, initial)
        seed_chrom = self._decode(seed_chrom, initial, target)
        logger.info(
            "GA seed: %d candidates, %d after decode",
            len(candidates), len(seed_chrom),
        )
        population = [seed_chrom]
        for _ in range(population_size - 1):
            population.append(self._random_chromosome(candidates, rng))

        best_chrom: list[str] | None = None
        best_fit = float("inf")
        best_feasible = False

        for gen in range(generations):
            scored = [
                (
                    self._fitness(
                        c, initial, target, max_budget, max_weeks, weights
                    ),
                    c,
                )
                for c in population
            ]
            scored.sort(key=lambda x: x[0][0])

            if scored[0][0][0] < best_fit:
                best_fit = scored[0][0][0]
                best_chrom = scored[0][1]
                best_feasible = scored[0][0][1]

            new_population = [s[1] for s in scored[:elite_size]]
            while len(new_population) < population_size:
                k = 3
                aspirants = rng.sample(scored, k=min(k, len(scored)))
                p1 = min(aspirants, key=lambda x: x[0][0])[1]
                aspirants = rng.sample(scored, k=min(k, len(scored)))
                p2 = min(aspirants, key=lambda x: x[0][0])[1]
                c1, c2 = self._crossover_ox(p1, p2, rng)
                c1 = self._mutate_swap(c1, mutation_rate, rng)
                c2 = self._mutate_swap(c2, mutation_rate, rng)
                new_population.append(c1)
                if len(new_population) < population_size:
                    new_population.append(c2)
            population = new_population

        if best_chrom is None or not best_feasible:
            logger.warning(
                "GA did not find a feasible solution in %d generations",
                generations,
            )
            return None

        path = self._decode(best_chrom, initial, target)
        return GAResult(
            path=path,
            total_cost=sum(self.cost_by_name.get(c, 0) for c in path),
            total_time=sum(self.duration_by_name.get(c, 0) for c in path),
            fitness=best_fit,
            generations=generations,
            feasible=True,
        )
