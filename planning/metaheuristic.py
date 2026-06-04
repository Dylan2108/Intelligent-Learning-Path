"""
Genetic Algorithm metaheuristic for the career-path planning problem.

Representation: permutation (sequence) of candidate course indices.
Fitness:     weighted sum of (time, cost, uncovered_skills, prereq_violations).
Operators:   Order Crossover (OX), swap mutation.
Repair:      after crossover/mutation, repair by topological sort so that
             prerequisites appear before their dependents. Invalid courses
             that fall outside the candidate pool are removed.
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

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
    GA-based metaheuristic for career path planning.
    """

    def __init__(
        self,
        courses_path: str = "data/courses.json",
        prerequisites_path: str = "data/prerequisites.json",
        careers_path: str = "data/careers.json",
    ):
        self.courses = json.loads(Path(courses_path).read_text())
        self.careers = json.loads(Path(careers_path).read_text())
        self.prereq_pairs: list[tuple[str, str]] = [
            (p["course"], p["prerequisite"])
            for p in json.loads(Path(prerequisites_path).read_text())
        ]
        self.duration_by_name = {c["name"]: c["duration"] for c in self.courses}
        self.cost_by_name = {c["name"]: c["cost"] for c in self.courses}

    def _target_skills(self, career_name: str) -> set[str]:
        for c in self.careers:
            if c["career"] == career_name:
                return set(c["skills"])
        return set()

    def _candidate_courses(self, initial: set[str], target: set[str]) -> list[str]:
        """
        Returns the list of courses that are potentially relevant: every
        target course plus every transitive prerequisite of any target.
        """
        # Build dependency closure
        closure: set[str] = set(target)
        stack = list(target)
        while stack:
            cur = stack.pop()
            for (course, pre) in self.prereq_pairs:
                if course == cur and pre not in closure and pre not in initial:
                    closure.add(pre)
                    stack.append(pre)
        return [c["name"] for c in self.courses if c["name"] in closure and c["name"] not in initial]

    def _prereqs_of(self, course: str) -> list[str]:
        return [pre for (c, pre) in self.prereq_pairs if c == course]

    def _topological_repair(self, sequence: list[str]) -> list[str]:
        """
        Reorders a sequence so that for every (course, prereq) edge,
        prereq appears before course. If a prereq is missing from the
        sequence but is in initial_skills, it is silently treated as
        satisfied (handled by the caller).
        """
        # Kahn's algorithm restricted to nodes in the sequence
        present = set(sequence)
        in_degree: dict[str, int] = {n: 0 for n in present}
        adj: dict[str, list[str]] = {n: [] for n in present}
        for (course, pre) in self.prereq_pairs:
            if course in present and pre in present:
                adj[pre].append(course)
                in_degree[course] += 1

        ready = sorted([n for n, d in in_degree.items() if d == 0])
        ordered: list[str] = []
        while ready:
            n = ready.pop(0)
            ordered.append(n)
            for m in adj[n]:
                in_degree[m] -= 1
                if in_degree[m] == 0:
                    ready.append(m)
        # If there's a cycle (shouldn't happen with valid prereqs), append
        # the remaining nodes in arbitrary order.
        if len(ordered) < len(present):
            leftovers = [n for n in present if n not in ordered]
            ordered.extend(leftovers)
        return ordered

    def _decode(self, chromosome: list[str], initial: set[str], target: set[str]) -> list[str]:
        """
        Decodes a chromosome into a valid path: keeps only courses that
        contribute to covering the target (or its prerequisites) and
        reorders them topologically.
        """
        kept = [c for c in chromosome if c in target or self._is_prereq_of_target(c, target)]
        return self._topological_repair(kept)

    def _is_prereq_of_target(self, course: str, target: set[str]) -> bool:
        stack = list(target)
        seen: set[str] = set()
        while stack:
            cur = stack.pop()
            if cur == course:
                return True
            if cur in seen:
                continue
            seen.add(cur)
            for (c, pre) in self.prereq_pairs:
                if c == cur:
                    stack.append(pre)
        return False

    def _fitness(
        self,
        chromosome: list[str],
        initial: set[str],
        target: set[str],
        max_budget: int | None,
        max_weeks: int | None,
        weights: dict[str, float],
    ) -> tuple[float, bool]:
        """
        Lower fitness is better. The chromosome is first decoded into a
        valid path. Penalties are applied for uncovered target skills
        and for violating hard constraints.
        """
        path = self._decode(chromosome, initial, target)
        covered = initial | set(path)
        total_cost = sum(self.cost_by_name.get(c, 0) for c in path)
        total_time = sum(self.duration_by_name.get(c, 0) for c in path)
        missing = target - covered
        feasibility = (
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
        if not feasibility:
            fit += 1000.0
        return fit, feasibility

    def _random_chromosome(self, candidates: list[str], rng: random.Random) -> list[str]:
        chrom = list(candidates)
        rng.shuffle(chrom)
        return chrom

    def _crossover_ox(
        self,
        parent_a: list[str],
        parent_b: list[str],
        rng: random.Random,
    ) -> tuple[list[str], list[str]]:
        """
        Order Crossover (OX): preserves relative order from each parent.
        """
        n = len(parent_a)
        if n < 2:
            return list(parent_a), list(parent_b)
        i, j = sorted(rng.sample(range(n), 2))

        def ox(p1: list[str], p2: list[str]) -> list[str]:
            middle = p1[i:j]
            remaining = [x for x in p2 if x not in middle]
            return remaining[:i] + middle + remaining[i:]

        return ox(parent_a, parent_b), ox(parent_b, parent_a)

    def _mutate_swap(self, chrom: list[str], rate: float, rng: random.Random) -> list[str]:
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
        """
        Runs the GA. Returns the best feasible individual found, or None
        if no feasible solution was found in the allotted generations.
        """
        rng = random.Random(seed)
        initial = set(initial_skills)
        target = self._target_skills(target_career)
        if not target:
            return None
        weights = weights or {"time": 1.0, "cost": 0.5, "missing": 100.0, "length": 0.0}
        candidates = self._candidate_courses(initial, target)
        if not candidates:
            return None

        # Seed population: always include a heuristic-friendly seed
        # (the A*-style ordering) to speed up convergence.
        seed_chrom = self._topological_repair(list(candidates))
        population = [seed_chrom]
        for _ in range(population_size - 1):
            population.append(self._random_chromosome(candidates, rng))

        best_chrom: list[str] | None = None
        best_fit = float("inf")
        best_feasible = False

        for gen in range(generations):
            scored = [
                (self._fitness(c, initial, target, max_budget, max_weeks, weights), c)
                for c in population
            ]
            scored.sort(key=lambda x: x[0][0])

            if scored[0][0][0] < best_fit:
                best_fit = scored[0][0][0]
                best_chrom = scored[0][1]
                best_feasible = scored[0][0][1]

            # Elitism
            new_population = [s[1] for s in scored[:elite_size]]
            while len(new_population) < population_size:
                # Tournament selection
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
            logger.warning("GA did not find a feasible solution in %d generations", generations)
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
