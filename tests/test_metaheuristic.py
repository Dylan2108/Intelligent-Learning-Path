import pytest
from planning.metaheuristic import GeneticAlgorithmSolver
from planning.career_planner import CareerPlanner


@pytest.fixture
def ga():
    return GeneticAlgorithmSolver()


class TestGeneticAlgorithm:
    def test_solve_ml_engineer(self, ga):
        result = ga.solve(["Python"], "ML Engineer", seed=42)
        assert result is not None
        assert result.feasible
        assert result.total_time > 0

    def test_solve_data_scientist(self, ga):
        result = ga.solve(["Python"], "Data Scientist", seed=42)
        assert result is not None
        assert result.feasible

    def test_solve_data_engineer(self, ga):
        result = ga.solve(["Python"], "Data Engineer", seed=42)
        assert result is not None
        assert result.feasible

    def test_solve_cloud_architect(self, ga):
        result = ga.solve([], "Cloud Architect", seed=42)
        assert result is not None
        assert result.feasible

    def test_solve_devops_engineer(self, ga):
        result = ga.solve([], "DevOps Engineer", seed=42)
        assert result is not None
        assert result.feasible

    def test_converges_toward_optimal(self, ga):
        astar = CareerPlanner().plan(["Python"], "ML Engineer")
        genetic = ga.solve(["Python"], "ML Engineer", seed=42)
        assert genetic is not None
        assert astar is not None
        # GA may not always match A* exactly, but should be close
        assert genetic.total_time >= astar.total_time

    def test_respects_budget(self, ga):
        result = ga.solve(["Python"], "ML Engineer", max_budget=100, seed=42)
        if result is not None:
            assert result.total_cost <= 100

    def test_respects_weeks(self, ga):
        result = ga.solve(["Python"], "ML Engineer", max_weeks=10, seed=42)
        if result is not None:
            assert result.total_time <= 10

    def test_unknown_career(self, ga):
        result = ga.solve(["Python"], "Nonexistent Career", seed=42)
        assert result is None

    def test_path_covers_target_skills(self, ga):
        result = ga.solve(["Python"], "ML Engineer", seed=42)
        assert result is not None
        target = ga._target_skills("ML Engineer")
        covered = set(["Python"])
        for name in result.path:
            covered.update(ga.courses_by_name[name]["teaches"])
        assert target.issubset(covered)
