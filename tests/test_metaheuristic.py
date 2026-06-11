import pytest
from planning.metaheuristic import GeneticAlgorithmSolver


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

    def test_converges_to_optimal(self, ga):
        from planning.career_planner import CareerPlanner
        astar = CareerPlanner().plan(["Python"], "ML Engineer")
        genetic = ga.solve(["Python"], "ML Engineer", seed=42)
        assert genetic is not None
        assert astar.total_time == genetic.total_time

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
