import pytest
from planning.greedy import GreedySolver
from planning.career_planner import CareerPlanner


@pytest.fixture
def greedy():
    return GreedySolver()


class TestGreedySolver:
    def test_solve_ml_engineer(self, greedy):
        result = greedy.solve(["Python"], "ML Engineer")
        assert result is not None
        assert result.total_time > 0
        assert result.courses_taken > 0
        assert len(result.path) == result.courses_taken

    def test_solve_data_scientist(self, greedy):
        result = greedy.solve(["Python"], "Data Scientist")
        assert result is not None
        assert result.total_time > 0

    def test_solve_data_engineer(self, greedy):
        result = greedy.solve(["Python"], "Data Engineer")
        assert result is not None
        assert result.total_time > 0

    def test_solve_cloud_architect(self, greedy):
        result = greedy.solve([], "Cloud Architect")
        assert result is not None
        assert result.total_time > 0

    def test_solve_devops_engineer(self, greedy):
        result = greedy.solve([], "DevOps Engineer")
        assert result is not None
        assert result.total_time > 0

    def test_unknown_career(self, greedy):
        result = greedy.solve(["Python"], "Nonexistent Career")
        assert result is None

    def test_already_has_skills(self, greedy):
        result = greedy.solve(
            ["Python", "Machine Learning", "Deep Learning", "Docker"], "ML Engineer"
        )
        assert result is not None
        assert result.courses_taken == 0
        assert result.total_time == 0

    def test_greedy_path_respects_prerequisites(self, greedy):
        result = greedy.solve(["Python"], "ML Engineer")
        assert result is not None
        covered = set(["Python"])
        for name in result.path:
            course = greedy.courses_by_name[name]
            assert greedy.constraints.can_take(covered, course)
            covered.update(course["teaches"])

    def test_greedy_covers_target_skills(self, greedy):
        result = greedy.solve(["Python"], "ML Engineer")
        assert result is not None
        target = greedy._target_skills("ML Engineer")
        covered = set()
        for name in result.path:
            covered.update(greedy.courses_by_name[name]["teaches"])
        assert target.issubset(covered)

    def test_greedy_vs_astar(self, greedy):
        astar = CareerPlanner()
        r_greedy = greedy.solve(["Python"], "ML Engineer")
        r_astar = astar.plan(["Python"], "ML Engineer")
        assert r_greedy is not None
        assert r_astar is not None
        assert r_greedy.total_time >= r_astar.total_time
