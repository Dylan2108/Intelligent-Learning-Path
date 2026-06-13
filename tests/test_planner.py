import pytest
from planning.career_planner import CareerPlanner


@pytest.fixture
def planner():
    return CareerPlanner()


class TestCareerPlanner:
    def test_target_skills_ml(self, planner):
        skills = planner.target_skills("ML Engineer")
        assert skills == {"Python", "Machine Learning", "Deep Learning", "Docker"}

    def test_target_skills_unknown(self, planner):
        skills = planner.target_skills("Nonexistent Career")
        assert skills == set()

    def test_get_course(self, planner):
        course = planner.get_course("Python (Intro)")
        assert course["name"] == "Python (Intro)"
        assert course["duration"] > 0
        assert "teaches" in course

    def test_get_course_unknown(self, planner):
        course = planner.get_course("Nonexistent")
        assert course == {}

    def test_heuristic_empty(self, planner):
        h = planner.heuristic(set(), {"Python", "Machine Learning"})
        assert h > 0

    def test_heuristic_completed(self, planner):
        h = planner.heuristic({"Python"}, {"Python"})
        assert h == 0

    def test_heuristic_admissible(self, planner):
        target = planner.target_skills("ML Engineer")
        h = planner.heuristic(set(), target)
        result = planner.plan(["Python"], "ML Engineer")
        assert result is not None
        assert h <= result.total_time

    def test_plan_ml_engineer(self, planner):
        result = planner.plan(["Python"], "ML Engineer")
        assert result is not None
        assert result.total_time > 0
        assert result.total_cost > 0
        assert len(result.path) > 0

    def test_plan_data_scientist(self, planner):
        result = planner.plan(["Python"], "Data Scientist")
        assert result is not None
        assert result.total_time > 0

    def test_plan_data_engineer(self, planner):
        result = planner.plan(["Python"], "Data Engineer")
        assert result is not None

    def test_plan_no_skills(self, planner):
        result = planner.plan([], "Data Engineer")
        assert result is not None

    def test_plan_returns_optimal(self, planner):
        r1 = planner.plan(["Python"], "ML Engineer")
        r2 = planner.plan(["Python"], "ML Engineer")
        assert r1 is not None and r2 is not None
        assert r1.total_time == r2.total_time
        assert r1.total_cost == r2.total_cost

    def test_plan_covered_skills(self, planner):
        result = planner.plan(["Python"], "ML Engineer")
        assert result is not None
        target = planner.target_skills("ML Engineer")
        assert target.issubset(result.covered_skills)

    def test_plan_path_respects_prereqs(self, planner):
        result = planner.plan(["Python"], "ML Engineer")
        assert result is not None
        covered = set(["Python"])
        for name in result.path:
            course = planner.get_course(name)
            assert planner.constraints.can_take(covered, course), (
                f"Cannot take {name}: covered skills {covered}"
            )
            covered.update(course.get("teaches", []))
