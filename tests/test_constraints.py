import pytest
from planning.constraints import ConstraintManager


@pytest.fixture
def cm():
    return ConstraintManager("data/prerequisites.json", "data/courses.json")


class TestConstraintManager:
    def test_prerequisites_of_ml(self, cm):
        prereqs = cm.prerequisites_of_skill("Machine Learning")
        assert "Python" in prereqs
        assert "Statistics" in prereqs

    def test_prerequisites_of_python(self, cm):
        prereqs = cm.prerequisites_of_skill("Python")
        assert prereqs == set()

    def test_prerequisites_of_deep_learning(self, cm):
        prereqs = cm.prerequisites_of_skill("Deep Learning")
        assert "Machine Learning" in prereqs

    def test_can_take_no_prereqs(self, cm):
        course = {"teaches": ["Python"]}
        assert cm.can_take(set(), course)

    def test_cannot_take_ml_without_stats(self, cm):
        course = {"teaches": ["Machine Learning"]}
        assert not cm.can_take({"Python"}, course)

    def test_can_take_ml_with_all_prereqs(self, cm):
        course = {"teaches": ["Machine Learning"]}
        assert cm.can_take({"Python", "Statistics", "Linear Algebra"}, course)

    def test_can_take_docker_with_linux(self, cm):
        course = {"teaches": ["Docker"]}
        assert cm.can_take({"Linux"}, course)

    def test_can_take_multi_skill_course(self, cm):
        # Course that teaches ML+DL covers DL's prereq (ML) itself
        course = {"teaches": ["Machine Learning", "Deep Learning"]}
        assert cm.can_take({"Python", "Statistics", "Linear Algebra"}, course)
        # But it doesn't cover ML's own prereqs (Statistics, Linear Algebra)
        assert not cm.can_take({"Python"}, course)
        assert cm.can_take(
            {"Python", "Statistics", "Linear Algebra", "Machine Learning"}, course
        )
