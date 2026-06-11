import pytest
from planning.constraints import ConstraintManager


@pytest.fixture
def cm():
    return ConstraintManager("data/prerequisites.json")


class TestConstraintManager:
    def test_prerequisites_of_ml(self, cm):
        prereqs = cm.prerequisites_of("Machine Learning")
        assert "Python" in prereqs
        assert "Statistics" in prereqs

    def test_prerequisites_of_python(self, cm):
        prereqs = cm.prerequisites_of("Python")
        assert prereqs == []

    def test_prerequisites_of_deep_learning(self, cm):
        prereqs = cm.prerequisites_of("Deep Learning")
        assert "Machine Learning" in prereqs

    def test_can_take_python_always(self, cm):
        assert cm.can_take(set(), "Python")

    def test_cannot_take_ml_without_stats(self, cm):
        assert not cm.can_take({"Python"}, "Machine Learning")

    def test_can_take_ml_with_all_prereqs(self, cm):
        assert cm.can_take({"Python", "Statistics", "Linear Algebra"}, "Machine Learning")

    def test_can_take_docker_with_linux(self, cm):
        assert cm.can_take({"Linux"}, "Docker")
