import pytest
from unittest.mock import patch, MagicMock
from llm.parser import GoalParser, GoalSchema
from llm.client import OllamaClient


class TestGoalSchema:
    def test_basic_goal(self):
        g = GoalSchema(target_career="ML Engineer", initial_skills=["Python"])
        assert g.target_career == "ML Engineer"
        assert g.initial_skills == ["Python"]
        assert g.max_budget is None
        assert g.max_weeks is None

    def test_goal_with_constraints(self):
        g = GoalSchema(
            target_career="Data Scientist",
            initial_skills=["Python", "SQL"],
            max_budget=200,
            max_weeks=30,
        )
        assert g.max_budget == 200
        assert g.max_weeks == 30

    def test_goal_empty_skills(self):
        g = GoalSchema(target_career="Cloud Architect", initial_skills=[])
        assert g.initial_skills == []


class TestGoalParser:
    def test_parse_returns_schema(self):
        mock_response = {
            "target_career": "ML Engineer",
            "initial_skills": ["Python"],
            "max_budget": 200,
            "max_weeks": 30,
            "notes": "",
        }
        with patch.object(OllamaClient, "chat_json", return_value=mock_response):
            parser = GoalParser(client=OllamaClient())
            result = parser.parse("Quiero ser ML Engineer y sé Python")
            assert isinstance(result, GoalSchema)
            assert result.target_career == "ML Engineer"
            assert "Python" in result.initial_skills

    def test_parse_coerces_string_skills(self):
        mock_response = {
            "target_career": "Data Scientist",
            "initial_skills": "Python, SQL",
            "max_budget": None,
            "max_weeks": None,
            "notes": "",
        }
        with patch.object(OllamaClient, "chat_json", return_value=mock_response):
            parser = GoalParser(client=OllamaClient())
            result = parser.parse("Quiero ser Data Scientist")
            assert isinstance(result.initial_skills, list)
            assert "Python" in result.initial_skills
            assert "SQL" in result.initial_skills

    def test_chat_json_retries_on_invalid_json(self):
        client = OllamaClient()
        bad_responses = [
            "not valid json",
            '{"incomplete":',
            '{"target_career": "Data Engineer", "initial_skills": [], "max_budget": null, "max_weeks": null, "notes": ""}',
        ]
        call_count = [0]

        def mock_chat(prompt, system=None, json_mode=False, temperature=0.0):
            idx = call_count[0]
            call_count[0] += 1
            return bad_responses[idx]

        with patch.object(client, "chat", side_effect=mock_chat):
            result = client.chat_json(prompt="test")
            assert result["target_career"] == "Data Engineer"
            assert call_count[0] == 3
