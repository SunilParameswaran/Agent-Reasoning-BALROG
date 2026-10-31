"""Behavioural checks for the planning agent without model calls."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "BALROG"))
from balrog.agents.custom import CustomAgent
from balrog.client import LLMResponse
from balrog.prompt_builder.history import HistoryPromptBuilder


class FakeClient:
    def __init__(self, texts):
        self.texts = iter(texts)
        self.prompts = []

    def generate(self, messages):
        self.prompts.append([m.content for m in messages])
        text, stop = next(self.texts)
        return LLMResponse("fake", text, stop, 10, 12, None)


def make_agent(texts):
    client = FakeClient(texts)
    history = HistoryPromptBuilder(max_text_history=16, max_image_history=0)
    history.update_instruction_prompt("Mission: pick up the blue ball.")
    return CustomAgent(lambda: client, history), client


OBS = {"text": {"long_term_context": "a blue ball 2 steps forward", "short_term_context": ""}, "image": None}
VALID = "PLAN: Approach the ball then pick it up.\nSUBGOAL: Approach the ball.\nACTION: go forward"


def test_plan_persists_and_reset_clears_all_episode_state():
    agent, client = make_agent([(VALID, "stop"), ("PLAN: KEEP\nSUBGOAL: Pick up the ball.\nACTION: pick up", "stop"), (VALID, "stop")])
    first = agent.act(OBS)
    second = agent.act(OBS, prev_action=first.completion)
    assert "Saved plan: Approach the ball then pick it up." in client.prompts[1][-1]
    assert agent.plan == "Approach the ball then pick it up."
    assert agent.subgoal == "Pick up the ball."
    assert second.completion == "pick up"
    assert second.input_tokens == 10 and second.output_tokens == 12
    assert len(client.prompts) == 2
    agent.reset()
    assert agent.plan is None and agent.subgoal is None and agent.last_decision is None
    agent.act(OBS)
    assert "Saved plan: Not yet set" in client.prompts[2][-1]
    assert len(client.prompts[2]) == 2  # Mission plus only the new observation.


@pytest.mark.parametrize("text, action, issue", [
    ("PLAN: KEEP\nSUBGOAL: KEEP\nACTION: turn left\nACTION: turn right", "", "missing_or_duplicate_action"),
    ("PLAN: KEEP\nSUBGOAL: KEEP\nACTION: turn left and go forward", "", "invalid_action"),
    ("turn right", "turn right", "bare_action"),
    ("PLAN: Change the plan.\nACTION: turn left", "turn left", "missing_or_duplicate_subgoal"),
    ("PLAN: KEEP\nSUBGOAL: KEEP\nACTION: teleport", "", "invalid_action"),
    ("PLAN: " + "word " * 25 + "\nSUBGOAL: New subgoal\nACTION: turn left", "turn left", "plan_too_long"),
])
def test_malformed_responses_do_not_corrupt_memory(text, action, issue):
    agent, client = make_agent([(VALID, "stop"), (text, "stop")])
    agent.act(OBS)
    memory = (agent.plan, agent.subgoal)
    response = agent.act(OBS)
    assert response.completion == action
    assert (agent.plan, agent.subgoal) == memory
    decision = json.loads(response.reasoning)
    assert issue in decision["issues"]
    assert bool(decision["fallback"]) == (action == "")
    assert len(client.prompts) == 2  # No hidden repair call.


def test_normalizes_legal_action_but_not_ambiguous_text():
    agent, _ = make_agent([(VALID.replace("go forward", "`Go   Forward`"), "stop")])
    assert agent.act(OBS).completion == "go forward"
    assert agent.last_decision["issues"] == []


def test_truncated_response_does_not_commit_memory():
    agent, _ = make_agent([(VALID, "length")])
    assert agent.act(OBS).completion == "go forward"
    assert agent.plan is None
    assert "truncated_response" in agent.last_decision["issues"]


def test_keep_cannot_initialize_memory():
    agent, _ = make_agent([("PLAN: KEEP\nSUBGOAL: KEEP\nACTION: turn left", "stop")])
    agent.act(OBS)
    assert agent.plan is None and agent.subgoal is None
    assert "keep_without_plan" in agent.last_decision["issues"]
