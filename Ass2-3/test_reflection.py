import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "BALROG"))
from balrog.client import LLMResponse
from balrog.agents.naive import NaiveAgent
from balrog.prompt_builder.history import HistoryPromptBuilder
from agent.reflection import ReflectionAgent


class Client:
    def __init__(self, critiques=None):
        self.critiques = iter(critiques or [("Repeated actions did not help. Recheck the current target position.", "stop")]*10)
        self.messages = []

    def generate(self, messages):
        self.messages.append([(m.role, m.content) for m in messages])
        text, stop = next(self.critiques) if messages[0].content.startswith("You are reviewing") else ("turn left", "stop")
        return LLMResponse("test", text, stop, 10, 2, None)


def observation(text):
    return {"text": {"long_term_context": text, "short_term_context": ""}, "image": None}


def build(enabled=True, client=None, cls=ReflectionAgent):
    client = client or Client()
    builder = HistoryPromptBuilder(max_text_history=16, max_image_history=0)
    builder.update_instruction_prompt("Mission: go to the red ball. Actions: turn left, turn right, go forward.")
    return (cls(lambda: client, builder, enabled=enabled) if cls == ReflectionAgent else cls(lambda: client, builder)), client


def test_no_reflection_matches_naive_prompts_and_actions():
    agent, client = build(enabled=False)
    naive, baseline_client = build(cls=NaiveAgent)
    previous = None
    for step in range(40):
        obs = observation(f"View {step % 4}")
        response = agent.act(obs, previous)
        baseline = naive.act(obs, previous)
        assert response.completion == baseline.completion
        assert response.input_tokens == baseline.input_tokens
        previous = response.completion
    assert client.messages == baseline_client.messages
    assert agent.reviews == 0


def test_cycle_detection_cooldown_cap_and_reset():
    agent, client = build()
    events = []
    agent.on_reflection = events.append
    previous = None
    for step in range(70):
        previous = agent.act(observation(f"Facing direction {step % 4}"), previous).completion
    assert [e["after_actions"] for e in events] == [8, 16, 24, 32]
    assert all(e["trigger"] == "repeated_transition_cycle_4" for e in events)
    assert agent.reviews == 4 and len(client.messages) == 74
    assert agent.advice in client.messages[-1][-1][1]
    agent.reset()
    agent.act(observation("New episode"))
    assert agent.reviews == 0 and agent.advice is None and not agent.transitions
    assert agent.usage["critic"]["calls"] == 0


def test_scheduled_review_and_actual_fallback_action():
    agent, _ = build()
    previous = None
    for step in range(17):
        response = agent.act(observation(f"Distinct view {step}"), previous)
        previous = "invalid candidate"
    assert agent.last_decision["reflection"]["trigger"] == "scheduled_review"
    assert agent.transitions[-1]["executed_action"] == "go forward"
    assert agent.transitions[-1]["candidate_action"] == "invalid candidate"
    assert response.input_tokens == 20 and response.output_tokens == 4
    assert agent.usage["actor"]["calls"] == 17
    assert agent.usage["critic"]["calls"] == 1


@pytest.mark.parametrize("bad,stop,issue", [
    ("", "stop", "empty_advice"),
    ("word "*61, "stop", "advice_too_long"),
    ("Try moving toward the door.", "length", "truncated_advice"),
    ("turn right", "stop", "bare_action_instead_of_critique"),
])
def test_bad_critique_preserves_previous_advice_and_consumes_budget(bad, stop, issue):
    client = Client([("Check the target's position before moving.", "stop"), (bad, stop)])
    agent, _ = build(client=client)
    for step in range(17):
        agent.act(observation("Unchanged wall"), "turn left" if step else None)
    assert agent.reviews == 2 and agent.accepted_reviews == 1
    assert agent.advice == "Check the target's position before moving."
    assert issue in agent.last_decision["reflection"]["issues"]
    assert agent.last_decision["reflection"]["trigger"] == "four_unchanged_observations"
    assert client.allow_empty_response is False
