"""Bounded within-episode reflection on observed BabyAI transitions."""

import json
from collections import deque

from balrog.agents.base import BaseAgent
from balrog.agents.naive import NaiveAgent
from balrog.prompt_builder.history import Message


class ReflectionAgent(NaiveAgent):
    ACTIONS = ("turn left", "turn right", "go forward", "pick up", "drop", "toggle")
    INTERVAL = 16
    COOLDOWN = 8
    MAX_REVIEWS = 4
    WINDOW = 8
    ACTOR_INSTRUCTION = "You always have to output one of the above actions at a time and no other text. You always have to output an action until the episode terminates."
    CRITIC_INSTRUCTION = """You are reviewing an agent's recent behavior in BabyAI.
Use ONLY the mission and observed transitions below. Assess whether its actions
and previous advice are helping. An attempted action is not evidence of success.
Repeated observations or a loop alert are reasons to inspect, not proof of failure.
Return at most 60 words of plain text: a brief evidence-based diagnosis followed
by one concrete corrective instruction, or explain why continuing is appropriate.
Do not invent objects, positions, inventory, or completed subgoals. If evidence
is insufficient, say so and recommend checking the current observation.
Prefer retaining useful advice over reversing it without evidence. Do not output
a list of game actions, a conversation, or a new mission."""

    def __init__(self, client_factory, prompt_builder, enabled=True):
        BaseAgent.__init__(self, client_factory, prompt_builder)
        self.enabled = enabled
        self.on_reflection = None
        self.reset()

    def reset(self):
        super().reset()
        self.transitions = deque(maxlen=self.WINDOW)
        self.previous_observation = None
        self.advice = None
        self.completed_actions = 0
        self.last_review_step = 0
        self.reviews = 0
        self.accepted_reviews = 0
        self.last_decision = None
        self.call_role = "actor"
        self.usage = {role: {"calls": 0, "input_tokens": 0, "output_tokens": 0}
                      for role in ("actor", "critic")}

    @staticmethod
    def observation_text(obs):
        return "\n".join(str(obs["text"].get(k, "")) for k in
                         ("short_term_context", "long_term_context")).strip()

    def reflection_trigger(self):
        if not self.enabled or self.reviews >= self.MAX_REVIEWS:
            return None
        if self.completed_actions - self.last_review_step < self.COOLDOWN:
            return None
        recent = list(self.transitions)
        if len(recent) >= 4 and all(t["before"] == t["after"] for t in recent[-4:]):
            return "four_unchanged_observations"
        signatures = [(t["before"], t["executed_action"], t["after"]) for t in recent]
        for period in range(2, self.WINDOW // 2 + 1):
            if len(signatures) >= 2 * period and signatures[-period:] == signatures[-2*period:-period]:
                return f"repeated_transition_cycle_{period}"
        if self.completed_actions % self.INTERVAL == 0:
            return "scheduled_review"
        return None

    def generate(self, messages, role):
        self.call_role = role
        self.client.allow_empty_response = role == "critic"
        try:
            response = self.client.generate(messages)
            self.usage[role]["calls"] += 1
            self.usage[role]["input_tokens"] += response.input_tokens
            self.usage[role]["output_tokens"] += response.output_tokens
            return response
        finally:
            self.call_role = "actor"
            self.client.allow_empty_response = False

    def reflect(self, trigger):
        messages = [Message(role="user", content=self.CRITIC_INSTRUCTION + "\n\n" + json.dumps({
            "mission_and_game_instructions": self.prompt_builder.system_prompt,
            "trigger": trigger,
            "previous_advice": self.advice,
            "recent_transitions": list(self.transitions),
        }, ensure_ascii=False))]
        response = self.generate(messages, "critic")
        self.reviews += 1
        self.last_review_step = self.completed_actions
        advice = (response.completion or "").strip()
        issues = []
        if not advice:
            issues.append("empty_advice")
        if len(advice.split()) > 60:
            issues.append("advice_too_long")
        if response.stop_reason == "length":
            issues.append("truncated_advice")
        if advice.lower() in self.ACTIONS:
            issues.append("bare_action_instead_of_critique")
        before = self.advice
        if not issues:
            self.advice = advice
            self.accepted_reviews += 1
        event = {"after_actions": self.completed_actions, "trigger": trigger,
                 "advice_before": before, "advice_after": self.advice,
                 "raw_advice": advice, "accepted": not issues, "issues": issues,
                 "input_tokens": response.input_tokens, "output_tokens": response.output_tokens}
        if self.on_reflection:
            self.on_reflection(event)
        return response, event

    def act(self, obs, prev_action=None):
        current = self.observation_text(obs)
        if self.previous_observation is not None:
            # BALROG passes the candidate, even when it executed its fallback.
            executed = prev_action if prev_action in self.ACTIONS else "go forward"
            self.transitions.append({"before": self.previous_observation,
                                     "candidate_action": prev_action,
                                     "executed_action": executed, "after": current})
            self.completed_actions += 1
        trigger = self.reflection_trigger()
        critique, event = self.reflect(trigger) if trigger else (None, None)
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        messages = self.prompt_builder.get_prompt()
        if self.advice:
            messages[-1].content += ("\n\nRecent reviewer advice (may be mistaken; check against the current observation):\n" + self.advice)
        messages[-1].content += "\n\n" + self.ACTOR_INSTRUCTION
        raw = self.generate(messages, "actor")
        response = self._extract_final_answer(raw)  # Same parser as the naive baseline.
        self.previous_observation = current
        self.last_decision = {"action_step": self.completed_actions,
                              "candidate_action": response.completion,
                              "executed_action": response.completion if response.completion in self.ACTIONS else "go forward",
                              "advice": self.advice, "reflection": event}
        return response._replace(
            input_tokens=response.input_tokens + (critique.input_tokens if critique else 0),
            output_tokens=response.output_tokens + (critique.output_tokens if critique else 0),
            reasoning=json.dumps(self.last_decision))

    def episode_stats(self):
        return {"reviews": self.reviews, "accepted_reviews": self.accepted_reviews,
                "usage": self.usage}
