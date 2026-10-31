"""Exercise 2: short plan and subgoal memory, evaluated on BabyAI only."""

import json
import re

from balrog.agents.base import BaseAgent


class CustomAgent(BaseAgent):
    """One model call per action; persistent memory is cleared every episode."""

    ACTIONS = ("turn left", "turn right", "go forward", "pick up", "drop", "toggle")
    INSTRUCTIONS = """Use the mission and observations to maintain a short plan.
Choose the next action toward your current subgoal. Only mark a subgoal complete
when the observation supports it; an attempted action is not proof of success.
Use only the game observations and history. Do not invent objects or completed steps.
Return exactly three lines, without explanations or markdown:
PLAN: <a short plan of at most 24 words, or KEEP if unchanged>
SUBGOAL: <the current subgoal in at most 10 words, or KEEP if unchanged>
ACTION: <exactly one of: turn left, turn right, go forward, pick up, drop, toggle>
On the first step, provide an actual plan and subgoal, not KEEP."""

    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.plan = None
        self.subgoal = None
        self.last_decision = None

    def reset(self):
        super().reset()
        self.plan = None
        self.subgoal = None
        self.last_decision = None

    @staticmethod
    def _normalize_action(text):
        return " ".join(text.strip().strip("`\"'").lower().split())

    def _parse(self, text):
        fields = {}
        issues = []
        for line in text.strip().splitlines():
            match = re.fullmatch(r"\s*(PLAN|SUBGOAL|ACTION)\s*:\s*(.*?)\s*", line, re.I)  # noqa: FURB167
            if not match:
                if line.strip():
                    issues.append("unexpected_text")
                continue
            name, value = match.group(1).upper(), match.group(2)
            fields.setdefault(name, []).append(value)
        for name in ("PLAN", "SUBGOAL", "ACTION"):
            if len(fields.get(name, [])) != 1 or not fields[name][0]:
                issues.append(f"missing_or_duplicate_{name.lower()}")
        action = ""
        if len(fields.get("ACTION", [])) == 1:
            candidate = self._normalize_action(fields["ACTION"][0])
            if candidate in self.ACTIONS:
                action = candidate
        elif not fields:
            candidate = self._normalize_action(text)
            if candidate in self.ACTIONS:
                action = candidate
                issues.append("bare_action")
        if not action:
            issues.append("invalid_action")
        return fields, action, issues

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        messages = self.prompt_builder.get_prompt()
        if not messages or messages[-1].role != "user":
            raise ValueError("CustomAgent requires a final user observation.")
        memory_before = {"plan": self.plan, "subgoal": self.subgoal}
        messages[-1].content += (
            f"\n\nSaved plan: {self.plan or 'Not yet set'}"
            f"\nCurrent subgoal: {self.subgoal or 'Not yet set'}\n\n{self.INSTRUCTIONS}"
        )
        response = self.client.generate(messages)
        fields, action, issues = self._parse(response.completion or "")
        if response.stop_reason == "length":
            issues.append("truncated_response")
        proposed = {}
        if not issues:
            for name, attribute, limit in [
                ("PLAN", "plan", 24),
                ("SUBGOAL", "subgoal", 10),
            ]:
                value = fields[name][0]
                if value.upper() == "KEEP":
                    value = getattr(self, attribute)
                    if value is None:
                        issues.append(f"keep_without_{attribute}")
                elif len(value.split()) > limit:
                    issues.append(f"{attribute}_too_long")
                proposed[attribute] = value
        if not issues:
            self.plan, self.subgoal = proposed["plan"], proposed["subgoal"]
        self.last_decision = {
            "memory_before": memory_before,
            "memory_after": {"plan": self.plan, "subgoal": self.subgoal},
            "action": action,
            "issues": issues,
            "fallback": "BALROG default: go forward" if not action else None,
        }
        # Empty action is rejected and counted by BALROG, rather than silently repaired.
        return response._replace(
            completion=action, reasoning=json.dumps(self.last_decision)
        )
