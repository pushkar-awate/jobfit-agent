"""Never trust the brain blindly. Validate the choice and the output."""
from __future__ import annotations


class GuardrailError(Exception):
    pass


class Guardrails:
    def __init__(self, banned_chars=("—", "–"), max_repeat=3):
        # banned_chars: em dash, en dash (a real document rule)
        self.banned_chars = banned_chars
        self.max_repeat = max_repeat

    def check_decision(self, decision, state):
        """Stop the agent looping forever on the same tool."""
        recent = [h.decision.tool for h in state["history"][-self.max_repeat:]]
        if (len(recent) == self.max_repeat and len(set(recent)) == 1
                and decision.tool == recent[-1]):
            raise GuardrailError("repeated tool %r too many times" % decision.tool)

    def check_output(self, decision, output, state):
        """Applied to generative output: no banned dashes, no invented skills."""
        if decision.tool == "draft_bullets" and isinstance(output, str):
            for ch in self.banned_chars:
                if ch in output:
                    raise GuardrailError("drafted text contains a banned em/en dash")
            self._no_invented_skills(output, state)

    @staticmethod
    def _no_invented_skills(text, state):
        missing = state["artifacts"].get("gaps", {}).get("missing", [])
        low = text.lower()
        invented = [s for s in missing if s.lower() in low]
        if invented:
            raise GuardrailError(
                "draft references skills not in the resume: " + ", ".join(invented))
