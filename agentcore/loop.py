"""The agent loop. Knows nothing about resumes or Kubernetes on purpose."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from .memory import Memory
from .guardrails import Guardrails, GuardrailError


@dataclass
class Decision:
    tool: str
    args: dict
    rationale: str = ""


@dataclass
class StepResult:
    decision: Decision
    output: Any
    ok: bool
    note: str = ""


class Agent:
    """Runs the loop: perceive -> reason -> guardrail -> act -> verify -> remember."""

    def __init__(self, brain, tools, memory=None, guardrails=None, max_steps=12):
        self.brain = brain
        self.tools = tools
        self.memory = memory or Memory()
        self.guardrails = guardrails or Guardrails()
        self.max_steps = max_steps

    def run(self, goal: str, context: dict) -> dict:
        state = {"goal": goal, "context": context, "brain": self.brain,
                 "done": False, "history": [], "artifacts": {}, "flags": []}
        for step in range(1, self.max_steps + 1):
            observation = self._perceive(state)                      # 1. PERCEIVE
            decision = self.brain.decide(observation, self.tools.specs())  # 2. REASON
            try:                                                     # 3. GUARDRAIL
                self.guardrails.check_decision(decision, state)
            except GuardrailError as e:
                decision = Decision("raise_flag", {"reason": str(e)},
                                    "blocked by guardrail")
            output, ok, note = self._act(decision, state)           # 4. ACT
            if ok:                                                   # 5. VERIFY
                try:
                    self.guardrails.check_output(decision, output, state)
                except GuardrailError as e:
                    ok, note = False, "output rejected: %s" % e
            result = StepResult(decision, output, ok, note)         # 6. REMEMBER
            self.memory.append(step, result)
            state["history"].append(result)
            if decision.tool == "finish":
                state["done"] = True
                break
        return state

    def _perceive(self, state) -> dict:
        return {
            "goal": state["goal"],
            "context_keys": list(state["context"].keys()),
            "completed": [h.decision.tool for h in state["history"] if h.ok],
            "artifacts": sorted(state["artifacts"].keys()),
            "recent": [(h.decision.tool, h.ok) for h in state["history"][-5:]],
        }

    def _act(self, decision, state):
        fn = self.tools.get(decision.tool)
        if fn is None:
            return None, False, "unknown tool %r" % decision.tool
        try:
            return fn(state, **decision.args), True, "ok"
        except Exception as e:  # a tool must never crash the loop
            return None, False, "tool error: %s" % e
