"""A tiny tool registry. Built-in tools: finish, raise_flag."""
from __future__ import annotations


class ToolRegistry:
    def __init__(self):
        self._tools = {}
        self._specs = {}
        self.register("finish", lambda state, **k: "done",
                      "Call when the goal is complete.")
        self.register("raise_flag", self._raise_flag,
                      "Record a problem the agent could not handle.")

    def register(self, name, fn, spec):
        self._tools[name] = fn
        self._specs[name] = spec

    def get(self, name):
        return self._tools.get(name)

    def specs(self):
        return dict(self._specs)

    @staticmethod
    def _raise_flag(state, reason="unspecified", **k):
        state.setdefault("flags", []).append(reason)
        return {"flag": reason}
