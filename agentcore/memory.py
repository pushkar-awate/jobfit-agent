"""Append-only decision ledger. This is both the audit trail and the demo."""
from __future__ import annotations
import json
import time


class Memory:
    def __init__(self, path=None):
        self.path = path
        self.entries = []

    def append(self, step, result):
        entry = {
            "step": step,
            "ts": round(time.time(), 3),
            "tool": result.decision.tool,
            "args": result.decision.args,
            "rationale": result.decision.rationale,
            "ok": result.ok,
            "note": result.note,
        }
        self.entries.append(entry)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

    def ledger(self):
        return list(self.entries)
