"""The pluggable reasoning backend - the "brain".

A brain is the swappable intelligence behind an agent (or a judge). It exposes
two task-agnostic methods:

    decide(observation, tool_specs) -> Decision   # choose the next tool to run
    complete(prompt) -> str                        # do the actual LLM reasoning

MockBrain is deterministic and needs no key: `decide` follows a plan you give it
(or the available tool order), and `complete` returns nothing - so a run is
reproducible and key-free by default. GroqBrain overrides `complete` to reason
with a hosted LLM; that is the ONLY method that touches the network, and it is
task-agnostic - the caller supplies the prompt.

Nothing here is task-specific. An app's plan and its prompts live in the app
layer, not in the shared brain - that is what makes this core reusable across
jobfit-agent, selfheal-mlops and agenteval.
"""
from __future__ import annotations
import json
import os
import sys
import time
import urllib.error
import urllib.request

from .loop import Decision


def extract_json(text):
    """Best-effort: pull the first {...} object out of an LLM reply."""
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start >= 0 and end > start else text


def tool_names(tool_specs):
    """Normalise a tool-spec collection to an ordered list of tool names.

    Registries differ: some hand `decide` a dict {name: spec}, some a list of
    names, some a list of {"name": ...} dicts. The brain shouldn't care."""
    if not tool_specs:
        return []
    if isinstance(tool_specs, dict):
        return list(tool_specs.keys())
    names = []
    for t in tool_specs:
        if isinstance(t, str):
            names.append(t)
        elif isinstance(t, dict):
            names.append(t.get("name"))
    return names


# agentcore's built-in control tools (see ToolRegistry / loop.py). A brain that
# auto-derives a plan from the registry must skip these - they are how the loop
# stops and flags problems, not task steps.
CONTROL_TOOLS = ("finish", "raise_flag")


class Brain:
    def decide(self, observation, tool_specs):
        raise NotImplementedError

    def complete(self, prompt):
        raise NotImplementedError


class MockBrain(Brain):
    """Deterministic reasoning: no network, no key.

    `decide` runs a plan (a list of tool names) if one is given, otherwise it
    falls back to the order the tools are registered in. `complete` is a stub -
    real semantic work needs a real LLM brain (below).
    """

    def __init__(self, plan=None):
        self.plan = plan

    def decide(self, observation, tool_specs):
        completed = observation.get("completed", [])
        plan = self.plan or [n for n in tool_names(tool_specs)
                             if n not in CONTROL_TOOLS]
        for tool in plan:
            if tool and tool not in completed:
                return Decision(tool, {}, "plan step: %s" % tool)
        return Decision("finish", {}, "plan complete")

    def complete(self, prompt):
        return ""


class GroqBrain(MockBrain):
    """Real LLM reasoning over Groq. Task-agnostic: the caller owns the prompt."""

    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, model="openai/gpt-oss-20b", api_key=None, plan=None):
        super().__init__(plan)
        self.model = model
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY not set")

    def complete(self, prompt):
        try:
            return self._chat(prompt)
        except Exception as e:
            print("[groq] complete failed: %s" % e, file=sys.stderr)
            return ""

    def _chat(self, prompt, temperature=0.2, _retries=2):
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.URL, data=body, method="POST",
            headers={"Authorization": "Bearer %s" % self.api_key,
                     "Content-Type": "application/json",
                     "User-Agent": "agentcore/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                resp = json.loads(r.read().decode("utf-8"))
            return resp["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code == 429 and _retries > 0:      # rate limited: back off, retry
                time.sleep(4)
                return self._chat(prompt, temperature, _retries - 1)
            raise
