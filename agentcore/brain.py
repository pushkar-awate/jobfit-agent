"""The pluggable reasoning backend.

A brain implements:
  decide(observation, tool_specs) -> Decision   # which tool to run next
  generate(task, payload) -> str                # produce text/JSON for a step

MockBrain is deterministic and needs no network or API key. GroqBrain reuses
MockBrain's deterministic *planner* (tool order is fixed, so no LLM call is
wasted on plumbing) and overrides only generate() to do the semantic work with
a hosted LLM. That keeps a full run to two API calls - assess_fit and
draft_bullets - well within free-tier rate limits.
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
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start >= 0 and end > start else text


class Brain:
    def decide(self, observation, tool_specs):
        raise NotImplementedError

    def generate(self, task, payload):
        raise NotImplementedError


class MockBrain(Brain):
    """Deterministic planner + rule-based generation. No key needed."""

    PLAN = ["parse_jd", "parse_resume", "score_match", "identify_gaps",
            "assess_fit", "draft_bullets", "write_report", "finish"]

    def decide(self, observation, tool_specs):
        completed = observation.get("completed", [])
        for tool in self.PLAN:
            if tool not in completed:
                return Decision(tool, {}, "plan step: %s" % tool)
        return Decision("finish", {}, "plan complete")

    def generate(self, task, payload):
        if task == "draft_bullets":
            kws = payload.get("matched_keywords", [])
            bullets = payload.get("resume_bullets", [])[:4]
            role = payload.get("role", "the role")
            top = ", ".join(kws[:6]) if kws else "the listed skills"
            out = ["Tailored for %s. Emphasized strengths: %s." % (role, top)]
            for b in bullets:
                out.append("- %s" % b)
            if kws:
                out.append("- Directly relevant hands-on experience with %s." % top)
            return "\n".join(out)
        if task == "assess_fit":
            km = payload.get("keyword_match", {})
            return json.dumps({
                "score": km.get("score", 0),
                "strengths": km.get("matched", []),
                "gaps": km.get("missing", []),
                "rationale": "Deterministic keyword match (no LLM reasoning). "
                             "Run with --llm for a semantic assessment.",
                "source": "keyword",
            })
        return ""


class GroqBrain(MockBrain):
    """LLM generation over Groq, on top of MockBrain's deterministic planner."""

    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, model="openai/gpt-oss-20b", api_key=None):
        self.model = model
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY not set")

    def generate(self, task, payload):
        if task == "draft_bullets":
            prompt = (
                "Rewrite these resume bullets to be crisp, quantified where the "
                "original allows, and tailored to the role. Keep every claim "
                "truthful to the original bullets: do NOT invent tools, skills or "
                "metrics, and do NOT stuff keywords in unnaturally. Where genuinely "
                "relevant, reflect these real strengths: %s. Do NOT use em dashes or "
                "en dashes. Return 4-6 plain bullets starting with '- '.\n"
                "Role: %s\nOriginal bullets:\n%s"
                % (", ".join(payload.get("matched_keywords", [])[:6]),
                   payload.get("role"), "\n".join(payload.get("resume_bullets", [])))
            )
        elif task == "assess_fit":
            prompt = (
                "Assess how well a candidate fits a role. Read the JOB DESCRIPTION "
                "and RESUME and return ONLY a JSON object:\n"
                "{\"score\": <0-100 integer>, \"strengths\": [short phrases the "
                "candidate genuinely has that the role wants], \"gaps\": [things the "
                "role wants that the resume does not show], \"rationale\": <2-3 "
                "sentences>}.\nTreat semantic equivalents as matches (e.g. 'agent "
                "loops' ~ 'agentic workflows', 'Claude' ~ 'Anthropic'). Do NOT invent "
                "experience the resume lacks.\n\nJOB DESCRIPTION:\n%s\n\nRESUME:\n%s"
                % (payload.get("jd_text", "")[:4000], payload.get("resume_text", "")[:4000])
            )
        else:
            return ""
        try:
            return self._chat(prompt)
        except Exception as e:
            print("[groq] generate(%s) failed: %s" % (task, e), file=sys.stderr)
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
                     "User-Agent": "jobfit-agent/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                resp = json.loads(r.read().decode("utf-8"))
            return resp["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code == 429 and _retries > 0:      # rate limited: back off and retry
                time.sleep(4)
                return self._chat(prompt, temperature, _retries - 1)
            raise
