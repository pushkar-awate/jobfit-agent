"""The pluggable reasoning backend.

A brain implements:
  decide(observation, tool_specs) -> Decision   # which tool to run next
  generate(task, payload) -> str                # produce text/JSON for a step

MockBrain is deterministic and needs no network or API key. GroqBrain calls a
hosted LLM when GROQ_API_KEY is set. Same interface -> pluggable.
"""
from __future__ import annotations
import json
import os
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


class GroqBrain(Brain):
    """Hosted-LLM brain via Groq's OpenAI-compatible API (stdlib only)."""

    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, model="llama-3.3-70b-versatile", api_key=None):
        self.model = model
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY not set")

    def decide(self, observation, tool_specs):
        tools_desc = "\n".join("- %s: %s" % (n, s) for n, s in tool_specs.items())
        prompt = (
            "You are the planner of a job-fit agent. Choose the SINGLE next tool.\n"
            "Prerequisites: parse_jd and parse_resume must run before score_match; "
            "score_match before identify_gaps and assess_fit; then draft_bullets, "
            "then write_report, then finish.\n"
            "Goal: %s\nTools:\n%s\nAlready completed: %s\n"
            "Reply with ONLY JSON: {\"tool\": <name>, \"args\": {}, \"rationale\": <short>}"
            % (observation.get("goal"), tools_desc, observation.get("completed"))
        )
        try:
            data = json.loads(extract_json(self._chat(prompt)))
            return Decision(data["tool"], data.get("args", {}),
                            data.get("rationale", ""))
        except Exception:
            return Decision("finish", {}, "unparseable brain output")

    def generate(self, task, payload):
        if task == "draft_bullets":
            prompt = (
                "Rewrite these resume bullets for the role '%s'. Emphasize ONLY "
                "these already-present skills: %s. Do NOT invent any skill not "
                "listed. Do NOT use em dashes or en dashes. Return plain bullets.\n"
                "Bullets:\n%s"
                % (payload.get("role"), ", ".join(payload.get("matched_keywords", [])),
                   "\n".join(payload.get("resume_bullets", [])))
            )
            return self._chat(prompt)
        if task == "assess_fit":
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
            return self._chat(prompt)
        return ""

    def _chat(self, prompt, temperature=0.2):
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.URL, data=body, method="POST",
            headers={"Authorization": "Bearer %s" % self.api_key,
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            resp = json.loads(r.read().decode("utf-8"))
        return resp["choices"][0]["message"]["content"]
