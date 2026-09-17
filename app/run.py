"""CLI entry point.

  python -m app.run --jd JD.txt --resume RESUME.txt            # mock brain
  python -m app.run --jd JD.txt --resume RESUME.txt --llm      # force Groq
  python -m app.run --jd JD.txt --resume RESUME.txt --auto     # escalate only if unsure
  python -m app.run --jd JD.txt --resume RESUME.txt --json     # machine-readable output
"""
from __future__ import annotations
import argparse
import json
import os
import sys

from agentcore.loop import Agent
from agentcore.memory import Memory
from agentcore.guardrails import Guardrails
from agentcore.tools import ToolRegistry
from agentcore.brain import MockBrain, GroqBrain
from app import tools_jobfit as T
from app.skills import extract_skills

# confidence bands for --auto routing (keyword score, 0-100)
CONFIDENT_FIT = 55     # at/above this: clearly a fit, mock is enough
CONFIDENT_NOFIT = 5    # at/below this: clearly unrelated, mock is enough


def build_registry():
    reg = ToolRegistry()
    reg.register("parse_jd", T.parse_jd,
                 "Extract role title and required skills from the job description.")
    reg.register("parse_resume", T.parse_resume,
                 "Extract skills and bullets from the resume.")
    reg.register("score_match", T.score_match,
                 "Compute an ATS-style keyword match score between JD and resume.")
    reg.register("identify_gaps", T.identify_gaps,
                 "List JD skills missing from the resume.")
    reg.register("assess_fit", T.assess_fit,
                 "Assess overall fit with reasoning (LLM) or keywords (mock).")
    reg.register("draft_bullets", T.draft_bullets,
                 "Draft tailored resume bullets (no invented skills).")
    reg.register("write_report", T.write_report,
                 "Assemble the final markdown report.")
    return reg


def pick_brain(use_llm=False):
    # Mock is the default so a fresh clone runs with zero keys. The LLM is
    # opt-in, and still falls back to mock if the key is missing.
    if use_llm:
        try:
            return GroqBrain(), "groq"
        except Exception as e:
            print("[warn] LLM requested but Groq unavailable (%s); "
                  "falling back to mock" % e, file=sys.stderr)
    return MockBrain(), "mock"


def keyword_score(jd_text, resume_text):
    jd = set(extract_skills(jd_text))
    res = set(extract_skills(resume_text))
    return round(100 * len(jd & res) / max(1, len(jd)))


def route(jd_text, resume_text, use_llm, auto):
    """Pick a brain. --llm forces the LLM; --auto escalates only when unsure."""
    if use_llm:
        return pick_brain(use_llm=True)
    if auto:
        ks = keyword_score(jd_text, resume_text)
        if CONFIDENT_NOFIT < ks < CONFIDENT_FIT and os.environ.get("GROQ_API_KEY"):
            brain, mode = pick_brain(use_llm=True)
            if mode == "groq":
                print("[router] keyword score %d%% is uncertain -> escalating to LLM"
                      % ks, file=sys.stderr)
                return brain, "groq (auto)"
        reason = ("confident fit" if ks >= CONFIDENT_FIT
                  else "confident nofit" if ks <= CONFIDENT_NOFIT
                  else "no key, cannot escalate")
        print("[router] keyword score %d%% -> %s -> mock" % (ks, reason),
              file=sys.stderr)
    return pick_brain(use_llm=False)


def main():
    ap = argparse.ArgumentParser(description="Resume <-> JD job-fit agent")
    ap.add_argument("--jd", required=True)
    ap.add_argument("--resume", required=True)
    ap.add_argument("--out", default="report.md")
    ap.add_argument("--ledger", default="decisions.jsonl")
    ap.add_argument("--llm", action="store_true",
                    help="use the Groq LLM brain (needs GROQ_API_KEY)")
    ap.add_argument("--auto", action="store_true",
                    help="escalate to the LLM only when the keyword score is uncertain")
    ap.add_argument("--json", action="store_true", dest="as_json",
                    help="print the result as JSON instead of a summary line")
    args = ap.parse_args()

    jd_text = open(args.jd, encoding="utf-8").read()
    resume_text = open(args.resume, encoding="utf-8").read()

    brain, mode = route(jd_text, resume_text, args.llm, args.auto)
    open(args.ledger, "w", encoding="utf-8").close()  # truncate (delete may be blocked)

    agent = Agent(brain=brain, tools=build_registry(),
                  memory=Memory(path=args.ledger), guardrails=Guardrails())
    state = agent.run(goal="Assess job fit and tailor the resume",
                      context={"jd_text": jd_text, "resume_text": resume_text})

    report = state["artifacts"].get("report", "")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)

    art = state["artifacts"]
    fit = art.get("fit", {})
    if args.as_json:
        print(json.dumps({
            "brain": mode,
            "title": art.get("jd", {}).get("title"),
            "fit": fit,
            "keyword_match": art.get("match", {}),
            "learned": art.get("jd", {}).get("learned", []),
            "report_path": args.out,
        }, indent=2))
    else:
        print("[brain=%s] fit score: %s%% (%s)  ->  %s   (ledger: %s)"
              % (mode, fit.get("score", "?"), fit.get("source", "?"),
                 args.out, args.ledger))


if __name__ == "__main__":
    main()
