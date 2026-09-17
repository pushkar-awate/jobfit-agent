"""Measure the matcher against labeled cases.

  python eval/run_eval.py                 # mock brain
  python eval/run_eval.py --llm           # Groq brain (needs GROQ_API_KEY)
  python eval/run_eval.py --threshold 30  # tune the fit/nofit cutoff
"""
from __future__ import annotations
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agentcore.loop import Agent
from agentcore.memory import Memory
from app.run import build_registry, pick_brain


def fit_score(jd_text, resume_text, brain):
    agent = Agent(brain=brain, tools=build_registry(), memory=Memory())
    state = agent.run("Assess job fit", {"jd_text": jd_text, "resume_text": resume_text})
    return state["artifacts"]["fit"]["score"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", default=os.path.join(ROOT, "examples", "sample_resume.txt"))
    ap.add_argument("--threshold", type=int, default=25)
    ap.add_argument("--llm", action="store_true")
    args = ap.parse_args()

    cases = json.load(open(os.path.join(ROOT, "eval", "cases.json"), encoding="utf-8"))
    resume_text = open(args.resume, encoding="utf-8").read()
    brain, mode = pick_brain(use_llm=args.llm)

    tp = fp = tn = fn = 0
    print("%-20s %-6s %-7s %-9s %s" % ("case", "label", "score", "predicted", "ok"))
    print("-" * 52)
    for c in cases:
        jd = open(os.path.join(ROOT, c["jd"]), encoding="utf-8").read()
        score = fit_score(jd, resume_text, brain)
        pred = "fit" if score >= args.threshold else "nofit"
        ok = pred == c["label"]
        if c["label"] == "fit":
            tp += ok; fn += (not ok)
        else:
            tn += ok; fp += (not ok)
        print("%-20s %-6s %-7s %-9s %s"
              % (c["name"], c["label"], "%d%%" % score, pred, "OK" if ok else "X"))

    total = len(cases)
    acc = round(100 * (tp + tn) / total)
    prec = round(100 * tp / max(1, tp + fp))
    rec = round(100 * tp / max(1, tp + fn))
    print("-" * 52)
    print("[brain=%s  threshold=%d]  accuracy %d%% (%d/%d)  precision %d%%  recall %d%%"
          % (mode, args.threshold, acc, tp + tn, total, prec, rec))


if __name__ == "__main__":
    main()
