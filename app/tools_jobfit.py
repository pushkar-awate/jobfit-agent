"""Task-specific tools. Each takes (state, **args) and returns a result."""
from __future__ import annotations
import json

from .skills import extract_skills, learn_from


def _extract_json(text):
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start >= 0 and end > start else text


def parse_jd(state, **kw):
    text = state["context"]["jd_text"]
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    _t = lines[0] if lines else "Unknown role"
    if len(_t) > 70:
        _t = _t[:67].rstrip() + "..."
    learned = learn_from(text)
    jd = {"title": _t, "skills": extract_skills(text),
          "learned": learned, "raw": text}
    state["artifacts"]["jd"] = jd
    return {"title": jd["title"], "skills": jd["skills"], "learned": learned}


def parse_resume(state, **kw):
    text = state["context"]["resume_text"]
    bullets = [l.strip(" -*•\t") for l in text.splitlines()
               if l.strip().startswith(("-", "*", "•"))]
    resume = {"skills": extract_skills(text), "bullets": bullets, "raw": text}
    state["artifacts"]["resume"] = resume
    return {"skills": resume["skills"], "bullets": len(bullets)}


def score_match(state, **kw):
    jd = set(state["artifacts"]["jd"]["skills"])
    res = set(state["artifacts"]["resume"]["skills"])
    matched, missing = sorted(jd & res), sorted(jd - res)
    match = {"score": round(100 * len(matched) / max(1, len(jd))),
             "matched": matched, "missing": missing, "jd_total": len(jd)}
    state["artifacts"]["match"] = match
    return match


def identify_gaps(state, **kw):
    missing = state["artifacts"]["match"]["missing"]
    gaps = {"missing": missing, "priority": missing[:5]}
    state["artifacts"]["gaps"] = gaps
    return gaps


def assess_fit(state, **kw):
    """Semantic fit assessment. LLM brain reasons; mock brain reuses keywords."""
    brain = state["brain"]
    km = state["artifacts"].get("match", {})
    payload = {
        "jd_text": state["artifacts"]["jd"]["raw"],
        "resume_text": state["artifacts"]["resume"]["raw"],
        "keyword_match": km,
    }
    raw = brain.generate("assess_fit", payload)
    try:
        d = json.loads(_extract_json(raw))
        fit = {
            "score": int(d.get("score", km.get("score", 0))),
            "strengths": d.get("strengths", km.get("matched", [])),
            "gaps": d.get("gaps", km.get("missing", [])),
            "rationale": d.get("rationale", ""),
            "source": d.get("source", "llm"),
        }
    except Exception:
        fit = {"score": km.get("score", 0), "strengths": km.get("matched", []),
               "gaps": km.get("missing", []),
               "rationale": "Could not parse model output; used keyword match.",
               "source": "keyword"}
    state["artifacts"]["fit"] = fit
    return fit


def draft_bullets(state, **kw):
    brain = state["brain"]
    payload = {
        "role": state["artifacts"]["jd"]["title"],
        "matched_keywords": state["artifacts"]["match"]["matched"],
        "resume_bullets": state["artifacts"]["resume"]["bullets"],
    }
    text = brain.generate("draft_bullets", payload)
    state["artifacts"]["bullets"] = text
    return text


def write_report(state, **kw):
    jd = state["artifacts"]["jd"]
    match = state["artifacts"]["match"]
    fit = state["artifacts"].get("fit", {})
    bullets = state["artifacts"].get("bullets", "")
    src = fit.get("source", "keyword")
    label = "LLM reasoning" if src == "llm" else "keyword match"

    out = ["# Job-fit report: %s" % jd["title"], ""]
    out += ["**Fit score:** %s%% (%s)" % (fit.get("score", match["score"]), label), ""]
    if fit.get("rationale"):
        out += ["> " + fit["rationale"], ""]
    out += ["## Strengths for this role",
            ", ".join(fit.get("strengths", match["matched"])) or "_none_", ""]
    out += ["## Gaps to address",
            ", ".join(fit.get("gaps", match["missing"])) or "_none_", ""]
    out += ["## ATS keyword match (deterministic)",
            "%d%% - matched: %s" % (match["score"], ", ".join(match["matched"]) or "none"),
            ""]
    out += ["## Tailored bullet suggestions", bullets or "_none_"]
    learned = jd.get("learned", [])
    if learned:
        out += ["", "## New skills the agent learned from this posting",
                ", ".join(learned)]
    report = "\n".join(out)
    state["artifacts"]["report"] = report
    return report
