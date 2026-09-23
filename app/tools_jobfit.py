"""Task-specific tools. Each takes (state, **args) and returns a result.

The reasoning prompts live HERE, in the app layer - not in the shared
agentcore brain. A tool builds a prompt for its task and asks the brain to
`complete` it; if the brain returns nothing (the deterministic MockBrain, or a
failed LLM call), the tool falls back to a keyword/heuristic result. That split
is what keeps agentcore task-agnostic and reusable across projects.
"""
from __future__ import annotations
import json

from .skills import extract_skills, learn_from


def _extract_json(text):
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start >= 0 and end > start else text


# --- prompt builders (task-specific, app-owned) ---------------------------

def _assess_fit_prompt(jd_text, resume_text):
    return (
        "Assess how well a candidate fits a role. Read the JOB DESCRIPTION and "
        "RESUME and return ONLY a JSON object:\n"
        '{"score": <0-100 integer>, "strengths": [...], "gaps": [...], '
        '"rationale": <2-3 sentences>}.\n'
        "Treat semantic equivalents as matches (e.g. 'agent loops' ~ 'agentic "
        "workflows', 'Claude' ~ 'Anthropic'). Do NOT invent experience the "
        "resume lacks.\n\nJOB DESCRIPTION:\n%s\n\nRESUME:\n%s"
        % (jd_text[:4000], resume_text[:4000]))


def _draft_bullets_prompt(role, matched_keywords, resume_bullets):
    return (
        "Rewrite these resume bullets to be crisp, quantified where the "
        "original allows, and tailored to the role. Keep every claim truthful: "
        "do NOT invent tools, skills or metrics, and do NOT stuff keywords in "
        "unnaturally. Where genuinely relevant, reflect these real strengths: "
        "%s. Do NOT use em dashes or en dashes. Return 4-6 plain bullets "
        "starting with '- '.\nRole: %s\nOriginal bullets:\n%s"
        % (", ".join(matched_keywords[:6]), role, "\n".join(resume_bullets)))


def _draft_bullets_mock(role, matched_keywords, resume_bullets):
    """Deterministic tailoring: only re-states real bullets and real strengths,
    never invents a skill the resume lacks."""
    kws = matched_keywords
    top = ", ".join(kws[:6]) if kws else "the listed skills"
    lines = ["Tailored for %s. Emphasized strengths: %s." % (role, top)]
    for b in resume_bullets[:4]:
        lines.append("- %s" % b)
    if kws:
        lines.append("- Directly relevant hands-on experience with %s." % top)
    return "\n".join(lines)


# --- tools -----------------------------------------------------------------

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
    """Semantic fit assessment. An LLM brain reasons over the prompt; a mock
    brain returns nothing, so we fall back to the deterministic keyword match."""
    brain = state["brain"]
    km = state["artifacts"].get("match", {})
    prompt = _assess_fit_prompt(state["artifacts"]["jd"]["raw"],
                                state["artifacts"]["resume"]["raw"])
    raw = brain.complete(prompt)
    fit = None
    if raw:
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
            fit = None
    if fit is None:
        fit = {"score": km.get("score", 0), "strengths": km.get("matched", []),
               "gaps": km.get("missing", []),
               "rationale": ("Deterministic keyword match (no LLM reasoning). "
                             "Run with --llm for a semantic assessment."),
               "source": "keyword"}
    state["artifacts"]["fit"] = fit
    return fit


def draft_bullets(state, **kw):
    brain = state["brain"]
    fit = state["artifacts"].get("fit", {})
    # prefer the (LLM) fit assessment's strengths; fall back to keyword matches
    strengths = fit.get("strengths") or state["artifacts"]["match"]["matched"]
    role = state["artifacts"]["jd"]["title"]
    resume_bullets = state["artifacts"]["resume"]["bullets"]
    raw = brain.complete(_draft_bullets_prompt(role, strengths, resume_bullets))
    text = raw or _draft_bullets_mock(role, strengths, resume_bullets)
    text = text.replace("‑", "-")  # non-breaking hyphen -> hyphen
    text = "\n".join(line.rstrip() for line in text.splitlines() if line.strip())
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
