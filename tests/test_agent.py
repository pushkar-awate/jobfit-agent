"""Runnable with plain `python tests/test_agent.py` (no pytest needed)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from agentcore.loop import Agent, Decision
from agentcore.memory import Memory
from agentcore.brain import MockBrain
from agentcore.guardrails import Guardrails, GuardrailError
from app.run import build_registry


def _samples():
    jd = open(os.path.join(ROOT, "examples", "sample_jd.txt"), encoding="utf-8").read()
    resume = open(os.path.join(ROOT, "examples", "sample_resume.txt"), encoding="utf-8").read()
    return jd, resume


def test_end_to_end_mock():
    jd, resume = _samples()
    agent = Agent(brain=MockBrain(), tools=build_registry(), memory=Memory())
    state = agent.run("Assess job fit and tailor the resume",
                      {"jd_text": jd, "resume_text": resume})
    assert state["done"], "agent did not finish"
    match = state["artifacts"]["match"]
    fit = state["artifacts"]["fit"]
    assert fit["source"] == "keyword"
    assert fit["score"] == match["score"], "mock fit should equal keyword score"
    assert "terraform" in match["missing"] and "pytorch" in match["missing"]
    assert state["artifacts"]["report"].startswith("# Job-fit report")
    draft = state["artifacts"]["bullets"].lower()
    for gap in match["missing"]:
        assert gap not in draft, "invented skill leaked into draft: %s" % gap
    print("PASS: end-to-end mock  (fit=%d%% source=%s)" % (fit["score"], fit["source"]))


def test_guardrail_blocks_invented_skill():
    state = {"artifacts": {"gaps": {"missing": ["terraform"]}}}
    dec = Decision("draft_bullets", {}, "")
    try:
        Guardrails().check_output(dec, "Experienced with terraform.", state)
    except GuardrailError as e:
        print("PASS: guardrail blocked invented skill  (%s)" % e)
        return
    raise AssertionError("guardrail did NOT block an invented skill")


class StubLLM(MockBrain):
    """Simulates the Groq brain's assess_fit output (no network)."""
    def generate(self, task, payload):
        if task == "assess_fit":
            return ('```json\n{"score": 78, "strengths": ["agentic AI", "Python", '
                    '"Kubernetes"], "gaps": ["MCP", "LangGraph"], "rationale": '
                    '"Strong agentic and cloud background; missing MCP and LangGraph.", '
                    '"source": "llm"}\n```')
        return super().generate(task, payload)


def test_llm_path_with_stub():
    jd, resume = _samples()
    agent = Agent(brain=StubLLM(), tools=build_registry(), memory=Memory())
    state = agent.run("Assess job fit", {"jd_text": jd, "resume_text": resume})
    fit = state["artifacts"]["fit"]
    report = state["artifacts"]["report"]
    assert fit["source"] == "llm", "expected LLM-sourced fit"
    assert fit["score"] == 78, "expected parsed score 78, got %s" % fit["score"]
    assert "(LLM reasoning)" in report and "78%" in report
    assert "missing MCP and LangGraph" in report
    print("PASS: LLM path via stub  (fit=%d%% source=%s, robust JSON parse)"
          % (fit["score"], fit["source"]))



def test_router_bands():
    import os as _os
    from app.run import keyword_score, route
    mlops = open(_os.path.join(ROOT, "eval", "cases", "mlops_engineer.txt"), encoding="utf-8").read()
    nurse = open(_os.path.join(ROOT, "eval", "cases", "registered_nurse.txt"), encoding="utf-8").read()
    resume = open(_os.path.join(ROOT, "examples", "sample_resume.txt"), encoding="utf-8").read()
    assert keyword_score(mlops, resume) >= 55, "mlops should be a confident fit"
    assert keyword_score(nurse, resume) <= 5, "nurse should be a confident nofit"
    _os.environ.pop("GROQ_API_KEY", None)
    _, mode = route(mlops, resume, use_llm=False, auto=True)
    assert mode == "mock", "no key -> must stay on mock"
    print("PASS: router bands (mlops>=55, nurse<=5, no-key stays mock)")


def test_job_board_scoring():
    from app.jobs import sample_jobs, _strip_html
    from app.skills import extract_skills
    resume = open(os.path.join(ROOT, "examples", "sample_resume.txt"), encoding="utf-8").read()
    rskills = set(extract_skills(resume))

    assert _strip_html("<p>Python &amp; <b>PyTorch</b></p>") == "Python & PyTorch"

    def score(job):
        jd = set(extract_skills(job["match_text"]))
        matched = jd & rskills
        nm, njd, nr = len(matched), len(jd), len(rskills)
        if nm == 0 or njd == 0 or nr == 0:
            return 0.0, nm
        recall, precision = nm / njd, nm / nr
        f1 = 2 * recall * precision / (recall + precision)
        return f1, nm

    # zero-overlap posting must not divide-by-zero and must score 0
    assert score({"match_text": "barista latte espresso"})[0] == 0.0

    jobs = sample_jobs()
    ranked = sorted(jobs, key=lambda j: (score(j)[0], score(j)[1]), reverse=True)
    order = [j["title"] for j in ranked]
    pos = {t: i for i, t in enumerate(order)}

    # the old bug: "Applied Scientist" (3 skills matched) outranked richer roles.
    # F1 must now place roles where the user matches MORE skills above it.
    for richer in ["Machine Learning Engineer", "MLOps Engineer",
                   "Machine Learning Platform Engineer"]:
        assert pos[richer] < pos["Applied Scientist, Forecasting"], (
            "%s (more skills matched) should outrank the thin 3-skill role" % richer)

    top = ranked[0]
    assert top["title"] == "Machine Learning Engineer", (
        "most-matching role should rank first, got %r" % top["title"])
    print("PASS: job-board F1 ranking  (top=%s, thin role demoted)" % top["title"])


def test_query_filter():
    from app.jobs import sample_jobs, matches_query
    jobs = sample_jobs()

    def hits(q):
        return {j["title"] for j in jobs if matches_query(j, q)}

    # different searches must return different, on-topic sets (the reported bug
    # was every search returning the same list)
    assert hits("ai engineer") != hits("mlops"), "distinct queries must differ"
    assert "AI Engineer (LLM / Agents)" in hits("ai engineer")
    assert "MLOps Engineer" in hits("mlops")
    # an off-topic search matches nothing (app then shows a closest-by-fit note)
    assert hits("registered nurse") == set()
    # a query of only generic words can't narrow, so it keeps everything
    assert matches_query(jobs[0], "engineer")
    print("PASS: query filter  (ai!=mlops, nurse=0, generic keeps all)")


if __name__ == "__main__":
    test_end_to_end_mock()
    test_guardrail_blocks_invented_skill()
    test_llm_path_with_stub()
    test_router_bands()
    test_job_board_scoring()
    test_query_filter()
    print("\nAll tests passed.")
