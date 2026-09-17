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


if __name__ == "__main__":
    test_end_to_end_mock()
    test_guardrail_blocks_invented_skill()
    test_llm_path_with_stub()
    test_router_bands()
    print("\nAll tests passed.")
