# jobfit-agent

A from-scratch AI agent that reads a job description and your resume, scores how
well they match, tells you which skills are missing, and drafts tailored resume
bullets that never claim a skill you do not have.

It runs with **zero dependencies and zero API keys** out of the box, so anyone
can clone it and see it work in under a minute. That default path is a
deterministic keyword-and-synonym matcher. An optional LLM brain (`--llm`) adds
real semantic reasoning on top of the same agent.

```
$ python -m app.run --jd examples/sample_jd.txt --resume examples/sample_resume.txt
[brain=mock] fit score: 69% (keyword)  ->  report.md   (ledger: decisions.jsonl)
```

## Why this exists

Job seekers apply into a black box: an applicant-tracking system keyword-matches
the resume before a human ever reads it. This agent makes that step transparent.
For every posting it produces a fit score, the matched and missing skills, and a
set of tailored bullets you can paste in, with a guardrail that refuses to invent
experience you do not have.

## What it does (real output)

```
# Job-fit report: MLOps Engineer

Fit score: 69% (keyword match)

## Strengths for this role
aws, ci/cd, docker, git, kubernetes, linux, machine learning, mlflow, mlops, python, tensorflow

## Gaps to address
fastapi, grafana, prometheus, pytorch, terraform

## ATS keyword match (deterministic)
69% - matched: aws, ci/cd, docker, git, ...

## Tailored bullet suggestions
...
```

## A real LLM run (why the LLM brain matters)

On a real "AI Agent Engineer" posting, the deterministic keyword brain scored the
match at **20%** - it only matched the literal words "agentic ai" and "anthropic".
The LLM brain (`--llm`, Groq `openai/gpt-oss-20b`) scored the same pairing at
**70%**, because it saw the semantic fit that keywords miss:

> The candidate brings solid AI/ML engineering and full-stack infrastructure
> skills, including agentic workflows and Claude Code usage, which align with the
> role's focus on AI-driven features. However, the resume lacks key details on MCP
> integration, LangChain/LangGraph, spec-first development, and secure
> coding/testing practices that the job explicitly requires, resulting in a
> moderate fit score.

That gap - 20% literal vs 70% semantic - is the whole reason the pluggable LLM
brain exists. A run makes just two model calls (the fit assessment and the
drafting); everything else is deterministic.

## How it works

This is not a single prompt to an LLM. It is a small agent loop, written from
scratch, that runs six steps every turn:

1. **Perceive** - read the current state (what has been done, what artifacts exist).
2. **Reason** - the brain picks the single next tool to run.
3. **Guardrail** - validate that choice before doing anything (e.g. stop it looping).
4. **Act** - run the chosen tool.
5. **Verify** - validate the *output* (no invented skills, no banned characters).
6. **Remember** - append the decision, rationale and result to a ledger.

The tools are `parse_jd`, `parse_resume`, `score_match`, `identify_gaps`,
`assess_fit`, `draft_bullets`, `write_report`, plus built-in `finish` and
`raise_flag`.

### The brain is pluggable

The reasoning step is an interface, not a hard-coded model:

- **MockBrain** - a deterministic planner and rule-based matcher. No network, no
  key. This is the **default**, which is why a fresh clone runs immediately and
  the tests are reproducible.
- **GroqBrain** - calls a hosted Llama model, opt-in via `--llm` and a free
  `GROQ_API_KEY`, using only the Python standard library.

```
export GROQ_API_KEY=your_free_key
python -m app.run --jd JD.txt --resume RESUME.txt --llm
```

Without `--llm` the agent uses the mock brain. With `--llm` it uses Groq for two
things: a **semantic fit score** (it reads the JD and resume and reasons about the
match, recognizing that e.g. "agent loops" ~ "agentic workflows") and genuinely
rewritten resume bullets. The deterministic keyword score is always shown
alongside for comparison.

> **Note on the LLM path.** The GroqBrain request/response handling is covered by
> an automated test that feeds a simulated model response (`test_llm_path_with_stub`),
> so the parsing and report path are verified. Confirm the *live* end-to-end call
> by running `--llm` against the real API on a machine with network access to Groq.

### Guardrails

The agent is not trusted blindly. The drafting step is checked and rejected if it:

- references a skill that appears in the job description but **not** in the
  resume (no invented experience), or
- contains an em dash or en dash (a document-style rule).

```
$ python tests/test_agent.py
PASS: end-to-end mock  (fit=69% source=keyword)
PASS: guardrail blocked invented skill  (draft references skills not in the resume: terraform)
PASS: LLM path via stub  (fit=78% source=llm, robust JSON parse)
PASS: router bands (mlops>=55, nurse<=5, no-key stays mock)
```

### Every decision is logged

`decisions.jsonl` is the agent's audit trail - one line per step, with the tool,
the rationale, and whether it succeeded:

```json
{"step": 3, "tool": "score_match", "rationale": "plan step: score_match", "ok": true}
{"step": 5, "tool": "draft_bullets", "rationale": "plan step: draft_bullets", "ok": true}
```

## Project layout

```
agentcore/     the reusable, task-agnostic core (loop, brain, tools, memory, guardrails)
app/           the Resume <-> JD task layer built on top of it
eval/          a labeled smoke-test set and a runner
examples/      a sample job description and resume
tests/         end-to-end, guardrail, LLM-stub and router tests (run with plain python)
```

The split is deliberate. `agentcore/` knows nothing about resumes. Swap the tool
set and the same loop, brain, memory and guardrails drive a different agent -
the intended next use is the controller of a self-healing MLOps pipeline, where
the same loop watches a live model, reasons about drift, and decides to retrain,
promote or roll back.

## Modes

```
--llm     use the Groq LLM brain (needs GROQ_API_KEY): semantic fit score + real drafting
--auto    cost-aware routing: stay on the cheap mock brain when the keyword score is
          confident (a clear fit or a clear no-fit), escalate to the LLM only when unsure
--json    machine-readable output for piping into other tools
```

## Troubleshooting

`diagnose.py` is a standalone check for the `--llm` path: run `python diagnose.py` (with `GROQ_API_KEY` set) to see the real Groq response and the list of models your key can use.

## Evaluation

`eval/` holds a small **labeled smoke test**: six job postings marked fit / no-fit
against one sample resume, scored and compared to the labels.

```
$ python eval/run_eval.py
[brain=mock  threshold=25]  accuracy 83% (5/6)  precision 100%  recall 75%
```

This is the **mock (keyword) brain**, not the LLM, and six cases is a smoke test,
not a statistically meaningful benchmark - read it as "does the pipeline behave
sensibly," not as a headline accuracy claim. Its one miss is the agentic role,
where keyword matching cannot see the semantic fit; `python eval/run_eval.py --llm`
measures the LLM brain on the same set.

## Limitations

Honest boundaries, by design:

- **The default brain is keyword-based**, so it misses semantic equivalents the
  alias map does not cover. That is what `--llm` is for.
- **The eval is a 6-case smoke test** against a single resume, with no independent
  ground truth or human baseline. It shows the pipeline works, not how accurate it
  is in general. Growing it to a labeled set of 30-50 postings is the next step.
- **The self-learning vocabulary is a heuristic, not a curated taxonomy.** It only
  records 3-5 letter acronyms and CamelCase tech names, capped in size, and is not
  guaranteed to improve matching - treat `learned_skills.json` as a convenience.
- **The live LLM path needs a real run to confirm** (see the note above); it is
  currently verified only against a simulated response in tests.

## Quickstart

```
git clone https://github.com/pushkar-awate/jobfit-agent.git
cd jobfit-agent
python -m app.run --jd examples/sample_jd.txt --resume examples/sample_resume.txt
# -> writes report.md and decisions.jsonl
```

Requires Python 3.8+. No pip install needed.
