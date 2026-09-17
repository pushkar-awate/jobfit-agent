# jobfit-agent

An AI agent that reads a job description and your resume, scores the match the
way an applicant-tracking system would, tells you exactly which skills are
missing, and drafts tailored resume bullets that never claim a skill you do not
have.

It runs with **zero dependencies and zero API keys** out of the box, so anyone
can clone it and see it work in under a minute.

```
$ python -m app.run --jd examples/sample_jd.txt --resume examples/sample_resume.txt
[brain=mock] match score: 69%  ->  report.md   (ledger: decisions.jsonl)
```

## Why this exists

Job seekers apply into a black box: an ATS keyword-matches the resume before a
human ever reads it. This agent makes that step transparent. For every posting
it produces a match score, the matched and missing skills, and a set of tailored
bullets you can paste in, with a guardrail that refuses to invent experience you
do not have.

## What it does (example output)

```
# Job-fit report: MLOps Engineer

Match score: 69% (11 of 16 required skills)

## Matched skills
aws, ci/cd, docker, git, kubernetes, linux, machine learning, mlflow, mlops, python, tensorflow

## Gaps to address (in the JD, not in your resume)
fastapi, grafana, prometheus, pytorch, terraform

## Tailored bullet suggestions
...
```

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
`draft_bullets`, `write_report`, plus built-in `finish` and `raise_flag`.

### The brain is pluggable

The reasoning step is an interface, not a hard-coded model:

- **MockBrain** - a deterministic planner. No network, no key. This is why a
  fresh clone runs immediately and the tests are reproducible.
- **GroqBrain** - calls a hosted Llama model when `GROQ_API_KEY` is set, using
  only the Python standard library. The agent picks it automatically if the key
  is present.

```
export GROQ_API_KEY=your_free_key      # optional; from groq.com
python -m app.run --jd JD.txt --resume RESUME.txt --llm
```

Without `--llm` the agent uses the mock brain (zero setup). With `--llm` it uses Groq for two things: a **semantic fit score** (it reads the JD and resume and reasons about the match, recognizing that e.g. "agent loops" ~ "agentic workflows") and genuinely rewritten resume bullets. The deterministic keyword score is always shown alongside it for comparison.

### Guardrails

The agent is not trusted blindly. The drafting step is checked and rejected if it:

- references a skill that appears in the job description but **not** in the
  resume (no invented experience), or
- contains an em dash or en dash (a real document-style rule).

```
$ python tests/test_agent.py
PASS: end-to-end mock  (score=69%, steps=7, gaps=fastapi, grafana, prometheus, pytorch, terraform)
PASS: guardrail blocked invented skill  (draft references skills not in the resume: terraform)
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
examples/      a sample job description and resume
tests/         end-to-end and guardrail tests (run with plain python)
```

The split is deliberate. `agentcore/` knows nothing about resumes. Swap the tool
set and the same loop, brain, memory and guardrails drive a different agent -
which is exactly how this core is reused as the controller in a follow-up
project: a self-healing MLOps pipeline where the same loop watches a live model,
reasons about drift, and decides to retrain, promote or roll back.

## Quickstart

```
git clone <this repo>
cd jobfit-agent
python -m app.run --jd examples/sample_jd.txt --resume examples/sample_resume.txt
# -> writes report.md and decisions.jsonl
```

Requires Python 3.8+. No pip install needed.

## Modes

```
--llm     use the Groq LLM brain (needs GROQ_API_KEY): semantic fit score + real drafting
--auto    cost-aware routing: stay on the cheap mock brain when the keyword score is
          confident (a clear fit or a clear no-fit), escalate to the LLM only when unsure
--json    machine-readable output for piping into other tools
```

## Evaluation

A labeled test set lives in `eval/`. It measures the matcher against jobs known to
fit or not fit, so "the LLM is better" is a number, not a claim:

```
$ python eval/run_eval.py
[brain=mock  threshold=25]  accuracy 83% (5/6)  precision 100%  recall 75%
```

The mock brain's one miss is the agentic role, where keyword matching can't see the
semantic fit. Run `python eval/run_eval.py --llm` to measure the LLM brain on the same set.
