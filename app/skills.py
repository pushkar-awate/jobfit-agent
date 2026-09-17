"""Skill extraction that (a) knows synonyms and (b) learns new terms over time.

- SKILL_LEXICON: curated, high-precision known skills.
- ALIASES: many surface forms -> one canonical concept, so 'agent loops',
  'agentic workflows' and 'claude code' all count as the same thing.
- learned_skills.json: terms the agent discovered in past postings. It grows
  every run, so the agent gets less blind the more job descriptions it sees.
"""
from __future__ import annotations
import json
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
LEARNED_PATH = os.path.join(_HERE, "data", "learned_skills.json")

SKILL_LEXICON = [
    # languages
    "python", "java", "c++", "sql", "nosql", "bash",
    # infra / devops
    "git", "docker", "kubernetes", "k8s", "terraform", "ci/cd", "linux",
    "agile", "microservices", "rest api", "servicenow",
    # cloud
    "aws", "azure", "gcp", "sagemaker", "aks", "gke", "eks", "ec2", "ecs",
    "fargate", "lambda", "s3", "dynamodb",
    # ml / ds
    "pytorch", "tensorflow", "keras", "scikit-learn", "xgboost", "pandas",
    "numpy", "optuna", "hyperparameter tuning", "spark", "kafka", "airflow",
    "kestra", "mlflow", "machine learning", "deep learning",
    "reinforcement learning", "nlp", "computer vision", "mlops", "data mining",
    "data science",
    # geospatial
    "geopandas", "qgis",
    # serving / monitoring
    "fastapi", "flask", "django", "prometheus", "grafana", "dynatrace",
    "observability", "model drift",
    # llm / agents
    "llm", "rag", "transformers", "hugging face", "agentic", "agent loop",
    "mcp", "langchain", "langgraph", "anthropic", "openai", "vertex ai",
    "gemini", "claude", "tool-calling", "tool calling", "spec-first",
    "prompt engineering", "full stack", "full-stack", "coding agent",
    "agent tooling", "ai agent",
]

# surface form -> canonical concept (both JD and resume are normalized through this)
ALIASES = {
    "agent loop": "agentic ai", "agent loops": "agentic ai",
    "agentic": "agentic ai", "agentic workflows": "agentic ai",
    "ai coding agent": "agentic ai", "coding agent": "agentic ai",
    "claude code": "agentic ai", "coding agent": "agentic ai",
    "agent tooling": "agentic ai", "ai agent": "agentic ai",
    "tool-calling": "tool use", "tool calling": "tool use",
    "full-stack": "full stack",
    "k8s": "kubernetes",
    "claude": "anthropic",
}

# acronyms/words that look like skills but are not
_STOP = {"gst", "ai", "ml", "pty", "ltd", "abn", "eod", "wfh", "us", "usa",
         "uk", "au", "hr", "it", "qa", "ui", "ux", "ceo", "cto", "rmit",
         "json", "csv", "pdf", "faq", "gpa"}


def _load_learned():
    try:
        with open(LEARNED_PATH, encoding="utf-8") as f:
            return list(json.load(f))
    except Exception:
        return []


def _save_learned(terms):
    try:
        os.makedirs(os.path.dirname(LEARNED_PATH), exist_ok=True)
        with open(LEARNED_PATH, "w", encoding="utf-8") as f:
            json.dump(sorted(set(terms)), f, indent=2)
    except Exception:
        pass


def _known_terms():
    return set(SKILL_LEXICON) | set(_load_learned())


def canon(term):
    return ALIASES.get(term, term)


def extract_raw(text):
    """Every known surface form present in the text (before normalizing)."""
    low = text.lower()
    found = []
    for skill in _known_terms():
        pat = r"(?<![a-z0-9])" + re.escape(skill) + r"s?(?![a-z0-9])"
        if re.search(pat, low):
            found.append(skill)
    return found


def extract_skills(text):
    """Canonical skill concepts present in the text."""
    return sorted({canon(t) for t in extract_raw(text)})


def discover_terms(text):
    """Skill-like terms in the text: CamelCase tech and non-stop acronyms."""
    cands = set()
    for m in re.findall(r"\b[A-Z][a-z]+[A-Z][A-Za-z]+\b", text):   # LangGraph
        cands.add(m.lower())
    for m in re.findall(r"\b[A-Z]{3,5}\b", text):                  # MCP, GKE
        low = m.lower()
        if low not in _STOP:
            cands.add(low)
    return cands


LEARNED_CAP = 300  # keep the self-learning store bounded


def learn_from(text):
    """Add genuinely new discovered terms to the persistent vocabulary.

    Heuristic and deliberately conservative: 3-5 letter acronyms or CamelCase
    tech names only, never seen before, bounded by LEARNED_CAP. This is a
    convenience, not a curated taxonomy - see the Limitations note in README.
    """
    known = _known_terms()
    new = sorted(t for t in discover_terms(text)
                 if t not in known and t not in ALIASES and 3 <= len(t) <= 30)
    if new:
        merged = sorted(set(_load_learned()) | set(new))[:LEARNED_CAP]
        _save_learned(merged)
    return new
