"""agentcore: a small, from-scratch agent core.

Task-agnostic building blocks: a perceive-reason-guardrail-act-verify-remember
loop, a pluggable reasoning brain, a tool registry, an append-only decision
ledger, and guardrails. Swap the tool set and the same core drives a different
agent. That reuse is the point: this package later powers the self-healing
MLOps controller with a different set of tools.
"""
