"""
evaluator.py — RAG evaluation using direct Groq API calls. Zero ragas dependency.

Metrics implemented from scratch (label-free, no ground truth needed):
  - faithfulness        : fraction of answer claims supported by retrieved context
  - context_utilization : whether retrieved context was useful for the answer
"""
from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

import config

RAGAS_AVAILABLE = True   # always True — no external eval library needed
RAGAS_ERROR = ""


def _ask_groq(prompt: str) -> str:
    """Call Groq API and return the response text."""
    from groq import Groq
    client = Groq(api_key=config.GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=256,
    )
    return resp.choices[0].message.content.strip()


def _score_faithfulness(question: str, answer: str, contexts: list[str]) -> float:
    """Ask LLM to score how well the answer is grounded in the context (0.0 - 1.0)."""
    context_text = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    prompt = f"""You are evaluating whether an AI answer is factually grounded in the provided context.

QUESTION: {question}

CONTEXT:
{context_text}

ANSWER: {answer}

Task: Evaluate each claim made in the answer. Count how many claims are directly supported by the context vs. how many are not.

Respond with ONLY a JSON object in this exact format:
{{"supported": <number>, "total": <number>, "score": <float between 0 and 1>}}

Where score = supported / total. If the answer makes no claims, return score 1.0."""

    try:
        raw = _ask_groq(prompt)
        match = re.search(r'\{[^}]+\}', raw)
        if match:
            data = json.loads(match.group())
            return float(data.get("score", 0.0))
    except Exception:
        pass
    return 0.0


def _score_context_utilization(question: str, answer: str, contexts: list[str]) -> float:
    """Ask LLM to score how much the context was used in generating the answer (0.0 - 1.0)."""
    context_text = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    prompt = f"""You are evaluating whether the retrieved context was actually useful for answering a question.

QUESTION: {question}

CONTEXT:
{context_text}

ANSWER: {answer}

Task: Score how much of the provided context was relevant and used in generating the answer.
- 1.0 = context was highly relevant and clearly used to form the answer
- 0.5 = context was partially relevant or only partly used
- 0.0 = context was ignored or irrelevant to the answer

Respond with ONLY a JSON object in this exact format:
{{"score": <float between 0 and 1>, "reason": "<one sentence>"}}"""

    try:
        raw = _ask_groq(prompt)
        match = re.search(r'\{[^}]+\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return float(data.get("score", 0.0))
    except Exception:
        pass
    return 0.0


def run_evaluation(
    qa_pairs: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Evaluate QA pairs. Each pair needs: question, answer, contexts (list[str]).
    Returns (per_question_df, aggregate_scores_dict).
    """
    if not qa_pairs:
        raise ValueError("No QA pairs provided.")

    rows = []
    for p in qa_pairs:
        q = p["question"]
        a = p["answer"]
        c = p["contexts"]
        faith = _score_faithfulness(q, a, c)
        util  = _score_context_utilization(q, a, c)
        rows.append({
            "question":            q,
            "answer":              a[:120] + "…" if len(a) > 120 else a,
            "faithfulness":        faith,
            "context_utilization": util,
        })

    df = pd.DataFrame(rows)
    agg = {
        "faithfulness":        round(float(df["faithfulness"].mean()), 4),
        "context_utilization": round(float(df["context_utilization"].mean()), 4),
    }
    return df, agg


def score_color(score: float) -> str:
    if score >= 0.75:
        return "#22c55e"
    if score >= 0.50:
        return "#f59e0b"
    return "#ef4444"


def interpret_score(metric: str, score: float) -> str:
    thresholds = {
        "faithfulness":        ("Hallucination risk",  "Partially grounded", "Well-grounded"),
        "context_utilization": ("Context ignored",     "Partially used",     "Context well-used"),
    }
    labels = thresholds.get(metric, ("Poor", "Moderate", "Good"))
    if score >= 0.75:
        return labels[2]
    if score >= 0.50:
        return labels[1]
    return labels[0]