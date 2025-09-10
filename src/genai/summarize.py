# src/genai/summarize.py
from __future__ import annotations
import os, textwrap
from typing import Iterable, List

def _chunk_text(items: Iterable[str], max_chars: int = 3000) -> List[str]:
    buf, total = [], 0
    cur = []
    for s in items:
        s = (s or "").strip()
        if not s: continue
        if total + len(s) > max_chars and cur:
            buf.append("\n".join(cur)); cur=[s]; total=len(s)
        else:
            cur.append(s); total += len(s)
    if cur: buf.append("\n".join(cur))
    return buf

def offline_extractive_summary(texts: List[str], max_sentences: int = 8) -> str:
    # very small, dependency-free fallback: frequency scoring over words
    import re, math
    text = " ".join(texts)
    sents = re.split(r"(?<=[.!?])\s+", text)
    if not sents: return "No content to summarize."
    words = re.findall(r"[A-Za-z']+", text.lower())
    if not words: return "No content to summarize."
    stop = set("a an the is are to of for in on with and or at from by this that it".split())
    freq = {}
    for w in words:
        if w in stop: continue
        freq[w] = freq.get(w, 0) + 1
    scores=[]
    for s in sents:
        tokens = re.findall(r"[A-Za-z']+", s.lower())
        if not tokens: continue
        sc = sum(freq.get(w,0) for w in tokens) / (len(tokens) ** 0.7)
        scores.append((sc, s))
    top = [s for _, s in sorted(scores, reverse=True)[:max_sentences]]
    return " ".join(top)

def summarize_reviews(texts: List[str], prompt_hint: str | None = None) -> str:
    """
    Best-effort: use OpenAI if OPENAI_API_KEY present; else fallback.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return offline_extractive_summary(texts)

    # Lightweight OpenAI call; keep optional to avoid hard dep
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        chunks = _chunk_text(texts, max_chars=6000)
        points=[]
        for i,ch in enumerate(chunks,1):
            sys = "You are an analyst producing a concise customer-experience digest with bullets and themes."
            usr = f"""Summarize key customer pain points, root causes, and opportunities.
{('Context: ' + prompt_hint) if prompt_hint else ''}

Reviews:
{ch}
"""
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":sys},{"role":"user","content":usr}],
                temperature=0.2, max_tokens=400
            )
            points.append(resp.choices[0].message.content.strip())
        return "\n\n".join(points)
    except Exception:
        return offline_extractive_summary(texts)