"""
Deterministic keyword/intent router — used ONLY when no LLM provider is
configured (no GROQ_API_KEY / OPENAI_API_KEY). It's a real, rule-based
NLU router (not a placeholder that always returns one tool), covering the
same task taxonomy the LLM function-calling router uses:

  describe   -> VQA / land-cover description
  ground     -> referring segmentation ("highlight/locate X")
  change     -> bi-temporal change detection
  fuse       -> optical + SAR fusion
  trend      -> change trend Q&A ("has X increased/decreased")
"""
from __future__ import annotations

TASK_KEYWORDS = {
    "trend": ["increased", "decreased", "grown", "shrunk", "trend", "unchanged", "remained"],
    "change": ["change", "changed", "between these two dates", "compare", "difference", "before and after"],
    "fuse": ["sar", "fuse", "fusion", "optical and sar", "together", "both images"],
    "ground": ["highlight", "locate", "point to", "where is", "referred to", "mark the"],
    "describe": ["describe", "what is", "land cover", "land-cover", "objects", "identify"],
}


def route(query: str, has_pair: bool, has_optical_sar: bool) -> str:
    q = query.lower()
    scores = {task: sum(1 for kw in kws if kw in q) for task, kws in TASK_KEYWORDS.items()}
    best_task = max(scores, key=scores.get)

    # Input-config overrides (mirrors the PS's "input configuration" clause)
    if has_optical_sar:
        return "fuse"
    if scores[best_task] == 0:
        return "change" if has_pair else "describe"
    return best_task
