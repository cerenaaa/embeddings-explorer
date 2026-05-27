"""
Semantic search with cosine similarity and MMR re-ranking for diversity.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class SearchResult:
    text: str
    score: float
    index: int


def cosine_search(query_emb: np.ndarray, corpus_embs: np.ndarray,
                  texts: list[str], top_k: int = 10) -> list[SearchResult]:
    q = query_emb / (np.linalg.norm(query_emb) + 1e-9)
    C = corpus_embs / (np.linalg.norm(corpus_embs, axis=1, keepdims=True) + 1e-9)
    scores = C @ q
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [SearchResult(texts[i], float(scores[i]), int(i)) for i in top_idx]


def mmr_rerank(query_emb: np.ndarray, results: list[SearchResult],
               corpus_embs: np.ndarray, top_k: int = 5, lambda_: float = 0.5) -> list[SearchResult]:
    """
    Maximal Marginal Relevance: balances relevance and diversity.
    lambda_=1.0 → pure relevance, lambda_=0.0 → pure diversity.
    """
    if not results:
        return []
    indices = [r.index for r in results]
    embs = corpus_embs[indices]
    q = query_emb / (np.linalg.norm(query_emb) + 1e-9)
    relevance = embs @ q

    selected, remaining = [], list(range(len(results)))
    while len(selected) < top_k and remaining:
        if not selected:
            best = int(np.argmax(relevance[remaining]))
        else:
            sel_embs = embs[selected]
            sim_to_selected = (embs[remaining] @ sel_embs.T).max(axis=1)
            mmr_scores = lambda_ * relevance[remaining] - (1 - lambda_) * sim_to_selected
            best = remaining[int(np.argmax(mmr_scores))]
        selected.append(best)
        remaining.remove(best)

    return [SearchResult(results[i].text, float(relevance[i]), results[i].index) for i in selected]
