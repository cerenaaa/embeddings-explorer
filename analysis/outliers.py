"""Embedding-space outlier detection using isolation forest and LOF."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import normalize


@dataclass
class OutlierResult:
    indices: list[int]
    scores: np.ndarray
    method: str
    contamination: float


class EmbeddingOutlierDetector:
    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state

    def isolation_forest(self, embeddings: np.ndarray) -> OutlierResult:
        X = normalize(embeddings)
        clf = IsolationForest(contamination=self.contamination, random_state=self.random_state)
        preds = clf.fit_predict(X)
        scores = -clf.score_samples(X)
        outlier_idx = [i for i, p in enumerate(preds) if p == -1]
        return OutlierResult(indices=outlier_idx, scores=scores, method="isolation_forest", contamination=self.contamination)

    def lof(self, embeddings: np.ndarray, n_neighbors: int = 20) -> OutlierResult:
        X = normalize(embeddings)
        clf = LocalOutlierFactor(n_neighbors=min(n_neighbors, len(X)-1), contamination=self.contamination)
        preds = clf.fit_predict(X)
        scores = -clf.negative_outlier_factor_
        outlier_idx = [i for i, p in enumerate(preds) if p == -1]
        return OutlierResult(indices=outlier_idx, scores=scores, method="lof", contamination=self.contamination)
