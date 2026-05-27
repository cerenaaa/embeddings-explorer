"""
Semantic clustering of text embeddings.
K-Means with silhouette-based K selection + DBSCAN for density-based clusters.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize


@dataclass
class ClusterResult:
    labels: np.ndarray
    n_clusters: int
    silhouette: float
    method: str
    cluster_sizes: dict


class EmbeddingClusterer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def kmeans(self, embeddings: np.ndarray, k: int = None, k_range: tuple = (2, 10)) -> ClusterResult:
        X = normalize(embeddings)
        if k is None:
            scores = {}
            for ki in range(k_range[0], min(k_range[1]+1, len(X))):
                km = KMeans(n_clusters=ki, random_state=self.random_state, n_init=10)
                labels = km.fit_predict(X)
                scores[ki] = silhouette_score(X, labels)
            k = max(scores, key=scores.get)
            print(f"Selected K={k} (silhouette={scores[k]:.3f})")

        km = KMeans(n_clusters=k, random_state=self.random_state, n_init=20)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels) if k > 1 else 0.0
        sizes = {int(i): int(np.sum(labels == i)) for i in range(k)}
        return ClusterResult(labels=labels, n_clusters=k, silhouette=sil, method="kmeans", cluster_sizes=sizes)

    def dbscan(self, embeddings: np.ndarray, eps: float = 0.3, min_samples: int = 3) -> ClusterResult:
        X = normalize(embeddings)
        db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
        labels = db.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise = int(np.sum(labels == -1))
        sil = silhouette_score(X, labels) if n_clusters > 1 and noise < len(labels) else 0.0
        sizes = {int(i): int(np.sum(labels == i)) for i in set(labels)}
        print(f"DBSCAN: {n_clusters} clusters, {noise} noise points")
        return ClusterResult(labels=labels, n_clusters=n_clusters, silhouette=sil, method="dbscan", cluster_sizes=sizes)

    def get_cluster_representatives(self, texts: list[str], embeddings: np.ndarray, labels: np.ndarray) -> dict[int, str]:
        """Find the text closest to each cluster centroid."""
        reps = {}
        for cluster_id in set(labels):
            if cluster_id == -1:
                continue
            mask = labels == cluster_id
            cluster_embs = embeddings[mask]
            cluster_texts = [t for t, m in zip(texts, mask) if m]
            centroid = cluster_embs.mean(axis=0)
            sims = cluster_embs @ centroid / (np.linalg.norm(cluster_embs, axis=1) * np.linalg.norm(centroid) + 1e-9)
            reps[cluster_id] = cluster_texts[sims.argmax()]
        return reps
