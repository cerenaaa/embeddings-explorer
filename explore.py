"""CLI for embedding exploration tasks."""
import argparse
import numpy as np

SAMPLE_TEXTS = [
    "gradient descent optimization in neural networks",
    "stochastic gradient descent and learning rate schedules",
    "customer churn prediction with machine learning",
    "subscription cancellation risk modeling",
    "transformer attention mechanisms explained",
    "BERT and GPT language model architectures",
    "pricing optimization using demand elasticity",
    "dynamic pricing strategies for SaaS",
    "knowledge graph construction from text",
    "entity and relation extraction with NLP",
    "RAG retrieval augmented generation pipelines",
    "vector databases for semantic search",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["cluster","search","outlier"], default="cluster")
    parser.add_argument("--query", default="machine learning optimization")
    args = parser.parse_args()

    try:
        from embeddings.encoder import TextEncoder
        encoder = TextEncoder()
        embs = encoder.encode(SAMPLE_TEXTS)
        print(f"Encoded {len(SAMPLE_TEXTS)} texts, dim={embs.shape[1]}")
    except ImportError:
        print("sentence-transformers not installed — using random embeddings for demo")
        embs = np.random.default_rng(42).standard_normal((len(SAMPLE_TEXTS), 64)).astype(np.float32)

    if args.task == "cluster":
        from analysis.clustering import EmbeddingClusterer
        clusterer = EmbeddingClusterer()
        result = clusterer.kmeans(embs, k_range=(2, 5))
        reps = clusterer.get_cluster_representatives(SAMPLE_TEXTS, embs, result.labels)
        print(f"\n{result.n_clusters} clusters (silhouette={result.silhouette:.3f}):")
        for cid, rep in reps.items():
            size = result.cluster_sizes[cid]
            print(f"  Cluster {cid} ({size} items): {rep[:60]}...")

    elif args.task == "search":
        from search.semantic_search import cosine_search, mmr_rerank
        try:
            from embeddings.encoder import TextEncoder
            q_emb = TextEncoder().encode_single(args.query)
        except ImportError:
            q_emb = np.random.default_rng(0).standard_normal(64).astype(np.float32)
        results = cosine_search(q_emb, embs, SAMPLE_TEXTS, top_k=5)
        print(f"\nTop results for: '{args.query}'")
        for r in results:
            print(f"  [{r.score:.3f}] {r.text}")

    elif args.task == "outlier":
        from analysis.outliers import EmbeddingOutlierDetector
        detector = EmbeddingOutlierDetector(contamination=0.1)
        result = detector.isolation_forest(embs)
        print(f"\nOutliers ({len(result.indices)} found):")
        for i in result.indices:
            print(f"  [{result.scores[i]:.3f}] {SAMPLE_TEXTS[i]}")

if __name__ == "__main__":
    main()
