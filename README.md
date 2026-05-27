# Embeddings Explorer

[![CI](https://github.com/cerenaaa/embeddings-explorer/actions/workflows/ci.yml/badge.svg)](https://github.com/cerenaaa/embeddings-explorer/actions)

Toolkit for understanding and working with text embeddings: dimensionality reduction for visualization, semantic clustering, similarity search, and outlier detection.

## What's inside

| Module | Purpose |
|---|---|
| `embeddings/encoder.py` | Batch encoding with caching and normalization |
| `analysis/clustering.py` | K-Means + DBSCAN on embedding space with auto-labeling |
| `analysis/dimensionality.py` | PCA + UMAP reduction for 2D/3D visualization |
| `analysis/outliers.py` | Embedding-space anomaly detection |
| `search/semantic_search.py` | Cosine similarity search with MMR re-ranking |

## Quickstart
```bash
pip install -r requirements.txt
python explore.py --task cluster --input data/sample_texts.txt
python explore.py --task search --query "machine learning optimization"
```
