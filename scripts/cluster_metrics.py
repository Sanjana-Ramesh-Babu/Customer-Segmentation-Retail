"""
Metrics for intra-cluster compactness and inter-cluster separation (K-Means context).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def total_sum_of_squares(X: np.ndarray) -> float:
    """Total SS around the global mean (for variance decomposition)."""
    mu = X.mean(axis=0)
    return float(np.sum((X - mu) ** 2))


def between_cluster_sum_of_squares(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """Between-cluster SS (spread of centroids weighted by cluster size)."""
    overall = X.mean(axis=0)
    bss = 0.0
    for k in range(centroids.shape[0]):
        mask = labels == k
        nk = int(mask.sum())
        if nk == 0:
            continue
        bss += nk * float(np.sum((centroids[k] - overall) ** 2))
    return bss


def cluster_metrics_bundle(
    X: np.ndarray,
    labels: np.ndarray,
    inertia: float,
    centroids: np.ndarray,
) -> dict:
    """
    Returns sklearn metrics plus BSS/TSS ratio (higher = more separation relative to total variance).
    """
    X = np.asarray(X)
    labels = np.asarray(labels)
    tss = total_sum_of_squares(X)
    bss = between_cluster_sum_of_squares(X, labels, centroids)
    wss = float(inertia)
    ratio_bss_tss = bss / tss if tss > 0 else np.nan

    sil = float(silhouette_score(X, labels, metric="euclidean"))
    db = float(davies_bouldin_score(X, labels))
    ch = float(calinski_harabasz_score(X, labels))

    # Mean pairwise centroid distance (higher → more separated cluster centers)
    k = centroids.shape[0]
    if k < 2:
        centroid_separation = 0.0
    else:
        d = []
        for i in range(k):
            for j in range(i + 1, k):
                d.append(float(np.linalg.norm(centroids[i] - centroids[j])))
        centroid_separation = float(np.mean(d)) if d else 0.0

    return {
        "silhouette_score": sil,
        "davies_bouldin": db,
        "calinski_harabasz": ch,
        "inertia_wcss": wss,
        "total_ss": tss,
        "between_cluster_ss": bss,
        "within_cluster_ss": wss,
        "between_total_variance_ratio": ratio_bss_tss,
        "mean_centroid_distance": centroid_separation,
    }


def metrics_summary_rows(metrics: dict) -> pd.DataFrame:
    """Human-readable table for UI / export."""
    rows = [
        ("Silhouette score (higher = better separation)", metrics["silhouette_score"]),
        ("Davies–Bouldin index (lower = better)", metrics["davies_bouldin"]),
        ("Calinski–Harabasz score (higher = better)", metrics["calinski_harabasz"]),
        ("K-Means inertia (within-cluster SS, PCA space)", metrics["inertia_wcss"]),
        ("Between-cluster SS / Total SS", metrics["between_total_variance_ratio"]),
        ("Mean distance between cluster centroids (PCA space)", metrics["mean_centroid_distance"]),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])
