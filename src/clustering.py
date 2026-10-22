from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans

from src.icd_clustering import fit_icd_clustering


def apply_kmeans_clustering(
    df: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int | None = 42,
):
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(df)

    return {
        "labels": labels.tolist(),
        "centroids": model.cluster_centers_.tolist(),
        "inertia": float(model.inertia_),
        "iterations": int(model.n_iter_),
    }


def compute_elbow_curve(
    df: pd.DataFrame,
    k_values: list[int],
    random_state: int | None = 42,
):
    if not k_values:
        return {
            "k_values": [],
            "inertias": [],
        }

    inertias = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        model.fit(df)
        inertias.append(float(model.inertia_))

    return {
        "k_values": k_values,
        "inertias": inertias,
    }


def apply_icd_clustering(
    df: pd.DataFrame,
    n_clusters: int = 4,
    max_iter: int = 20,
    random_state: int | None = 42,
):
    result = fit_icd_clustering(
        df=df,
        n_clusters=n_clusters,
        max_iter=max_iter,
        random_state=random_state,
    )

    return {
        "labels": result.labels.tolist(),
        "centroids": result.centroids.tolist(),
        "icd_per_cluster": {str(key): float(value) for key, value in result.icd_per_cluster.items()},
        "total_icd": float(result.total_icd),
        "iterations": result.iterations,
        "converged": result.converged,
    }
