from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ICDClusteringResult:
    labels: np.ndarray
    centroids: np.ndarray
    icd_per_cluster: dict[int, float]
    total_icd: float
    iterations: int
    converged: bool


def _average_pairwise_distance(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0

    diffs = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(diffs, axis=2)
    upper = distances[np.triu_indices(len(points), k=1)]
    return float(upper.mean()) if upper.size else 0.0


def _total_icd_score(data: np.ndarray, labels: np.ndarray, n_clusters: int) -> tuple[dict[int, float], float]:
    icd_per_cluster: dict[int, float] = {}
    total = 0.0

    for cluster_id in range(n_clusters):
        members = data[labels == cluster_id]
        cluster_icd = _average_pairwise_distance(members)
        icd_per_cluster[cluster_id] = cluster_icd
        total += cluster_icd

    return icd_per_cluster, float(total)


def _recompute_centroids(data: np.ndarray, labels: np.ndarray, centroids: np.ndarray, n_clusters: int) -> np.ndarray:
    updated = centroids.copy()
    for cluster_id in range(n_clusters):
        members = data[labels == cluster_id]
        if len(members):
            updated[cluster_id] = members.mean(axis=0)
    return updated


def _initialize_labels(data: np.ndarray, n_clusters: int, random_state: int | None) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    seed_indices = rng.choice(len(data), size=n_clusters, replace=False)
    centroids = data[seed_indices].copy()

    distances = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
    labels = distances.argmin(axis=1)

    for cluster_id in range(n_clusters):
        if np.any(labels == cluster_id):
            continue
        farthest_idx = np.argmax(distances.min(axis=1))
        labels[farthest_idx] = cluster_id
        centroids[cluster_id] = data[farthest_idx]

    centroids = _recompute_centroids(data, labels, centroids, n_clusters)
    return labels, centroids


def fit_icd_clustering(
    df: pd.DataFrame,
    n_clusters: int = 4,
    max_iter: int = 20,
    tol: float = 1e-4,
    random_state: int | None = 42,
) -> ICDClusteringResult:
    data = np.asarray(df, dtype=float)
    if data.ndim != 2 or len(data) == 0:
        raise ValueError("Input data must be a non-empty 2D matrix.")
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2.")
    if len(data) < n_clusters:
        raise ValueError("n_clusters cannot exceed the number of samples.")

    labels, centroids = _initialize_labels(data, n_clusters, random_state)
    _, previous_total_icd = _total_icd_score(data, labels, n_clusters)
    converged = False

    for iteration in range(1, max_iter + 1):
        changed = False

        for sample_idx in range(len(data)):
            current_cluster = int(labels[sample_idx])
            current_members = np.where(labels == current_cluster)[0]
            if len(current_members) <= 1:
                continue

            best_cluster = current_cluster
            best_score = previous_total_icd

            for candidate_cluster in range(n_clusters):
                trial_labels = labels.copy()
                trial_labels[sample_idx] = candidate_cluster
                _, trial_total_icd = _total_icd_score(data, trial_labels, n_clusters)
                if trial_total_icd + tol < best_score:
                    best_score = trial_total_icd
                    best_cluster = candidate_cluster

            if best_cluster != current_cluster:
                labels[sample_idx] = best_cluster
                previous_total_icd = best_score
                changed = True

        centroids = _recompute_centroids(data, labels, centroids, n_clusters)
        icd_per_cluster, total_icd = _total_icd_score(data, labels, n_clusters)

        if not changed or abs(previous_total_icd - total_icd) <= tol:
            converged = True
            return ICDClusteringResult(
                labels=labels,
                centroids=centroids,
                icd_per_cluster=icd_per_cluster,
                total_icd=total_icd,
                iterations=iteration,
                converged=converged,
            )

        previous_total_icd = total_icd

    icd_per_cluster, total_icd = _total_icd_score(data, labels, n_clusters)
    return ICDClusteringResult(
        labels=labels,
        centroids=centroids,
        icd_per_cluster=icd_per_cluster,
        total_icd=total_icd,
        iterations=max_iter,
        converged=converged,
    )
