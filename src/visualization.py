from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd


def prepare_scatter_data(pca_data, labels):
    try:
        scatter = []

        for index, point in enumerate(pca_data):
            scatter.append(
                {
                    "x": point[0],
                    "y": point[1],
                    "cluster": int(labels[index]),
                    "sample_id": index,
                }
            )

        return {"scatter_data": scatter}

    except Exception as e:
        return {"error": str(e)}


def prepare_cluster_distribution(labels):
    counts = Counter(int(label) for label in labels)
    ordered = sorted(counts.items(), key=lambda item: item[0])
    return {
        "labels": [f"Cluster {cluster_id}" for cluster_id, _ in ordered],
        "counts": [count for _, count in ordered],
    }


def prepare_heatmap_data(df: pd.DataFrame):
    corr = df.corr(numeric_only=True).round(4)
    return {
        "columns": corr.columns.tolist(),
        "matrix": corr.values.tolist(),
        "min": float(corr.values.min()),
        "max": float(corr.values.max()),
    }


def prepare_distribution_data(
    df: pd.DataFrame,
    kmeans_labels: list[int],
    icd_labels: list[int],
    top_n_genes: int = 12,
    bins: int = 18,
):
    values = df.to_numpy(dtype=float).ravel()
    counts, edges = np.histogram(values, bins=bins)
    histogram_labels = [f"{edges[index]:.1f} to {edges[index + 1]:.1f}" for index in range(len(edges) - 1)]

    variances = df.var(axis=0).sort_values(ascending=False)
    top_variances = variances.head(top_n_genes)

    def summarize_clusters(labels: list[int]):
        summary = []
        unique_labels = sorted(set(int(label) for label in labels))
        for cluster_id in unique_labels:
            cluster_values = df.iloc[[index for index, label in enumerate(labels) if int(label) == cluster_id]].to_numpy(dtype=float).ravel()
            summary.append(
                {
                    "cluster": cluster_id,
                    "mean": round(float(cluster_values.mean()), 4),
                    "std": round(float(cluster_values.std()), 4),
                    "min": round(float(cluster_values.min()), 4),
                    "max": round(float(cluster_values.max()), 4),
                }
            )
        return summary

    return {
        "summary_stats": {
            "mean": round(float(values.mean()), 4),
            "median": round(float(np.median(values)), 4),
            "std": round(float(values.std()), 4),
            "min": round(float(values.min()), 4),
            "max": round(float(values.max()), 4),
        },
        "histogram": {
            "labels": histogram_labels,
            "counts": counts.tolist(),
        },
        "top_variable_genes": {
            "genes": top_variances.index.tolist(),
            "variances": [round(float(value), 4) for value in top_variances.tolist()],
        },
        "kmeans_cluster_summary": summarize_clusters(kmeans_labels),
        "icd_cluster_summary": summarize_clusters(icd_labels),
    }
