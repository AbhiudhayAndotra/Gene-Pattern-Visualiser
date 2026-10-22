from __future__ import annotations

import numpy as np
import pandas as pd


def load_csv(file):
    try:
        df = pd.read_csv(file)

        return {
            "dataframe": df,
            "rows": df.shape[0],
            "columns": df.shape[1],
            "column_names": list(df.columns),
            "source": "uploaded_csv",
        }

    except Exception as e:
        return {"error": str(e)}


def generate_synthetic_gene_data(
    n_samples: int = 480,
    n_features: int = 96,
    n_clusters: int = 4,
    random_state: int = 42,
):
    if n_samples < 200 or n_samples > 500:
        raise ValueError("Synthetic dataset must have between 200 and 500 samples.")
    if n_features < 50 or n_features > 100:
        raise ValueError("Synthetic dataset must have between 50 and 100 gene features.")

    rng = np.random.default_rng(random_state)
    cluster_sizes = np.full(n_clusters, n_samples // n_clusters)
    cluster_sizes[: n_samples % n_clusters] += 1

    blocks = []
    base_means = np.linspace(-6, 6, n_clusters)

    for cluster_id, cluster_size in enumerate(cluster_sizes):
        mean_vector = rng.normal(loc=base_means[cluster_id], scale=0.8, size=n_features)
        block = rng.normal(loc=mean_vector, scale=1.1, size=(cluster_size, n_features))
        blocks.append(block)

    data = np.vstack(blocks)
    data = data[rng.permutation(n_samples)]
    columns = [f"Gene_{index + 1}" for index in range(n_features)]
    df = pd.DataFrame(data, columns=columns)

    return {
        "dataframe": df,
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": columns,
        "source": "synthetic_dataset",
    }
