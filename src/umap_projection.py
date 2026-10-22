from __future__ import annotations

import numpy as np
import umap


def apply_umap(df, n_components=2, random_state=42, n_neighbors=15, min_dist=0.1):
    try:
        sample_count = len(df)
        if sample_count < 3:
            array = np.asarray(df, dtype=float)
            if array.shape[1] >= n_components:
                reduced = array[:, :n_components]
            else:
                padding = np.zeros((sample_count, n_components - array.shape[1]))
                reduced = np.hstack([array, padding])
            return {"umap_data": reduced.tolist()}

        adaptive_neighbors = min(n_neighbors, sample_count - 1)
        reducer = umap.UMAP(
            n_components=n_components,
            random_state=random_state,
            n_neighbors=adaptive_neighbors,
            min_dist=min_dist,
        )
        transformed = reducer.fit_transform(df)

        return {
            "umap_data": transformed.tolist(),
        }
    except Exception as e:
        return {"error": str(e)}
