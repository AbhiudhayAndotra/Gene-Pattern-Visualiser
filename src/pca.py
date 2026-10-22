from sklearn.decomposition import PCA


def apply_pca(df, n_components=2):
    try:
        pca = PCA(n_components=n_components, random_state=42)
        transformed = pca.fit_transform(df)

        return {
            "pca_data": transformed.tolist(),
            "explained_variance": pca.explained_variance_ratio_.tolist(),
            "components": pca.components_.tolist(),
        }

    except Exception as e:
        return {"error": str(e)}
