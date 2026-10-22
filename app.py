from __future__ import annotations

from datetime import datetime

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from sklearn.preprocessing import StandardScaler

from src.clustering import apply_icd_clustering, apply_kmeans_clustering, compute_elbow_curve
from src.data_loader import generate_synthetic_gene_data, load_csv
from src.pca import apply_pca
from src.report import create_pdf_report
from src.umap_projection import apply_umap
from src.validation import validate_data
from src.visualization import (
    prepare_cluster_distribution,
    prepare_distribution_data,
    prepare_heatmap_data,
    prepare_scatter_data,
)

app = FastAPI(title="GenePatternX")
app.mount("/static", StaticFiles(directory="static"), name="static")


def _load_input_dataset(file: UploadFile | None):
    if file is not None and file.filename:
        load_result = load_csv(file.file)
    else:
        load_result = generate_synthetic_gene_data()

    if "error" in load_result:
        raise HTTPException(status_code=400, detail=load_result["error"])

    validation = validate_data(load_result["dataframe"])
    if "error" in validation:
        raise HTTPException(status_code=400, detail=validation["error"])

    clean_df = validation["dataframe"]
    source_label = "Uploaded CSV" if load_result["source"] == "uploaded_csv" else "Synthetic gene expression dataset"
    return clean_df, load_result["source"], source_label


def _select_cluster_count(num_samples: int) -> int:
    base = max(3, num_samples // 80)
    return min(max(2, base), max(2, num_samples - 1))


def _run_analysis(df, source: str, source_label: str):
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df)
    scaled_df = df.copy()
    scaled_df.loc[:, :] = scaled_array

    cluster_count = _select_cluster_count(len(df))
    pca_result = apply_pca(scaled_df, n_components=2)
    pca_3d_result = apply_pca(scaled_df, n_components=3)
    umap_result = apply_umap(scaled_df, n_components=2, random_state=42)
    kmeans_result = apply_kmeans_clustering(scaled_df, n_clusters=cluster_count, random_state=42)
    icd_result = apply_icd_clustering(scaled_df, n_clusters=cluster_count, max_iter=15, random_state=42)
    elbow_max_k = min(8, max(2, len(df) - 1))
    elbow_k_values = list(range(2, elbow_max_k + 1)) if elbow_max_k >= 2 else []
    elbow_curve = compute_elbow_curve(scaled_df, k_values=elbow_k_values, random_state=42)
    icd_scatter = prepare_scatter_data(pca_result["pca_data"], icd_result["labels"])
    kmeans_scatter = prepare_scatter_data(pca_result["pca_data"], kmeans_result["labels"])
    umap_scatter = prepare_scatter_data(umap_result["umap_data"], icd_result["labels"])
    icd_distribution = prepare_cluster_distribution(icd_result["labels"])
    kmeans_distribution = prepare_cluster_distribution(kmeans_result["labels"])
    heatmap = prepare_heatmap_data(df)
    distribution = prepare_distribution_data(df, kmeans_result["labels"], icd_result["labels"])

    top_features = list(df.columns[: min(6, len(df.columns))])
    icd_centroid_summary = []
    for cluster_index, centroid in enumerate(icd_result["centroids"]):
        icd_centroid_summary.append(
            {
                "cluster": cluster_index,
                "top_features": {
                    feature_name: round(float(centroid[df.columns.get_loc(feature_name)]), 4)
                    for feature_name in top_features
                },
            }
        )

    kmeans_centroid_summary = []
    for cluster_index, centroid in enumerate(kmeans_result["centroids"]):
        kmeans_centroid_summary.append(
            {
                "cluster": cluster_index,
                "top_features": {
                    feature_name: round(float(centroid[df.columns.get_loc(feature_name)]), 4)
                    for feature_name in top_features
                },
            }
        )

    icd_clusters = {
        "labels": icd_result["labels"],
        "centroids": icd_result["centroids"],
        "distribution": icd_distribution,
        "icd_per_cluster": icd_result["icd_per_cluster"],
        "total_icd": icd_result["total_icd"],
        "iterations": icd_result["iterations"],
        "converged": icd_result["converged"],
        "centroid_summary": icd_centroid_summary,
    }

    kmeans_clusters = {
        "labels": kmeans_result["labels"],
        "centroids": kmeans_result["centroids"],
        "distribution": kmeans_distribution,
        "iterations": kmeans_result["iterations"],
        "inertia": kmeans_result["inertia"],
        "centroid_summary": kmeans_centroid_summary,
    }

    return {
        "report_id": datetime.now().strftime("%Y%m%d%H%M%S"),
        "dataset": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "feature_names": df.columns.tolist(),
            "source": source,
            "source_label": source_label,
        },
        "pca": {
            "points": pca_result["pca_data"],
            "points_3d": pca_3d_result["pca_data"],
            "explained_variance": pca_result["explained_variance"],
            "explained_variance_3d": pca_3d_result["explained_variance"],
            "scatter": icd_scatter["scatter_data"],
            "icd_scatter": icd_scatter["scatter_data"],
            "kmeans_scatter": kmeans_scatter["scatter_data"],
        },
        "umap": {
            "points": umap_result["umap_data"],
            "scatter": umap_scatter["scatter_data"],
        },
        "clusters": icd_clusters,
        "icd_clusters": icd_clusters,
        "kmeans_clusters": kmeans_clusters,
        "elbow": elbow_curve,
        "distribution": distribution,
        "heatmap": heatmap,
    }


@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    with open("templates/index.html", "r", encoding="utf-8") as template:
        return HTMLResponse(content=template.read())


@app.post("/upload")
async def upload(file: UploadFile | None = File(default=None)):
    df, source, source_label = _load_input_dataset(file)
    return _run_analysis(df, source, source_label)


@app.post("/download-report")
async def download_report(file: UploadFile | None = File(default=None)):
    df, source, source_label = _load_input_dataset(file)
    analysis_payload = _run_analysis(df, source, source_label)
    pdf_path = create_pdf_report(df, analysis_payload)
    return FileResponse(pdf_path, filename="GenePatternX_Report.pdf", media_type="application/pdf")
