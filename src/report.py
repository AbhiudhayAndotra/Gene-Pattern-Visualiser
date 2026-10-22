from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


OUTPUT_DIR = Path("generated_reports")
OUTPUT_DIR.mkdir(exist_ok=True)


def _build_pca_plot(pca_array: np.ndarray, labels: list[int], output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    unique_labels = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab10")

    for cluster_id in unique_labels:
        cluster_mask = np.array(labels) == cluster_id
        cluster_points = pca_array[cluster_mask]
        ax.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            label=f"Cluster {cluster_id}",
            alpha=0.8,
            s=34,
            color=cmap(cluster_id % 10),
        )

    ax.set_title("PCA Scatter Plot")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _build_heatmap_plot(corr_matrix: np.ndarray, columns: list[str], output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    heatmap = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_title("Correlation Heatmap")

    max_labels = min(12, len(columns))
    tick_positions = np.linspace(0, len(columns) - 1, max_labels, dtype=int)
    tick_labels = [columns[index] for index in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=7)

    colorbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Correlation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _build_cluster_distribution_plot(cluster_labels: list[str], counts: list[int], output_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(cluster_labels, counts, color="#0f766e")
    ax.set_title("Cluster Size Distribution")
    ax.set_ylabel("Samples")
    ax.grid(axis="y", alpha=0.2)
    ax.bar_label(bars, padding=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_pdf_report(df, analysis_payload: dict):
    report_id = analysis_payload["report_id"]
    pdf_path = OUTPUT_DIR / f"GenePatternX_Report_{report_id}.pdf"
    pca_plot = OUTPUT_DIR / f"pca_{report_id}.png"
    heatmap_plot = OUTPUT_DIR / f"heatmap_{report_id}.png"
    cluster_plot = OUTPUT_DIR / f"clusters_{report_id}.png"

    pca_array = np.asarray(analysis_payload["pca"]["points"], dtype=float)
    labels = [point["cluster"] for point in analysis_payload["pca"]["scatter"]]
    corr_matrix = np.asarray(analysis_payload["heatmap"]["matrix"], dtype=float)
    cluster_labels = analysis_payload["clusters"]["distribution"]["labels"]
    counts = analysis_payload["clusters"]["distribution"]["counts"]

    _build_pca_plot(pca_array, labels, pca_plot)
    _build_heatmap_plot(corr_matrix, analysis_payload["heatmap"]["columns"], heatmap_plot)
    _build_cluster_distribution_plot(cluster_labels, counts, cluster_plot)

    styles = getSampleStyleSheet()
    story = [
        Paragraph("GenePatternX Analysis Report", styles["Title"]),
        Spacer(1, 0.2 * inch),
        Paragraph(f"Dataset source: {analysis_payload['dataset']['source_label']}", styles["Normal"]),
        Paragraph(f"Rows: {df.shape[0]}", styles["Normal"]),
        Paragraph(f"Columns: {df.shape[1]}", styles["Normal"]),
        Spacer(1, 0.16 * inch),
        Paragraph("PCA Scatter Plot", styles["Heading2"]),
        Image(str(pca_plot), width=6.5 * inch, height=3.9 * inch),
        Spacer(1, 0.16 * inch),
        Paragraph("Correlation Heatmap", styles["Heading2"]),
        Image(str(heatmap_plot), width=6.3 * inch, height=4.4 * inch),
        Spacer(1, 0.16 * inch),
        Paragraph("Cluster Distribution Graph", styles["Heading2"]),
        Image(str(cluster_plot), width=6.0 * inch, height=3.8 * inch),
        Spacer(1, 0.16 * inch),
        Paragraph("ICD Metrics", styles["Heading2"]),
    ]

    icd_rows = [["Cluster", "ICD", "Size"]]
    for index, label in enumerate(cluster_labels):
        cluster_key = str(index)
        icd_rows.append(
            [
                label,
                f"{analysis_payload['clusters']['icd_per_cluster'][cluster_key]:.4f}",
                str(counts[index]),
            ]
        )
    icd_rows.append(["Total ICD", f"{analysis_payload['clusters']['total_icd']:.4f}", str(sum(counts))])

    icd_table = Table(icd_rows, colWidths=[2.5 * inch, 1.4 * inch, 1.1 * inch])
    icd_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#94a3b8")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ]
        )
    )
    story.append(icd_table)

    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=28)
    doc.build(story)

    return str(pdf_path)
