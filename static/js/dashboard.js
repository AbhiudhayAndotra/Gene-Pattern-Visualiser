let analysisData = null;
let kmeansScatterChart = null;
let icdScatterChart = null;
let kmeansClusterChart = null;
let icdClusterChart = null;
let elbowChart = null;
let umapScatterChart = null;
let distributionHistogramChart = null;
let varianceChart = null;

const clusterPalette = [
    "#0f766e",
    "#f97316",
    "#2563eb",
    "#dc2626",
    "#7c3aed",
    "#0891b2",
    "#65a30d",
    "#c2410c",
];

document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll(".nav-link").forEach((button) => {
        button.addEventListener("click", () => activateTab(button.dataset.tab));
    });

    document.getElementById("analyzeBtn").addEventListener("click", runAnalysis);
    document.getElementById("reportBtn").addEventListener("click", downloadReport);
    document.getElementById("reportTabBtn").addEventListener("click", downloadReport);
});

function activateTab(tabId) {
    document.querySelectorAll(".nav-link").forEach((button) => {
        button.classList.toggle("active", button.dataset.tab === tabId);
    });

    document.querySelectorAll(".tab-panel").forEach((panel) => {
        panel.classList.toggle("active", panel.id === tabId);
    });
}

function buildFormData() {
    const formData = new FormData();
    const fileInput = document.getElementById("fileInput");
    if (fileInput.files.length > 0) {
        formData.append("file", fileInput.files[0]);
    }
    return formData;
}

async function runAnalysis() {
    updateStatus("Processing dataset and running ICD optimization...");
    try {
        const response = await fetch("/upload", {
            method: "POST",
            body: buildFormData(),
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || "Analysis request failed.");
        }

        analysisData = data;
        updateOverview(data);
        drawPCAComparisons(data);
        drawClusterComparisons(data);
        drawHeatmap(data);
        drawDistribution(data);
        renderICDMetrics(data);
        renderKMeansMetrics(data);
        renderCentroidSummary("kmeansCentroidSummary", data.kmeans_clusters.centroid_summary);
        renderCentroidSummary("icdCentroidSummary", data.icd_clusters.centroid_summary);
        renderDistributionSummary(data);
        renderReportSummary(data);
        activateTab("pca");

        const uploadMode = document.getElementById("uploadMode");
        uploadMode.textContent = data.dataset.source === "uploaded_csv"
            ? "Current mode: uploaded CSV analyzed successfully."
            : "Current mode: synthetic dataset generated and analyzed successfully.";
        updateStatus("Analysis complete. Visualizations and report data are ready.");
    } catch (error) {
        updateStatus(error.message);
    }
}

function updateOverview(data) {
    document.getElementById("rowsValue").textContent = data.dataset.rows;
    document.getElementById("columnsValue").textContent = data.dataset.columns;
    document.getElementById("icdValue").textContent = Number(data.icd_clusters.total_icd).toFixed(3);
    document.getElementById("sourceValue").textContent = data.dataset.source_label;
    document.getElementById("iterationsValue").textContent = `${data.kmeans_clusters.iterations} / ${data.icd_clusters.iterations}`;
    document.getElementById("convergedValue").textContent = data.icd_clusters.converged ? "ICD Yes" : "ICD No";

    const variance = data.pca.explained_variance
        .map((value, index) => `PC${index + 1}: ${(value * 100).toFixed(1)}%`)
        .join(" | ");
    document.getElementById("varianceValue").textContent = variance;
}

function drawPCAComparisons(data) {
    kmeansScatterChart = drawScatterChart(
        "kmeansScatterChart",
        data.pca.kmeans_scatter,
        kmeansScatterChart
    );
    icdScatterChart = drawScatterChart(
        "icdScatterChart",
        data.pca.icd_scatter,
        icdScatterChart
    );
    drawPCA3DPlot(data);
}

function drawScatterChart(canvasId, scatterPoints, chartInstance) {
    const grouped = new Map();
    scatterPoints.forEach((point) => {
        const key = point.cluster;
        if (!grouped.has(key)) {
            grouped.set(key, []);
        }
        grouped.get(key).push({ x: point.x, y: point.y });
    });

    const datasets = Array.from(grouped.entries()).map(([clusterId, points], index) => ({
        label: `Cluster ${clusterId}`,
        data: points,
        backgroundColor: clusterPalette[index % clusterPalette.length],
        pointRadius: 5,
        pointHoverRadius: 6,
    }));

    if (chartInstance) {
        chartInstance.destroy();
    }

    return new Chart(document.getElementById(canvasId), {
        type: "scatter",
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: "bottom" },
            },
            scales: {
                x: { title: { display: true, text: "Principal Component 1" } },
                y: { title: { display: true, text: "Principal Component 2" } },
            },
        },
    });
}

function drawClusterComparisons(data) {
    kmeansClusterChart = drawClusterChart("kmeansClusterChart", data.kmeans_clusters.distribution, kmeansClusterChart, false);
    icdClusterChart = drawClusterChart("icdClusterChart", data.icd_clusters.distribution, icdClusterChart, true);
    elbowChart = drawElbowChart(data.elbow, elbowChart);
    umapScatterChart = drawScatterChart("umapScatterChart", data.umap.scatter, umapScatterChart);
}

function drawPCA3DPlot(data) {
    if (!window.Plotly || !data.pca.points_3d) {
        return;
    }

    const grouped = new Map();
    data.pca.icd_scatter.forEach((point, index) => {
        const clusterId = point.cluster;
        if (!grouped.has(clusterId)) {
            grouped.set(clusterId, []);
        }
        grouped.get(clusterId).push(data.pca.points_3d[index]);
    });

    const traces = Array.from(grouped.entries()).map(([clusterId, points], index) => ({
        type: "scatter3d",
        mode: "markers",
        name: `Cluster ${clusterId}`,
        x: points.map((point) => point[0]),
        y: points.map((point) => point[1]),
        z: points.map((point) => point[2]),
        marker: {
            size: 4,
            color: clusterPalette[index % clusterPalette.length],
            opacity: 0.85,
        },
    }));

    Plotly.react("pca3dPlot", traces, {
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        margin: { l: 0, r: 0, t: 10, b: 0 },
        scene: {
            bgcolor: "rgba(0,0,0,0)",
            xaxis: { title: "PC1", color: "#cbd5e1", gridcolor: "rgba(148,163,184,0.18)" },
            yaxis: { title: "PC2", color: "#cbd5e1", gridcolor: "rgba(148,163,184,0.18)" },
            zaxis: { title: "PC3", color: "#cbd5e1", gridcolor: "rgba(148,163,184,0.18)" },
        },
        legend: { orientation: "h", y: -0.08, font: { color: "#cbd5e1" } },
    }, { responsive: true, displayModeBar: false });
}

function drawElbowChart(elbowData, chartInstance) {
    if (!elbowData || !elbowData.k_values || !elbowData.k_values.length) {
        return chartInstance;
    }

    if (chartInstance) {
        chartInstance.destroy();
    }

    return new Chart(document.getElementById("elbowChart"), {
        type: "line",
        data: {
            labels: elbowData.k_values,
            datasets: [{
                label: "Inertia",
                data: elbowData.inertias,
                borderColor: "#f97316",
                backgroundColor: "rgba(249, 115, 22, 0.18)",
                pointBackgroundColor: "#fb923c",
                pointRadius: 4,
                tension: 0.25,
                fill: false,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
            },
            scales: {
                x: {
                    title: { display: true, text: "Number of Clusters (k)" },
                    ticks: { color: "#cbd5e1" },
                    grid: { display: false },
                },
                y: {
                    title: { display: true, text: "Inertia" },
                    ticks: { color: "#cbd5e1" },
                    grid: { color: "rgba(148, 163, 184, 0.14)" },
                },
            },
        },
    });
}

function drawClusterChart(canvasId, distribution, chartInstance, emphasizeAxes = false) {
    if (chartInstance) {
        chartInstance.destroy();
    }

    return new Chart(document.getElementById(canvasId), {
        type: "bar",
        data: {
            labels: distribution.labels,
            datasets: [{
                label: "Samples",
                data: distribution.counts,
                backgroundColor: distribution.labels.map((_, index) => clusterPalette[index % clusterPalette.length]),
                borderRadius: 8,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            layout: {
                padding: emphasizeAxes ? { left: 12, right: 12, top: 8, bottom: 8 } : { left: 6, right: 6, top: 6, bottom: 4 },
            },
            plugins: {
                legend: { display: false },
            },
            scales: {
                x: {
                    ticks: {
                        display: true,
                        color: "#cbd5e1",
                        padding: 8,
                    },
                    grid: {
                        display: false,
                    },
                    border: {
                        display: true,
                        color: "rgba(148, 163, 184, 0.45)",
                    },
                },
                y: {
                    beginAtZero: true,
                    title: { display: true, text: "Samples" },
                    ticks: {
                        color: "#cbd5e1",
                        padding: 8,
                    },
                    grid: {
                        color: "rgba(148, 163, 184, 0.14)",
                    },
                    border: {
                        display: true,
                        color: "rgba(148, 163, 184, 0.45)",
                    },
                },
            },
        },
    });
}

function drawHeatmap(data) {
    const canvas = document.getElementById("heatmapCanvas");
    const ctx = canvas.getContext("2d");
    const matrix = data.heatmap.matrix;
    const columns = data.heatmap.columns;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!matrix.length) {
        return;
    }

    const paddingLeft = 120;
    const paddingTop = 100;
    const gridWidth = canvas.width - paddingLeft - 30;
    const gridHeight = canvas.height - paddingTop - 40;
    const cellSize = Math.min(gridWidth / columns.length, gridHeight / matrix.length);

    ctx.font = "12px 'Segoe UI', sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    matrix.forEach((row, rowIndex) => {
        row.forEach((value, colIndex) => {
            ctx.fillStyle = interpolateHeatmapColor(value);
            ctx.fillRect(
                paddingLeft + colIndex * cellSize,
                paddingTop + rowIndex * cellSize,
                cellSize,
                cellSize
            );
        });
    });

    ctx.strokeStyle = "rgba(15, 23, 42, 0.08)";
    for (let index = 0; index <= columns.length; index += 1) {
        const x = paddingLeft + index * cellSize;
        ctx.beginPath();
        ctx.moveTo(x, paddingTop);
        ctx.lineTo(x, paddingTop + matrix.length * cellSize);
        ctx.stroke();
    }

    for (let index = 0; index <= matrix.length; index += 1) {
        const y = paddingTop + index * cellSize;
        ctx.beginPath();
        ctx.moveTo(paddingLeft, y);
        ctx.lineTo(paddingLeft + columns.length * cellSize, y);
        ctx.stroke();
    }

    const labelStep = Math.max(1, Math.ceil(columns.length / 12));
    ctx.fillStyle = "#e2e8f0";
    columns.forEach((column, index) => {
        if (index % labelStep !== 0) {
            return;
        }

        const x = paddingLeft + index * cellSize + cellSize / 2;
        const y = paddingTop - 12;
        ctx.save();
        ctx.translate(x, y);
        ctx.rotate(-Math.PI / 4);
        ctx.fillText(column, 0, 0);
        ctx.restore();

        ctx.textAlign = "right";
        ctx.fillText(column, paddingLeft - 8, paddingTop + index * cellSize + cellSize / 2);
        ctx.textAlign = "center";
    });

    drawHeatmapLegend(ctx, canvas.width - 30, paddingTop, matrix.length * cellSize);
}

function drawDistribution(data) {
    if (!data.distribution || !data.distribution.histogram || !data.distribution.top_variable_genes) {
        clearDistributionViews("Distribution data is unavailable for this analysis run.");
        return;
    }

    distributionHistogramChart = drawDistributionHistogram(
        data.distribution.histogram,
        distributionHistogramChart
    );
    varianceChart = drawVarianceChart(
        data.distribution.top_variable_genes,
        varianceChart
    );
}

function clearDistributionViews(message) {
    if (distributionHistogramChart) {
        distributionHistogramChart.destroy();
        distributionHistogramChart = null;
    }
    if (varianceChart) {
        varianceChart.destroy();
        varianceChart = null;
    }

    const summaryRoot = document.getElementById("distributionSummary");
    const kmeansRoot = document.getElementById("kmeansDistributionSummary");
    const icdRoot = document.getElementById("icdDistributionSummary");

    if (summaryRoot) {
        summaryRoot.innerHTML = `<div><span class="metric-label">Status</span><strong>${message}</strong></div>`;
    }
    if (kmeansRoot) {
        kmeansRoot.innerHTML = "";
    }
    if (icdRoot) {
        icdRoot.innerHTML = "";
    }
}

function drawDistributionHistogram(histogram, chartInstance) {
    if (chartInstance) {
        chartInstance.destroy();
    }

    return new Chart(document.getElementById("distributionHistogramChart"), {
        type: "bar",
        data: {
            labels: histogram.labels,
            datasets: [{
                label: "Frequency",
                data: histogram.counts,
                backgroundColor: "rgba(34, 197, 94, 0.75)",
                borderRadius: 6,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
            },
            scales: {
                x: {
                    ticks: {
                        color: "#cbd5e1",
                        maxRotation: 60,
                        minRotation: 60,
                        autoSkip: true,
                    },
                    grid: { display: false },
                },
                y: {
                    beginAtZero: true,
                    ticks: { color: "#cbd5e1" },
                    title: { display: true, text: "Frequency" },
                    grid: { color: "rgba(148, 163, 184, 0.14)" },
                },
            },
        },
    });
}

function drawVarianceChart(varianceData, chartInstance) {
    if (chartInstance) {
        chartInstance.destroy();
    }

    return new Chart(document.getElementById("varianceChart"), {
        type: "bar",
        data: {
            labels: varianceData.genes,
            datasets: [{
                label: "Variance",
                data: varianceData.variances,
                backgroundColor: varianceData.genes.map((_, index) => clusterPalette[index % clusterPalette.length]),
                borderRadius: 6,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
            },
            scales: {
                x: {
                    ticks: {
                        color: "#cbd5e1",
                        maxRotation: 55,
                        minRotation: 55,
                    },
                    grid: { display: false },
                },
                y: {
                    beginAtZero: true,
                    ticks: { color: "#cbd5e1" },
                    title: { display: true, text: "Variance" },
                    grid: { color: "rgba(148, 163, 184, 0.14)" },
                },
            },
        },
    });
}

function interpolateHeatmapColor(value) {
    const normalized = (value + 1) / 2;
    const start = { r: 30, g: 64, b: 175 };
    const middle = { r: 248, g: 250, b: 252 };
    const end = { r: 185, g: 28, b: 28 };

    if (normalized <= 0.5) {
        const ratio = normalized / 0.5;
        return mixColor(start, middle, ratio);
    }

    const ratio = (normalized - 0.5) / 0.5;
    return mixColor(middle, end, ratio);
}

function mixColor(first, second, ratio) {
    const r = Math.round(first.r + (second.r - first.r) * ratio);
    const g = Math.round(first.g + (second.g - first.g) * ratio);
    const b = Math.round(first.b + (second.b - first.b) * ratio);
    return `rgb(${r}, ${g}, ${b})`;
}

function drawHeatmapLegend(ctx, xStart, yStart, height) {
    const legendWidth = 18;
    const gradient = ctx.createLinearGradient(0, yStart, 0, yStart + height);
    gradient.addColorStop(0, "rgb(185, 28, 28)");
    gradient.addColorStop(0.5, "rgb(248, 250, 252)");
    gradient.addColorStop(1, "rgb(30, 64, 175)");

    ctx.fillStyle = gradient;
    ctx.fillRect(xStart, yStart, legendWidth, height);

    ctx.fillStyle = "#e2e8f0";
    ctx.textAlign = "left";
    ctx.fillText("1.0", xStart + 24, yStart + 4);
    ctx.fillText("0.0", xStart + 24, yStart + height / 2);
    ctx.fillText("-1.0", xStart + 24, yStart + height - 4);
}

function renderICDMetrics(data) {
    const metricsRoot = document.getElementById("icdMetrics");
    metricsRoot.innerHTML = "";

    Object.entries(data.icd_clusters.icd_per_cluster).forEach(([clusterId, value], index) => {
        const card = document.createElement("div");
        card.className = "metric-pill";
        card.innerHTML = `
            <span class="swatch" style="background:${clusterPalette[index % clusterPalette.length]}"></span>
            <div>
                <strong>Cluster ${clusterId}</strong>
                <p>ICD ${Number(value).toFixed(4)}</p>
            </div>
        `;
        metricsRoot.appendChild(card);
    });

    const total = document.createElement("div");
    total.className = "metric-pill total-pill";
    total.innerHTML = `
        <div>
            <strong>Total ICD</strong>
            <p>${Number(data.icd_clusters.total_icd).toFixed(4)}</p>
        </div>
    `;
    metricsRoot.appendChild(total);
}

function renderKMeansMetrics(data) {
    const metricsRoot = document.getElementById("kmeansMetrics");
    metricsRoot.innerHTML = `
        <div class="metric-pill total-pill">
            <div>
                <strong>KMeans Inertia</strong>
                <p>${Number(data.kmeans_clusters.inertia).toFixed(4)}</p>
            </div>
        </div>
        <div class="metric-pill">
            <div>
                <strong>KMeans Iterations</strong>
                <p>${data.kmeans_clusters.iterations}</p>
            </div>
        </div>
        <div class="metric-pill">
            <div>
                <strong>Cluster Count</strong>
                <p>${data.kmeans_clusters.distribution.labels.length}</p>
            </div>
        </div>
    `;
}

function renderDistributionSummary(data) {
    if (!data.distribution || !data.distribution.summary_stats) {
        clearDistributionViews("Distribution data is unavailable for this analysis run.");
        return;
    }

    const summaryRoot = document.getElementById("distributionSummary");
    const stats = data.distribution.summary_stats;
    summaryRoot.innerHTML = `
        <div><span class="metric-label">Mean</span><strong>${stats.mean}</strong></div>
        <div><span class="metric-label">Median</span><strong>${stats.median}</strong></div>
        <div><span class="metric-label">Std Dev</span><strong>${stats.std}</strong></div>
        <div><span class="metric-label">Minimum</span><strong>${stats.min}</strong></div>
        <div><span class="metric-label">Maximum</span><strong>${stats.max}</strong></div>
    `;

    renderClusterDistributionSummary("kmeansDistributionSummary", data.distribution.kmeans_cluster_summary || []);
    renderClusterDistributionSummary("icdDistributionSummary", data.distribution.icd_cluster_summary || []);
}

function renderClusterDistributionSummary(rootId, summaryEntries) {
    const summaryRoot = document.getElementById(rootId);
    summaryRoot.innerHTML = "";

    summaryEntries.forEach((entry, index) => {
        const card = document.createElement("div");
        card.className = "centroid-entry";
        card.innerHTML = `
            <div class="centroid-head">
                <span class="swatch" style="background:${clusterPalette[index % clusterPalette.length]}"></span>
                <strong>Cluster ${entry.cluster}</strong>
            </div>
            <div class="centroid-values">
                <span>Mean: ${entry.mean}</span>
                <span>Std: ${entry.std}</span>
                <span>Min: ${entry.min}</span>
                <span>Max: ${entry.max}</span>
            </div>
        `;
        summaryRoot.appendChild(card);
    });
}

function renderCentroidSummary(rootId, centroidSummary) {
    const summaryRoot = document.getElementById(rootId);
    summaryRoot.innerHTML = "";

    centroidSummary.forEach((entry, index) => {
        const card = document.createElement("div");
        card.className = "centroid-entry";

        const values = Object.entries(entry.top_features)
            .map(([feature, value]) => `<span>${feature}: ${value}</span>`)
            .join("");

        card.innerHTML = `
            <div class="centroid-head">
                <span class="swatch" style="background:${clusterPalette[index % clusterPalette.length]}"></span>
                <strong>Cluster ${entry.cluster}</strong>
            </div>
            <div class="centroid-values">${values}</div>
        `;
        summaryRoot.appendChild(card);
    });
}

function renderReportSummary(data) {
    const summary = document.getElementById("reportSummary");
    summary.innerHTML = `
        <p><strong>Dataset:</strong> ${data.dataset.source_label}</p>
        <p><strong>Dimensions:</strong> ${data.dataset.rows} rows x ${data.dataset.columns} columns</p>
        <p><strong>PCA:</strong> ${(data.pca.explained_variance[0] * 100).toFixed(1)}% + ${(data.pca.explained_variance[1] * 100).toFixed(1)}% explained variance</p>
        <p><strong>KMeans:</strong> inertia ${Number(data.kmeans_clusters.inertia).toFixed(4)} across ${data.kmeans_clusters.distribution.labels.length} clusters</p>
        <p><strong>ICD:</strong> ${Number(data.icd_clusters.total_icd).toFixed(4)} total across ${data.icd_clusters.distribution.labels.length} clusters</p>
    `;
}

async function downloadReport() {
    updateStatus("Preparing PDF report...");
    try {
        const response = await fetch("/download-report", {
            method: "POST",
            body: buildFormData(),
        });

        if (!response.ok) {
            let errorMessage = "Report generation failed.";
            try {
                const errorPayload = await response.json();
                errorMessage = errorPayload.detail || errorMessage;
            } catch (_ignored) {
            }
            throw new Error(errorMessage);
        }

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = "GenePatternX_Report.pdf";
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
        updateStatus("PDF report downloaded.");
    } catch (error) {
        updateStatus(error.message);
    }
}

function updateStatus(message) {
    document.getElementById("statusText").textContent = message;
}
