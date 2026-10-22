# GenePatternX

**Advanced Gene Expression Visualization & Clustering Analysis**

GenePatternX is a modular bioinformatics dashboard built with FastAPI, Python, and modern browser visualizations. It supports gene expression uploads, automatic synthetic dataset fallback, ICD-optimized clustering, PCA/UMAP visualizations, and PDF report exports.

---

## 📚 Table of Contents

- [Why this project matters](#why-this-project-matters)
- [Quick start](#quick-start)
- [Project structure](#project-structure)
- [Main features](#main-features)
- [API endpoints](#api-endpoints)
- [Publish to GitHub](#publish-to-github)
- [Troubleshooting](#⚠️-troubleshooting)
- [Recommended workflow](#✅-recommended-workflow)
- [Notes](#📝-notes)
- [Support](#📞-support)

---

## ✨ Why this project matters

- 📤 Flexible data upload with CSV and synthetic fallback
- 🧠 Compare custom `ICD` clustering against `KMeans`
- 📉 PCA 2D/3D and UMAP embedding for visual discovery
- 📊 Interactive dashboard with charts, heatmaps, and summaries
- 📄 Export analysis results as a PDF report

---

## 🚀 Quick start

### 1. Open the project folder

```cmd
cd C:\Users\PRASHANT KUMAR\Desktop\Gene_viz_prototype
```

### 2. Create and activate the venv

#### PowerShell
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### CMD
```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```cmd
pip install -r requirements.txt
```

### 4. Run the app

```cmd
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

### 5. Open dashboard

```text
http://127.0.0.1:8000/dashboard
```

---

## 🧩 Project structure

```
Gene_viz_prototype/
├── app.py
├── requirements.txt
├── README.md
├── generated_reports/
├── old_prototype/
│   └── prototype.py
├── src/
│   ├── clustering.py
│   ├── data_loader.py
│   ├── icd_clustering.py
│   ├── pca.py
│   ├── report.py
│   ├── umap_projection.py
│   ├── validation.py
│   └── visualization.py
├── static/
│   ├── css/style.css
│   └── js/dashboard.js
└── templates/index.html
```

---

## 🔧 Main features

- **Data Input:** CSV upload or synthetic dataset fallback
- **Clustering:** ICD-optimized and KMeans baseline
- **Reduction:** PCA 2D/3D and UMAP visualization
- **Analytics:** heatmaps, elbow plots, gene ranking, cluster summaries
- **Reporting:** generate PDF reports automatically

---

## 📡 API endpoints

- `GET /dashboard` — open the UI
- `POST /upload` — submit CSV data for analysis
- `POST /download-report` — download the generated report

---

## 📦 Publish to GitHub

If this folder is not yet a Git repo, initialize local Git and commit the project.

```cmd
cd C:\Users\PRASHANT KUMAR\Desktop\Gene_viz_prototype
git init
git add .
git commit -m "Initial import of GenePatternX"
```

Create a repository on GitHub and connect it:

```cmd
git remote add origin https://github.com/<your-username>/Gene_viz_prototype.git
git branch -M main
git push -u origin main
```

Later updates:

```cmd
git add .
git commit -m "Update README and docs"
git push
```

---

## ⚠️ Troubleshooting

- If `8000` is busy, use: `uvicorn app:app --port 8001`
- In CMD, activate the venv with: `venv\Scripts\activate`
- If dependencies are missing: `pip install -r requirements.txt`
- If PDF creation fails, verify `ReportLab` is installed and `generated_reports/` is writable

---

## ✅ Recommended workflow

1. Start server
2. Open `/dashboard`
3. Upload CSV or use synthetic data
4. Run analysis
5. Download report and inspect visuals

---

## 📝 Notes

- `generated_reports/` is intended for runtime output
- Data should be CSV with genes as rows and samples as columns
- UMAP requires `umap-learn` in the venv
- Browser tested on Chrome, Firefox, Safari, and Edge

---

## 📞 Support

If you want to publish the project, create a GitHub repo and follow the commands above.
