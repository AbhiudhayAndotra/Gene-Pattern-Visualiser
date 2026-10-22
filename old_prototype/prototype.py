import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ==========================================
# 1. GENERATE SYNTHETIC GENE DATA
# ==========================================
print("🧬 Generating synthetic gene expression data...")
np.random.seed(42)

# Let's create 100 Genes (rows) across 5 Samples (columns)
n_genes = 100
n_samples = 5

# Create 3 distinct clusters of genes with different expression patterns
# Cluster 0: Low expression across all samples
cluster0 = np.random.randn(30, n_samples) * 0.5 + 1.0 
# Cluster 1: High expression in first 2 samples, low in others
cluster1 = np.random.randn(30, n_samples) * 0.5 + np.array([4, 4, 1, 1, 1])
# Cluster 2: High expression in last 3 samples, low in others
cluster2 = np.random.randn(40, n_samples) * 0.5 + np.array([1, 1, 4, 4, 4])

# Combine them
data_raw = np.vstack([cluster0, cluster1, cluster2])

# Create a DataFrame
gene_names = [f"Gene_{i}" for i in range(1, n_genes + 1)]
sample_names = [f"Sample_{i}" for i in range(1, n_samples + 1)]
df = pd.DataFrame(data_raw, columns=sample_names, index=gene_names)

# ==========================================
# 2. PREPROCESSING
# ==========================================
print("🧼 Preprocessing data (Log Transform + Standardization)...")

# Gene expression data is often skewed, so log-transform is standard practice
# We add 1 to avoid log(0) errors
df_log = np.log1p(df)

# Standardize so every gene has mean=0, variance=1 (Crucial for KMeans)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df_log)

# ==========================================
# 3. CLUSTERING (K-MEANS)
# ==========================================
print("🤖 Running K-Means Clustering...")

# We know we created 3 clusters, so let's set k=3 for the prototype
# In the full version, we will calculate optimal k automatically
k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(data_scaled)

# Add cluster labels to dataframe
df['Cluster'] = clusters

# Evaluate Quality
silhouette = silhouette_score(data_scaled, clusters)
print(f"✅ Clustering Complete. Silhouette Score: {silhouette:.3f}")

# ==========================================
# 4. DIMENSIONALITY REDUCTION (PCA)
# ==========================================
print("📉 Reducing dimensions with PCA...")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data_scaled)

df['PC1'] = pca_result[:, 0]
df['PC2'] = pca_result[:, 1]

# ==========================================
# 5. IMPROVED VISUALIZATION
# ==========================================
print("🎨 Generating Improved Visualizations...")

sns.set_theme(style="whitegrid")

# Plot 1: PCA Scatter Plot (Improved)
plt.figure(figsize=(10, 8))
scatter = sns.scatterplot(
    data=df, x='PC1', y='PC2', hue='Cluster', 
    palette='viridis', s=150, edgecolor='black', linewidth=0.5, alpha=0.9
)
plt.title(f'Gene Clusters in PCA Space\nSilhouette Score: {silhouette:.3f}', 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
           fontsize=12)
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
           fontsize=12)
plt.legend(title='Cluster', title_fontsize=10, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pca_clusters.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: Heatmap - Show ONLY representative genes (Much clearer!)
plt.figure(figsize=(10, 8))

# Select top 10 genes from each cluster (by distance to centroid)
representative_genes = []
for cluster_id in sorted(df['Cluster'].unique()):
    cluster_genes = df[df['Cluster'] == cluster_id]
    # Get the 10 genes closest to the cluster center
    cluster_indices = cluster_genes.index
    distances = np.linalg.norm(data_scaled[[df.index.get_loc(g) for g in cluster_indices]] - 
                               kmeans.cluster_centers_[cluster_id], axis=1)
    top_10 = cluster_genes.index[np.argsort(distances)[:10]]
    representative_genes.extend(top_10)

df_rep = df.loc[representative_genes].copy()
df_rep = df_rep.sort_values('Cluster')
heatmap_data = df_rep[sample_names]

# Create better heatmap
ax = sns.heatmap(
    heatmap_data, 
    cmap='RdYlBu_r',
    yticklabels=[f"{idx}" for idx in heatmap_data.index],
    xticklabels=True,
    cbar_kws={'label': 'Expression Level (Log)', 'shrink': 0.8},
    linewidths=0.5,
    linecolor='lightgray'
)

# Add cluster separation lines
cluster_boundaries = df_rep['Cluster'].value_counts().sort_index().cumsum()
for boundary in cluster_boundaries[:-1]:
    ax.axhline(y=boundary, color='black', linestyle='-', linewidth=2)

plt.title('Gene Expression Heatmap\n(10 Representative Genes per Cluster)', 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Samples', fontsize=12)
plt.ylabel('Genes', fontsize=12)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('expression_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 3: Expression Profiles for Representative Genes (NEW!)
plt.figure(figsize=(12, 6))
for cluster_id in sorted(df['Cluster'].unique()):
    cluster_genes = df[df['Cluster'] == cluster_id]
    # Plot mean expression profile for each cluster
    mean_profile = cluster_genes[sample_names].mean()
    plt.plot(range(1, n_samples + 1), mean_profile, 'o-', 
             linewidth=3, markersize=10, label=f'Cluster {cluster_id}')

plt.title('Average Expression Profile per Cluster', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Samples', fontsize=12)
plt.ylabel('Mean Expression Level (Log)', fontsize=12)
plt.legend(title='Cluster', title_fontsize=10, fontsize=10)
plt.xticks(range(1, n_samples + 1), sample_names)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('expression_profiles.png', dpi=300, bbox_inches='tight')
plt.show()

print("🎉 Prototype Complete!")
print("📁 Generated files:")
print("   - pca_clusters.png (PCA scatter plot)")
print("   - expression_heatmap.png (Improved heatmap with representative genes)")
print("   - expression_profiles.png (NEW: Expression trends per cluster)")