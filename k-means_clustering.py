import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.datasets import make_classification
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
from scipy.spatial import Voronoi, voronoi_plot_2d
# Removed unused import
from matplotlib import cm

#Clustering evaluation function
def evaluate_clustering(data, labels, n_clusters, ax=None, title_suffix=''):
    """
    Evaluate a clustering model using silhouette scores and the Davies-Bouldin index.
    
    Parameters:
    data (ndarray): Feature matrix.
    labels (array-like): Cluster labels assigned to each sample.
    n_clusters (int): The number of clusters in the model.
    ax: The subplot axes to plot on.
    title_suffix (str): Optional suffix for plot title
    
    Returns:
    None: Displays silhoutte scores and a silhouette plot.
    """
    if ax is None:
        ax = plt.gca()  # Get the current axis if none is provided
    
    # Calculate silhouette scores
    silhouette_avg = silhouette_score(data, labels, random_state=42)
    sample_silhouette_values = silhouette_samples(data, labels)

    # Plot silhouette analysis on the provided axis
    unique_labels = np.unique(labels)
    colormap = cm.tab10
    label_mapping = {label: numeric_label for numeric_label, label in enumerate(unique_labels)}
    color_dict = {label: colormap(float(label_mapping[label]) / n_clusters) for label in unique_labels}
    y_lower = 10
    for i in unique_labels:
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = color_dict[i]
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    
    ax.set_title(f'Silhouette Score for {title_suffix} \n' + 
                 f'Average Silhouette: {silhouette_avg:.2f}')
    ax.set_xlabel('Silhouette Coefficient')
    ax.set_xlim([-1, 1])  # Set the x-axis range to [-1, 1]
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_xlim([-0.25, 1])  # Set the x-axis range to [0, 1]

    ax.set_yticks([])
#Create synthetic data with four blobs to experiment with k-means clustering
# 🧪 Generate synthetic blob data
X, y = make_blobs(
    n_samples=500,
    n_features=2,
    centers=4,
    cluster_std=[1.0, 3, 5, 2],
    random_state=42
)

# 🧠 Apply KMeans clustering
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
y_kmeans = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_
colormap = cm.get_cmap("tab10")  # Correctly access the colormap

# 📊 Plot the results
plt.figure(figsize=(18, 6))

# 1️⃣ Raw Data with Centroids
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], s=50, alpha=0.6, edgecolor='k')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', alpha=0.9, label='Centroids')
plt.title(f'Synthetic Blobs with {n_clusters} Clusters')
FEATURE_1_LABEL = 'Feature 1'
FEATURE_2_LABEL = 'Feature 2'
plt.ylabel(FEATURE_2_LABEL)
plt.legend()

# 2️⃣ Clustered Data
plt.subplot(1, 3, 2)
colors = colormap(y_kmeans.astype(float) / n_clusters)
plt.scatter(X[:, 0], X[:, 1], c=colors, s=50, alpha=0.6, edgecolor='k')
plt.scatter(centers[:, 0], centers[:, 1], marker="o", c="white", alpha=1, s=200, edgecolor="k", label='Centroids')
for i, c in enumerate(centers):
    plt.scatter(c[0], c[1], marker=f"${i}$", alpha=1, s=50, edgecolor="k")
plt.title(f'KMeans Clustering with {n_clusters} Clusters')
plt.xlabel(FEATURE_1_LABEL)
plt.ylabel(FEATURE_2_LABEL)
plt.legend()

# 3️⃣ Evaluation: Silhouette + DB Index
plt.subplot(1, 3, 3)
evaluate_clustering(X, y_kmeans, n_clusters, title_suffix='KMeans Clustering')

plt.tight_layout()
plt.show()

#Cluster Stability

# Number of runs for K-Means with different initial seeds
n_runs = 8
inertia_values = []

# Calculate number of subplot rows and columns
n_cols = 2
n_rows = -(-n_runs // n_cols)  # Ceiling division
plt.figure(figsize=(14, n_rows * 5))  # Adjust size based on rows

for i in range(n_runs):
    kmeans = KMeans(n_clusters=4, random_state=42)  # Provide a fixed seed for reproducibility
    inertia_values.append(kmeans.fit(X).inertia_)

    # Plot each result
    plt.subplot(n_rows, n_cols, i + 1)
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='tab10', alpha=0.6, edgecolor='k')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                c='red', s=200, marker='x', label='Centroids')
    plt.title(f'K-Means Run {i + 1}\nInertia={kmeans.inertia_:.2f}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='upper right', fontsize='small')

plt.tight_layout()
plt.show()

# Print inertia scores
print("\n📊 Inertia Scores Across Runs:")
for i, inertia in enumerate(inertia_values, start=1):
    print(f'Run {i}: Inertia = {inertia:.2f}')

#Number of clusters
# Range of k values to test
k_values = range(2, 11)

# Store performance metrics
inertia_values = []
silhouette_scores = []
davies_bouldin_indices = [davies_bouldin_score(X, KMeans(n_clusters=k, random_state=42).fit_predict(X)) for k in k_values]

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    y_kmeans = kmeans.fit_predict(X)
    
    silhouette_scores.append(silhouette_score(X, y_kmeans, random_state=42))  # Ensure random_state is set in KMeans
    inertia_values.append(kmeans.inertia_)
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
CLUSTERS_LABEL = 'Number of Clusters (k)'  # Define once and reuse

# Plot silhouette scores
plt.subplot(1, 3, 2)
plt.plot(k_values, silhouette_scores, marker='o')
plt.title('Silhouette Score vs. k')
plt.plot(k_values, silhouette_scores[:len(k_values)], marker='o')  # Ensure matching dimensions
plt.ylabel('Silhouette Score')

# Plot Davies-Bouldin Index
plt.subplot(1, 3, 3)
plt.plot(k_values, davies_bouldin_indices, marker='o')
plt.title('Davies-Bouldin Index vs. k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Davies-Bouldin Index')
plt.title('Davies-Bouldin Index vs. k')
davies_bouldin_indices = [davies_bouldin_score(X, KMeans(n_clusters=k, random_state=42).fit_predict(X)) for k in k_values]
plt.plot(k_values, davies_bouldin_indices, marker='o')
plt.ylabel('Davies-Bouldin Index')

plt.tight_layout()
plt.show()

#task 1. Plot the blobs and the clustering results for k = 3, 4, and 5

# Plot setup
plt.figure(figsize=(18, 12))
colormap = cm.tab10

for i, k in enumerate([2, 3, 4]):
    # Fit KMeans and predict the labels
    kmeans = KMeans(n_clusters=k, random_state=42)
    y_kmeans = kmeans.fit_predict(X)

    # Create colors based on predicted labels
    colors = colormap(y_kmeans.astype(float) / k)

    # Clustering result subplot
    ax1 = plt.subplot(2, 3, i + 1)
    ax1.scatter(X[:, 0], X[:, 1], c=colors, s=50, alpha=0.6, edgecolor='k')
    ax1.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=200, marker='X', label='Centroids')

    centers = kmeans.cluster_centers_
    for j, c in enumerate(centers):
        ax1.scatter(c[0], c[1], marker=f"${j}$", alpha=1, s=50, edgecolor="k")

    ax1.set_title(f'K-Means Clustering Results, k={k}')
    ax1.set_xlabel(FEATURE_1_LABEL)
    ax1.set_ylabel(FEATURE_2_LABEL)
    ax1.legend()

    # Silhouette plot subplot
    ax2 = plt.subplot(2, 3, i + 4)
    evaluate_clustering(X, y_kmeans, k, ax=ax2, title_suffix=f' (k={k})')

plt.tight_layout()
plt.show()
#task 2. Are these results consistent with our previous results, where we analyzed the evaluation metric plots against k?
from sklearn.datasets import make_classification
from scipy.spatial import Voronoi, voronoi_plot_2d
# Generate synthetic classification data
X, y_true = make_classification(n_samples=300, n_features=2, n_informative=2, n_redundant=0,
                                n_clusters_per_class=1, n_classes=3, random_state=42)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

# Compute the Voronoi diagram
vor = Voronoi(centroids)

# Create a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Get consistent axis limits for all scatter plots
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

# Plot the true labels with Voronoi regions
colormap = cm.tab10
colors_true = colormap(y_true.astype(float) / 3)
axes[0, 0].scatter(X[:, 0], X[:, 1], c=colors_true, s=50, alpha=0.5, ec='k')
voronoi_plot_2d(vor, ax=axes[0, 0], show_vertices=False, line_colors='red', line_width=2, line_alpha=0.6, point_size=2)
axes[0, 0].set_title('Labelled Classification Data with Voronoi Regions')
axes[0, 0].set_xlabel(FEATURE_1_LABEL)
axes[0, 0].set_ylabel(FEATURE_2_LABEL)
axes[0, 0].set_xlim(x_min, x_max)
axes[0, 0].set_ylim(y_min, y_max)

# Call evaluate_clustering for true labels
evaluate_clustering(X, y_true, n_clusters=3, ax=axes[1, 0], title_suffix=' True Labels')
# Call evaluate_clustering for clustering labels
evaluate_clustering(X, y_kmeans, n_clusters=3, ax=axes[1, 0], title_suffix=' Clustering Labels')
colors_kmeans = colormap(y_kmeans.astype(float) / 3)
axes[0, 1].scatter(X[:, 0], X[:, 1], c=colors_kmeans, s=50, alpha=0.5, ec='k')
axes[0, 1].scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='x', label='Centroids')
voronoi_plot_2d(vor, ax=axes[0, 1], show_vertices=False, line_colors='red', line_width=2, line_alpha=0.6, point_size=2)

axes[0, 1].set_title('K-Means Clustering with Voronoi Regions')
axes[0, 1].set_xlabel(FEATURE_1_LABEL)
axes[0, 1].set_ylabel(FEATURE_2_LABEL)
axes[0, 1].set_xlim(x_min, x_max)
axes[0, 1].set_ylim(y_min, y_max)

# Call evaluate_clustering for K-Means labels
evaluate_clustering(X, y_kmeans, n_clusters=3, ax=axes[1, 1], title_suffix=' K-Means Clustering')

# Adjust layout and show plot
plt.tight_layout()
plt.show()
