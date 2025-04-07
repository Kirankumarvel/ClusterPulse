# stability_analysis.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = np.load('X.npy')
n_runs = 8
inertia_values = []

n_cols = 2
n_rows = -(-n_runs // n_cols)
plt.figure(figsize=(16, 16))

for i in range(n_runs):
    kmeans = KMeans(n_clusters=4, random_state=None)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)

    plt.subplot(n_rows, n_cols, i + 1)
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='tab10', alpha=0.6, edgecolor='k')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=200, marker='x', label='Centroids')
    plt.title(f'K-Means Run {i + 1}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='upper right', fontsize='small')

plt.tight_layout()
plt.show()

for i, inertia in enumerate(inertia_values, start=1):
    print(f'Run {i}: Inertia={inertia:.2f}')
