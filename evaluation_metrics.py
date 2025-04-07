# evaluation_metrics.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

X = np.load('X.npy')
k_values = range(2, 10)
inertias = []
silhouette_scores = []
db_indices = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)

    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, labels))
    db_indices.append(davies_bouldin_score(X, labels))

plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.plot(k_values, inertias, marker='o')
plt.title('Inertia (Elbow Method)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')

plt.subplot(1, 3, 2)
plt.plot(k_values, silhouette_scores, marker='o')
plt.title('Silhouette Scores')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')

plt.subplot(1, 3, 3)
plt.plot(k_values, db_indices, marker='o')
plt.title('Davies-Bouldin Index')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('DB Index (lower is better)')

plt.tight_layout()
plt.show()
