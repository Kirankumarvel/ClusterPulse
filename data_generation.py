# data_generation.py
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

# Generate synthetic data
X, y_true = make_blobs(n_samples=500, centers=4, cluster_std=1.0, random_state=42)

# Plot the data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], s=50, c=y_true, cmap='tab10', alpha=0.6, edgecolor='k')
plt.title('Generated Blobs Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.show()

# Save for use in other scripts
np.save('X.npy', X)
np.save('y_true.npy', y_true)
