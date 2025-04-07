# 📊 ClusterPulse

**ClusterPulse** is a hands-on machine learning project to explore, evaluate, and visualize the behavior of the K-Means clustering algorithm on synthetic data.

From synthetic blob generation to silhouette analysis, elbow method, and cluster stability tests — this project is your sandbox for understanding clustering dynamics.

---

## 🔍 What This Project Covers

- ✅ Generating synthetic datasets using `make_blobs`
- ✅ Applying K-Means clustering with varying `k` values
- ✅ Visualizing clusters and centroids
- ✅ Evaluating clustering with:
  - **Inertia (Elbow Method)**
  - **Silhouette Scores**
  - **Davies-Bouldin Index**
- ✅ Analyzing cluster stability by:
  - Varying initial centroid seeds
  - Comparing multiple K-Means runs
- ✅ Plotting all results with Matplotlib

---

## 🧪 Project Structure

```
ClusterPulse/
├── data_generation.py        # Create synthetic blobs with varying std
├── kmeans_clustering.py      # Apply KMeans and visualize results
├── stability_analysis.py     # Run KMeans multiple times, assess inertia
├── evaluation_metrics.py     # Plot Elbow, Silhouette, and DBI
├── cluster_comparison.py     # Compare results for k = 3, 4, 5
├── utils.py                  # Evaluation function (e.g. silhouette plot)
└── README.md                 # This file
```

---

## 📈 Example Visuals

- ✅ **K-Means clustering results with centroids**
- ✅ **Cluster stability across 8 random seeds**
- ✅ **Inertia vs. k (Elbow Method)**
- ✅ **Silhouette plots for k = 3, 4, 5**
- ✅ **Davies-Bouldin Index analysis**

---

## 🚀 How to Run

1. Clone the repo:

   ```bash
   git clone https://github.com/Kirankumarvel/ClusterPulse.git
   cd ClusterPulse
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the scripts in order or explore notebooks (optional):

   ```bash
   python data_generation.py
   python kmeans_clustering.py
   python stability_analysis.py
   python evaluation_metrics.py
   python cluster_comparison.py
   ```

---

## 🧠 Insights

- K-Means clustering is sensitive to centroid initialization
- Evaluation metrics like silhouette score and DB index help find the optimal k
- Inertia always decreases with higher k — beware of overfitting
- Synthetic data lets us explore ground truth vs. predicted clusters

---

## 📎 Requirements

- Python 3.7+
- numpy
- matplotlib
- scikit-learn




---

## 🔖 License

MIT License – free to use, fork, and learn from!
