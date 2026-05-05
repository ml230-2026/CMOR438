## Unsupervised Learning

Unsupervised learning is a class of machine learning algorithms that find hidden structure in unlabeled data. Unlike supervised learning, there are no correct answers to learn from — the algorithm must discover patterns on its own.

### What is Unsupervised Learning?

In unsupervised learning, we only have a feature matrix $X$ — no labels $y$. The algorithm explores the data and finds:
- Clusters — groups of similar data points
- Structure — lower-dimensional representations of the data
- Outliers — points that don't fit any pattern

### Two Main Types

| Type | Goal | Output | Example |
|---|---|---|---|
| Clustering | Group similar points together | Cluster labels | Which countries have similar deforestation patterns? |
| Dimensionality Reduction | Compress data into fewer dimensions | Lower-dim representation | What are the main axes of variation in bee colony data? |

### Algorithms Covered

| # | Algorithm | Type | Dataset | Key Idea |
|---|---|---|---|---|
| A | [PCA](A\)%20PCA/) | Dimensionality Reduction | Bee Colony Loss 🐝 | Project data onto directions of maximum variance |
| B | [K-Means Clustering](B\)%20K-Means-Clustering/) | Clustering | Tree Cover Loss 🌳 | Partition into k clusters by minimizing inertia |
| C | [DBSCAN](C\)%20DBSCAN/) | Clustering | Tree Cover Loss 🌳 | Find dense regions; label sparse points as noise |

### Dataset Used

**🐝 Bee Colony Loss Dataset (USDA)**
Annual U.S. honey bee colony data from 2015–2022, including colony counts, loss percentages, and environmental stressors. Used with PCA to find the main axes of variation across states and years.

**🌳 Tree Cover Loss Dataset (Global Forest Watch)**
Annual tree cover loss in hectares for 236 countries from 2001–2022. Used to cluster countries by deforestation patterns and identify outliers.

### How to Run the Notebooks

1. Clone the repository
2. Install dependencies: `pip install numpy pandas matplotlib scikit-learn`
3. Open any notebook with Jupyter: `jupyter notebook`
4. Run all cells from top to bottom

### Supervised vs Unsupervised — Quick Comparison

| | Supervised | Unsupervised |
|---|---|---|
| Labels | Required | Not needed |
| Goal | Predict known output | Discover hidden structure |
| Evaluation | Accuracy, MSE, R² | Silhouette score, inertia, visual inspection |
| Examples | Classification, Regression | Clustering, PCA |

### Key Concepts Across All Notebooks

- Distance Metrics — Euclidean distance drives both K-Means and DBSCAN
- Feature Scaling — critical for distance-based algorithms
- Explained Variance — how much information PCA components capture
- Silhouette Score — measures how well-separated clusters are
- Inertia — measures compactness of K-Means clusters
- Noise Points — DBSCAN's unique ability to identify outliers
