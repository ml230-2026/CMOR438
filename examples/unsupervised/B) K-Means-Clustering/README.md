### B) K-Means Clustering

**Overview88

K-Means is the most widely used clustering algorithm. It partitions data into k groups by iteratively assigning points to the nearest centroid and updating centroids to the mean of their assigned points. Unlike supervised learning, there are no labels — the algorithm discovers structure on its own.

**Dataset — Tree Cover Loss 🌳**

Source: Global Forest Watch via Kaggle
Features: Annual tree cover loss (hectares) for 236 countries from 2001 to 2022
Task: Cluster countries by their deforestation patterns over 21 years

Deforestation is a global crisis but it looks very different across countries — some have high chronic loss, some have accelerating loss, some have almost none. K-Means helps us answer: *"Which countries share similar deforestation stories?"* without being told the answer in advance.

**The Algorithm**

Given $k$ and data $X$:
1. Initialize — randomly select $k$ points as initial centroids
2. Assign — assign each point to the nearest centroid
   $$c_i = \arg\min_k \|x_i - \mu_k\|^2$$
3. Update — recompute each centroid as the mean of its assigned points
   $$\mu_k = \frac{1}{|C_k|} \sum_{i \in C_k} x_i$$
4. Repeat steps 2–3 until centroids stop moving

**Key Concepts Demonstrated**

- The elbow method — choosing the right k
- Inertia (within-cluster sum of squares) as a loss function
- How random initialization affects results
- Silhouette score for evaluating cluster quality
- Visualizing clusters with PCA (2D projection)
- Interpreting what each cluster means in context
- Comparing cluster profiles — average loss over time per cluster

**Notebook Structure**

1. Introduction — What is clustering? When do we use it?
2. Data Exploration — Which countries lost the most forest? Time trends?
3. Preprocessing — Standardize yearly loss columns, filter zero-loss countries
4. Elbow Method — Plot inertia vs k to find the "elbow"
5. Fit K-Means — Cluster countries by loss pattern
6. Silhouette Score — Quantify cluster quality
7. PCA Visualization — Plot countries in 2D colored by cluster
8. Cluster Profiles — Average loss over time per cluster
9. Who's In Each Cluster? — Name the countries in each group
10. Conclusion — What deforestation stories does K-Means reveal?

**Choosing k: The Elbow Method**

Plot inertia (total within-cluster distance) vs number of clusters. The "elbow" where the curve bends is the best k — adding more clusters beyond that gives diminishing returns.

**Inertia vs Silhouette Score**

| Metric | What it measures | Better when |
|---|---|---|
| Inertia | Within-cluster compactness | Lower |
| Silhouette | Separation between clusters | Higher (max 1.0) |

**Limitations**

- Must specify k in advance
- Assumes spherical clusters — struggles with irregular shapes
- Sensitive to outliers (they pull centroids)
- Results depend on random initialization
- Every point is assigned to a cluster — no concept of noise/outliers

**Connection to Other Algorithms**

K-Means assumes spherical clusters and assigns every point to a group. DBSCAN (next notebook) relaxes both assumptions — it finds clusters of any shape and explicitly labels outliers as noise.
