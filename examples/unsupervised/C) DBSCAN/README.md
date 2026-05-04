# DBSCAN

## Overview

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a clustering algorithm that groups together points that are **closely packed** and marks isolated points as **outliers**. Unlike K-Means, you don't need to specify the number of clusters in advance — DBSCAN discovers it automatically.

## Dataset — Tree Cover Loss 🌳

**Source:** Global Forest Watch via Kaggle
**Features:** Annual tree cover loss (hectares) for 236 countries from 2001 to 2022
**Task:** Cluster countries by deforestation pattern and identify outlier countries with unique loss profiles

K-Means forced every country into a cluster. DBSCAN asks a more honest question: *"Which countries genuinely group together, and which are so extreme or unusual that they don't belong to any group?"* Countries like Brazil — with catastrophic, accelerating deforestation — get labeled as outliers rather than lumped into a cluster with moderate deforesters.

## The Algorithm

Two parameters define the algorithm:
- **ε (eps)** — the radius of a neighborhood around a point
- **min_samples** — the minimum number of neighbors to be a core point

**Three types of points:**
- **Core point** — has ≥ min_samples neighbors within ε
- **Border point** — within ε of a core point but not a core itself
- **Noise point** — not within ε of any core point → labeled **-1**

**Steps:**
1. For each unvisited point, find its ε-neighborhood
2. If it has ≥ min_samples neighbors → it's a core point, start a new cluster
3. Expand the cluster by recursively adding all density-reachable points
4. Any remaining points are noise

## Key Concepts Demonstrated

- How to choose ε using the k-distance graph (elbow method)
- Core points, border points, and noise points explained visually
- Why DBSCAN doesn't need k specified in advance
- Handling outliers — K-Means can't do this, DBSCAN can
- Sensitivity analysis — how results change with ε and min_samples
- PCA visualization of clusters and outliers
- Average loss profile per cluster
- Silhouette score excluding noise points

## Notebook Structure

1. **Introduction** — Density-based clustering vs centroid-based
2. **Data Exploration** — Country-level deforestation patterns
3. **Preprocessing** — Standardize, filter zero-loss countries
4. **k-Distance Graph** — Choose ε from the elbow
5. **Fit DBSCAN** — Cluster countries, identify noise
6. **Explore Clusters** — Which countries are in each group?
7. **Outlier Analysis** — Which countries are noise and why?
8. **PCA Visualization** — Plot clusters in 2D
9. **Cluster Profiles** — Average loss over time per cluster
10. **Sensitivity Analysis** — How eps and min_samples affect results
11. **Comparison with K-Means** — Key differences on same dataset
12. **Conclusion** — What does density reveal about deforestation?

## DBSCAN vs K-Means

| | K-Means | DBSCAN |
|---|---|---|
| Number of clusters | Must specify k | Discovered automatically |
| Cluster shape | Spherical only | Any shape |
| Outliers | Every point assigned | Noise points labeled -1 |
| Parameters | k | ε, min_samples |
| Scales well | Yes | Slower for large data |

## Choosing Parameters

**ε (eps):** Use the k-distance graph — sort distances to the kth nearest neighbor and look for the elbow.

**min_samples:** A common starting point is `min_samples = 2 × n_features`. More samples = stricter core point definition = more noise.

## Limitations

- Sensitive to the choice of **ε and min_samples**
- Struggles with **varying density** — one ε can't capture both dense and sparse clusters
- High-dimensional data makes distances less meaningful
- Slower than K-Means: O(n²) in the worst case

## Connection to Other Algorithms

DBSCAN is the natural follow-up to K-Means on the same dataset — using both shows the tradeoffs between centroid-based and density-based clustering. PCA was used for visualization in both notebooks, tying all three unsupervised algorithms together.
