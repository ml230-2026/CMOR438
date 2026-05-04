# Principal Component Analysis (PCA)

## Overview

Principal Component Analysis is a **dimensionality reduction** technique that finds the directions of maximum variance in high-dimensional data and projects the data onto a lower-dimensional space. It's used to visualize complex datasets, remove noise, and compress features before feeding them into other algorithms.

## Dataset — Bee Colony Loss 🐝

**Source:** USDA National Agricultural Statistics Service via Kaggle
**Features:** Colony counts, loss percentages, stressor levels (varroa mites, pesticides, disease, unknown) across U.S. states and quarters 2015–2022
**Task:** Dimensionality reduction — find the main axes of variation in bee colony health across states and time

The bee colony dataset has many correlated features — states that lose colonies to varroa mites often also show pesticide stress. PCA helps us answer: *"What are the underlying drivers of colony health across the U.S.?"* by compressing correlated features into interpretable components.

## The Algorithm

PCA finds a new coordinate system where:
1. **PC1** points in the direction of maximum variance
2. **PC2** points in the direction of maximum remaining variance, orthogonal to PC1
3. Each subsequent PC captures the next most variance

**Steps:**
1. Center the data (subtract mean)
2. Compute the covariance matrix: $\Sigma = \frac{1}{n} X^T X$
3. Compute eigenvectors and eigenvalues of $\Sigma$
4. Sort eigenvectors by eigenvalue (descending)
5. Project data: $Z = X W$ where $W$ = top $k$ eigenvectors

## Key Concepts Demonstrated

- Explained variance ratio — how much information each component captures
- The scree plot — choosing how many components to keep
- Biplot — visualizing both samples and features in PC space
- How PCA removes correlated features
- Using PCA as preprocessing before clustering or classification
- The trade-off between compression and information loss

## Notebook Structure

1. **Introduction** — What is dimensionality? Why reduce it?
2. **Data Exploration** — Feature correlations, state-level trends
3. **Preprocessing** — Standardize (critical for PCA!)
4. **Fit PCA** — Compute components, explained variance
5. **Scree Plot** — Choose number of components
6. **Visualization** — Plot states in PC1-PC2 space
7. **Biplot** — Which stressors load on which components?
8. **PCA as Preprocessing** — Use components for clustering
9. **Conclusion** — What drives bee colony variation across the U.S.?

## Explained Variance

The explained variance ratio tells you how much information each component captures:

$$\text{EVR}_k = \frac{\lambda_k}{\sum_j \lambda_j}$$

A common rule: keep enough components to explain **85–95%** of total variance.

## Limitations

- **Linear only** — can't capture non-linear structure (use t-SNE or UMAP for that)
- Components are **hard to interpret** — they're combinations of all original features
- Sensitive to **outliers** (they inflate variance)
- **Must standardize** before applying — PCA is scale-dependent

## Connection to Other Algorithms

PCA is often used as a **preprocessing step** before K-Means or DBSCAN — reducing dimensions makes clustering faster and avoids the curse of dimensionality. It's also used to visualize high-dimensional clusters in 2D.
