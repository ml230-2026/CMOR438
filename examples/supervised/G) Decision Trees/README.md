# Decision & Regression Trees

## Overview

Decision Trees are interpretable models that learn a series of **if-then rules** to partition the feature space. They can handle both classification and regression tasks, require no feature scaling, and produce human-readable models — you can literally follow the logic of a prediction step by step.

## Dataset — Forest Cover Type 🌲

**Source:** UCI Machine Learning Repository (Roosevelt National Forest, Colorado)
**Features:** Elevation, slope, aspect, distance to water/roads/fire points, hillshade, soil type, wilderness area
**Task:** Both classification (predict tree cover type) and regression (predict elevation from other features)

Decision Trees are ideal for this dataset because forest ecology follows rule-like logic: *"If elevation > 3000m AND slope < 20° AND soil type is stony, then it's likely Spruce-Fir."* Trees make these implicit rules explicit and interpretable.

## The Algorithm

**Building the tree** (recursive binary splitting):
1. For each feature and each possible split point, compute impurity reduction
2. Choose the split that reduces impurity the most
3. Recursively split each child node
4. Stop when max depth reached or min samples per leaf is met

**Impurity measures:**
- Classification: **Gini impurity** or **Entropy**
- Regression: **Mean Squared Error**

$$\text{Gini} = 1 - \sum_{k} p_k^2 \qquad \text{Entropy} = -\sum_{k} p_k \log_2(p_k)$$

## Key Concepts Demonstrated

- How the splitting criterion works (Gini vs Entropy)
- Tree depth and its effect on overfitting
- Feature importance — which variables drive splits most?
- Visualizing the actual decision tree
- Pruning and regularization via max_depth and min_samples_split
- Regression trees vs classification trees

## Notebook Structure

1. **Introduction** — Decision trees as flowcharts
2. **Data Exploration** — Feature distributions by tree cover type
3. **Preprocessing** — No scaling needed! Encode categoricals
4. **Classification Tree** — Fit, visualize, evaluate
5. **Regression Tree** — Predict elevation from other features
6. **Depth Analysis** — Overfitting curve as depth increases
7. **Feature Importance** — Which features matter most?
8. **Conclusion** — What rules define each tree cover type?

## Overfitting and Depth

| max_depth | Training Accuracy | Test Accuracy | Notes |
|---|---|---|---|
| 1 | Low | Low | Underfits (stump) |
| 5 | Medium | High | Good generalization |
| None | 100% | Lower | Overfits perfectly |

## Advantages

- **Interpretable** — can visualize and explain every prediction
- **No scaling needed** — works with raw features
- **Handles mixed types** — numerical and categorical features
- **Fast** — O(n log n) to build, O(depth) to predict

## Limitations

- **High variance** — small changes in data = very different tree
- **Greedy** — local optimal splits may not be globally optimal
- Deep trees **overfit** easily

## Connection to Other Algorithms

A single Decision Tree is weak. Combine many trees via **bagging** → Random Forest. Combine them via **boosting** → Gradient Boosting. Both dramatically reduce variance while keeping the interpretability of individual trees.
