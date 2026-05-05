### G) Decision Trees

**Overview**

A Decision Tree is a supervised learning algorithm that learns a series of if-then rules to classify data into categories. It recursively splits the feature space into regions based on feature values, and at each leaf predicts the majority class of training points in that region. The result is a human-readable flowchart you can follow from root to leaf to understand exactly why a prediction was made.

**Dataset — Forest Cover Type 🌲**

Source: UCI Machine Learning Repository (Roosevelt National Forest, Colorado)
Features: Elevation, slope, aspect, distance to water/roads/fire points, hillshade, soil type, wilderness area
Task: Classification — predict one of 7 tree cover types from cartographic features

Forest ecology follows rule-like logic: *"If elevation > 3000m AND soil type is stony AND slope < 15°, then it's likely Spruce-Fir."* Decision Trees make these implicit rules explicit and interpretable — you can literally read the tree and understand what geographic conditions define each cover type.

**The Algorithm**

Building the tree (recursive binary splitting on impurity):
1. For each feature and each possible split point, compute the impurity reduction
2. Choose the split that reduces impurity the most:

$$\text{Gini} = 1 - \sum_{k=1}^{K} p_k^2 \qquad \text{Entropy} = -\sum_{k=1}^{K} p_k \log_2(p_k)$$

3. Recursively split each child node
4. Stop when max depth is reached or min samples per leaf is met
5. At each leaf, predict the **majority class**:

$$\hat{y} = \arg\max_k \sum_{i \in R} \mathbf{1}[y_i = k]$$

**Decision Tree vs Regression Tree**

| | Decision Tree | Regression Tree |
|---|---|---|
| Target | Discrete class label | Continuous number |
| Impurity measure | Gini / Entropy | Mean Squared Error |
| Leaf prediction | Majority class | Mean of targets |
| Evaluation | Accuracy, F1 | MSE, RMSE, R² |

**Key Concepts Demonstrated**

- How Gini impurity and Entropy work as splitting criteria
- Visualizing the actual decision tree structure
- Depth vs accuracy — how max_depth controls overfitting
- Feature importance — which cartographic features best separate tree cover types?
- Decision boundary visualization using 2 PCA components
- Comparing Gini vs Entropy as splitting criteria

**Notebook Structure**

1. Introduction — Decision trees as flowcharts you can read
2. Data Exploration — Feature distributions by cover type
3. Preprocessing — Train/test split, encode categoricals, no scaling needed
4. Fit Decision Tree — Train, visualize the tree structure
5. Depth Analysis — Overfitting curve as depth increases
6. Evaluation — Accuracy, confusion matrix, per-class F1
7. Feature Importance — Which features most separate cover types?
8. Decision Boundary — Visualize in 2D via PCA
9. Gini vs Entropy — Compare the two splitting criteria
10. Conclusion — What rules define each tree cover type in Colorado?

**Evaluation Metrics**

| Metric | Meaning |
|---|---|
| Accuracy | Overall fraction correctly classified |
| Confusion Matrix | Full breakdown of correct and incorrect predictions per class |
| Per-class F1 | Harmonic mean of precision and recall for each cover type |
| Feature Importance | Mean decrease in impurity contributed by each feature |

**Effect of max_depth**

| max_depth | Training Accuracy | Test Accuracy | Notes |
|---|---|---|---|
| 1 | Low | Low | Underfits — just a stump |
| 5 | Medium | High | Good generalization |
| None | 100% | Lower | Overfits perfectly |

**Advantages**

- Interpretable — visualize and explain every single prediction
- No scaling needed — splits use rank order, not magnitude
- Handles mixed types — numerical and categorical features together
- Fast — O(n log n) to build, O(depth) to predict

**Limitations**

- High variance — small changes in data produce very different trees
- Greedy — local optimal splits may not be globally optimal
- Deep trees overfit easily without pruning
- Less accurate than ensemble methods on complex data

**Connection to Other Algorithms**

A single Decision Tree has high variance. Random Forests (next notebook) fix this by training many trees on random data subsets and averaging their votes — dramatically reducing variance while keeping the power of tree-based splits.
