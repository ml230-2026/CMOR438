# Regression Trees

## Overview

A Regression Tree is a Decision Tree adapted for **predicting continuous numerical values** instead of class labels. It recursively splits the feature space into regions and predicts the **mean target value** of training points in each region. The result is a piecewise constant function that approximates any non-linear relationship without assuming linearity.

## Dataset — Forest Cover Type 🌲

**Source:** UCI Machine Learning Repository (Roosevelt National Forest, Colorado)
**Features:** Elevation, slope, aspect, distance to water/roads/fire points, hillshade, soil type, wilderness area
**Task:** Regression — predict elevation from other cartographic features

While the Forest Cover dataset is often used for classification, it contains rich continuous features that make it ideal for regression as well. Elevation is a continuous target with strong non-linear relationships to slope, hillshade, and distance to water — relationships a regression tree can capture without any manual feature engineering.

## The Algorithm

**Building the tree** (recursive binary splitting on MSE):
1. For each feature and each possible split point, compute the reduction in MSE
2. Choose the split that minimizes the weighted sum of child MSEs:

$$\text{MSE split} = \frac{n_L}{n} \text{MSE}_L + \frac{n_R}{n} \text{MSE}_R$$

3. Recursively split each child until stopping criteria are met
4. At each leaf, predict the **mean** of training targets in that region:

$$\hat{y} = \frac{1}{|R_k|} \sum_{i \in R_k} y_i$$

## Regression Tree vs Classification Tree

| | Classification Tree | Regression Tree |
|---|---|---|
| Target | Discrete class label | Continuous number |
| Impurity measure | Gini / Entropy | Mean Squared Error |
| Leaf prediction | Majority class | Mean of targets |
| Evaluation | Accuracy, F1 | MSE, RMSE, R² |

## Key Concepts Demonstrated

- How MSE splitting works step by step
- Visualizing the piecewise constant prediction surface
- Depth vs accuracy — how max_depth controls overfitting
- Residual plots to evaluate regression quality
- Feature importance — which cartographic features best predict elevation?
- Comparing regression tree to linear regression on the same data

## Notebook Structure

1. **Introduction** — From classification trees to regression trees
2. **Data Exploration** — Elevation distribution, correlations with other features
3. **Preprocessing** — Train/test split, no scaling needed
4. **Fit Regression Tree** — Train, visualize the tree structure
5. **Depth Analysis** — Overfitting curve as depth increases
6. **Evaluation** — MSE, RMSE, R² on test set
7. **Residual Plot** — Where does the model struggle?
8. **Feature Importance** — Which features drive elevation predictions?
9. **Comparison** — Regression Tree vs Linear Regression
10. **Conclusion** — What does elevation depend on in Colorado forests?

## Evaluation Metrics

| Metric | Formula | Meaning |
|---|---|---|
| MSE | $\frac{1}{n}\sum(y - \hat{y})^2$ | Average squared error |
| RMSE | $\sqrt{\text{MSE}}$ | Error in original units (meters) |
| R² | $1 - \frac{\text{SS}_{res}}{\text{SS}_{tot}}$ | Proportion of variance explained |

## Advantages

- **No scaling needed** — splits are based on rank order, not magnitude
- **Handles non-linearity** — captures complex relationships automatically
- **Interpretable** — you can visualize and follow every prediction
- **Handles mixed feature types** — numerical and categorical together

## Limitations

- **Piecewise constant** — predictions are step functions, not smooth curves
- **High variance** — small data changes produce very different trees
- Deep trees **overfit** easily
- Less accurate than ensemble methods (Random Forests, Gradient Boosting)

## Connection to Other Algorithms

A single Regression Tree is the building block of **Gradient Boosting** — one of the most powerful algorithms in machine learning. Gradient Boosting trains regression trees sequentially, where each tree corrects the residuals of the previous one, covered in the Ensemble Methods notebook.
