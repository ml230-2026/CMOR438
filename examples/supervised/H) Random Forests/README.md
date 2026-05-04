# Random Forests

## Overview

Random Forests are an **ensemble of Decision Trees** trained on random subsets of the data and features. By averaging the predictions of many diverse trees, Random Forests dramatically reduce the variance of a single tree while maintaining low bias — making them one of the most reliable and widely-used algorithms in machine learning.

## Dataset — Forest Cover Type 🌲

**Source:** UCI Machine Learning Repository (Roosevelt National Forest, Colorado)
**Features:** Elevation, slope, aspect, distance to water/roads/fire points, hillshade, soil type, wilderness area
**Task:** Multi-class classification — predict one of 7 tree cover types

The Forest Cover dataset is a fitting choice for Random Forests — both literally and algorithmically. The dataset's high dimensionality (54 features including soil types) and complex multi-class structure are exactly where random forests excel, since individual trees overfit but their ensemble generalizes well.

## The Algorithm

For each of `n_trees` trees:
1. **Bootstrap sample** — sample n points with replacement from the training data
2. **Random feature subset** — at each split, only consider $\sqrt{p}$ random features
3. **Grow a full tree** on this bootstrap sample
4. **Aggregate** — majority vote (classification) or average (regression)

The randomness (both in data and features) ensures the trees are **decorrelated** — their errors cancel out when averaged.

## Key Concepts Demonstrated

- Why a single decision tree has high variance
- How bagging reduces variance without increasing bias
- Feature randomization and tree decorrelation
- Out-of-bag (OOB) error estimation
- Feature importance via mean decrease in impurity
- Comparing 1 tree vs 10 trees vs 100 trees

## Notebook Structure

1. **Introduction** — The wisdom of crowds in ML
2. **Data Exploration** — 54 features, 7 classes, 581,012 samples
3. **Preprocessing** — Train/test split, subsampling for speed
4. **Single Tree Baseline** — Accuracy and overfitting
5. **Random Forest** — Fit, evaluate, tune n_trees
6. **Feature Importance** — Which features matter most for tree type?
7. **OOB Error** — Free validation without a test set
8. **Comparison** — Single Tree vs Random Forest vs KNN
9. **Conclusion** — What drives forest cover type in Colorado?

## Single Tree vs Random Forest

| | Single Tree | Random Forest |
|---|---|---|
| Variance | High | Low |
| Bias | Low | Low |
| Interpretability | High | Medium |
| Training time | Fast | Slower |
| Accuracy | Moderate | High |

## Hyperparameters

| Parameter | Effect |
|---|---|
| `n_trees` | More trees = lower variance, slower training |
| `max_depth` | Controls individual tree complexity |
| `max_features` | Controls decorrelation between trees |
| `min_samples_split` | Prevents overfitting on small nodes |

## Limitations

- **Less interpretable** than a single tree
- **Slower to train and predict** than a single tree
- Memory intensive for large forests
- Feature importance can be **biased** toward high-cardinality features

## Connection to Other Algorithms

Random Forests use **bagging** (bootstrap aggregating). The other major ensemble strategy is **boosting** (Gradient Boosting, AdaBoost) — covered in the Ensemble Methods notebook.
