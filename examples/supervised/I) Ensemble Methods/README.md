### I) Ensemble Methods

**Overview**

Ensemble methods combine multiple machine learning models to produce a prediction that is better than any individual model. The core idea: weak learners become strong learners when combined wisely. This notebook covers three strategies — Hard Voting, Bagging, and Gradient Boosting.

**Dataset — Water Potability 💧**

Source: Kaggle
Features: pH, hardness, solids, chloramines, sulfate, conductivity, organic carbon, trihalomethanes, turbidity
Task: Binary classification — predict whether water is safe to drink

Water potability is a high-stakes classification problem where false negatives (labeling unsafe water as safe) have real consequences. Ensemble methods are particularly well-suited here because combining multiple models reduces the chance of any single model's blind spots affecting the final prediction.

**Three Ensemble Strategies**

1. Hard Voting
Combine multiple different classifiers and take the majority vote.

$$\hat{y} = \text{mode}(\hat{y}_1, \hat{y}_2, \ldots, \hat{y}_k)$$

Best when you have diverse models with different strengths.

2. Bagging (Bootstrap Aggregating)
Train the same model on different random subsets of the data, then average.

$$\hat{y} = \frac{1}{B} \sum_{b=1}^{B} f_b(x)$$

Reduces variance. Random Forests are a special case of bagging.

3. Gradient Boosting
Train models sequentially, where each new model corrects the errors of the previous one.

$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

Where $h_m$ fits the residuals of $F_{m-1}$. Reduces both bias and variance.

**Key Concepts Demonstrated**

- Why individual models fail and how ensembles help
- Diversity in ensemble members is key
- Bagging vs Boosting — when to use each
- How Gradient Boosting builds trees sequentially
- Learning rate and n_estimators in boosting
- Comparing all three methods on water potability

**Notebook Structure**

1. Introduction — The wisdom of crowds
2. Data Exploration — Chemical feature distributions, class imbalance
3. Preprocessing — Imputation, scaling, train/test split
4. Baseline — Single Decision Tree performance
5. Hard Voting — Combine Logistic Regression, KNN, Decision Tree
6. Bagging — BaggingClassifier with decision tree base
7. Gradient Boosting — Sequential boosting, learning curves
8. Comparison Table — All methods head to head
9. Conclusion — Which ensemble works best for water safety?

**Bagging vs Boosting**

| | Bagging | Boosting |
|---|---|---|
| Trees trained | In parallel | Sequentially |
| Focus | Reducing variance | Reducing bias + variance |
| Overfitting risk | Low | Higher (if too many rounds) |
| Speed | Faster | Slower |
| Example | Random Forest | Gradient Boosting |

**Hyperparameters (Gradient Boosting)**

| Parameter | Effect |
|---|---|
| `n_estimators` | Number of boosting rounds |
| `learning_rate` | Shrinks each tree's contribution |
| `max_depth` | Depth of each weak learner |

**Limitations**

- Harder to interpret than a single model
- Gradient Boosting is sensitive to hyperparameters
- Training time increases with more estimators
- Can overfit with too many boosting rounds

**Connection to Other Algorithms**

Ensemble Methods build on Decision Trees (covered in the previous notebook). The Random Forest is a special case of Bagging.
