# Linear Regression

## Overview

Linear Regression is the foundational algorithm for predicting continuous numerical values. It models the relationship between input features and a target by fitting a straight line (or hyperplane) through the data using **gradient descent**.

## Dataset — Bee Colony Loss 🐝

**Source:** USDA National Agricultural Statistics Service via Kaggle
**Features:** Year, quarter, state, number of colonies, colony loss percentage, stressors (varroa mites, pesticides, disease)
**Task:** Regression — predict the number of bee colonies lost given environmental and seasonal features

Bee colony collapse is one of the most pressing ecological crises affecting agriculture. This dataset lets us ask: can we predict colony loss from measurable stressors? Linear Regression is the natural first model for this kind of continuous prediction problem.

## The Algorithm

Linear Regression fits a model of the form:

$$\hat{y} = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b$$

Parameters are learned by minimizing **Mean Squared Error (MSE)**:

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

Using **gradient descent**:

$$w \leftarrow w - \eta \cdot \frac{\partial \text{MSE}}{\partial w}$$

## Key Concepts Demonstrated

- Gradient descent and how the loss decreases over epochs
- The meaning of weights (slope) and bias (intercept)
- Visualizing the regression line against real data
- MSE and R² as evaluation metrics
- Multivariate regression with multiple features

## Notebook Structure

1. **Introduction** — What is regression? When do we use it?
2. **Data Exploration** — Colony trends over time, stressor distributions
3. **Preprocessing** — Handle missing values, encode states, scale features
4. **Training** — Gradient descent, loss curve visualization
5. **Evaluation** — MSE, RMSE, R² score
6. **Interpretation** — Which stressors most predict colony loss?
7. **Conclusion** — What does this tell us about bee health?

## Evaluation Metrics

| Metric | Formula | Meaning |
|---|---|---|
| MSE | $\frac{1}{n}\sum(y - \hat{y})^2$ | Average squared error |
| RMSE | $\sqrt{\text{MSE}}$ | Error in original units |
| R² | $1 - \frac{\text{SS}_{res}}{\text{SS}_{tot}}$ | Proportion of variance explained |

## Limitations

- Assumes a **linear relationship** between features and target
- Sensitive to **outliers** (they inflate MSE disproportionately)
- Assumes **independence** of features (multicollinearity is a problem)

## Connection to Other Algorithms

Linear Regression + sigmoid activation = **Logistic Regression**. Add hidden layers = **Neural Network**. The gradient descent update rule used here is the same one powering deep learning.
