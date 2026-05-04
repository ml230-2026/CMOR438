# Multilayer Perceptron (MLP)

## Overview

The Multilayer Perceptron is a **feedforward neural network** with one or more hidden layers. By stacking layers of neurons with nonlinear activation functions, the MLP can learn complex, non-linear decision boundaries that a single Perceptron or Logistic Regression cannot.

## Dataset — Water Potability 💧

**Source:** Kaggle
**Features:** pH, hardness, solids, chloramines, sulfate, conductivity, organic carbon, trihalomethanes, turbidity
**Task:** Binary classification — predict whether water is safe to drink

Water quality is a global public health challenge. With 9 continuous chemical measurements, this dataset is a perfect fit for a neural network — the relationship between water chemistry and potability is highly non-linear and involves complex feature interactions that simpler models miss.

## The Algorithm

An MLP consists of:
- **Input layer** — one node per feature
- **Hidden layers** — learned representations with nonlinear activations
- **Output layer** — sigmoid for binary classification

**Forward pass:**
$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}, \quad a^{(l)} = \sigma(z^{(l)})$$

**Backpropagation** computes gradients layer by layer using the chain rule, updating all weights to minimize binary cross-entropy loss.

## Key Concepts Demonstrated

- Forward pass through multiple layers
- Backpropagation and the chain rule explained visually
- How hidden layers learn representations
- The effect of network depth and width
- Training vs validation loss — detecting overfitting
- Activation functions: sigmoid, ReLU, tanh

## Notebook Structure

1. **Introduction** — Why do we need hidden layers? The XOR problem
2. **Data Exploration** — Chemical distributions, missing values, class balance
3. **Preprocessing** — Imputation, scaling, train/test split
4. **Architecture** — Choosing layers and activation functions
5. **Training** — Loss curves, learning rate sensitivity
6. **Evaluation** — Accuracy, confusion matrix, ROC curve
7. **Comparison** — MLP vs Logistic Regression on the same data
8. **Conclusion** — What chemistry predicts safe water?

## Why MLP Over Logistic Regression?

| | Logistic Regression | MLP |
|---|---|---|
| Decision boundary | Linear only | Non-linear (any shape) |
| Feature interactions | Manual engineering needed | Learned automatically |
| Interpretability | High | Lower |
| Data needed | Less | More |

## Hyperparameters

| Parameter | What it controls |
|---|---|
| `layers` | Network architecture (nodes per hidden layer) |
| `learning_rate` | Step size for gradient descent |
| `epochs` | How many passes through the training data |

## Limitations

- **Black box** — hard to interpret what the network learned
- Requires more data than simpler models
- Sensitive to hyperparameter choices
- Can overfit without regularization
