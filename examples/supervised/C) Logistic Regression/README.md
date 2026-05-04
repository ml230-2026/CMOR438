# Logistic Regression

## Overview

Logistic Regression is a **probabilistic binary classifier** that predicts the probability that an input belongs to a class. Despite the name, it is a classification algorithm — it uses a linear model combined with the **sigmoid function** to output probabilities between 0 and 1.

## Dataset — AKC Dog Breeds 🐶

**Source:** American Kennel Club via Kaggle
**Features:** Size, energy level, trainability, shedding, lifespan, breed group
**Task:** Binary classification — predict whether a dog breed is good for first-time owners

Dog breeds have rich, measurable characteristics that correlate with trainability and temperament. Logistic Regression gives us not just a prediction but a **probability** — how likely is this breed to be beginner-friendly?

## The Algorithm

Logistic Regression applies the sigmoid function to a linear combination of features:

$$\hat{p} = \sigma(w \cdot x + b) = \frac{1}{1 + e^{-(w \cdot x + b)}}$$

Predictions are made by thresholding the probability:

$$\hat{y} = \begin{cases} 1 & \text{if } \hat{p} \geq 0.5 \\ 0 & \text{if } \hat{p} < 0.5 \end{cases}$$

Parameters are learned by minimizing **Binary Cross-Entropy (Log Loss)**:

$$\mathcal{L} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

## Key Concepts Demonstrated

- The sigmoid function and why it produces probabilities
- Binary cross-entropy loss vs MSE — why log loss is better for classification
- Decision boundary visualization
- The effect of the classification threshold
- Precision, recall, and the F1 score

## Notebook Structure

1. **Introduction** — From regression to classification
2. **Data Exploration** — Breed traits, class distributions
3. **Preprocessing** — Encoding, scaling, handling class imbalance
4. **Training** — Gradient descent on log loss, loss curve
5. **Evaluation** — Accuracy, confusion matrix, precision, recall, F1
6. **Threshold Analysis** — How changing 0.5 affects precision vs recall
7. **Conclusion** — What predicts a beginner-friendly dog?

## Evaluation Metrics

| Metric | Meaning |
|---|---|
| **Accuracy** | Overall fraction correct |
| **Precision** | Of predicted positives, how many are actually positive? |
| **Recall** | Of actual positives, how many did we catch? |
| **F1 Score** | Harmonic mean of precision and recall |
| **Confusion Matrix** | Full breakdown of TP, FP, TN, FN |

## Perceptron vs Logistic Regression

| | Perceptron | Logistic Regression |
|---|---|---|
| Output | Hard label (-1 or 1) | Probability (0 to 1) |
| Loss | Misclassification count | Binary cross-entropy |
| Update | Only on errors | Every step |
| Converges? | Only if separable | Always |

## Limitations

- Still assumes a **linear decision boundary**
- Binary only (multi-class needs one-vs-rest or softmax)
- Struggles when features are highly correlated
