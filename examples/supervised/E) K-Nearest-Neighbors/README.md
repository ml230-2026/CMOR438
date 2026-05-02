# K-Nearest Neighbors

This directory contains example code and notes for the K-Nearest Neighbors algorithm in supervised learning.

---

## Algorithm

**K-Nearest Neighbors (KNN)** is a **nonparametric** machine learning algorithm — meaning it makes no assumptions about the shape or distribution of the data. Unlike parametric models (Perceptron, Logistic Regression, MLP) which learn a fixed set of weights during training, KNN has no training phase at all. It simply stores the entire training dataset and does all computation at prediction time.

The core idea is:

> *"Birds of a feather flock together"* — similar data points exist in close proximity.

To classify a new point, KNN:
1. Computes the **Euclidean distance** from the query point to every training example
2. Sorts all training examples by distance
3. Selects the **k closest** examples
4. Takes a **majority vote** of their labels
5. Returns the winning label as the prediction

The Euclidean distance between two points $p$ and $q$ is:

$$d(p, q) = \sqrt{(p - q)^T(p - q)}$$

### Parametric vs. Nonparametric

| | Parametric (MLP, Logistic Regression) | Nonparametric (KNN) |
|---|---|---|
| **Training** | Learns weights from data | Just stores the data |
| **Prediction** | Uses learned weights only | Searches through stored data |
| **Assumptions** | Assumes a functional form | Makes almost none |
| **Speed** | Fast at prediction time | Slow at prediction time |

### Key Hyperparameters

| Hyperparameter | Description |
|---|---|
| `k` | Number of nearest neighbors to use for voting. Use an odd number to avoid ties. |

### Choosing K

- **Small k** → high variance, sensitive to noise
- **Large k** → high bias, overly smooth predictions
- Best practice: plot classification error vs. k and pick the value with the lowest test error

---

## Data

### Dataset: Forest Cover Type

**Source:** [Kaggle — Forest Cover Type by UCI](https://www.kaggle.com/datasets/uciml/forest-cover-type-dataset)

> ⚠️ The dataset file is too large to store in this repository (581,012 rows, ~75MB). Download `covtype.csv` from the link above, rename it to `Forest_Cover.csv`, and place it in the `data/` folder before running the notebook.

**Motivation:** Predicting forest cover type from cartographic (map-based) variables has real-world value for conservation and land management. Instead of sending teams into the field to manually survey forest composition, a well-trained model could predict cover type from satellite data and topographic maps alone. The dataset covers wilderness areas in Roosevelt National Forest in northern Colorado — areas minimally disturbed by humans, making them ideal for studying natural forest ecosystems.

We chose this dataset for KNN because forest cover type is a **multi-class classification problem** (7 possible tree types) with no strong prior about the functional relationship between cartographic features and tree species — exactly where a nonparametric model shines.

### Input Features

| Feature | Description |
|---|---|
| `Elevation` | Elevation in meters |
| `Aspect` | Aspect in degrees azimuth |
| `Slope` | Slope in degrees |
| `Horizontal_Distance_To_Hydrology` | Horizontal distance to nearest water (meters) |
| `Vertical_Distance_To_Hydrology` | Vertical distance to nearest water (meters) |
| `Horizontal_Distance_To_Roadways` | Horizontal distance to nearest road (meters) |
| `Hillshade_9am` | Hillshade index at 9am (0–255) |
| `Hillshade_Noon` | Hillshade index at noon (0–255) |
| `Hillshade_3pm` | Hillshade index at 3pm (0–255) |
| `Horizontal_Distance_To_Fire_Points` | Horizontal distance to nearest wildfire ignition point (meters) |

### Labels

| Label | Forest Cover Type |
|---|---|
| 1 | Spruce/Fir |
| 2 | Lodgepole Pine |
| 3 | Ponderosa Pine |
| 4 | Cottonwood/Willow |
| 5 | Aspen |
| 6 | Douglas Fir |
| 7 | Krummholz |

### Preprocessing

1. **Sampling** — 5,000 rows sampled from the full 581,012 (pure-Python KNN is slow at full scale)
2. **Feature selection** — 10 continuous cartographic features used (binary wilderness/soil columns excluded)
3. **Train/test split** — 80% training, 20% test (random seed 42)
4. **Feature scaling** — StandardScaler applied; critical for KNN since distance is scale-sensitive

### Files

```
E) K-Nearest-Neighbors/
├── data/
│   └── Forest_Cover.csv   ← download from Kaggle link above
├── KNN.ipynb
└── README.md
```
