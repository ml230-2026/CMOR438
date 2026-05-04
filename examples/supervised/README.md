## Supervised Learning

Supervised learning is a class of machine learning algorithms that learn from labeled data (datasets where each example has a known input and a known output). The algorithm learns a mapping from inputs to outputs and uses that mapping to make predictions on new data.

### What is Supervised Learning?

In supervised learning, we train a model on a dataset of $(X, y)$ pairs where:
- $X$ is the feature matrix — the input data
- $y$ is the target vector — the correct answers

The model learns to predict $y$ from $X$. Once trained, it can generalize to new inputs it has never seen before.

### Two Types of Supervised Learning

| Type | Goal | Output | Example |
|---|---|---|---|
| Classification | Predict a category | Discrete label | Is this dog a sporting breed? (yes/no) |
|*Regression | Predict a number | Continuous value | How many bee colonies were lost this year? |

### Algorithms Covered

| # | Algorithm | Type | Dataset | Key Idea |
|---|---|---|---|---|
| A | [Perceptron](A%29%20The-Perceptron/) | Classification | Dog Breeds 🐶 | Binary linear classifier inspired by a neuron |
| B | [Linear Regression](B%29%20Linear%20Regression/) | Regression | Bee Colony Loss 🐝 | Fit a line to predict continuous values |
| C | [Logistic Regression](C%29%20Logistic%20Regression/) | Classification | Dog Breeds 🐶 | Probabilistic classifier using the sigmoid function |
| D | [Multilayer Perceptron](D%29%20Multilayer%20Perceptron/) | Classification | Water Potability 💧 | Neural network with hidden layers |
| E | [K-Nearest Neighbors](E%29%20K-Nearest-Neighbors/) | Classification | Forest Cover Type 🌲 | Classify by majority vote of nearest neighbors |
| F | [Regression Trees](F%29%20Regression%20Trees/) | Regression | Forest Cover Type 🌲 | Predict continuous values via recursive splitting |
| G | [Decision Trees](G%29%20Decision%20Trees/) | Classification | Forest Cover Type 🌲 | Classify via recursive binary splitting on features |
| H | [Random Forests](H%29%20Random%20Forests/) | Classification | Forest Cover Type 🌲 | Ensemble of decision trees via bagging |
| I | [Ensemble Methods](I%29%20Ensemble%20Methods/) | Classification | Water Potability 💧 | Combine weak learners into a strong model |

### Datasets Used

🐶 Dog Breeds Dataset
A collection of 277 AKC-recognized dog breeds with features including size, energy level, trainability, lifespan, and group classification. Used for binary and multi-class classification tasks.

🐝 Bee Colony Loss Dataset (USDA)
Annual U.S. honey bee colony data collected by the USDA from 2015–2022, including colony counts, losses, and stressors. Used to explore regression on ecological time-series data.

💧 Water Potability Dataset
Chemical measurements (pH, hardness, chloramines, etc.) for 3,276 water samples, each labeled as potable or not. Used for binary classification.

🌲 Forest Cover Type Dataset (UCI)
Cartographic features (elevation, slope, soil type, distance to landmarks) for forest plots in Roosevelt National Forest, Colorado. Used to classify one of 7 tree cover types.

### How to Run the Notebooks

1. Clone the repository
2. Install dependencies: `pip install numpy pandas matplotlib scikit-learn`
3. Open any notebook with Jupyter: `jupyter notebook`
4. Run all cells from top to bottom

### Key Concepts Across All Notebooks

- Train/Test Split — splitting data so we evaluate on unseen examples
- Evaluation Metrics — accuracy, precision, recall, F1, MSE, R²
- Overfitting vs Underfitting — the bias-variance tradeoff
- Feature Scaling — standardizing inputs so distance-based methods work correctly