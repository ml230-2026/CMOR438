## CMOR 438: Data Science & Machine Learning
Hi, my name is Michelle Lee and this repository is for CMOR 438 taught by Professor Randy Davila. This course explores the fundamentals of data science and machine learning. The goal of this repository is to collect a sample of core machine learning algorithms and provide descriptions in subdirectories with companion jupyter notebooks in a beginner-friendly way. The underlying goal is to explore some environmental concepts and learn new things through these models.  
![Cover page art](drawing.png)

### Table of Contents: Examples 

Supervised Learning
- [The Perceptron](examples/supervised/A%29%20The-Perceptron/)
- [Linear Regression](examples/supervised/B%29%20Linear%20Regression/)
- [Logistic Regression](examples/supervised/C%29%20Logistic%20Regression/)
- [Multilayer Perceptron](examples/supervised/D%29%20Multilayer%20Perceptron/)
- [K-Nearest Neighbors](examples/supervised/E%29%20K-Nearest-Neighbors/)
- [Regression Trees](examples/supervised/F%29%20Regression%20Trees/)
- [Decision Trees](examples/supervised/G%29%20Decision%20Trees/)
- [Random Forests](examples/supervised/H%29%20Random%20Forests/)
- [Ensemble Methods](examples/supervised/I%29%20Ensemble%20Methods/)

Unsupervised Learning
- [PCA](examples/unsupervised/A%29%20PCA/)
- [K-Means Clustering](examples/unsupervised/B%29%20K-Means-Clustering/)
- [DBSCAN](examples/unsupervised/C%29%20DBSCAN/)

### What is Machine Learning?

Machine Learning is the science and art of programming computers so they can learn from data rather than being explicitly told every rule.

Key vocabulary words:
- Training set — the examples the system uses to learn from
- Training instance (or sample) — a single example in the training set
- Model — the part of the ML system that learns from the training set and makes predictions

### Types of Machine Learning

**I. By How Much Supervision They Get During Training**

*Supervised Learning*
The training set is fed to the algorithm with the answers already included. The model learns by comparing its predictions to the correct answers.
- Classification — predicting a category
- Regression — predicting a numeric value given a set of features

*Unsupervised Learning* 
The training data is unlabeled — never tells the algorithm which group a data point belongs to. It has to find the patterns on its own.
- Good for: finding hidden clusters, detecting anomalies, simplifying data
- Includes dimensionality reduction (AKA feature extraction) — simplifying data by merging correlated features into one

**II. By Whether the System Can Learn Incrementally**

*Batch Learning*
- Trained all at once on available data, then deployed
- Cannot learn incrementally from new data
- If you want the model to know about new data, you have to retrain from scratch on the full dataset (old + new)
- Also called offline learning
- Downside: limited resources, takes up a lot of space

*Online Learning*
- Trains the system incrementally by feeding it data instances sequentially — either individually or in small groups called mini-batches
- Great for huge datasets that can't fit in one machine's memory (AKA out-of-core learning)
- The learning rate controls how fast the system adapts to changing data
- Downside: if bad data is fed, the system's performance will decline


**III. By How They Generalize to New Cases**

*Instance-Based Learning*
- Learns examples by heart, then generalizes to new cases using a similarity measure
- Example: K-Nearest Neighbors

*Model-Based Learning*
- Builds a model from examples, then uses that model to make predictions
- Example: Linear Regression, Decision Trees


### How Do You Know If Your Model Is Good?

Before you can use your model, you need to define its parameter values. To figure out which values make it perform best, you specify a performance measure:

- Utility function — measures how good the model is
- Cost function — measures how bad it is (we try to minimize this)

The general process for any ML project:
1. Study the data
2. Select a model
3. Train on training data
4. Apply the model to make predictions on new cases (*inference*)

### What Can Go Wrong?

**Bad Data**
- Non-representative data — training data must be representative of new cases
- Sampling noise — sample too small
- Sampling bias — sample too large but skewed
- Poor quality — full of errors, outliers, and noise
- Irrelevant features — need a good set of features to train on (feature engineering)
  - Feature selection — selecting the most useful features
  - Feature extraction — combining existing features to produce a more useful one
  - Creating new features from new data

**Bad Model**
- Overfitting — performs well on training data but not on new data
- Underfitting — too simple to learn the underlying structure of the data

To detect these issues, people split their data into a training set and a test set. The generalization error (or out-of-sample error) is the error rate on new cases — it tells us how well the model will perform on instances it has never seen before.


### The Machine Learning Process

Every notebook in this repo follows these steps:

1. Look at the big picture
2. Get the data
3. Explore and visualize data to gain insights
4. Prepare data for the ML algorithm
5. Select a model and train it
6. Fine-tune the model
7. Present the solution
8. Launch, monitor, and maintain the system

### This Repository Structure
```
CMOR438/
├── .github/
│   └── workflows/
│       └── Tests.yml
├── Examples/
│   ├── supervised/
│   │   ├── A) The-Perceptron/
│   │   ├── B) Linear Regression/
│   │   ├── C) Logistic Regression/
│   │   ├── D) Multilayer Perceptron/
│   │   ├── E) K-Nearest-Neighbors/
│   │   ├── F) Regression Trees/
│   │   ├── G) Decision Trees/
│   │   ├── H) Random Forests/
│   │   └── I) Ensemble Methods/
│   └── unsupervised/
│       ├── A) PCA/
│       ├── B) K-Means-Clustering/
│       └── C) DBSCAN/
├── Python Package/
│   └── final_ml/
│       ├── __init__.py
│       ├── supervised_learning/
│       │   ├── decision_tree.py
│       │   ├── ensemble.py
│       │   ├── knn.py
│       │   ├── linear_regression.py
│       │   ├── logistic_regression.py
│       │   ├── mlp.py
│       │   ├── perceptron.py
│       │   ├── random_forest.py
│       │   └── regression_tree.py
│       └── unsupervised_learning/
│           ├── dbscan.py
│           ├── kmeans.py
│           └── pca.py
├── Tests/
│   └── unit/
│       ├── test_dbscan.py
│       ├── test_decision_tree_classifier.py
│       ├── test_decision_tree_regressor.py
│       ├── test_ensemble.py
│       ├── test_kmeans.py
│       ├── test_knn.py
│       ├── test_linear_regression.py
│       ├── test_logistic_regression.py
│       ├── test_mlp.py
│       ├── test_pca.py
│       ├── test_perceptron.py
│       └── test_random_forest.py
├── .gitignore
├── LICENSE
├── README.md
└── pyproject.toml
```

## Installation
```bash
git clone https://github.com/ml230-2026/CMOR438.git
cd CMOR438
pip install -e .
```
## Running the Tests
```bash
pytest Tests/unit/ -v
```
145 tests covering all supervised and unsupervised learning algorithms. Tests include correctness checks, edge cases, and shape validation.
## Running the Notebooks
Download the datasets from Kaggle and place them in the `data/` folder inside each algorithm directory. Or open the notebook in VS Code and click **Run All**.

### Data Sets
🌲 [Forest Cover Type](https://www.kaggle.com/datasets/uciml/forest-cover-type-dataset)
Compiled in 1998 by Jock Blackard, Denis Dean, and Charles Anderson from the Remote Sensing and GIS Program at Colorado State University, using data collected by the US Geological Survey and US Forest Service. The dataset captures naturally and minimally disturbed forest ecosystems across wilderness areas in Roosevelt National Forest in northern Colorado. Features are entirely cartographic — tree type, shadow coverage, distance to nearby landmarks, soil type, and local topography — with no satellite or remote sensing data. Originally compiled to research neural networks for forest cover classification. Source: UCI Machine Learning Repository

🐝 [Bee Colony Loss](https://www.kaggle.com/datasetsthedevastatorbee-colony-census-and-loss-data)
Compiled by Brenda Griffith of the Bee Informed Partnership (BIP) from 2010–2024, combining data from two government and nonprofit sources: USDA Agricultural Statistics Service colony census surveys (at county and state levels) and BIP annual colony loss reports. The dataset brings together a comprehensive picture of honeybee health across the U.S., created to analyze colony health trends and inform decisions about saving the honeybee population. Source: USDA Agricultural Statistics Service & Bee Informed Partnership via Kaggle

🐾 [Dog Breeds & Lifespan](https://www.kaggle.com/datasets/mexwell/dog-breeds-dataset)
Created in 2024 by Telmo Silva Filho as a personal passion project, this dataset covers all 277 officially recognized American Kennel Club dog breeds. Features include size, energy level, trainability, shedding, lifespan, and breed group. All data rights belong to the AKC. Source: American Kennel Club via Kaggle

🌊 [Water Potability](https://www.kaggle.com/datasets/adityakadiwal/water-potability)
Created by Aditya Kadiwal, this dataset contains water quality measurements for 3,276 water bodies around the world, testing chemical and physical properties — pH, hardness, solids, chloramines, sulfate, conductivity, organic carbon, trihalomethanes, and turbidity — against WHO and US EPA safety standards. Motivated by global health concern, the dataset reflects the principle that "safe drinking water is essential to health, a basic human right and component of effective policy for health protection." Source: Kaggle


🌲 [Global Tree Cover Loss](https://www.kaggle.com/datasets/karnikakapoorglobal-forest-data-2001-2022)
Created by Kanika Kapoor using data from Global Forest Watch, the World Resources Institute, Maryland's GLAD Laboratory, and Google, based on research by Hansen et al. and Harris et al. Satellite imagery was used to track forest change across the planet from 2001–2022. Designed to support strategies in forest conservation and carbon management. Importantly, "loss" does not mean deforestation specifically — it refers to any removal or mortality of tree cover, whether from logging, fire, or natural causes. Source: Global Forest Watch / World Resources Institute via Kaggle

### Resources
Programming Tools & Libraries Used
- Python 3.13
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- jupyter
- pytest
- git
- GitHub Actions 
- pip
- pyproject.oml
- virtual environemt (myenv)

Books
- Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems by Aurélien Géron