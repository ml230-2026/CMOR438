# Computational Applied Mathematics and Operations Research 438: Data Science & Machine Learning
Hi, my name is Michelle Lee and this repository is for CMOR 438 taught by Professor Randy Davila. This course explores the fundamentals of data science and machine learning. This beginner-friendly repository collects a sample of core machine learning algorithms and provide descriptions in subdirectories with companion jupyter notebooks. 
![Cover page art](drawing.png)

## Table of Contents
**Supervised Learning**
- [The Perceptron](Supervised-Learning/A\)%20The-Perceptron/)
- [Linear Regression](Supervised-Learning/B\)%20Linear%20Regression/)
- [Logistic Regression](Supervised-Learning/C\)%20Logistic%20Regression/)
- [Multilayer Perceptron](Supervised-Learning/D\)%20Multilayer%20Perceptron/)
- [K-Nearest Neighbors](Supervised-Learning/E\)%20K-Nearest-Neighbors/)
- [Decision/Regression Trees](Supervised-Learning/F\)%20Decision%20Random%20Trees/)
- [Random Forests](Supervised-Learning/G\)%20Random%20Forests/)
- [Ensemble Methods](Supervised-Learning/H\)%20Ensemble%20Methods/)

**Unsupervised Learning**
- [K-Means Clustering](Unsupervised-Learning/B\%20K-Means-Clustering/)
- [PCA](Unsupervised-Learning/A\)%20PCA/)
- [DBSCAN](Unsupervised-Learning/C\)%20DBSCAN/)

## What is Machine Learning?

**Machine Learning** is the science and art of programming computers so they can learn from data rather than being explicitly told every rule.

Key vocabulary words:

- **Training set** — the examples the system uses to learn from
- **Training instance** (or sample) — a single example in the training set
- **Model** — the part of the ML system that learns from the training set and makes predictions

## Types of Machine Learning

### I. By How Much Supervision They Get During Training

**Supervised Learning**  
The training set is fed to the algorithm *with* the answers already included. The model learns by comparing its predictions to the correct answers.
- **Classification** — predicting a category (e.g. is this dog high energy or not?)
- **Regression** — predicting a numeric value given a set of features (e.g. how long will this dog breed live?)

**Unsupervised Learning**  
The training data is *unlabeled* — we never tell the algorithm which group a data point belongs to. It has to find the patterns on its own.
- Good for: finding hidden clusters, detecting anomalies, simplifying data
- Includes *dimensionality reduction* (AKA feature extraction) — simplifying data by merging correlated features into one


### II. By Whether the System Can Learn Incrementally

**Batch Learning**
- Trained all at once on available data, then deployed
- Cannot learn incrementally from new data
- If you want the model to know about new data, you have to retrain from scratch on the full dataset (old + new)
- Also called *offline learning*
- Downside: limited resources, takes up a lot of space

**Online Learning**
- Trains the system incrementally by feeding it data instances sequentially — either individually or in small groups called *mini-batches*
- Great for huge datasets that can't fit in one machine's memory (AKA *out-of-core learning*)
- The *learning rate* controls how fast the system adapts to changing data
- Downside: if bad data is fed, the system's performance will decline


### III. By How They Generalize to New Cases

**Instance-Based Learning**
- Learns examples by heart, then generalizes to new cases using a *similarity measure*
- Example: K-Nearest Neighbors

**Model-Based Learning**
- Builds a model from examples, then uses that model to make predictions
- Example: Linear Regression, Decision Trees


## How Do You Know If Your Model Is Good?

Before you can use your model, you need to define its parameter values. To figure out which values make it perform best, you specify a **performance measure:**

- **Utility function** — measures how *good* the model is
- **Cost function** — measures how *bad* it is (we try to minimize this)

The general process for any ML project:
1. Study the data
2. Select a model
3. Train on training data
4. Apply the model to make predictions on new cases (*inference*)

## What Can Go Wrong?

### Bad Data
- **Non-representative data** — training data must be representative of new cases
- **Sampling noise** — sample too small
- **Sampling bias** — sample too large but skewed
- **Poor quality** — full of errors, outliers, and noise
- **Irrelevant features** — need a good set of features to train on (*feature engineering*)
  - *Feature selection* — selecting the most useful features
  - *Feature extraction* — combining existing features to produce a more useful one
  - Creating new features from new data

### Bad Model
- **Overfitting** — performs well on training data but not on new data
- **Underfitting** — too simple to learn the underlying structure of the data

To detect these issues, we split our data into a **training set** and a **test set**. The *generalization error* (or out-of-sample error) is the error rate on new cases — it tells us how well our model will perform on instances it has never seen before.


## The Machine Learning Process

Every notebook in this repo follows these steps:

1. Look at the big picture
2. Get the data
3. Explore and visualize data to gain insights
4. Prepare data for the ML algorithm
5. Select a model and train it
6. Fine-tune the model
7. Present the solution
8. Launch, monitor, and maintain the system

## Data Sets
🌲 [Forest Cover Type](https://www.kaggle.com/datasets/uciml/forest-cover-type-dataset)

🐝 [Bee Colony Loss](https://www.kaggle.com/datasetsthedevastatorbee-colony-census-and-loss-data)

🐾 [Dog Breeds & Lifespan](https://www.kaggle.com/datasets/mexwell/dog-breeds-dataset)

🌊 [Water Potability](https://www.kaggle.com/datasets/adityakadiwal/water-potability)

🌲 [Global Tree Cover Loss](https://www.kaggle.com/datasets/karnikakapoorglobal-forest-data-2001-2022)


## Resources
**Programming Tools & Libraries Used**

**Books**