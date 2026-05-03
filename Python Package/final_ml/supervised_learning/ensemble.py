"""
ensemble.py
===========
A beginner-friendly implementation of the four main ensemble methods
covered in Lectures 9.1 and 9.2:

  1. HardVotingClassifier  – combine different model types by majority vote
  2. BaggingClassifier     – many copies of the same model on bootstrap samples
  3. AdaBoostClassifier    – sequential models that fix previous mistakes (reweighting)
  4. GradientBoostingClassifier – sequential models that fit the residual errors

Each class follows the same simple contract:
  .fit(X, y)      – train the model
  .predict(X)     – return class labels
  .score(X, y)    – return accuracy (correct / total)
"""

import numpy as np
from collections import Counter


# ===========================================================================
# 1. HARD VOTING CLASSIFIER
# ===========================================================================

class HardVotingClassifier:
    """
    Hard Voting Classifier
    ----------------------
    Combines predictions from several *different* models by taking a
    majority vote.  Each model gets one vote, and whichever class gets
    the most votes wins.

    Why is this useful?
    -------------------
    Different model types make different kinds of mistakes.
    • A Decision Tree might fail on smooth boundaries.
    • An SVM might fail on non-linear patterns.
    • Logistic Regression might fail when classes overlap badly.

    When we combine them, each model's weakness is covered by the others,
    so the final prediction is more reliable than any single model alone.

    This is called "hard" voting because every model votes for a single
    class label (as opposed to "soft" voting where models vote with
    probabilities).

    Parameters
    ----------
    estimators : list of (name, model) tuples
        Each element is a 2-tuple: a string label and a scikit-learn
        compatible model (must have .fit() and .predict() methods).

    Example
    -------
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.svm import SVC
    >>> clf = HardVotingClassifier([
    ...     ('tree', DecisionTreeClassifier()),
    ...     ('lr',   LogisticRegression()),
    ...     ('svm',  SVC()),
    ... ])
    >>> clf.fit(X_train, y_train)
    >>> clf.predict(X_test)
    """

    def __init__(self, estimators):
        self.estimators = estimators

    def fit(self, X, y):
        """Train every sub-model on the full training set."""
        for name, model in self.estimators:
            model.fit(X, y)
            print(f"  [HardVoting] Trained: {name}")
        return self

    def predict(self, X):
        """
        For each sample, collect one prediction per model and return
        the class that appeared most often (majority vote).
        """
        # Shape: (n_models, n_samples)
        all_preds = np.array([model.predict(X) for _, model in self.estimators])

        final = []
        for i in range(X.shape[0]):
            # all predictions for sample i
            votes = all_preds[:, i]
            winner = Counter(votes).most_common(1)[0][0]
            final.append(winner)
        return np.array(final)

    def score(self, X, y):
        """Return the fraction of correctly classified samples."""
        return np.mean(self.predict(X) == y)


# ===========================================================================
# 2. BAGGING CLASSIFIER
# ===========================================================================

class BaggingClassifier:
    """
    Bagging Classifier  (Bootstrap AGGregating)
    --------------------------------------------
    Trains many copies of the *same* base model, each on a different
    random bootstrap sample of the training data, then combines their
    predictions by majority vote.

    What is a bootstrap sample?
    ---------------------------
    A bootstrap sample is drawn by randomly picking N samples *with
    replacement* from the original N training points.  This means some
    samples may appear more than once and others not at all (~37 % are
    left out on average).

    Why does this help?
    -------------------
    Each copy sees slightly different data, so it makes slightly
    different errors.  When we average/vote across many copies, those
    errors cancel out — this is called *variance reduction*.

    Key idea: we are turning one high-variance model (e.g. a deep
    decision tree) into a low-variance ensemble.

    Parameters
    ----------
    base_estimator : scikit-learn compatible classifier
        The model to clone for each bag.
    n_estimators   : int
        How many bags (copies) to train.
    random_state   : int or None
        Seed for reproducibility.
    """

    def __init__(self, base_estimator, n_estimators=100, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators   = n_estimators
        self.random_state   = random_state
        self.estimators_    = []   # filled by .fit()

    def _clone(self):
        """Return a fresh copy of the base estimator with the same params."""
        import copy
        return copy.deepcopy(self.base_estimator)

    def fit(self, X, y):
        """
        For each of the n_estimators bags:
          1. Draw a bootstrap sample (same size as training set, with replacement).
          2. Fit a fresh copy of the base model on that sample.
          3. Store the trained model.
        """
        rng = np.random.default_rng(self.random_state)
        n   = X.shape[0]
        self.estimators_ = []

        for i in range(self.n_estimators):
            # Bootstrap: pick n indices WITH replacement
            idx     = rng.integers(0, n, size=n)
            X_boot  = X[idx]
            y_boot  = y[idx]

            model = self._clone()
            model.fit(X_boot, y_boot)
            self.estimators_.append(model)

        print(f"  [Bagging] Trained {self.n_estimators} estimators.")
        return self

    def predict(self, X):
        """Majority vote across all trained models."""
        # Shape: (n_estimators, n_samples)
        all_preds = np.array([m.predict(X) for m in self.estimators_])

        final = []
        for i in range(X.shape[0]):
            votes  = all_preds[:, i]
            winner = Counter(votes).most_common(1)[0][0]
            final.append(winner)
        return np.array(final)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


# ===========================================================================
# 3. ADABOOST CLASSIFIER  (Adaptive Boosting)
# ===========================================================================

class AdaBoostClassifier:
    """
    AdaBoost Classifier  (Adaptive Boosting)
    -----------------------------------------
    Trains models *sequentially*: each new model focuses extra attention
    on the samples the previous models got wrong.

    How it works (step-by-step):
    1.  Start with equal weights for every training sample.
    2.  Train a weak learner (usually a depth-1 decision tree, a "stump").
    3.  Compute the model's weighted error rate.
    4.  Give the model a vote weight (α) based on how well it did —
        better models get bigger votes.
    5.  Increase the sample weights for misclassified points, so the
        next model focuses on them.
    6.  Repeat steps 2-5 for n_estimators rounds.
    7.  Final prediction = sign of the weighted sum of all models' votes.

    The key insight: instead of changing the data (like Bagging), we
    change the *importance* of each sample.  Hard examples get more
    attention as training progresses.

    Parameters
    ----------
    base_estimator : scikit-learn compatible classifier
        Typically a DecisionTreeClassifier(max_depth=1).
    n_estimators   : int
        Number of sequential models to train.
    learning_rate  : float
        Shrinks each model's contribution.  Smaller = more conservative.
    random_state   : int or None
    """

    def __init__(self, base_estimator, n_estimators=50,
                 learning_rate=1.0, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators   = n_estimators
        self.learning_rate  = learning_rate
        self.random_state   = random_state
        self.estimators_    = []
        self.alphas_        = []

    def _clone(self):
        import copy
        return copy.deepcopy(self.base_estimator)

    def fit(self, X, y):
        """
        AdaBoost training loop.
        Expects binary labels encoded as {0, 1} — we convert internally
        to {-1, +1} for the standard AdaBoost math.
        """
        n = X.shape[0]

        # Convert to {-1, +1}
        classes = np.unique(y)
        self.classes_ = classes
        y_ = np.where(y == classes[0], -1, 1)

        # Equal weights to start
        w = np.ones(n) / n

        self.estimators_ = []
        self.alphas_     = []

        for t in range(self.n_estimators):
            model = self._clone()
            # scikit-learn supports sample_weight in .fit()
            model.fit(X, y_, sample_weight=w)

            preds = model.predict(X)

            # Weighted error: fraction of weight on wrong samples
            err = np.sum(w * (preds != y_))

            # Avoid divide-by-zero or log(0)
            err = np.clip(err, 1e-10, 1 - 1e-10)

            # Model vote weight (alpha)
            alpha = self.learning_rate * 0.5 * np.log((1 - err) / err)

            # Update sample weights: wrong → heavier, right → lighter
            w *= np.exp(-alpha * y_ * preds)
            w /= np.sum(w)   # normalize so weights sum to 1

            self.estimators_.append(model)
            self.alphas_.append(alpha)

        print(f"  [AdaBoost] Trained {self.n_estimators} estimators.")
        return self

    def decision_function(self, X):
        """Weighted sum of all models' signed predictions."""
        total = np.zeros(X.shape[0])
        for alpha, model in zip(self.alphas_, self.estimators_):
            total += alpha * model.predict(X)
        return total

    def predict(self, X):
        """Return the class with the highest weighted vote."""
        scores = self.decision_function(X)
        # Map sign back to original class labels
        return np.where(scores < 0, self.classes_[0], self.classes_[1])

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


# ===========================================================================
# 4. GRADIENT BOOSTING CLASSIFIER
# ===========================================================================

class GradientBoostingClassifier:
    """
    Gradient Boosting Classifier
    ----------------------------
    Like AdaBoost, models are trained sequentially.  The key difference:
    instead of reweighting samples, each new tree is fitted to the
    *residual errors* (pseudo-residuals) of the current ensemble.

    Intuition (regression analogy from Lecture 9.2):
    -------------------------------------------------
    Suppose you're trying to predict a value y.
    • Tree 1 predicts y1.  Error = y - y1.
    • Tree 2 tries to predict the *error* (y - y1).
    • Tree 3 tries to predict the remaining error, and so on.
    • Final prediction = y1 + y2 + y3 + ...

    For classification we use log-loss and predict log-odds, then pass
    the result through a sigmoid to get probabilities.

    Parameters
    ----------
    n_estimators   : int     – number of trees
    learning_rate  : float   – shrinks each tree's contribution (0 < lr ≤ 1)
    max_depth      : int     – max depth of each tree
    random_state   : int or None
    """

    def __init__(self, n_estimators=100, learning_rate=0.1,
                 max_depth=3, random_state=None):
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.max_depth     = max_depth
        self.random_state  = random_state
        self.trees_        = []
        self.F0_           = 0.0    # initial prediction (log-odds)

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def fit(self, X, y):
        """
        Train the gradient boosting ensemble for binary classification.
        Internally uses the log-loss / deviance gradient.
        """
        from sklearn.tree import DecisionTreeRegressor

        n = X.shape[0]
        self.classes_ = np.unique(y)
        assert len(self.classes_) == 2, "Only binary classification supported."

        # Encode as 0 / 1
        y_ = (y == self.classes_[1]).astype(float)

        # Initial prediction: log-odds of the positive class
        p0 = np.clip(y_.mean(), 1e-6, 1 - 1e-6)
        self.F0_ = np.log(p0 / (1 - p0))

        # Running predictions (log-odds)
        F = np.full(n, self.F0_)

        self.trees_ = []
        rng = np.random.default_rng(self.random_state)

        for t in range(self.n_estimators):
            # Pseudo-residuals: gradient of log-loss w.r.t. F
            # = (observed probability) - (predicted probability)
            p   = self._sigmoid(F)
            r   = y_ - p

            # Fit a regression tree to the pseudo-residuals
            tree = DecisionTreeRegressor(
                max_depth    = self.max_depth,
                random_state = rng.integers(0, 2**31)
            )
            tree.fit(X, r)

            # Update ensemble
            F += self.learning_rate * tree.predict(X)
            self.trees_.append(tree)

        print(f"  [GradientBoosting] Trained {self.n_estimators} trees.")
        return self

    def predict_proba(self, X):
        """Return probability estimates for the positive class."""
        F = np.full(X.shape[0], self.F0_)
        for tree in self.trees_:
            F += self.learning_rate * tree.predict(X)
        p_pos = self._sigmoid(F)
        return np.column_stack([1 - p_pos, p_pos])

    def predict(self, X):
        proba  = self.predict_proba(X)[:, 1]
        labels = np.where(proba >= 0.5, self.classes_[1], self.classes_[0])
        return labels

    def score(self, X, y):
        return np.mean(self.predict(X) == y)
