import numpy as np
from collections import Counter


class RegressionTreeNode:
    """
    A single node in a Regression Tree.

    Either an internal decision node (has a split rule and two children)
    or a leaf node (predicts the mean of the samples that reached it).
    """

    def __init__(self):
        self.feature    = None   # which feature to split on
        self.threshold  = None   # split value (go left if x <= threshold)
        self.left       = None   # left child
        self.right      = None   # right child
        self.prediction = None   # mean value at leaf node


class RegressionTreeRegressor:
    """
    Regression Tree built from scratch.

    Works just like a Decision Tree but for predicting numbers instead
    of class labels. At each node it finds the split that minimizes the
    variance of the two resulting groups. At leaf nodes it predicts the
    mean of all samples that ended up there.

    Parameters
    ----------
    max_depth : int
        Maximum depth of the tree. Default is 5.
    min_samples_split : int
        Minimum samples required to split a node. Default is 2.
    """

    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None


    def _variance(self, y):
        """Variance of a set of values. Used to measure split quality."""
        if len(y) == 0:
            return 0.0
        return np.var(y)


    def _best_split(self, X, y):
        """
        Find the feature and threshold that minimizes weighted variance
        across the two resulting groups.
        """
        best_feature   = None
        best_threshold = None
        best_variance  = float('inf')

        for feature_index in range(X.shape[1]):
            for threshold in np.unique(X[:, feature_index]):

                left_mask  = X[:, feature_index] <= threshold
                right_mask = ~left_mask

                y_left  = y[left_mask]
                y_right = y[right_mask]

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                n = len(y)
                weighted_var = (
                    (len(y_left)  / n) * self._variance(y_left) +
                    (len(y_right) / n) * self._variance(y_right)
                )

                if weighted_var < best_variance:
                    best_variance  = weighted_var
                    best_feature   = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold


    def _build_tree(self, X, y, depth):
        """Recursively build the tree."""

        # Stopping condition 1: max depth reached
        if depth >= self.max_depth:
            node = RegressionTreeNode()
            node.prediction = np.mean(y)
            return node

        # Stopping condition 2: too few samples
        if len(y) < self.min_samples_split:
            node = RegressionTreeNode()
            node.prediction = np.mean(y)
            return node

        # Stopping condition 3: all values are the same
        if len(np.unique(y)) == 1:
            node = RegressionTreeNode()
            node.prediction = y[0]
            return node

        # Find the best split
        feature, threshold = self._best_split(X, y)

        # Stopping condition 4: no valid split found
        if feature is None:
            node = RegressionTreeNode()
            node.prediction = np.mean(y)
            return node

        # Create a decision node and recurse
        node = RegressionTreeNode()
        node.feature   = feature
        node.threshold = threshold

        left_mask  = X[:, feature] <= threshold
        right_mask = ~left_mask

        node.left  = self._build_tree(X[left_mask],  y[left_mask],  depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return node


    def fit(self, X, y):
        """
        Build the regression tree from training data.

        Parameters
        ----------
        X : numpy array of shape (num_samples, num_features)
        y : numpy array of shape (num_samples,) — continuous target values
        """
        self.root = self._build_tree(X, y, depth=0)


    def _predict_one(self, xi, node):
        """Traverse the tree for one sample and return its prediction."""
        if node.prediction is not None:
            return node.prediction
        if xi[node.feature] <= node.threshold:
            return self._predict_one(xi, node.left)
        else:
            return self._predict_one(xi, node.right)


    def predict(self, X):
        """
        Predict continuous values for all samples in X.

        Parameters
        ----------
        X : numpy array of shape (num_samples, num_features)

        Returns
        -------
        numpy array of predicted values
        """
        return np.array([self._predict_one(xi, self.root) for xi in X])


    def mean_squared_error(self, X, y):
        """
        Compute Mean Squared Error on a dataset.

        MSE = (1/N) * sum((y_hat - y)^2)

        Lower is better.
        """
        predictions = self.predict(X)
        return np.mean((predictions - y) ** 2)


    def r_squared(self, X, y):
        """
        Compute R-squared (coefficient of determination).

        R² = 1 - (residual variance / total variance)

        1.0 = perfect, 0.0 = no better than predicting the mean.
        """
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot
