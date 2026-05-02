import numpy as np
from collections import Counter


class DecisionTreeNode:
    """
    A single node in a Decision Tree.

    Either an internal decision node (has a split rule and two children)
    or a leaf node (predicts the majority class of samples that reached it).
    """

    def __init__(self):
        self.feature    = None   # which feature to split on
        self.threshold  = None   # split value (go left if x <= threshold)
        self.left       = None   # left child
        self.right      = None   # right child
        self.prediction = None   # majority class at leaf node


class DecisionTreeClassifier:
    """
    Decision Tree Classifier built from scratch.

    Works by recursively splitting the training data into smaller groups.
    At each node it finds the feature and threshold that creates the purest
    possible groups — where most samples share the same class label.
    At leaf nodes it predicts the majority class of the samples there.

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


    def _impurity(self, y):
        """Measure of how mixed the labels are in a group."""
        if len(y) == 0:
            return 0.0
        counts = Counter(y)
        total = len(y)
        score = 1.0
        for count in counts.values():
            score -= (count / total) ** 2
        return score


    def _best_split(self, X, y):
        """
        Find the feature and threshold that creates the purest groups.
        Tries every feature and every unique value as a candidate threshold.
        """
        best_feature   = None
        best_threshold = None
        best_score     = float('inf')

        for feature_index in range(X.shape[1]):
            for threshold in np.unique(X[:, feature_index]):

                left_mask  = X[:, feature_index] <= threshold
                right_mask = ~left_mask

                y_left  = y[left_mask]
                y_right = y[right_mask]

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                n = len(y)
                score = (
                    (len(y_left)  / n) * self._impurity(y_left) +
                    (len(y_right) / n) * self._impurity(y_right)
                )

                if score < best_score:
                    best_score     = score
                    best_feature   = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold


    def _build_tree(self, X, y, depth):
        """Recursively build the tree."""

        # Stopping condition 1: max depth reached
        if depth >= self.max_depth:
            node = DecisionTreeNode()
            node.prediction = Counter(y).most_common(1)[0][0]
            return node

        # Stopping condition 2: too few samples
        if len(y) < self.min_samples_split:
            node = DecisionTreeNode()
            node.prediction = Counter(y).most_common(1)[0][0]
            return node

        # Stopping condition 3: perfectly pure node
        if len(set(y)) == 1:
            node = DecisionTreeNode()
            node.prediction = y[0]
            return node

        # Find the best split
        feature, threshold = self._best_split(X, y)

        # Stopping condition 4: no valid split found
        if feature is None:
            node = DecisionTreeNode()
            node.prediction = Counter(y).most_common(1)[0][0]
            return node

        # Create a decision node and recurse
        node = DecisionTreeNode()
        node.feature   = feature
        node.threshold = threshold

        left_mask  = X[:, feature] <= threshold
        right_mask = ~left_mask

        node.left  = self._build_tree(X[left_mask],  y[left_mask],  depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return node


    def fit(self, X, y):
        """
        Build the decision tree from training data.

        Parameters
        ----------
        X : numpy array of shape (num_samples, num_features)
        y : numpy array of shape (num_samples,)
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
        Predict class labels for all samples in X.

        Parameters
        ----------
        X : numpy array of shape (num_samples, num_features)

        Returns
        -------
        numpy array of predicted labels
        """
        return np.array([self._predict_one(xi, self.root) for xi in X])


    def score(self, X, y):
        """
        Compute classification accuracy.

        Returns
        -------
        float — accuracy between 0.0 and 1.0
        """
        return np.sum(self.predict(X) == y) / len(y)
