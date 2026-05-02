import numpy as np
from collections import Counter
from final_ml.supervised_learning.decision_tree import DecisionTreeClassifier


class RandomForestClassifier:
    """
    Random Forest Classifier built from scratch.

    Builds many Decision Trees and combines their predictions through
    majority voting. Two sources of randomness make each tree different:

    1. Bootstrap sampling — each tree trains on a random sample of the
       data with replacement, so each tree sees a slightly different dataset.
    2. Random feature subsets — at each split, only a random subset of
       features is considered, forcing trees to learn different patterns.

    Because the trees are diverse, their errors are uncorrelated and
    cancel out when averaged — producing better accuracy than any single tree.

    Parameters
    ----------
    n_trees : int
        Number of decision trees to build. Default is 10.
    max_depth : int
        Maximum depth of each tree. Default is 5.
    min_samples_split : int
        Minimum samples required to split a node. Default is 2.
    max_features : int or None
        Number of features to consider at each split.
        If None, uses sqrt(num_features).
    random_state : int or None
        Seed for reproducibility. Default is None.
    """

    def __init__(self, n_trees=10, max_depth=5, min_samples_split=2,
                 max_features=None, random_state=None):
        self.n_trees           = n_trees
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.max_features      = max_features
        self.random_state      = random_state
        self.trees             = []


    def fit(self, X, y):
        """
        Build the random forest by training n_trees decision trees.

        For each tree:
        1. Draw a bootstrap sample of the training data
        2. Randomly select max_features features to use
        3. Train a decision tree on that sample using only those features

        Parameters
        ----------
        X : numpy array of shape (num_samples, num_features)
        y : numpy array of shape (num_samples,)
        """
        rng = np.random.default_rng(self.random_state)
        num_features = X.shape[1]

        if self.max_features is None:
            self.max_features = int(np.sqrt(num_features))

        self.trees = []

        for i in range(self.n_trees):

            # Step 1: Bootstrap sample — sample with replacement
            indices = rng.integers(0, len(X), size=len(X))
            X_sample = X[indices]
            y_sample = y[indices]

            # Step 2: Random feature subset
            feature_indices = rng.choice(num_features,
                                         size=self.max_features,
                                         replace=False)

            # Step 3: Train a decision tree on this sample
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_sample[:, feature_indices], y_sample)

            self.trees.append((tree, feature_indices))
            print(f'Tree {i + 1}/{self.n_trees} trained')


    def predict(self, X):
        """
        Predict class labels using majority voting across all trees.

        Parameters
        ----------
        X : numpy array of shape (num_samples, num_features)

        Returns
        -------
        numpy array of predicted labels
        """
        # Collect predictions from every tree
        all_predictions = np.array([
            tree.predict(X[:, feature_indices])
            for tree, feature_indices in self.trees
        ])

        # Majority vote for each sample
        final_predictions = []
        for sample_idx in range(X.shape[0]):
            votes = all_predictions[:, sample_idx]
            majority = Counter(votes).most_common(1)[0][0]
            final_predictions.append(majority)

        return np.array(final_predictions)


    def score(self, X, y):
        """
        Compute classification accuracy.

        Returns
        -------
        float — accuracy between 0.0 and 1.0
        """
        return np.sum(self.predict(X) == y) / len(y)
