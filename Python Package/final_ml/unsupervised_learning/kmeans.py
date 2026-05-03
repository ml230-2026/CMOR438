"""
kmeans.py
=========
Custom implementation of K-Means Clustering following Lecture 10.1.

K-Means partitions n data points into k clusters by iterating two steps:
  1. Assign each point to its nearest centroid (Euclidean distance).
  2. Move each centroid to the mean of all points assigned to it.

These steps repeat until centroids converge or max_iter is reached.
"""

import numpy as np


class KMeans:
    """K-Means Clustering algorithm implemented from scratch.

    Partitions data into k clusters by minimising the total
    within-cluster sum of squared distances (inertia).

    Follows the professor's lecture structure exactly:
      1. Randomly initialise k centroids from the data.
      2. Assign every point to its nearest centroid.
      3. Move each centroid to the mean of its assigned points.
      4. Repeat steps 2-3 until convergence or max_iter is reached.

    Parameters
    ----------
    n_clusters : int, optional
        Number of clusters k to form (default is 3).
    max_iter : int, optional
        Maximum number of assign-and-update iterations (default is 100).
    random_state : int or None, optional
        Seed for reproducible centroid initialisation (default is None).

    Attributes
    ----------
    cluster_centers_ : np.ndarray, shape (n_clusters, n_features)
        Coordinates of the final cluster centroids after fitting.
    labels_ : np.ndarray, shape (n_samples,)
        Cluster index assigned to each training sample after fitting.
    inertia_ : float
        Total within-cluster sum of squared distances to centroids.
        Lower is better (tighter clusters).

    Examples
    --------
    >>> import numpy as np
    >>> from final_ml.unsupervised_learning.kmeans import KMeans
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]], dtype=float)
    >>> km = KMeans(n_clusters=2, random_state=0)
    >>> km.fit(X)
    >>> km.cluster_centers_.shape == (2, 2)
    True
    """

    def __init__(self, n_clusters=3, max_iter=100, random_state=None):
        self.n_clusters       = n_clusters
        self.max_iter         = max_iter
        self.random_state     = random_state
        self.cluster_centers_ = None
        self.labels_          = None
        self.inertia_         = None

    def _distance(self, point_a, point_b):
        """Compute the Euclidean distance between two points.

            Parameters
            ----------
            point_a : np.ndarray, shape (n_features,)
                First point.
            point_b : np.ndarray, shape (n_features,)
                Second point.

            Returns
            -------
            float
                Euclidean distance between point_a and point_b.

            Examples
            --------
            >>> import numpy as np
            >>> from final_ml.unsupervised_learning.kmeans import KMeans
            >>> km = KMeans()
            >>> km._distance(np.array([0.0, 0.0]), np.array([3.0, 4.0]))
            5.0
        """
        return float(np.sqrt(np.sum((point_a - point_b) ** 2)))

    def _assign_clusters(self, X):
        """Assign each sample to its nearest centroid.

            Parameters
            ----------
            X : np.ndarray, shape (n_samples, n_features)
                Data matrix to assign.

            Returns
            -------
            np.ndarray, shape (n_samples,)
                Cluster index for each sample (0 to n_clusters - 1).

            Examples
            --------
            >>> import numpy as np
            >>> from final_ml.unsupervised_learning.kmeans import KMeans
            >>> km = KMeans(n_clusters=2)
            >>> km.cluster_centers_ = np.array([[0.0, 0.0], [10.0, 10.0]])
            >>> km._assign_clusters(np.array([[1.0, 1.0], [9.0, 9.0]])).tolist()
            [0, 1]
        """
        labels = []
        for point in X:
            distances = [self._distance(point, c) for c in self.cluster_centers_]
            labels.append(int(np.argmin(distances)))
        return np.array(labels)

    def _update_centers(self, X, labels):
        """Move each centroid to the mean of its assigned points.

            Parameters
            ----------
            X : np.ndarray, shape (n_samples, n_features)
                Data matrix.
            labels : np.ndarray, shape (n_samples,)
                Current cluster assignment for each sample.

            Returns
            -------
            np.ndarray, shape (n_clusters, n_features)
                Updated centroid positions.

            Examples
            --------
            >>> import numpy as np
            >>> from final_ml.unsupervised_learning.kmeans import KMeans
            >>> km = KMeans(n_clusters=2)
            >>> X = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 0.0]])
            >>> labels = np.array([0, 0, 1])
            >>> new_centers = km._update_centers(X, labels)
            >>> new_centers.shape == (2, 2)
            True
        """
        new_centers = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            points_in_cluster = X[labels == k]
            if len(points_in_cluster) > 0:
                new_centers[k] = points_in_cluster.mean(axis=0)
            else:
                # If a cluster is empty reinitialise its center randomly
                rng = np.random.default_rng(self.random_state)
                new_centers[k] = X[rng.integers(0, len(X))]
        return new_centers

    def fit(self, X):
        """Fit K-Means to the data by iterating assign and update steps.

            Parameters
            ----------
            X : np.ndarray, shape (n_samples, n_features)
                Training data. Should be standardized before calling fit
                so no single feature dominates the distance calculation.

            Returns
            -------
            self : KMeans
                The fitted KMeans object (allows method chaining).

            Raises
            ------
            ValueError
                If n_clusters is greater than n_samples.

            Examples
            --------
            >>> import numpy as np
            >>> from final_ml.unsupervised_learning.kmeans import KMeans
            >>> X = np.array([[1, 2], [1, 4], [1, 0],
            ...               [10, 2], [10, 4], [10, 0]], dtype=float)
            >>> km = KMeans(n_clusters=2, random_state=0)
            >>> fitted = km.fit(X)
            >>> fitted is km
            True
        """
        X = np.array(X, dtype=float)
        n_samples = X.shape[0]

        if self.n_clusters > n_samples:
            raise ValueError(
                f"n_clusters={self.n_clusters} cannot be greater than "
                f"n_samples={n_samples}."
            )

        # Step 1: Randomly pick k data points as starting centers
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(n_samples, size=self.n_clusters, replace=False)
        self.cluster_centers_ = X[idx].copy()

        # Steps 2–3: Iterate assign and update until convergence
        for _ in range(self.max_iter):
            old_centers = self.cluster_centers_.copy()
            labels      = self._assign_clusters(X)
            self.cluster_centers_ = self._update_centers(X, labels)

            # Stop early if centers have not moved
            if np.allclose(old_centers, self.cluster_centers_):
                break

        # Store final labels and inertia
        self.labels_  = self._assign_clusters(X)
        self.inertia_ = float(sum(
            self._distance(X[i], self.cluster_centers_[self.labels_[i]]) ** 2
            for i in range(n_samples)
        ))

        return self

    def predict(self, X):
        """Assign new data points to the nearest fitted centroid.

            Parameters
            ----------
            X : np.ndarray, shape (n_samples, n_features)
                New data to cluster. Must have the same number of features
                as the training data passed to fit().

            Returns
            -------
            np.ndarray, shape (n_samples,)
                Cluster index for each sample (0 to n_clusters - 1).

            Raises
            ------
            RuntimeError
                If predict is called before fit.

            Examples
            --------
            >>> import numpy as np
            >>> from final_ml.unsupervised_learning.kmeans import KMeans
            >>> X = np.array([[1, 2], [1, 4], [10, 2], [10, 4]], dtype=float)
            >>> km = KMeans(n_clusters=2, random_state=0)
            >>> km.fit(X)
            >>> labels = km.predict(X)
            >>> labels.shape == (4,)
            True
        """
        if self.cluster_centers_ is None:
            raise RuntimeError("Call fit() before predict().")
        return self._assign_clusters(np.array(X, dtype=float))

    def fit_predict(self, X):
        """Fit the model and return cluster labels for the training data.

            Equivalent to calling fit(X) followed by predict(X).

            Parameters
            ----------
            X : np.ndarray, shape (n_samples, n_features)
                Training data.

            Returns
            -------
            np.ndarray, shape (n_samples,)
                Cluster index for each sample.

            Examples
            --------
            >>> import numpy as np
            >>> from final_ml.unsupervised_learning.kmeans import KMeans
            >>> X = np.array([[1, 2], [1, 4], [10, 2], [10, 4]], dtype=float)
            >>> km = KMeans(n_clusters=2, random_state=0)
            >>> labels = km.fit_predict(X)
            >>> labels.shape == (4,)
            True
        """
        return self.fit(X).predict(X)

    def score(self, X):
        """Return the negative inertia of the model on X.

            A higher (less negative) score means tighter, better clusters.
            Follows sklearn's convention where higher score = better model.

            Parameters
            ----------
            X : np.ndarray, shape (n_samples, n_features)
                Data to score.

            Returns
            -------
            float
                Negative inertia (negative total within-cluster distance²).

            Raises
            ------
            RuntimeError
                If score is called before fit.

            Examples
            --------
            >>> import numpy as np
            >>> from final_ml.unsupervised_learning.kmeans import KMeans
            >>> X = np.array([[1, 2], [1, 4], [10, 2], [10, 4]], dtype=float)
            >>> km = KMeans(n_clusters=2, random_state=0)
            >>> km.fit(X)
            >>> km.score(X) <= 0
            True
        """
        if self.cluster_centers_ is None:
            raise RuntimeError("Call fit() before score().")
        X      = np.array(X, dtype=float)
        labels = self._assign_clusters(X)
        inertia = float(sum(
            self._distance(X[i], self.cluster_centers_[labels[i]]) ** 2
            for i in range(len(X))
        ))
        return -inertia
