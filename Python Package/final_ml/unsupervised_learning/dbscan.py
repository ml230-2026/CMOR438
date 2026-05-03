"""
dbscan.py
=========
Custom implementation of DBSCAN (Density-Based Spatial Clustering of
Applications with Noise) following the course lecture on unsupervised learning.

DBSCAN groups together points that are closely packed (many nearby neighbors)
and marks points in low-density regions as outliers (noise, label = -1).
Unlike K-Means, it does not require the number of clusters to be specified
in advance and can find arbitrarily shaped clusters.

Algorithm Overview
------------------
1. For each unvisited point p:
   a. Find all points within distance eps of p  (its "epsilon-neighborhood").
   b. If |neighborhood| >= min_samples -> p is a *core point*; start / extend
      a cluster by recursively adding all density-reachable points.
   c. Otherwise p is temporarily marked as noise (it may later become a
      *border point* if it falls inside another core point's neighborhood).
2. Repeat until every point has been visited.

Point Types
-----------
- Core point   : has >= min_samples neighbors within distance eps.
- Border point : within eps of a core point but fewer than min_samples neighbors.
- Noise point  : not within eps of any core point -> label = -1.

References
----------
Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996).
A density-based algorithm for discovering clusters in large spatial databases
with noise. KDD-96 Proceedings, 226-231.
"""

import numpy as np


class DBSCAN:
    """
    Density-Based Spatial Clustering of Applications with Noise (DBSCAN).

    Parameters
    ----------
    eps : float, optional (default=0.5)
        The maximum distance between two points for them to be considered
        in the same neighborhood.  Equivalent to the radius of the
        epsilon-ball drawn around each point.
    min_samples : int, optional (default=5)
        The minimum number of points (including the point itself) required
        inside the eps-neighborhood for a point to be classified as a
        *core point*.

    Attributes
    ----------
    labels_ : np.ndarray, shape (n_samples,)
        Cluster label assigned to each training sample.
        Noise points are labeled -1.
        Valid cluster labels start at 0.
    core_sample_indices_ : np.ndarray, shape (n_core_samples,)
        Indices of all core points found during fit.
    n_clusters_ : int
        Number of clusters found (does not count the noise label -1).
    n_noise_ : int
        Number of points labeled as noise.

    Examples
    --------
    >>> import numpy as np
    >>> from dbscan import DBSCAN
    >>> X = np.array([[1, 2], [1.1, 2.1], [5, 6], [5.1, 6.1], [99, 99]])
    >>> db = DBSCAN(eps=0.5, min_samples=2)
    >>> db.fit(X)
    >>> db.labels_
    array([ 0,  0,  1,  1, -1])
    >>> db.n_clusters_
    2
    """

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

        # Set during fit
        self.labels_ = None
        self.core_sample_indices_ = None
        self.n_clusters_ = None
        self.n_noise_ = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _euclidean_distance(self, a, b):
        """
        Compute the Euclidean distance between two 1-D vectors.

        Parameters
        ----------
        a : np.ndarray, shape (n_features,)
            First point.
        b : np.ndarray, shape (n_features,)
            Second point.

        Returns
        -------
        float
            Euclidean distance between a and b.
        """
        return np.sqrt(np.sum((a - b) ** 2))

    def _get_neighbors(self, X, point_idx):
        """
        Return the indices of all points within eps of X[point_idx].

        The point itself is included in the result (distance = 0).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            The dataset.
        point_idx : int
            Index of the query point.

        Returns
        -------
        list of int
            Indices of all points (including point_idx itself) within
            distance eps of X[point_idx].
        """
        neighbors = []
        for i in range(len(X)):
            if self._euclidean_distance(X[point_idx], X[i]) <= self.eps:
                neighbors.append(i)
        return neighbors

    def _expand_cluster(self, X, labels, point_idx, neighbors, cluster_id):
        """
        Grow a cluster starting from a confirmed core point.

        Recursively visits all points in the neighborhood; if a visited
        point is itself a core point, its neighbors are added to the
        expansion queue.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            The dataset.
        labels : np.ndarray, shape (n_samples,)
            Working label array (modified in-place).
        point_idx : int
            Index of the seed core point.
        neighbors : list of int
            Indices of the initial neighborhood of point_idx.
        cluster_id : int
            The cluster label to assign.

        Returns
        -------
        None
            labels is modified in-place.
        """
        labels[point_idx] = cluster_id

        i = 0
        while i < len(neighbors):
            n_idx = neighbors[i]

            if labels[n_idx] == -1:
                # Noise -> promote to border point of this cluster
                labels[n_idx] = cluster_id

            elif labels[n_idx] == -2:
                # Unvisited -> assign to cluster
                labels[n_idx] = cluster_id

                # Check if this neighbor is itself a core point
                n_neighbors = self._get_neighbors(X, n_idx)
                if len(n_neighbors) >= self.min_samples:
                    # It is a core point -> add its neighbors to the queue
                    for nb in n_neighbors:
                        if nb not in neighbors:
                            neighbors.append(nb)

            i += 1

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, X):
        """
        Fit the DBSCAN model to the dataset X.

        Iterates over every point, finds its epsilon-neighborhood, and
        either starts a new cluster (if the point is a core point) or
        marks it as noise.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data.  Must be a 2-D numeric array.

        Returns
        -------
        self : DBSCAN
            The fitted estimator (allows method chaining).

        Raises
        ------
        TypeError
            If X is not a numpy ndarray.
        ValueError
            If X is not 2-dimensional.

        Examples
        --------
        >>> db = DBSCAN(eps=1.0, min_samples=3)
        >>> db.fit(X_scaled)
        >>> db.labels_
        array([ 0,  0,  1, ..., -1])
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy ndarray.")
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional (n_samples, n_features).")

        n_samples = len(X)

        # -2 = unvisited sentinel (avoids confusion with noise label -1)
        labels = np.full(n_samples, -2, dtype=int)

        cluster_id = 0
        core_indices = []

        for i in range(n_samples):
            if labels[i] != -2:
                # Already visited
                continue

            neighbors = self._get_neighbors(X, i)

            if len(neighbors) < self.min_samples:
                # Not enough neighbors -> noise (for now)
                labels[i] = -1
            else:
                # Core point -> expand a new cluster
                core_indices.append(i)
                self._expand_cluster(X, labels, i, neighbors, cluster_id)
                cluster_id += 1

        self.labels_ = labels
        self.core_sample_indices_ = np.array(core_indices, dtype=int)
        self.n_clusters_ = cluster_id
        self.n_noise_ = int(np.sum(labels == -1))

        return self

    def fit_predict(self, X):
        """
        Fit the model and return cluster labels in one step.

        Equivalent to calling fit(X) then accessing labels_.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        labels : np.ndarray, shape (n_samples,)
            Cluster label for each point. Noise points have label -1.

        Examples
        --------
        >>> labels = DBSCAN(eps=3.5, min_samples=4).fit_predict(X_scaled)
        >>> labels
        array([ 0,  0,  1, ..., -1])
        """
        self.fit(X)
        return self.labels_

    def score(self, X, labels_true):
        """
        Compute the accuracy of the assigned cluster labels against
        ground-truth labels using the best Hungarian assignment.

        .. note::
            DBSCAN is unsupervised and does not predict fixed label IDs.
            This method exhaustively tries all mappings between predicted
            cluster IDs and true class IDs and returns the accuracy of
            the best assignment.  Noise points (label = -1) are always
            counted as misclassified.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Data (used only to verify fit has been called).
        labels_true : np.ndarray, shape (n_samples,)
            Ground-truth class labels (integers starting at 0).

        Returns
        -------
        float
            Best-assignment accuracy in [0.0, 1.0].

        Raises
        ------
        ValueError
            If fit() has not been called before score().

        Examples
        --------
        >>> acc = db.score(X_scaled, y_true)
        >>> round(acc, 4)
        0.8142
        """
        if self.labels_ is None:
            raise ValueError("Call fit() before score().")

        from itertools import permutations

        predicted = self.labels_
        n = len(labels_true)
        unique_pred = [l for l in set(predicted) if l != -1]
        unique_true = list(set(labels_true))

        best_acc = 0.0
        # Try all mappings of predicted cluster ids -> true class ids
        for perm in permutations(unique_true, min(len(unique_pred), len(unique_true))):
            mapping = dict(zip(unique_pred, perm))
            correct = sum(
                mapping.get(predicted[i], -999) == labels_true[i]
                for i in range(n)
                if predicted[i] != -1
            )
            acc = correct / n
            if acc > best_acc:
                best_acc = acc

        return best_acc
