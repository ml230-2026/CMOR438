"""
pca.py
======
Custom implementation of Principal Component Analysis (PCA) using
Singular Value Decomposition (SVD), following Lecture 10.2.

PCA reduces the dimensionality of a dataset by projecting data onto
the directions (principal components) that capture the most variance.

Steps (from the lecture):
  1. Standardize the data (center and scale to z-scores).
  2. Compute the covariance matrix S = (1/(n-1)) * A^T * A.
  3. Find eigenvectors of S via SVD of A.
  4. Sort principal components by descending eigenvalue.
  5. Project the data onto the top k components.
"""

import numpy as np


class PCA:
    """Dimensionality reduction using Principal Component Analysis (PCA).

    Computes principal components via Singular Value Decomposition (SVD)
    of the centered data matrix, following the derivation in Lecture 10.2.

    The first principal component points in the direction of greatest
    variance in the data. Each subsequent component is orthogonal to all
    previous ones and captures the next greatest variance.

    Parameters
    ----------
    n_components : int or None, optional
        Number of principal components to keep. If None, all components
        are kept (default is None).

    Attributes
    ----------
    components_ : np.ndarray, shape (n_components, n_features)
        The principal component directions (rows are PCs, sorted by
        descending explained variance). Equivalent to Vt[:n_components]
        from the SVD decomposition A = U * Sigma * Vt.

    explained_variance_ : np.ndarray, shape (n_components,)
        The variance explained by each principal component.
        Computed as sigma_i^2 / (n_samples - 1).

    explained_variance_ratio_ : np.ndarray, shape (n_components,)
        The fraction of total variance explained by each principal
        component. All ratios sum to 1.0 (or to a value <= 1.0 when
        n_components < n_features).

    mean_ : np.ndarray, shape (n_features,)
        Per-feature mean of the training data. Subtracted during
        transform so that new data is centered the same way.

    Examples
    --------
    >>> import numpy as np
    >>> from final_ml.unsupervised_learning.pca import PCA
    >>> X = np.array([[1, 2, 3],
    ...               [4, 5, 6],
    ...               [7, 8, 9],
    ...               [10, 11, 12]], dtype=float)
    >>> pca = PCA(n_components=2)
    >>> pca.fit(X)
    >>> pca.components_.shape == (2, 3)
    True
    """

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_               = None
        self.explained_variance_       = None
        self.explained_variance_ratio_ = None
        self.mean_                     = None

    def fit(self, X):
        """Fit the PCA model by computing principal components from X.

        Centers the data, performs SVD, and stores the principal
        component directions and explained variance ratios.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data. Rows are observations, columns are features.
            Should be standardized (z-scored) before calling fit.

        Returns
        -------
        self : PCA
            The fitted PCA object (allows method chaining).

        Raises
        ------
        ValueError
            If n_components is greater than min(n_samples, n_features).

        Examples
        --------
        >>> import numpy as np
        >>> from final_ml.unsupervised_learning.pca import PCA
        >>> X = np.array([[2.5, 2.4],
        ...               [0.5, 0.7],
        ...               [2.2, 2.9],
        ...               [1.9, 2.2],
        ...               [3.1, 3.0]], dtype=float)
        >>> pca = PCA(n_components=1)
        >>> fitted = pca.fit(X)
        >>> fitted is pca
        True
        """
        X = np.array(X, dtype=float)
        n_samples, n_features = X.shape

        # Validate n_components
        max_components = min(n_samples, n_features)
        if self.n_components is not None and self.n_components > max_components:
            raise ValueError(
                f"n_components={self.n_components} must be <= "
                f"min(n_samples, n_features)={max_components}."
            )

        # Step 1: Center the data (subtract column means)
        # The professor's notes: form matrix A = X - mean(X, axis=0)
        self.mean_ = X.mean(axis=0)
        A = X - self.mean_

        # Step 2 & 3: SVD — A = U * diag(sigma) * Vt
        # The rows of Vt are the principal component directions,
        # sorted from largest to smallest singular value automatically.
        _, sigma, Vt = np.linalg.svd(A, full_matrices=False)

        # Step 4: Compute explained variance per PC
        # Eigenvalue of covariance matrix = sigma_i^2 / (n_samples - 1)
        explained_variance = (sigma ** 2) / (n_samples - 1)
        total_variance     = explained_variance.sum()

        # Step 5: Keep only the top n_components
        k = self.n_components if self.n_components is not None else n_features

        self.components_               = Vt[:k]
        self.explained_variance_       = explained_variance[:k]
        self.explained_variance_ratio_ = explained_variance[:k] / total_variance

        return self

    def transform(self, X):
        """Project X onto the fitted principal components.

        Subtracts the training mean (computed during fit) and then
        multiplies by the principal component matrix to obtain the
        lower-dimensional representation.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to project. Must have the same number of features as
            the training data passed to fit().

        Returns
        -------
        np.ndarray, shape (n_samples, n_components)
            Data projected onto the top n_components principal components.
            Each column corresponds to one principal component axis.

        Raises
        ------
        RuntimeError
            If transform is called before fit.

        Examples
        --------
        >>> import numpy as np
        >>> from final_ml.unsupervised_learning.pca import PCA
        >>> X = np.array([[2.5, 2.4],
        ...               [0.5, 0.7],
        ...               [2.2, 2.9]], dtype=float)
        >>> pca = PCA(n_components=1)
        >>> pca.fit(X)
        >>> X_projected = pca.transform(X)
        >>> X_projected.shape == (3, 1)
        True
        """
        if self.components_ is None:
            raise RuntimeError("Call fit() before transform().")

        X = np.array(X, dtype=float)

        # Center using the mean learned during fit
        A = X - self.mean_

        # Project: A · Vt.T  (n_samples × n_features) · (n_features × k)
        return A @ self.components_.T

    def fit_transform(self, X):
        """Fit the model and project X in a single step.

        Equivalent to calling fit(X) followed by transform(X), but
        slightly more efficient because it reuses the SVD computation.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data to fit and project.

        Returns
        -------
        np.ndarray, shape (n_samples, n_components)
            Data projected onto the top n_components principal components.

        Examples
        --------
        >>> import numpy as np
        >>> from final_ml.unsupervised_learning.pca import PCA
        >>> X = np.array([[1.0, 2.0, 3.0],
        ...               [4.0, 5.0, 6.0],
        ...               [7.0, 8.0, 9.0]], dtype=float)
        >>> pca = PCA(n_components=2)
        >>> X_projected = pca.fit_transform(X)
        >>> X_projected.shape == (3, 2)
        True
        """
        return self.fit(X).transform(X)

    def get_loading_scores(self, feature_names):
        """Return a summary of how each original feature contributes to each PC.

        Loading scores are the entries of each principal component vector.
        A large absolute value means that feature strongly drives that PC.

        Parameters
        ----------
        feature_names : list of str
            Names of the original features, in the same order as the
            columns of X passed to fit(). Length must equal n_features.

        Returns
        -------
        dict
            A dictionary mapping each PC label (e.g. 'PC1', 'PC2') to
            a dict of {feature_name: loading_score} pairs, sorted by
            descending absolute loading score.

        Raises
        ------
        RuntimeError
            If called before fit().
        ValueError
            If len(feature_names) does not match n_features.

        Examples
        --------
        >>> import numpy as np
        >>> from final_ml.unsupervised_learning.pca import PCA
        >>> X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        >>> pca = PCA(n_components=2)
        >>> pca.fit(X)
        >>> scores = pca.get_loading_scores(['height', 'weight'])
        >>> 'PC1' in scores
        True
        """
        if self.components_ is None:
            raise RuntimeError("Call fit() before get_loading_scores().")

        n_features = self.components_.shape[1]
        if len(feature_names) != n_features:
            raise ValueError(
                f"Expected {n_features} feature names, got {len(feature_names)}."
            )

        result = {}
        for i, component in enumerate(self.components_):
            label = f"PC{i + 1}"
            # Sort features by descending absolute loading score
            pairs = sorted(
                zip(feature_names, component),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            result[label] = dict(pairs)
        return result
