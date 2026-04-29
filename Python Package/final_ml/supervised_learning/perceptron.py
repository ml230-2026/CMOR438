import numpy as np
# numpy is our math library — it lets us do fast operations on arrays
# (lists of numbers). Instead of looping through numbers one by one,
# numpy can handle entire vectors at once, which is much faster.


class Perceptron:
    """
    A Perceptron binary classifier implemented from scratch.

    The Perceptron is the oldest machine learning algorithm — a single
    artificial neuron inspired by how biological neurons fire. It learns
    a linear decision boundary between two classes by correcting its
    mistakes using the update rule:

        w <- w - alpha * (y_hat - y) * x
        b <- b - alpha * (y_hat - y)

    Note: Labels must be -1 or 1 (not 0 and 1).

    Parameters
    ----------
    learning_rate : float, optional
        Step size alpha for weight updates (default is 0.01).
        Too large → model overcorrects. Too small → model learns slowly.
    n_epochs : int, optional
        Number of full passes over the training data (default is 1000).

    Returns
    -------
    None

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    >>> y = np.array([1, 1, -1, -1])
    >>> model = Perceptron(learning_rate=0.1, n_epochs=10)
    >>> model = model.fit(X, y)
    >>> isinstance(model.weights, np.ndarray)
    True
    """

    def __init__(self, learning_rate=0.01, n_epochs=1000):
        # __init__ is the constructor — it runs automatically when you
        # create a new Perceptron object like: model = Perceptron()
        # It sets up all the starting values before any training happens.

        # learning_rate (alpha) controls how big each correction step is.
        # Too big → the model overcorrects and bounces around wildly.
        # Too small → the model learns very slowly.
        # 0.01 is a safe default starting point.
        self.learning_rate = learning_rate

        # n_epochs = how many times we loop through the ENTIRE training set.
        # One epoch = one full pass over all training examples.
        # More epochs = more chances to correct mistakes = better learning.
        self.n_epochs = n_epochs

        # weights will become a vector — one weight per feature.
        # Set to None now because we don't know the number of features
        # until we actually see the training data inside fit().
        self.weights = None

        # bias is a single number that shifts the decision boundary.
        # Without it, the boundary would always pass through the origin.
        self.bias = None

        # Tracks how many mistakes are made per epoch.
        # If learning is working, this number should decrease over time.
        self.errors_per_epoch = []


    def _activation(self, z):
        """
        Apply the step activation function Phi(z) from the professor's notes.

        Converts the preactivation value z into a binary prediction.
        If z is positive the neuron fires (output 1).
        If z is zero or negative the neuron stays silent (output -1).

        Parameters
        ----------
        z : float or np.ndarray
            Preactivation value(s) — the raw weighted sum before activation.

        Returns
        -------
        int or np.ndarray
            1 if z > 0, -1 if z <= 0.

        Raises
        ------
        TypeError
            If z is not a numeric type or numpy array.

        Examples
        --------
        >>> model = Perceptron()
        >>> model._activation(2.5)
        1
        >>> model._activation(-1.0)
        -1
        >>> model._activation(0)
        -1
        """
        # np.where(condition, value_if_true, value_if_false)
        # Works like a vectorized if/else — handles entire arrays at once.
        # Every positive z → 1, every zero or negative z → -1.
        return np.where(z > 0, 1, -1)


    def fit(self, X, y):
        """
        Train the Perceptron on labeled training data.

        Loops over all training examples for n_epochs passes. For each
        example, computes a prediction and applies the update rule if
        the prediction was wrong. Only misclassified examples change
        the weights — correct predictions leave the model unchanged.

        Parameters
        ----------
        X : np.ndarray
            Training feature matrix of shape (n_samples, n_features).
            Each row is one training example, each column is one feature.
        y : np.ndarray
            True binary labels of shape (n_samples,).
            Values must be -1 or 1.

        Returns
        -------
        self : Perceptron
            The trained Perceptron object with updated weights and bias.

        Raises
        ------
        ValueError
            If y contains values other than -1 or 1.
        TypeError
            If X or y are not numpy arrays.

        Examples
        --------
        >>> import numpy as np
        >>> X = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        >>> y = np.array([1, 1, -1, -1])
        >>> model = Perceptron(learning_rate=0.1, n_epochs=10)
        >>> model = model.fit(X, y)
        >>> model.weights is not None
        True
        """
        # X.shape returns (n_samples, n_features)
        # n_samples = number of training examples (rows)
        # n_features = number of features per example (columns)
        n_samples, n_features = X.shape

        # Initialize weights as zeros — one weight per feature.
        # All features start equally unimportant. Training will
        # adjust each weight based on how useful that feature is.
        self.weights = np.zeros(n_features)

        # Initialize bias to zero.
        # It will shift during training to move the decision boundary.
        self.bias = 0.0

        # Reset error tracker in case fit() is called more than once.
        self.errors_per_epoch = []

        # ── OUTER LOOP: repeat for n_epochs full passes over the data ──
        for epoch in range(self.n_epochs):

            # Fresh mistake counter for this epoch
            errors = 0

            # ── INNER LOOP: look at one training example at a time ──
            for i in range(n_samples):
                # X[i] is the feature vector for the i-th training example.
                # For a dog breed it might look like:
                # [55.88, 63.50, 22.68, 9.0, 1.0, ...]
                #  height  height weight  life  trainability...

                # ── STEP 1: PREACTIVATION ──
                # z = w^T * x + b
                # Multiply each feature by its weight, sum everything up,
                # then add the bias. Result is a single number.
                # np.dot computes the dot product: w1*x1 + w2*x2 + ...
                z = np.dot(self.weights, X[i]) + self.bias

                # ── STEP 2: POST-ACTIVATION ──
                # Pass z through the step function to get a prediction.
                # z > 0  →  y_hat = 1  (predicts: high energy!)
                # z <= 0 →  y_hat = -1 (predicts: not high energy!)
                y_hat = self._activation(z)

                # ── STEP 3: COMPUTE ERROR ──
                # error = y_hat - y[i]  (predicted minus actual)
                #
                # error = 0  → correct prediction → no update needed
                # error = 2  → predicted 1, true was -1 → pull weights down
                # error = -2 → predicted -1, true was 1 → push weights up
                error = y_hat - y[i]

                # ── STEP 4: APPLY UPDATE RULE ──
                # w <- w - alpha * error * x[i]
                # Nudge every weight in the direction that reduces the error.
                # Features that are larger get bigger nudges because they
                # contributed more to the wrong prediction.
                self.weights -= self.learning_rate * error * X[i]

                # b <- b - alpha * error
                # Bias update — same idea but no x[i] multiplication
                # because bias doesn't depend on the input features.
                self.bias -= self.learning_rate * error

                # Count this as a mistake if error was non-zero
                if error != 0:
                    errors += 1

            # Record total mistakes made this epoch.
            # Plotting this over time shows the learning curve!
            self.errors_per_epoch.append(errors)

        # Return self allows method chaining:
        # model = Perceptron().fit(X_train, y_train)
        return self


    def predict(self, X):
        """
        Predict binary class labels for input data.

        Runs the forward pass — computes the preactivation z for every
        example and applies the step function to get predictions.
        Does not update weights — this is inference only.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
            Must have the same number of features as the training data.

        Returns
        -------
        np.ndarray
            Predicted labels of shape (n_samples,).
            Each value is either -1 or 1.

        Raises
        ------
        TypeError
            If X is not a numpy array.

        Examples
        --------
        >>> import numpy as np
        >>> X_train = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        >>> y_train = np.array([1, 1, -1, -1])
        >>> model = Perceptron(learning_rate=0.1, n_epochs=100)
        >>> model = model.fit(X_train, y_train)
        >>> predictions = model.predict(X_train)
        >>> all(p in [-1, 1] for p in predictions)
        True
        """
        # Compute preactivation for ALL examples at once using matrix math.
        # np.dot(X, weights) multiplies every row of X by the weights vector
        # giving one z value per example — much faster than a loop!
        z = np.dot(X, self.weights) + self.bias

        # Apply the step function to every z value at once.
        # Returns an array of -1s and 1s — one prediction per example.
        return self._activation(z)


    def score(self, X, y):
        """
        Compute the classification accuracy on a labeled dataset.

        Accuracy = number of correct predictions / total predictions.
        A score of 1.0 means every prediction was correct.
        A score of 0.5 means the model is no better than random guessing.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            True labels of shape (n_samples,). Values must be -1 or 1.

        Returns
        -------
        float
            Accuracy score between 0.0 and 1.0.

        Raises
        ------
        TypeError
            If X or y are not numpy arrays.

        Examples
        --------
        >>> import numpy as np
        >>> X = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        >>> y = np.array([1, 1, -1, -1])
        >>> model = Perceptron(learning_rate=0.1, n_epochs=100)
        >>> model = model.fit(X, y)
        >>> 0.0 <= model.score(X, y) <= 1.0
        True
        """
        # Get predictions for all examples
        predictions = self.predict(X)

        # predictions == y returns a boolean array e.g. [True, False, True, True]
        # np.mean() treats True as 1 and False as 0
        # so the mean = fraction of correct predictions = accuracy
        # Example: [True, True, False, True] → 3/4 = 0.75 → 75% accurate
        return np.mean(predictions == y)