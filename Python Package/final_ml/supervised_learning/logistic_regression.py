import numpy as np


class LogisticRegression:
    """
    Logistic Regression using Stochastic Gradient Descent.

    Unlike Linear Regression which predicts a number, Logistic Regression
    predicts a PROBABILITY between 0 and 1, then converts that to a class
    label (0 or 1) using a decision threshold.

    The key idea is the sigmoid function, which squashes any real number
    into the range (0, 1):

        sigma(z) = 1 / (1 + exp(-z))

    The full prediction pipeline is:
        z      = w * x + b          (linear combination, same as linear regression)
        y_hat  = sigma(z)           (squash to probability using sigmoid)
        label  = 1 if y_hat >= 0.5 else 0   (convert probability to class)

    Parameters
    ----------
    learning_rate : float
        How big of a step to take each update. Default is 0.01.
    epochs : int
        How many times to loop through the entire training dataset. Default is 50.
    threshold : float
        The probability cutoff for predicting class 1. Default is 0.5.
    """

    def __init__(self, learning_rate=0.01, epochs=50, threshold=0.5):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.threshold = threshold

        # These will be set when we call fit()
        self.weights = None
        self.bias = None
        self.loss_history = []  # tracks cross-entropy loss after each epoch


    def _sigmoid(self, z):
        """
        The sigmoid activation function.

        Squashes any real number into the range (0, 1).
        This is what turns our linear output into a probability.

            sigma(z) = 1 / (1 + exp(-z))

        """
        return 1 / (1 + np.exp(-z))


    def fit(self, X, y):
        """
        Train the model on input data X and binary target labels y.

        X should be a 2D array of shape (num_samples, num_features).
        y should be a 1D array of shape (num_samples,) with values 0 or 1.
        """

        num_samples = X.shape[0]
        num_features = X.shape[1]

        # Start weights and bias at zero
        self.weights = np.zeros(num_features)
        self.bias = 0.0

        # Loop through the data 'epochs' number of times
        for epoch in range(self.epochs):

            # Track total cross-entropy loss for this epoch
            total_loss = 0.0

            # Loop through every single training example one at a time (SGD)
            for i in range(num_samples):

                # Step 1: Get this single training example
                xi = X[i]   # one feature vector
                yi = y[i]   # the correct label (0 or 1)

                # Step 2: Compute the linear combination
                z = np.dot(xi, self.weights) + self.bias

                # Step 3: Apply the sigmoid to get a probability
                y_hat = self._sigmoid(z)

                # Step 4: Compute the error
                error = y_hat - yi

                # Step 5: Update the weights using gradient descent
                # The gradient of cross-entropy loss w.r.t. w is: error * xi
                self.weights = self.weights - self.learning_rate * error * xi

                # Step 6: Update the bias
                # The gradient of cross-entropy loss w.r.t. b is: error
                self.bias = self.bias - self.learning_rate * error

                # Step 7: Accumulate binary cross-entropy loss for this example
                # We clip y_hat to avoid log(0) which is undefined
                y_hat_clipped = np.clip(y_hat, 1e-15, 1 - 1e-15)
                total_loss += -(yi * np.log(y_hat_clipped) + (1 - yi) * np.log(1 - y_hat_clipped))

            # Record average loss for this epoch
            self.loss_history.append(total_loss / num_samples)


    def predict_proba(self, X):
        """
        Return the predicted probability of class 1 for each sample.

        X should be a 2D array of shape (num_samples, num_features).
        Returns a 1D array of probabilities between 0 and 1.
        """

        # Compute linear combination then apply sigmoid
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)


    def predict(self, X):
        """
        Return the predicted class label (0 or 1) for each sample.

        Converts probabilities to labels using the decision threshold.
        Default threshold is 0.5 — if probability >= 0.5, predict class 1.

        X should be a 2D array of shape (num_samples, num_features).
        Returns a 1D array of 0s and 1s.
        """

        probabilities = self.predict_proba(X)
        return (probabilities >= self.threshold).astype(int)


    def accuracy(self, X, y):
        """
        Calculate the accuracy of the model on a dataset.

        Accuracy = (number of correct predictions) / (total predictions)

        1.0 = perfect, 0.0 = wrong every time.
        """

        predictions = self.predict(X)
        correct = np.sum(predictions == y)
        return correct / len(y)


    def binary_cross_entropy(self, X, y):
        """
        Calculate the Binary Cross-Entropy loss on a dataset.

        This is the standard loss function for binary classification:

            L = -(1/N) * sum(y * log(y_hat) + (1 - y) * log(1 - y_hat))

        Lower is better. 0 would mean perfect confidence in correct answers.
        """

        probabilities = self.predict_proba(X)
        probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)
        loss = -np.mean(y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities))
        return loss
