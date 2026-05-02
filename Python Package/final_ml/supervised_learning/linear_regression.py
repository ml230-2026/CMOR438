import numpy as np


class LinearRegression:
    """
    Linear Regression using Stochastic Gradient Descent.

    Predicts a continuous number by fitting a straight line to the data.
    The line equation is:  y_hat = w * x + b

    Parameters
    ----------
    learning_rate : float
        How big of a step to take each update. Default is 0.01.
    epochs : int
        How many times to loop through the entire training dataset. Default is 50.
    """

    def __init__(self, learning_rate=0.01, epochs=50):
        self.learning_rate = learning_rate
        self.epochs = epochs

        # These will be set when we call fit()
        self.weights = None     # one weight per feature
        self.bias = None        # the intercept
        self.loss_history = []  # tracks MSE loss after each epoch so we can plot it later


    def fit(self, X, y):
        """
        Train the model on input data X and target labels y.

        X should be a 2D array of shape (num_samples, num_features).
        y should be a 1D array of shape (num_samples,).
        """

        # Figure out how many training examples and features we have
        num_samples = X.shape[0]
        num_features = X.shape[1]

        # Start weights at zero and bias at zero
        self.weights = np.zeros(num_features)
        self.bias = 0.0

        # Loop through the data 'epochs' number of times
        for epoch in range(self.epochs):

            # Keep track of total squared error for this epoch
            total_error = 0.0

            # Loop through every single training example one at a time
            # This is called Stochastic Gradient Descent (SGD)
            for i in range(num_samples):

                # Step 1: Get this single training example
                xi = X[i]   # one feature vector
                yi = y[i]   # the correct label for this example

                # Step 2: Make a prediction with current weights and bias
                # prediction = w1*x1 + w2*x2 + ... + b
                y_hat = np.dot(xi, self.weights) + self.bias

                # Step 3: Calculate the error (how wrong were we?)
                error = y_hat - yi

                # Step 4: Update the weights using the gradient descent rule
                # Gradient of MSE with respect to w is: error * xi
                # We subtract because we want to move downhill (reduce the loss)
                self.weights = self.weights - self.learning_rate * error * xi

                # Step 5: Update the bias using the gradient descent rule
                # Gradient of MSE with respect to b is just: error
                self.bias = self.bias - self.learning_rate * error

                # Add this example's squared error to the running total
                total_error += 0.5 * (error ** 2)

            # After going through all examples, record average MSE for this epoch
            mse = total_error / num_samples
            self.loss_history.append(mse)


    def predict(self, X):
        """
        Use the trained weights and bias to make predictions on new data.

        X should be a 2D array of shape (num_samples, num_features).
        Returns a 1D array of predicted values.
        """

        # For each row in X, compute: prediction = w1*x1 + w2*x2 + ... + b
        predictions = np.dot(X, self.weights) + self.bias
        return predictions


    def mean_squared_error(self, X, y):
        """
        Calculate the Mean Squared Error on a dataset.

        MSE = (1/N) * sum( (y_hat - y)^2 )

        Lower is better. 0 would mean perfect predictions.
        """

        predictions = self.predict(X)
        squared_errors = (predictions - y) ** 2
        mse = np.mean(squared_errors)
        return mse


    def r_squared(self, X, y):
        """
        Calculate R-squared (coefficient of determination).

        R-squared tells us how much of the variance in y our model explains.
          - R2 = 1.0  means perfect predictions
          - R2 = 0.0  means no better than just predicting the mean
          - R2 < 0    means worse than predicting the mean
        """

        predictions = self.predict(X)

        # Total variance in the true labels (how spread out y actually is)
        total_variance = np.sum((y - np.mean(y)) ** 2)

        # Variance our model still couldn't explain (the leftover residual errors)
        residual_variance = np.sum((y - predictions) ** 2)

        # R2 = 1 - (what we couldn't explain / total variance)
        r2 = 1 - (residual_variance / total_variance)
        return r2