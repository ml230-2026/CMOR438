import numpy as np


# ── Activation functions ──────────────────────────────────────────────────────

def sigmoid(z):
    """
    The sigmoid activation function.
    Squashes any real number into the range (0, 1).

        sigma(z) = 1 / (1 + exp(-z))
    """
    return 1.0 / (1.0 + np.exp(-z))


def d_sigmoid(z):
    """
    The derivative of the sigmoid function.
    Used during backpropagation to compute gradients.

        sigma'(z) = sigma(z) * (1 - sigma(z))
    """
    return sigmoid(z) * (1.0 - sigmoid(z))


# ── Helper functions ──────────────────────────────────────────────────────────

def initialize_weights(layers):
    """
    Randomly initialize the weight matrices and bias vectors for each layer.

    We use the He initialization scaling factor sqrt(2 / n_in) to help
    gradients flow well during early training.

    Parameters
    ----------
    layers : list of int
        Number of nodes in each layer, including input and output.
        Example: [9, 64, 32, 1] means 9 inputs, two hidden layers, 1 output.

    Returns
    -------
    W : list of numpy arrays
        W[i] is the weight matrix connecting layer i-1 to layer i.
        W[0] is a placeholder (not used).
    B : list of numpy arrays
        B[i] is the bias vector for layer i.
        B[0] is a placeholder (not used).
    """
    W = [[0.0]]  # placeholder for index 0
    B = [[0.0]]  # placeholder for index 0

    for i in range(1, len(layers)):
        scale = np.sqrt(2 / layers[i - 1])
        w_temp = np.random.randn(layers[i], layers[i - 1]) * scale
        b_temp = np.random.randn(layers[i], 1) * scale
        W.append(w_temp)
        B.append(b_temp)

    return W, B


def forward_pass(W, B, xi):
    """
    Run a single input xi through the network (feedforward phase).

    For each layer l:
        z^l = W^l @ a^(l-1) + b^l      (pre-activation)
        a^l = sigmoid(z^l)              (post-activation)

    Parameters
    ----------
    W : list of weight matrices
    B : list of bias vectors
    xi : numpy array of shape (num_features, 1)
        A single input feature vector (must be a column vector).

    Returns
    -------
    Z : list of pre-activation values per layer
    A : list of post-activation values per layer (A[0] = xi)
    """
    Z = [[0.0]]   # placeholder for index 0
    A = [xi]      # A[0] is the input itself

    L = len(W) - 1  # number of non-input layers

    for i in range(1, L + 1):
        z = W[i] @ A[i - 1] + B[i]   # pre-activation
        Z.append(z)
        a = sigmoid(z)                 # post-activation
        A.append(a)

    return Z, A


def mse_single(a, y):
    """
    Mean Squared Error for a single training example.

        MSE = 0.5 * sum((a_k - y_k)^2)

    Parameters
    ----------
    a : numpy array — network output (predictions)
    y : numpy array — true label (column vector)

    Returns
    -------
    float
    """
    return 0.5 * float(np.sum((a - y) ** 2))


def MSE(W, B, X, y):
    """
    Average MSE over an entire dataset.

    Parameters
    ----------
    W, B : weight matrices and bias vectors
    X : list of input column vectors
    y : list of target column vectors

    Returns
    -------
    float
    """
    total = 0.0
    for xi, yi in zip(X, y):
        a = forward_pass(W, B, xi)[1][-1]
        total += mse_single(a, yi)
    return total / len(X)


# ── Main class ────────────────────────────────────────────────────────────────

class DenseNetwork:
    """
    A fully connected (dense) multilayer neural network trained with
    Stochastic Gradient Descent and Backpropagation.

    Architecture
    ------------
    - Input layer:   one node per feature
    - Hidden layers: arbitrary number and size (defined by 'layers')
    - Output layer:  one node per output class

    All layers use the sigmoid activation function.
    Loss function is Mean Squared Error (MSE).

    Parameters
    ----------
    layers : list of int
        Number of nodes in each layer including input and output.
        Example: [9, 64, 32, 1] means 9 inputs, two hidden layers, 1 output.
    """

    def __init__(self, layers):
        self.layers = layers
        self.W, self.B = initialize_weights(layers)
        self.loss_history = []


    def train(self, X_train, y_train, alpha=0.01, epochs=10):
        """
        Train the network using Stochastic Gradient Descent + Backpropagation.

        Backpropagation algorithm:
        1. For each training example (xi, yi):
        2.   Feedforward xi through the network → get Z, A at each layer
        3.   Compute output error:
                 delta^L = (a^L - yi) * sigma'(z^L)
        4.   Backpropagate error through hidden layers:
                 delta^l = (W^(l+1)^T @ delta^(l+1)) * sigma'(z^l)
        5.   Update weights and biases:
                 W^l  ←  W^l  - alpha * delta^l @ a^(l-1)^T
                 b^l  ←  b^l  - alpha * delta^l

        Parameters
        ----------
        X_train : list of numpy arrays of shape (num_features, 1)
        y_train : list of numpy arrays of shape (num_outputs, 1)
        alpha   : float — learning rate
        epochs  : int   — number of full passes through the training data
        """

        L = len(self.layers) - 1   # number of non-input layers

        # Record starting loss before any training
        starting_loss = MSE(self.W, self.B, X_train, y_train)
        self.loss_history.append(starting_loss)
        print(f'Starting MSE = {starting_loss:.4f}')

        for epoch in range(epochs):

            # Loop over every single training example (SGD)
            for xi, yi in zip(X_train, y_train):

                # ── Step 1: Feedforward ───────────────────────────────────────
                Z, A = forward_pass(self.W, self.B, xi)

                # ── Step 2: Compute output layer error (delta) ────────────────
                deltas = {}
                deltas[L] = (A[L] - yi) * d_sigmoid(Z[L])

                # ── Step 3: Backpropagate through hidden layers ────────────────
                for i in range(L - 1, 0, -1):
                    deltas[i] = (self.W[i + 1].T @ deltas[i + 1]) * d_sigmoid(Z[i])

                # ── Step 4: Update weights and biases ─────────────────────────
                for i in range(1, L + 1):
                    self.W[i] -= alpha * deltas[i] @ A[i - 1].T
                    self.B[i] -= alpha * deltas[i]

            # Record MSE after this epoch
            epoch_loss = MSE(self.W, self.B, X_train, y_train)
            self.loss_history.append(epoch_loss)
            print(f'Epoch {epoch + 1} MSE = {epoch_loss:.4f}')


    def predict_proba(self, xi):
        """
        Return the raw output of the network for a single input xi.
        For a binary classification problem with 1 output node,
        this is the predicted probability of class 1.

        Parameters
        ----------
        xi : numpy array of shape (num_features, 1)

        Returns
        -------
        float — predicted probability
        """
        _, A = forward_pass(self.W, self.B, xi)
        return float(A[-1].flatten()[0])


    def predict(self, xi, threshold=0.5):
        """
        Return a binary class label (0 or 1) for a single input xi.

        Parameters
        ----------
        xi        : numpy array of shape (num_features, 1)
        threshold : float — decision boundary (default 0.5)

        Returns
        -------
        int — 0 or 1
        """
        return int(self.predict_proba(xi) >= threshold)


    def accuracy(self, X, y_labels, threshold=0.5):
        """
        Compute classification accuracy over a dataset.

        Parameters
        ----------
        X        : list of input column vectors
        y_labels : list of true binary labels (0 or 1, not one-hot)
        threshold: float — decision boundary

        Returns
        -------
        float — accuracy between 0 and 1
        """
        correct = sum(
            int(self.predict(xi, threshold) == int(yi))
            for xi, yi in zip(X, y_labels)
        )
        return correct / len(y_labels)
