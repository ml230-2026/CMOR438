import numpy as np


def distance(p, q):
    """
    Compute the Euclidean distance between two points p and q.

    Euclidean distance is the straight-line distance between two points
    in n-dimensional space:

        d(p, q) = sqrt((p - q)^T (p - q))

    Parameters
    ----------
    p : numpy array — first point
    q : numpy array — second point

    Returns
    -------
    float — the distance between p and q
    """
    return np.sqrt((p - q) @ (p - q))


def k_nearest_neighbors(point, training_features, training_labels, k):
    """
    Find the k nearest neighbors to a given point in the training data.

    For each training example, we compute its distance to the query point,
    collect all (feature, label, distance) triples, sort by distance,
    and return the k closest ones.

    Parameters
    ----------
    point             : numpy array — the query point we want neighbors for
    training_features : numpy array of shape (num_samples, num_features)
    training_labels   : numpy array of shape (num_samples,)
    k                 : int — number of neighbors to return

    Returns
    -------
    list of [feature_vector, label, distance] — the k nearest neighbors,
    sorted from closest to farthest
    """
    neighbors = []

    for p, label in zip(training_features, training_labels):
        d = distance(point, p)
        neighbors.append([p, label, d])

    # Sort by distance (smallest first)
    neighbors.sort(key=lambda x: x[-1])

    return neighbors[:k]


def KNN_Predict(point, training_features, training_labels, k, regression=False):
    """
    Predict the label for a single query point using KNN.

    For classification: returns the most common label among the k neighbors (mode).
    For regression:     returns the average label among the k neighbors (mean).

    Parameters
    ----------
    point             : numpy array — the query point
    training_features : numpy array of shape (num_samples, num_features)
    training_labels   : numpy array of shape (num_samples,)
    k                 : int — number of neighbors to use
    regression        : bool — if True, return mean (regression); else return mode (classification)

    Returns
    -------
    predicted label (class or number)
    """
    neighbors = k_nearest_neighbors(point, training_features, training_labels, k)

    if not regression:
        # Classification: return the most common label (mode)
        labels = [x[1] for x in neighbors]
        return max(labels, key=labels.count)
    else:
        # Regression: return the average label (mean)
        return sum(x[1] for x in neighbors) / k


def classification_error(test_features, test_labels, training_features, training_labels, k):
    """
    Compute the classification error rate on a test set.

    Error rate = (number of wrong predictions) / (total predictions)

    A perfect model has error rate 0.0.
    A model that is always wrong has error rate 1.0.

    Parameters
    ----------
    test_features     : numpy array — features of the test set
    test_labels       : numpy array — true labels of the test set
    training_features : numpy array — features of the training set
    training_labels   : numpy array — labels of the training set
    k                 : int — number of neighbors

    Returns
    -------
    float — error rate between 0.0 and 1.0
    """
    error = 0
    for point, label in zip(test_features, test_labels):
        prediction = KNN_Predict(point, training_features, training_labels, k)
        error += int(label != prediction)
    return error / len(test_features)


class KNNClassifier:
    """
    K-Nearest Neighbors Classifier.

    A nonparametric algorithm that classifies a new point based on
    the majority label among its k closest training examples.

    Unlike parametric models (Perceptron, Logistic Regression, MLP),
    KNN has NO training phase — it simply memorizes the training data
    and does all computation at prediction time.

    Parameters
    ----------
    k : int
        Number of nearest neighbors to use for voting. Default is 5.
        Use an odd number to avoid ties in binary classification.
    """

    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None


    def fit(self, X, y):
        """
        'Train' the model — for KNN, this simply stores the training data.

        There are no weights to learn. All computation happens at predict time.

        Parameters
        ----------
        X : numpy array of shape (num_samples, num_features)
        y : numpy array of shape (num_samples,)
        """
        self.X_train = X
        self.y_train = y


    def predict(self, X):
        """
        Predict class labels for all samples in X.

        For each query point, find its k nearest neighbors in the training set
        and return the most common label among them.

        Parameters
        ----------
        X : numpy array of shape (num_samples, num_features)

        Returns
        -------
        numpy array of predicted labels, shape (num_samples,)
        """
        predictions = []
        for point in X:
            pred = KNN_Predict(point, self.X_train, self.y_train, self.k)
            predictions.append(pred)
        return np.array(predictions)


    def score(self, X, y):
        """
        Compute classification accuracy on a dataset.

        Accuracy = 1 - error rate = (correct predictions) / (total predictions)

        Parameters
        ----------
        X : numpy array of shape (num_samples, num_features)
        y : numpy array of shape (num_samples,) — true labels

        Returns
        -------
        float — accuracy between 0.0 and 1.0
        """
        predictions = self.predict(X)
        return np.sum(predictions == y) / len(y)
