import numpy as np
import pytest

from final_ml.supervised_learning.perceptron import Perceptron


X_sep = np.array([[1, 1], [2, 2], [-1, -1], [-2, -2]], dtype=float)
y_sep = np.array([1, 1, -1, -1])


# ---- __init__ ----

def test_learning_rate_stored():
    p = Perceptron(learning_rate=0.05, n_epochs=20)
    assert p.learning_rate == 0.05

def test_n_epochs_stored():
    p = Perceptron(learning_rate=0.1, n_epochs=20)
    assert p.n_epochs == 20

def test_weights_none_before_fit():
    assert Perceptron().weights is None

def test_bias_none_before_fit():
    assert Perceptron().bias is None

def test_errors_per_epoch_empty_before_fit():
    assert Perceptron().errors_per_epoch == []


# ---- fit ----

def test_weights_shape_after_fit():
    p = Perceptron(learning_rate=0.1, n_epochs=10)
    p.fit(X_sep, y_sep)
    assert p.weights.shape == (X_sep.shape[1],)

def test_errors_per_epoch_length_matches_n_epochs():
    p = Perceptron(learning_rate=0.1, n_epochs=30)
    p.fit(X_sep, y_sep)
    assert len(p.errors_per_epoch) == 30

def test_converges_to_zero_errors_on_separable_data():
    p = Perceptron(learning_rate=0.1, n_epochs=200)
    p.fit(X_sep, y_sep)
    assert p.errors_per_epoch[-1] == 0

def test_errors_non_negative():
    p = Perceptron(learning_rate=0.1, n_epochs=50)
    p.fit(X_sep, y_sep)
    assert all(e >= 0 for e in p.errors_per_epoch)


# ---- predict ----

def test_predict_output_shape():
    p = Perceptron(learning_rate=0.1, n_epochs=100)
    p.fit(X_sep, y_sep)
    assert p.predict(X_sep).shape == (len(X_sep),)

def test_predict_labels_are_minus_one_or_one():
    p = Perceptron(learning_rate=0.1, n_epochs=100)
    p.fit(X_sep, y_sep)
    assert set(p.predict(X_sep)).issubset({-1, 1})

def test_predict_correct_on_separable_data():
    p = Perceptron(learning_rate=0.1, n_epochs=200)
    p.fit(X_sep, y_sep)
    np.testing.assert_array_equal(p.predict(X_sep), y_sep)

def test_positive_point_predicts_one():
    p = Perceptron(learning_rate=0.1, n_epochs=200)
    p.fit(X_sep, y_sep)
    assert p.predict(np.array([[3, 3]]))[0] == 1

def test_negative_point_predicts_minus_one():
    p = Perceptron(learning_rate=0.1, n_epochs=200)
    p.fit(X_sep, y_sep)
    assert p.predict(np.array([[-3, -3]]))[0] == -1


# ---- score ----

def test_score_perfect_on_training_data():
    p = Perceptron(learning_rate=0.1, n_epochs=200)
    p.fit(X_sep, y_sep)
    assert p.score(X_sep, y_sep) == 1.0

def test_score_between_zero_and_one():
    p = Perceptron(learning_rate=0.1, n_epochs=100)
    p.fit(X_sep, y_sep)
    assert 0.0 <= p.score(X_sep, y_sep) <= 1.0
