import numpy as np
import pytest

from final_ml.supervised_learning.logistic_regression import LogisticRegression


rng = np.random.default_rng(42)
X_a = rng.normal(loc=[0.0, 0.0], scale=0.3, size=(20, 2))
X_b = rng.normal(loc=[4.0, 4.0], scale=0.3, size=(20, 2))
X_bin = np.vstack([X_a, X_b])
y_bin = np.array([0] * 20 + [1] * 20)


# ---- __init__ ----

def test_learning_rate_stored():
    lr = LogisticRegression(learning_rate=0.05, epochs=100, threshold=0.4)
    assert lr.learning_rate == 0.05

def test_threshold_stored():
    lr = LogisticRegression(threshold=0.4)
    assert lr.threshold == 0.4

def test_weights_none_before_fit():
    assert LogisticRegression().weights is None

def test_bias_none_before_fit():
    assert LogisticRegression().bias is None

def test_loss_history_empty_before_fit():
    assert LogisticRegression().loss_history == []


# ---- fit ----

def test_weights_shape_after_fit():
    lr = LogisticRegression(learning_rate=0.1, epochs=10)
    lr.fit(X_bin, y_bin)
    assert lr.weights.shape == (X_bin.shape[1],)

def test_loss_history_length_matches_epochs():
    lr = LogisticRegression(learning_rate=0.1, epochs=50)
    lr.fit(X_bin, y_bin)
    assert len(lr.loss_history) == 50

def test_loss_decreases_over_training():
    lr = LogisticRegression(learning_rate=0.1, epochs=300)
    lr.fit(X_bin, y_bin)
    assert lr.loss_history[-1] < lr.loss_history[0]


# ---- predict_proba ----

def test_predict_proba_shape():
    lr = LogisticRegression(learning_rate=0.1, epochs=100)
    lr.fit(X_bin, y_bin)
    assert lr.predict_proba(X_bin).shape == (len(X_bin),)

def test_predict_proba_between_zero_and_one():
    lr = LogisticRegression(learning_rate=0.1, epochs=100)
    lr.fit(X_bin, y_bin)
    proba = lr.predict_proba(X_bin)
    assert np.all(proba >= 0) and np.all(proba <= 1)


# ---- predict ----

def test_predict_output_shape():
    lr = LogisticRegression(learning_rate=0.1, epochs=100)
    lr.fit(X_bin, y_bin)
    assert lr.predict(X_bin).shape == (len(X_bin),)

def test_predict_labels_are_zero_or_one():
    lr = LogisticRegression(learning_rate=0.1, epochs=100)
    lr.fit(X_bin, y_bin)
    assert set(lr.predict(X_bin)).issubset({0, 1})

def test_predict_high_accuracy_on_separable_data():
    lr = LogisticRegression(learning_rate=0.5, epochs=500)
    lr.fit(X_bin, y_bin)
    assert lr.accuracy(X_bin, y_bin) > 0.95


# ---- accuracy ----

def test_accuracy_between_zero_and_one():
    lr = LogisticRegression(learning_rate=0.1, epochs=100)
    lr.fit(X_bin, y_bin)
    assert 0.0 <= lr.accuracy(X_bin, y_bin) <= 1.0

def test_accuracy_near_one_after_convergence():
    lr = LogisticRegression(learning_rate=0.5, epochs=500)
    lr.fit(X_bin, y_bin)
    assert lr.accuracy(X_bin, y_bin) > 0.95
