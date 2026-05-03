import numpy as np
import pytest

from final_ml.supervised_learning.random_forest import RandomForestClassifier


rng = np.random.default_rng(7)
X_a = rng.normal([0, 0], 0.3, (20, 2))
X_b = rng.normal([5, 5], 0.3, (20, 2))
X_rf = np.vstack([X_a, X_b])
y_rf = np.array([0] * 20 + [1] * 20)


# ---- __init__ ----

def test_n_trees_stored():
    assert RandomForestClassifier(n_trees=10).n_trees == 10

def test_max_depth_stored():
    assert RandomForestClassifier(max_depth=4).max_depth == 4


# ---- fit ----

def test_correct_number_of_trees_after_fit():
    rf = RandomForestClassifier(n_trees=7)
    rf.fit(X_rf, y_rf)
    assert len(rf.trees) == 7

def test_fit_runs_without_error():
    rf = RandomForestClassifier(n_trees=3)
    rf.fit(X_rf, y_rf)

def test_trees_populated_after_fit():
    rf = RandomForestClassifier(n_trees=4)
    rf.fit(X_rf, y_rf)
    assert rf.trees is not None and len(rf.trees) > 0


# ---- predict ----

def test_predict_output_shape():
    rf = RandomForestClassifier(n_trees=5)
    rf.fit(X_rf, y_rf)
    assert rf.predict(X_rf).shape == (len(X_rf),)

def test_predict_labels_are_valid_classes():
    rf = RandomForestClassifier(n_trees=5)
    rf.fit(X_rf, y_rf)
    assert set(rf.predict(X_rf)).issubset({0, 1})

def test_predict_returns_numpy_array():
    rf = RandomForestClassifier(n_trees=5)
    rf.fit(X_rf, y_rf)
    assert isinstance(rf.predict(X_rf), np.ndarray)


# ---- score ----

def test_score_between_zero_and_one():
    rf = RandomForestClassifier(n_trees=5)
    rf.fit(X_rf, y_rf)
    assert 0.0 <= rf.score(X_rf, y_rf) <= 1.0

def test_score_high_on_separable_data():
    rf = RandomForestClassifier(n_trees=10, max_depth=5)
    rf.fit(X_rf, y_rf)
    assert rf.score(X_rf, y_rf) > 0.9
