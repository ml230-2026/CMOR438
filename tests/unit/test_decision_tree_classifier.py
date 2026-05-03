import numpy as np
import pytest

from final_ml.supervised_learning.decision_tree import DecisionTreeClassifier


X_simple = np.array([[1], [2], [8], [9]], dtype=float)
y_simple = np.array([0, 0, 1, 1])

X_multi = np.array([[1, 2], [1, 3], [5, 1], [6, 2]], dtype=float)
y_multi = np.array([0, 0, 1, 1])


# ---- __init__ ----

def test_max_depth_stored():
    dt = DecisionTreeClassifier(max_depth=5)
    assert dt.max_depth == 5

def test_min_samples_split_stored():
    dt = DecisionTreeClassifier(min_samples_split=3)
    assert dt.min_samples_split == 3


# ---- fit ----

def test_root_not_none_after_fit():
    dt = DecisionTreeClassifier(max_depth=3)
    dt.fit(X_simple, y_simple)
    assert dt.root is not None


# ---- predict ----

def test_predict_output_shape():
    dt = DecisionTreeClassifier(max_depth=3)
    dt.fit(X_simple, y_simple)
    assert dt.predict(X_simple).shape == (len(X_simple),)

def test_predict_correct_on_simple_data():
    dt = DecisionTreeClassifier(max_depth=5)
    dt.fit(X_simple, y_simple)
    np.testing.assert_array_equal(dt.predict(X_simple), y_simple)

def test_predict_correct_on_multi_feature_data():
    dt = DecisionTreeClassifier(max_depth=5)
    dt.fit(X_multi, y_multi)
    np.testing.assert_array_equal(dt.predict(X_multi), y_multi)

def test_predict_new_point_left_class():
    dt = DecisionTreeClassifier(max_depth=3)
    dt.fit(X_simple, y_simple)
    assert dt.predict(np.array([[1.5]]))[0] == 0

def test_predict_new_point_right_class():
    dt = DecisionTreeClassifier(max_depth=3)
    dt.fit(X_simple, y_simple)
    assert dt.predict(np.array([[8.5]]))[0] == 1

def test_labels_are_valid_classes():
    dt = DecisionTreeClassifier(max_depth=3)
    dt.fit(X_simple, y_simple)
    assert set(dt.predict(X_simple)).issubset({0, 1})


# ---- score ----

def test_score_perfect_on_simple_data():
    dt = DecisionTreeClassifier(max_depth=5)
    dt.fit(X_simple, y_simple)
    assert dt.score(X_simple, y_simple) == 1.0

def test_score_between_zero_and_one():
    dt = DecisionTreeClassifier(max_depth=3)
    dt.fit(X_simple, y_simple)
    assert 0.0 <= dt.score(X_simple, y_simple) <= 1.0

def test_depth_one_stump_still_predicts():
    dt = DecisionTreeClassifier(max_depth=1)
    dt.fit(X_simple, y_simple)
    assert dt.predict(X_simple).shape == (len(X_simple),)
