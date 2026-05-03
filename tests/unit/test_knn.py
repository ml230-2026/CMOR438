import numpy as np
import pytest

from final_ml.supervised_learning.knn import KNNClassifier


X_clf = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_clf = np.array([0, 0, 1, 1])


# ---- __init__ ----

def test_k_stored():
    assert KNNClassifier(k=3).k == 3

def test_default_k_is_five():
    assert KNNClassifier().k == 5


# ---- fit ----

def test_fit_runs_without_error():
    clf = KNNClassifier(k=3)
    clf.fit(X_clf, y_clf)

def test_predict_works_after_fit():
    clf = KNNClassifier(k=1)
    clf.fit(X_clf, y_clf)
    preds = clf.predict(X_clf)
    assert preds is not None


# ---- predict ----

def test_predict_output_shape():
    clf = KNNClassifier(k=3)
    clf.fit(X_clf, y_clf)
    assert clf.predict(X_clf).shape == (len(X_clf),)

def test_predict_correct_on_simple_data():
    clf = KNNClassifier(k=3)
    clf.fit(X_clf, y_clf)
    preds = clf.predict(np.array([[0.1, 0.1], [0.9, 0.9]]))
    assert preds.tolist() == [0, 1]

def test_predict_perfect_with_k1():
    clf = KNNClassifier(k=1)
    clf.fit(X_clf, y_clf)
    np.testing.assert_array_equal(clf.predict(X_clf), y_clf)

def test_labels_are_valid_classes():
    clf = KNNClassifier(k=1)
    clf.fit(X_clf, y_clf)
    assert set(clf.predict(X_clf)).issubset({0, 1})

def test_bottom_left_predicts_zero():
    clf = KNNClassifier(k=1)
    clf.fit(X_clf, y_clf)
    assert clf.predict(np.array([[0.05, 0.05]]))[0] == 0

def test_top_right_predicts_one():
    clf = KNNClassifier(k=1)
    clf.fit(X_clf, y_clf)
    assert clf.predict(np.array([[0.95, 0.95]]))[0] == 1


# ---- score ----

def test_score_perfect_with_k1():
    clf = KNNClassifier(k=1)
    clf.fit(X_clf, y_clf)
    assert clf.score(X_clf, y_clf) == 1.0

def test_score_between_zero_and_one():
    clf = KNNClassifier(k=3)
    clf.fit(X_clf, y_clf)
    assert 0.0 <= clf.score(X_clf, y_clf) <= 1.0
