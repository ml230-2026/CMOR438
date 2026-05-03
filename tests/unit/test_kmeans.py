import numpy as np
import pytest

from final_ml.unsupervised_learning.kmeans import KMeans


rng = np.random.default_rng(42)
X_a = rng.normal([0.0, 0.0], 0.1, (10, 2))
X_b = rng.normal([10.0, 10.0], 0.1, (10, 2))
X_two = np.vstack([X_a, X_b])

centers = [[0, 0], [10, 0], [0, 10], [10, 10]]
X_four = np.vstack([rng.normal(c, 0.2, (8, 2)) for c in centers])


# ---- __init__ ----

def test_n_clusters_stored():
    km = KMeans(n_clusters=3)
    assert km.n_clusters == 3

def test_cluster_centers_none_before_fit():
    assert KMeans(n_clusters=2).cluster_centers_ is None

def test_inertia_none_before_fit():
    assert KMeans(n_clusters=2).inertia_ is None


# ---- fit ----

def test_fit_returns_self():
    km = KMeans(n_clusters=2, random_state=0)
    assert km.fit(X_two) is km

def test_cluster_centers_shape():
    km = KMeans(n_clusters=2, random_state=0).fit(X_two)
    assert km.cluster_centers_.shape == (2, X_two.shape[1])

def test_inertia_is_positive():
    km = KMeans(n_clusters=2, random_state=0).fit(X_two)
    assert km.inertia_ > 0

def test_four_clusters_centers_shape():
    km = KMeans(n_clusters=4, random_state=0).fit(X_four)
    assert km.cluster_centers_.shape == (4, X_four.shape[1])

def test_deterministic_with_same_random_state():
    km1 = KMeans(n_clusters=2, random_state=5).fit(X_two)
    km2 = KMeans(n_clusters=2, random_state=5).fit(X_two)
    np.testing.assert_array_almost_equal(km1.cluster_centers_, km2.cluster_centers_)


# ---- predict ----

def test_predict_output_shape():
    km = KMeans(n_clusters=2, random_state=0).fit(X_two)
    assert km.predict(X_two).shape == (len(X_two),)

def test_predict_labels_within_range():
    km = KMeans(n_clusters=2, random_state=0).fit(X_two)
    labels = km.predict(X_two)
    assert labels.min() >= 0 and labels.max() < 2

def test_predict_finds_two_distinct_labels():
    km = KMeans(n_clusters=2, random_state=0).fit(X_two)
    assert len(set(km.predict(X_two))) == 2

def test_correct_grouping_of_separated_blobs():
    km = KMeans(n_clusters=2, random_state=0).fit(X_two)
    labels = km.predict(X_two)
    assert len(set(labels[:10])) == 1
    assert len(set(labels[10:])) == 1
    assert labels[0] != labels[10]


# ---- fit_predict ----

def test_fit_predict_matches_fit_then_predict():
    km1 = KMeans(n_clusters=2, random_state=0)
    labels1 = km1.fit_predict(X_two)
    km2 = KMeans(n_clusters=2, random_state=0)
    km2.fit(X_two)
    labels2 = km2.predict(X_two)
    np.testing.assert_array_equal(labels1, labels2)


# ---- score ----

def test_score_is_non_positive():
    km = KMeans(n_clusters=2, random_state=0).fit(X_two)
    assert km.score(X_two) <= 0

def test_more_clusters_lower_inertia():
    km2 = KMeans(n_clusters=2, random_state=0).fit(X_four)
    km4 = KMeans(n_clusters=4, random_state=0).fit(X_four)
    assert km4.inertia_ <= km2.inertia_
