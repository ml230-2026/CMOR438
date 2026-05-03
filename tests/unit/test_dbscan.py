import numpy as np
import pytest

from final_ml.unsupervised_learning.dbscan import DBSCAN


# Two tight clusters + one isolated outlier
X_basic = np.array([
    [1.0, 2.0], [1.1, 2.1], [1.2, 2.0],   # cluster 0
    [9.0, 9.0], [9.1, 9.1], [9.2, 9.0],   # cluster 1
    [99.0, 99.0],                           # noise
], dtype=float)

# Single tight cluster with no noise
X_one_cluster = np.array([
    [0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1]
], dtype=float)


# ---- __init__ ----

def test_params_stored():
    db = DBSCAN(eps=0.5, min_samples=3)
    assert db.eps == 0.5
    assert db.min_samples == 3


def test_labels_none_before_fit():
    db = DBSCAN(eps=0.5, min_samples=2)
    assert db.labels_ is None


def test_core_sample_indices_none_before_fit():
    db = DBSCAN(eps=0.5, min_samples=2)
    assert db.core_sample_indices_ is None


# ---- fit ----

def test_fit_returns_self():
    db = DBSCAN(eps=0.5, min_samples=2)
    assert db.fit(X_basic) is db


def test_labels_shape_after_fit():
    db = DBSCAN(eps=0.5, min_samples=2).fit(X_basic)
    assert db.labels_.shape == (len(X_basic),)


def test_finds_two_clusters():
    db = DBSCAN(eps=0.5, min_samples=2).fit(X_basic)
    assert db.n_clusters_ == 2


def test_finds_one_noise_point():
    db = DBSCAN(eps=0.5, min_samples=2).fit(X_basic)
    assert db.n_noise_ == 1


def test_noise_label_is_minus_one():
    db = DBSCAN(eps=0.5, min_samples=2).fit(X_basic)
    assert db.labels_[-1] == -1


def test_cluster_labels_start_at_zero():
    db = DBSCAN(eps=0.5, min_samples=2).fit(X_basic)
    cluster_labels = db.labels_[db.labels_ != -1]
    assert cluster_labels.min() == 0


def test_fit_raises_type_error_for_list():
    with pytest.raises(TypeError):
        DBSCAN(eps=0.5, min_samples=2).fit(X_basic.tolist())


def test_fit_raises_value_error_for_1d():
    with pytest.raises(ValueError):
        DBSCAN(eps=0.5, min_samples=2).fit(np.array([1, 2, 3], dtype=float))


# ---- fit_predict ----

def test_fit_predict_output_shape():
    db = DBSCAN(eps=0.5, min_samples=2)
    labels = db.fit_predict(X_basic)
    assert labels.shape == (len(X_basic),)


def test_fit_predict_matches_fit_then_labels():
    db1 = DBSCAN(eps=0.5, min_samples=2)
    labels1 = db1.fit_predict(X_basic)

    db2 = DBSCAN(eps=0.5, min_samples=2)
    db2.fit(X_basic)
    labels2 = db2.labels_

    np.testing.assert_array_equal(labels1, labels2)


# ---- n_clusters_ and n_noise_ ----

def test_n_clusters_and_n_noise_consistent():
    db = DBSCAN(eps=0.5, min_samples=2).fit(X_basic)
    assert db.n_clusters_ == len(set(db.labels_[db.labels_ != -1]))
    assert db.n_noise_ == int(np.sum(db.labels_ == -1))


def test_one_tight_cluster_no_noise():
    db = DBSCAN(eps=0.5, min_samples=2).fit(X_one_cluster)
    assert db.n_clusters_ == 1
    assert db.n_noise_ == 0


def test_large_eps_merges_everything():
    """With eps large enough all points join one cluster."""
    db = DBSCAN(eps=200.0, min_samples=2).fit(X_basic)
    assert db.n_clusters_ == 1
    assert db.n_noise_ == 0


def test_tiny_eps_makes_everything_noise():
    """With eps near zero no point has neighbors → all noise."""
    db = DBSCAN(eps=1e-9, min_samples=2).fit(X_basic)
    assert db.n_clusters_ == 0
    assert db.n_noise_ == len(X_basic)


# ---- core_sample_indices_ ----

def test_core_sample_indices_subset_of_all_indices():
    db = DBSCAN(eps=0.5, min_samples=2).fit(X_basic)
    all_idx = set(range(len(X_basic)))
    assert set(db.core_sample_indices_).issubset(all_idx)


def test_outlier_not_in_core_samples():
    db = DBSCAN(eps=0.5, min_samples=2).fit(X_basic)
    # Last point (index 6) is the outlier
    assert 6 not in db.core_sample_indices_
