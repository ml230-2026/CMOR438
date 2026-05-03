import numpy as np
import pytest

from final_ml.unsupervised_learning.pca import PCA


rng = np.random.default_rng(0)
X_pca = rng.standard_normal((10, 4))

X_1d = np.column_stack([
    np.linspace(0, 10, 20),
    np.linspace(0, 10, 20) + rng.normal(0, 0.01, 20),
    rng.normal(0, 0.01, 20),
])


# ---- __init__ ----

def test_n_components_stored():
    pca = PCA(n_components=2)
    assert pca.n_components == 2

def test_components_none_before_fit():
    assert PCA(n_components=2).components_ is None

def test_explained_variance_none_before_fit():
    assert PCA(n_components=2).explained_variance_ is None


# ---- fit ----

def test_fit_returns_self():
    pca = PCA(n_components=2)
    assert pca.fit(X_pca) is pca

def test_components_shape_after_fit():
    pca = PCA(n_components=2).fit(X_pca)
    assert pca.components_.shape == (2, X_pca.shape[1])

def test_explained_variance_shape():
    pca = PCA(n_components=2).fit(X_pca)
    assert pca.explained_variance_.shape == (2,)

def test_explained_variance_ratio_leq_one():
    pca = PCA(n_components=2).fit(X_pca)
    assert pca.explained_variance_ratio_.sum() <= 1.0 + 1e-9

def test_explained_variance_ratio_non_negative():
    pca = PCA(n_components=2).fit(X_pca)
    assert np.all(pca.explained_variance_ratio_ >= 0)

def test_components_are_orthogonal():
    pca = PCA(n_components=2).fit(X_pca)
    dot = pca.components_[0] @ pca.components_[1]
    assert abs(dot) < 1e-9


# ---- transform ----

def test_transform_output_shape():
    pca = PCA(n_components=2).fit(X_pca)
    assert pca.transform(X_pca).shape == (X_pca.shape[0], 2)

def test_transform_reduces_dimensions():
    pca = PCA(n_components=2).fit(X_pca)
    X_t = pca.transform(X_pca)
    assert X_t.shape[1] == 2


# ---- fit_transform ----

def test_fit_transform_output_shape():
    pca = PCA(n_components=3)
    X_t = pca.fit_transform(X_pca)
    assert X_t.shape == (X_pca.shape[0], 3)

def test_fit_transform_matches_fit_then_transform():
    pca1 = PCA(n_components=2)
    X_t1 = pca1.fit_transform(X_pca)
    pca2 = PCA(n_components=2)
    pca2.fit(X_pca)
    X_t2 = pca2.transform(X_pca)
    np.testing.assert_allclose(np.abs(X_t1), np.abs(X_t2), atol=1e-9)

def test_first_component_captures_most_variance():
    pca = PCA(n_components=2).fit(X_1d)
    assert pca.explained_variance_ratio_[0] > 0.95
