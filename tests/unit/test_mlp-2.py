import numpy as np
import pytest

from final_ml.supervised_learning.mlp import DenseNetwork


# Single samples for predict/predict_proba (takes xi, not X matrix)
xi_zero = np.array([0.0, 0.0])
xi_one  = np.array([1.0, 1.0])


# ---- __init__ ----

def test_layers_stored():
    mlp = DenseNetwork(layers=[4, 2])
    assert mlp.layers == [4, 2]

def test_single_hidden_layer_stored():
    mlp = DenseNetwork(layers=[8])
    assert mlp.layers == [8]

def test_loss_history_empty_before_training():
    mlp = DenseNetwork(layers=[4])
    assert mlp.loss_history == []


# ---- predict_proba ----

def test_predict_proba_returns_float():
    mlp = DenseNetwork(layers=[4])
    result = mlp.predict_proba(xi_zero)
    assert isinstance(result, float) or np.isscalar(result)

def test_predict_proba_between_zero_and_one():
    mlp = DenseNetwork(layers=[4])
    result = mlp.predict_proba(xi_zero)
    assert 0.0 <= result <= 1.0

def test_predict_proba_same_input_consistent():
    mlp = DenseNetwork(layers=[4])
    p1 = mlp.predict_proba(xi_zero)
    p2 = mlp.predict_proba(xi_zero)
    assert p1 == p2


# ---- predict ----

def test_predict_returns_zero_or_one():
    mlp = DenseNetwork(layers=[4])
    result = mlp.predict(xi_zero)
    assert result in {0, 1}

def test_predict_default_threshold_is_half():
    mlp = DenseNetwork(layers=[4])
    proba = mlp.predict_proba(xi_zero)
    label = mlp.predict(xi_zero)
    expected = 1 if proba >= 0.5 else 0
    assert label == expected

def test_predict_threshold_zero_always_predicts_one():
    mlp = DenseNetwork(layers=[4])
    # threshold=0 means any probability >= 0 → always class 1
    assert mlp.predict(xi_zero, threshold=0.0) == 1

def test_predict_threshold_one_always_predicts_zero():
    mlp = DenseNetwork(layers=[4])
    # threshold=1 means probability must be >= 1 → always class 0
    assert mlp.predict(xi_zero, threshold=1.0) == 0
