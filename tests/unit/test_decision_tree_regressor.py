import numpy as np
import pytest

from final_ml.supervised_learning.regression_tree import RegressionTreeRegressor


X_reg = np.array([[1], [2], [3], [8], [9], [10]], dtype=float)
y_reg = 2 * X_reg.ravel()   # [2, 4, 6, 16, 18, 20]


# ---- __init__ ----

def test_max_depth_stored():
    dt = RegressionTreeRegressor(max_depth=4)
    assert dt.max_depth == 4

def test_min_samples_split_stored():
    dt = RegressionTreeRegressor(min_samples_split=2)
    assert dt.min_samples_split == 2


# ---- fit ----

def test_root_not_none_after_fit():
    dt = RegressionTreeRegressor(max_depth=3)
    dt.fit(X_reg, y_reg)
    assert dt.root is not None


# ---- predict ----

def test_predict_output_shape():
    dt = RegressionTreeRegressor(max_depth=5)
    dt.fit(X_reg, y_reg)
    assert dt.predict(X_reg).shape == (len(X_reg),)

def test_predict_outputs_are_floats():
    dt = RegressionTreeRegressor(max_depth=5)
    dt.fit(X_reg, y_reg)
    assert dt.predict(X_reg).dtype in (np.float32, np.float64)

def test_predict_close_to_truth_deep_tree():
    dt = RegressionTreeRegressor(max_depth=10)
    dt.fit(X_reg, y_reg)
    np.testing.assert_allclose(dt.predict(X_reg), y_reg, atol=1e-6)

def test_predict_left_region_smaller_than_right():
    dt = RegressionTreeRegressor(max_depth=3)
    dt.fit(X_reg, y_reg)
    left_pred = dt.predict(np.array([[2.0]]))[0]
    right_pred = dt.predict(np.array([[9.0]]))[0]
    assert left_pred < right_pred

def test_depth_one_stump_still_predicts():
    dt = RegressionTreeRegressor(max_depth=1)
    dt.fit(X_reg, y_reg)
    assert dt.predict(X_reg).shape == (len(X_reg),)
