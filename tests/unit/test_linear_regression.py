import numpy as np
import pytest

from final_ml.supervised_learning.linear_regression import LinearRegression


X_line = np.array([[1], [2], [3], [4], [5]], dtype=float)
y_line = 2 * X_line.ravel() + 1   # [3, 5, 7, 9, 11]


# ---- __init__ ----

def test_learning_rate_stored():
    lr = LinearRegression(learning_rate=0.001, epochs=500)
    assert lr.learning_rate == 0.001

def test_epochs_stored():
    lr = LinearRegression(epochs=500)
    assert lr.epochs == 500

def test_weights_none_before_fit():
    assert LinearRegression().weights is None

def test_bias_none_before_fit():
    assert LinearRegression().bias is None


# ---- fit ----

def test_weights_shape_after_fit():
    lr = LinearRegression(learning_rate=0.01, epochs=100)
    lr.fit(X_line, y_line)
    assert lr.weights.shape == (X_line.shape[1],)

def test_recovers_slope():
    lr = LinearRegression(learning_rate=0.01, epochs=5000)
    lr.fit(X_line, y_line)
    assert abs(lr.weights[0] - 2.0) < 0.1

def test_recovers_intercept():
    lr = LinearRegression(learning_rate=0.01, epochs=5000)
    lr.fit(X_line, y_line)
    assert abs(lr.bias - 1.0) < 0.3


# ---- predict ----

def test_predict_output_shape():
    lr = LinearRegression(learning_rate=0.01, epochs=1000)
    lr.fit(X_line, y_line)
    assert lr.predict(X_line).shape == (len(X_line),)

def test_predict_close_to_truth():
    lr = LinearRegression(learning_rate=0.01, epochs=5000)
    lr.fit(X_line, y_line)
    np.testing.assert_allclose(lr.predict(X_line), y_line, atol=0.5)

def test_predict_outputs_are_floats():
    lr = LinearRegression(learning_rate=0.01, epochs=100)
    lr.fit(X_line, y_line)
    assert lr.predict(X_line).dtype in (np.float32, np.float64)


# ---- mean_squared_error ----

def test_mse_is_non_negative():
    lr = LinearRegression(learning_rate=0.01, epochs=1000)
    lr.fit(X_line, y_line)
    assert lr.mean_squared_error(X_line, y_line) >= 0

def test_mse_near_zero_after_convergence():
    lr = LinearRegression(learning_rate=0.01, epochs=5000)
    lr.fit(X_line, y_line)
    assert lr.mean_squared_error(X_line, y_line) < 0.5


# ---- r_squared ----

def test_r_squared_near_one_after_convergence():
    lr = LinearRegression(learning_rate=0.01, epochs=5000)
    lr.fit(X_line, y_line)
    assert lr.r_squared(X_line, y_line) > 0.95

def test_r_squared_between_zero_and_one():
    lr = LinearRegression(learning_rate=0.01, epochs=1000)
    lr.fit(X_line, y_line)
    assert 0.0 <= lr.r_squared(X_line, y_line) <= 1.0
