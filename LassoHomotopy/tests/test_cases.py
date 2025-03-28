import numpy as np
import pytest
from ..model.LassoHomotopy import LassoHomotopyModel

def test_constant_data():
    # When X is constant and y is constant, we expect zero coefficients and intercept equal to y constant.
    X = np.ones((10, 1))
    y = np.full(10, 5.0)
    model = LassoHomotopyModel()
    results = model.fit(X, y)
    preds = results.predict(X)
    np.testing.assert_allclose(preds, 5.0, atol=1e-6)

def test_zero_data():
    # Test with X and y equal to zero.
    X = np.zeros((10, 2))
    y = np.zeros(10)
    model = LassoHomotopyModel()
    results = model.fit(X, y)
    preds = results.predict(X)
    np.testing.assert_allclose(preds, 0.0, atol=1e-6)
    np.testing.assert_allclose(results.coef, np.zeros(2), atol=1e-6)

def test_collinear_data():
    # Create nearly collinear features.
    np.random.seed(0)
    X_base = np.random.randn(100, 1)
    X = np.hstack([X_base, X_base * 0.99])
    y = X_base.flatten() * 3.0 + 1.0
    model = LassoHomotopyModel()
    results = model.fit(X, y)
    # Expect at least one coefficient to be nearly zero for sparsity.
    assert np.sum(np.abs(results.coef) < 1e-3) >= 1

def test_more_features_than_samples():
    # Test with more features than samples.
    np.random.seed(1)
    X = np.random.randn(5, 10)
    y = np.random.randn(5)
    model = LassoHomotopyModel()
    results = model.fit(X, y)
    preds = results.predict(X)
    assert preds.shape[0] == 5

def test_overfitting_low_lambda():
    # When noise is very low, the model should predict very close to y.
    np.random.seed(2)
    X = np.random.randn(50, 3)
    true_coef = np.array([2.0, -1.5, 0.5])
    y = X.dot(true_coef) + 0.01 * np.random.randn(50)
    model = LassoHomotopyModel()
    results = model.fit(X, y)
    preds = results.predict(X)
    np.testing.assert_allclose(preds, y, atol=0.5)

def test_monotonic_lambda_path():
    # The lambda values along the path should be monotonic decreasing.
    np.random.seed(3)
    X = np.random.randn(30, 4)
    y = X.dot(np.array([1.0, -2.0, 3.0, 0.5])) + 0.1 * np.random.randn(30)
    model = LassoHomotopyModel()
    results = model.fit(X, y)
    lambdas = results.path_lambdas
    for i in range(1, len(lambdas)):
        assert lambdas[i] <= lambdas[i - 1] + 1e-6

def test_intercept_calculation():
    # Verify that the intercept is computed correctly.
    np.random.seed(4)
    X = np.random.randn(40, 2)
    true_coef = np.array([1.0, -1.0])
    intercept_true = 3.0
    y = X.dot(true_coef) + intercept_true
    model = LassoHomotopyModel()
    results = model.fit(X, y)
    y_pred = results.predict(X)
    np.testing.assert_allclose(np.mean(y_pred), np.mean(y), atol=1e-1)

def test_prediction_dimensions():
    # The prediction output should have the same number of samples as input.
    np.random.seed(5)
    X = np.random.randn(20, 5)
    y = np.random.randn(20)
    model = LassoHomotopyModel()
    results = model.fit(X, y)
    preds = results.predict(X)
    assert preds.shape == (20,)

def test_consistency_of_solution():
    # Repeating the fit on the same data should give the same solution.
    np.random.seed(6)
    X = np.random.randn(30, 3)
    y = X.dot(np.array([2.0, -1.0, 0.5])) + 1.0
    model = LassoHomotopyModel()
    results1 = model.fit(X, y)
    results2 = model.fit(X, y)
    np.testing.assert_allclose(results1.coef, results2.coef, atol=1e-6)
    np.testing.assert_allclose(results1.intercept, results2.intercept, atol=1e-6)

def test_negative_coefficients():
    # For data that should yield negative coefficients, check that the learned coefficients are non-positive.
    np.random.seed(7)
    X = np.random.randn(50, 2)
    true_coef = np.array([-2.0, -3.0])
    y = X.dot(true_coef) + 0.1 * np.random.randn(50)
    model = LassoHomotopyModel()
    results = model.fit(X, y)
    assert np.all(results.coef <= 0.1)

def test_path_length():
    # Ensure that the solution path contains more than one step.
    np.random.seed(8)
    X = np.random.randn(40, 4)
    y = X.dot(np.array([1.0, 2.0, -1.0, 0.5])) + 0.1 * np.random.randn(40)
    model = LassoHomotopyModel()
    results = model.fit(X, y)
    assert len(results.path_lambdas) > 1

def test_perfect_fit():
    # When the data lies exactly on a hyperplane, the prediction error should be (nearly) zero.
    np.random.seed(9)
    X = np.random.randn(50, 3)
    true_coef = np.array([1.0, -2.0, 3.0])
    intercept = 4.0
    y = X.dot(true_coef) + intercept
    model = LassoHomotopyModel()
    results = model.fit(X, y)
    preds = results.predict(X)
    np.testing.assert_allclose(preds, y, atol=1e-6)
