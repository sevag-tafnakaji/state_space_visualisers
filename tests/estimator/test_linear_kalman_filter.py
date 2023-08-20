"""
Module containing unit tests for linear kalman filter.
"""
from unittest.mock import Mock
import numpy as np
from src.model.linear_state_space_model import StateSpace
from src.estimator.linear_kalman_filter import LinearKalmanFilter


A = np.array([[-8, 1], [3, -20.3]])
B = np.array([[1], [2]])
C = np.array([[1, 0]])
N = np.eye(2)
Q = np.eye(2)
R = np.array([1]).reshape(1, 1)

test_model = StateSpace(A, B, C, N, Q, R)

test_model_discrete = test_model.convert_to_discrete(1e-3)


def test_prediction_step():
    """
    GIVEN
        - Linear Kalman filter instance, starting values, example input

    WHEN
        - Calling 'prediction_step' method.

    THEN
        - Verify that results match MATLAB's calculations.
    """
    x_0 = np.array([[1], [1]])
    p_0 = np.eye(2)
    kalman = LinearKalmanFilter(test_model_discrete, x_0, p_0)
    expected_estimate = np.array([[0.9930], [0.9829]])
    expected_p_matrix = np.array([[1.9841, 0.0039], [0.0039, 1.9602]])

    actual_predicted_estimate, actual_predicted_p = kalman.prediction_step(
        given_input=np.array([[0]]))

    print(actual_predicted_estimate, actual_predicted_p)
    print(expected_estimate, expected_p_matrix)

    assert np.allclose(actual_predicted_p, expected_p_matrix, [1.00001e10, 1e-2])
    assert np.allclose(actual_predicted_estimate, expected_estimate, [1.00001e10, 1e-2])


def test_update_step():
    """
    GIVEN
        - Linear Kalman filter instance, starting values, example measurement

    WHEN
        - Calling 'update_step' method.

    THEN
        - Verify that results match MATLAB's calculations.
    """
    x_0 = np.array([[1], [1]])
    p_0 = np.eye(2)
    kalman = LinearKalmanFilter(test_model_discrete, x_0, p_0)
    expected_estimate = np.array([[0.5], [1]])
    expected_p_matrix = np.array([[0.5, 0.0], [0.0, 1.0]])
    expected_residual = -0.5

    actual_predicted_estimate, actual_predicted_p, actual_residual = kalman.update_step(0, x_0, p_0)

    assert np.allclose(actual_predicted_p, expected_p_matrix, [1.00001e10, 1e-2])
    assert np.allclose(actual_predicted_estimate, expected_estimate, [1.00001e10, 1e-2])
    assert expected_residual == actual_residual


def test_get_next_estimate():
    """
    GIVEN
        - Linear Kalman filter instance, starting values, example input, and measurement

    WHEN
        - Calling 'get_next_estimate' method.

    THEN
        - Verify that prediction and update step calculation functions are called once.
    """
    x_0 = np.array([[1], [1]])
    p_0 = np.eye(2)
    kalman = LinearKalmanFilter(test_model_discrete, x_0, p_0)
    # Given random return values (accuracy not necessary).
    kalman.prediction_step = Mock(return_value=(1, 1))
    kalman.update_step = Mock(return_value=(1, 1, 1))

    _ = kalman.get_next_estimate(0, 0)

    kalman.prediction_step.assert_called_once()
    kalman.update_step.assert_called_once()
