from unittest.mock import Mock
import numpy as np
import pytest
from src.model.linear_state_space_model import StateSpace
from src.controller.lqr_solver import get_lqr_gains
from utils.exception_types import SizeError


A = np.array([[-8, 1], [3, -20.3]])
B = np.array([[1], [2]])
C = np.array([[1, 0]])

continuous_model = StateSpace(A, B, C)
discrete_model = continuous_model.convert_to_discrete(1e-2)


def test_get_lqr_gains():
    """
    GIVEN
        - discrete model.

    WHEN
        - Calling 'get_lqr_gains' method.

    THEN
        - Verify that calculated gains matches gains found through MATLAB.
    """
    l_gains = get_lqr_gains(discrete_model, np.eye(2), np.array([1]).reshape((1, 1)))

    assert (l_gains - np.array([[0.07123481, 0.04916482]]) <= 1e-5).all()


def test_get_lqr_gains_with_continuous():
    """
    GIVEN
        - continuous model.

    WHEN
        - Calling 'get_lqr_gains' method.

    THEN
        - Verify that :
            - continuous model is converted to a discrete model.
            - calculated gains from discrete model matches gains found through MATLAB.
    """
    continuous_model.convert_to_discrete = Mock()
    continuous_model.convert_to_discrete.return_value = discrete_model

    l_gains = get_lqr_gains(continuous_model, np.eye(2), np.array([1]).reshape((1, 1)))

    continuous_model.convert_to_discrete.assert_called_once()
    assert (l_gains - np.array([[0.07123481, 0.04916482]]) <= 1e-5).all()


def test_get_lqr_gains_wrong_q_dim():
    """
    GIVEN
        - discrete model with Q of wrong dimension.

    WHEN
        - Calling 'get_lqr_gains' method.

    THEN
        - Verify that error is raised with descriptive error.
    """
    with pytest.raises(SizeError,
                       match=r"Expected Q to have shape: \(2, 2\). Got instead: \(2, 1\)"):
        _ = get_lqr_gains(discrete_model, np.array([[1], [1]]), np.array([1]).reshape((1, 1)))


def test_get_lqr_gains_wrong_r_dim():
    """
    GIVEN
        - discrete model with R of wrong dimension.

    WHEN
        - Calling 'get_lqr_gains' method.

    THEN
        - Verify that error is raised with descriptive error.
    """
    with pytest.raises(SizeError,
                       match=r"Expected R to have shape: \(1, 1\). Got instead: \(2, 2\)"):
        _ = get_lqr_gains(discrete_model, np.eye(2), np.eye(2))
