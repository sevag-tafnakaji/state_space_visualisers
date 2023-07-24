from unittest import TestCase
from unittest.mock import Mock
import numpy as np
import pytest
from src.model.state_space_model import StateSpace
from src.controller.lqr_solver import get_LQR_gains
from utils.exception_types import SizeError


A = np.array([[-8, 1], [3, -20.3]])
B = np.array([[1], [2]])
C = np.array([[1, 0]])

continuous_model = StateSpace(A, B, C)
discrete_model = continuous_model.convert_to_discrete(1e-2)


def test_get_LQR_gains():
    L = get_LQR_gains(discrete_model, np.eye(2), np.array([1]).reshape((1, 1)))
    print((L - np.array([[0.07123481, 0.04916482]]) <= 1e-5).all())
    assert (L - np.array([[0.07123481, 0.04916482]]) <= 1e-5).all()


def test_get_LQR_gains_with_continuous():
    continuous_model.convert_to_discrete = Mock()
    continuous_model.convert_to_discrete.return_value = discrete_model

    L = get_LQR_gains(continuous_model, np.eye(2), np.array([1]).reshape((1, 1)))

    continuous_model.convert_to_discrete.assert_called_once()
    assert (L - np.array([[0.07123481, 0.04916482]]) <= 1e-5).all()


def test_get_LQR_gains_wrong_Q_dim():
    with pytest.raises(SizeError, match = r"Expected Q to have shape: \(2, 2\). Got instead: \(2, 1\)"):
        L = get_LQR_gains(discrete_model, np.array([[1], [1]]), np.array([1]).reshape((1, 1)))

def test_get_LQR_gains_wrong_R_dim():
    with pytest.raises(SizeError, match = r"Expected R to have shape: \(1, 1\). Got instead: \(2, 2\)"):
        L = get_LQR_gains(discrete_model, np.eye(2), np.eye(2))
