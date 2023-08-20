import numpy as np
from src.model.linear_state_space_model import StateSpace
from src.controller.lqr_controller import LQRController


A = np.array([[-8, 1], [3, -20.3]])
B = np.array([[1], [2]])
C = np.array([[1, 0]])

continuous_model = StateSpace(A, B, C)
discrete_model = continuous_model.convert_to_discrete(1e-2)
solver = LQRController(
    discrete_model, np.array([0.0, 0.0]), np.eye(2), np.eye(2), np.array([1]).reshape((1, 1)))


def test_get_lqr_gains():
    """
    GIVEN
        - discrete model.

    WHEN
        - Calling 'get_lqr_gains' method.

    THEN
        - Verify that calculated gains matches gains found through MATLAB.
    """
    l_gains = solver.get_lqr_gains()

    assert (l_gains - np.array([[0.07123481, 0.04916482]]) <= 1e-5).all()
