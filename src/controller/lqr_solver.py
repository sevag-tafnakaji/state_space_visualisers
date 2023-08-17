import numpy as np
from scipy import linalg
from src.model.linear_state_space_model import StateSpace
from utils.exception_types import SizeError


def get_lqr_gains(model: StateSpace, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Calculate LQR gain using scale values.

    Args:
        model (StateSpace): state space model to apply LQR with.
        Q (np.ndarray): scales on states.
        R (np.ndarray): scales on inputs.

    Raises:
        SizeError: Q matrix wrong size.
        SizeError: R matrix wrong size.

    Returns:
        np.ndarray: gain matrix.
    """
    n = model.A.shape[0]
    m = model.B.shape[1]

    model: StateSpace = model.ensure_discrete()

    if not model.is_stabilizable():
        print("Cannot create controller for non-stabilisable models.")
        return None

    if Q.shape != (n, n):
        raise SizeError(f"Expected Q to have shape: {(n, n)}. Got instead: {Q.shape}")

    if R.shape != (m, m):
        raise SizeError(f"Expected R to have shape: {(m, m)}. Got instead: {R.shape}")

    # Solve Ricatti equations to find S
    S = linalg.solve_discrete_are(model.A, model.B, Q, R)

    l_gains = linalg.inv(model.B.T @ S @ model.B + R) @ model.B.T @ S @ model.A

    return l_gains
