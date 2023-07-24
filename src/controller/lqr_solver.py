import numpy as np
from scipy import linalg
from src.model.state_space_model import StateSpace
from utils.exception_types import SizeError

def get_LQR_gains(model: StateSpace, Q: np.ndarray, R: np.ndarray):
    n = model.A.shape[0]
    m = model.B.shape[1]

    model = model.ensure_discrete()

    if Q.shape != (n, n):
        raise SizeError(f"Expected Q to have shape: {(n, n)}. Got instead: {Q.shape}")

    if R.shape != (m, m):
        raise SizeError(f"Expected R to have shape: {(m, m)}. Got instead: {R.shape}")

    S = linalg.solve_discrete_are(model.A, model.B, Q, R)

    L = linalg.inv(model.B.T @ S @ model.B + R) @ model.B.T @ S @ model.A

    return L
