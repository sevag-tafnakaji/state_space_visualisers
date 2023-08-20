"""
Module containing LQR controller class.
"""

import numpy as np
from scipy import linalg
from src.model.linear_state_space_model import StateSpace


class LQRController():
    """
    Class for solving Riccati equations for LQR controllers.
    """

    def __init__(self, model: StateSpace, reference_value: np.ndarray,
                 m_matrix: np.ndarray, q_matrix: np.ndarray = np.array([]),
                 r_matrix: np.ndarray = np.array([])):
        """
        Args:
            model (StateSpace): Model to be controlled
            q_matrix (np.ndarray, optional): . Defaults to np.array([]).
            r_matrix (np.ndarray, optional): _description_. Defaults to np.array([]).

        Attributes:
            model (StateSpace):
        """
        self.model: StateSpace = model.ensure_discrete()
        self.num_of_states = model.A.shape[0]
        self.num_of_inputs = model.B.shape[1]
        self.q_matrix = np.eye(self.num_of_states) if q_matrix.size == 0 else q_matrix
        self.r_matrix = np.eye(self.num_of_inputs) if r_matrix.size == 0 else r_matrix
        self.reference_value = reference_value
        self.m_matrix = m_matrix
        self.l_gains = None

    def get_lqr_gains(self) -> np.ndarray:
        """
        Calculate LQR gain using scale values.

        Raises:
            SizeError: Q matrix wrong size.
            SizeError: R matrix wrong size.

        Rxeturns:
            np.ndarray: gain matrix.
        """
        if not self.model.is_stabilizable():
            print("Cannot create controller for non-stabilisable models.")
            return None

        # Solve Ricatti equations to find S
        if self.l_gains is not None:
            return self.l_gains
        s_matrix = linalg.solve_discrete_are(
            self.model.A, self.model.B, self.q_matrix, self.r_matrix)

        l_gains: np.ndarray = linalg.inv(
            self.model.B.T @ s_matrix @ self.model.B + self.r_matrix) @ self.\
            model.B.T @ s_matrix @ self.model.A
        return l_gains

    def get_input(self, state: np.ndarray) -> np.ndarray:
        """
        Use LQR to find optimal input based on current state.

        Args:
            state (np.ndarray): state value (estimate or exact)

        Returns:
            np.ndarray: best input according to LQR principles
        """
        if self.l_gains is None:
            self.l_gains = self.get_lqr_gains()
        lr_matrix = np.negative(np.linalg.pinv(
            self.m_matrix @ np.linalg.inv(
                self.model.A - self.model.B @ self.l_gains) @ self.model.B))

        return np.negative(self.l_gains @ state) + lr_matrix @ self.reference_value
