"""
Module containing a state space model class.
"""
from __future__ import annotations
from scipy import linalg
import numpy as np
from sympy.matrices import Matrix, eye
from sympy import Symbol


class StateSpace:
    """
    Class for state space.
    """

    def __init__(self,
                 A_matrix: np.ndarray,
                 B_matrix: np.ndarray,
                 C_matrix: np.ndarray,
                 N_matrix: np.ndarray = np.array([]),
                 Q_matrix: np.ndarray = np.array([]),
                 R_matrix: np.ndarray = np.array([]),
                 sampling_time: float = 0.0):
        """
        Attributes:
            - discrete (bool): if sampling time is non-zero.
            - A (np.ndarray): A matrix of model. Update of state based on current state values.
            - B (np.ndarray): B matrix of model. Update of state based on inputs.
            - C (np.ndarray): C matrix of model. Update of output based on current state values.
            - N (np.ndarray): N matrix of model. Effect of process noise on state update.
            - Q (np.ndarray): Covariance of process noise.
            - R (np.ndarray): Covariance of measurement noise.
            - sampling_time (float): sampling time of (possibly) discrete model
        """

        self.discrete = sampling_time != 0.0
        self.A = A_matrix
        self.B = B_matrix
        self.C = C_matrix
        self.N = N_matrix
        self.Q = Q_matrix
        self.R = R_matrix
        self.sampling_time = sampling_time

    def convert_to_discrete(self, sampling_time: float) -> StateSpace:
        """
        Return a new, discrete model if current model is continuous.

        Args:
            sampling_time (float): desired sampling time of new model.

        Raises:
            TypeError: Raised if current model is already discrete.

        Returns:
            StateSpace: new, discrete, state space model.
        """
        if self.discrete:
            raise TypeError("Trying to discretise an already discrete model.")
        num_of_states = self.A.shape[0]
        num_of_inputs = self.B.shape[1]

        continuous_set = np.column_stack((np.row_stack((self.A, np.zeros((num_of_inputs,
                                                                          num_of_states)))),
                                          np.row_stack((self.B, np.zeros((num_of_inputs,
                                                                          num_of_inputs))))))

        discrete_set = linalg.expm(continuous_set * sampling_time)

        A_d = discrete_set[0:self.A.shape[0], 0:self.A.shape[1]]
        B_d = discrete_set[0:self.B.shape[0], self.A.shape[1]:self.A.shape[1]+self.B.shape[1]]

        return StateSpace(A_d, B_d, self.C, self.N, self.Q, self.R, sampling_time)

    def ensure_discrete(self, dt: float = 1e-2) -> StateSpace:
        """
        if current model is not discrete, return a discrete model

        Args:
            dt (float, optional): sampling time if conversion is needed. Defaults to 1e-2.

        Returns:
            StateSpace: discrete model
        """
        if not self.discrete:
            print(f"Model not discrete, converting to discrete with sampling time {dt}")
            return self.convert_to_discrete(dt)

        return self

    def calc_stability(self) -> bool:
        """
        Calculate stability of model. Different methods based on if model is discrete or continous.

        Returns:
            bool: if model is stable or not
        """
        eigen_values = np.linalg.eigvals(self.A)
        if self.discrete:
            return eigen_values[(abs(eigen_values) < 1.0)].shape == eigen_values.shape
        else:
            return eigen_values[(eigen_values < 0.0)].shape == eigen_values.shape

    def is_controllable(self) -> bool:
        """
        Calculate if model is controllable.

        Returns:
            bool: if model is controllable.
        """
        n = self.A.shape[0]
        S = np.array([])
        for i in range(n):
            if S.size == 0:
                S = self.B
            else:
                S = np.concatenate((S, np.linalg.matrix_power(self.A, i) @ self.B), axis=1)
        return np.linalg.matrix_rank(S) == n

    def is_observable(self) -> bool:
        """
        Calculate if model is observable.

        Returns:
            bool: if model is observable.
        """
        n = self.A.shape[0]
        observability_matrix = np.array([])
        for i in range(n):
            if observability_matrix.size == 0:
                observability_matrix = self.C
            else:
                observability_matrix = np.concatenate(
                    (observability_matrix, self.C @ np.linalg.matrix_power(self.A, i)), axis=0)
        return np.linalg.matrix_rank(observability_matrix) == n

    def is_stabilizable(self) -> bool:
        """
        Calculate if model is stabilizable.

        Returns:
            bool: if model is stabilizable.
        """
        n = self.A.shape[0]
        lbd = Symbol('lbd')
        lbd_mat = lbd * eye(n)
        D: Matrix = lbd_mat - self.A
        D: Matrix = D.col_insert(n, Matrix(self.B))
        return D.rank() == n

    def is_detectible(self) -> bool:
        """
        Calculate if model is detectible.

        Returns:
            bool: if model is detectible.
        """
        n = self.A.shape[0]
        lbd = Symbol('lbd')
        lbd_mat = lbd * eye(n)
        D: Matrix = lbd_mat - self.A
        D: Matrix = D.row_insert(n, Matrix(self.C))
        return D.rank() == n
