"""
Module containing a state space model class.
"""
from typing import Tuple
from scipy import linalg
import numpy as np
from sympy.matrices import Matrix, eye
from sympy import Symbol

class StateSpace:
    """
    Class for state space.
    """

    # TODO: Kalman filter state estimate to use.
    #       Add input function using LQR.

    def __init__(self,
                 A_matrix: np.ndarray,
                 B_matrix: np.ndarray,
                 C_matrix: np.ndarray,
                 N_matrix: np.ndarray = np.array([]),
                 Q_matrix: np.ndarray = np.array([]),
                 R_matrix: np.ndarray = np.array([]),
                 sampling_time: float = 0.0):

        self.discrete = sampling_time != 0.0
        self.A = A_matrix
        self.B = B_matrix
        self.C = C_matrix
        self.N = N_matrix
        self.Q = Q_matrix
        self.R = R_matrix
        self.sampling_time = sampling_time

    def convert_to_discrete(self, sampling_time: float):
        if self.discrete:
            raise TypeError("Trying to discretise an already discrete model.")
        continuous_set = np.column_stack((np.row_stack((self.A, np.zeros(self.A.shape[1]))),\
            np.row_stack((self.B, np.zeros(self.B.shape[1])))))

        discrete_set = linalg.expm(continuous_set * sampling_time)

        A_d = discrete_set[0:self.A.shape[0], 0:self.A.shape[1]]
        B_d = discrete_set[0:self.B.shape[0], self.A.shape[1]:self.A.shape[1]+self.B.shape[1]]

        # self.A = A_d
        # self.B = B_d

        # self.sampling_time = sampling_time
        # self.discrete = True
        return StateSpace(A_d, B_d, self.C, self.N, self.Q, self.R, sampling_time)

    def ensure_discrete(self, dt=1e-2):
        if not self.discrete:
            print(f"Model not discrete, converting to discrete with sampling time {dt}")
            return self.convert_to_discrete(dt)

        return self

    def calc_stability(self):
        eigen_values = np.linalg.eigvals(self.A)
        if self.discrete:
            return eigen_values[(abs(eigen_values) < 1.0)].shape == eigen_values.shape
        else:
            return eigen_values[(eigen_values < 0.0)].shape == eigen_values.shape

    def is_controllable(self):
        n = self.A.shape[0]
        S = np.array([])
        for i in range(n):
            # print(np.linalg.matrix_power(self.A, i)*self.B)
            if S.size == 0:
                S = self.B
            else:
                S = np.concatenate((S, np.linalg.matrix_power(self.A, i)*self.B), axis=1)
        return np.linalg.matrix_rank(S) == n

    def is_observable(self):
        n = self.A.shape[0]
        O = np.array([])
        for i in range(n):
            if O.size == 0:
                O = self.C
            else:
                O = np.concatenate((O, self.C*np.linalg.matrix_power(self.A, i)), axis=0)
        return np.linalg.matrix_rank(O) == n

    def is_stabilizable(self):
        n = self.A.shape[0]
        lbd = Symbol('lbd')
        lbd_mat = lbd * eye(n)
        D: Matrix = lbd_mat - self.A
        D: Matrix = D.col_insert(n, Matrix(self.B))
        return D.rank() == n

    def is_detectible(self):
        n = self.A.shape[0]
        lbd = Symbol('lbd')
        lbd_mat = lbd * eye(n)
        D: Matrix = lbd_mat - self.A
        D: Matrix = D.row_insert(n, Matrix(self.C))
        return D.rank() == n
