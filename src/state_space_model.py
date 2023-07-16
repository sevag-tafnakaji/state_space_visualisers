"""
Module containing a state space model class.
"""
from typing import Tuple
from scipy import linalg
import numpy as np


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

    def calc_stability(self):
        eigen_values = np.linalg.eigvals(self.A)
        if self.discrete:
            return eigen_values[(abs(eigen_values) < 1.0)].shape == eigen_values.shape
        else:
            return eigen_values[(eigen_values < 0.0)].shape == eigen_values.shape
