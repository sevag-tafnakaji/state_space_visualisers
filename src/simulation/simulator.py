"""
Module containing simulation class
"""
from typing import Tuple
import numpy as np
from src.model.linear_state_space_model import StateSpace
from src.estimator.linear_kalman_filter import LinearKalmanFilter
from src.controller.lqr_controller import LQRController


class Simulator():
    """
    Class for simulations
    """

    def __init__(self, model: StateSpace, x_0: np.ndarray, p_0: np.ndarray,
                 q_matrix: np.ndarray, r_matrix: np.ndarray, reference_value: np.ndarray,
                 m_matrix: np.ndarray, delta_t: float = 0.05):
        self.model = model.ensure_discrete(delta_t * 0.5)

        self.q_matrix = q_matrix
        self.r_matrix = r_matrix

        self.kalman = LinearKalmanFilter(model, x_0, p_0)
        self.lqr_controller = LQRController(self.model, reference_value, m_matrix,
                                            self.q_matrix, self.r_matrix)

        self.delta_t = delta_t

    def update(self, previous_state: np.ndarray,
               previous_estimate: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                       np.ndarray, np.ndarray]:
        """
        Calculate one step in simulation (i.e. go from time step k to k+1)

        Args:
            previous_state (np.ndarray): x_{k-1}
            previous_estimate (np.ndarray): x_{k-1|k-1}

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: next state, measurement,
            next state estimate, next state estimate covariance.
        """
        current_input = self.lqr_controller.get_input(previous_estimate)
        num_of_states = self.model.A.shape[0]
        num_of_process_noise = self.model.Q.shape[0]

        process_noise = np.random.multivariate_normal(
            np.zeros(num_of_process_noise), self.model.Q).reshape(num_of_process_noise, 1)

        if self.model.C.shape[0] > 1:
            measurement_noise = np.random.multivariate_normal(
                np.zeros(self.model.C.shape[0]), self.model.R).reshape(self.model.C.shape[0], 1)
        else:
            measurement_noise = np.random.normal(0, np.sqrt(self.model.R[0]))

        next_state = (self.model.A @ previous_state + self.model.B @ current_input) + (
            self.model.N @ process_noise).reshape(num_of_states, 1)
        output = self.model.C @ next_state + measurement_noise
        next_estimate, next_p, _, _, _ = self.kalman.get_next_estimate(current_input, output)

        return next_state.reshape(next_estimate.shape), output, next_estimate, current_input, next_p

    def simulate(self, initial_state: np.ndarray,
                 simulation_time: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete full simulation starting from given initial state

        Args:
            initial_state (np.ndarray): given initial state. Can differ from x_0 for kalman filter
            simulation_time (float): total time for simulation

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: all states, all measurements,
            all state estimates, all time values simulated. Note that measurements and estimates
            have one less value (i.e. if there are N = 20 states, then there are
            N = 19 measurements and estimates).
        """
        N = int(simulation_time / self.delta_t)

        n = initial_state.shape[0]
        m = self.model.C.shape[0]

        T = np.linspace(0, simulation_time, N, True)

        states = np.zeros((n, N))
        state_estimates = np.zeros((n, N-1))
        outputs = np.zeros((m, N-1))
        inputs = np.zeros((m, N-1))
        states[:, 0:1] = initial_state
        state_estimates[:, 0:1] = initial_state

        for i in range(N-1):
            if i == 0:
                states[:, i + 1: i + 2], outputs[:, i:i + 1], _, inputs[:, i:i + 1], _ = \
                    self.update(states[:, i:i + 1], state_estimates[:, i:i + 1])
            else:
                states[:, i + 1: i + 2], outputs[:, i:i + 1], \
                    state_estimates[:, i:i + 1], inputs[:, i:i + 1], _ = \
                    self.update(states[:, i:i + 1], state_estimates[:, i - 1:i])

        return states, outputs, state_estimates, inputs, T
