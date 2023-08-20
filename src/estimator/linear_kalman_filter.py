"""
Module containing linear kalman filter.
"""

from typing import Tuple
import numpy as np
from numpy import ndarray
from scipy import linalg as la
from src.model.linear_state_space_model import StateSpace


class LinearKalmanFilter():
    """
    Linear Kalman filter class

    Uses stored estimates from previous time step to calculate state estimate & covariance
    predictions and updates.

    N.B. if provided state-space model is not discrete, it will be converted to discrete with 0.01s
    sampling time.
    """

    def __init__(self, model: StateSpace, x_0: ndarray, p_0: ndarray):
        """
        Attributes:
            - model (StateSpace): provided state space model (guaranteed discrete).
            - previous_estimate (ndarray): previous updated estimate (x_{k-1|k-1}). initially x_0
            - previous_p (ndarray): previous updated estimate covariance (P_{k-1|k-1}).
            initially P_0.
        """
        self.model = model.ensure_discrete()
        self.previous_estimate = x_0
        self.previous_p = p_0
        if self.model.Q.size == 0 or self.model.R.size == 0:
            print("Provided model does not have noise. Estimates are not necessary.")

    def prediction_step(self, given_input: ndarray) -> Tuple[ndarray, ndarray]:
        """
        Perform prediction step

        Args:
            given_input (ndarray): input at time k, u_k

        Returns:
            Tuple[ndarray, ndarray]: predicted estimate of states and their covariance
            (x_{k|k-1} and P_{k|k-1}).
        """
        prediction_p = self.model.A @ self.previous_p @ self.model.A.T + (
            self.model.N @ self.model.Q @ self.model.N.T)

        prediction_estimate = self.model.A @ self.previous_estimate + self.model.B @ given_input

        return prediction_estimate, prediction_p

    def update_step(self, given_measurement: ndarray, prediction_estimate: ndarray,
                    prediction_p: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
        """
        Perform update step

        Args:
            given_measurement (ndarray): measurement at time k, y_k
            prediction_estimate (ndarray): predicted estimates of states (x_{k|k-1})
            prediction_p (ndarray): covariance of predicted estimate of states (P_{k|k-1})

        Returns:
            Tuple[ndarray, ndarray, ndarray]: updated estimate of states,
            and their covariance matrix.
        """
        a_dim = self.model.A.shape[0]
        innovation: ndarray = given_measurement - self.model.C @ prediction_estimate
        innovation_covariance: ndarray = \
            self.model.C @ prediction_p @ self.model.C.T + self.model.R
        kalman_gain: ndarray = prediction_p @ self.model.C.T @ la.inv(innovation_covariance)

        updated_estimate: ndarray = prediction_estimate + kalman_gain @ innovation
        updated_p: ndarray = (np.eye(a_dim) - kalman_gain @ self.model.C) @ prediction_p
        post_fit_residual: ndarray = given_measurement - self.model.C @ updated_estimate
        self.previous_estimate = updated_estimate
        self.previous_p = updated_p

        return updated_estimate, updated_p, post_fit_residual

    def get_next_estimate(self, given_input: ndarray, given_measurement: ndarray):
        """
        Combine both prediction and update steps. Using estimate from time k-1, input and
            measurement from time k, find best estimate of states at time k
            (i.e. x_{k|k} and P_{k|k} using x_{k-1|k-1}, P_{k-1|k-1}, u_k, y_k).

        Args:
            given_input (ndarray): input at time k, u_k.
            given_measurement (ndarray): measurement at time k, y_k.

        Returns:
            Tuple[ndarray, ndarray, ndarray, ndarray, ndarray]: updated estimate and p, predicted
            estimate and p, and post fit residual (difference between updated estimated measurement
            and actual measurement).
        """
        prediction_estimate, prediction_p = self.prediction_step(given_input)

        updated_estimate, updated_p, post_fit_residual = self.update_step(given_measurement,
                                                                          prediction_estimate,
                                                                          prediction_p)

        return updated_estimate, updated_p, prediction_estimate, prediction_p, post_fit_residual
