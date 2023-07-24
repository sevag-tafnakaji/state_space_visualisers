import numpy as np
import matplotlib.pyplot as plt
from src.model.state_space_model import StateSpace


class Simulator():
    """
    Class for simulations
    """

    def __init__(self, model: StateSpace, dt: float = 0.05):
        self.model = model
        self.dt = dt

    def update(self, previous_state: np.ndarray, input: float):
        # TODO: Add LQR controller
        next_state = self.model.A @ previous_state
        output = self.model.C @ next_state

        return next_state, output

    def simulate(self, initial_state: np.ndarray, simulation_time: float):
        N = int(simulation_time / self.dt)

        n = initial_state.shape[0]
        m = self.model.C.shape[0]

        T = np.linspace(0, simulation_time, N, True)

        states = np.zeros((n, N))
        outputs = np.zeros((m, N-1))
        states[:,0] = initial_state

        if not self.model.discrete:
            print("Cannot simulate continuous model, converting to discrete.")
            self.model = self.model.convert_to_discrete(self.dt)
            if not self.model.calc_stability():
                print("Model not stable, please provide stable model.")
        else:
            self.dt = self.model.sampling_time

        for i in range(N-1):
            states[:,i+1], outputs[:,i] = self.update(states[:,i], 0.0)

        return states, outputs, T

