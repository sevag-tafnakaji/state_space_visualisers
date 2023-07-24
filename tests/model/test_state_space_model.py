from unittest import TestCase
import numpy as np
from src.model.state_space_model import StateSpace
import pytest


class TestStateSpaceModel(TestCase):

    def setUp(self):
        A = np.array([[-8, 1], [3, -20.3]])
        B = np.array([[1], [2]])
        C = np.array([[1, 0]])

        self.test_model = StateSpace(A, B, C)


    def test_calc_stability(self):
        """
        GIVEN
            - State space model instance.

        WHEN
            - Calling 'calc_stability' method.

        THEN
            - Verify that the example model pases the (continuous) stability test.
        """
        is_stable = self.test_model.calc_stability()
        self.assertEqual(is_stable, True)

    def test_convert_to_discrete(self):
        """
        GIVEN
            - Continuous state space model instance.

        WHEN
            - Calling 'convert_to_discrete' method.

        THEN
            - Verify that a new discrete model is found.
        """

        discrete_test_model = self.test_model.convert_to_discrete(1e-2)

        self.assertTrue(isinstance(discrete_test_model, StateSpace))

        self.assertEqual(discrete_test_model.sampling_time, 1e-2)

        self.assertEqual(discrete_test_model.discrete, True)
