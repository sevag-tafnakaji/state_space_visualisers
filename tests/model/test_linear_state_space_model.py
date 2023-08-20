from unittest import TestCase
import numpy as np
from src.model.linear_state_space_model import StateSpace


class TestStateSpaceModel(TestCase):

    def setUp(self):
        A = np.array([[-8, 1], [3, -20.3]])
        B = np.array([[1], [2]])
        C = np.array([[1, 0]])

        self.test_model = StateSpace(A, B, C)

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

    def test_ensure_discrete(self):
        """
        GIVEN
            - Continuous state space model.

        WHEN
            - Calling 'ensure_discrete' method.

        THEN
            - Verify that a discrete model is returned. If originally continuous then a
            sampling time of 1e-2 is used (by default) to convert to discrete. Assumes no delays.
        """

        # WHEN
        discrete_model = self.test_model.ensure_discrete()

        # THEN
        assert discrete_model.discrete
        assert discrete_model.sampling_time == 1e-2

    def test_calc_stability(self):
        """
        GIVEN
            - State space model instance.

        WHEN
            - Calling 'calc_stability' method.

        THEN
            - Verify that the example model pases the stability test.
        """
        # WHEN
        is_stable = self.test_model.calc_stability()

        # THEN
        self.assertTrue(is_stable)

        # WHEN
        is_stable = self.test_model.ensure_discrete().calc_stability()

        # THEN
        self.assertTrue(is_stable)

    def test_is_controllable(self):
        """
        GIVEN
            - State space model instance.

        WHEN
            - Calling 'is_controllable' method.

        THEN
            - Verify that the example model is controllable.
        """

        expected_result = True

        # WHEN
        controllable = self.test_model.is_controllable()

        # THEN
        assert expected_result == controllable

    def test_is_observable(self):
        """
        GIVEN
            - State space model instance.

        WHEN
            - Calling 'is_observable' method.

        THEN
            - Verify that the example model is observable.
        """

        expected_result = True

        # WHEN
        observable = self.test_model.is_observable()

        # THEN
        assert expected_result == observable

    def test_is_stabilizable(self):
        """
        GIVEN
            - State space model instance.

        WHEN
            - Calling 'is_stabilizable' method.

        THEN
            - Verify that the example model is stabilizable.
        """

        expected_result = True

        # WHEN
        stabilizable = self.test_model.is_stabilizable()

        # THEN
        assert expected_result == stabilizable

    def test_is_detectible(self):
        """
        GIVEN
            - State space model instance.

        WHEN
            - Calling 'is_detectible' method.

        THEN
            - Verify that the example model is detectible.
        """

        expected_result = True

        # WHEN
        detectible = self.test_model.is_detectible()

        # THEN
        assert expected_result == detectible
