"""Tests that utils.py is properly functioning"""

import numpy as np

from utils import compute_ionic_viscosity


class TestComputeIonicViscosity:
    """Unit tests for the compute_ionic_viscosity utility function."""

    def test_pure_resistance(self):
        """
        Test with pure resistance (Phase = 0).
        cos(0) = 1.
        Conductance (G) = 1 / Z.
        Viscosity = 1 / G.
        Therefore, Viscosity should exactly equal Z.
        """
        z_ohm = np.array([100.0, 50.0, 10.0])
        phase_deg = np.array([0.0, 0.0, 0.0])

        expected_viscosity = np.array([100.0, 50.0, 10.0])
        result = compute_ionic_viscosity(z_ohm, phase_deg)

        np.testing.assert_allclose(result, expected_viscosity, rtol=1e-5)

    def test_with_phase_angle(self):
        """
        Test with a phase angle to ensure trigonometry is applied correctly.
        cos(60 deg) = 0.5.
        """
        z_ohm = np.array([100.0])
        phase_deg = np.array([60.0])

        # G = cos(60) / 100 = 0.5 / 100 = 0.005
        # Viscosity = 1 / G = 1 / 0.005 = 200.0
        expected_viscosity = np.array([200.0])
        result = compute_ionic_viscosity(z_ohm, phase_deg)

        np.testing.assert_allclose(result, expected_viscosity, rtol=1e-5)

    def test_constants_applied(self):
        """Test that cell_constant_cm_inv and walden_constant scale the output correctly."""
        z_ohm = np.array([100.0])
        phase_deg = np.array([0.0])
        cell_constant = 2.0
        walden_constant = 5.0

        # G = 1 / 100 = 0.01
        # Conductivity = G * 2.0 = 0.02
        # Viscosity = 5.0 / 0.02 = 250.0
        expected_viscosity = np.array([250.0])
        result = compute_ionic_viscosity(
            z_ohm,
            phase_deg,
            cell_constant_cm_inv=cell_constant,
            walden_constant=walden_constant,
        )

        np.testing.assert_allclose(result, expected_viscosity, rtol=1e-5)

    def test_invalid_physical_state(self):
        """
        Test that angles resulting in negative conductance (e.g., >90 or <-90 deg)
        are masked out and return NaN.
        """
        z_ohm = np.array([100.0, 100.0])
        phase_deg = np.array([120.0, -120.0])  # cos(+/-120 deg) = -0.5

        result = compute_ionic_viscosity(z_ohm, phase_deg)

        # Output array should contain NaNs where conductance was <= 0
        assert np.isnan(result[0])
        assert np.isnan(result[1])

    def test_zero_impedance_clamping(self):
        """
        Test that a Z of 0 Ohms does not raise a ZeroDivisionError and
        is safely clamped using the 1e-12 fallback.
        """
        z_ohm = np.array([0.0])
        phase_deg = np.array([0.0])

        result = compute_ionic_viscosity(z_ohm, phase_deg)

        # G = cos(0) / max(0, 1e-12) = 1 / 1e-12 = 1e12
        # Viscosity = 1 / 1e12 = 1e-12
        expected_viscosity = np.array([1e-12])
        np.testing.assert_allclose(result, expected_viscosity, rtol=1e-5)

    def test_negative_impedance_handled_by_abs(self):
        """Test that negative Z values are processed via their absolute value."""
        z_ohm = np.array([-100.0])
        phase_deg = np.array([0.0])

        # abs(-100) = 100. Viscosity should be 100.
        expected_viscosity = np.array([100.0])
        result = compute_ionic_viscosity(z_ohm, phase_deg)

        np.testing.assert_allclose(result, expected_viscosity, rtol=1e-5)
