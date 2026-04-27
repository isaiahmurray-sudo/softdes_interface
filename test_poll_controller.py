"""Tests the whole poll collection data managment is working as expected"""

import time
import pytest
import pandas as pd

from poll_controller import PollController
from datastructs.poll_data import PollParameters, TrialInfo
from controllers.adc_controller import ADCControllerDebug
from controllers.lcr_controller import LCRControllerDebug
from controllers.leash_controller import (
    LeashControllerDebug,
    ExposureParameter,
    ExposureImage,
)


class TestPollController:
    """Unit tests for the PollController orchestrator."""

    @pytest.fixture
    def debug_controllers(self):
        """Fixture providing initialized debug controllers."""
        leash = LeashControllerDebug()
        lcr = LCRControllerDebug()
        adc = ADCControllerDebug()

        # Simulate connection
        leash.connect("127.0.0.1")
        lcr.connect("4")

        return leash, lcr, adc

    @pytest.fixture
    def brief_poll_parameters(self):
        """Fixture providing a very short poll configuration to keep tests fast."""
        return PollParameters(
            exposure_parameters=[
                ExposureParameter(duration=0.2, intensity=0.5, image=ExposureImage.V2)
            ],
            frequency_hz=1000,
            minimum_exposure_time_s=0.0,
            trial_info=TrialInfo(trial_name="Test Trial", resin_type="TestResin"),
            push_force=70.0,
            push_repeats=1,
            push_delay_s=0.0,
        )

    def test_uninitialized_poll_data_access(self, debug_controllers):
        """Ensure accessing poll_data before start_poll raises a ValueError."""
        leash, lcr, adc = debug_controllers
        controller = PollController(leash, lcr, adc)

        with pytest.raises(ValueError, match="Poll data is not initialized"):
            _ = controller.poll_data

    def test_enforce_minimum_exposure_time(
        self, debug_controllers, brief_poll_parameters
    ):
        """Test that the controller pads the exposure schedule if it is too short."""
        leash, lcr, adc = debug_controllers
        controller = PollController(leash, lcr, adc)

        # Set a minimum exposure time longer than the scheduled 0.2s
        brief_poll_parameters.minimum_exposure_time_s = 0.5

        # pylint: disable="protected-access"
        controller._enforce_minimum_exposure_time(brief_poll_parameters)
        # pylint: enable="protected-access"

        # The schedule should now have 2 items (the original 0.2s + 0.3s padding)
        schedule = brief_poll_parameters.exposure_parameters
        assert len(schedule) == 2
        assert schedule[-1].duration == pytest.approx(0.3)
        assert schedule[-1].intensity == 0.0  # Padding must be 0 intensity

    def test_full_poll_execution(self, debug_controllers, brief_poll_parameters):
        """Test that start_poll runs the threads and populates the data structures."""
        leash, lcr, adc = debug_controllers
        controller = PollController(leash, lcr, adc)

        # Start a blocking poll
        controller.start_poll(brief_poll_parameters, wait_for_completion=True)

        poll_data = controller.poll_data

        # Check DataFrames were created and populated
        assert isinstance(poll_data.lcr_measurements, pd.DataFrame)
        assert isinstance(poll_data.adc_measurements, pd.DataFrame)
        assert isinstance(poll_data.poi_measurements, pd.DataFrame)

        assert not poll_data.lcr_measurements.empty, "LCR thread failed to collect data"
        assert not poll_data.adc_measurements.empty, "ADC thread failed to collect data"
        assert not poll_data.poi_measurements.empty, "POI logs failed to collect data"

        # Check POI logic tracked the poll start and exposure events
        poi_labels = poll_data.poi_measurements["Label"].values
        assert "poll_start_time" in poi_labels
        assert "exposure_0_call_started" in poi_labels
        assert "exposure_0_call_complete" in poi_labels

    def test_progress_tracking(self, debug_controllers, brief_poll_parameters):
        """Verify that the progress property scales from 0.0 to 1.0."""
        leash, lcr, adc = debug_controllers
        controller = PollController(leash, lcr, adc)

        # Extend the duration slightly so we can catch it mid-poll
        brief_poll_parameters.exposure_parameters[0].duration = 0.5

        # Run non-blocking
        controller.start_poll(brief_poll_parameters, wait_for_completion=False)

        assert controller.polling is True

        # Give the threads a moment to start
        time.sleep(0.2)

        # Progress should be greater than 0 but less than 1
        mid_progress = controller.progress
        assert 0.0 < mid_progress < 1.0

        # Wait for it to finish
        controller.wait_for_poll_completion()

        assert controller.polling is False
        assert controller.progress == 1.0

    def test_get_analysis_dataframe(self, debug_controllers, brief_poll_parameters):
        """Test that the orchestrator correctly hands off data to the utility math functions."""
        leash, lcr, adc = debug_controllers
        controller = PollController(leash, lcr, adc)

        controller.start_poll(brief_poll_parameters, wait_for_completion=True)

        # Extract analysis dataframe
        df = controller.get_analysis_dataframe(
            cell_constant_cm_inv=1.5, walden_constant=2.0
        )

        # Verify the structure matches what we expect
        expected_columns = ["Time", "Impedance_Ohm", "Phase_Deg", "Ionic_Viscosity"]
        for col in expected_columns:
            assert col in df.columns

        assert len(df) == len(controller.poll_data.lcr_measurements)
