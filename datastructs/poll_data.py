"""Data structures and serialization helpers for poll execution results."""

import dataclasses
from dataclasses import field, dataclass
from datetime import datetime as dt
import os
import uuid
import time
import json
from enum import Enum
import pandas as pd
from controllers.lcr_controller import MeasurementMode, MeasurementSpeed
from controllers.leash_controller import ExposureParameter


class CustomJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles enums and exposure parameter dataclasses."""

    # pylint: disable="arguments-renamed"
    def default(self, obj):
        """Serialize custom domain objects used by poll metadata payloads."""
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, ExposureParameter):
            return obj.to_dict()
        return super().default(obj)
    
@dataclass
class TrialInfo:
    """Descriptive metadata captured before and after a trial."""

    trial_name: str
    resin_type: str
    trial_index: int | None = None
    pre_trial_notes: str | None = None
    post_trial_notes: str | None = None
    thickness: float | None = None


# pylint: disable=too-many-instance-attributes
@dataclass
class PollParameters:
    """Configuration inputs required to execute a poll run."""

    exposure_parameters: list[ExposureParameter]
    frequency_hz: int
    minimum_exposure_time_s: float = 0.0
    measurement_mode: MeasurementMode = MeasurementMode.IMPEDANCE
    measurement_speed: MeasurementSpeed = MeasurementSpeed.FAST
    trial_info: TrialInfo = field(
        default_factory=lambda: TrialInfo(
            trial_name="Default Trial", resin_type="Unknown"
        )
    )
    push_force: float = 70
    push_repeats: int = 2
    push_delay_s: float = 3


class PollData:
    """Container for poll metadata and time-series measurements."""

    def __init__(self, poll_parameters: PollParameters):
        """Initialize empty measurement tables for a new poll run."""
        self.id = str(uuid.uuid4())
        self.timestamp = time.time()

        self._poll_parameters = poll_parameters
        self.trial_info = None
        self._lcr_measurements = pd.DataFrame(columns=["Time", "Z", "Phase"])
        self._adc_measurements = pd.DataFrame(columns=["Time", "V"])
        self._poi_measurements = pd.DataFrame(columns=["Time", "Label", "Value"])

    @property
    def poll_parameters(self):
        """Return the immutable poll parameters associated with this run."""
        return self._poll_parameters

    @property
    def lcr_measurements(self):
        """Return the LCR measurements dataframe."""
        return self._lcr_measurements

    @property
    def adc_measurements(self):
        """Return the ADC measurements dataframe."""
        return self._adc_measurements

    @property
    def poi_measurements(self):
        """Return the points-of-interest dataframe."""
        return self._poi_measurements

    @property
    def data_summary(self):
        """Build a lightweight summary dictionary for quick inspection."""
        summary = {
            "id": self.id,
            "timestamp": self.timestamp,
            "poll_parameters": dataclasses.asdict(self._poll_parameters),
            "trial_info": (
                dataclasses.asdict(self.trial_info) if self.trial_info else None
            ),
            "lcr_measurements_count": len(self._lcr_measurements),
            "adc_measurements_count": len(self._adc_measurements),
            "poi_measurements_count": len(self._poi_measurements),
        }
        return summary

    def add_lcr_measurement(self, measurement: dict):
        """Append a single LCR measurement row."""
        self._lcr_measurements.loc[len(self._lcr_measurements)] = measurement

    def add_adc_measurement(self, measurement: dict):
        """Append a single ADC measurement row."""
        self._adc_measurements.loc[len(self._adc_measurements)] = measurement

    def add_poi_measurement(self, label: str, value: str):
        """Append a point-of-interest record with current timestamp."""
        measurement = {"Time": time.time(), "Label": label, "Value": value}
        self._poi_measurements.loc[len(self._poi_measurements)] = measurement

    def save_to_file(self):
        """Persist raw CSV measurements and metadata JSON for this poll."""
        root_dir = "softdes_interface/data/polls"
        index, filename = self._prep_file_prefix()
        self._poll_parameters.trial_info.trial_index = index
        os.makedirs(f"{root_dir}/raw/{filename}", exist_ok=True)
        self._lcr_measurements.to_csv(
            f"{root_dir}/raw/{filename}/lcr_measurements.csv", index=False
        )
        self._adc_measurements.to_csv(
            f"{root_dir}/raw/{filename}/adc_measurements.csv", index=False
        )
        self._poi_measurements.to_csv(
            f"{root_dir}/raw/{filename}/poi_measurements.csv", index=False
        )

        metadata_dict = {
            "id": self.id,
            "timestamp": self.timestamp,
            "poll_parameters": dataclasses.asdict(self._poll_parameters),
            "trial_info": (
                dataclasses.asdict(self.trial_info) if self.trial_info else None
            ),
            "raw_data_files": {
                "lcr_measurements": f"{root_dir}/raw/{filename}/lcr_measurements.csv",
                "adc_measurements": f"{root_dir}/raw/{filename}/adc_measurements.csv",
                "poi_measurements": f"{root_dir}/raw/{filename}/poi_measurements.csv",
            },
        }

        with open(f"{root_dir}/{filename}.json", "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f, indent=2, cls=CustomJSONEncoder)

    def _prep_file_prefix(self):
        """Allocate next poll index and generate timestamped filename stem."""
        now = dt.now()
        with open("test_id.txt", "r+", encoding="utf-8") as f:
            test_id = int(f.read())
            f.seek(0)
            f.write(str(test_id + 1))
        return (
            test_id,
            f"PL_{test_id:04d}_{self.poll_parameters.trial_info.resin_type}_{now.strftime("%H%M")}",
        )


@dataclass
class POIRecord:
    """Typed representation for a point-of-interest event."""

    label: str
    value: float
    timestamp: float
