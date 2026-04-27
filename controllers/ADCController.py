"""ADC controller interfaces used by the polling pipeline.

The debug implementation emits fixed synthetic voltage values, while the
hardware implementation is currently a placeholder to preserve API shape.
"""

import time


class ADCController:
    """Base interface for retrieving ADC voltage samples."""

    def __init__(self):
        """Initialize the ADC controller base type."""
        pass

    def measure(self):
        """Return a single ADC sample dictionary."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class ADCControllerDebug(ADCController):
    """Debug ADC implementation that returns a constant voltage sample."""

    def measure(self):
        """Return a deterministic sample payload for testing."""
        return {"V": 1.23, "Time": time.time()}


class ADCControllerHardware(ADCController):
    """Placeholder hardware ADC implementation."""

    def measure(self):
        """Return a default hardware-shaped sample payload."""
        return {"V": 0.0, "Time": time.time()}
