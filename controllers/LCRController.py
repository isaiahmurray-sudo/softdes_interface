"""LCR controller abstractions and implementations for debug and hardware modes."""

import pyvisa
import math
import re
from enum import Enum
import time
import logging


class MeasurementMode(Enum):
    CAPACITANCE = "CPD"
    INDUCTANCE = "LCR"
    RESISTANCE = "RX"
    IMPEDANCE = "ZTH"


class MeasurementSpeed(Enum):
    FAST = "FAST"
    SLOW = "SLOW"


class LCRController:
    """Base interface for configuring and sampling an LCR instrument."""

    def __init__(self):
        """Initialize connection state and instrument handle."""
        self._connected = False
        self.instrument = None

    @property
    def connected(self):
        return self._connected

    @property
    def debug(self):
        return True

    def configureMeasurement(
        self, frequency: float, mode: MeasurementMode, speed: MeasurementSpeed
    ):
        """Configure measurement mode, frequency, and speed on the device."""
        pass

    def measure(self):
        """Return a single measurement sample from the instrument."""
        pass

    def disconnect(self):
        """Close any active connection to the instrument."""
        pass

    def connect(self):
        """Open a connection to the instrument transport."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


class LCRControllerDebug(LCRController):
    """Simulated LCR implementation that emits deterministic sine-wave values."""

    def __init__(self):
        """Initialize configuration fields and mark debug controller connected."""
        self.mode: MeasurementMode = None
        self.speed: MeasurementSpeed = None
        self.frequency = None
        self._connected = True

    @property
    def connected(self):
        return self._connected

    @property
    def debug(self):
        return True

    def configureMeasurement(
        self, frequency: float, mode: MeasurementMode, speed: MeasurementSpeed
    ):
        self.frequency = frequency
        self.mode = mode
        self.speed = speed

    def measure(self):
        if self.mode is None:
            raise Exception("Measurement mode not configured.")
        return {
            "Z": self._dummy_impedance(),
            "Phase": self._dummy_phase(),
            "Time": time.time(),
        }

    def disconnect(self):
        self._connected = False

    def connect(self, com_port):
        logging.info(f"Simulated connection to LCRController on {com_port}")
        self._connected = True
        logging.info("Simulated connection established.")

    def _dummy_impedance(self):
        return math.sin(time.time() / 10)

    def _dummy_phase(self):
        return math.sin(time.time() / 15) * 90


class LCRControllerHardware(LCRController):
    """PyVISA-backed LCR implementation for serial COM communication."""

    def __init__(self, com_port: str | None = None):
        """Create VISA resource manager and optionally connect to a COM port."""
        super().__init__()
        self.rm = pyvisa.ResourceManager()

        try:
            if com_port is not None:
                self.connect(com_port)
        except Exception as e:
            self.instrument = None
            logging.error(f"Failed to connect to instrument: {e}")
            raise e

        self.frequency = None
        self.mode: MeasurementMode = None
        self.speed: MeasurementSpeed = None

    @staticmethod
    def _normalize_com_port(com_port: str) -> str:
        """Normalize COM values like ``COM4`` to ``4``."""
        value = (com_port or "").strip().upper()
        if value.startswith("COM"):
            value = value[3:]
        value = value.strip()
        if not value:
            raise ValueError("COM port cannot be empty.")
        return value

    @property
    def connected(self):
        return self._connected

    @property
    def debug(self):
        return False

    def _write(self, command: str):
        """Write a SCPI command if connected, otherwise log a critical error."""
        if self.connected:
            self.instrument.write(command)
        else:
            logging.critical("Attempted to write to instrument while not connected.")

    def _query(self, command: str):
        """Send a SCPI query and return the instrument response text."""
        if self.connected:
            return self.instrument.query(command)
        else:
            logging.critical("Attempted to query instrument while not connected.")
            return ""

    def is_connected(self):
        """
        Checks if the instrument is still connected by sending an *IDN? query.
        """
        try:
            idn = self._query("*IDN?")
            return True
        except:
            return False

    def configureMeasurement(
        self, frequency: float, mode: MeasurementMode, speed: MeasurementSpeed
    ):
        """Send SCPI commands needed to configure an impedance measurement."""
        self._write(f"MEAS:FUNC {mode.value}")
        self._write(f"FREQuency {frequency}")
        self._write(f"MEAS:SPEED {speed.value}")
        self.frequency = frequency
        self.mode = mode
        self.speed = speed

    def measure(self):
        """Read and parse a measurement response into canonical numeric fields."""
        if self.mode is None:
            raise Exception("Measurement mode not configured.")

        response = self._query("MEASurement:RESUlt?")
        primary, secondary = self._parse_measurement_response(response)
        return {"Z": primary, "Phase": secondary, "Time": time.time()}

    def _parse_measurement_response(self, response: str):
        """Parse comma-separated primary/secondary values from instrument output."""
        parts = [part.strip() for part in response.strip().split(",")]
        if len(parts) < 2:
            raise ValueError(f"Unexpected measurement response: {response!r}")

        primary = self._parse_engineering_value(parts[0])
        secondary = self._parse_engineering_value(parts[1])
        return primary, secondary

    def _parse_engineering_value(self, value_text: str) -> float:
        """Convert engineering notation with units into a float value."""
        match = re.match(
            r"^([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*([A-Za-zµΩ]*)", value_text
        )
        if not match:
            raise ValueError(f"Could not parse numeric value from {value_text!r}")

        numeric_value = float(match.group(1))
        unit = match.group(2).lower().replace(" ", "")
        unit = unit.replace("ω", "ohm").replace("µ", "u")

        multipliers = {
            "": 1.0,
            "ohm": 1.0,
            "ohms": 1.0,
            "k": 1e3,
            "kohm": 1e3,
            "m": 1e-3,
            "u": 1e-6,
            "uohm": 1e-6,
            "deg": 1.0,
            "rad": 1.0,
        }

        if unit.startswith("mohm"):
            multiplier = 1e6
        elif unit in multipliers:
            multiplier = multipliers[unit]
        else:
            multiplier = 1.0

        return numeric_value * multiplier

    def connect(self, com_port: str = "4"):
        """Open the PyVISA serial resource for the configured COM port."""
        try:
            port = self._normalize_com_port(com_port)
            self.instrument = self.rm.open_resource(
                f"ASRL{port}::INSTR", write_termination="\n", read_termination="\n"
            )
            self.instrument.timeout = 10000
            self._connected = True
        except Exception as e:
            self._connected = False
            logging.error(f"Failed to connect to instrument: {e}")
            raise e

    def disconnect(self):
        """Close the instrument handle and update connection state."""
        try:
            self.instrument.close()
            self._connected = False
            logging.info("Instrument closed.")
        except Exception as e:
            logging.error(f"Failed to close instrument: {e}")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Run LCR controller diagnostics on a serial COM port."
    )
    parser.add_argument(
        "--com-port", default="4", help="Serial COM port number, default: 4"
    )
    args = parser.parse_args()

    logging.info("Starting LCR diagnostics on ASRL%s::INSTR", args.com_port)

    rm = pyvisa.ResourceManager()
    logging.info("Available VISA resources: %s", rm.list_resources())

    controller = LCRControllerHardware()
    try:
        controller.connect(args.com_port)
        logging.info("Connected: %s", controller.connected)
        logging.info("Debug mode: %s", controller.debug)
        logging.info("Instrument timeout: %s", controller.instrument.timeout)

        try:
            idn = controller._query("*IDN?")
            logging.info("*IDN? response: %r", idn)
        except Exception as exc:
            logging.exception("*IDN? query failed: %s", exc)

        try:
            controller.configureMeasurement(
                1000, MeasurementMode.IMPEDANCE, MeasurementSpeed.FAST
            )
            logging.info(
                "Measurement configured: frequency=%s, mode=%s, speed=%s",
                controller.frequency,
                controller.mode,
                controller.speed,
            )
        except Exception as exc:
            logging.exception("configureMeasurement failed: %s", exc)

        try:
            measurement = controller.measure()
            logging.info("MEAS:RESUL? response parsed as: %s", measurement)
        except Exception as exc:
            logging.exception("measure() failed: %s", exc)

    finally:
        controller.disconnect()
