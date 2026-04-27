"""Leash controller abstractions for debug and hardware-backed runtimes.

This module defines shared controller contracts, simulation behavior, and the
remote SSH-based hardware implementation used to execute Leash commands.
"""

import dataclasses
import json
import logging
import os
import socket
import threading
import time
import uuid
from enum import Enum
from pathlib import Path

import paramiko

try:
    from controllers.timing_recorder import TimingRecorder
except ImportError:
    try:
        from controllers.timing_recorder import TimingRecorder
    except ImportError:
        try:
            from softdes_interface.controllers.timing_recorder import TimingRecorder
        except ImportError as import_error:
            print(f"Error importing TimingRecorder: {import_error}")


class ConfidentialDataAccessError(RuntimeError):
    """Raised when local confidential runtime data is unavailable or invalid."""


CONFIDENTIAL_DIR_ENV_VAR = "LEASH_CONFIDENTIAL_DIR"
DEFAULT_CONFIDENTIAL_DIR = ".local_confidential"
DEBUG_STATUS_FILENAME = "debug_status.json"
COMMAND_TEMPLATES_FILENAME = "leash_command_templates.json"


def _get_confidential_dir() -> Path:
    """Return the directory where confidential runtime JSON payloads are stored."""
    configured = os.environ.get(CONFIDENTIAL_DIR_ENV_VAR)
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parent / DEFAULT_CONFIDENTIAL_DIR


def _load_confidential_json(filename: str, description: str) -> dict:
    """Load a confidential JSON object and validate its basic shape."""
    path = _get_confidential_dir() / filename
    if not path.exists():
        raise ConfidentialDataAccessError(
            f"Missing {description} at '{path}'. No access to confidential data."
        )

    try:
        with path.open("r", encoding="utf-8") as infile:
            payload = json.load(infile)
    except json.JSONDecodeError as exc:
        raise ConfidentialDataAccessError(
            f"Invalid JSON in {description} at '{path}'. No access to confidential data."
        ) from exc

    if not isinstance(payload, dict):
        raise ConfidentialDataAccessError(
            f"Invalid {description} at '{path}': expected a JSON object."
        )

    return payload


class ExposureImage(Enum):
    """Image options for exposure measurement."""

    V1 = "test.png"
    V2 = "test_square_v2.png"


@dataclasses.dataclass
class ExposureParameter:
    """Exposure configuration describing duration, intensity, and image."""

    duration: float
    intensity: float
    image: ExposureImage

    def to_dict(self):
        """Serialize this parameter into a JSON-friendly dictionary."""
        return {
            "duration": self.duration,
            "intensity": self.intensity,
            "image": self.image.value,
        }


class LeashController:
    """Base interface for Leash controller implementations."""

    def __init__(self):
        """Initialize shared timing recorder and connection/status state."""
        self._timing_recorder = TimingRecorder("LeashControllerTiming.json")
        self._status = None
        self._status_timestamp = None
        self._connected = False

    @property
    def status_timestamp(self):
        """Get the timestamp when status was last updated."""
        return self._status_timestamp

    @property
    def time_since_last_status(self):
        """Get the time elapsed since the last status update."""
        if self._status_timestamp is None:
            return None
        return time.time() - self._status_timestamp

    @property
    def connected(self):
        """Check if the controller is connected."""
        return self._connected

    @property
    def debug(self):
        """Check if running in debug mode."""
        return True

    @property
    def status(self):
        """Get the current status payload."""
        return self._status

    def expose(self, parameters: ExposureParameter):
        """Execute an exposure with the given parameters."""
        raise NotImplementedError

    def set_z_position(self, position: float):
        """Set the Z position to the specified value."""
        raise NotImplementedError

    def press_z(self, force: float):
        """Press Z with the specified force."""
        raise NotImplementedError

    def get_z_position(self):
        """Get the current Z position."""
        raise NotImplementedError

    def get_status(self):
        """Return the most recent status payload and refresh status timestamp."""
        self._status_timestamp = time.time()
        return self._status

    def disconnect(self):
        """Disconnect from the controller."""
        raise NotImplementedError

    def connect(self, ip):
        """Connect to the controller at the specified IP address."""
        raise NotImplementedError

    def load_image(self, image: ExposureImage):
        """Load an image for exposure operations."""
        raise NotImplementedError

    def start_leash_session(self):
        """Start a Leash session."""
        raise NotImplementedError

    def close_leash_session(self):
        """Close an active Leash session."""
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._timing_recorder.save_records()
        self.close_leash_session()
        self.disconnect()


class LeashControllerDebug(LeashController):
    """Simulated Leash implementation that uses recorded timing delays."""

    def __init__(self):
        """Initialize timing-derived delays and local mock Z position."""
        super().__init__()
        self._timing_recorder.calculate_delay_cache()
        self._zpos = 0.0

    def expose(self, parameters: ExposureParameter):
        """Execute exposure with the given parameters."""
        logging.info("Exposing with parameters: %s", parameters)
        self._timing_recorder.delay("expose")
        logging.info("Exposure complete.")

    def set_z_position(self, position: float):
        """Set Z position to the specified value."""
        logging.info("Setting Z position to: %s", position)
        self._timing_recorder.delay("set_z_position")
        logging.info("Z position set.")
        self._zpos = position

    def press_z(self, force: float):
        """Press Z with the specified force."""
        logging.info("Pressing Z with force: %s", force)
        self._timing_recorder.delay("press_z")
        self._zpos = 0.1
        logging.info("Press complete.")

    def load_image(self, image: ExposureImage):
        """Load an image for exposure."""
        logging.info("Loading image: %s", image)
        self._timing_recorder.delay("load_image")
        logging.info("Image loaded.")

    def start_leash_session(self):
        """Start a Leash session."""
        logging.info("Starting leash session.")
        self._timing_recorder.delay("start_leash_session")
        logging.info("Leash session started.")

    def close_leash_session(self):
        """Close a Leash session."""
        logging.info("Closing leash session.")
        self._timing_recorder.delay("close_leash_session")
        logging.info("Leash session closed.")

    def get_status(self):
        """Fetch current status from debug payload."""
        self._timing_recorder.delay("get_status")
        self._status = _load_confidential_json(
            DEBUG_STATUS_FILENAME, "debug status payload"
        )
        self._status_timestamp = time.time()
        return super().get_status()

    def get_z_position(self):
        """Get the current Z position."""
        self.get_status()
        return self._zpos

    def disconnect(self):
        """Disconnect from debug controller."""
        logging.info("Disconnecting LeashControllerDebug.")
        self._connected = False

    def connect(self, ip):
        """Establish a simulated connection."""
        logging.info("Simulated connection to LeashController at %s", ip)
        self._timing_recorder.delay("connect")
        logging.info("Simulated connection established.")
        logging.info("LeashControllerDebug connected.")
        self._connected = True


# pylint: disable=too-many-instance-attributes
class LeashControllerHardware(LeashController):
    """SSH-backed Leash implementation that executes remote Python commands."""

    def __init__(self):
        """Prepare SSH state, command locking, and command templates."""
        super().__init__()
        self._ip = None
        self._port = None
        self.timeout = 10.0
        self._client = None
        self._command_lock = threading.Lock()
        self._loaded_image: ExposureImage | None = None
        self._leash_channel = None
        self._command_templates = _load_confidential_json(
            COMMAND_TEMPLATES_FILENAME, "leash command templates"
        )

    def _get_command_template(self, key: str) -> str:
        """Fetch a command template by key and validate that it is non-empty."""
        template = self._command_templates.get(key)
        if not isinstance(template, str) or not template.strip():
            raise ConfidentialDataAccessError(
                f"Missing command template '{key}' in {COMMAND_TEMPLATES_FILENAME}."
            )
        return template

    def _render_command_template(self, key: str, **kwargs) -> str:
        """Render a configured command template with runtime placeholders."""
        template = self._get_command_template(key)
        try:
            return template.format(**kwargs)
        except KeyError as exc:
            placeholder = exc.args[0]
            raise ConfidentialDataAccessError(
                f"Command template '{key}' is missing placeholder value '{placeholder}'."
            ) from exc

    @property
    def client(self):
        """Get the SSH client instance."""
        return self._client

    def connect(
        self,
        ip,
        port="22",
        timeout=10,
        fetch_status=True,
        initialize_leash_session=True,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Open SSH transport, optionally initialize session, and fetch status."""
        task_id = self._timing_recorder.start_record("connect")
        if self._connected:
            self.disconnect()

        try:
            self.timeout = float(timeout)
            sock = socket.create_connection((ip, port), timeout=timeout)
            transport = paramiko.Transport(sock)
            transport.start_client(timeout=timeout)
            transport.auth_none("root")

            self._client = paramiko.SSHClient()
            # Note: using _transport as paramiko doesn't expose alternative
            self._client._transport = transport  # pylint: disable=protected-access
            self._ip = ip
            self._port = port
            self._connected = True
            self._timing_recorder.end_record(task_id)
            logging.info("LeashController connected.")

            if initialize_leash_session:
                self.start_leash_session()

            if fetch_status:
                self.get_status()
                logging.info("Initial status fetched.")
        except OSError as connection_error:
            logging.error("Failed to connect to LeashController: %s", connection_error)
            self.close_leash_session()
            if self._client is not None:
                try:
                    self._client.close()
                except OSError:
                    pass
                self._client = None
            self._connected = False
            self._timing_recorder.end_record(task_id)
            raise

    def disconnect(self):
        """Close any active session/channel and release SSH client resources."""
        self.close_leash_session()
        if self._client is not None:
            self._client.close()
            logging.info("LeashController disconnected.")
            self._client = None
        self._connected = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self._timing_recorder.save_records()
        finally:
            self.disconnect()
        return False

    def _run_command(self, command: str) -> str:
        """Run a one-off remote shell command and return stdout text."""
        if not self.connected:
            raise RuntimeError("Not connected to LeashController.")

        _, stdout, stderr = self._client.exec_command(command)
        exit_code = stdout.channel.recv_exit_status()
        output = stdout.read().decode().strip()
        error = stderr.read().decode().strip()
        if exit_code != 0:
            logging.error("Command failed: %s\nError: %s", command, error)
            raise RuntimeError(f"Remote command failed: {error}")
        return output

    def _wait_for_marker(self, marker: str, timeout: float | None) -> str:
        """Read shell output until a unique marker string is observed."""
        if self._leash_channel is None:
            raise RuntimeError("Leash session is not active")

        deadline = None if timeout is None else (time.time() + timeout)
        chunks: list[str] = []

        while True:
            if deadline is not None and time.time() >= deadline:
                break

            if self._leash_channel.recv_ready():
                chunk = self._leash_channel.recv(65535)
                if not chunk:
                    raise RuntimeError("Leash session closed unexpectedly")
                text = chunk.decode(errors="replace")
                chunks.append(text)
                combined = "".join(chunks)
                if marker in combined:
                    return combined
            elif self._leash_channel.closed:
                raise RuntimeError("Leash session closed unexpectedly")
            else:
                time.sleep(0.01)

        raise TimeoutError(f"Timed out waiting for leash marker: {marker}")

    def _clean_repl_output(
        self, raw_output: str, marker: str, python_lines: list[str]
    ) -> str:
        """Remove REPL echoes and marker tokens from session command output."""
        output = raw_output.replace("\r", "")
        output = output.replace(marker, "")

        cleaned_lines: list[str] = []
        command_lines = set(python_lines)

        for line in output.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped in {">>>", "..."}:
                continue
            if stripped.startswith(">>>") or stripped.startswith("..."):
                stripped = stripped[3:].strip()
                if not stripped:
                    continue
            if stripped in command_lines:
                continue
            cleaned_lines.append(stripped)

        return "\n".join(cleaned_lines)

    def start_leash_session(self):
        """Start an interactive remote Leash session channel if needed."""
        if not self.connected:
            raise RuntimeError("Cannot start leash session: not connected.")

        if self._leash_channel is not None and not self._leash_channel.closed:
            return

        task_id = self._timing_recorder.start_record("start_leash_session")
        try:
            self._leash_channel = self._client.invoke_shell(width=200, height=40)
            time.sleep(0.1)
            while self._leash_channel.recv_ready():
                self._leash_channel.recv(65535)

            enter_session_command = self._get_command_template("enter_leash_session")
            self._leash_channel.send(f"{enter_session_command}\n")
            self._loaded_image = None
            ready_marker = f"__LEASH_READY_{uuid.uuid4().hex}__"
            self._leash_channel.send(f'print("{ready_marker}")\n')
            self._wait_for_marker(ready_marker, timeout=max(5.0, self.timeout))
            logging.info("Leash session started successfully.")
            self._timing_recorder.end_record(task_id)
        except OSError as session_error:
            self.close_leash_session()
            logging.error("Failed to start leash session: %s", session_error)
            self._timing_recorder.delete_record(task_id)
            raise RuntimeError("Failed to start leash session.") from session_error

    def close_leash_session(self):
        """Close the interactive Leash session channel if one is open."""
        task_id = self._timing_recorder.start_record("close_leash_session")
        if self._leash_channel is not None:
            try:
                self._leash_channel.close()
            except OSError:
                pass
            self._loaded_image = None
            self._leash_channel = None
        self._timing_recorder.end_record(task_id)

    def run_leash_persistent_python(
        self, python_lines: list[str], timeout: float | None = None
    ) -> str:
        """Run python lines in a long-lived remote leash runtime.

        timeout=None waits indefinitely for the command marker.
        """
        with self._command_lock:
            for attempt in (1, 2):
                self.start_leash_session()
                marker = f"__LEASH_DONE_{uuid.uuid4().hex}__"

                for line in python_lines:
                    self._leash_channel.send(f"{line}\n")
                self._leash_channel.send(f'print("{marker}")\n')

                try:
                    raw_output = self._wait_for_marker(marker, timeout=timeout)
                except RuntimeError as channel_error:
                    if "closed unexpectedly" in str(channel_error) and attempt == 1:
                        self.close_leash_session()
                        logging.warning(
                            "Leash session was closed unexpectedly; restarting once."
                        )
                        continue
                    raise RuntimeError(
                        f"Failed to run leash python command: {channel_error}"
                    ) from channel_error

                output = self._clean_repl_output(raw_output, marker, python_lines)

                if "Traceback (most recent call last):" in output:
                    raise RuntimeError(f"Remote leash python error:\n{output}")

                return output

            raise RuntimeError(
                "Failed to run leash python command after reconnect attempt"
            )

    def _run_leash_python(self, python_lines: list[str]) -> str:
        """Run python statements as a one-off leash process."""
        script = "\n".join(python_lines)
        command = f"leash <<'EOF'\n{script}\nEOF"
        return self._run_command(command)

    def load_image(self, image: ExposureImage):
        """Load an image into the remote Leash runtime for exposure."""
        task_id = self._timing_recorder.start_record("load_image")
        image_path = f"/data/images/{image.value}"
        quoted_path = json.dumps(image_path)
        self.run_leash_persistent_python(
            [self._render_command_template("load_image", image_path_json=quoted_path)],
            timeout=None,
        )
        self._loaded_image = image
        self._timing_recorder.end_record(task_id)

    def expose(self, parameters: ExposureParameter):
        """Apply intensity mask, load image, and trigger timed exposure."""
        task_id = self._timing_recorder.start_record("expose")

        self.run_leash_persistent_python(
            [
                self._render_command_template(
                    "set_calibration_mask", intensity=parameters.intensity
                )
            ],
            timeout=None,
        )

        self.load_image(parameters.image)

        result = self.run_leash_persistent_python(
            [
                self._render_command_template(
                    "expose_image", duration=parameters.duration
                )
            ],
            timeout=None,
        )

        self._timing_recorder.end_record(task_id)
        return result

    def set_z_position(self, position: float):
        """Set absolute Z position through a one-off Leash command."""
        task_id = self._timing_recorder.start_record("set_z_position")

        result = self._run_leash_python(
            [self._render_command_template("set_z_position", position=position)]
        )

        self._timing_recorder.end_record(task_id)
        return result

    def press_z(self, force: float):
        """Execute a force-controlled Z press command."""
        task_id = self._timing_recorder.start_record("press_z")

        result = self._run_leash_python(
            [self._render_command_template("press_z", force=force)]
        )

        self._timing_recorder.end_record(task_id)
        return result

    def get_z_position(self):
        """Return Z position extracted from a freshly queried status payload."""
        status_payload = self.get_status()
        return status_payload.get("z_position")

    def get_status(self, try_run_persistent=True, attempts=3):
        """Fetch and parse Leash status, retrying persistent mode when enabled."""
        time_start = time.time()
        logging.info("Fetching status from LeashController...")
        status_python_line = self._get_command_template("get_status_python_line")

        def process_status_output(raw_output: str) -> dict:
            try:
                response = raw_output.split("___partition___")[1]
                return json.loads(response)
            except json.JSONDecodeError as json_error:
                logging.error(
                    "Failed to parse status output: %s, raw output: %s",
                    json_error,
                    raw_output,
                )
                raise RuntimeError("Failed to parse status output") from json_error

        task_id = self._timing_recorder.start_record("get_status")
        if try_run_persistent:
            for attempt_num in range(attempts):
                try:
                    output = self.run_leash_persistent_python(
                        [status_python_line], timeout=3.0
                    )
                    status_payload = process_status_output(output)
                    self._timing_recorder.end_record(task_id)
                    self._status_timestamp = time.time()
                    self._status = status_payload
                    logging.info(
                        "Status fetched successfully using persistent leash session."
                        " (Duration: %.3fs, attempt %d/%d)",
                        time.time() - time_start,
                        attempt_num + 1,
                        attempts,
                    )
                    return self._status
                except OSError as status_error:
                    logging.error(
                        "Failed to get status from LeashController from persistent"
                        " session: %s",
                        status_error,
                    )
        response = self._run_command(f"leash <<'EOF'\n{status_python_line}\nEOF")
        status_payload = process_status_output(response)
        self._status_timestamp = time.time()
        self._status = status_payload
        self._timing_recorder.end_record(task_id)
        logging.info(
            "Status fetched successfully using one-off leash command."
            " (Duration: %.2fs)",
            time.time() - time_start,
        )
        return status_payload


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    with LeashControllerHardware() as controller:

        controller.connect("10.35.14.234")
        status = controller.get_status()

        while True:
            status = controller.get_status()
