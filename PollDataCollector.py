"""
GUI application for collecting poll data from Leash, LCR, and ADC controllers.
Features:
- User-friendly interface to configure poll parameters (exposure schedule, frequency, push settings)
- Real-time progress updates and live graphing of LCR and ADC data during polling
- Automatic saving of poll data to JSON files with timestamped filenames

(Ideally this should be ignored, this file may be the application but isn't meant to represent any
part of the assignment. It's just a way to view the data live and save it in a more convenient way than the Jupyter notebooks.)
"""

import sys
import json
import glob
import os
import logging
from datetime import datetime
import numpy as np

from PyQt6.QtCore import QSettings, QTimer, pyqtSignal, QObject
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QCheckBox,
    QScrollArea,
    QGroupBox,
    QPushButton,
    QLineEdit,
    QLabel,
    QFileDialog,
    QTextEdit,
    QComboBox,
    QListWidget,
    QAbstractItemView,
    QProgressBar,
    QSplitter,
    QDialog,
    QFormLayout,
    QDialogButtonBox,
)
from PyQt6.QtCore import Qt

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pyvisa

from controllers.LeashController import (
    LeashControllerHardware,
    LeashControllerDebug,
    ExposureParameter,
    ExposureImage,
)
from controllers.LCRController import (
    LCRControllerHardware,
    LCRControllerDebug,
    MeasurementMode,
    MeasurementSpeed,
)
from controllers.ADCController import ADCControllerHardware, ADCControllerDebug
from datastructs.PollData import PollParameters, TrialInfo
from PollController import PollController

DEFAULT_LCR_COM_PORT = "4"


class PollWorker(QObject):
    """Qt helper object that starts a poll and emits progress/completion signals."""

    progress_updated = pyqtSignal(float)
    poll_finished = pyqtSignal()

    def __init__(self, poll_controller, poll_params):
        """Store poll dependencies and initialize a progress timer."""
        super().__init__()
        self.poll_controller = poll_controller
        self.poll_params = poll_params
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_progress)

    def start_poll(self):
        """Start polling asynchronously and forward completion to Qt signals."""

        def on_completion():
            self.timer.stop()
            self.poll_finished.emit()

        self.poll_controller.start_poll(
            self.poll_params, wait_for_completion=False, on_completion=on_completion
        )
        self.timer.start(100)  # Still keep timer for progress updates

    def check_progress(self):
        """Emit current normalized progress from the backing controller."""
        progress = self.poll_controller.progress
        self.progress_updated.emit(progress)


class PollDataCollectorApp(QMainWindow):
    """Main GUI window used to configure, run, and visualize polling trials."""

    def __init__(self):
        """Initialize UI state, timers, and persisted settings."""
        super().__init__()
        self.setWindowTitle("Poll Data Collector")
        self.setGeometry(100, 100, 1600, 700)

        self.poll_controller = None
        self.poll_worker = None
        self.current_poll_data = None
        self.exposure_segments = []  # List of (duration_edit, intensity_edit, layout)
        self.leash_ip = "10.35.14.234"
        self.lcr_com_port = DEFAULT_LCR_COM_PORT
        self.graph_update_timer = QTimer()
        self.graph_update_timer.timeout.connect(self.update_graphs)

        self.init_ui()
        self.load_settings()

    def init_ui(self):
        """Construct the full Qt layout for controls, notes, and live plots."""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Sidebar
        sidebar_group = QGroupBox("Poll Settings")
        sidebar_layout = QVBoxLayout(sidebar_group)

        # Debug mode
        self.debug_checkbox = QCheckBox("Debug Mode (Simulated Hardware)")
        self.debug_checkbox.setChecked(True)
        sidebar_layout.addWidget(self.debug_checkbox)

        self.scan_devices_button = QPushButton("Scan USB / VISA Devices")
        self.scan_devices_button.clicked.connect(self.scan_usb_visa_devices)
        sidebar_layout.addWidget(self.scan_devices_button)

        self.connection_settings_button = QPushButton("Connection Settings")
        self.connection_settings_button.clicked.connect(self.open_connection_settings)
        sidebar_layout.addWidget(self.connection_settings_button)

        self.connection_settings_label = QLabel("")
        self.connection_settings_label.setWordWrap(True)
        sidebar_layout.addWidget(self.connection_settings_label)

        # Trial Info
        sidebar_layout.addWidget(QLabel("Trial Name:"))
        self.trial_name_input = QLineEdit("Default Trial")
        sidebar_layout.addWidget(self.trial_name_input)

        sidebar_layout.addWidget(QLabel("Resin Type:"))
        self.resin_dropdown = QComboBox()
        self.resin_dropdown.addItems(
            [
                "Gray",
                "White",
                "Black",
                "FastModel",
                "FlameRetardant",
                "Durable",
                "HighTemp",
                "Rigid10K",
                "SurgicalGuide",
                "Clear",
                "PrecisionModel",
                "MRC201",
                "Unknown",
            ]
        )
        sidebar_layout.addWidget(self.resin_dropdown)

        # Exposure segments
        exposure_group = QGroupBox("Exposure Segments")
        exposure_layout = QVBoxLayout(exposure_group)

        self.segments_layout = QVBoxLayout()
        exposure_layout.addLayout(self.segments_layout)

        add_segment_button = QPushButton("Add Segment")
        add_segment_button.clicked.connect(self.add_exposure_segment)
        exposure_layout.addWidget(add_segment_button)

        # Default segments
        self.add_exposure_segment(duration=5.0, intensity=0.0)
        self.add_exposure_segment(duration=5.0, intensity=1.0)
        self.add_exposure_segment(duration=10.0, intensity=0.0)

        sidebar_layout.addWidget(exposure_group)

        # Frequency
        sidebar_layout.addWidget(QLabel("Frequency (Hz):"))
        self.frequency_input = QLineEdit("1000")
        sidebar_layout.addWidget(self.frequency_input)

        sidebar_layout.addWidget(QLabel("Minimum Exposure Time (s):"))
        self.minimum_exposure_time_input = QLineEdit("0")
        sidebar_layout.addWidget(self.minimum_exposure_time_input)

        # Push parameters
        sidebar_layout.addWidget(QLabel("Push Force:"))
        self.push_force_input = QLineEdit("70")
        sidebar_layout.addWidget(self.push_force_input)

        sidebar_layout.addWidget(QLabel("Push Repeats:"))
        self.push_repeats_input = QLineEdit("2")
        sidebar_layout.addWidget(self.push_repeats_input)

        sidebar_layout.addWidget(QLabel("Push Delay (s):"))
        self.push_delay_input = QLineEdit("3")
        sidebar_layout.addWidget(self.push_delay_input)

        # Start button
        self.start_button = QPushButton("Start Poll")
        self.start_button.setStyleSheet(
            "background-color: #2ecc71; color: white; font-weight: bold;"
        )
        self.start_button.clicked.connect(self.start_poll)
        sidebar_layout.addWidget(self.start_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        sidebar_layout.addWidget(self.progress_bar)

        sidebar_layout.addStretch()

        # Create main content area (notes + graphs)
        content_layout = QVBoxLayout()

        # Pre-test notes
        content_layout.addWidget(QLabel("Pre-Test Notes:"))
        self.pre_test_notes = QTextEdit()
        content_layout.addWidget(self.pre_test_notes, 1)

        # Post-test section (initially disabled)
        post_group = QGroupBox("Post-Test Input")
        post_layout = QVBoxLayout(post_group)

        post_layout.addWidget(QLabel("Post-Test Notes:"))
        self.post_test_notes = QTextEdit()
        self.post_test_notes.setEnabled(False)
        post_layout.addWidget(self.post_test_notes)

        post_layout.addWidget(QLabel("Thickness (mm):"))
        self.thickness_input = QLineEdit()
        self.thickness_input.setEnabled(False)
        post_layout.addWidget(self.thickness_input)

        self.save_button = QPushButton("Save Data")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_data)
        post_layout.addWidget(self.save_button)

        content_layout.addWidget(post_group)

        # Create graph area on the right
        graph_layout = QVBoxLayout()

        # LCR Graph
        lcr_group = QGroupBox("LCR Impedance (Z)")
        lcr_layout = QVBoxLayout(lcr_group)
        self.lcr_figure = Figure(figsize=(5, 3), dpi=100)
        self.lcr_canvas = FigureCanvas(self.lcr_figure)
        lcr_layout.addWidget(self.lcr_canvas)
        graph_layout.addWidget(lcr_group, 1)

        # ADC Graph
        adc_group = QGroupBox("ADC Voltage (V)")
        adc_layout = QVBoxLayout(adc_group)
        self.adc_figure = Figure(figsize=(5, 3), dpi=100)
        self.adc_canvas = FigureCanvas(self.adc_figure)
        adc_layout.addWidget(self.adc_canvas)
        graph_layout.addWidget(adc_group, 1)

        # Add sidebar and content to main layout
        main_layout.addWidget(sidebar_group, 0)
        main_layout.addLayout(content_layout, 2)
        main_layout.addLayout(graph_layout, 2)

        # Initialize empty graphs
        self.init_empty_graphs()
        self.update_connection_settings_label()

    def add_exposure_segment(self, duration=5.0, intensity=0.0):
        """Append an editable exposure segment row to the schedule panel."""
        segment_layout = QHBoxLayout()

        duration_edit = QLineEdit(str(duration))
        intensity_edit = QLineEdit(str(intensity))

        segment_layout.addWidget(QLabel("Duration (s):"))
        segment_layout.addWidget(duration_edit)
        segment_layout.addWidget(QLabel("Intensity (0-1):"))
        segment_layout.addWidget(intensity_edit)

        remove_button = QPushButton("Remove")
        remove_button.clicked.connect(
            lambda: self.remove_exposure_segment(segment_layout)
        )
        segment_layout.addWidget(remove_button)

        self.segments_layout.addLayout(segment_layout)
        self.exposure_segments.append((duration_edit, intensity_edit, segment_layout))

    def remove_exposure_segment(self, layout):
        """Remove a previously created exposure segment row from the UI."""
        for i, (dur, intens, lay) in enumerate(self.exposure_segments):
            if lay == layout:
                # Remove from layout
                while layout.count():
                    item = layout.takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()
                self.segments_layout.removeItem(layout)
                del self.exposure_segments[i]
                break

    def update_exposure_schedule(self):
        """Deprecated no-op kept for compatibility with older UI callbacks."""
        # This method is no longer needed since segments are editable directly
        pass

    def scan_usb_visa_devices(self):
        """Log currently visible VISA/USB resources for connection diagnostics."""
        rm = None
        try:
            rm = pyvisa.ResourceManager()
            resources = list(rm.list_resources())
            usb_resources = [res for res in resources if res.upper().startswith("USB")]

            lines = []
            lines.append("VISA backend initialized successfully.")
            lines.append("")
            lines.append(f"Total VISA resources: {len(resources)}")
            if resources:
                lines.extend([f"  - {res}" for res in resources])
            else:
                lines.append("  (none found)")

            lines.append("")
            lines.append(f"USB resources: {len(usb_resources)}")
            if usb_resources:
                lines.extend([f"  - {res}" for res in usb_resources])
            else:
                lines.append("  (none found)")

            logging.info("USB / VISA Device Scan:\n%s", "\n".join(lines))
        except Exception as e:
            logging.exception(
                "USB / VISA Device Scan Failed: could not enumerate VISA resources: %s",
                e,
            )
        finally:
            if rm is not None:
                try:
                    rm.close()
                except Exception:
                    pass

    def normalize_com_port(self, com_port_text, default=DEFAULT_LCR_COM_PORT):
        """Normalize COM text into a plain numeric port string."""
        value = "" if com_port_text is None else str(com_port_text)
        value = value.strip().upper()
        if value.startswith("COM"):
            value = value[3:]
        value = value.strip()

        if not value:
            return default
        if not value.isdigit():
            return default
        return str(int(value))

    def update_connection_settings_label(self):
        """Refresh the sidebar label that displays active connection settings."""
        self.connection_settings_label.setText(
            f"Leash IP: {self.leash_ip}\nLCR COM Port: {self.lcr_com_port}"
        )

    def open_connection_settings(self):
        """Open a modal dialog to edit Leash IP and LCR COM port values."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Connection Settings")
        layout = QVBoxLayout(dialog)

        form_layout = QFormLayout()
        leash_ip_input = QLineEdit(self.leash_ip)
        lcr_com_input = QLineEdit(self.lcr_com_port)
        form_layout.addRow("Leash IP Address:", leash_ip_input)
        form_layout.addRow("LCR COM Port:", lcr_com_input)
        layout.addLayout(form_layout)

        dialog_buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        dialog_buttons.accepted.connect(dialog.accept)
        dialog_buttons.rejected.connect(dialog.reject)
        layout.addWidget(dialog_buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        leash_ip = leash_ip_input.text().strip()
        com_port = self.normalize_com_port(lcr_com_input.text())
        if not leash_ip:
            logging.warning(
                "Invalid connection settings: leash IP address cannot be empty."
            )
            return

        self.leash_ip = leash_ip
        self.lcr_com_port = com_port
        self.update_connection_settings_label()
        self.save_current_settings()

    def init_empty_graphs(self):
        """Initialize graphs with empty placeholder state."""
        try:
            # Initialize LCR graph
            self.lcr_figure.clear()
            ax = self.lcr_figure.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "Ready to start poll...",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_title("LCR Data")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Impedance (Ω)")
            self.lcr_canvas.draw()

            # Initialize ADC graph
            self.adc_figure.clear()
            ax = self.adc_figure.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "Ready to start poll...",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_title("ADC Data")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Voltage (V)")
            self.adc_canvas.draw()
        except Exception as e:
            print(f"Error initializing graphs: {e}")

    def start_poll(self):
        """Collect UI parameters, initialize controllers, and start polling."""
        self.save_current_settings()

        # Get parameters
        debug = self.debug_checkbox.isChecked()
        trial_name = self.trial_name_input.text() or "Default Trial"
        resin_type = self.resin_dropdown.currentText()
        pre_notes = self.pre_test_notes.toPlainText()

        # Collect exposure segments
        exposure_params = []
        for duration_edit, intensity_edit, _ in self.exposure_segments:
            try:
                duration = float(duration_edit.text())
                intensity = float(intensity_edit.text())
                exposure_params.append(
                    ExposureParameter(
                        duration=duration, intensity=intensity, image=ExposureImage.V2
                    )
                )
            except ValueError:
                logging.warning("Invalid duration or intensity in exposure segments.")
                return

        try:
            frequency = int(self.frequency_input.text())
            minimum_exposure_time_s = float(self.minimum_exposure_time_input.text())
            push_force = float(self.push_force_input.text())
            push_repeats = int(self.push_repeats_input.text())
            push_delay = float(self.push_delay_input.text())
        except ValueError as e:
            logging.warning("Invalid numeric input: %s", e)
            return

        if minimum_exposure_time_s < 0:
            logging.warning("Minimum exposure time must be 0 or greater.")
            return

        trial_info = TrialInfo(
            trial_name=trial_name, resin_type=resin_type, pre_trial_notes=pre_notes
        )

        poll_params = PollParameters(
            exposure_parameters=exposure_params,
            frequency_hz=frequency,
            minimum_exposure_time_s=minimum_exposure_time_s,
            trial_info=trial_info,
            push_force=push_force,
            push_repeats=push_repeats,
            push_delay_s=push_delay,
        )

        # Initialize controllers
        try:
            leash = LeashControllerHardware() if not debug else LeashControllerDebug()
            adc = ADCControllerHardware() if not debug else ADCControllerDebug()
            lcr = LCRControllerHardware() if not debug else LCRControllerDebug()

            self.poll_controller = PollController(
                leash_controller=leash, lcr_controller=lcr, adc_controller=adc
            )

            # Connect if not debug
            if not debug:
                leash.connect(self.leash_ip)
                com_port = self.normalize_com_port(self.lcr_com_port)
                self.lcr_com_port = com_port
                lcr.connect(com_port)

            # Start poll in worker thread
            self.poll_worker = PollWorker(self.poll_controller, poll_params)
            self.poll_worker.progress_updated.connect(self.update_progress)
            self.poll_worker.poll_finished.connect(self.poll_completed)
            self.poll_worker.start_poll()

            # Set current poll data so graphs can update during polling
            self.current_poll_data = self.poll_controller.poll_data

            self.start_button.setEnabled(False)
            self.progress_bar.setValue(0)
            self.graph_update_timer.start(200)  # Update graphs every 200ms

        except Exception as e:
            logging.exception("Failed to initialize controllers: %s", e)
            self.start_button.setEnabled(True)

    def update_progress(self, progress):
        """Update the progress bar from a normalized float value."""
        self.progress_bar.setValue(int(progress * 100))

    def poll_completed(self):
        """Finalize UI state after polling and trigger automatic save."""
        self.current_poll_data = self.poll_controller.poll_data
        self.graph_update_timer.stop()
        self.update_graphs()  # Final update
        self._auto_save_poll_data()
        self.start_button.setEnabled(True)

    def update_graphs(self):
        """Update LCR and ADC graphs with latest poll data."""
        if self.current_poll_data is None:
            return

        try:
            # Update LCR graph with impedance, phase, and ionic viscosity
            self.lcr_figure.clear()
            ax1 = self.lcr_figure.add_subplot(111)

            lcr_data = self.current_poll_data.lcr_measurements
            if len(lcr_data) > 0:
                times = lcr_data["Time"].values
                times = times - times[0]  # Start from 0
                z_values = lcr_data["Z"].values
                phase_values = lcr_data["Phase"].values

                # Calculate ionic viscosity using the same algorithm as EasyDataViewer
                ionic_viscosity = self._compute_ionic_viscosity(z_values, phase_values)

                # Plot impedance on left axis (ax1)
                line1 = ax1.plot(
                    times,
                    z_values,
                    "b-",
                    linewidth=1.5,
                    alpha=0.8,
                    label="Impedance (Z)",
                )
                ax1.set_xlabel("Time (s)", fontweight="bold")
                ax1.set_ylabel("Impedance (Ω)", fontweight="bold", color="b")
                ax1.tick_params(axis="y", labelcolor="b")

                # Create right axis for phase (ax2)
                ax2 = ax1.twinx()
                line2 = ax2.plot(
                    times,
                    phase_values,
                    "g--",
                    linewidth=1.5,
                    alpha=0.8,
                    label="Phase (°)",
                )
                ax2.set_ylabel(
                    "Phase (°)", fontweight="bold", color="g", rotation=270, labelpad=15
                )
                ax2.tick_params(axis="y", labelcolor="g")

                # Create third axis for ionic viscosity (ax3), offset to the right
                ax3 = ax1.twinx()
                ax3.spines["right"].set_position(("outward", 60))
                line3 = ax3.plot(
                    times,
                    ionic_viscosity,
                    "r-",
                    linewidth=1.5,
                    alpha=0.8,
                    label="Ionic Viscosity",
                )
                ax3.set_ylabel(
                    "Ionic Viscosity",
                    fontweight="bold",
                    color="r",
                    rotation=270,
                    labelpad=25,
                )
                ax3.tick_params(axis="y", labelcolor="r")

                ax1.set_title(f"LCR Data ({len(lcr_data)} samples)", fontweight="bold")
                ax1.grid(True, alpha=0.3)

                # Combined legend
                lines = line1 + line2 + line3
                labels = [l.get_label() for l in lines]
                ax1.legend(lines, labels, loc="upper left", fontsize="small")
            else:
                ax1.text(
                    0.5,
                    0.5,
                    "Waiting for LCR data...",
                    ha="center",
                    va="center",
                    transform=ax1.transAxes,
                )
                ax1.set_title("LCR Data")

            self.lcr_figure.tight_layout()
            self.lcr_canvas.draw()
        except Exception as e:
            print(f"Error updating LCR graph: {e}")

        try:
            # Update ADC graph
            self.adc_figure.clear()
            ax = self.adc_figure.add_subplot(111)

            adc_data = self.current_poll_data.adc_measurements
            if len(adc_data) > 0:
                times = adc_data["Time"].values
                times = times - times[0]  # Start from 0
                v_values = adc_data["V"].values

                ax.plot(times, v_values, "r-", linewidth=1.5, label="Voltage (V)")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Voltage (V)")
                ax.set_title(f"ADC Data ({len(adc_data)} samples)")
                ax.grid(True, alpha=0.3)
                ax.legend()
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Waiting for ADC data...",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title("ADC Data")

            self.adc_canvas.draw()
        except Exception as e:
            print(f"Error updating ADC graph: {e}")

    def _auto_save_poll_data(self):
        """Persist finished poll data without requiring manual user action."""
        if self.poll_controller is None:
            return

        try:
            self.poll_controller.save_to_file()
            logging.info("Poll data saved automatically.")
        except Exception as e:
            logging.exception("Failed to auto-save poll data: %s", e)

    def _compute_ionic_viscosity(
        self,
        z_ohm: np.ndarray,
        phase_deg: np.ndarray,
        cell_constant_cm_inv: float = 1.0,
        walden_constant: float = 1.0,
    ) -> np.ndarray:
        """
        Compute ionic viscosity from impedance and phase.

        Based on ionic_viscosity_analysis.py implementation.
        Algorithm:
        1. Calculate parallel conductance from impedance and phase
        2. Calculate bulk conductivity using cell constant
        3. Calculate ionic viscosity using Walden constant

        Args:
            z_ohm: Impedance values in ohms
            phase_deg: Phase values in degrees
            cell_constant_cm_inv: Cell constant in cm^-1
            walden_constant: Walden constant for the material

        Returns:
            Array of ionic viscosity values
        """
        phase_rad = np.deg2rad(phase_deg)

        # 1. Calculate Parallel Conductance (Siemens)
        z_safe = np.maximum(np.abs(z_ohm), 1e-12)
        conductance_s = np.cos(phase_rad) / z_safe

        # Mask out invalid physical states (conductance must be positive)
        conductance_s = np.where(conductance_s > 0, conductance_s, np.nan)

        # 2. Calculate Bulk Conductivity (S/cm)
        conductivity_s_per_cm = conductance_s * cell_constant_cm_inv

        # 3. Calculate Ionic Viscosity using the Walden constant
        ionic_viscosity = walden_constant / conductivity_s_per_cm

        return ionic_viscosity

    def save_data(self):
        """Manually save poll data and reset parts of the interface state."""
        if self.current_poll_data is None or self.poll_controller is None:
            return

        try:
            self.poll_controller.save_to_file()
            logging.info("Poll data saved successfully.")
        except Exception as e:
            logging.exception("Failed to save poll data: %s", e)

        self.pre_test_notes.clear()
        self.progress_bar.setValue(0)
        self.init_empty_graphs()

    def save_current_settings(self):
        """Persist current UI configuration using ``QSettings``."""
        settings = QSettings("MyCompany", "PollDataCollector")
        settings.setValue("debug", self.debug_checkbox.isChecked())
        settings.setValue("trial_name", self.trial_name_input.text())
        settings.setValue("resin", self.resin_dropdown.currentText())
        settings.setValue("frequency", self.frequency_input.text())
        settings.setValue(
            "minimum_exposure_time_s", self.minimum_exposure_time_input.text()
        )
        settings.setValue("push_force", self.push_force_input.text())
        settings.setValue("push_repeats", self.push_repeats_input.text())
        settings.setValue("push_delay", self.push_delay_input.text())
        settings.setValue("leash_ip", self.leash_ip)
        settings.setValue("lcr_com_port", self.normalize_com_port(self.lcr_com_port))

    def load_settings(self):
        """Restore last-used UI configuration from ``QSettings``."""
        settings = QSettings("MyCompany", "PollDataCollector")
        self.debug_checkbox.setChecked(settings.value("debug", True, type=bool))
        self.trial_name_input.setText(settings.value("trial_name", "Default Trial"))
        self.resin_dropdown.setCurrentText(settings.value("resin", "Unknown"))
        self.frequency_input.setText(settings.value("frequency", "1000"))
        self.minimum_exposure_time_input.setText(
            settings.value("minimum_exposure_time_s", "0")
        )
        self.push_force_input.setText(settings.value("push_force", "70"))
        self.push_repeats_input.setText(settings.value("push_repeats", "2"))
        self.push_delay_input.setText(settings.value("push_delay", "3"))
        self.leash_ip = settings.value("leash_ip", "10.35.14.234")
        self.lcr_com_port = self.normalize_com_port(
            settings.value("lcr_com_port", DEFAULT_LCR_COM_PORT)
        )
        self.update_connection_settings_label()

    def closeEvent(self, event):
        """Save settings before shutdown and accept the close request."""
        self.save_current_settings()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PollDataCollectorApp()
    window.show()
    sys.exit(app.exec())
