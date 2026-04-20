"""Core polling orchestrator for Leash, LCR, and ADC controllers.

This module coordinates exposure scheduling, background sampling threads, data
queue processing, progress reporting, and persistence through ``PollData``.
"""

import json
import pandas as pd
import numpy as np

from controllers.LeashController import LeashController, ExposureParameter, ExposureImage
from controllers.LCRController import LCRController
from controllers.ADCController import ADCController
from controllers.TimingRecorder import TimingRecorder
from datastructs.PollData import PollData, PollParameters, TrialInfo
from threading import Thread
from queue import Queue
import logging
import time

POLLING_INTERVAL_SECONDS = 0.025

class PollController:
    """Coordinate a complete polling run across all controller interfaces."""

    def __init__(self, leash_controller: LeashController, lcr_controller: LCRController, adc_controller: ADCController):
        """Store controller dependencies and initialize poll state."""
        self._leash_controller : LeashController = leash_controller
        self._lcr_controller : LCRController = lcr_controller
        self._adc_controller : ADCController = adc_controller
        self._progress: float = 0.0
        self._poll_data: PollData | None = None
        self.polling = False

    @property
    def poll_data(self) -> PollData | None:
        """Return current poll data once a poll has been started."""
        if self._poll_data is None:
            raise ValueError("Poll data is not initialized until a poll is started.")
        return self._poll_data

    @property
    def progress(self) -> float:
        """Return normalized poll progress in the range [0.0, 1.0]."""
        return self._progress

    def get_analysis_dataframe(self, cell_constant_cm_inv: float = 1.0, walden_constant: float = 1.0) -> pd.DataFrame:
        """
        Generate a pandas DataFrame with impedance, phase, and calculated ionic viscosity.
        
        Args:
            cell_constant_cm_inv: Cell constant in cm^-1 (default: 1.0)
            walden_constant: Walden constant for ionic viscosity calculation (default: 1.0)
            
        Returns:
            DataFrame with columns: Time, Impedance_Ohm, Phase_Deg, Ionic_Viscosity
            
        Raises:
            ValueError: If no LCR measurements are available.
        """
        lcr_df = self._poll_data._lcr_measurements
        
        if lcr_df.empty:
            raise ValueError("No LCR measurements available. Has the poll been started and completed?")
        
        # Extract arrays from DataFrame
        times = lcr_df["Time"].values
        z_ohm = lcr_df["Z"].values
        phase_deg = lcr_df["Phase"].values
        
        # Calculate ionic viscosity
        ionic_viscosity = self._compute_ionic_viscosity(z_ohm, phase_deg, cell_constant_cm_inv, walden_constant)
        
        # Create analysis DataFrame
        analysis_df = pd.DataFrame({
            'Time': times,
            'Impedance_Ohm': z_ohm,
            'Phase_Deg': phase_deg,
            'Ionic_Viscosity': ionic_viscosity
        })
        
        return analysis_df

    def _compute_ionic_viscosity(self, z_ohm: np.ndarray, phase_deg: np.ndarray,
                                  cell_constant_cm_inv: float = 1.0, walden_constant: float = 1.0) -> np.ndarray:
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

    def start_poll(self, poll_parameters: PollParameters, wait_for_completion: bool = True, on_completion: callable = None):
        """Start a polling session and spawn worker threads for data collection.

        Args:
            poll_parameters: Poll settings and exposure schedule.
            wait_for_completion: If True, block until worker threads end.
            on_completion: Optional callback invoked when polling completes.
        """
        logging.info("Starting poll with parameters: %s", poll_parameters)
        self.polling = True
        self._on_completion = on_completion

        if not poll_parameters.exposure_parameters or len(poll_parameters.exposure_parameters) == 0:
            raise ValueError("At least one exposure parameter must be provided.")

        self._enforce_minimum_exposure_time(poll_parameters)
        
        self._poll_data = PollData(poll_parameters)

        logging.info("Poll data initialized. Starting Leash session and loading image.")
        self._leash_controller.start_leash_session()
        self._leash_controller.load_image(poll_parameters.exposure_parameters[0].image)
        logging.info("Leash session started and image loaded. Configuring LCR controller.")
        self._lcr_controller.configureMeasurement(poll_parameters.frequency_hz, poll_parameters.MeasurementMode, poll_parameters.MeasurementSpeed)
        logging.info("LCR controller configured. Starting measurements.")

        self._leash_controller.set_z_position(70)
        for _ in range(1,self._poll_data._poll_parameters.push_repeats):
            self._leash_controller.press_z(self._poll_data._poll_parameters.push_force)
            time.sleep(self._poll_data._poll_parameters.push_delay_s)

        overall_start_time = time.time()

        dataqueue = Queue()

        self._data_handler_thread = Thread(target=self._handle_data, args=(dataqueue,))
        self._lcr_thread = Thread(target=self._poll_lcr_loop, args=(dataqueue,))
        self._adc_thread = Thread(target=self._poll_adc_loop, args=(dataqueue,))
        self._printer_thread = Thread(target=self._handle_printer, args=(dataqueue,))

        logging.info("Starting threads.")
        self._data_handler_thread.start()
        self._lcr_thread.start()
        self._adc_thread.start()
        self._printer_thread.start()
        logging.info("Threads started. Poll is now running.")
        logging.info(f"Threads started after {(time.time() - overall_start_time):.2f} seconds since poll start.")
        if wait_for_completion:
            logging.info("Waiting for poll completion.")
            self.wait_for_poll_completion()
            logging.info(f"Poll completed after {(time.time() - overall_start_time):.2f} seconds.")

    def _enforce_minimum_exposure_time(self, poll_parameters: PollParameters):
        """Pad exposure schedule with a zero-intensity segment when required."""
        min_duration = max(0.0, float(getattr(poll_parameters, "minimum_exposure_time_s", 0.0)))
        if min_duration <= 0:
            return

        schedule = poll_parameters.exposure_parameters
        scheduled_duration = sum(exp.duration for exp in schedule)
        if scheduled_duration >= min_duration:
            return

        buffer_duration = min_duration - scheduled_duration
        previous_image = schedule[-1].image
        schedule.append(
            ExposureParameter(duration=buffer_duration, intensity=0.0, image=previous_image)
        )
        logging.info(
            "Exposure schedule padded by %.2fs with zero intensity to satisfy minimum exposure time %.2fs.",
            buffer_duration,
            min_duration,
        )

    def save_to_file(self, filename: str | None = None):
        """Persist current poll data to disk using ``PollData`` defaults."""
        self._poll_data.save_to_file()
        logging.info("Poll data saved to file.")
    
    def wait_for_poll_completion(self, timeout: float | None = None):
        """Join worker threads and wait until polling has finished."""
        if not self.polling:
            logging.warning("wait_for_poll_completion called but polling is not active.")
            return
        try:
            self._printer_thread.join(timeout=timeout)
            self._lcr_thread.join(timeout=timeout)
            self._adc_thread.join(timeout=timeout)
            self._data_handler_thread.join(timeout=timeout)
        except AttributeError as e:
            logging.error(f"Ensure threads are initialized before calling join: {e}")

    def _handle_printer(self, dataqueue: Queue, status_interval: float | None = None):
        """Drive exposure timing and enqueue points-of-interest/status records."""
        start_time = time.time()
        dataqueue.put({"type": "poi", "label": "poll_start_time", "value": start_time})
        exposure_schedule = self._poll_data._poll_parameters.exposure_parameters
        total_poll_duration = sum([exp.duration for exp in exposure_schedule])

        while self.polling:
            current_time = time.time()
            elapsed_time = current_time - start_time
            self._progress = min(elapsed_time / total_poll_duration, 1.0)

            def handle_status(dataqueue: Queue, elapsed_time: float):
                raw_status = self._leash_controller.get_status()
                try:
                    status = {
                        "label": "printer_status",
                        "value": json.dumps({
                            "z_force": raw_status["info"]["forceAndPositionReadings"]["zForce"],
                            "z_position": raw_status["info"]["forceAndPositionReadings"]["zPosition"],
                        })
                    }
                    dataqueue.put({"type": "poi", "data": status})
                except Exception as e:
                    logging.error(f"Failed to parse status from LeashController: {e}")

            for i, exposure in enumerate(exposure_schedule):
                exp_start_time = sum([exp.duration for exp in exposure_schedule[:i]])
                if elapsed_time >= exp_start_time and not hasattr(exposure, "_started"):
                    if exposure.intensity > 0:
                        logging.info(f"Starting exposure for segment {i} at elapsed time {elapsed_time:.2f} seconds.")
                        dataqueue.put({"type": "poi", "data": {"label": f"exposure_{i}_call_started", "value": (current_time,exposure.duration, exposure.intensity)}})
                        self._leash_controller.expose(exposure)
                        dataqueue.put({"type": "poi", "data": {"label": f"exposure_{i}_call_complete", "value": (current_time, exposure.duration, exposure.intensity)}})
                        logging.info(f"Exposure call for segment {i} completed.")
                    else:
                        logging.info(f"Skipping exposure for segment #{i} due to zero intensity.")
                    setattr(exposure, "_started", True)
            if elapsed_time >= total_poll_duration:
                logging.info(f"Total poll duration of {total_poll_duration} seconds has elapsed. Ending poll.")
                self.polling = False
                logging.critical(f"controller is {self._leash_controller.connected}")
                self._leash_controller.set_z_position(200)
                if self._on_completion:
                    self._on_completion()
            if status_interval is not None and self._leash_controller.time_since_last_status is not None and self._leash_controller.time_since_last_status >= status_interval:
                handle_status(dataqueue, elapsed_time)
                time.sleep(POLLING_INTERVAL_SECONDS)
            else:
                handle_status(dataqueue, elapsed_time)
                


    def _handle_data(self, dataqueue: Queue):
        """Consume queued measurements and append them to ``PollData`` tables."""
        while self.polling or not dataqueue.empty():
            try:
                data_item = dataqueue.get(timeout=0.1)
                logging.debug(f"Handling data item of type {data_item['type']} with data {data_item['data']} from queue.")
                if data_item["type"] == "lcr":
                    logging.debug(f"Adding LCR measurement to poll data: {data_item['data']}")
                    self._poll_data.add_lcr_measurement(data_item["data"])
                elif data_item["type"] == "adc":
                    logging.debug(f"Adding ADC measurement to poll data: {data_item['data']}")
                    self._poll_data.add_adc_measurement(data_item["data"])
                elif data_item["type"] == "poi":
                    logging.debug(f"Adding POI measurement to poll data: label={data_item['data']['label']}, value={data_item['data']['value']}")
                    self._poll_data.add_poi_measurement(label=data_item["data"]["label"], value=data_item["data"]["value"])
                else:
                    logging.warning(f"Received unknown data type in data handler: {data_item['type']}")
            except Exception:
                if dataqueue.empty():
                    continue
                else:
                    logging.error(f"Error handling data item from queue", exc_info=True)

    def _poll_lcr_loop(self, dataqueue: Queue):
        """Continuously sample the LCR controller while polling is active."""
        while self.polling:
            try:
                measurement = self._lcr_controller.measure()
                dataqueue.put({"type": "lcr", "data": measurement})
            except Exception as e:
                logging.error(f"Error during LCR measurement: {e}")
            time.sleep(POLLING_INTERVAL_SECONDS)

    def _poll_adc_loop(self, dataqueue: Queue):
        """Continuously sample the ADC controller while polling is active."""
        while self.polling:
            try:
                measurement = self._adc_controller.measure()
                dataqueue.put({"type": "adc", "data": measurement})
            except Exception as e:
                logging.error(f"Error during ADC measurement: {e}")
            time.sleep(POLLING_INTERVAL_SECONDS)

if __name__ == "__main__":
    import argparse
    from controllers.LeashController import LeashControllerHardware, LeashControllerDebug
    from controllers.LCRController import LCRControllerHardware, LCRControllerDebug
    from controllers.ADCController import ADCControllerHardware, ADCControllerDebug

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Demo of PollController functionality.")
    parser.add_argument("--debug", action="store_true", help="Use dummy controllers with simulated data for testing.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging for debugging.")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    l = None
    with LeashControllerHardware() if not args.debug else LeashControllerDebug() as leash:
        l = leash
    a = None
    with ADCControllerHardware() if not args.debug else ADCControllerDebug() as adc:
        a = adc
    lcr = None
    with LCRControllerHardware() if not args.debug else LCRControllerDebug() as lcr_instance:
        lcr = lcr_instance

    l.connect("10.35.14.234")
    lcr.connect("4")
    
    if l is None or a is None or lcr is None:
        logging.error("Failed to initialize controllers. Exiting.")

    poll_controller = PollController(leash_controller=l, lcr_controller=lcr, adc_controller=a)
    exposure_params = [
        ExposureParameter(image=ExposureImage.V2, duration=5, intensity=0),
        ExposureParameter(image=ExposureImage.V2, duration=5, intensity=1),
        ExposureParameter(image=ExposureImage.V2, duration=5, intensity=0)
    ]
    poll_params = PollParameters(exposure_parameters=exposure_params, frequency_hz=1000)
    poll_controller.start_poll(poll_params)
    