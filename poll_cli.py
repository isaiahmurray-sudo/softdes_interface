"""Command-line interface for configuring and running sensor polling trials.

This module collects trial parameters from a user, initializes either debug or
hardware-backed controllers, runs a poll through ``PollController``, and prints
summary statistics for quick validation.
"""

import json
from pathlib import Path
import time
import logging
import argparse
from controllers.LeashController import LeashControllerHardware, LeashControllerDebug, ExposureParameter, ExposureImage
from controllers.LCRController import LCRControllerHardware, LCRControllerDebug, MeasurementMode, MeasurementSpeed
from controllers.ADCController import ADCControllerHardware, ADCControllerDebug
from datastructs.PollData import PollParameters, TrialInfo
from PollController import PollController

def get_connection_parameters():
    """Prompt for runtime connection settings.

    Returns:
        dict: Keys include ``debug``, ``leash_ip``, and ``lcr_port``.
    """
    print("\n--- Connection Setup ---")
    
    debug = input("Use debug mode (simulated hardware)? (y/n) [default: y]: ").strip().lower()
    debug = debug != 'n'  # default to debug
    
    if not debug:
        leash_ip = input("Leash controller IP [default: 10.35.14.234]: ").strip()
        leash_ip = leash_ip if leash_ip else "10.35.14.234"
        
        lcr_port = input("LCR controller port [default: 4]: ").strip()
        lcr_port = lcr_port if lcr_port else "4"
    else:
        leash_ip = None
        lcr_port = None

    return {
        "debug": debug,
        "leash_ip": leash_ip,
        "lcr_port": lcr_port
    }

def get_poll_parameters():
    """Prompt for trial and poll configuration values.

    Returns:
        PollParameters: Parameters used by ``PollController.start_poll``.
    """
    print("\n--- Poll Configuration ---")
    
    # Trial Info
    trial_name = input("Enter trial name [default: Default Trial]: ").strip()
    trial_name = trial_name if trial_name else "Default Trial"
    
    resin_type = input("Enter resin type [default: Unknown]: ").strip()
    resin_type = resin_type if resin_type else "Unknown"
    
    pre_notes = input("Pre-trial notes (optional): ").strip()
    thickness = input("Resin thickness (mm, optional): ").strip()
    thickness = float(thickness) if thickness else None
    
    # Frequency
    frequency = input("Enter frequency (Hz) [default: 1000]: ").strip()
    frequency = int(frequency) if frequency else 1000
    
    # Exposure Parameters
    exposure_params = []
    print("\n--- Exposure Schedule ---")
    while True:
        duration = input("Enter exposure duration (seconds) [default: 5]: ").strip()
        duration = float(duration) if duration else 5.0
        
        intensity = input("Enter exposure intensity (0-1) [default: 0]: ").strip()
        intensity = float(intensity) if intensity else 0.0
        
        image_str = input("Enter exposure image (V1/V2) [default: V2]: ").strip().upper()
        image = ExposureImage.V2 if image_str == "V2" else ExposureImage.V1
        
        exposure_params.append(ExposureParameter(duration=duration, intensity=intensity, image=image))
        
        more = input("Add another exposure segment? (y/n) [default: n]: ").strip().lower()
        if more != 'y':
            break
    
    if not exposure_params:
        print("At least one exposure parameter required. Using default.")
        exposure_params = [ExposureParameter(duration=5, intensity=0, image=ExposureImage.V2)]
    
    # Other parameters with defaults
    push_force = input("Push force [default: 70]: ").strip()
    push_force = float(push_force) if push_force else 70.0
    
    push_repeats = input("Push repeats [default: 2]: ").strip()
    push_repeats = int(push_repeats) if push_repeats else 2
    
    push_delay = input("Push delay (s) [default: 3]: ").strip()
    push_delay = float(push_delay) if push_delay else 3.0
    
    trial_info = TrialInfo(
        trial_name=trial_name,
        resin_type=resin_type,
        pre_trial_notes=pre_notes if pre_notes else None,
        thickness=thickness
    )
    
    return PollParameters(
        exposure_parameters=exposure_params,
        frequency_hz=frequency,
        trial_info=trial_info,
        push_force=push_force,
        push_repeats=push_repeats,
        push_delay_s=push_delay
    )


def run_poll(params, conn_params):
    """Execute one poll run and return collected data.

    Args:
        params: Poll configuration object.
        conn_params: Connection settings from ``get_connection_parameters``.

    Returns:
        PollData | None: Populated poll data if successful, otherwise ``None``.
    """
    print(f"\nInitializing controllers (debug={conn_params['debug']})...")
    
    try:
        leash = LeashControllerHardware() if not conn_params['debug'] else LeashControllerDebug()
        adc = ADCControllerHardware() if not conn_params['debug'] else ADCControllerDebug()
        lcr = LCRControllerHardware() if not conn_params['debug'] else LCRControllerDebug()
        
        with leash, adc, lcr:
            if not conn_params['debug']:
                print("Connecting to hardware...")
                leash.connect(conn_params['leash_ip'])
                lcr.connect(conn_params['lcr_port'])
                print("Connected.")
            
            poll_controller = PollController(leash_controller=leash, lcr_controller=lcr, adc_controller=adc)
            
            print("Starting poll...")
            poll_controller.start_poll(params, wait_for_completion=True)
            print("Poll completed.")
            
            return poll_controller.poll_data
            
    except Exception as e:
        print(f"Error during poll: {e}")
        logging.error(f"Poll error: {e}", exc_info=True)
        return None


def print_summary(poll_data):
    """Print key counters and measurement ranges for a completed poll."""
    if poll_data is None:
        print("No poll data available.")
        return
    
    print("\n--- Poll Summary ---")
    print(f"Trial: {poll_data.poll_parameters.trial_info.trial_name}")
    print(f"Resin: {poll_data.poll_parameters.trial_info.resin_type}")
    print(f"Frequency: {poll_data.poll_parameters.frequency_hz} Hz")
    print(f"LCR Measurements: {len(poll_data.lcr_measurements)}")
    print(f"ADC Measurements: {len(poll_data.adc_measurements)}")
    print(f"POI Measurements: {len(poll_data.poi_measurements)}")
    
    if len(poll_data.lcr_measurements) > 0:
        z_values = poll_data.lcr_measurements['Z']
        print(f"LCR Z - Min: {z_values.min():.2e}, Max: {z_values.max():.2e}, Avg: {z_values.mean():.2e}")
    
    if len(poll_data.adc_measurements) > 0:
        v_values = poll_data.adc_measurements['V']
        print(f"ADC V - Min: {v_values.min():.2f}, Max: {v_values.max():.2f}, Avg: {v_values.mean():.2f}")


def main():
    """Run the interactive CLI loop until the user exits."""
    logging.basicConfig(level=logging.INFO)
    
    print("=== Sensor Poll CLI ===")
    
    conn_params = get_connection_parameters()
    
    while True:
        params = get_poll_parameters()
        poll_data = run_poll(params, conn_params)
        poll_data.save_to_file()
        print_summary(poll_data)
        
        again = input("\nRun another poll? (y/n): ").strip().lower()
        if again != 'y':
            print("Exiting.")
            break


if __name__ == "__main__":
    main()