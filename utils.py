"""Utils for poll collectiona nd processing"""

import numpy as np


def compute_ionic_viscosity(
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
