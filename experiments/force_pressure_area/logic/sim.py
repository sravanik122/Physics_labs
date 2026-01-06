import json
import math
from pathlib import Path
from typing import List

import numpy as np

from .types import (
    ExperimentInputs,
    FinalState,
    Timeline,
    SimulationResult,
    TrainingRow,
)
from .helpers import (
    set_random_seed,
    safe_divide,
    normal_force_component,
    surface_contact_factor,
    shape_concentration_factor,
    add_gaussian_noise,
)

CONFIG_PATH = Path(__file__).parents[1] / "config" / "sim_config.json"

with open(CONFIG_PATH, "r") as f:
    SIM_CONFIG = json.load(f)

def simulate(inputs: ExperimentInputs) -> SimulationResult:
    """
    Run deterministic simulation for force, pressure, and area relationship.
    """

    set_random_seed(inputs.seed)

    # Time setup 
    total_time = inputs.duration
    dt = inputs.sampling_interval
    t_values = np.arange(0, total_time + dt, dt)

    # Constants 
    noise_sigma = SIM_CONFIG["noise_sigma"]
    constants = SIM_CONFIG["constants"]

    #  Normal force (angle-aware) 
    normal_force = normal_force_component(
        inputs.applied_force,
        inputs.force_application_angle
    )

    # Shape & surface factors 
    shape_factor = shape_concentration_factor(
        inputs.contact_shape,
        constants["contact_shape_factor"]
    )

    surface_factor = surface_contact_factor(inputs.surface_roughness)

    hardness_factor = constants["material_hardness_factor"].get(
        inputs.material_hardness, 1.0
    )

    distribution_factor = constants["pressure_distribution_factor"].get(
        inputs.pressure_distribution, 1.0
    )

    resistance_factor = constants["external_resistance_factor"].get(
        inputs.external_resistance, 1.0
    )

    # Effective pressure amplitude (A) 
    A_pressure = safe_divide(
        normal_force * shape_factor * distribution_factor,
        inputs.contact_area
    )

    # Rate constant (k) 
    k = safe_divide(
        hardness_factor * surface_factor,
        resistance_factor
    )

    # Time series generation 
    pressure_series = []
    deformation_series = []
    normal_force_series = []

    for t in t_values:
        pressure_t = A_pressure * (1 - math.exp(-k * t))
        deformation_t = safe_divide(pressure_t, hardness_factor * 1e6)

        pressure_t = add_gaussian_noise(pressure_t, noise_sigma)
        deformation_t = add_gaussian_noise(deformation_t, noise_sigma)

        pressure_series.append(pressure_t)
        deformation_series.append(deformation_t)
        normal_force_series.append(normal_force)

    # Final values 
    final_pressure = pressure_series[-1]
    final_deformation = deformation_series[-1]

    stress = safe_divide(final_pressure, surface_factor)

    final_state = FinalState(
        pressure=final_pressure,
        normal_force=normal_force,
        stress=stress,
        deformation_depth=final_deformation
    )

    timeline = Timeline(
        t=t_values.tolist(),
        applied_force=[inputs.applied_force] * len(t_values),
        normal_force=normal_force_series,
        pressure=pressure_series,
        deformation_depth=deformation_series
    )

    visual = SIM_CONFIG["visual"]

    return SimulationResult(
        final_state=final_state,
        timeline=timeline,
        visual=visual
    )

def simulate_at_time(inputs: ExperimentInputs, t: float) -> FinalState:
    """
    Compute final state at a specific time t.
    """

    inputs = inputs.copy()
    inputs.duration = t

    result = simulate(inputs)
    return result.final_state

def generate_teacher_dataset(
    inputs_list: List[ExperimentInputs]
) -> List[TrainingRow]:
    """
    Generate training dataset for ML model.
    """

    dataset = []

    for inp in inputs_list:
        result = simulate(inp)

        dataset.append(
            TrainingRow(
                inputs={
                    "applied_force": inp.applied_force,
                    "contact_area": inp.contact_area,
                    "surface_roughness": inp.surface_roughness,
                    "force_application_angle": inp.force_application_angle,
                    "material_hardness": float(
                        SIM_CONFIG["constants"]["material_hardness_factor"].get(
                            inp.material_hardness, 1.0
                        )
                    )
                },
                outputs={
                    "pressure": result.final_state.pressure,
                    "deformation_depth": result.final_state.deformation_depth
                }
            )
        )

    return dataset
