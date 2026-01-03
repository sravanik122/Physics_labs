import json
import math
from pathlib import Path
from typing import List

from .types import (
    ExperimentInputs,
    FinalState,
    Timeline,
    SimulationResult,
    VisualData,
    TrainingRow,
)
from .helpers import (
    set_random_seed,
    safe_divide,
    temperature_factor,
    cross_section_area_mm2_to_m2,
    rms_value,
    add_gaussian_noise,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "sim_config.json"

with open(CONFIG_PATH, "r") as f:
    SIM_CONFIG = json.load(f)

def simulate(inputs: ExperimentInputs) -> SimulationResult:
    """
    Run deterministic electric circuit simulation.
    """

    set_random_seed(inputs.seed)

    dt = inputs.sampling_interval or SIM_CONFIG["default_sampling_interval"]
    total_time = inputs.observation_duration or SIM_CONFIG["default_total_time"]
    noise_sigma = SIM_CONFIG["noise_sigma"]

    t_values: List[float] = []
    current_series: List[float] = []
    voltage_series: List[float] = []
    resistance_series: List[float] = []
    power_series: List[float] = []
    temperature_series: List[float] = []

    material = inputs.wire_material
    rho_0 = SIM_CONFIG["constants"]["material_resistivity"][material]
    alpha = SIM_CONFIG["constants"]["temperature_coefficient"][material]
    T_ref = SIM_CONFIG["constants"]["reference_temperature"]

    area_m2 = cross_section_area_mm2_to_m2(inputs.wire_thickness)

    temp_factor = temperature_factor(
        inputs.temperature, T_ref, alpha
    )

    wire_resistance = rho_0 * temp_factor * inputs.wire_length / area_m2
    total_resistance = wire_resistance + inputs.contact_resistance

    source = SIM_CONFIG["constants"]["power_source_types"][inputs.power_source_type]
    stability = source["stability_factor"]

    t = 0.0
    while t <= total_time:

        if inputs.switch_state:
            current = safe_divide(inputs.voltage, total_resistance) * stability
        else:
            current = 0.0

        if inputs.power_source_type == "ac":
            current = rms_value(current)

        power = inputs.voltage * current

        # Add noise
        current = add_gaussian_noise(current, noise_sigma)
        power = add_gaussian_noise(power, noise_sigma)

        t_values.append(t)
        current_series.append(current)
        voltage_series.append(inputs.voltage)
        resistance_series.append(total_resistance)
        power_series.append(power)
        temperature_series.append(inputs.temperature)

        t += dt

    final_state = FinalState(
        current=current_series[-1],
        voltage=inputs.voltage,
        total_resistance=total_resistance,
        power=power_series[-1],
        switch_state=inputs.switch_state,
        power_source_type=inputs.power_source_type,
    )

    visual = VisualData(
        current_flow={
            "enabled": inputs.switch_state,
            "flow_intensity": abs(current_series[-1]),
        },
        bulb={
            "glow_intensity": abs(power_series[-1]),
        },
        wire_heating={
            "temperature_level": inputs.temperature,
            "color_map": SIM_CONFIG["visual"]["wire_heat_color_map"],
        },
        circuit_state={
            "switch": "on" if inputs.switch_state else "off"
        },
        animation={
            "show_current_flow": inputs.switch_state,
            "show_bulb_glow": True,
            "show_wire_heating": True,
            "recommended_fps": SIM_CONFIG["visual"]["animation"]["recommended_fps"],
        },
    )

    timeline = Timeline(
        t=t_values,
        current=current_series,
        voltage=voltage_series,
        resistance=resistance_series,
        power=power_series,
        temperature=temperature_series,
    )

    return SimulationResult(
        final_state=final_state,
        timeline=timeline,
        visual=visual,
    )

def simulate_at_time(inputs: ExperimentInputs, t: float) -> FinalState:
    """
    Return circuit state at a specific time.
    """
    result = simulate(inputs)
    index = min(
        int(t / inputs.sampling_interval),
        len(result.timeline.t) - 1,
    )
    return result.final_state

def generate_teacher_dataset(
    input_samples: List[ExperimentInputs],
) -> List[TrainingRow]:
    """
    Generate labeled dataset using deterministic simulation.
    """

    dataset: List[TrainingRow] = []

    for inp in input_samples:
        result = simulate(inp)

        dataset.append(
            TrainingRow(
                inputs=inp.dict(exclude={"seed"}, exclude_none=True),
                outputs={
                    "effective_resistance": result.final_state.total_resistance,
                    "current": result.final_state.current,
                    "power": result.final_state.power,
                },
            )
        )

    return dataset
