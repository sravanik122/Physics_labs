import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import TensorBoard

from .types import (
    ExperimentInputs,
    FinalState,
    Timeline,
    VisualData,
    SimulationResult,
    TrainingRow,
)
from .helpers import clamp, safe_divide, rms_value

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "model_config.json"

with open(CONFIG_PATH, "r") as f:
    MODEL_CONFIG = json.load(f)

def build_model(input_dim: int, output_dim: int) -> tf.keras.Model:
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    for units in MODEL_CONFIG["model_layers"]:
        model.add(layers.Dense(units, activation="relu"))

    model.add(layers.Dense(output_dim, activation="linear"))

    optimizer = optimizers.Adam(
        learning_rate=MODEL_CONFIG["optimizer"]["learning_rate"]
    )

    model.compile(
        optimizer=optimizer,
        loss=MODEL_CONFIG["loss"],
        metrics=MODEL_CONFIG["metrics"],
    )

    return model

def train_model(
    training_data: List[TrainingRow],
    input_keys: List[str],
    output_keys: List[str],
) -> Dict:
    X, y = [], []

    for row in training_data:
        X.append([float(row.inputs[k]) for k in input_keys])
        y.append([float(row.outputs[k]) for k in output_keys])

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    model = build_model(X.shape[1], y.shape[1])

    log_dir = MODEL_CONFIG["tensorboard_logdir"]
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    tensorboard_cb = TensorBoard(log_dir=log_dir)

    history = model.fit(
        X,
        y,
        epochs=MODEL_CONFIG["epochs"],
        batch_size=MODEL_CONFIG["batch_size"],
        validation_split=MODEL_CONFIG["validation_split"],
        callbacks=[tensorboard_cb],
        verbose=0,
    )

    model.save(Path(log_dir) / "model.keras")

    return {
        "loss": history.history["loss"],
        "val_loss": history.history.get("val_loss", []),
        "epochs": MODEL_CONFIG["epochs"],
    }

def load_model(model_path: Path) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path)

def params_to_timeline(
    params: Dict[str, float],
    voltage: float,
    switch_state: bool,
    power_source_type: str,
    total_time: float,
    sampling_interval: float,
) -> Timeline:
    t_values = []
    current_series = []
    voltage_series = []
    resistance_series = []
    power_series = []
    temperature_series = []

    t = 0.0
    while t <= total_time:

        if switch_state:
            current = safe_divide(voltage, params["effective_resistance"])
        else:
            current = 0.0

        if power_source_type == "ac":
            current = rms_value(current)

        power = voltage * current

        t_values.append(t)
        current_series.append(current)
        voltage_series.append(voltage)
        resistance_series.append(params["effective_resistance"])
        power_series.append(power)
        temperature_series.append(0.0)

        t += sampling_interval

    return Timeline(
        t=t_values,
        current=current_series,
        voltage=voltage_series,
        resistance=resistance_series,
        power=power_series,
        temperature=temperature_series,
    )

def predict(
    model: tf.keras.Model,
    inputs: ExperimentInputs,
    input_keys: List[str],
    total_time: float,
    sampling_interval: float,
) -> SimulationResult:
    X = np.asarray(
        [[float(getattr(inputs, k)) for k in input_keys]],
        dtype=np.float32,
    )

    predictions = model.predict(X, verbose=0)[0]

    params = {
        "effective_resistance": clamp(predictions[0], 1e-6, 1e9),
        "current": clamp(predictions[1], 0.0, 1e6),
        "power": clamp(predictions[2], 0.0, 1e9),
    }

    timeline = params_to_timeline(
        params=params,
        voltage=inputs.voltage,
        switch_state=inputs.switch_state,
        power_source_type=inputs.power_source_type,
        total_time=total_time,
        sampling_interval=sampling_interval,
    )

    final_state = FinalState(
        current=timeline.current[-1],
        voltage=inputs.voltage,
        total_resistance=params["effective_resistance"],
        power=timeline.power[-1],
        switch_state=inputs.switch_state,
        power_source_type=inputs.power_source_type,
    )

    visual = VisualData(
        current_flow={
            "enabled": inputs.switch_state,
            "flow_intensity": abs(timeline.current[-1]),
        },
        bulb={
            "glow_intensity": abs(timeline.power[-1]),
        },
        wire_heating={
            "temperature_level": 0.0,
            "color_map": "thermal",
        },
        circuit_state={
            "switch": "on" if inputs.switch_state else "off"
        },
        animation={
            "predicted": True
        },
    )

    return SimulationResult(
        final_state=final_state,
        timeline=timeline,
        visual=visual,
    )
