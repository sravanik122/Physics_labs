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
from .helpers import (
    clamp,
    temperature_gradient,
    convection_mode,
)

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
    inputs: ExperimentInputs,
    total_time: float,
    sampling_interval: float,
) -> Timeline:
    t_values = []
    temperature_series = []
    heat_rate_series = []
    cond_series = []
    conv_series = []
    rad_series = []

    t = 0.0
    delta_t = inputs.temperature_difference

    while t <= total_time:

        grad = temperature_gradient(delta_t, inputs.distance_from_source)

        q_cond = params["effective_conductivity"] * grad
        q_conv = params["effective_heat_transfer_coefficient"] * delta_t
        q_rad = params["effective_emissivity"] * ((delta_t + 273.15) ** 4 - 273.15 ** 4)

        q_total = q_cond + q_conv + q_rad

        temperature = delta_t * (1 - (t / total_time))

        t_values.append(t)
        temperature_series.append(temperature)
        heat_rate_series.append(q_total)
        cond_series.append(q_cond)
        conv_series.append(q_conv)
        rad_series.append(q_rad)

        t += sampling_interval

    return Timeline(
        t=t_values,
        temperature=temperature_series,
        heat_transfer_rate=heat_rate_series,
        conduction_rate=cond_series,
        convection_rate=conv_series,
        radiation_rate=rad_series,
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
        "effective_conductivity": clamp(predictions[0], 0.0, 1e6),
        "effective_heat_transfer_coefficient": clamp(predictions[1], 0.0, 1e6),
        "effective_emissivity": clamp(predictions[2], 0.0, 1.0),
    }

    timeline = params_to_timeline(
        params=params,
        inputs=inputs,
        total_time=total_time,
        sampling_interval=sampling_interval,
    )

    final_state = FinalState(
        final_temperature=timeline.temperature[-1],
        total_heat_transfer=sum(timeline.heat_transfer_rate) * sampling_interval,
        conduction_heat=sum(timeline.conduction_rate) * sampling_interval,
        convection_heat=sum(timeline.convection_rate) * sampling_interval,
        radiation_heat=sum(timeline.radiation_rate) * sampling_interval,
    )

    visual = VisualData(
        conduction={
            "heat_gradient_intensity": abs(timeline.conduction_rate[-1]),
            "color_map": "thermal",
        },
        convection={
            "flow_strength": inputs.airflow_speed,
            "arrow_density": 20,
        },
        radiation={
            "wave_intensity": params["effective_emissivity"],
            "wave_count": 15,
        },
        animation={
            "predicted": True,
        },
    )

    return SimulationResult(
        final_state=final_state,
        timeline=timeline,
        visual=visual,
    )
