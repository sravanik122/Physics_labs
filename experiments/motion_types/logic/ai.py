import json
import math
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
from .helpers import clamp

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "model_config.json"

with open(CONFIG_PATH, "r") as f:
    MODEL_CONFIG = json.load(f)

def build_model(input_dim: int, output_dim: int) -> tf.keras.Model:
    """
    Build and compile Keras model.
    """

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
    """
    Train ML model and return training history.
    """

    X, y = [], []

    for row in training_data:
        X.append([float(row.inputs[k]) for k in input_keys])
        y.append([float(row.outputs[k]) for k in output_keys])

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    model = build_model(
        input_dim=X.shape[1],
        output_dim=y.shape[1],
    )

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
    """
    Load trained model from disk.
    """
    return tf.keras.models.load_model(model_path)

def params_to_timeline(
    motion_type: str,
    params: Dict[str, float],
    total_time: float,
    sampling_interval: float,
) -> Timeline:
    """
    Reconstruct motion timeline using same equations as sim.py.
    """
    t_values = []
    x_series, y_series = [], []
    speed_series = []

    t = 0.0
    while t <= total_time:

        if motion_type == "linear":
            v = params["speed"]
            x = v * t
            y = 0.0
            vx, vy = v, 0.0

        elif motion_type == "circular":
            r = params["radius_of_path"]
            w = params["angular_velocity"]
            x = r * math.cos(w * t)
            y = r * math.sin(w * t)
            vx = -r * w * math.sin(w * t)
            vy = r * w * math.cos(w * t)

        elif motion_type == "periodic":
            A = params["amplitude"]
            f = params["frequency"]
            w = 2 * math.pi * f
            x = A * math.sin(w * t)
            y = 0.0
            vx = A * w * math.cos(w * t)
            vy = 0.0

        else:
            raise ValueError("Invalid motion type")

        speed = math.sqrt(vx**2 + vy**2)

        t_values.append(t)
        x_series.append(x)
        y_series.append(y)
        speed_series.append(speed)

        t += sampling_interval

    return Timeline(
        t=t_values,
        position=[{"x": x, "y": y} for x, y in zip(x_series, y_series)],
        velocity=[{"x": 0.0, "y": 0.0} for _ in t_values],
        acceleration=[{"x": 0.0, "y": 0.0} for _ in t_values],
        x=x_series,
        y=y_series,
        speed=speed_series,
    )

def predict(
    model: tf.keras.Model,
    inputs: ExperimentInputs,
    input_keys: List[str],
    total_time: float,
    sampling_interval: float,
) -> SimulationResult:
    """
    Predict compact parameters and reconstruct full motion output.
    """

    X = np.asarray(
        [[float(getattr(inputs, k)) for k in input_keys]],
        dtype=np.float32,
    )

    predictions = model.predict(X, verbose=0)[0]

    params = {}
    for key, value in zip(input_keys, predictions):
        params[key] = clamp(value, 0.0, float("inf"))

    timeline = params_to_timeline(
        motion_type=inputs.motion_type,
        params=params,
        total_time=total_time,
        sampling_interval=sampling_interval,
    )

    final_state = FinalState(
        final_position={"x": timeline.x[-1], "y": timeline.y[-1]},
        final_velocity={"x": 0.0, "y": 0.0},
        total_distance=sum(timeline.speed) * sampling_interval,
        average_speed=sum(timeline.speed) / len(timeline.speed),
        motion_type=inputs.motion_type,
    )

    visual = VisualData(
        trajectory={"x": timeline.x, "y": timeline.y},
        velocity_vectors=[],
        path_type=inputs.motion_type,
        animation={"predicted": True},
    )

    return SimulationResult(
        final_state=final_state,
        timeline=timeline,
        visual=visual,
    )
