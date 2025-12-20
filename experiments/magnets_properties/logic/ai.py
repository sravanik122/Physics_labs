import json
import math
from pathlib import Path
from typing import List, Dict

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
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
    Build and compile a Keras model using config parameters.
    """

    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    for units in MODEL_CONFIG["model_layers"]:
        model.add(
            layers.Dense(
                units,
                activation="relu",
                kernel_regularizer=regularizers.l2(
                    MODEL_CONFIG["regularization"]["l2"]
                ),
            )
        )
        model.add(layers.Dropout(MODEL_CONFIG["regularization"]["dropout"]))

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
    Train the ML model and return training history.
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

    model_path = Path(log_dir) / "model.keras"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    return {
        "loss": history.history["loss"],
        "val_loss": history.history.get("val_loss", []),
        "epochs": MODEL_CONFIG["epochs"],
    }

def load_model(model_path: Path) -> tf.keras.Model:
    """
    Load a trained Keras model from disk.
    """
    return tf.keras.models.load_model(model_path)

def params_to_timeline(
    A: float,
    k: float,
    total_time: float,
    sampling_interval: float,
) -> Timeline:
    """
    Reconstruct magnetic field & force timeline using
    the same equations as sim.py.
    """

    t_values = []
    field_series = []
    force_series = []

    t = 0.0
    while t <= total_time:
        field = A * math.exp(-k * t)
        force = field * k

        t_values.append(t)
        field_series.append(field)
        force_series.append(force)

        t += sampling_interval

    return Timeline(
        t=t_values,
        attraction_force=force_series,
        field_intensity=field_series,
    )

def predict(
    model: tf.keras.Model,
    inputs: ExperimentInputs,
    input_keys: List[str],
    total_time: float,
    sampling_interval: float,
) -> SimulationResult:
    """
    Predict compact parameters (A, k) and reconstruct
    full simulation output.
    """

    X = np.asarray(
        [[float(getattr(inputs, k)) for k in input_keys]],
        dtype=np.float32,
    )

    # Predict compact parameters
    A_pred, k_pred = model.predict(X, verbose=0)[0]

    A_pred = clamp(A_pred, 0.0, float("inf"))
    k_pred = clamp(k_pred, 0.0, float("inf"))

    timeline = params_to_timeline(
        A=A_pred,
        k=k_pred,
        total_time=total_time,
        sampling_interval=sampling_interval,
    )

    final_force = timeline.attraction_force[-1]
    final_field = timeline.field_intensity[-1]

    attraction_type = (
        "attract" if final_force > 0 else
        "repel" if final_force < 0 else
        "none"
    )

    final_state = FinalState(
        attraction_force=final_force,
        field_intensity=final_field,
        attraction_type=attraction_type,
        effective_range=1 / (k_pred + 1e-6),
        temperature_loss_factor=1.0,
    )

    visual = VisualData(
        field_lines=[],
        force_vectors=[],
        pole_positions=[],
        animation={"predicted": True},
    )

    return SimulationResult(
        final_state=final_state,
        timeline=timeline,
        visual=visual,
    )
