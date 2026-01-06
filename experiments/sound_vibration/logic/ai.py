import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

from .types import (
    ExperimentInputs,
    SimulationResult,
    FinalState,
    Timeline,
    TrainingRow
)
from .helpers import (
    sine_wave,
    exponential_decay,
    wavelength_from_frequency,
    sound_intensity_from_amplitude,
    loudness_db,
    distance_attenuation
)

CONFIG_PATH = Path(__file__).parents[1] / "config" / "model_config.json"

with open(CONFIG_PATH, "r") as f:
    MODEL_CONFIG = json.load(f)

def build_model(input_dim: int, output_dim: int) -> tf.keras.Model:
    """
    Build regression model.
    """
    model = models.Sequential()

    model.add(layers.Input(shape=(input_dim,)))

    for units in MODEL_CONFIG["model_layers"]:
        model.add(layers.Dense(units, activation="relu"))

    model.add(layers.Dense(output_dim, activation="linear"))

    optimizer = optimizers.Adam(
        learning_rate=MODEL_CONFIG["learning_rate"]
    )

    model.compile(
        optimizer=optimizer,
        loss=MODEL_CONFIG["loss"],
        metrics=MODEL_CONFIG["metrics"]
    )

    return model

def train_model(
    training_data: List[TrainingRow],
    input_keys: List[str],
    output_keys: List[str]
) -> Dict:
    """
    Train ML model using teacher dataset.
    """

    X = []
    y = []

    for row in training_data:
        X.append([float(row.inputs[k]) for k in input_keys])
        y.append([float(row.outputs[k]) for k in output_keys])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    model = build_model(
        input_dim=X.shape[1],
        output_dim=y.shape[1]
    )

    log_dir = MODEL_CONFIG["tensorboard_logdir"]
    tb_callback = callbacks.TensorBoard(log_dir=log_dir)

    history = model.fit(
        X,
        y,
        epochs=MODEL_CONFIG["epochs"],
        batch_size=MODEL_CONFIG["batch_size"],
        validation_split=MODEL_CONFIG["validation_split"],
        callbacks=[tb_callback],
        verbose=0
    )

    model.save(Path(log_dir) / "model.keras")

    return {
        "loss": history.history["loss"],
        "val_loss": history.history.get("val_loss", [])
    }

def load_model(model_path: Path) -> tf.keras.Model:
    """
    Load trained model.
    """
    return tf.keras.models.load_model(model_path)

def params_to_timeline(
    predicted_params: Dict[str, float],
    inputs: ExperimentInputs
) -> SimulationResult:
    """
    Convert predicted parameters into full sound timeline.
    """

    sampling_interval = inputs.sampling_interval
    total_time = inputs.observation_time

    times = []
    displacement = []
    intensities = []
    loudness_levels = []
    wave_amplitudes = []

    t = 0.0
    while t <= total_time:
        disp = sine_wave(
            predicted_params["amplitude"],
            predicted_params["frequency"],
            t
        )

        amp = exponential_decay(
            abs(disp),
            predicted_params["damping"],
            t
        )

        intensity = sound_intensity_from_amplitude(amp)
        intensity = distance_attenuation(
            intensity,
            inputs.distance_from_source
        )

        loudness = loudness_db(
            intensity,
            predicted_params["reference_intensity"]
        )

        times.append(t)
        displacement.append(disp)
        intensities.append(intensity)
        loudness_levels.append(loudness)
        wave_amplitudes.append(amp)

        t += sampling_interval

    final_state = FinalState(
        sound_intensity=intensities[-1],
        loudness_level=loudness_levels[-1],
        wavelength=predicted_params["wavelength"],
        energy_decay=wave_amplitudes[-1]
    )

    timeline = Timeline(
        t=times,
        displacement=displacement,
        sound_intensity=intensities,
        loudness_level=loudness_levels,
        wave_amplitude=wave_amplitudes
    )

    return SimulationResult(
        final_state=final_state,
        timeline=timeline,
        visual={}
    )

def predict(
    model: tf.keras.Model,
    inputs: ExperimentInputs,
    input_keys: List[str]
) -> SimulationResult:
    """
    Predict sound behavior using trained ML model.
    """

    X = np.array(
        [[getattr(inputs, k) for k in input_keys]],
        dtype=np.float32
    )

    predictions = model.predict(X, verbose=0)[0]

    predicted_params = {
        "sound_intensity": predictions[0],
        "loudness_level": predictions[1],
        "wavelength": predictions[2],
        "amplitude": inputs.amplitude,
        "frequency": inputs.frequency,
        "damping": inputs.damping,
        "reference_intensity": 1e-12
    }

    return params_to_timeline(predicted_params, inputs)
