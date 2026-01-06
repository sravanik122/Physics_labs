from typing import List, Dict, Optional
from pydantic import BaseModel

class ExperimentInputs(BaseModel):
    frequency: float
    amplitude: float
    medium_type: str
    tension: float
    length_of_vibrating_body: float
    density: float
    distance_from_source: float
    air_pressure: float
    damping: float
    resonance: bool
    source_type: str
    observation_time: float

    sampling_interval: float
    seed: Optional[int] = None

class FinalState(BaseModel):
    sound_intensity: float
    loudness_level: float
    wavelength: float
    energy_decay: float

class Timeline(BaseModel):
    t: List[float]
    displacement: List[float]
    sound_intensity: List[float]
    loudness_level: List[float]
    wave_amplitude: List[float]

class SimulationResult(BaseModel):
    final_state: FinalState
    timeline: Timeline
    visual: Dict

class TrainingRow(BaseModel):
    inputs: Dict[str, float]
    outputs: Dict[str, float]

class ModelPrediction(BaseModel):
    predicted_params: Dict[str, float]
