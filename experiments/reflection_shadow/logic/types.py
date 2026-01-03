from pydantic import BaseModel
from typing import List, Dict, Optional, Union

class ExperimentInputs(BaseModel):
    angle_of_incidence: float
    object_distance: float
    light_intensity: float

    mirror_type: str
    surface_smoothness: float

    object_size: float
    screen_distance: float

    ambient_light: float
    wavelength: float

    observation_duration: float
    sampling_interval: float

    seed: Optional[int] = None

class FinalState(BaseModel):
    angle_of_reflection: float
    image_distance: float
    shadow_size: float
    shadow_intensity: float
    reflected_intensity: float

class Timeline(BaseModel):
    t: List[float]

    angle_of_incidence: List[float]
    angle_of_reflection: List[float]

    reflected_intensity: List[float]
    shadow_size: List[float]
    shadow_intensity: List[float]

class VisualData(BaseModel):
    rays: Dict[str, Union[int, float]]
    shadow: Dict[str, float]
    mirror: Dict[str, str]
    light: Dict[str, Union[str, float]]
    animation: Dict[str, Union[bool, int]]

class SimulationResult(BaseModel):
    final_state: FinalState
    timeline: Timeline
    visual: VisualData

class TrainingRow(BaseModel):
    inputs: Dict[str, Union[float, int, str]]
    outputs: Dict[str, float]

class ModelPrediction(BaseModel):
    parameters: Dict[str, float]
    confidence: Optional[float] = None
