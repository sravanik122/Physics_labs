from pydantic import BaseModel
from typing import List, Dict, Optional, Union

class ExperimentInputs(BaseModel):
    motion_type: str  

    displacement: float
    speed: float

    radius_of_path: float
    angular_velocity: float

    amplitude: float
    frequency: float

    medium_resistance: float
    initial_position: float

    observation_duration: float
    sampling_interval: float

    seed: Optional[int] = None

class FinalState(BaseModel):
    final_position: Union[float, Dict[str, float]]
    final_velocity: Union[float, Dict[str, float]]

    total_distance: float
    average_speed: float

    motion_type: str

class Timeline(BaseModel):
    t: List[float]

    position: List[Union[float, Dict[str, float]]]
    velocity: List[Union[float, Dict[str, float]]]
    acceleration: List[Union[float, Dict[str, float]]]

    x: List[float]
    y: List[float]

    speed: List[float]

class VelocityVector(BaseModel):
    x: float
    y: float
    magnitude: float
    direction: float

class VisualData(BaseModel):
    trajectory: Dict[str, List[float]]
    velocity_vectors: List[VelocityVector]

    path_type: str  

    animation: Dict[str, Union[bool, int, float]]
  
class SimulationResult(BaseModel):
    final_state: FinalState
    timeline: Timeline
    visual: VisualData
  
class TrainingRow(BaseModel):
    inputs: Dict[str, Union[float, int, str]]
    outputs: Dict[str, Union[float, int]]

class ModelPrediction(BaseModel):
    parameters: Dict[str, float]
    confidence: Optional[float] = None
