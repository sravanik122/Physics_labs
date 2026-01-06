from typing import List, Dict, Optional
from pydantic import BaseModel

class ExperimentInputs(BaseModel):
    applied_force: float
    contact_area: float
    object_mass: float
    surface_type: str
    gravity: float
    deformation: float
    material_hardness: str
    pressure_distribution: str
    duration: float
    external_resistance: str
    contact_shape: str
    surface_roughness: float
    force_application_angle: float

    sampling_interval: float
    seed: Optional[int] = None

class FinalState(BaseModel):
    pressure: float
    normal_force: float
    stress: float
    deformation_depth: float

class Timeline(BaseModel):
    t: List[float]
    applied_force: List[float]
    normal_force: List[float]
    pressure: List[float]
    deformation_depth: List[float]

class SimulationResult(BaseModel):
    final_state: FinalState
    timeline: Timeline
    visual: Dict

class TrainingRow(BaseModel):
    inputs: Dict[str, float]
    outputs: Dict[str, float]

class ModelPrediction(BaseModel):
    predicted_params: Dict[str, float]
