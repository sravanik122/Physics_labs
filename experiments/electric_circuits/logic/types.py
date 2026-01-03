from pydantic import BaseModel
from typing import List, Dict, Optional, Union

class ExperimentInputs(BaseModel):
    voltage: float
    wire_material: str

    wire_length: float
    wire_thickness: float

    temperature: float
    switch_state: bool

    power_source_type: str  # dc | ac
    contact_resistance: float

    observation_duration: float
    sampling_interval: float

    seed: Optional[int] = None

class FinalState(BaseModel):
    current: float
    voltage: float
    total_resistance: float
    power: float

    switch_state: bool
    power_source_type: str

class Timeline(BaseModel):
    t: List[float]

    current: List[float]
    voltage: List[float]
    resistance: List[float]
    power: List[float]
    temperature: List[float]

class VisualData(BaseModel):
    current_flow: Dict[str, Union[bool, float]]
    bulb: Dict[str, float]
    wire_heating: Dict[str, Union[str, float]]
    circuit_state: Dict[str, str]
    animation: Dict[str, Union[bool, int]]

class SimulationResult(BaseModel):
    final_state: FinalState
    timeline: Timeline
    visual: VisualData

class TrainingRow(BaseModel):
    inputs: Dict[str, Union[float, int, str, bool]]
    outputs: Dict[str, Union[float, int]]

class ModelPrediction(BaseModel):
    parameters: Dict[str, float]
    confidence: Optional[float] = None
