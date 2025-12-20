from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal, Union

class ExperimentInputs(BaseModel):
    magnet_strength: float = Field(
        ..., ge=0.1, le=1.0,
        description="Relative strength of the magnet"
    )

    distance_from_object: float = Field(
        ..., ge=0.0, le=50.0,
        description="Distance between magnet and object (mm)"
    )

    material_type: Literal[
        "iron", "steel", "nickel", "cobalt",
        "copper", "aluminum", "plastic", "wood"
    ]

    pole_orientation: Literal["N-S", "N-N", "S-S"]

    magnetic_field_intensity: float = Field(
        ..., ge=0.01, le=2.0,
        description="Initial magnetic field intensity (Tesla)"
    )

    temperature: float = Field(
        ..., ge=0.0, le=100.0,
        description="Ambient temperature (°C)"
    )

    shape_of_magnet: Literal[
        "bar", "horseshoe", "ring", "cylindrical"
    ]

    surface_contact: Literal["touching", "not_touching"]

    number_of_magnets: int = Field(
        ..., ge=1, le=5,
        description="Number of magnets used"
    )

    surrounding_medium: Literal["vacuum", "air", "water"]

    total_time: Optional[float] = None
    sampling_interval: Optional[float] = None
    seed: Optional[int] = None

class FinalState(BaseModel):
    attraction_force: float
    field_intensity: float
    attraction_type: Literal["attract", "repel", "none"]
    effective_range: float
    temperature_loss_factor: float

class Timeline(BaseModel):
    t: List[float]
    attraction_force: List[float]
    field_intensity: List[float]

class FieldLine(BaseModel):
    x: float
    y: float
    strength: float

class ForceVector(BaseModel):
    x: float
    y: float
    magnitude: float
    direction: float  

class PolePosition(BaseModel):
    pole: Literal["N", "S"]
    x: float
    y: float

class VisualData(BaseModel):
    field_lines: List[FieldLine]
    force_vectors: List[ForceVector]
    pole_positions: List[PolePosition]

    animation: Dict[str, Union[float, bool]]

class SimulationResult(BaseModel):
    final_state: FinalState
    timeline: Timeline
    visual: VisualData

class TrainingRow(BaseModel):
    inputs: Dict[str, Union[float, int, str]]
    outputs: Dict[str, float]

class ModelPrediction(BaseModel):
    final_state: FinalState
    visual: VisualData
