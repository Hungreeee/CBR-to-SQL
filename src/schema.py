from pydantic import BaseModel
from typing import List
from enum import Enum


class EntityType(str, Enum):
    CONDITION = "CONDITION"
    MEASUREMENT = "MEASUREMENT"
    DRUG = "DRUG"
    PATIENT = "PATIENT"
    CODE = "CODE"

class Entity(BaseModel):
    name: str
    type: EntityType 

class EntityExtractionResult(BaseModel):
    entities: List[Entity]
