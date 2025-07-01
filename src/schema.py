from pydantic import BaseModel, Field
from typing import List
from enum import Enum

class Entity(BaseModel):
    value: str = Field(...)
    label: str = Field(...)

class MaskingResults(BaseModel):
    masked_sentence: str = Field(...)
    redacted_entities: List[Entity] = Field(..., description="List of redacted values. Leave blank if nothing match.")

class Verdict(str, Enum):
    CREATE = "CREATE"
    IGNORE = "IGNORE"

class RetainVerdict(BaseModel):
    action: Verdict = Field(...)