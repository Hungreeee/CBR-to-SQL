from pydantic import BaseModel, Field
from typing import List, Optional


class EntityExtraction(BaseModel):
    """Schema for entity value extraction"""
    entities: List[str] = Field(
        description="List of specific medical values that need to be matched against the database (NOT generic terms like 'patient', 'disease', etc.)"
    )


class TagAssignment(BaseModel):
    """Schema for tag assignment with schema linking"""
    label: str = Field(
        description="Semantic tag (e.g., CONDITION, PROCEDURE, DRUG, EQUIPMENT, TIME, etc.)"
    )
    best_match_index: int = Field(
        description="Index (0-4) of the best matching database entity from the provided list"
    )


class EntityValidation(BaseModel):
    """Schema for validating lexical/string matching quality"""
    is_acceptable: bool = Field(
        description="Whether the matched entity has good lexical/string similarity to the noun phrase (NOT whether it's medically correct)"
    )
    feedback: Optional[str] = Field(
        default=None,
        description="If not acceptable, provide specific feedback on lexical mismatch (e.g., 'No word overlap between X and Y. Look for matches containing [key words]')"
    )


class SemanticRichnessScore(BaseModel):
    """Schema for semantic richness evaluation"""
    score: float = Field(
        description="Semantic richness score from 0.0 (not meaningful) to 1.0 (highly meaningful)",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        description="Brief explanation of the score"
    )
    examples_analysis: str = Field(
        description="What do these values represent? Are they descriptive?"
    )
