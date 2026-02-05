from pydantic import BaseModel, Field
from typing import List, Optional


class TaggedEntity(BaseModel):
    """Schema for a single tagged entity"""
    value: str = Field(
        description="Entity value span extracted from the question in its original form (including typos, special characters, parentheses, slashes, etc.)"
    )
    label: str = Field(
        description="Temporary semantic tag based on sentence context (e.g., CONDITION, PROCEDURE, DRUG, SUBJECT_ID, AGE, YEAR, ICD_CODE, etc.)"
    )
    is_semantic: bool = Field(
        description="True if text-based (needs fuzzy lookup): words, phrases, names, medical terms, locations, attributes. False if literal value (exact match): IDs, numbers, codes, years, dates, counts."
    )


class EntityExtraction(BaseModel):
    """Schema for entity value extraction with temporary tagging"""
    entities: List[TaggedEntity] = Field(
        description="List of all specific values extracted from the question with temporary tags and semantic flags"
    )


class TagAssignment(BaseModel):
    """Schema for refined tag assignment with schema linking"""
    label: str = Field(
        description="Refined semantic tag based on database schema (e.g., CONDITION, PROCEDURE, DRUG, etc.). Use 'NO_MATCH' if rejecting all candidates."
    )
    best_match_index: int = Field(
        description="Index (0-4) of the best matching database entity, or -1 if no suitable match found and all candidates should be rejected"
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


class SqlGeneration(BaseModel):
    sql_query: str = Field(description="Return a SQL query for the question.")


class SqlGenerationOptional(BaseModel):
    """Think hard about why the question is answerable, and what evidence do you have. Decide to 
    write a SQL query if possible, and write "None" if impossible to answer based on given context.
    """
    sql_query: str | None = Field(description="Return a SQL query if question is answerable, and 'None' if the question is impossible.")
    reasoning: str = Field(description="Very brief explanation: Why is the SQL query written that way. Explain all evidences for every function /condition, etc you wrote.")


class PromptDecomposition(BaseModel):
    decomposed_prompts: List[str] = Field(description="Return the list of decomposed queries here for more nuanced retrieval.")