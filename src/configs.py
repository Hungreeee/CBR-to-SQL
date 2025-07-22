from typing import List
from dataclasses import dataclass


class FlaskConfig:
    DEBUG = True

@dataclass
class RetrieverConfig:
    collection_name: str = "default"
    chunk_size: int = 1000
    chunk_overlap: int = 500
    embedding_dim: int = 384
    device: str = "cpu"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    @classmethod
    def default(cls):
        return cls()

@dataclass
class LLMConfig:
    # model: str = "gpt-4.1-mini"
    model: str = "gpt-4o"
    temperature: int = 0.
    top_p: int = None

    @classmethod
    def default(cls):
        return cls()


@dataclass
class RAGConfig:
    top_k: int = 5
    brittle_retrieval: bool = True
    return_response: bool = False
    template_construction: bool = True
    source_discovery: bool = True

    @classmethod
    def default(cls):
        return cls()
