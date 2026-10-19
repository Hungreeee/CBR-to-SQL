from typing import Literal
from dataclasses import dataclass


@dataclass
class RetrieverConfig:
    collection_name: str = "default"
    chunk_size: int = 1000
    chunk_overlap: int = 500
    embedding_dim: int = 384
    device: str = "cpu"
    dense_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    sparse_embedding_model: str = "Qdrant/bm25"

    @classmethod
    def default(cls):
        return cls()

@dataclass
class LLMConfig:
    # model: str = "gpt-4.1-mini"
    model: str = "gpt-4o" # Azure maps this to the gpt-4o-2024-08-06 deployment; use the dated name only for direct OpenAI
    # model: str = "Qwen/Qwen3-VL-8B-Instruct-FP8:de-c91946d076db1154"
    temperature: int = 0.
    top_p: int = None

    @classmethod
    def default(cls):
        return cls()


@dataclass
class RAGConfig:
    top_k: int = 5
    rerank_top_k: int = 50
    brittle_retrieval: bool = False
    hybrid_retrieval: bool = False
    prompt_extension: bool = False
    doc_reranking: bool = False
    template_construction: bool = True
    source_discovery: bool = True
    refined_tagging: bool = True # False: mask with the generic semantic tags; the lookup table still resolves best_match/table/column
    reranker_embedding_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2" # Qwen/Qwen3-Reranker-0.6B
    dataset: Literal["mimicsql", "ehrsql", "ehrsql24"] = "mimicsql"

    @classmethod
    def default(cls):
        return cls()
