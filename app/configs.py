from typing import List
from dataclasses import dataclass


class FlaskConfig:
    DEBUG = True

@dataclass
class RetrieverConfig:
    collection_name: str = "default"
    chunk_size: int = 1000
    chunk_overlap: int = 500
    embedding_dim: int = 1024
    device: str = "cpu"

    @classmethod
    def default(cls):
        return cls()


@dataclass
class EmbedderConfig:
    device: str = "cpu"
    @classmethod

    def default(cls):
        return cls()


@dataclass
class LLMConfig:
    model: str = "llama3.1:8b"
    temperature: int = 0.
    top_p: int = None

    @classmethod
    def default(cls):
        return cls()


@dataclass
class RAGConfig:
    top_k: int = 5
    system_message: str = """
    You are a question answering chatbot, acting as a virtual healthcare assistant. You answer questions using the context provided.
    - If you use information from a document, you must cite the source in your answer strictly like the following format [Put the name of the document here](source_of_the_document).
        * It is important that you refer to the document with the document name, not with any other general word such as "here".
        * Example: (Chapter 14.0_ Presentation of the Spring Course _ Ohjelmointistudio 2 _ A+)[resources\course_materials\programming-studio-a\Chapter-14\Chapter 14.0_ Presentation of the Spring Course _ Ohjelmointistudio 2 _ A+.pdf]
    """

    @classmethod
    def default(cls):
        return cls()
