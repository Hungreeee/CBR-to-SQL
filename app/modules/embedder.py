import os
from typing import List

from langchain_openai import AzureOpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from configs import EmbedderConfig


class BaseEmbedder:
    def __init__(
        self,
        config: EmbedderConfig = EmbedderConfig.default(),
    ):
        self.embedding_model = None
        self.config = config

    def encode_single_query(self, text: str):
        self.embedding_model.embed_query(text)

    def encode_documents(self, documents: List[str]):
        self.embedding_model.embed_documents(documents)


class HuggingFaceEmbedder(BaseEmbedder):
    def __init__(
        self,
        model: str = os.environ.get("HUGGINGFACE_EMBEDDING_MODEL", "BAAI/bge-m3"),
        config: EmbedderConfig = EmbedderConfig.default(),
    ):
        super().__init__(config)

        self.embedding_model = HuggingFaceEmbeddings(model_name=model)


class AzureEmbedder(BaseEmbedder):
    def __init__(
        self,
        api_key: str = os.environ.get("AZURE_OPENAI_API_KEY"),
        api_version: str = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        base_url: str = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://cv-poc-openai.openai.azure.com"),
        model: str = os.environ.get("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"),
        config: EmbedderConfig = EmbedderConfig.default(),
    ):
        super().__init__(config)

        self.embedding_model = AzureOpenAIEmbeddings(
            model=model,
            api_key=api_key,
            azure_endpoint=base_url,
            openai_api_version=api_version,
        )