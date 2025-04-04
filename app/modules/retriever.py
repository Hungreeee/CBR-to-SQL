import os
import uuid
from typing import List, Dict

from qdrant_client import QdrantClient, models

from langchain.schema.document import Document
from langchain_qdrant import QdrantVectorStore
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer

from configs import RetrieverConfig
from modules.generator import BaseGenerator
from modules.embedder import BaseEmbedder

from dotenv import load_dotenv

load_dotenv()


class BaseRetriever:
    def __init__(
        self, 
        embedder: BaseEmbedder,
        config: RetrieverConfig = RetrieverConfig.default(),
    ):
        self.config = config
        self.embedder = embedder
        self.vectorstore = None
    
    def update_config(self, **kwargs):
        self.config.update(kwargs)
    
    def add(self, documents: List[Dict], chunking: bool = True):
        documents_langchain = []

        for doc in documents:
            parent_id = doc["parent_id"]
            metadata = {key: value for key, value in doc.items() if key != "page_content"}
            metadata["parent_id"] = parent_id

            documents_langchain.append(Document(
                page_content=doc["page_content"], 
                metadata=metadata,
            ))

        if chunking:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size, 
                chunk_overlap=self.config.chunk_overlap,
            )
            documents_langchain = text_splitter.split_documents(documents_langchain)

            for chunk in documents_langchain:
                chunk.metadata["chunk_id"] = str(uuid.uuid4())

        self.vectorstore.add_documents(documents_langchain)

    def update(self, documents: List[Dict]):
        raise NotImplementedError
    
    def delete(self, node_ids: List[str]):
        raise NotImplementedError

    def retrieve(self, query: str, top_k: int):
        raise NotImplementedError


class SimpleVectorRetriever(BaseRetriever):
    def __init__(
        self, 
        config: RetrieverConfig = RetrieverConfig.default(), 
        vectorstore_path: str = None,
    ):
        super().__init__(config)

        self.vectorstore = None

        if vectorstore_path and os.path.exists(vectorstore_path):
            self.vectorstore = FAISS.load_local(
                folder_path=vectorstore_path, 
                embeddings=self.embedding_model,
                allow_dangerous_deserialization=True,
            )

    def ingest(
        self, 
        documents: List[dict], 
        vectorstore_path: str = "./vectorstore/"
    ):
        documents_langchain = []

        for doc in documents:
            metadata = {key: value for key, value in doc.items() if key != "page_content"}
            documents_langchain.append(Document(
                page_content=doc["page_content"], 
                metadata=metadata,
            ))

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size, 
            chunk_overlap=self.config.chunk_overlap,
        )
        document_chunks = text_splitter.split_documents(documents_langchain)

        self.vectorstore = FAISS.from_documents(
            documents=document_chunks, 
            embedding=self.embedding_model
        )
        self.vectorstore.save_local(vectorstore_path)
        
    def retrieve(self, query: str, top_k: int = 5):
        relevant_docs = self.vectorstore.similarity_search(query=query, k=top_k)
        return relevant_docs
    
    def reset(self):
        pass


class QdrantRetriever(BaseRetriever):
    def __init__(
        self,
        embedder: BaseEmbedder,
        collection_name: str = os.environ.get("QDRANT_INDEX_NAME", "default"),
        base_url: str = os.environ.get("QDRANT_ENDPOINT", "http://localhost:6333"),
        config: RetrieverConfig = RetrieverConfig.default(),
    ):
        super().__init__(embedder, config)

        self.client = QdrantClient(url=base_url)
        self.collection_name = collection_name
        self._ensure_collection_exists()

        self.vectorstore = QdrantVectorStore(
            client=self.client, 
            collection_name=self.collection_name,
            embedding=self.embedder.embedding_model,
        )

    def _ensure_collection_exists(self):
        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name, 
                vectors_config=models.VectorParams(
                    size=self.config.embedding_dim, 
                    distance=models.Distance.COSINE,
                ),
            )
    
    def delete(self, node_ids: List[str]):
        self.vectorstore.delete(node_ids)

    def update(self, documents: List[Dict]):
        self.add(documents)

    def reset(self):
        self.client.delete_collection(collection_name=self.config.collection_name)
        self._ensure_collection_exists()

    def retrieve(
        self, 
        query: str, 
        top_k: int,
        filter: models.Filter = None,
    ):
        retrieve_documents = self.vectorstore.similarity_search(
            query=query, 
            k=top_k, 
            filter=filter,
        )
        return retrieve_documents