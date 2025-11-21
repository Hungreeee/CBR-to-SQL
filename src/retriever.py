from typing import List, Dict

from qdrant_client import QdrantClient, models

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from configs import RetrieverConfig

from dotenv import load_dotenv
load_dotenv()


class BaseRetriever:
    def __init__(
        self, 
        collection_name: str, 
        config: RetrieverConfig = RetrieverConfig.default()
    ):
        self.config = config
        self.collection_name = collection_name

        self.embedder = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
            model_kwargs={"device": self.config.device},
        )

    def retrieve(self, query: str, top_k: int) -> List[Dict]:
        raise NotImplementedError()

    def ingest(self, documents: List[Dict], indexed_field: str) -> None:
        raise NotImplementedError()

    def reset(self) -> None:
        raise NotImplementedError()


class QdrantRetriever(BaseRetriever):
    def __init__(
        self, 
        collection_name: str = "main",
        config: RetrieverConfig = RetrieverConfig.default(), 
        base_url: str = "http://localhost:6333",
    ):
        super().__init__(collection_name, config)

        self.client = QdrantClient(url=base_url)
        self._ensure_collection_exists()

        self.vectorstore = QdrantVectorStore(
            client=self.client, 
            collection_name=self.collection_name,
            embedding=self.embedder,
            # sparse_embedding=,
            # retrieval_mode=RetrievalMode.HYBRID,
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

    def ingest(self, documents: List[Dict], indexed_field: str):
        documents_langchain = []
        for doc in documents:
            metadata = {key: value for key, value in doc.items() if key != indexed_field}
            documents_langchain.append(Document(
                page_content=doc[indexed_field], 
                metadata=metadata,
            ))
        self.vectorstore.add_documents(documents_langchain)
    
    def delete(self, node_ids: List):
        self.vectorstore.delete(node_ids)

    def reset(self):
        self.client.delete_collection(collection_name=self.collection_name)
        self._ensure_collection_exists()

    def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        filter: models.Filter = None,
        score_threshold: float = None,
    ) -> List[Dict]:
        retrieve_documents = self.vectorstore.similarity_search(
            query=query, 
            k=top_k, 
            filter=filter,
            score_threshold=score_threshold,
        )
        return retrieve_documents