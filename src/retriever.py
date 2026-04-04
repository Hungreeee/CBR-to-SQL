from typing import List, Dict, Literal

from qdrant_client import QdrantClient, models
from qdrant_client.models import SparseVectorParams, SparseIndexParams

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
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

        self.dense_embedder = HuggingFaceEmbeddings(
            model_name=self.config.dense_embedding_model,
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

        self.is_lookup_table = "lookup_table" in collection_name
        
        self.sparse_embedder = FastEmbedSparse(
            model_name=self.config.sparse_embedding_model
        )

        self.client = QdrantClient(url=base_url)
        self._ensure_collection_exists()

        if self.is_lookup_table:
            self.vectorstore = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.dense_embedder,
            )
            self.sparse_vectorstore = None  # Not supported for lookup tables
            self._ingest_store = self.vectorstore
        else:
            self.vectorstore = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.dense_embedder,
                retrieval_mode=RetrievalMode.DENSE,
            )
            self.sparse_vectorstore = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.dense_embedder,
                sparse_embedding=self.sparse_embedder,
                retrieval_mode=RetrievalMode.SPARSE,
            )
            # Used only for ingestion (writes both dense + sparse vectors)
            self._ingest_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.dense_embedder,
                sparse_embedding=self.sparse_embedder,
                retrieval_mode=RetrievalMode.HYBRID,
            )

    def _ensure_collection_exists(self):
        if not self.client.collection_exists(collection_name=self.collection_name):
            vectors_config = models.VectorParams(
                size=self.config.embedding_dim, 
                distance=models.Distance.COSINE,
            )
            
            # Lookup table: no sparse vectors
            if self.is_lookup_table:
                self.client.create_collection(
                    collection_name=self.collection_name, 
                    vectors_config=vectors_config,
                )
            else:
                # Other collections: with sparse vectors
                self.client.create_collection(
                    collection_name=self.collection_name, 
                    vectors_config=vectors_config,
                    sparse_vectors_config={
                        "langchain-sparse": SparseVectorParams(
                            index=SparseIndexParams(on_disk=False)
                        )
                    }
                )

    def ingest(self, documents: List[Dict], indexed_field: str):
        documents_langchain = []
        for doc in documents:
            metadata = {key: value for key, value in doc.items() if key != indexed_field}
            documents_langchain.append(Document(
                page_content=doc[indexed_field], 
                metadata=metadata,
            ))
        self._ingest_store.add_documents(documents_langchain)
    
    def delete(self, node_ids: List):
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=node_ids
        )

    def reset(self):
        self.client.delete_collection(collection_name=self.collection_name)
        self._ensure_collection_exists()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter: models.Filter = None,
        score_threshold: float = None,
        mode: Literal["dense", "sparse"] = "dense",
    ) -> List[Document]:
        if mode == "sparse" and self.sparse_vectorstore is not None:
            vectorstore = self.sparse_vectorstore
        else:
            vectorstore = self.vectorstore

        return vectorstore.similarity_search(
            query=query,
            k=top_k,
            filter=filter,
            score_threshold=score_threshold,
        )
