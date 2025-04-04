from modules.retriever import BaseRetriever
from modules.generator import BaseGenerator
from configs import RAGConfig


class RAGPipeline:
    def __init__(
        self,
        retriever: BaseRetriever,
        llm_generator: BaseGenerator,
        config: RAGConfig = RAGConfig.default(),
    ):
        self.retriever = retriever
        self.llm_generator = llm_generator

        self.config = config

    def query(self, query: str, stream: bool = False):
        retrieved_documents = self.retriever.retrieve(query, top_k=self.config.top_k) 
        documents_content_string = "\n\n".join(
            f"Source: {doc.metadata['source']}\nContent: {doc.page_content}" for doc in retrieved_documents
        )
        retrieved_documents = [doc.dict() for doc in retrieved_documents]
        print(retrieved_documents)

        rag_messages = [
            ("system", self.config.system_message),
            ("human", f"Question: {query}"),
            ("human", f"Context: {documents_content_string}")
        ]

        response = self.llm_generator.generate(rag_messages, stream=stream) 
        return response, retrieved_documents