from retriever import BaseRetriever
from generator import BaseGenerator
from configs import RAGConfig
import prompt_factory 

from langchain_community.utilities.sql_database import SQLDatabase


class RAGPipeline:
    def __init__(
        self,
        retriever: BaseRetriever,
        generator: BaseGenerator,
        config: RAGConfig = RAGConfig.default(),
    ):
        self.retriever = retriever
        self.generator = generator

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

        response = self.generator.generate(rag_messages, stream=stream) 
        return response, retrieved_documents
    

class CBR_SQLAgent:
    def __init__(
        self,
        retriever: BaseRetriever,
        generator: BaseGenerator,
        sql_db: SQLDatabase,
        config: RAGConfig = RAGConfig.default(),
    ):
        self.retriever = retriever
        self.generator = generator
        self.sql_db = sql_db
        self.config = config

    def query(self, query: str):
        sql_query = self.formulate_SQL_query(query)
        sql_response = self.sql_db.run(sql_query)

        messages = [
            ("system", prompt_factory.question_answering),
            ("human", f"Question: {query}"),
            ("human", f"Context: {sql_response}")
        ]
        response = self.generator.generate(messages) 
        return response
    
    def formulate_SQL_query(self, query: str):
        # Retrieve
        retrieved_cases = self.retriever.retrieve(query, top_k=self.config.top_k) 
        formatted_string = "\n\n".join(
            f"Query: {doc.page_content}\nSQL Query: {doc.metadata['source']}" for doc in retrieved_cases
        )

        # Revise
        messages = [
            ("system", prompt_factory.case_revising),
            ("human", f"Question: {query}"),
            ("human", f"Context: {formatted_string}")
        ]
        sql_query = self.generator.generate(messages) 
        return sql_query
    
    def retain(self, query: str, correct_SQL: str):
        self.retriever.ingest()

    
def get_masked_question(query: str, generator: BaseGenerator):
    messages = [
        ("system", prompt_factory.entity_masking),
        ("human", f"Input text: {query}")
        ("human", f"Masked text:")
    ]

    masked_text = generator.generate(messages)
    return masked_text