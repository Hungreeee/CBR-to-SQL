import json
from typing import Tuple, List, Dict

from retriever import BaseRetriever
from generator import BaseGenerator
from configs import RAGConfig
import prompt_factory 
from schema import EntityExtractionResult

from langchain_community.utilities.sql_database import SQLDatabase


class RAG2SQL:
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
        sql_query, retrieved_cases = self.formulate_sql_query(query)
        try:
            sql_response = self.sql_db.run(sql_query)
        except: 
            sql_response = "DATA FAILED"

        messages = [
            ("system", prompt_factory.question_answering),
            ("human", f"Question: {query}"),
            ("human", f"Retrieved Data: {sql_response}"),
            ("human", f"Answer:"),
        ]
        response = self.generator.generate(messages) if self.config.return_response else None

        return {
            "response": response,
            "sql_query": sql_query,
            "sql_response": sql_response,
            "retrieved_cases": retrieved_cases,
        }
    
    def formulate_sql_query(self, query: str):
        # Retrieve
        retrieved_cases = self.retriever.retrieve(query, top_k=self.config.top_k) 
        formatted_string = "\n\n".join(
            f"Query: {doc.metadata['case']}\nSQL Query: {doc.metadata['sql_query']}" for doc in retrieved_cases
        )

        # Revise
        messages = [
            ("system", prompt_factory.case_revising),
            ("system", f"Schema Information:{self.sql_db.get_table_info()}"),
            ("human", f"Question: {query}"),
            ("human", f"Past SQL Examples: {formatted_string}"),
            ("human", f"Revised SQL Query:"),
        ]
        sql_query = self.generator.generate(messages) 
        return sql_query, retrieved_cases
    
    def retain(self, query: str, correct_sql: str):
        documents = [{
            "case": query,
            "sql_query": correct_sql,
        }]
        self.retriever.ingest(documents, "case")
    

class CBR2SQL(RAG2SQL):
    def __init__(
        self,
        retriever: BaseRetriever,
        generator: BaseGenerator,
        sql_db: SQLDatabase,
        lookup_table: BaseRetriever,
        config: RAGConfig = RAGConfig.default(),
    ):
        super().__init__(retriever, generator, sql_db, config)
        self.lookup_table = lookup_table

    def query(self, query: str) -> Dict:
        sql_query, retrieved_cases = self.formulate_sql_query(query)
        try:
            sql_response = self.sql_db.run(sql_query)
        except: 
            sql_response = "DATA FAILED"

        messages = [
            ("system", prompt_factory.question_answering),
            ("human", f"Question: {query}"),
            ("human", f"Retrieved Data: {sql_response}"),
            ("human", f"Answer:"),
        ]
        response = self.generator.generate(messages) if self.config.return_response else None

        return {
            "response": response,
            "sql_query": sql_query,
            "sql_response": sql_response,
            "retrieved_cases": retrieved_cases,
        }
    
    def formulate_sql_query(self, query: str):
        # Retrieve solution template
        masked_query, extracted_entities = self.get_masked_question(query)
        retrieved_cases = self.retriever.retrieve(masked_query, top_k=self.config.top_k) 
        formatted_string = "\n\n".join(
            f"Query: {doc.metadata['case']}\nSQL Query: {doc.metadata['sql_query']}" for doc in retrieved_cases
        )

        # Revise solution based on examples
        messages = [
            ("system", prompt_factory.template_formulation),
            ("system", f"Schema Information:{self.sql_db.get_table_info()}"),
            ("human", f"Question: {query}"),
            ("human", f"Past SQL Examples: {formatted_string}"),
            ("human", f"Revised SQL Query:"),
        ]
        sql_template = self.generator.generate(messages) 

        # Extract relevant entities
        relevant_entity_match = []
        for entity in extracted_entities:
            matches = self.lookup_table.retrieve(entity, self.config.top_k)
            relevant_entity_match.append({
                "entity": entity,
                "matches": [relevant_entity.model_dump() for relevant_entity in matches]
            })

        formatted_string = "\n\n".join(
            f"Entity: {doc['entity']}" + "\nRelevant matches:\n".join([
                f"\nRelevant Entity: {match['entity']}, Table: {match['table']}, Column: {match['column']}" \
                    for match in doc["matches"]
            ]) for doc in relevant_entity_match
        )

        # Revise solution based on examples
        messages = [
            ("system", prompt_factory.source_discovery),
            ("human", f"SQL Template: {sql_template}"),
            ("human", f"Relevant entity matches: {formatted_string}"),
            ("system", f"Schema Information:{self.sql_db.get_table_info()}"),
            ("human", f"Revised SQL Query:"),
        ]
        sql_query = self.generator.generate(messages) 
        return sql_query, retrieved_cases

    def retain(self, query: str, correct_sql: str):
        """
        Construct an abstract problem-solving case, and retain it.
        """
        documents = [{
            "masked_case": self.get_masked_question(query)[0],
            "case": query,
            "sql_query": correct_sql,
            "explanation": self.get_explanation(query, correct_sql)
        }]
        self.retriever.ingest(documents, "masked_case")

    def get_explanation(self, query: str, correct_sql: str) -> str:
        """
        Generate an explanation of the formulation of the SQL query from the NL query.
        """
        messages = [
            ("system", prompt_factory.case_explanation),
            ("human", f"Natural Language Query: {query}"),
            ("human", f"SQL: {correct_sql}"),
            ("human", f"Explanation:"),
        ]
        explanation = self.generator.generate(messages)
        return explanation

    def get_masked_question(self, query: str) -> Tuple[str, List[Dict]]:
        """
        Mask entities out of the query, and return the extracted entities.
        """
        # Bind function calling to LLM
        llm_extraction = self.generator.client.bind_tools([EntityExtractionResult], strict=True)

        # Extract entities from the query
        messages = [
            ("system", prompt_factory.entity_extraction),
            ("human", f"Input text: {query}"),
        ]
        response = llm_extraction.generate(messages)
        extracted_entities = json.loads(response.additional_kwargs['tool_calls'][0]["function"]["arguments"])["entities"]

        # Mask the entities from the query
        masked_text = query
        for entity in extracted_entities:
            masked_text = masked_text.replace(entity.name, f"{entity.type.upper()}")

        return masked_text, extracted_entities