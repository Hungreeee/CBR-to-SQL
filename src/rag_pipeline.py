
import nltk
import json
from typing import Tuple, List, Dict

from utils import *
import prompt_factory 
from retriever import BaseRetriever
from generator import BaseGenerator
from configs import RAGConfig
from schema import MaskingResults

from langchain.callbacks import get_openai_callback
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

    def query(self, query: str) -> Dict:
        with get_openai_callback() as cb:
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
            "token_usage": {
                "total_tokens": cb.total_tokens,
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "successful_requests": cb.successful_requests,
            }
        }
    
    def formulate_sql_query(self, query: str):
        # Retrieve
        retrieved_cases = self.retriever.retrieve(query, top_k=self.config.top_k) 

        if self.config.brittle_retrieval:
            retrieved_cases = drop_cases(retrieved_cases)

        formatted_string = "\n\n".join(
            f"Query: {doc.page_content}\nSQL Query: {doc.metadata['sql_query']}" for doc in retrieved_cases
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
        return remove_sql_wrapper(sql_query), retrieved_cases
    
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
        with get_openai_callback() as cb:
            masked_query, extracted_entities = self.get_masked_question(query)
            
            if self.config.template_construction:
                sql_query, retrieved_cases = self.formulate_sql_template(query, masked_query, extracted_entities)
            else:
                sql_query, retrieved_cases = self.formulate_sql_query(query)

            if self.config.source_discovery:
                sql_query, relevant_entity_match = self.discover_sources(query, sql_query, extracted_entities)
            else:
                relevant_entity_match = []

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
            "relevant_entities": relevant_entity_match,
            "token_usage": {
                "total_tokens": cb.total_tokens,
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "successful_requests": cb.successful_requests,
            }
        }
    
    def retain(self, query: str, correct_sql: str) -> bool:
        """
        Construct a masked case, and retain it.
        """
        masked_case, extracted_entities = self.get_masked_question(query)
        documents = [{
            "masked_case": masked_case,
            "related_entities": extracted_entities,
            "case": query,
            "sql_query": correct_sql,
        }]
        self.retriever.ingest(documents, "masked_case")

    def get_masked_question(self, query: str) -> Tuple[str, List[Dict]]:
        """
        Mask entities out of the query, and return the extracted entities.
        """
        # Bind function calling to LLM
        llm_extraction = self.generator.client.bind_tools([MaskingResults], strict=True)

        # Extract entities from the query
        messages = [
            ("system", prompt_factory.entity_extraction),
            ("human", query),
        ]
        response = llm_extraction.invoke(messages)  

        if "tool_calls" in response.additional_kwargs:
            tool_call_results = json.loads(response.additional_kwargs['tool_calls'][0]["function"]["arguments"])
            extracted_entities = tool_call_results["redacted_entities"]
            masked_text = tool_call_results["masked_sentence"]

        else:
            masked_text = query
            extracted_entities = []

        return masked_text, extracted_entities

    def formulate_sql_template(
        self, 
        query: str, 
        masked_query: str,
        extracted_entities: List[Dict],
    ) -> Tuple[str, List[Dict]]:
        """
        Write a SQL template based on similar past cases.
        """
        # Retrieve solution template
        retrieved_cases = self.retriever.retrieve(masked_query, top_k=self.config.top_k) 

        if self.config.brittle_retrieval:
            retrieved_cases = drop_cases(retrieved_cases)

        formatted_string = "\n\n-----\n\n".join(
            f"Query: {doc.metadata['case']}\nSQL Query: {doc.metadata['sql_query']}" for doc in retrieved_cases
        )
        
        masking_entities = [entity for entity in extracted_entities if entity["label"] in [
            "CONDITION", "PROCEDURE", "ETHNICITY",
            "DRUG", "NAME", "RELIGION", "EQUIPMENT",
        ]]

        # Construct solution template based on examples
        messages = [
            ("system", prompt_factory.template_formulation),
            ("system", f"Schema Information:{self.sql_db.get_table_info()}"),
            ("human", f"Entities to highlight: {masking_entities}"),
            ("human", f"User Question: {query}"),
            ("human", f"Past SQL Examples: {formatted_string}"),
            ("human", f"Revised SQL Query:"),
        ]
        response = self.generator.generate(messages, return_content=False) 
        sql_template = response.content
        return remove_sql_wrapper(sql_template), retrieved_cases

    def discover_sources(
        self, 
        query: str, 
        sql_template: str, 
        extracted_entities: List[Dict]
    ) -> Tuple[str, List[Dict]]:
        """
        Modify templates to adapt to true sources of the entities.
        """
        # Extract relevant entities
        # To-do: Re-rank matches using Levenshtein distance
        relevant_entity_match = []
        
        for entity in extracted_entities:
            if entity["label"] in [
                "CONDITION", "PROCEDURE", "ETHNICITY",
                "DRUG", "NAME", "RELIGION", "EQUIPMENT",
            ]:
                relevant_matches = self.lookup(entity["value"])
                relevant_entity_match.append({
                    "entity": entity["value"],
                    "matches": relevant_matches,
                })

        formatted_string = "\n-----\n".join(
            f"Relevant matches for entity: `{doc['entity']}`\n" + "\n".join([
                f"Relevant match #{idx}: `{match['page_content']}`, Table: `{match['metadata']['table']}`, Column: `{match['metadata']['column']}`" \
                    for idx, match in enumerate(doc["matches"])
            ]) for doc in relevant_entity_match
        )

        # Revise solution based on examples
        if relevant_entity_match:
            messages = [
                ("system", prompt_factory.source_discovery),
                ("human", f"User Question: {query}"),
                ("human", f"Initial SQL Template: {sql_template}"),
                ("human", f"Relevant entity matches: {formatted_string}"),
                ("human", f"Revised SQL Query:"),
            ]
            sql_query = self.generator.generate(messages) 
            
        else:
            sql_query = sql_template

        return remove_sql_wrapper(sql_query), relevant_entity_match
    
    def lookup(self, query: str, retrieval_range: int = 100, top_k: int = 5) -> List[Dict]:
        """
        Retrieve similar (real) entities given a seed value
        Retrieve with cosine similarity, re-rank using Levenshtein distance
        """
        matches = self.lookup_table.retrieve(query, top_k=retrieval_range)
        reranked_results = []

        for match in matches:
            match_dict = match.model_dump()
            reranked_results.append({
                **match_dict,
                "score": nltk.edit_distance(
                    " ".join(sorted(tokenize(query))), 
                    " ".join(sorted(tokenize(match_dict["page_content"])))
                )
            })

        reranked_results = sorted(reranked_results, key=lambda x: x["score"])[:top_k]
        return reranked_results
