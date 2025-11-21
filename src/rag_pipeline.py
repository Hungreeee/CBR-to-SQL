import nltk
import json
from typing import Tuple, List, Dict

import prompt_factory 
from utils import drop_cases, remove_sql_wrapper, tokenize
from retriever import BaseRetriever
from generator import BaseGenerator
from configs import RAGConfig
from schema import EntityExtraction, TagAssignment, EntityValidation

from langchain_community.callbacks import get_openai_callback
from langchain_community.utilities.sql_database import SQLDatabase


class RAGtoSQL:
    def __init__(
        self, 
        retriever: BaseRetriever,
        generator: BaseGenerator, 
        sql_db: SQLDatabase,
        config: RAGConfig = RAGConfig.default()
    ):
        self.retriever = retriever
        self.generator = generator
        self.sql_db = sql_db
        self.config = config

    def handle_request(self, question: str) -> Dict:
        with get_openai_callback() as callback:
            sql_query, retrieved_cases = self.generate_sql(question)
            execution_results = self._execute_sql(sql_query)
        
        return {
            "sql_query": sql_query,
            "retrieved_cases": retrieved_cases,
            "execution_results": execution_results,
            "token_usage": {
                "total_tokens": callback.total_tokens,
                "prompt_tokens": callback.prompt_tokens,
                "completion_tokens": callback.completion_tokens,
                "successful_requests": callback.successful_requests,
            }
        }

    def generate_sql(self, question: str) -> Tuple[str, List]:
        retrieved_cases = self.retriever.retrieve(question, top_k=self.config.top_k)

        if self.config.brittle_retrieval:
            retrieved_cases = drop_cases(retrieved_cases)

        formatted_examples = "\n\n".join(
            f"Query: {doc.page_content}\nSQL: {doc.metadata['sql_query']}" 
            for doc in retrieved_cases
        )
        
        messages = [
            ("system", prompt_factory.case_revising),
            ("system", f"Schema:\n{self.sql_db.get_table_info()}"),
            ("human", f"Question: {question}"),
            ("human", f"Examples:\n{formatted_examples}"),
            ("human", "SQL Query:"),
        ]
        
        sql_query = self.generator.generate(messages) 
        return remove_sql_wrapper(sql_query), retrieved_cases
    
    def _execute_sql(self, sql_query: str) -> str:
        try:
            return self.sql_db.run(sql_query)
        except: 
            return "EXECUTION FAILED"
    
    def retain_case(self, question: str, sql_query: str) -> None:
        self.retriever.ingest(
            documents=[{"case": question, "sql_query": sql_query}],
            indexed_field="case"
        )


class CBRtoSQL(RAGtoSQL):
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

    def handle_request(self, question: str) -> Dict:
        with get_openai_callback() as callback:
            sql_query, retrieved_cases, entities = self.generate_sql(question)
            execution_results = self._execute_sql(sql_query)
        
        return {
            "sql_query": sql_query,
            "retrieved_cases": retrieved_cases,
            "entities": entities,
            "execution_results": execution_results,
            "token_usage": {
                "total_tokens": callback.total_tokens,
                "prompt_tokens": callback.prompt_tokens,
                "completion_tokens": callback.completion_tokens,
                "successful_requests": callback.successful_requests,
            }
        }
    
    def generate_sql(self, question: str) -> Tuple[str, List, List]:
        """Pipeline: Source Discovery -> Template Construction -> Slot Filling"""
        masked_question, entities = self.source_discovery(question)
        sql_template, retrieved_cases = self._construct_template(masked_question)
        final_sql = self._fill_slots(question, sql_template, entities)
        return remove_sql_wrapper(final_sql), retrieved_cases, entities
    
    def retain_case(self, question: str, sql_query: str) -> None:
        """Store case with offline entity tagging"""
        masked_question, entities = self.source_discovery(question)
        
        self.retriever.ingest(
            documents=[{
                "masked_case": masked_question,
                "case": question,
                "sql_query": sql_query,
                "entities": entities
            }],
            indexed_field="masked_case"
        )
    
    # ========== SOURCE DISCOVERY: 3-ROUND ITERATIVE REFINEMENT ==========
    
    def source_discovery(self, question: str, max_rounds: int = 2) -> Tuple[str, List[Dict]]:
        """
        Iterative entity discovery with feedback-based refinement.
        
        Each round:
        1. Extract noun phrases
        2. Schema linking (k-NN search)
        3. Tag assignment + validation
        
        If validation fails, feedback is provided for the next round.
        """
        tagged_entities = []
        feedback_history = []
        
        for round_num in range(max_rounds):
            # Round context: include feedback from previous rounds
            round_context = {
                "round": round_num + 1,
                "feedback": feedback_history
            }
            
            # Step 1: Extract noun phrases (with feedback context)
            noun_phrases = self._extract_noun_phrases(question, round_context)
            print(noun_phrases)
            
            if not noun_phrases:
                break
            
            # Track entities that need refinement in next round
            entities_needing_refinement = []
            
            # Step 2 & 3: For each noun phrase, link to schema and assign tags
            for noun_phrase in noun_phrases:
                # Skip if already successfully tagged in previous rounds
                if any(e["original"] == noun_phrase for e in tagged_entities):
                    continue
                
                # Step 2: Schema linking via k-NN search
                linked_entities = self._lookup(noun_phrase)
                
                if not linked_entities:
                    # No matches found - needs refinement
                    entities_needing_refinement.append({
                        "noun_phrase": noun_phrase,
                        "issue": "no_matches_found",
                        "feedback": "No database entities found. Try extracting more specific or alternative phrasings."
                    })
                    continue
                
                # Step 3: Tag assignment
                tag_result = self._assign_tag(question, noun_phrase, linked_entities)
                print(tag_result)
                print(linked_entities)
                print("------")
                
                # Validate the assigned tag and match
                validation = self._validate_entity(question, noun_phrase, tag_result, linked_entities)
                print(validation)
                print(tag_result)
                print("======")
                
                if validation["is_acceptable"]:
                    # Entity successfully tagged and validated
                    tagged_entities.append({
                        "original": noun_phrase,
                        "label": tag_result["label"],
                        "best_match": tag_result["best_match"],
                        "table": tag_result.get("table"),
                        "column": tag_result.get("column"),
                        "round": round_num + 1
                    })
                else:
                    # Validation failed - needs refinement
                    entities_needing_refinement.append({
                        "noun_phrase": noun_phrase,
                        "issue": "validation_failed",
                        "feedback": validation.get("feedback", ""),
                        "attempted_match": tag_result.get("best_match")
                    })
            
            # If all entities validated successfully, we're done
            if not entities_needing_refinement:
                break
            
            # Store feedback for next round
            feedback_history.append({
                "round": round_num + 1,
                "entities_needing_refinement": entities_needing_refinement
            })
        
        # Generate masked question from successfully tagged entities
        masked_question = self._mask_question(question, tagged_entities)
        
        return masked_question, tagged_entities
    
    # ========== STEP 1: NOUN PHRASE EXTRACTION ==========
    
    def _extract_noun_phrases(self, question: str, round_context: Dict) -> List[str]:
        """
        Extract noun phrases with feedback from previous rounds.
        """
        llm = self.generator.client.bind_tools([EntityExtraction], strict=True)

        messages = [("system", prompt_factory.entity_extraction)]

        # Previous round feedback
        if round_context.get("feedback"):
            feedback_text = self._format_feedback(round_context["feedback"])
            messages.append(("system", f"Feedback from previous rounds:\n{feedback_text}"))
            messages.append(("system", "Please refine your extraction based on the feedback above."))

        messages.append(("human", f"Question: {question}"))

        response = llm.invoke(messages)

        tool_calls = getattr(response, "tool_calls", None)

        if not tool_calls:
            return []

        args = tool_calls[0].get("args", {})
        return args.get("entities", [])

    
    def _format_feedback(self, feedback_history: List[Dict]) -> str:
        """Format feedback history for LLM context"""
        formatted = []
        for entry in feedback_history:
            formatted.append(f"Round {entry['round']}:")
            for entity in entry["entities_needing_refinement"]:
                formatted.append(f"  - '{entity['noun_phrase']}': {entity['feedback']}")
        return "\n".join(formatted)
    
    # ========== STEP 2: SCHEMA LINKING ==========
    
    def _lookup(self, noun_phrase: str, top_k: int = 5) -> List[Dict]:
        """
        k-NN search against database entities (schema linking)
        """
        matches = self.lookup_table.retrieve(noun_phrase, top_k=100)

        print(matches)
        
        scored_matches = []
        for match in matches:
            match_dict = match.model_dump()
            score = nltk.edit_distance(
                " ".join(sorted(tokenize(noun_phrase))),
                " ".join(sorted(tokenize(match_dict["page_content"])))
            )
            scored_matches.append({
                "value": match_dict["page_content"],
                "table": match_dict["metadata"].get("table"),
                "column": match_dict["metadata"].get("column"),
                "score": score
            })
        
        return sorted(scored_matches, key=lambda x: x["score"])[:top_k]
    
    # ========== STEP 3: TAG ASSIGNMENT ==========
    
    def _assign_tag(self, question: str, noun_phrase: str, linked_entities: List[Dict]) -> Dict:
        """
        Assign semantic tag based on schema context
        """
        formatted_entities = "\n".join([
            f"{i+1}. Value: '{e['value']}', Table: {e['table']}, Column: {e['column']}, Score: {e['score']}"
            for i, e in enumerate(linked_entities)
        ])
        
        llm = self.generator.client.bind_tools([TagAssignment], strict=True)
        
        messages = [
            ("system", prompt_factory.tag_assignment),
            ("human", f"Question: {question}"),
            ("human", f"Noun phrase: '{noun_phrase}'"),
            ("human", f"Linked database entities:\n{formatted_entities}"),
        ]
        
        response = llm.invoke(messages)
        
        tool_calls = getattr(response, "tool_calls", None)
        if tool_calls:
            try:
                args = tool_calls[0].get("args", {})
                best_match_idx = args.get("best_match_index", 0)

                if 0 <= best_match_idx < len(linked_entities):
                    best = linked_entities[best_match_idx]
                    return {
                        "label": args.get("label", "MASKED"),
                        "best_match": best["value"],
                        "table": best["table"],
                        "column": best["column"],
                    }

            except (KeyError, IndexError, TypeError):
                pass

        
        # Fallback
        return {
            "label": "MASKED",
            "best_match": linked_entities[0]["value"] if linked_entities else noun_phrase,
            "table": linked_entities[0].get("table") if linked_entities else None,
            "column": linked_entities[0].get("column") if linked_entities else None
        }
    
    # ========== VALIDATION WITH FEEDBACK ==========
    
    def _validate_entity(
        self, 
        question: str, 
        noun_phrase: str, 
        tag_result: Dict,
        linked_entities: List[Dict]
    ) -> Dict:
        """
        Validate if the tag assignment is acceptable, provide feedback if not
        """
        llm = self.generator.client.bind_tools([EntityValidation], strict=True)
        
        formatted_entities = "\n".join([
            f"{i+1}. '{e['value']}' (Table: {e['table']}, Column: {e['column']})"
            for i, e in enumerate(linked_entities)
        ])
        
        messages = [
            ("system", prompt_factory.entity_validation),
            ("human", f"Question: {question}"),
            ("human", f"Noun phrase: '{noun_phrase}'"),
            ("human", f"Assigned tag: {tag_result['label']}"),
            ("human", f"Selected match: '{tag_result['best_match']}' (Table: {tag_result.get('table')}, Column: {tag_result.get('column')})"),
            ("human", f"All available matches:\n{formatted_entities}"),
        ]
        
        response = llm.invoke(messages)
        
        tool_calls = getattr(response, "tool_calls", None)

        if tool_calls:
            try:
                args = tool_calls[0].get("args", {})
                return args
            except (KeyError, TypeError):
                pass

        # Default: accept the match
        return {"is_acceptable": True}
    
    # ========== TEMPLATE CONSTRUCTION & SLOT FILLING ==========
    
    def _construct_template(self, masked_question: str) -> Tuple[str, List]:
        """Generate SQL template from masked question"""
        retrieved_cases = self.retriever.retrieve(masked_question, top_k=self.config.top_k)
        
        if self.config.brittle_retrieval:
            retrieved_cases = drop_cases(retrieved_cases)
        
        formatted_examples = "\n\n".join(
            f"Query: {doc.metadata['case']}\nSQL: {doc.metadata['sql_query']}" 
            for doc in retrieved_cases
        )
        
        messages = [
            ("system", prompt_factory.template_formulation),
            ("system", f"Schema:\n{self.sql_db.get_table_info()}"),
            ("human", f"Masked Question: {masked_question}"),
            ("human", f"Examples:\n{formatted_examples}"),
            ("human", "SQL Template:"),
        ]
        
        sql_template = self.generator.generate(messages)
        return remove_sql_wrapper(sql_template), retrieved_cases
    
    def _fill_slots(self, question: str, sql_template: str, entities: List[Dict]) -> str:
        """Replace entity tags with actual values"""
        if not entities:
            return sql_template
        
        entity_info = "\n".join([
            f"[{e['label']}] -> '{e['best_match']}' (Table: {e.get('table')}, Column: {e.get('column')})"
            for e in entities
        ])
        
        messages = [
            ("system", prompt_factory.slot_filling),
            ("human", f"Question: {question}"),
            ("human", f"Template:\n{sql_template}"),
            ("human", f"Entities:\n{entity_info}"),
            ("human", f"Schema:\n{self.sql_db.get_table_info()}"),
            ("human", "Final SQL:"),
        ]
        
        return self.generator.generate(messages)
    
    def _mask_question(self, question: str, entities: List[Dict]) -> str:
        """Replace entity values with tags"""
        masked = question
        for entity in sorted(entities, key=lambda x: len(x["original"]), reverse=True):
            masked = masked.replace(entity["original"], f"[{entity['label']}]")
        return masked