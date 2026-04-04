import nltk
from typing import Tuple, List, Dict, Optional

import prompt_factory
from utils import drop_cases, remove_sql_wrapper, tokenize, is_content_filter_error, rrf_fuse
from retriever import BaseRetriever
from generator import BaseGenerator
from configs import RAGConfig
from schema import EntityExtraction, TagAssignment, PromptExtension

from langchain_core.documents import Document
from langchain_community.callbacks import get_openai_callback
from langchain_community.utilities.sql_database import SQLDatabase

from sentence_transformers import CrossEncoder


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
        if config.doc_reranking:
            self.reranker = CrossEncoder(config.reranker_embedding_model)

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
        top_k_multiplier = 5 if self.config.doc_reranking and \
            not self.config.prompt_extension else 1

        # Retrieve cases from main question
        retrieved_cases: List[Document] = self._retrieve(
            question, top_k=self.config.top_k * top_k_multiplier
        )

        if self.config.prompt_extension:
            subquestions = self._decompose_prompt(question)

            for subq in subquestions:
                sub_cases = self._retrieve(subq, top_k=self.config.top_k)
                retrieved_cases.extend(sub_cases)

            # Deduplicate after all sub-questions are processed
            seen = set()
            unique_cases = []
            for doc in retrieved_cases:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    unique_cases.append(doc)
            retrieved_cases = unique_cases

            if self.config.doc_reranking:
                retrieved_cases = self._rerank_documents(
                    query=question,
                    documents=retrieved_cases,
                    top_k=self.config.top_k
                )

            if self.config.brittle_retrieval:
                retrieved_cases = drop_cases(retrieved_cases, len(retrieved_cases))

            formatted_examples = "\n\n".join(
                f"Query: {doc.page_content}\nSQL: {doc.metadata['sql_query']}"
                for doc in retrieved_cases
            )
        else:
            if self.config.doc_reranking:
                retrieved_cases = self._rerank_documents(
                    query=question,
                    documents=retrieved_cases,
                    top_k=self.config.top_k
                )

            if self.config.brittle_retrieval:
                retrieved_cases = drop_cases(retrieved_cases, len(retrieved_cases))

            formatted_examples = "\n\n".join(
                f"Query: {doc.page_content}\nSQL: {doc.metadata['sql_query']}" 
                for doc in retrieved_cases
            )
        
        messages = [
            ("system", self._get_prompt("case_revising")),
            ("system", f"Schema:\n{self.sql_db.get_table_info()}"),
            ("user", f"Question: {question}"),
            ("user", f"Examples:\n{formatted_examples}"),
            ("user", "SQL Query:"),
        ]
        
        sql_query = self.generator.generate(messages) 
        return remove_sql_wrapper(sql_query), retrieved_cases
    
    def _retrieve(self, query: str, top_k: int) -> List[Document]:
        """Single-collection retrieval; fuses dense+sparse via RRF when hybrid is enabled."""
        if self.config.hybrid_retrieval:
            dense = self.retriever.retrieve(query, top_k=top_k, mode="dense")
            sparse = self.retriever.retrieve(query, top_k=top_k, mode="sparse")
            return rrf_fuse(dense, sparse, top_k=top_k)
        return self.retriever.retrieve(query, top_k=top_k, mode="dense")

    def _execute_sql(self, sql_query: str) -> str:
        try:
            return self.sql_db.run(sql_query)
        except Exception:
            return "EXECUTION FAILED"
    
    def retain_case(self, question: str, sql_query: str) -> None:
        self.retriever.ingest(
            documents=[{"case": question, "sql_query": sql_query}],
            indexed_field="case"
        )

    def _get_prompt(self, prompt_type: str) -> str:
        """Get dataset-specific prompt"""
        return getattr(
            getattr(prompt_factory, f"{self.config.dataset}_prompts"),
            prompt_type
        )
    
    def _decompose_prompt(self, question: str) -> List[str]:
        llm = self.generator.client.bind_tools([PromptExtension], tool_choice="PromptExtension", strict=True)
        
        messages = [
            ("system", self._get_prompt("prompt_extension")),
            ("user", f"# Question: {question}"),
        ]

        try: 
            response = llm.invoke(messages)
        except Exception as e:
            raise e
        
        tool_calls = getattr(response, "tool_calls", None)
        if tool_calls:
            try:
                args = tool_calls[0].get("args", {})
                decomposed_prompts = args.get("decomposed_prompts", [])
                return decomposed_prompts

            except (KeyError, IndexError, TypeError):
                pass

        # Fallback: rejection (parsing failed or invalid response)
        return []

    def _rerank_documents(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: int = 5
    ) -> List:
        """
        Re-rank documents using cross-encoder based on relevance to query
        
        Args:
            query: The original query to rank against
            documents: List of retrieved documents
            top_k: Number of top documents to return
            
        Returns:
            Top-k re-ranked documents
        """
        if not documents:
            return []
        
        # Create query-document pairs
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Score all pairs with cross-encoder
        scores = self.reranker.predict(pairs)
        
        # Combine documents with scores and sort
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: -x[1])  # Sort by score descending
        
        # Return top k documents
        return [doc for doc, _ in doc_scores[:top_k]]


class CBRtoSQL(RAGtoSQL):
    def __init__(
        self,
        retriever: BaseRetriever,
        generator: BaseGenerator,
        sql_db: SQLDatabase,
        lookup_table: BaseRetriever,
        config: RAGConfig = RAGConfig.default(),
        fallback_generator: Optional[BaseGenerator] = None,
        rag_retriever: Optional[BaseRetriever] = None,
    ):
        super().__init__(retriever, generator, sql_db, config)
        self.lookup_table = lookup_table
        self.fallback_generator = fallback_generator
        self.rag_retriever = rag_retriever

    def handle_request(self, question: str) -> Dict:
        with get_openai_callback() as callback:
            sql_query, retrieved_cases, entities = self.generate_sql(question)
            # execution_results = self._execute_sql(sql_query)
            execution_results = None
        
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
        final_sql, retrieved_cases = self._construct_and_fill_sql(masked_question, question, entities)
        return remove_sql_wrapper(final_sql), retrieved_cases, entities
    
    def retain_case(self, question: str, sql_query: str) -> None:
        """Store case with offline entity tagging"""
        try:
            masked_question, entities = self.source_discovery(question)
        except Exception as e:
            print(f"Error occurred for question '{question}': {str(e)}\n\n")
            masked_question, entities = question, []
        
        self.retriever.ingest(
            documents=[{
                "masked_case": masked_question,
                "case": question,
                "sql_query": sql_query,
                "entities": entities
            }],
            indexed_field="masked_case"
        )
    
    def source_discovery(self, question: str) -> Tuple[str, List[Dict]]:
        """
        Two-phase entity discovery:
        Phase 1: Extract all entities and assign temporary context-based tags
        Phase 2: For semantic entities, perform lookup and refined tagging
        Phase 3: Perform masking using tagged entities
        """
        # PHASE 1: Extract entities and assign 'temporary' tags
        extracted_entities = self._extract_all_entities(question)
        
        if not extracted_entities:
            return question, []
        
        # Separate entities by whether they need source discovery
        entities_needing_discovery = []
        entities_no_discovery = []
        
        for entity_info in extracted_entities:
            if entity_info["requires_source_discovery"]:
                entities_needing_discovery.append(entity_info)
            else:
                # Structural entities: keep temporary tag, use original text
                entities_no_discovery.append({
                    "original": entity_info["text"],
                    "label": entity_info["temp_label"],
                    "best_match": entity_info["text"],
                    "table": None,
                    "column": None,
                    "source_discovery": False
                })
        
        # PHASE 2: Lookup and refined tagging for semantic entities
        tagged_entities = entities_no_discovery.copy()
        
        if self.config.source_discovery:
            for entity_info in entities_needing_discovery:
                entity_text = entity_info["text"]
                
                # STEP 1: Schema linking via k-NN search
                linked_entities = self._lookup(entity_text)
                
                if not linked_entities:
                    # No matches found - use original text with temporary tag
                    tagged_entities.append({
                        "original": entity_text,
                        "label": entity_info["temp_label"],
                        "best_match": entity_text,
                        "table": None,
                        "column": None,
                        "source_discovery": True,
                        "match_found": False
                    })
                    continue
                
                # STEP 2: Refined tag assignment (select best match)
                tag_result = self._assign_tag(question, entity_text, linked_entities)
                
                # Check if match was rejected (best_match_index = -1)
                if tag_result is None or tag_result.get("label") == "NO_MATCH":
                    # Rejected - use original text with temporary tag
                    tagged_entities.append({
                        "original": entity_text,
                        "label": entity_info["temp_label"],  # Keep original temp tag
                        "best_match": entity_text,
                        "table": None,
                        "column": None,
                        "source_discovery": True,
                        "match_found": False
                    })
                else:
                    # Accepted - use matched result
                    tagged_entities.append({
                        "original": entity_text,
                        "label": tag_result["label"],  # Use refined tag
                        "best_match": tag_result["best_match"],
                        "table": tag_result.get("table"),
                        "column": tag_result.get("column"),
                        "source_discovery": True,
                        "match_found": True
                    })
        
        # PHASE 3: Generate masked question from all tagged entities
        masked_question = self._mask_question(question, tagged_entities)
        
        return masked_question, tagged_entities
    
    def _extract_all_entities(self, question: str) -> List[Dict]:
        """
        Extract ALL entities (semantic + structural) and assign temporary context-based tags in ONE call
        """
        llm = self.generator.client.bind_tools([EntityExtraction], strict=True, tool_choice="EntityExtraction")
        messages = [
            ("system", self._get_prompt("entity_extraction")),
            ("user", f"Question: {question}")
        ]
        
        try: 
            response = llm.invoke(messages)
        except Exception as e:
            if is_content_filter_error(e) and self.fallback_generator:
                llm = self.fallback_generator.client.bind_tools([EntityExtraction], strict=True, tool_choice="EntityExtraction")
                response = llm.invoke(messages)
            else:
                raise

        tool_calls = getattr(response, "tool_calls", None)
        
        if not tool_calls:
            return []
        
        args = tool_calls[0].get("args", {})
        tagged_entities = args.get("entities", [])
        
        if not tagged_entities:
            return []
        
        # Convert to internal format
        entity_info_list = []
        for entity in tagged_entities:
            entity_info_list.append({
                "text": entity["value"],
                "temp_label": entity["label"],
                "requires_source_discovery": entity["is_semantic"],
            })
        
        return entity_info_list
    
    def _lookup(self, entity: str, top_k: int = 5) -> List[Dict]:
        """
        k-NN search against database entities (schema linking)
        Uses minimum of order-sensitive and order-insensitive scores
        """
        matches = self.lookup_table.retrieve(entity, top_k=1000)
        
        scored_matches = []
        for match in matches:
            match_dict = match.model_dump()
            match_value = match_dict["page_content"]
            
            # Score 1: Order-sensitive (preserves word order)
            score_ordered = nltk.edit_distance(
                entity.lower(),
                match_value.lower()
            )
            
            # Score 2: Order-insensitive (bag of words)
            score_unordered = nltk.edit_distance(
                " ".join(sorted(tokenize(entity))),
                " ".join(sorted(tokenize(match_value)))
            )
            
            # Take the best (lowest) score from either approach
            final_score = min(score_ordered, score_unordered)
            
            scored_matches.append({
                "value": match_value,
                "table": match_dict["metadata"].get("table"),
                "column": match_dict["metadata"].get("column"),
                "score": final_score,
            })
        
        return sorted(scored_matches, key=lambda x: x["score"])[:top_k]
    
    def _assign_tag(self, question: str, entity: str, linked_entities: List[Dict]) -> Optional[Dict]:
        """
        Assign refined semantic tag based on schema context
        Returns None if best_match_index = -1 (rejected)
        """
        formatted_entities = "\n".join([
            f"Index {i}:, Value: '{e['value']}', Table: {e['table']}, Column: {e['column']}"
            for i, e in enumerate(linked_entities)
        ])
        
        llm = self.generator.client.bind_tools([TagAssignment], tool_choice="TagAssignment", strict=True)
        
        messages = [
            ("system", self._get_prompt("tag_assignment")),
            ("user", f"# Question: {question}"),
            ("user", f"# Entity: '{entity}'"),
            ("user", f"# Proposed related entities:\n{formatted_entities}"),
        ]

        try: 
            response = llm.invoke(messages)
        except Exception as e:
            if is_content_filter_error(e) and self.fallback_generator:
                llm = self.fallback_generator.client.bind_tools([TagAssignment], tool_choice="TagAssignment", strict=True)
                response = llm.invoke(messages)
            else:
                raise
        
        tool_calls = getattr(response, "tool_calls", None)
        if tool_calls:
            try:
                args = tool_calls[0].get("args", {})
                best_match_idx = args.get("best_match_index", -1)

                # Check if rejected
                if best_match_idx == -1 or args.get("label") == "NO_MATCH":
                    return None 
                
                # Valid match
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
        
        # Fallback: rejection (parsing failed or invalid response)
        return None
    
    def _mask_question(self, question: str, entities: List[Dict]) -> str:
        """Replace entity values with tags"""
        masked = question
        for entity in sorted(entities, key=lambda x: len(x["original"]), reverse=True):
            masked = masked.replace(entity["original"], f"[{entity['label']}]")
        return masked
    
    def _construct_and_fill_sql(
        self, 
        masked_question: str, 
        original_question: str, 
        entities: List[Dict]
    ) -> Tuple[str | None, List]:
        """Generate SQL directly from masked question with entity mappings in ONE call"""
        
        if not self.config.template_construction:
            masked_question = original_question

        top_k_multiplier = 5 if self.config.doc_reranking and \
            not self.config.prompt_extension else 1

        retrieved_cases: List[Document] = self._cbr_retrieve(
            masked_question, original_question,
            top_k=self.config.top_k * top_k_multiplier,
        )

        if self.config.prompt_extension:
            subquestions = self._decompose_prompt(masked_question)
            # print(f"Decomposed into {len(subquestions)} sub-questions:", subquestions)

            all_retrieved = list(retrieved_cases)

            for subq in subquestions:
                # Sub-questions are in masked form — dense only
                sub_cases = self.retriever.retrieve(subq, top_k=self.config.top_k, mode="dense")
                all_retrieved.extend(sub_cases)
            
            seen_content = set()
            unique_cases = []
            for doc in all_retrieved:
                if doc.page_content not in seen_content:
                    seen_content.add(doc.page_content)
                    unique_cases.append(doc)
            
            if self.config.doc_reranking:
                retrieved_cases = self._rerank_documents(
                    query=masked_question,
                    documents=unique_cases,
                    top_k=self.config.top_k
                )
            else:
                retrieved_cases = unique_cases[:self.config.top_k]
            # print(f"After re-ranking: kept top {len(retrieved_cases)} documents")

            if self.config.brittle_retrieval:
                retrieved_cases = drop_cases(retrieved_cases, len(retrieved_cases))
            
            formatted_examples = "\n---\n".join(
                f"* Example question: {doc.metadata['case']}\n"
                f"* Masked question form: {doc.page_content}\n"
                f"* Example SQL query: {doc.metadata.get('sql_query', 'Unanswerable question!')}" 
                for doc in retrieved_cases 
                if doc.metadata.get('case') and doc.metadata.get('sql_query')
            )
        else:
            if self.config.doc_reranking:
                retrieved_cases = self._rerank_documents(
                    query=masked_question,
                    documents=retrieved_cases,
                    top_k=self.config.top_k
                )
            
            if self.config.brittle_retrieval:
                retrieved_cases = drop_cases(retrieved_cases, len(retrieved_cases))

            formatted_examples = "\n---\n".join(
                f"* Example question: {doc.metadata['case']}\n"
                f"* Masked question form: {doc.page_content}\n"
                f"* Example SQL query: {doc.metadata.get('sql_query', 'Unanswerable question!')}" 
                for doc in retrieved_cases 
                if doc.metadata.get('case') and doc.metadata.get('sql_query')
            )
        
        # Format entity mappings
        entity_info = "\n---\n".join([
            f"[{e['label']}] -> '{e['best_match']}'" + 
            (f" (Table: {e.get('table')}, Column: {e.get('column')})" if e.get('table') else "")
            for e in entities
            if e.get('table') and e.get('column')
        ]) if entities else "No entities to replace"

        # # Bind tool call structure 
        # if self.config.dataset == "ehrsql":
        #     llm = self.generator.client.bind_tools([SqlGenerationOptional], tool_choice="SqlGenerationOptional", strict=True)
        # elif self.config.dataset == "ehrsql24":
        #     llm = self.generator.client.bind_tools([SqlGenerationOptional], tool_choice="SqlGenerationOptional", strict=True)
        # else:
        #     llm = self.generator.client.bind_tools([SqlGeneration], tool_choice="SqlGeneration", strict=True)

        if self.config.template_construction:
            messages = [
                ("system", self._get_prompt("sql_generation")),
                ("user", f"# Original Question: {original_question}"),
                ("user", f"# Masked Question (with entity spans extracted): {masked_question}"),
                ("user", f"# Retrieved Examples:\n\n{formatted_examples}"),
                ("user", f"# Entity Mappings:\n\n{entity_info}"),
                ("user", f"# Schema:\n\n{self.sql_db.get_table_info()}"),
                ("user", "# Adapted SQL query:"),
            ]
        else:
            messages = [
                ("system", self._get_prompt("sql_generation")),
                ("user", f"# Original Question: {original_question}"),
                ("user", f"# Retrieved Examples:\n\n{formatted_examples}"),
                ("user", f"# Entity Mappings:\n\n{entity_info}"),
                ("user", f"# Schema:\n\n{self.sql_db.get_table_info()}"),
                ("user", "# Adapted SQL query:"),
            ]

        try: 
            sql_query = self.generator.generate(messages) 
            # response = llm.invoke(messages)
            # print(response)
            # sql_query = response.tool_calls[0]["args"]["sql_query"]
        except Exception as e:
            if is_content_filter_error(e) and self.fallback_generator:
                sql_query = self.fallback_generator.generate(messages)
            else:
                raise e

        return remove_sql_wrapper(sql_query), retrieved_cases

    def _cbr_retrieve(
        self, masked_question: str, original_question: str, top_k: int
    ) -> List[Document]:
        """CBR retrieval: dense over masked CBR collection; if hybrid, fuse with sparse
        over the RAG collection (original questions) using the original question."""
        dense = self.retriever.retrieve(masked_question, top_k=top_k, mode="dense")
        if self.config.hybrid_retrieval:
            if self.rag_retriever is None:
                raise ValueError("hybrid_retrieval=True requires rag_retriever to be set in CBRtoSQL")
            sparse = self.rag_retriever.retrieve(original_question, top_k=top_k, mode="sparse")
            return rrf_fuse(dense, sparse, top_k=top_k)
        return dense

    def _decompose_prompt(self, question: str) -> List[str]:
        """Decompose question into sub-questions (excluding original)"""
        llm = self.generator.client.bind_tools([PromptExtension], tool_choice="PromptExtension", strict=True)
        
        messages = [
            ("system", self._get_prompt("prompt_extension")),
            ("user", f"# Question: {question}"),
        ]

        try: 
            response = llm.invoke(messages)
        except Exception as e:
            if is_content_filter_error(e) and self.fallback_generator:
                llm = self.fallback_generator.client.bind_tools(
                    [PromptExtension], 
                    tool_choice="PromptExtension", 
                    strict=True
                )
                response = llm.invoke(messages)
            else:
                raise
        
        tool_calls = getattr(response, "tool_calls", None)
        if tool_calls:
            try:
                args = tool_calls[0].get("args", {})
                decomposed_prompts = args.get("decomposed_prompts", [])
                return decomposed_prompts
            except (KeyError, IndexError, TypeError):
                pass
        
        return []