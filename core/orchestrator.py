# offline_chat_bot/core/orchestrator.py
import uuid 
import os      
import shutil  
import time    
import gc      

# Conditional imports
if __name__ == '__main__' and __package__ is None:
    import sys
    project_root_for_direct_run = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Renamed variable
    if project_root_for_direct_run not in sys.path:
        sys.path.insert(0, project_root_for_direct_run)
    from mmu.mmu_manager import MemoryManagementUnit
    from core.llm_service_wrapper import LLMServiceWrapper
    from core.agents import KnowledgeRetrieverAgent, SummarizationAgent, FactExtractionAgent
    from core.tokenizer_utils import count_tokens 
    from core.config_loader import get_config, get_project_root # <--- ENSURE get_project_root IS IMPORTED HERE
    from chromadb import Documents, EmbeddingFunction, Embeddings 

else: # When imported as part of a package
    from mmu.mmu_manager import MemoryManagementUnit
    from .llm_service_wrapper import LLMServiceWrapper
    from .agents import KnowledgeRetrieverAgent, SummarizationAgent, FactExtractionAgent
    from .tokenizer_utils import count_tokens
    from .config_loader import get_config, get_project_root # <--- AND HERE

class ConversationOrchestrator:
    def __init__(self, 
                 mmu: MemoryManagementUnit, 
                 lsw: LLMServiceWrapper,
                 knowledge_retriever: KnowledgeRetrieverAgent,
                 summarizer: SummarizationAgent,
                 fact_extractor: FactExtractionAgent,
                 config: dict = None): # Accept optional config
        
        if config is None:
            # print("CO: Loading global configuration...") # Optional print
            config = get_config()
        # else:
            # print("CO: Using provided configuration dictionary.") # Optional print

        expected_args = [
            (mmu, MemoryManagementUnit), (lsw, LLMServiceWrapper),
            (knowledge_retriever, KnowledgeRetrieverAgent),
            (summarizer, SummarizationAgent), (fact_extractor, FactExtractionAgent) 
        ]
        if not all(isinstance(arg, expected_type) for arg, expected_type in expected_args):
            raise TypeError("Invalid type for one or more arguments during CO initialization.")

        self.mmu = mmu
        self.lsw = lsw
        self.knowledge_retriever = knowledge_retriever
        self.summarizer = summarizer
        self.fact_extractor = fact_extractor
        
        co_config = config.get('orchestrator', {})
        lsw_config = config.get('lsw', {}) # For LSW default chat model as fallback

        # Use CO's specific default model if defined in config, else LSW's default from its own config loading
        self.default_llm_model = co_config.get('default_llm_model', lsw.default_chat_model)
        
        self.target_max_prompt_tokens = co_config.get('target_max_prompt_tokens', 7000)
        self.default_top_k_vector_search = co_config.get('default_top_k_vector_search', 3)
        self.min_tokens_for_vs_learn = co_config.get('min_tokens_for_user_statement_to_vector_store', 10)
        self.max_tokens_ltm_chunk = co_config.get('max_tokens_per_ltm_vector_chunk', 500)
        self.target_summary_tokens_ltm = co_config.get('target_summary_tokens_for_ltm_chunk', 150)

        # --- Load New STM Management Settings ---
        self.manage_stm_within_prompt = co_config.get('manage_stm_within_prompt', True) # Default to True if not in config
        self.stm_condensation_strategy = co_config.get('stm_condensation_strategy', "truncate") # Default to "truncate"
        self.target_stm_tokens_budget_ratio = co_config.get('target_stm_tokens_budget_ratio', 0.5) # Default to 0.5 (50%)
        self.stm_summary_target_tokens = co_config.get('stm_summary_target_tokens', 200) # Default to 200
        # --- End Loading New STM Management Settings ---

        fe_cfg = co_config.get('fact_extraction', {})
        self.fe_min_meaningful_length = fe_cfg.get('min_meaningful_length_for_filter', 3)
        self.fe_common_fillers = fe_cfg.get('common_fillers_for_filter', [])
        self.fe_question_indicators_check = fe_cfg.get('question_indicators_for_filter', [])
        
        self.active_conversation_id = None
        
        print("ConversationOrchestrator initialized (using configuration).")
        print(f"  CO - Default LLM model for responses: {self.default_llm_model}")
        print(f"  CO - Target max prompt tokens: {self.target_max_prompt_tokens}")
        print(f"  CO - Min tokens for VS learn: {self.min_tokens_for_vs_learn}")
        # --- Add print statements for new configs for verification ---
        print(f"  CO - Manage STM in prompt: {self.manage_stm_within_prompt}")
        print(f"  CO - STM Condensation Strategy: {self.stm_condensation_strategy}")
        print(f"  CO - STM Token Budget Ratio: {self.target_stm_tokens_budget_ratio}")
        print(f"  CO - STM Summary Target Tokens: {self.stm_summary_target_tokens}")
        # --- End print statements ---

    def _build_prompt_with_context(self,
                                   conversation_id: str, # No longer needed as param if self.active_conversation_id is used
                                   user_query: str,
                                   retrieved_knowledge: dict
                                  ) -> list:
        #print("  CO: Building prompt with context...")
        target_ollama_model_for_tokens = self.default_llm_model 
        summarization_llm_model = self.summarizer.default_model_name

        # 1. System Prompt
        system_message_content = (
            "You are a helpful and knowledgeable AI assistant. " 
            "Your primary goal is to answer the user's questions accurately and coherently. "
            "You have access to your short-term conversation memory and relevant information retrieved from your long-term knowledge base. "
            "Prioritize using the 'RETRIEVED KNOWLEDGE CONTEXT' section to answer if it's relevant. "
            "If the retrieved context does not contain enough information, use your general knowledge but clearly state that the information is not from your specific knowledge base. "
            "If you cannot answer the question based on either retrieved context or general knowledge, say so clearly. "
            "Do not invent facts or information. Be concise unless asked for detail. "
            "If you use information from the 'RETRIEVED KNOWLEDGE CONTEXT', try to subtly indicate that you are recalling specific information (e.g., 'Based on what we discussed earlier about X...' or 'I found some information relating to Y...')."
        )
        system_tokens = count_tokens(system_message_content, target_ollama_model_for_tokens)
        current_total_tokens = system_tokens
        print(f"    Tokens - System message: {system_tokens}")
        messages = [{"role": "system", "content": system_message_content}]
        knowledge_context_str_parts = []
        
        # SKB Facts
        if retrieved_knowledge.get("skb_fact_results"):
            added_skb_title = False
            for fact_idx, fact in enumerate(retrieved_knowledge["skb_fact_results"][:3]): 
                fact_str = f"Fact: {fact.get('subject')} {fact.get('predicate')} {fact.get('object')}."
                fact_tokens = count_tokens(fact_str, target_ollama_model_for_tokens)
                title_tokens = 0
                if not added_skb_title:
                    title = "Retrieved Facts from Knowledge Base:"
                    title_tokens = count_tokens(title, target_ollama_model_for_tokens)
                if current_total_tokens + title_tokens + fact_tokens < self.target_max_prompt_tokens: # USE SELF.
                    if not added_skb_title:
                         knowledge_context_str_parts.append(title); current_total_tokens += title_tokens; added_skb_title = True
                    knowledge_context_str_parts.append(f"  - {fact_str}"); current_total_tokens += fact_tokens
                else: break 
        
        # Vector Store results
        if retrieved_knowledge.get("vector_results"):
            added_vs_title = False
            # Use self.default_top_k_vector_search if results are not pre-sliced
            for res_idx, res in enumerate(retrieved_knowledge["vector_results"][:self.default_top_k_vector_search]): # USE SELF.
                chunk_text_to_use = res.get('text_chunk', '')
                original_chunk_tokens = count_tokens(chunk_text_to_use, target_ollama_model_for_tokens)
                if original_chunk_tokens > self.max_tokens_ltm_chunk: # USE SELF.
                    summary = self.summarizer.summarize_text(
                        text_to_summarize=chunk_text_to_use, model_name=summarization_llm_model,
                        max_summary_length=self.target_summary_tokens_ltm # USE SELF.
                    )
                    if summary: chunk_text_to_use = f"[Summary of a longer document]: {summary}"
                res_str = f"From source (metadata: {res.get('metadata', {})}): \"{chunk_text_to_use}\""
                res_tokens = count_tokens(res_str, target_ollama_model_for_tokens)
                title_tokens = 0
                if not added_vs_title:
                    title = "\nRetrieved Similar Information from Past Interactions/Documents:"
                    title_tokens = count_tokens(title, target_ollama_model_for_tokens)
                if current_total_tokens + title_tokens + res_tokens < self.target_max_prompt_tokens: # USE SELF.
                    if not added_vs_title:
                        knowledge_context_str_parts.append(title); current_total_tokens += title_tokens; added_vs_title = True
                    knowledge_context_str_parts.append(f"  - {res_str}"); current_total_tokens += res_tokens
                else: break
        retrieved_knowledge_prompt_str = "\n".join(knowledge_context_str_parts)
        print(f"    Tokens - System + LTM context: {current_total_tokens}")

        # --- New STM Management Logic ---
        stm_history_turns_original = self.mmu.get_stm_history() # Get all current STM turns

        # The user_query for the *current* turn is NOT yet in stm_history_turns_original
        # because we add it to STM *after* the LLM call in handle_user_message.
        # So, stm_history_turns_original represents history *before* the current user_query.

        managed_stm_turns_for_prompt = [] # This will hold the STM turns we decide to include

        if self.manage_stm_within_prompt and stm_history_turns_original:
            user_query_tokens = count_tokens(user_query, target_ollama_model_for_tokens)
            
            # Calculate remaining budget *before* considering STM or the current user_query
            remaining_budget_for_stm_and_query = self.target_max_prompt_tokens - current_total_tokens
            
            # Calculate the specific budget for STM based on the ratio
            stm_token_budget = int(remaining_budget_for_stm_and_query * self.target_stm_tokens_budget_ratio)
            print(f"    Tokens - Budget available for STM & User Query: {remaining_budget_for_stm_and_query}")
            print(f"    Tokens - Calculated STM Token Budget (ratio {self.target_stm_tokens_budget_ratio}): {stm_token_budget}")

            if self.stm_condensation_strategy == "truncate":
                current_stm_tokens_in_budget = 0
                # Iterate from newest to oldest STM turns (reversed) to prioritize recent history
                truncated_older_turn = False
                for turn in reversed(stm_history_turns_original):
                    turn_str_temp = f"{turn['role'].capitalize()}: {turn['content']}" # For token counting
                    turn_tokens_temp = count_tokens(turn_str_temp, target_ollama_model_for_tokens)
                    
                    if current_stm_tokens_in_budget + turn_tokens_temp <= stm_token_budget:
                        managed_stm_turns_for_prompt.insert(0, turn) # Add to beginning to maintain order
                        current_stm_tokens_in_budget += turn_tokens_temp
                    else:
                        # This turn_tokens_temp is the one that didn't fit
                        print(f"    STM Truncation: Oldest considered STM turn ({turn_tokens_temp} tokens) for '{turn_str_temp[:30]}...' did not fit remaining STM budget ({stm_token_budget - current_stm_tokens_in_budget} left).")
                        truncated_older_turn = True
                        break 
                                # This print might be more informative
                if not stm_history_turns_original:
                    print(f"    Tokens - STM: No prior STM turns to consider for this prompt.")
                elif not truncated_older_turn and stm_history_turns_original:
                    print(f"    Tokens - STM: All {len(stm_history_turns_original)} STM turns fit within budget.")
                print(f"    Tokens - STM after truncation: {current_stm_tokens_in_budget} from {len(managed_stm_turns_for_prompt)} turns.")
            
            elif self.stm_condensation_strategy == "summarize":
                # Placeholder for summarization logic (to be implemented in a later step)
                print(f"    STM Condensation: Summarization strategy selected but not yet implemented. Using original STM for now.")
                managed_stm_turns_for_prompt = list(stm_history_turns_original) # Fallback for now
            
            else: # Unknown strategy or feature disabled implicitly
                managed_stm_turns_for_prompt = list(stm_history_turns_original)

        else: # STM management disabled or no STM history
            managed_stm_turns_for_prompt = list(stm_history_turns_original)
        # --- End New STM Management Logic ---

        # Now, build the stm_prompt_parts string from managed_stm_turns_for_prompt
        # And ensure the *total* prompt doesn't exceed target_max_prompt_tokens
        stm_prompt_parts = []
        # `current_total_tokens` here is still system_msg + LTM_context.
        # We need to add the tokens of the chosen STM turns to it.

        actual_stm_tokens_added = 0
        user_query_tokens = count_tokens(user_query, target_ollama_model_for_tokens) # Recalculate for safety or pass from above

        for turn in managed_stm_turns_for_prompt: # Iterate through the (potentially truncated) STM turns
            turn_str = f"{turn['role'].capitalize()}: {turn['content']}"
            turn_tokens = count_tokens(turn_str, target_ollama_model_for_tokens)
            
            # Check against the *overall* prompt budget
            if current_total_tokens + actual_stm_tokens_added + turn_tokens + user_query_tokens < self.target_max_prompt_tokens:
                stm_prompt_parts.append(turn_str) # Append to maintain chronological order for final string
                actual_stm_tokens_added += turn_tokens
            else:
                # This is a final safety break if, even after STM budget management,
                # adding the next managed STM turn would exceed the *total* prompt limit.
                print(f"    Tokens - STM turn skipped (FINAL BUDGET CHECK): {turn_tokens} for '{turn_str[:30]}...'")
                break
                
        current_total_tokens += actual_stm_tokens_added # Update total tokens with actual STM used
        stm_history_prompt_str = "\n".join(stm_prompt_parts)
        print(f"    Tokens - Actual STM history added to prompt: {actual_stm_tokens_added}")

        # Final Assembly (this part remains largely the same, but uses the new stm_history_prompt_str)
        full_user_prompt_content_parts = []
        if stm_history_prompt_str: # Only add if there's STM content
            full_user_prompt_content_parts.append("Relevant Conversation History:\n")
            full_user_prompt_content_parts.append(stm_history_prompt_str)

        if retrieved_knowledge_prompt_str:
            if full_user_prompt_content_parts: 
                full_user_prompt_content_parts.append("\n") # Add a newline if STM was also present
            full_user_prompt_content_parts.append("RETRIEVED KNOWLEDGE CONTEXT:\n")
            full_user_prompt_content_parts.append(retrieved_knowledge_prompt_str)

        if full_user_prompt_content_parts: # If either STM or LTM context was added
            full_user_prompt_content_parts.append("\n\n") # Add spacing before user query
            
        full_user_prompt_content_parts.append(f"User Query: {user_query}")
        final_user_content = "".join(full_user_prompt_content_parts)

        # The system message is added separately when constructing the `messages` list for the LLM
        # So, the token count for `final_user_content` is just for the user role part.
        # final_user_content_tokens = count_tokens(final_user_content, target_ollama_model_for_tokens)

        # `current_total_tokens` now includes system, LTM, and chosen STM. Add user_query for final check.
        final_prompt_total_tokens = current_total_tokens + user_query_tokens

        print(f"  CO: Final total prompt tokens: {final_prompt_total_tokens} (Target: {self.target_max_prompt_tokens})")
        if final_prompt_total_tokens > self.target_max_prompt_tokens:
            print(f"  CO: WARNING - Final prompt tokens ({final_prompt_total_tokens}) exceeded target AFTER STM management.")
            # Potentially, we could truncate final_user_content here as a last resort,
            # but it's better if the budgeting prevents this.
            
        messages.append({"role": "user", "content": final_user_content})
        return messages

    def handle_user_message(self, user_message: str, conversation_id: str = None) -> iter:
        """
        Main handler for processing a user's message.
        Yields response chunks for streaming.
        """
        if not user_message.strip(): yield "Please say something!"; return
        
        is_new_conversation_session = False
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            is_new_conversation_session = True
        elif self.active_conversation_id != conversation_id:
            is_new_conversation_session = True
        
        if is_new_conversation_session:
            print(f"CO: {'New conversation started' if not self.active_conversation_id else 'Switched to/started new conversation ID'}: {conversation_id} (was {self.active_conversation_id})")
            self.mmu.clear_stm()
            print(f"  CO: STM cleared for new conversation session '{conversation_id}'.")
        
        self.active_conversation_id = conversation_id 
        #print(f"\nCO: Handling user message for conversation '{conversation_id}': '{user_message}'")
        
        # Use self.fe_question_indicators_check from config
        is_likely_question = user_message.endswith("?") or \
                             any(user_message.lower().startswith(q_word) for q_word in self.fe_question_indicators_check)
        if is_likely_question: print(f"  CO_INFO: User message detected as a likely question.")
        self.mmu.add_stm_turn(role="user", content=user_message)

        print(f"  CO: Attempting to extract facts from user message: \"{user_message[:100]}...\"")
        extracted_facts_from_user = self.fact_extractor.extract_facts(text_to_process=user_message)
        
        if extracted_facts_from_user: # Facts were potentially extracted by agent
            #print(f"  CO: Raw extracted {len(extracted_facts_from_user)} fact(s) from user message by FactExtractionAgent.")
            filtered_facts_to_store = []
            if is_likely_question: 
                print(f"    CO_FILTER: User message appears to be a question. Discarding for SKB.")
            else:
                for fact in extracted_facts_from_user:
                    s = str(fact.get('subject', '')).strip(); p = str(fact.get('predicate', '')).strip(); o = str(fact.get('object', '')).strip()
                    if not s or not p or not o: continue
                    if s.lower() in self.fe_common_fillers and o.lower() in self.fe_common_fillers and \
                       len(s) <= self.fe_min_meaningful_length and len(o) <= self.fe_min_meaningful_length: continue
                    if (s.lower().startswith("my ") or s.lower().startswith("user's ")) and \
                       (p.lower() in self.fe_question_indicators_check or p.lower() == "what"): continue
                    if s.lower() in ["my", "i"] and p.lower() in self.fe_common_fillers and o.lower() in self.fe_common_fillers: continue
                    filtered_facts_to_store.append({"subject": s, "predicate": p, "object": o}) 
            
            if filtered_facts_to_store: # This block only runs if not a question AND facts survived filtering
                print(f"  CO: Storing {len(filtered_facts_to_store)} filtered fact(s) to LTM/SKB.")
                for fact_to_store in filtered_facts_to_store:
                     #print(f"    CO: Storing fact: S='{fact_to_store['subject']}', P='{fact_to_store['predicate']}', O='{fact_to_store['object']}'")
                     self.mmu.store_ltm_fact(subject=fact_to_store['subject'],predicate=fact_to_store['predicate'],object_value=fact_to_store['object'],confidence=0.80)

        # KRA Search (uses self.default_top_k_vector_search from config)
        retrieved_knowledge = self.knowledge_retriever.search_knowledge(
            query_text=user_message, top_k_vector=self.default_top_k_vector_search, search_skb_facts=True
        )

        # Build Prompt (uses self.active_conversation_id, other self. attributes for limits)
        llm_messages = self._build_prompt_with_context(
            conversation_id=self.active_conversation_id, # Pass current active ID
            user_query=user_message, retrieved_knowledge=retrieved_knowledge
        )
        
        # LSW Call (uses self.default_llm_model from config)
        assistant_response_stream = self.lsw.generate_chat_completion(
            messages=llm_messages, model_name=self.default_llm_model, temperature=0.5, stream=True
        )

        if assistant_response_stream is None: # Should be an empty iterator from LSW on error
            print("  CO: LSW returned None for stream (should be empty iterator). Yielding error message.")
            error_message = "I'm sorry, I encountered an issue and couldn't generate a response right now."
            yield error_message
            # Log this failure to LTM (User turn logged later, then this error as assistant turn)
            # This logging needs to happen *after* all chunks are processed or if stream fails.
            # For simplicity now, if stream itself fails to start, we log just this error.
            # A more robust way is to collect full response then log.
            # We'll log after the loop for successful streams.
            # This path is for LSW failing to even return a stream.
            _current_ltm_hist = self.mmu.get_ltm_conversation_history(self.active_conversation_id)
            _user_turn_seq_id = len(_current_ltm_hist) + 1
            self.mmu.log_ltm_interaction(self.active_conversation_id, _user_turn_seq_id, "user", user_message)
            self.mmu.log_ltm_interaction(self.active_conversation_id, _user_turn_seq_id + 1, "assistant_error", error_message, metadata={"error": "LSW stream init failed"})
            self.mmu.add_stm_turn(role="assistant", content=error_message)
            return # Stop generation

        accumulated_response_for_ltm = ""
        has_yielded_content = False
        try:
            for chunk in assistant_response_stream:
                if chunk: # Ensure chunk is not empty
                    # print(f"CO yielding chunk: '{chunk}'") # Debug: very verbose
                    yield chunk
                    accumulated_response_for_ltm += chunk
                    has_yielded_content = True
            
            if not has_yielded_content and not accumulated_response_for_ltm: # Stream was empty
                print("  CO: LLM stream was empty.")
                fallback_message = "I couldn't come up with a response for that."
                yield fallback_message
                accumulated_response_for_ltm = fallback_message

        except Exception as e:
            print(f"  CO: Error consuming LLM stream: {e}")
            error_message = "Sorry, there was an error while I was generating my response."
            yield error_message # Yield error to user
            accumulated_response_for_ltm = error_message # Log this error
             
        #print(f"  CO: Full assistant response (accumulated): \"{accumulated_response_for_ltm[:100].strip()}...\"")
        self.mmu.add_stm_turn(role="assistant", content=accumulated_response_for_ltm)

        # LTM Logging
        print(f"  CO: Preparing to log to LTM for conversation '{self.active_conversation_id}'.")
        current_ltm_history_len = len(self.mmu.get_ltm_conversation_history(self.active_conversation_id))
        user_turn_seq_id = current_ltm_history_len + 1
        #print(f"    LTM logging: User turn seq ID: {user_turn_seq_id} for '{user_message[:30]}...'")
        user_turn_ltm_entry_id = self.mmu.log_ltm_interaction(self.active_conversation_id, user_turn_seq_id, "user", user_message)

        # Vector Store Learning (uses self.min_tokens_for_vs_learn)
        if user_message and not is_likely_question:
            user_message_tokens = count_tokens(user_message, self.default_llm_model)
            if user_message_tokens >= self.min_tokens_for_vs_learn:
                print(f"  CO: User message is significant ({user_message_tokens} tokens). Storing to Vector Store.")
                doc_id_vs = f"user_stmt_{self.active_conversation_id}_{user_turn_seq_id}" 
                metadata_vs = {
                    "type": "user_statement", "conversation_id": self.active_conversation_id,
                    "turn_sequence_id": user_turn_seq_id,
                    "ltm_raw_log_entry_id": user_turn_ltm_entry_id if user_turn_ltm_entry_id else "unknown",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) 
                }
                self.mmu.add_document_to_ltm_vector_store(text_chunk=user_message, metadata=metadata_vs, doc_id=f"user_stmt_{self.active_conversation_id}_{user_turn_seq_id}")
                vector_store_add_id = self.mmu.add_document_to_ltm_vector_store(
                    text_chunk=user_message, metadata=metadata_vs, doc_id=doc_id_vs
                )
                if vector_store_add_id: print(f"    CO: User statement stored in Vector Store with ID: {vector_store_add_id}")
                else: print(f"    CO: Failed to store user statement in Vector Store.")
            else:
                print(f"  CO: User message not stored in Vector Store (tokens: {user_message_tokens} < {self.min_tokens_for_vs_learn}).")
        # --- End Vector Store Learning for user message ---
        
        # Log Assistant Turn
        assistant_turn_log_seq_id = user_turn_seq_id + 1
        print(f"    LTM logging: Assistant turn seq ID: {assistant_turn_log_seq_id} for '{accumulated_response_for_ltm[:30]}...'")
        self.mmu.log_ltm_interaction(
            conversation_id=self.active_conversation_id, turn_sequence_id=assistant_turn_log_seq_id,
            role="assistant", content=accumulated_response_for_ltm,
            llm_model_used=self.default_llm_model,
            metadata={
                "retrieved_vector_ids": [res['id'] for res in retrieved_knowledge.get("vector_results",[]) if 'id' in res],
                "retrieved_fact_ids": [fact['fact_id'] for fact in retrieved_knowledge.get("skb_fact_results",[]) if 'fact_id' in fact],
                "llm_streamed": True # Add metadata that it was streamed
            }
        )

if __name__ == "__main__":
    print("--- Testing CO (Config-Driven) with Fact Extraction, VS Learning & Summarization ---")

    # Load global application config
    app_config = get_config()

    # Test paths (use new ones to ensure fresh LTM for this specific test)
    # These paths will be overridden by the test_specific_co_config
    test_base_dir = os.path.join(get_project_root(), "test_co_config_data")
    test_co_ltm_sqlite_db_path = os.path.join(test_base_dir, 'co_test_ltm.db')
    test_co_ltm_chroma_dir = os.path.join(test_base_dir, 'co_test_ltm_chroma')
    test_co_mtm_db_path = os.path.join(test_base_dir, 'co_test_mtm.json')

    # Create a specific test configuration dictionary to override default paths
    # and potentially other settings for focused testing.
    # This dictionary will be passed to the CO and its dependencies.
    test_specific_config_override = {
        "data_dir": os.path.relpath(test_base_dir, get_project_root()), # Base for relative paths
        "mmu": {
            "stm_max_turns": 5, 
            "mtm_use_tinydb": True, 
            "mtm_db_path": os.path.relpath(test_co_mtm_db_path, test_base_dir), # Path relative to data_dir
            "ltm_sqlite_db_path": os.path.relpath(test_co_ltm_sqlite_db_path, test_base_dir),
            "ltm_chroma_persist_directory": os.path.relpath(test_co_ltm_chroma_dir, test_base_dir)
        },
        "lsw": app_config.get("lsw", {}), # Use LSW settings from main config
        "tokenizer": app_config.get("tokenizer", {}), # Use tokenizer settings from main config
        "orchestrator": { # Override some orchestrator settings for testing
            "target_max_prompt_tokens": 600, # Smaller limit to test truncation/summarization
            "default_top_k_vector_search": 2,
            "min_tokens_for_user_statement_to_vector_store": 5,
            "max_tokens_per_ltm_vector_chunk": 100, # Force summarization
            "target_summary_tokens_for_ltm_chunk": 30,
            "fact_extraction": app_config.get("orchestrator", {}).get("fact_extraction", {}) # Use main config for filters
        },
        "agents": app_config.get("agents", {}) # Use agent model settings from main config
    }
    # Ensure test_base_dir exists
    os.makedirs(test_base_dir, exist_ok=True)

    if os.path.exists(test_co_mtm_db_path): os.remove(test_co_mtm_db_path)
    if os.path.exists(test_co_ltm_sqlite_db_path): os.remove(test_co_ltm_sqlite_db_path)
    if os.path.exists(test_co_ltm_chroma_dir): shutil.rmtree(test_co_ltm_chroma_dir)
    os.makedirs(test_co_ltm_chroma_dir, exist_ok=True) # Recreate Chroma dir

    test_mmu = None; test_lsw = None; knowledge_retriever = None; 
    summarizer = None; fact_extractor = None; orchestrator = None
    
    # --- ADD CHROMA TYPES IMPORT FOR THE LSW WRAPPER FUNCTION CLASS ---
    # This ensures EmbeddingFunction, Documents, Embeddings are defined when the class is parsed.
    try:
        from chromadb import Documents, EmbeddingFunction, Embeddings
    except ImportError:
        print("CRITICAL: chromadb library not found. Cannot run CO tests that use embedding functions.")
        print("Please install it: pip install chromadb")
        # Define dummy classes so the rest of the script doesn't immediately crash on NameError,
        # though tests involving embeddings will effectively be skipped or fail differently.
        class EmbeddingFunction: pass
        class Documents(list): pass
        class Embeddings(list): pass
        # This isn't ideal, but allows script to proceed further to show other init messages if Chroma is missing.
    # --- END ADDITION ---

    # --- Mock Embedding Function for MMU direct init in this test ---
    # (This is only if MMU init still needs one directly for some reason, but it should get it from LSW via main.py)
    # MMU's __init__ now expects an embedding_function. We need to provide one.
    # LSW should be initialized first.
    try:
        print("\nInitializing LSW for CO test (using test_specific_config_override)...")
        test_lsw = LLMServiceWrapper(config=test_specific_config_override) 
        if not test_lsw.client: raise Exception("Ollama client in LSW failed.")

        # --- NEW: Define an EmbeddingFunction class that uses our LSW ---
        class LSWEmbeddingFunctionForChroma(EmbeddingFunction): # Now EmbeddingFunction is defined
            def __init__(self, lsw_instance: LLMServiceWrapper, embedding_source="st"):
                # ... (rest of LSWEmbeddingFunctionForChroma as defined in the previous step) ...
                self.lsw = lsw_instance
                self.source = embedding_source 
                self.model_name = None
                if self.source == "st":
                    self.model_name = self.lsw.default_embedding_model_st
                    if "all-MiniLM-L6-v2" in self.model_name: self.dim = 384
                    else: self.dim = 768 
                elif self.source == "ollama":
                    self.model_name = self.lsw.default_embedding_model_ollama
                    if "nomic-embed-text" in self.model_name: self.dim = 768 
                    else: self.dim = 768 
                else:
                    raise ValueError("Unsupported embedding source")
                print(f"  LSWEmbeddingFunctionForChroma: Initialized for source '{self.source}', model '{self.model_name}', dim {self.dim}")

            def __call__(self, input: Documents) -> Embeddings: # Now Documents & Embeddings are defined
                embeddings_list: Embeddings = [] 
                if not input: return embeddings_list
                print(f"  LSWEmbeddingFunctionForChroma: __call__ received {len(input)} document(s). Using LSW source '{self.source}'.")
                for text_doc in input:
                    emb = self.lsw.generate_embedding(text_to_embed=str(text_doc), source=self.source, model_name=self.model_name) 
                    if emb: embeddings_list.append(emb)
                    else:
                        print(f"    WARNING: Embedding failed via LSW for text: {str(text_doc)[:50]}... Appending dummy.")
                        embeddings_list.append([0.0] * self.dim) 
                return embeddings_list
        # --- End LSWEmbeddingFunctionForChroma class ---

        # Instantiate the wrapper
        lsw_embedder = LSWEmbeddingFunctionForChroma(lsw_instance=test_lsw, embedding_source="st")

        print("\nInitializing MMU for CO test (using test_specific_config_override)...")
        test_mmu = MemoryManagementUnit(
            embedding_function=lsw_embedder, # <--- PASS THE INSTANCE OF THE WRAPPER CLASS
            config=test_specific_config_override
        )
        
        print("\nInitializing Agents for CO test...")
        agents_cfg = test_specific_config_override.get('agents', {})
        lsw_cfg_for_agents = test_specific_config_override.get('lsw', {})
        summarizer_model = agents_cfg.get('summarization_agent_model', lsw_cfg_for_agents.get('default_chat_model'))
        fact_extractor_model = agents_cfg.get('fact_extraction_agent_model', lsw_cfg_for_agents.get('default_chat_model'))
        knowledge_retriever = KnowledgeRetrieverAgent(mmu=test_mmu)
        summarizer = SummarizationAgent(lsw=test_lsw, default_model_name=summarizer_model)
        fact_extractor = FactExtractionAgent(lsw=test_lsw, default_model_name=fact_extractor_model)
        print("\nInitializing ConversationOrchestrator (using test_specific_config_override)...")
        orchestrator = ConversationOrchestrator(
            mmu=test_mmu, lsw=test_lsw,
            knowledge_retriever=knowledge_retriever,
            summarizer=summarizer, fact_extractor=fact_extractor,
            config=test_specific_config_override
        )

        # Test VS 1, VS 2, VS 3 (these are orchestrator tests, not just MMU tests)
        vs_learn_convo_id = "test_co_config_learn_001"
        print(f"\n--- Test CO_CONFIG 1: User makes a significant statement ---")
        user_statement_for_vs = "The annual Quantum Computing Expo will be in Geneva and Dr. Eva Rostova is the keynote."
        # Consume the generator for testing purposes
        response_gen1 = orchestrator.handle_user_message(user_statement_for_vs, vs_learn_convo_id)
        response_vs1_str = "".join(list(response_gen1))
        print(f"Chatbot Response CO_CONFIG_1: {response_vs1_str}")

        print(f"\n--- Test CO_CONFIG 2: User asks a semantically similar question (Vector Store recall) ---")
        user_question_vs = "Tell me about the quantum conference in Geneva featuring Dr. Rostova."
        response_gen2 = orchestrator.handle_user_message(user_question_vs, vs_learn_convo_id )
        response_vs2_str = "".join(list(response_gen2))
        print(f"Chatbot Response CO_CONFIG_2: {response_vs2_str}")

        print("\nCO config-driven tests finished.")

    except Exception as e:
        print(f"FATAL: Error during CO test environment setup or execution: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
    
    finally: # Ensure cleanup runs even if tests error out
        # ... (Final cleanup logic as before) ...
        print("\nAttempting final cleanup of CO config test files...")
        # ... (del orchestrator etc.) ...
        # ... (os.remove and shutil.rmtree for test_co_config_data paths) ...
        if 'orchestrator' in locals() and orchestrator is not None: del orchestrator
        if 'knowledge_retriever' in locals() and knowledge_retriever is not None: del knowledge_retriever
        if 'summarizer' in locals() and summarizer is not None: del summarizer
        if 'fact_extractor' in locals() and fact_extractor is not None: del fact_extractor
        if 'test_lsw' in locals() and test_lsw is not None: del test_lsw
        if 'test_mmu' in locals() and test_mmu is not None:
            if hasattr(test_mmu.mtm, 'is_persistent') and test_mmu.mtm.is_persistent and hasattr(test_mmu.mtm, 'db') and test_mmu.mtm.db:
                test_mmu.mtm.db.close()
            del test_mmu
        if 'gc' in globals(): gc.collect(); time.sleep(0.1) # Check if gc was imported
        
        if os.path.exists(test_co_mtm_db_path): 
            try: os.remove(test_co_mtm_db_path)
            except Exception as e_clean: print(f"  Cleanup Error: {test_co_mtm_db_path}: {e_clean}")
        if os.path.exists(test_co_ltm_sqlite_db_path): 
            try: os.remove(test_co_ltm_sqlite_db_path)
            except Exception as e_clean: print(f"  Cleanup Error: {test_co_ltm_sqlite_db_path}: {e_clean}")
        if os.path.exists(test_co_ltm_chroma_dir): 
            try: shutil.rmtree(test_co_ltm_chroma_dir, ignore_errors=False)
            except Exception as e_clean: print(f"  Cleanup Error removing dir {test_co_ltm_chroma_dir}: {e_clean}")
        print("CO config test file cleanup attempt finished.")