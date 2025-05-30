# offline_chat_bot/core/orchestrator.py
import uuid # For generating conversation IDs if not provided
import os
import shutil
import time
import gc 

# Conditional imports
if __name__ == '__main__' and __package__ is None:
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from mmu.mmu_manager import MemoryManagementUnit
    from core.llm_service_wrapper import LLMServiceWrapper
    from core.agents import KnowledgeRetrieverAgent, SummarizationAgent, FactExtractionAgent
    from core.tokenizer_utils import count_tokens # <--- ADD THIS IMPORT
else:
    from mmu.mmu_manager import MemoryManagementUnit
    from .llm_service_wrapper import LLMServiceWrapper
    from .agents import KnowledgeRetrieverAgent, SummarizationAgent, FactExtractionAgent
    from .tokenizer_utils import count_tokens # <--- ADD THIS IMPORT

# --- Constants ---
# DEFAULT_MAX_CONTEXT_TOKENS_APPROX = 3000 # Words approx // 5
# Replace with token-based limit. Assuming 8192 context for gemma3:4b-it-fp16 via Ollama.
# Target prompt tokens should be less to leave room for generation.
TARGET_MAX_PROMPT_TOKENS = 7000 # <--- NEW CONSTANT (Adjust if your assumed context window is different)
DEFAULT_TOP_K_VECTOR_SEARCH = 3
MIN_TOKENS_FOR_USER_STATEMENT_TO_VECTOR_STORE = 10 # User statements longer than this may be stored
# NEW CONSTANT for summarizing LTM chunks
MAX_TOKENS_PER_LTM_VECTOR_CHUNK = 500 # If a VS chunk exceeds this, try to summarize it
TARGET_SUMMARY_TOKENS_FOR_LTM_CHUNK = 150 # Target length for the summary of a long chunk
# --- End Constants ---

class ConversationOrchestrator:
    def __init__(self, 
                 mmu: MemoryManagementUnit, 
                 lsw: LLMServiceWrapper,
                 knowledge_retriever: KnowledgeRetrieverAgent,
                 summarizer: SummarizationAgent,
                 fact_extractor: FactExtractionAgent, # <--- ADD fact_extractor parameter
                 default_llm_model: str = None):
        
        # Updated type checks
        expected_args = [
            (mmu, MemoryManagementUnit), (lsw, LLMServiceWrapper),
            (knowledge_retriever, KnowledgeRetrieverAgent),
            (summarizer, SummarizationAgent),
            (fact_extractor, FactExtractionAgent) # <--- ADD type check
        ]
        if not all(isinstance(arg, expected_type) for arg, expected_type in expected_args):
            raise TypeError("Invalid type for one or more arguments during CO initialization.")

        self.mmu = mmu
        self.lsw = lsw
        self.knowledge_retriever = knowledge_retriever
        self.summarizer = summarizer
        self.fact_extractor = fact_extractor # <--- STORE fact_extractor instance
        self.default_llm_model = default_llm_model if default_llm_model else self.lsw.default_chat_model
        self.active_conversation_id = None
        
        print("ConversationOrchestrator initialized.")
        print(f"  Default LLM model for responses: {self.default_llm_model}")
        print(f"  Target max prompt tokens: {TARGET_MAX_PROMPT_TOKENS}")
        print(f"  Min tokens for user statement to be considered for Vector Store: {MIN_TOKENS_FOR_USER_STATEMENT_TO_VECTOR_STORE}")
        print(f"  Max tokens per LTM vector chunk (before summary): {MAX_TOKENS_PER_LTM_VECTOR_CHUNK}")
        print(f"  Target summary tokens for long LTM chunk: {TARGET_SUMMARY_TOKENS_FOR_LTM_CHUNK}")

    def _build_prompt_with_context(self,
                                   conversation_id: str,
                                   user_query: str,
                                   retrieved_knowledge: dict
                                  ) -> list: # Return type is list of messages
        """
        Constructs the full prompt to be sent to the LLM, incorporating context
        and managing total token count.
        """
        print("  CO: Building prompt with context...")
        # The LLM model this prompt is being built for (to pass to count_tokens)
        # For now, assume it's the CO's default_llm_model. If different models
        # are used for different tasks within CO, this would need to be dynamic.
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

        # 2. Retrieved Knowledge Context (LTM) - Prioritize this after system message
        # This section will try to add LTM results one by one until limit is approached.
        # More sophisticated ranking could be added here.
        knowledge_context_str_parts = []
        
        # Add SKB facts (assuming these are generally short)
        if retrieved_knowledge.get("skb_fact_results"):
            added_skb_title = False
            for fact_idx, fact in enumerate(retrieved_knowledge["skb_fact_results"][:3]): 
                fact_str = f"Fact: {fact.get('subject')} {fact.get('predicate')} {fact.get('object')}."
                fact_tokens = count_tokens(fact_str, target_ollama_model_for_tokens)
                
                title_tokens = 0
                if not added_skb_title:
                    title = "Retrieved Facts from Knowledge Base:"
                    title_tokens = count_tokens(title, target_ollama_model_for_tokens)

                if current_total_tokens + title_tokens + fact_tokens < TARGET_MAX_PROMPT_TOKENS:
                    if not added_skb_title:
                         knowledge_context_str_parts.append(title)
                         current_total_tokens += title_tokens
                         added_skb_title = True
                    knowledge_context_str_parts.append(f"  - {fact_str}")
                    current_total_tokens += fact_tokens
                    print(f"    Tokens - Added SKB Fact {fact_idx+1} ({fact_tokens} tokens). Cumulative: {current_total_tokens}")
                else:
                    print(f"    Tokens - SKIPPING SKB Fact {fact_idx+1} (would exceed limit).")
                    break 
        
        # Add Vector Store results, with summarization for long chunks
        if retrieved_knowledge.get("vector_results"):
            added_vs_title = False
            for res_idx, res in enumerate(retrieved_knowledge["vector_results"][:DEFAULT_TOP_K_VECTOR_SEARCH]):
                chunk_text_to_use = res.get('text_chunk', '')
                original_chunk_tokens = count_tokens(chunk_text_to_use, target_ollama_model_for_tokens)
                
                # --- NEW: Summarize if too long ---
                if original_chunk_tokens > MAX_TOKENS_PER_LTM_VECTOR_CHUNK:
                    print(f"    Tokens - Vector chunk {res_idx+1} is long ({original_chunk_tokens} tokens). Attempting summarization...")
                    summary = self.summarizer.summarize_text(
                        text_to_summarize=chunk_text_to_use,
                        model_name=summarization_llm_model, # Use summarizer's default or a specific one
                        max_summary_length=TARGET_SUMMARY_TOKENS_FOR_LTM_CHUNK 
                    )
                    if summary:
                        chunk_text_to_use = f"[Summary of a longer document]: {summary}" # Prepend context
                        print(f"      Summarized chunk to ~{count_tokens(chunk_text_to_use, target_ollama_model_for_tokens)} tokens.")
                    else:
                        print(f"      Summarization failed for chunk {res_idx+1}. Using original (might be skipped).")
                        # Keep original chunk_text_to_use if summarization fails
                # --- End Summarization ---

                res_str = f"From source (metadata: {res.get('metadata', {})}): \"{chunk_text_to_use}\""
                res_tokens = count_tokens(res_str, target_ollama_model_for_tokens)
                
                title_tokens = 0
                if not added_vs_title:
                    title = "\nRetrieved Similar Information from Past Interactions/Documents:"
                    title_tokens = count_tokens(title, target_ollama_model_for_tokens)

                if current_total_tokens + title_tokens + res_tokens < TARGET_MAX_PROMPT_TOKENS:
                    if not added_vs_title:
                        knowledge_context_str_parts.append(title)
                        current_total_tokens += title_tokens
                        added_vs_title = True
                    knowledge_context_str_parts.append(f"  - {res_str}")
                    current_total_tokens += res_tokens
                    print(f"    Tokens - Added Vector Result {res_idx+1} ({res_tokens} tokens). Cumulative: {current_total_tokens}")
                else:
                    print(f"    Tokens - SKIPPING Vector Result {res_idx+1} (would exceed limit).")
                    break
        
        retrieved_knowledge_prompt_str = "\n".join(knowledge_context_str_parts)
    
        # STM History (logic as before, but now respecting token counts more accurately)
        stm_history_turns = self.mmu.get_stm_history()
        stm_prompt_parts = []
        user_query_tokens = count_tokens(user_query, target_ollama_model_for_tokens) # Calculate once
        
        stm_for_prompt_build = stm_history_turns
        if stm_for_prompt_build and stm_for_prompt_build[-1].get("role") == "user" and stm_for_prompt_build[-1].get("content") == user_query:
            stm_for_prompt_build = stm_history_turns[:-1]

        for turn_idx, turn in enumerate(reversed(stm_for_prompt_build)):
            turn_str = f"{turn['role'].capitalize()}: {turn['content']}"
            turn_tokens = count_tokens(turn_str, target_ollama_model_for_tokens)
            
            if current_total_tokens + turn_tokens + user_query_tokens < TARGET_MAX_PROMPT_TOKENS:
                stm_prompt_parts.insert(0, turn_str)
                current_total_tokens += turn_tokens
                print(f"    Tokens - Added STM Turn (from end) {len(stm_for_prompt_build)-turn_idx} ({turn_tokens} tokens). Cumulative: {current_total_tokens}")
            else:
                print(f"    Tokens - SKIPPING older STM Turn (from end) {len(stm_for_prompt_build)-turn_idx} (would exceed limit).")
                break
        
        stm_history_prompt_str = "\n".join(stm_prompt_parts)

        # 4. Construct final user message content
        full_user_prompt_content_parts = []
        if stm_history_prompt_str:
            full_user_prompt_content_parts.append("Relevant Conversation History:")
            full_user_prompt_content_parts.append(stm_history_prompt_str)
        if retrieved_knowledge_prompt_str:
            if full_user_prompt_content_parts: full_user_prompt_content_parts.append("\n") 
            full_user_prompt_content_parts.append("RETRIEVED KNOWLEDGE CONTEXT:")
            full_user_prompt_content_parts.append(retrieved_knowledge_prompt_str)
        if full_user_prompt_content_parts: full_user_prompt_content_parts.append("\n\n") 
        full_user_prompt_content_parts.append(f"User Query: {user_query}")
        final_user_content = "".join(full_user_prompt_content_parts)
        
        # Recalculate total tokens based on final assembled prompt parts for user message
        # (System message is separate)
        final_user_content_tokens = count_tokens(final_user_content, target_ollama_model_for_tokens)
        final_prompt_total_tokens = system_tokens + final_user_content_tokens
        
        print(f"    Tokens - Final User Content Block: {final_user_content_tokens}")
        print(f"  CO: Final total prompt tokens: {final_prompt_total_tokens} (Target: {TARGET_MAX_PROMPT_TOKENS})")

        # Defensive check: if somehow we still went over, we might need to truncate final_user_content
        # For now, we rely on the iterative building to prevent this.
        if final_prompt_total_tokens > TARGET_MAX_PROMPT_TOKENS:
            print(f"  CO: WARNING - Final prompt tokens ({final_prompt_total_tokens}) slightly exceeded target. This might indicate a flaw in iterative counting or very large user query.")
            # Simplistic truncation of user content if it happens - not ideal
            # A better approach would be more careful iterative building or summarizing the user_content block itself
            # This part is complex to do perfectly without re-tokenizing many times.
            # For now, this warning is key.

        messages.append({"role": "user", "content": final_user_content})
        return messages

    def handle_user_message(self, user_message: str, conversation_id: str = None) -> iter: # Now returns an iterator (generator)
        """
        Main handler for processing a user's message.
        Yields response chunks for streaming.
        """
        if not user_message.strip(): 
            yield "Please say something!" # Yield instead of return
            return # Must use return in a generator to stop it
        
        # --- STM Management & Determine if message is a question (DEFINE is_likely_question EARLIER) ---
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
        print(f"\nCO: Handling user message for conversation '{conversation_id}': '{user_message}'")
        
        # Define is_likely_question here, based on the raw user_message
        question_indicators_for_check = ["what", "where", "when", "who", "why", "how", "do ", "can ", "is ", "are "] # For startswith
        is_likely_question = user_message.endswith("?") or \
                             any(user_message.lower().startswith(q_word) for q_word in question_indicators_for_check)
        if is_likely_question:
            print(f"  CO_INFO: User message \"{user_message[:50]}...\" detected as a likely question.")
        # --- End STM Management & Question Check ---

        self.mmu.add_stm_turn(role="user", content=user_message)

        print(f"  CO: Attempting to extract facts from user message: \"{user_message[:100]}...\"")
        extracted_facts_from_user = self.fact_extractor.extract_facts(text_to_process=user_message)
        
        if extracted_facts_from_user: # Facts were potentially extracted by agent
            print(f"  CO: Raw extracted {len(extracted_facts_from_user)} fact(s) from user message by FactExtractionAgent.")
            filtered_facts_to_store = []
            # common_fillers and min_meaningful_length are used here
            common_fillers = ["is", "are", "was", "were", "am", "be", "the", "a", "an", "my", "me", "i", "it", "this", "that", "and", "or", "for", "to"]
            min_meaningful_length = 3

            if is_likely_question: # Use the already defined variable
                print(f"    CO_FILTER: User message appears to be a question. Discarding extracted 'facts' for SKB storage.")
                # No need to re-assign extracted_facts_from_user = [], just don't populate filtered_facts_to_store
            else: # Only process/filter if not a question (or if we decide to store facts from questions differently)
                for fact in extracted_facts_from_user:
                    s = str(fact.get('subject', '')).strip()
                    p = str(fact.get('predicate', '')).strip() 
                    o = str(fact.get('object', '')).strip()
                    if not s or not p or not o:
                        # print(f"    CO_FILTER: Skipping fact with empty S/P/O: {fact}") # Optional print
                        continue
                    if s.lower() in common_fillers and o.lower() in common_fillers and len(s) <= min_meaningful_length and len(o) <= min_meaningful_length:
                        # print(f"    CO_FILTER: Skipping fact with trivial subject and object: {fact}") # Optional print
                        continue
                    # Simplified question part filter as is_likely_question already handles broader case
                    if (s.lower().startswith("my ") or s.lower().startswith("user's ")) and (p.lower() == "what"):
                        # print(f"    CO_FILTER: Skipping fact that looks like part of a question: {fact}") # Optional print
                        continue
                    if s.lower() in ["my", "i"] and p.lower() in common_fillers and o.lower() in common_fillers:
                        # print(f"    CO_FILTER: Skipping overly generic first-person fact: {fact}") # Optional print
                        continue
                    filtered_facts_to_store.append({"subject": s, "predicate": p, "object": o}) 
            
            if filtered_facts_to_store: # This block only runs if not a question AND facts survived filtering
                print(f"  CO: Storing {len(filtered_facts_to_store)} filtered fact(s) to LTM/SKB.")
                for fact_to_store in filtered_facts_to_store:
                     print(f"    CO: Storing fact: S='{fact_to_store['subject']}', P='{fact_to_store['predicate']}', O='{fact_to_store['object']}'")
                     self.mmu.store_ltm_fact(subject=fact_to_store['subject'],predicate=fact_to_store['predicate'],object_value=fact_to_store['object'],confidence=0.80)

        retrieved_knowledge = self.knowledge_retriever.search_knowledge(query_text=user_message, top_k_vector=DEFAULT_TOP_K_VECTOR_SEARCH, search_skb_facts=True)
        # Build Prompt
        llm_messages = self._build_prompt_with_context(conversation_id=conversation_id, user_query=user_message, retrieved_knowledge=retrieved_knowledge)
        
        print(f"  CO: Sending request to LLM (model: {self.default_llm_model}) for streaming response...")
        
        # --- MODIFIED SECTION for LSW call and yielding ---
        assistant_response_stream = self.lsw.generate_chat_completion(
            messages=llm_messages,
            model_name=self.default_llm_model,
            temperature=0.5, # Or your preferred temp
            stream=True      # <--- IMPORTANT: Request streaming from LSW
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
            _current_ltm_hist = self.mmu.get_ltm_conversation_history(conversation_id)
            _user_turn_seq_id = len(_current_ltm_hist) + 1
            self.mmu.log_ltm_interaction(conversation_id, _user_turn_seq_id, "user", user_message)
            self.mmu.log_ltm_interaction(conversation_id, _user_turn_seq_id + 1, "assistant_error", error_message, metadata={"error": "LSW stream init failed"})
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
        
        # Now that streaming is complete (or an error occurred and was yielded),
        # update STM and log the full interaction to LTM.
        print(f"  CO: Full assistant response (accumulated): \"{accumulated_response_for_ltm[:100].strip()}...\"")
        self.mmu.add_stm_turn(role="assistant", content=accumulated_response_for_ltm)

        # LTM Logging
        print(f"  CO: Preparing to log to LTM for conversation '{conversation_id}'.")
        current_ltm_history_len_before_log = len(self.mmu.get_ltm_conversation_history(conversation_id))
        # The user turn might have already been logged if an LSW stream init error occurred.
        # This logic needs to be robust. Let's assume user turn is logged once.
        # A simple way is to only log user turn here if it wasn't logged in an error path.
        # For now, let's refine the sequence ID logic based on current LTM state.
        
        # Log User Turn First
        user_turn_log_seq_id = current_ltm_history_len_before_log + 1 
        # Check if this user turn was already logged (e.g. from an earlier LSW error path)
        # This is complex. A simpler model: LTM logging only happens *here* at the end for a successful interaction.
        # If LSW fails to give a stream, that earlier error logging handles it.

        print(f"    LTM logging: User turn seq ID: {user_turn_log_seq_id} for '{user_message[:30]}...'")
        user_turn_ltm_entry_id = self.mmu.log_ltm_interaction(
            conversation_id=conversation_id, turn_sequence_id=user_turn_log_seq_id,
            role="user", content=user_message)

        # --- Vector Store Learning (uses is_likely_question defined earlier) ---
        if user_message and not is_likely_question: 
            user_message_tokens = count_tokens(user_message, self.default_llm_model) 
            if user_message_tokens >= MIN_TOKENS_FOR_USER_STATEMENT_TO_VECTOR_STORE:
                print(f"  CO: User message is significant ({user_message_tokens} tokens). Storing to Vector Store.")
                doc_id_vs = f"user_stmt_{conversation_id}_{user_turn_log_seq_id}" 
                metadata_vs = {
                    "type": "user_statement", "conversation_id": conversation_id,
                    "turn_sequence_id": user_turn_log_seq_id,
                    "ltm_raw_log_entry_id": user_turn_ltm_entry_id if user_turn_ltm_entry_id else "unknown",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) 
                }
                vector_store_add_id = self.mmu.add_document_to_ltm_vector_store(
                    text_chunk=user_message, metadata=metadata_vs, doc_id=doc_id_vs
                )
                if vector_store_add_id: print(f"    CO: User statement stored in Vector Store with ID: {vector_store_add_id}")
                else: print(f"    CO: Failed to store user statement in Vector Store.")
            else:
                print(f"  CO: User message not stored in Vector Store (tokens: {user_message_tokens} < {MIN_TOKENS_FOR_USER_STATEMENT_TO_VECTOR_STORE}).")
        # --- End Vector Store Learning for user message ---
        
        # Log Assistant Turn
        assistant_turn_log_seq_id = user_turn_log_seq_id + 1
        print(f"    LTM logging: Assistant turn seq ID: {assistant_turn_log_seq_id} for '{accumulated_response_for_ltm[:30]}...'")
        self.mmu.log_ltm_interaction(
            conversation_id=conversation_id, turn_sequence_id=assistant_turn_log_seq_id,
            role="assistant", content=accumulated_response_for_ltm,
            llm_model_used=self.default_llm_model,
            metadata={
                "retrieved_vector_ids": [res['id'] for res in retrieved_knowledge.get("vector_results",[]) if 'id' in res],
                "retrieved_fact_ids": [fact['fact_id'] for fact in retrieved_knowledge.get("skb_fact_results",[]) if 'fact_id' in fact],
                "llm_streamed": True # Add metadata that it was streamed
            }
        )

if __name__ == "__main__":
    print("--- Testing CO with Context Summarization ---")

    # Test paths (use new ones to ensure fresh LTM for this specific test)
    test_co_sum_ltm_sqlite_db_path = 'test_co_sum_ltm_sqlite.db'
    test_co_sum_ltm_chroma_dir = 'test_co_sum_ltm_chroma'
    test_co_sum_mtm_db_path = 'test_co_sum_mtm_store.json'

    # Cleanup
    if os.path.exists(test_co_sum_mtm_db_path): os.remove(test_co_sum_mtm_db_path)
    if os.path.exists(test_co_sum_ltm_sqlite_db_path): os.remove(test_co_sum_ltm_sqlite_db_path)
    if os.path.exists(test_co_sum_ltm_chroma_dir): shutil.rmtree(test_co_sum_ltm_chroma_dir)

    # ... (Initialize MMU, LSW, Agents, Orchestrator as before, using these new paths) ...
    # Ensure all necessary instances are created, including summarizer for CO.
    test_mmu_sum = None
    test_lsw_sum = None
    knowledge_retriever_sum = None
    summarizer_agent_sum = None # Renamed to avoid conflict if you run other agent tests
    fact_extractor_sum = None
    orchestrator_sum = None

    try:
        print("\nInitializing MMU for CO Summarization test...")
        test_mmu_sum = MemoryManagementUnit(
            ltm_sqlite_db_path=test_co_sum_ltm_sqlite_db_path,
            ltm_chroma_persist_dir=test_co_sum_ltm_chroma_dir
        )
        print("\nInitializing LSW for CO Summarization test...")
        test_lsw_sum = LLMServiceWrapper(default_chat_model="gemma3:1b-it-fp16") 
        if not test_lsw_sum.client: raise Exception("Ollama client in LSW failed.")
        
        print("\nInitializing Agents for CO Summarization test...")
        knowledge_retriever_sum = KnowledgeRetrieverAgent(mmu=test_mmu_sum)
        summarizer_agent_sum = SummarizationAgent(lsw=test_lsw_sum) # CRITICAL: Use this instance
        fact_extractor_sum = FactExtractionAgent(lsw=test_lsw_sum)

        print("\nInitializing CO with Summarization capability...")
        orchestrator_sum = ConversationOrchestrator(
            mmu=test_mmu_sum, lsw=test_lsw_sum,
            knowledge_retriever=knowledge_retriever_sum,
            summarizer=summarizer_agent_sum, # PASS THE CORRECT SUMMARIZER INSTANCE
            fact_extractor=fact_extractor_sum
        )
    except Exception as e:
        print(f"FATAL: Error during CO Summarization test environment setup: {e}")
        exit()

    # --- Pre-populate LTM Vector Store with a VERY LONG document ---
    print("\nPre-populating LTM Vector Store with a long document...")
    # Create a long string (e.g., > MAX_TOKENS_PER_LTM_VECTOR_CHUNK which is 500)
    # Repeat a paragraph multiple times. One paragraph is ~50-70 words, ~70-100 tokens.
    # 6-7 repetitions should exceed 500 tokens.
    paragraph = (
        "The study of artificial intelligence (AI) is a dynamic and rapidly evolving field that "
        "encompasses a wide range of theories, methodologies, and applications. At its core, AI "
        "aims to create systems capable of performing tasks that typically require human intelligence, "
        "such as learning, problem-solving, perception, language understanding, and decision-making. "
        "Key subfields include machine learning, natural language processing, computer vision, and robotics. "
        "Ethical considerations are also paramount as AI capabilities advance, prompting discussions "
        "about bias, accountability, and the societal impact of autonomous systems. "
    ) # This paragraph is approx 95 words, likely > 100 tokens.
    
    long_document_text = (paragraph + " ") * 7 # Repeat 7 times 
    # This should be around 700+ tokens, exceeding MAX_TOKENS_PER_LTM_VECTOR_CHUNK (500)

    # Check if LTM's vector store is usable
    ltm_vector_store_ready_sum_test = False
    if hasattr(test_mmu_sum.ltm, 'vector_store') and test_mmu_sum.ltm.vector_store and \
       hasattr(test_mmu_sum.ltm.vector_store, 'collection') and test_mmu_sum.ltm.vector_store.collection and \
       hasattr(test_mmu_sum.ltm.vector_store.collection, '_embedding_function') and \
       test_mmu_sum.ltm.vector_store.collection._embedding_function is not None:
        ltm_vector_store_ready_sum_test = True

    if ltm_vector_store_ready_sum_test:
        long_doc_id = "long_doc_ai_ethics_001"
        orchestrator_sum.mmu.add_document_to_ltm_vector_store( # Use orchestrator's MMU
            text_chunk=long_document_text,
            metadata={"source": "ai_ethics_treatise", "topic": "AI Ethics Detailed Overview"},
            doc_id=long_doc_id
        )
        print(f"  Long document (approx {count_tokens(long_document_text, orchestrator_sum.default_llm_model)} tokens) stored in Vector Store with ID: {long_doc_id}")
        time.sleep(1) # Give Chroma a moment
    else:
        print("  LTM Vector Store embedding function not available. Skipping long document population for summarization test.")
        # If VS not ready, this test won't be very effective.
        exit() # Exit if vector store not ready, as test depends on it.


    # --- Test Conversation Flow with Potential Summarization ---
    sum_test_convo_id = "sum_test_conv_001"

    print(f"\n--- Test SUM 1: User asks about AI Ethics (should retrieve long doc and summarize it) ---")
    user_question_sum = "Can you tell me about the core concepts of AI ethics and its applications?"
    
    response_sum1 = orchestrator_sum.handle_user_message(
        user_message=user_question_sum,
        conversation_id=sum_test_convo_id
    )
    print(f"Chatbot Response SUM1: {response_sum1}")
    # Expected: 
    # - KRA retrieves the long document from Vector Store.
    # - CO._build_prompt_with_context detects it's > MAX_TOKENS_PER_LTM_VECTOR_CHUNK.
    # - CO calls self.summarizer.summarize_text().
    # - A summary like "[Summary of a longer document]: ..." is included in the LLM prompt.
    # - Bot's response uses this summarized information.

    print("\nCO Summarization test finished.")
    # ... (Final cleanup logic using test_co_sum_* paths) ...
    print("\nAttempting final cleanup of CO Summarization test files...")
    del orchestrator_sum
    del knowledge_retriever_sum
    del summarizer_agent_sum # Delete the summarizer instance
    del fact_extractor_sum
    del test_lsw_sum
    if hasattr(test_mmu_sum, 'mtm') and test_mmu_sum.mtm.is_persistent and hasattr(test_mmu_sum.mtm, 'db') and test_mmu_sum.mtm.db:
        test_mmu_sum.mtm.db.close()
    del test_mmu_sum
    gc.collect()
    time.sleep(0.1)
    # ... (os.remove and shutil.rmtree for test_co_sum_* files/dirs) ...
    if os.path.exists(test_co_sum_mtm_db_path): 
        try: os.remove(test_co_sum_mtm_db_path)
        except Exception as e: print(f"  Could not remove {test_co_sum_mtm_db_path}: {e}")
    if os.path.exists(test_co_sum_ltm_sqlite_db_path): 
        try: os.remove(test_co_sum_ltm_sqlite_db_path)
        except Exception as e: print(f"  Could not remove {test_co_sum_ltm_sqlite_db_path}: {e}")
    if os.path.exists(test_co_sum_ltm_chroma_dir): 
        try: shutil.rmtree(test_co_sum_ltm_chroma_dir, ignore_errors=False)
        except Exception as e: print(f"  Could not remove directory {test_co_sum_ltm_chroma_dir}: {e}")
    print("CO Summarization test file cleanup attempt finished.")