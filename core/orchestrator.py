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
        
        # Add SKB facts first (often more precise)
        if retrieved_knowledge.get("skb_fact_results"):
            # knowledge_context_str_parts.append("Retrieved Facts from Knowledge Base:") # Title token cost
            for fact_idx, fact in enumerate(retrieved_knowledge["skb_fact_results"][:3]): # Still limit to top 3 SKB for now
                fact_str = f"Fact: {fact.get('subject')} {fact.get('predicate')} {fact.get('object')}."
                fact_tokens = count_tokens(fact_str, target_ollama_model_for_tokens)
                if current_total_tokens + fact_tokens < TARGET_MAX_PROMPT_TOKENS:
                    if not knowledge_context_str_parts: # Add title only if we add content
                         title = "Retrieved Facts from Knowledge Base:"
                         knowledge_context_str_parts.append(title)
                         current_total_tokens += count_tokens(title, target_ollama_model_for_tokens)
                    knowledge_context_str_parts.append(f"  - {fact_str}")
                    current_total_tokens += fact_tokens
                    print(f"    Tokens - Added SKB Fact {fact_idx+1} ({fact_tokens} tokens). Cumulative: {current_total_tokens}")
                else:
                    print(f"    Tokens - SKIPPING SKB Fact {fact_idx+1} (would exceed limit).")
                    break 
        
        # Add Vector Store results
        if retrieved_knowledge.get("vector_results"):
            # temp_vector_title_added = False
            for res_idx, res in enumerate(retrieved_knowledge["vector_results"][:DEFAULT_TOP_K_VECTOR_SEARCH]):
                # Future: If res.get('text_chunk') is very long, summarize it first using self.summarizer
                # For now, we might truncate it or include as is if it fits.
                # Let's try to include a portion if it's too long.
                chunk_text = res.get('text_chunk', '')
                # Simple truncation for now if a single chunk is massive (e.g. > 1000 tokens itself)
                # A better way would be to summarize.
                # max_chunk_tokens = 1000 
                # chunk_tokens_initial = count_tokens(chunk_text, target_ollama_model_for_tokens)
                # if chunk_tokens_initial > max_chunk_tokens:
                #     # This is a placeholder for summarization or smarter truncation
                #     print(f"    Tokens - Vector chunk {res_idx+1} too long ({chunk_tokens_initial}), needs summarization/truncation logic.")
                #     # For now, just skip if too long to avoid complex logic here.
                #     # Or, try to take a prefix, but even that needs token counting.
                #     continue # Or implement truncation to X tokens.

                res_str = f"From source (metadata: {res.get('metadata', {})}): \"{chunk_text}\""
                res_tokens = count_tokens(res_str, target_ollama_model_for_tokens)

                if current_total_tokens + res_tokens < TARGET_MAX_PROMPT_TOKENS:
                    if not knowledge_context_str_parts or "Retrieved Similar Information" not in knowledge_context_str_parts[-1 if knowledge_context_str_parts and "Fact" in knowledge_context_str_parts[0] else 0]: # Add title if needed
                        title = "\nRetrieved Similar Information from Past Interactions/Documents:"
                        knowledge_context_str_parts.append(title)
                        current_total_tokens += count_tokens(title, target_ollama_model_for_tokens)
                    knowledge_context_str_parts.append(f"  - {res_str}")
                    current_total_tokens += res_tokens
                    print(f"    Tokens - Added Vector Result {res_idx+1} ({res_tokens} tokens). Cumulative: {current_total_tokens}")
                else:
                    print(f"    Tokens - SKIPPING Vector Result {res_idx+1} (would exceed limit).")
                    break
        
        retrieved_knowledge_prompt_str = "\n".join(knowledge_context_str_parts)
        # Note: current_total_tokens already includes tokens for retrieved_knowledge_prompt_str parts

        # 3. Short-Term Memory (STM) - Add recent turns, newest first from available STM
        #    but ensure they are presented in chronological order in the prompt.
        stm_history_turns = self.mmu.get_stm_history()
        stm_prompt_parts = []

        # STM usually includes the current user query as the last item if add_stm_turn was called before prompt building.
        # We want to build history *before* the current query for the prompt.
        stm_for_prompt_build = stm_history_turns
        if stm_for_prompt_build and stm_for_prompt_build[-1].get("role") == "user" and stm_for_prompt_build[-1].get("content") == user_query:
            stm_for_prompt_build = stm_history_turns[:-1] # Exclude current user query

        for turn_idx, turn in enumerate(reversed(stm_for_prompt_build)):
            turn_str = f"{turn['role'].capitalize()}: {turn['content']}"
            turn_tokens = count_tokens(turn_str, target_ollama_model_for_tokens)
            
            # Check if adding this turn (plus user_query tokens) exceeds the limit
            # User query tokens will be added last, so reserve space for them.
            user_query_tokens = count_tokens(user_query, target_ollama_model_for_tokens) # Calculate once
            if current_total_tokens + turn_tokens + user_query_tokens < TARGET_MAX_PROMPT_TOKENS:
                stm_prompt_parts.insert(0, turn_str) # Insert at beginning to maintain chronological order
                current_total_tokens += turn_tokens
                print(f"    Tokens - Added STM Turn (from end) {len(stm_for_prompt_build)-turn_idx} ({turn_tokens} tokens). Cumulative: {current_total_tokens}")
            else:
                print(f"    Tokens - SKIPPING older STM Turn (from end) {len(stm_for_prompt_build)-turn_idx} (would exceed limit).")
                break
        
        stm_history_prompt_str = "\n".join(stm_prompt_parts)
        # Note: current_total_tokens already includes tokens for stm_history_prompt_str

        # 4. Construct final user message content
        full_user_prompt_content_parts = []
        if stm_history_prompt_str:
            full_user_prompt_content_parts.append("Relevant Conversation History:")
            full_user_prompt_content_parts.append(stm_history_prompt_str)
        
        if retrieved_knowledge_prompt_str:
            if full_user_prompt_content_parts: full_user_prompt_content_parts.append("\n") # Add separator
            full_user_prompt_content_parts.append("RETRIEVED KNOWLEDGE CONTEXT:")
            full_user_prompt_content_parts.append(retrieved_knowledge_prompt_str)

        if full_user_prompt_content_parts: full_user_prompt_content_parts.append("\n\n") # Add separator
        full_user_prompt_content_parts.append(f"User Query: {user_query}")
        
        final_user_content = "".join(full_user_prompt_content_parts)
        user_query_tokens_final = count_tokens(final_user_content, target_ollama_model_for_tokens) # This is the token count of the whole user part
        
        # This final check isn't perfect because current_total_tokens was for system + LTM + STM strings separately.
        # A more accurate way is to tokenize the whole user content string.
        # For now, this is an estimate.
        final_prompt_tokens_estimate = system_tokens + user_query_tokens_final
        print(f"    Tokens - Final User Content Block: {user_query_tokens_final}")
        print(f"  CO: Estimated total prompt tokens: {final_prompt_tokens_estimate} (Target: {TARGET_MAX_PROMPT_TOKENS})")

        messages.append({"role": "user", "content": final_user_content})
        
        return messages


    def handle_user_message(self, user_message: str, conversation_id: str = None) -> str or None:
        if not user_message.strip(): return "Please say something!"
        
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
                    # print(f"    CO: Storing fact: S='{fact_to_store['subject']}', P='{fact_to_store['predicate']}', O='{fact_to_store['object']}'") # Optional
                    self.mmu.store_ltm_fact(subject=fact_to_store['subject'], predicate=fact_to_store['predicate'], object_value=fact_to_store['object'], confidence=0.80)
            elif extracted_facts_from_user and not is_likely_question: # Raw facts existed, not a question, but all got filtered
                print("  CO: All raw extracted facts were filtered out. No facts stored in SKB.")
            # If it was a question, the earlier filter message already covered it.
        
        elif extracted_facts_from_user is None: 
             print("  CO: Fact extraction from user message failed or encountered an error (agent returned None).")
        else: # Agent returned [], meaning it found no facts
             print("  CO: No facts extracted from user message by agent (agent returned empty list).")
        
        # ... (KRA search, _build_prompt_with_context, LSW call - these are fine) ...
        retrieved_knowledge = self.knowledge_retriever.search_knowledge(query_text=user_message, top_k_vector=DEFAULT_TOP_K_VECTOR_SEARCH, search_skb_facts=True)
        llm_messages = self._build_prompt_with_context(conversation_id=conversation_id, user_query=user_message, retrieved_knowledge=retrieved_knowledge)
        print(f"  CO: Sending request to LLM (model: {self.default_llm_model})...")
        assistant_response_text = self.lsw.generate_chat_completion(messages=llm_messages, model_name=self.default_llm_model, temperature=0.5)
        if assistant_response_text is None:
            assistant_response_text = "I'm sorry, I encountered an issue and couldn't generate a response right now."
        print(f"  CO: LLM Raw Response: \"{assistant_response_text[:100].strip()}...\"")
        self.mmu.add_stm_turn(role="assistant", content=assistant_response_text)


        print(f"  CO: Preparing to log to LTM for conversation '{conversation_id}'.")
        current_ltm_history_len_before_log = len(self.mmu.get_ltm_conversation_history(conversation_id))
        user_turn_log_seq_id = current_ltm_history_len_before_log + 1
        assistant_turn_log_seq_id = user_turn_log_seq_id + 1 
        print(f"    LTM logging: User turn seq ID: {user_turn_log_seq_id} for '{user_message[:30]}...'")
        user_turn_ltm_entry_id = self.mmu.log_ltm_interaction(
            conversation_id=conversation_id, turn_sequence_id=user_turn_log_seq_id,
            role="user", content=user_message,
        )

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
        
        print(f"    LTM logging: Assistant turn seq ID: {assistant_turn_log_seq_id} for '{assistant_response_text[:30]}...'")
        self.mmu.log_ltm_interaction(
            conversation_id=conversation_id, turn_sequence_id=assistant_turn_log_seq_id,
            role="assistant", content=assistant_response_text, llm_model_used=self.default_llm_model,
            metadata={
                "retrieved_vector_ids": [res['id'] for res in retrieved_knowledge.get("vector_results",[]) if 'id' in res],
                "retrieved_fact_ids": [fact['fact_id'] for fact in retrieved_knowledge.get("skb_fact_results",[]) if 'fact_id' in fact],
                "llm_call_failed": assistant_response_text is None 
            }
        )
        return assistant_response_text

if __name__ == "__main__":
    print("--- Testing CO with Fact Extraction & Vector Store Learning ---")

    # Test paths (use new ones to ensure fresh LTM for this specific test)
    test_co_vs_ltm_sqlite_db_path = 'test_co_vs_learn_ltm_sqlite.db'
    test_co_vs_ltm_chroma_dir = 'test_co_vs_learn_ltm_chroma'
    test_co_vs_mtm_db_path = 'test_co_vs_learn_mtm_store.json'

    # Cleanup
    if os.path.exists(test_co_vs_mtm_db_path): os.remove(test_co_vs_mtm_db_path)
    if os.path.exists(test_co_vs_ltm_sqlite_db_path): os.remove(test_co_vs_ltm_sqlite_db_path)
    if os.path.exists(test_co_vs_ltm_chroma_dir): shutil.rmtree(test_co_vs_ltm_chroma_dir)

    test_mmu_vs = None # Use unique names for test instances
    test_lsw_vs = None
    knowledge_retriever_vs = None
    summarizer_vs = None
    fact_extractor_vs = None
    orchestrator_vs = None

    try:
        print("\nInitializing MMU for CO VS learning test...")
        test_mmu_vs = MemoryManagementUnit(
            ltm_sqlite_db_path=test_co_vs_ltm_sqlite_db_path,
            ltm_chroma_persist_dir=test_co_vs_ltm_chroma_dir
        )
        print("\nInitializing LSW for CO VS learning test...")
        test_lsw_vs = LLMServiceWrapper(default_chat_model="gemma3:1b-it-fp16") 
        if not test_lsw_vs.client: raise Exception("Ollama client in LSW failed.")
        
        print("\nInitializing Agents for CO VS learning test...")
        knowledge_retriever_vs = KnowledgeRetrieverAgent(mmu=test_mmu_vs)
        summarizer_vs = SummarizationAgent(lsw=test_lsw_vs)
        fact_extractor_vs = FactExtractionAgent(lsw=test_lsw_vs)

        print("\nInitializing CO with Fact Extraction & VS Learning capability...")
        orchestrator_vs = ConversationOrchestrator(
            mmu=test_mmu_vs, lsw=test_lsw_vs,
            knowledge_retriever=knowledge_retriever_vs,
            summarizer=summarizer_vs, fact_extractor=fact_extractor_vs
        )
    except Exception as e:
        print(f"FATAL: Error during CO VS learning test environment setup: {e}")
        exit()
    
    # --- Test Conversation Flow with Vector Store Learning ---
    vs_learn_convo_id = "vs_learn_conv_001"

    print(f"\n--- Test VS 1: User makes a significant statement ---")
    user_statement_for_vs = "The upcoming Galactic Summit on AI Ethics will feature keynote speaker Dr. Aris Thorne and will discuss the future of sentient machines."
    # This should exceed MIN_TOKENS_FOR_USER_STATEMENT_TO_VECTOR_STORE
    
    response_vs1 = orchestrator_vs.handle_user_message(
        user_message=user_statement_for_vs,
        conversation_id=vs_learn_convo_id
    )
    print(f"Chatbot Response VS1: {response_vs1}")

    # Check Vector Store (this is an indirect check, we'd need to query it)
    # For now, we rely on the CO's print statements:
    # "CO: User message is significant... Storing to Vector Store."
    # "CO: User statement stored in Vector Store with ID: ..."

    print(f"\n--- Test VS 2: User asks a semantically similar question (Vector Store recall) ---")
    # This question uses different wording but is semantically related to user_statement_for_vs
    user_question_vs = "Tell me about the conference on machine sentience and Dr. Thorne."
    response_vs2 = orchestrator_vs.handle_user_message(
        user_message=user_question_vs,
        conversation_id=vs_learn_convo_id 
    )
    print(f"Chatbot Response VS2: {response_vs2}")
    # Expected: KRA should find user_statement_for_vs in Vector Store.
    # Bot should use this retrieved context.

    print(f"\n--- Test VS 3: User makes a short, non-significant statement ---")
    user_short_statement = "Okay, thanks." # Should be less than MIN_TOKENS...
    response_vs3 = orchestrator_vs.handle_user_message(
        user_message=user_short_statement,
        conversation_id=vs_learn_convo_id
    )
    print(f"Chatbot Response VS3: {response_vs3}")
    # Expected: "CO: User message not stored in Vector Store (tokens: X < Y)."

    print("\nCO Vector Store learning tests finished.")
    # ... (Final cleanup logic using test_co_vs_* paths) ...
    print("\nAttempting final cleanup of CO VS learning test files...")
    del orchestrator_vs
    del knowledge_retriever_vs
    del summarizer_vs
    del fact_extractor_vs
    del test_lsw_vs
    if hasattr(test_mmu_vs, 'mtm') and test_mmu_vs.mtm.is_persistent and hasattr(test_mmu_vs.mtm, 'db') and test_mmu_vs.mtm.db:
        test_mmu_vs.mtm.db.close()
    del test_mmu_vs
    gc.collect()
    time.sleep(0.1)
    if os.path.exists(test_co_vs_mtm_db_path): 
        try: os.remove(test_co_vs_mtm_db_path)
        except Exception as e: print(f"  Could not remove {test_co_vs_mtm_db_path}: {e}")
    if os.path.exists(test_co_vs_ltm_sqlite_db_path): 
        try: os.remove(test_co_vs_ltm_sqlite_db_path)
        except Exception as e: print(f"  Could not remove {test_co_vs_ltm_sqlite_db_path}: {e}")
    if os.path.exists(test_co_vs_ltm_chroma_dir): 
        try: shutil.rmtree(test_co_vs_ltm_chroma_dir, ignore_errors=False)
        except Exception as e: print(f"  Could not remove directory {test_co_vs_ltm_chroma_dir}: {e}")
    print("CO VS learning test file cleanup attempt finished.")