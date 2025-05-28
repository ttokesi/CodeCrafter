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
        print(f"  Target max prompt tokens: {TARGET_MAX_PROMPT_TOKENS}") # Log the target

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
        # ... (STM Management, initial print as before) ...
        if not user_message.strip(): return "Please say something!"
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
        self.mmu.add_stm_turn(role="user", content=user_message)

        # --- Step 2.A: Extract facts from user's message and store in LTM/SKB ---
        print(f"  CO: Attempting to extract facts from user message: \"{user_message[:100]}...\"")
        extracted_facts_from_user = self.fact_extractor.extract_facts(text_to_process=user_message)
        
        if extracted_facts_from_user: # Could be None (error) or [] (no facts found by agent)
            print(f"  CO: Raw extracted {len(extracted_facts_from_user)} fact(s) from user message by FactExtractionAgent.")
            
            filtered_facts_to_store = []
            # Define terms that suggest a question rather than a statement of fact
            question_indicators = ["what", "where", "when", "who", "why", "how", "do you", "can you", "is there", "are there"]
            # Define very short/common words that are often poor subjects/objects if not part of a larger phrase
            common_fillers = ["is", "are", "was", "were", "am", "be", "the", "a", "an", "my", "me", "i", "it", "this", "that", "and", "or", "for", "to"]
            min_meaningful_length = 3 # For S/O if they are not more specific types

            # First, check if the original user_message looks like a question
            is_likely_question = user_message.endswith("?") or any(user_message.lower().startswith(q_word) for q_word in ["what", "where", "when", "who", "why", "how", "do ", "can ", "is ", "are "])

            if is_likely_question:
                print(f"    CO_FILTER: User message \"{user_message[:50]}...\" appears to be a question. Extracted 'facts' from questions are usually not stored.")
                # We might still log what the LLM extracted from a question for debugging, but not store it in SKB.
                # For now, if it's a question, we don't store any "facts" derived from it.
                extracted_facts_from_user = [] # Effectively discard facts from questions for SKB storage

            for fact in extracted_facts_from_user: # Iterate remaining facts (or all if not a question)
                s = str(fact.get('subject', '')).strip()
                p = str(fact.get('predicate', '')).strip() # Keep predicate case as LLM gives it for now
                o = str(fact.get('object', '')).strip()

                # Rule 1: Skip if any part is empty after stripping
                if not s or not p or not o:
                    print(f"    CO_FILTER: Skipping fact with empty S/P/O: {fact}")
                    continue
                
                # Rule 2: Skip if subject AND object are both very short common words
                if s.lower() in common_fillers and o.lower() in common_fillers and len(s) <= min_meaningful_length and len(o) <= min_meaningful_length:
                    print(f"    CO_FILTER: Skipping fact with trivial subject and object: {fact}")
                    continue

                # Rule 3: Skip if the fact seems to be just rephrasing a question part (e.g. S:"my job", P:"What", O:"is")
                # This is harder to generalize, but we can check for question words in P or O if S is possessive.
                if (s.lower().startswith("my ") or s.lower().startswith("user's ")) and \
                   (p.lower() in question_indicators or o.lower() in question_indicators or p.lower() == "what"): # Added "what" here
                    print(f"    CO_FILTER: Skipping fact that looks like part of a question: {fact}")
                    continue
                
                # Rule 4: Subject should generally not be just "I" or "My" unless the prompt explicitly guided it.
                # The new fact extraction prompt guides towards "user's name" etc.
                # If the LLM still gives "I" or "My" as subject, it might be a less useful fact.
                # However, "(S:I, P:enjoy, O:hiking)" is okay. This rule needs care.
                # For now, let's allow "I" if P and O are substantial.
                # Let's focus on filtering very generic S-P-O like (S:My, P:is, O:favorite)

                if s.lower() in ["my", "i"] and p.lower() in common_fillers and o.lower() in common_fillers:
                    print(f"    CO_FILTER: Skipping overly generic first-person fact: {fact}")
                    continue

                # If it passed all filters, add it
                # Storing predicate in lowercase for potentially easier matching later,
                # but this depends on how we query. For now, let's keep case from LLM for P.
                filtered_facts_to_store.append({"subject": s, "predicate": p, "object": o}) 
            
            if filtered_facts_to_store:
                print(f"  CO: Storing {len(filtered_facts_to_store)} filtered fact(s) to LTM/SKB.")
                for fact_to_store in filtered_facts_to_store:
                    print(f"    CO: Storing fact: S='{fact_to_store['subject']}', P='{fact_to_store['predicate']}', O='{fact_to_store['object']}'")
                    self.mmu.store_ltm_fact(
                        subject=fact_to_store['subject'],
                        predicate=fact_to_store['predicate'],
                        object_value=fact_to_store['object'], # LTM method takes object_value
                        confidence=0.80 # Confidence for facts learned this way
                    )
            else:
                if extracted_facts_from_user: # Means raw facts were extracted, but all got filtered out
                    print("  CO: All raw extracted facts were filtered out. No facts stored.")
                # else: No raw facts were extracted in the first place (already handled below)

        elif extracted_facts_from_user is None: # Explicitly None means an error in FactExtractionAgent
             print("  CO: Fact extraction from user message failed or encountered an error (agent returned None).")
        else: # Empty list [] returned by agent, meaning it found no facts based on its logic
             print("  CO: No facts extracted from user message by agent (agent returned empty list).")

        retrieved_knowledge = self.knowledge_retriever.search_knowledge(
            query_text=user_message,
            top_k_vector=DEFAULT_TOP_K_VECTOR_SEARCH,
            search_skb_facts=True
        )

        llm_messages = self._build_prompt_with_context(
            conversation_id=conversation_id, # Pass current conversation_id
            user_query=user_message, # Pass current user_query
            retrieved_knowledge=retrieved_knowledge
        )

        print(f"  CO: Sending request to LLM (model: {self.default_llm_model})...")
        assistant_response_text = self.lsw.generate_chat_completion(
            messages=llm_messages,
            model_name=self.default_llm_model,
            temperature=0.5
        )

        # This is the fallback if LLM call itself fails (e.g. LSW returns None)
        # It does NOT mean the LLM said "I don't know" - that's a valid response.
        if assistant_response_text is None: # Check specifically for None from LSW
            print("  CO: LLM failed to generate a response (LSW returned None).")
            assistant_response_text = "I'm sorry, I encountered an issue and couldn't generate a response right now."
            # No need to log this specific failure to LTM here, as the main logging block will log
            # the user turn and this assistant_response_text. The LSW should log its own errors.
        
        print(f"  CO: LLM Raw Response: \"{assistant_response_text[:100].strip()}...\"")
        self.mmu.add_stm_turn(role="assistant", content=assistant_response_text)

        # --- LTM Logging with Debug ---
        print(f"  CO: Preparing to log to LTM for conversation '{conversation_id}'.")
        current_ltm_history_len_before_log = len(self.mmu.get_ltm_conversation_history(conversation_id))
        user_turn_log_seq_id = current_ltm_history_len_before_log + 1
        assistant_turn_log_seq_id = user_turn_log_seq_id + 1 
        print(f"    LTM logging: User turn seq ID: {user_turn_log_seq_id} for '{user_message[:30]}...'")
        user_log_id = self.mmu.log_ltm_interaction(
            conversation_id=conversation_id, turn_sequence_id=user_turn_log_seq_id,
            role="user", content=user_message,
        )
        # Now we could use user_log_id if we re-extracted facts and wanted precise source_turn_ids

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

# --- Update Test Block in orchestrator.py ---
if __name__ == "__main__":
    print("--- Testing ConversationOrchestrator with Fact Extraction ---")

    # --- Setup Test Environment ---
    test_co_ltm_sqlite_db_path = 'test_co_learn_ltm_sqlite.db' # New DB for this test
    test_co_ltm_chroma_dir = 'test_co_learn_ltm_chroma'
    test_co_mtm_db_path = 'test_co_learn_mtm_store.json'

    if os.path.exists(test_co_mtm_db_path): os.remove(test_co_mtm_db_path)
    if os.path.exists(test_co_ltm_sqlite_db_path): os.remove(test_co_ltm_sqlite_db_path)
    if os.path.exists(test_co_ltm_chroma_dir): shutil.rmtree(test_co_ltm_chroma_dir)

    test_mmu = None
    test_lsw = None
    knowledge_retriever = None
    summarizer = None
    fact_extractor_co_test = None # New instance for CO test
    orchestrator = None

    try:
        print("\nInitializing MMU for CO learning test...")
        test_mmu = MemoryManagementUnit(
            ltm_sqlite_db_path=test_co_ltm_sqlite_db_path,
            ltm_chroma_persist_dir=test_co_ltm_chroma_dir
        )

        print("\nInitializing LSW for CO learning test...")
        test_lsw = LLMServiceWrapper(default_chat_model="gemma3:1b-it-fp16") 
        if not test_lsw.client: raise Exception("Ollama client in LSW failed to initialize.")

        print("\nInitializing Agents for CO learning test...")
        knowledge_retriever = KnowledgeRetrieverAgent(mmu=test_mmu)
        summarizer = SummarizationAgent(lsw=test_lsw)
        fact_extractor_co_test = FactExtractionAgent(lsw=test_lsw)

        print("\nInitializing ConversationOrchestrator with Fact Extraction...")
        orchestrator = ConversationOrchestrator(
            mmu=test_mmu,
            lsw=test_lsw,
            knowledge_retriever=knowledge_retriever,
            summarizer=summarizer,
            fact_extractor=fact_extractor_co_test # Pass it to CO
        )
    except Exception as e:
        print(f"FATAL: Error during CO learning test environment setup: {e}")
        exit()
    
    # --- Test Conversation Flow with Learning ---
    learning_convo_id = "learn_conv_001"

    print(f"\n--- Test 1 (Learning): User states a fact ---")
    user_statement1 = "My favorite programming language is Python."
    response1 = orchestrator.handle_user_message(
        user_message=user_statement1,
        conversation_id=learning_convo_id
    )
    print(f"Chatbot Response 1: {response1}")

    # Check SKB for the learned fact
    print("\n  CO_TEST: Checking SKB for learned fact about Python...")
    # Query based on what was likely extracted and stored
    learned_facts_python = test_mmu.get_ltm_facts(object_value="Python") # LTM method uses object_value
    found_python_fact = False
    print(f"    Querying SKB for object='Python'. Found {len(learned_facts_python)} potential fact(s).")
    for fact in learned_facts_python:
        print(f"    SKB Fact: S='{fact['subject']}', P='{fact['predicate']}', O='{fact['object']}'")
        # Check if one of the good extractions was stored
        if (fact['subject'].lower() == 'my favorite programming language' and \
            fact['predicate'].lower() == 'is' and \
            fact['object'].lower() == 'python'):
            found_python_fact = True
            break
    if found_python_fact:
        print("  CO_TEST: SUCCESS - Fact about Python seems to have been learned and stored in SKB.")
    else:
        print("  CO_TEST: FAILED - Fact about Python not found in SKB as expected.")


    print(f"\n--- Test 2 (Recall): User asks about the learned fact ---")
    # Start a new "logical" user interaction, but can be same conversation_id
    # to see if STM + LTM (now with learned fact) helps.
    # Or, use a new conversation_id and see if LTM alone can recall it. Let's try same convo first.
    user_question1 = "What is my favorite programming language?"
    response2 = orchestrator.handle_user_message(
        user_message=user_question1,
        conversation_id=learning_convo_id 
    )
    print(f"Chatbot Response 2: {response2}")
    # Ideal response: "Your favorite programming language is Python." (or similar, using the learned fact)

    print(f"\n--- Test 3 (Learning another fact): User states another fact ---")
    user_statement2 = "The project codename is Phoenix."
    response3 = orchestrator.handle_user_message(
        user_message=user_statement2,
        conversation_id=learning_convo_id
    )
    print(f"Chatbot Response 3: {response3}")

    print("\n  CO_TEST: Checking SKB for learned fact about Project Phoenix...")
    learned_facts_phoenix = test_mmu.get_ltm_facts(object_value="Phoenix")
    found_phoenix_fact = False
    print(f"    Querying SKB for object='Phoenix'. Found {len(learned_facts_phoenix)} potential fact(s).")
    for fact in learned_facts_phoenix:
        print(f"    SKB Fact: S='{fact['subject']}', P='{fact['predicate']}', O='{fact['object']}'")
        if (('project' in fact['subject'].lower() or 'codename' in fact['subject'].lower()) and \
            fact['predicate'].lower() == 'is' and \
            fact['object'].lower() == 'phoenix'):
            found_phoenix_fact = True
            break
    if found_phoenix_fact:
        print("  CO_TEST: SUCCESS - Fact about Phoenix seems to have been learned and stored in SKB.")
    else:
        print("  CO_TEST: FAILED - Fact about Phoenix not found in SKB as expected.")

    print(f"\n--- Test 4 (Recall multiple facts): User asks a question that might use both ---")
    # This might be too complex for the current simple RAG and prompt, but let's see
    user_question2 = "Tell me about my favorite language and the project Phoenix."
    response4 = orchestrator.handle_user_message(
        user_message=user_question2,
        conversation_id=learning_convo_id 
    )
    print(f"Chatbot Response 4: {response4}")


    print("\nConversationOrchestrator learning tests finished.")
    # ... (Final cleanup logic as before, using test_co_learn_* paths) ...
    print("\nAttempting final cleanup of CO learning test files...")
    del orchestrator
    del knowledge_retriever
    del summarizer
    del fact_extractor_co_test # Delete the new agent instance
    if hasattr(test_lsw, 'client') and test_lsw.client: pass 
    del test_lsw
    if hasattr(test_mmu, 'mtm') and test_mmu.mtm.is_persistent and hasattr(test_mmu.mtm, 'db') and test_mmu.mtm.db:
        test_mmu.mtm.db.close()
    del test_mmu
    gc.collect()
    time.sleep(0.1)
    if os.path.exists(test_co_mtm_db_path): 
        try: os.remove(test_co_mtm_db_path)
        except Exception as e: print(f"  Could not remove {test_co_mtm_db_path}: {e}")
    if os.path.exists(test_co_ltm_sqlite_db_path): 
        try: os.remove(test_co_ltm_sqlite_db_path)
        except Exception as e: print(f"  Could not remove {test_co_ltm_sqlite_db_path}: {e}")
    if os.path.exists(test_co_ltm_chroma_dir): 
        try: shutil.rmtree(test_co_ltm_chroma_dir, ignore_errors=False)
        except Exception as e: print(f"  Could not remove directory {test_co_ltm_chroma_dir}: {e}")
    print("CO learning test file cleanup attempt finished.")