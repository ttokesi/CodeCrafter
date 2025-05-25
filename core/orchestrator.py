# offline_chat_bot/core/orchestrator.py
import uuid # For generating conversation IDs if not provided
import os
import shutil
import time
import gc 

# Conditional imports for MMU, LSW, and Agents based on execution context
if __name__ == '__main__' and __package__ is None:
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from mmu.mmu_manager import MemoryManagementUnit
    from core.llm_service_wrapper import LLMServiceWrapper
    from core.agents import KnowledgeRetrieverAgent, SummarizationAgent
else:
    from mmu.mmu_manager import MemoryManagementUnit
    from .llm_service_wrapper import LLMServiceWrapper
    from .agents import KnowledgeRetrieverAgent, SummarizationAgent

# Default configuration for the orchestrator
DEFAULT_MAX_CONTEXT_TOKENS_APPROX = 3000 # Approximate token limit for context fed to LLM
                                         # This needs to be less than the LLM's actual context window
                                         # (e.g., Llama 3 8k, Gemma 8k)
                                         # We'll use a simple word count as a proxy for now.
DEFAULT_TOP_K_VECTOR_SEARCH = 3

class ConversationOrchestrator:
    """
    Manages the overall conversation flow, orchestrating interactions between
    the user, MMU, LSW, and various agents/tools.
    """
    def __init__(self, 
                 mmu: MemoryManagementUnit, 
                 lsw: LLMServiceWrapper,
                 knowledge_retriever: KnowledgeRetrieverAgent,
                 summarizer: SummarizationAgent,
                 default_llm_model: str = None): # For main chat responses
        """
        Initializes the ConversationOrchestrator.

        Args:
            mmu (MemoryManagementUnit): Instance of the memory management unit.
            lsw (LLMServiceWrapper): Instance of the LLM service wrapper.
            knowledge_retriever (KnowledgeRetrieverAgent): Instance of the knowledge retriever agent.
            summarizer (SummarizationAgent): Instance of the summarization agent.
            default_llm_model (str, optional): Default LLM model to use for chat responses.
                                               If None, LSW's default_chat_model is used.
        """
        if not all(isinstance(arg, expected_type) for arg, expected_type in [
            (mmu, MemoryManagementUnit),
            (lsw, LLMServiceWrapper),
            (knowledge_retriever, KnowledgeRetrieverAgent),
            (summarizer, SummarizationAgent)
        ]):
            raise TypeError("Invalid type for one or more arguments during CO initialization.")

        self.mmu = mmu
        self.lsw = lsw
        self.knowledge_retriever = knowledge_retriever
        self.summarizer = summarizer
        self.default_llm_model = default_llm_model if default_llm_model else self.lsw.default_chat_model
        self.active_conversation_id = None # <--- ADDED for STM management
        print("ConversationOrchestrator initialized.")
        print(f"  Default LLM model for responses: {self.default_llm_model}")

    def _build_prompt_with_context(self,
                                   conversation_id: str,
                                   user_query: str,
                                   retrieved_knowledge: dict,
                                   max_context_words_approx: int = DEFAULT_MAX_CONTEXT_TOKENS_APPROX // 5 # Rough estimate: 1 token ~ 0.75 words, so 5 words ~ 3-4 tokens
                                  ) -> str:
        """
        Constructs the full prompt to be sent to the LLM, incorporating:
        - System instructions
        - Short-Term Memory (recent conversation history)
        - Retrieved knowledge from LTM (vector store, SKB facts)
        - The current user query
        Manages context length to avoid exceeding LLM limits.
        """
        
        # 1. System Prompt / Instructions
        # This needs careful crafting and iteration!
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
        # For now, we'll format the prompt as a single string for LSW's generate_chat_completion with system/user roles.
        # Later, we might use LSW's chat messages format more directly if needed.

        # 2. Short-Term Memory (STM)
        stm_history_turns = self.mmu.get_stm_history() # List of {"role": ..., "content": ...}
        
        # 3. Retrieved Knowledge Context (LTM)
        # We need to format and potentially condense this.
        context_parts = []
        
        # Add SKB facts
        if retrieved_knowledge.get("skb_fact_results"):
            context_parts.append("Retrieved Facts from Knowledge Base:")
            for fact in retrieved_knowledge["skb_fact_results"][:3]: # Limit facts displayed
                context_parts.append(f"  - Fact: {fact.get('subject')} {fact.get('predicate')} {fact.get('object')}.")
        
        # Add Vector Store results
        if retrieved_knowledge.get("vector_results"):
            context_parts.append("\nRetrieved Similar Information from Past Interactions/Documents:")
            for res in retrieved_knowledge["vector_results"][:DEFAULT_TOP_K_VECTOR_SEARCH]: # Already limited by top_k in search
                # For very long text_chunks, we might want to summarize them here using self.summarizer
                # For now, just take the chunk.
                context_parts.append(f"  - From source (metadata: {res.get('metadata', {})}): \"{res.get('text_chunk', '')}\"")
        
        retrieved_knowledge_str = "\n".join(context_parts)

        # 4. Construct the prompt messages list
        # This is a simplified approach. A more robust context manager would count tokens.
        # For now, we'll use word counts as a rough proxy for context length.
        
        messages = [{"role": "system", "content": system_message_content}]
        
        current_word_count = len(system_message_content.split())
        
        # Add retrieved knowledge if it fits (prioritize it)
        if retrieved_knowledge_str:
            knowledge_word_count = len(retrieved_knowledge_str.split())
            if current_word_count + knowledge_word_count < max_context_words_approx:
                # Using a "user" role for context is a common hack if the LLM doesn't have a dedicated context slot
                # or if we want the LLM to treat it as input it should pay close attention to.
                # A "system" message could also work.
                # Better: Some models support a specific format for RAG context.
                # For now, let's try embedding it naturally.
                # We'll build a context block to prepend to the user query.
                pass # Will be prepended later to user message if it fits.
            else:
                print(f"  CO: Warning - Retrieved knowledge too long ({knowledge_word_count} words), might be truncated or omitted. Max approx words: {max_context_words_approx}")
                # Potentially summarize retrieved_knowledge_str here if too long
                # For now, we might just not include it or include a part.

        # Add STM history, newest first, until context window is nearly full
        # (Actually, standard is oldest relevant STM first, then newer ones)
        temp_history_str_parts = []
        for turn in reversed(stm_history_turns[:-1]): # Exclude current user_query turn from STM history part
            turn_str = f"{turn['role'].capitalize()}: {turn['content']}"
            turn_word_count = len(turn_str.split())
            if current_word_count + knowledge_word_count + turn_word_count < max_context_words_approx:
                temp_history_str_parts.insert(0, turn_str) # Insert at beginning to maintain order
                current_word_count += turn_word_count
            else:
                break # Stop adding history if we are approaching the limit
        
        full_user_prompt_content = ""
        if temp_history_str_parts:
            full_user_prompt_content += "Relevant Conversation History:\n" + "\n".join(temp_history_str_parts) + "\n\n"
            
        if retrieved_knowledge_str and (len(full_user_prompt_content.split()) + len(retrieved_knowledge_str.split()) < max_context_words_approx):
            full_user_prompt_content += "RETRIEVED KNOWLEDGE CONTEXT:\n" + retrieved_knowledge_str + "\n\n"
        
        full_user_prompt_content += f"User Query: {user_query}"
        
        messages.append({"role": "user", "content": full_user_prompt_content})
        
        # print(f"  CO: Final constructed messages for LLM: {messages}")
        # print(f"  CO: Approx words in prompt: {current_word_count + len(user_query.split()) + knowledge_word_count}")
        return messages


    def handle_user_message(self, user_message: str, conversation_id: str = None) -> str or None:
        if not user_message.strip():
            return "Please say something!"

        # --- STM Management for conversation session ---
        is_new_conversation_session = False
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            print(f"CO: New conversation started (no ID provided), ID: {conversation_id}")
            is_new_conversation_session = True
        elif self.active_conversation_id != conversation_id:
            print(f"CO: Switched to/started new conversation ID: {conversation_id} (was {self.active_conversation_id})")
            is_new_conversation_session = True
        
        self.active_conversation_id = conversation_id 

        if is_new_conversation_session:
            self.mmu.clear_stm()
            print(f"  CO: STM cleared for new conversation session '{conversation_id}'.")
        # --- End STM Management ---

        print(f"\nCO: Handling user message for conversation '{conversation_id}': '{user_message}'")
        self.mmu.add_stm_turn(role="user", content=user_message)

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
        # It's crucial that get_ltm_conversation_history reflects transactions committed by previous turns.
        # SQLite default isolation should handle this per call.
        current_ltm_history_len_before_log = len(self.mmu.get_ltm_conversation_history(conversation_id))
        user_turn_log_seq_id = current_ltm_history_len_before_log + 1
        assistant_turn_log_seq_id = user_turn_log_seq_id + 1 # This will be the ID for the assistant's turn
        
        print(f"    LTM logging: User turn seq ID: {user_turn_log_seq_id} for '{user_message[:30]}...'")
        user_log_success = self.mmu.log_ltm_interaction( # Log user turn
            conversation_id=conversation_id,
            turn_sequence_id=user_turn_log_seq_id,
            role="user",
            content=user_message,
        )
        if not user_log_success:
            print(f"    LTM logging: FAILED to log user turn for seq ID {user_turn_log_seq_id}")
            # Decide how to handle this. For now, we'll proceed to log assistant.

        print(f"    LTM logging: Assistant turn seq ID: {assistant_turn_log_seq_id} for '{assistant_response_text[:30]}...'")
        self.mmu.log_ltm_interaction( # Log assistant turn
            conversation_id=conversation_id,
            turn_sequence_id=assistant_turn_log_seq_id,
            role="assistant", # Role should be 'assistant' even if content is an error message
            content=assistant_response_text,
            llm_model_used=self.default_llm_model,
            metadata={
                "retrieved_vector_ids": [res['id'] for res in retrieved_knowledge.get("vector_results",[]) if 'id' in res],
                "retrieved_fact_ids": [fact['fact_id'] for fact in retrieved_knowledge.get("skb_fact_results",[]) if 'fact_id' in fact],
                "llm_call_failed": assistant_response_text is None # Track if the LSW call itself failed
            }
        )
        # --- End LTM Logging ---

        return assistant_response_text

# --- Test Block ---
if __name__ == "__main__":
    print("--- Testing ConversationOrchestrator ---")

    # --- Setup Test Environment (MMU, LSW, Agents) ---
    # Define paths for test databases/stores for CO's MMU
    test_co_ltm_sqlite_db_path = 'test_co_ltm_sqlite.db'
    test_co_ltm_chroma_dir = 'test_co_ltm_chroma'
    test_co_mtm_db_path = 'test_co_mtm_store.json' # CO's MTM

    # Cleanup previous test files for CO
    if os.path.exists(test_co_mtm_db_path): os.remove(test_co_mtm_db_path)
    if os.path.exists(test_co_ltm_sqlite_db_path): os.remove(test_co_ltm_sqlite_db_path)
    import shutil
    if os.path.exists(test_co_ltm_chroma_dir): shutil.rmtree(test_co_ltm_chroma_dir)

    test_mmu = None
    test_lsw = None
    knowledge_retriever = None
    summarizer = None
    orchestrator = None

    try:
        print("\nInitializing MMU for CO test...")
        test_mmu = MemoryManagementUnit(
            ltm_sqlite_db_path=test_co_ltm_sqlite_db_path,
            ltm_chroma_persist_dir=test_co_ltm_chroma_dir
        )

        print("\nInitializing LSW for CO test...")
        # Ensure Ollama is running and "gemma3:1b-it-fp16" (or your chosen default) is pulled
        test_lsw = LLMServiceWrapper(default_chat_model="gemma3:1b-it-fp16") 
        if not test_lsw.client: raise Exception("Ollama client in LSW failed to initialize.")

        print("\nInitializing Agents for CO test...")
        knowledge_retriever = KnowledgeRetrieverAgent(mmu=test_mmu)
        summarizer = SummarizationAgent(lsw=test_lsw) # Uses LSW's default chat model

        print("\nInitializing ConversationOrchestrator...")
        orchestrator = ConversationOrchestrator(
            mmu=test_mmu,
            lsw=test_lsw,
            knowledge_retriever=knowledge_retriever,
            summarizer=summarizer
        )
    except Exception as e:
        print(f"FATAL: Error during CO test environment setup: {e}")
        print("Aborting CO tests.")
        exit()
    
    # --- Pre-populate LTM with some data for testing RAG ---
    print("\nPre-populating LTM for RAG test...")
    test_mmu.store_ltm_fact("The Conference", "location", "San Diego")
    test_mmu.store_ltm_fact("The Conference", "date", "October 26th")
    # Add to vector store only if LTM's embedding function is available
    ltm_vector_store_ready_co_test = False
    if hasattr(test_mmu.ltm, 'vector_store') and test_mmu.ltm.vector_store and \
       hasattr(test_mmu.ltm.vector_store, 'collection') and test_mmu.ltm.vector_store.collection and \
       hasattr(test_mmu.ltm.vector_store.collection, '_embedding_function') and \
       test_mmu.ltm.vector_store.collection._embedding_function is not None:
        ltm_vector_store_ready_co_test = True

    if ltm_vector_store_ready_co_test:
        test_mmu.add_document_to_ltm_vector_store(
            text_chunk="The upcoming AI conference will be held in San Diego. Key topics include LLMs and ethics.",
            metadata={"source": "conference_brochure", "topic": "AI Conference"},
            doc_id="conf_doc_1"
        )
        test_mmu.add_document_to_ltm_vector_store(
            text_chunk="Remember that the project deadline for 'Project Starlight' is November 15th.",
            metadata={"source": "internal_memo", "topic": "project_starlight"},
            doc_id="starlight_memo_1"
        )
        import time
        time.sleep(1) # Give Chroma a moment
        print("LTM populated with a fact and vector documents.")
    else:
        print("LTM populated with a fact. Vector store population skipped as embedding fn seems unavailable.")


    # --- Test Conversation Flow ---
    conversation_test_id = "test_conv_123"

    print(f"\n--- Test 1: User asks about the conference (expect RAG) ---")
    response1 = orchestrator.handle_user_message(
        user_message="Where is the AI conference being held?",
        conversation_id=conversation_test_id
    )
    print(f"Chatbot Response 1: {response1}")

    print(f"\n--- Test 2: User asks a follow-up (STM context should help) ---")
    response2 = orchestrator.handle_user_message(
        user_message="And what is its date?",
        conversation_id=conversation_test_id 
    )
    print(f"Chatbot Response 2: {response2}")
    
    print(f"\n--- Test 3: User asks about something not in LTM (general knowledge) ---")
    response3 = orchestrator.handle_user_message(
        user_message="What is the speed of light?",
        conversation_id=conversation_test_id
    )
    print(f"Chatbot Response 3: {response3}")

    print(f"\n--- Test 4: Start a new conversation (STM should be cleared for it) ---")
    new_convo_id = str(uuid.uuid4())
    response4 = orchestrator.handle_user_message(
        user_message="Tell me about Project Starlight.",
        conversation_id=new_convo_id # CO should recognize this as new
    )
    print(f"Chatbot Response 4 (New Convo {new_convo_id}): {response4}")
    # Check if STM for the new convo is fresh
    print(f"STM for new convo '{new_convo_id}': {orchestrator.mmu.get_stm_history()}")


    print("\n--- Checking LTM Contents After Conversation ---")
    history_conv_test_id = test_mmu.get_ltm_conversation_history(conversation_test_id)
    print(f"LTM history for '{conversation_test_id}' has {len(history_conv_test_id)} turns.")
    # for turn in history_conv_test_id:
    #     print(f"  - {turn['role']}: {turn['content'][:60]}...")

    history_new_convo_id = test_mmu.get_ltm_conversation_history(new_convo_id)
    print(f"LTM history for '{new_convo_id}' has {len(history_new_convo_id)} turns.")


    print("\nConversationOrchestrator tests finished.")

    # --- Final Cleanup of CO test files ---
    print("\nAttempting final cleanup of CO test files...")
    # Explicitly delete orchestrator and its components to help release resources
    del orchestrator
    del knowledge_retriever
    del summarizer
    if hasattr(test_lsw, 'client') and test_lsw.client: # LSW doesn't have a close for client
        pass 
    del test_lsw
    if hasattr(test_mmu, 'mtm') and test_mmu.mtm.is_persistent and hasattr(test_mmu.mtm, 'db') and test_mmu.mtm.db:
        test_mmu.mtm.db.close()
    del test_mmu
    
    import gc
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
    print("CO test file cleanup attempt finished.")