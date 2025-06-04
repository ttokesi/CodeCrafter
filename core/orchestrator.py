# offline_chat_bot/core/orchestrator.py
import uuid 
import os      
import shutil  
import time
import gc      
import json
from datetime import datetime, timezone


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

AVAILABLE_TOOLS = [
    {
        "tool_name": "knowledge_search",
        "description": "Searches your long-term memory (knowledge base of facts and records of past conversations) for information relevant to a specific query. Use this when you need to answer a user's question, find specific details from the past, or recall previously learned information. Do NOT use this for general knowledge questions if the information is unlikely to be in your specific memory.",
        "argument_schema": {
            "type": "object",
            "properties": {
                "query_text": {
                    "type": "string",
                    "description": "The specific question or search query to look up in the knowledge base. Be precise and use keywords from the user's request if applicable."
                }
            },
            "required": ["query_text"]
        }
    },
    {
        "tool_name": "text_summarizer",
        "description": "Generates a concise summary of a given piece of text. Use this tool if you have a long piece of text (e.g., a retrieved document or a long segment of conversation) that needs to be shortened while retaining the main points, or if the user explicitly asks for a summary.",
        "argument_schema": {
            "type": "object",
            "properties": {
                "text_to_summarize": {
                    "type": "string",
                    "description": "The text content that needs to be summarized."
                },
                "target_token_length": {
                    "type": "integer",
                    "description": "Optional. Suggest an approximate desired length of the summary in tokens (e.g., 50, 100, 150). If unsure, omit this."
                }
            },
            "required": ["text_to_summarize"]
        }
    },
    {
        "tool_name": "fact_extractor",
        "description": "Identifies and extracts new, explicit factual statements made by the user about themselves, their preferences, or specific entities they mention. Use this when the user provides a new piece of information that seems important to remember for future interactions (e.g., 'My name is John,' 'My favorite color is blue,' 'Project Alpha is due next week'). Do NOT use this for questions the user asks or for your own statements.",
        "argument_schema": {
            "type": "object",
            "properties": {
                "text_to_process": {
                    "type": "string",
                    "description": "The user's statement from which to extract new facts."
                }
            },
            "required": ["text_to_process"]
        }
    }
]
# --- End Definition of Available Tools ---

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
        #lsw_config = config.get('lsw', {}) # For LSW default chat model as fallback

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
        
        self.react_system_prompt_template = co_config.get('react_system_prompt_template', "Warning: ReAct system prompt template not found in config!")

        self.active_conversation_id = None
        
        print("ConversationOrchestrator initialized (using configuration).")
        print(f"  CO - Default LLM model for responses: {self.default_llm_model}")
        print(f"  CO - Target max prompt tokens: {self.target_max_prompt_tokens}")
        print(f"  CO - Min tokens for VS learn: {self.min_tokens_for_vs_learn}")
        print(f"  CO - Manage STM in prompt: {self.manage_stm_within_prompt}")
        print(f"  CO - STM Condensation Strategy: {self.stm_condensation_strategy}")
        print(f"  CO - STM Token Budget Ratio: {self.target_stm_tokens_budget_ratio}")
        print(f"  CO - STM Summary Target Tokens: {self.stm_summary_target_tokens}")
        print(f"  CO - Loaded ReAct System Prompt Template: {'Yes' if 'Warning:' not in self.react_system_prompt_template else 'NO - Check config!'}")
        
    def _format_available_tools_for_prompt(self) -> str:
        """
        Formats the AVAILABLE_TOOLS list into a string suitable for inclusion
        in the system prompt.
        """
        if not AVAILABLE_TOOLS: # Global constant
            return "No tools are currently available."

        tool_descriptions = []
        for i, tool_spec in enumerate(AVAILABLE_TOOLS):
            tool_info = f"{i+1}. Tool Name: {tool_spec['tool_name']}\n"
            tool_info += f"   Description: {tool_spec['description']}\n"
            tool_info += f"   Argument Schema:\n```json\n{json.dumps(tool_spec['argument_schema'], indent=4)}\n```\n---"
            tool_descriptions.append(tool_info)
        
        return "\n".join(tool_descriptions)

    def _build_prompt_with_context(self,
                                   user_query: str,
                                   retrieved_knowledge: dict,
                                   observation: str = None 
                                  ) -> list:
        #print("  CO: Building prompt with context...")
        target_ollama_model_for_tokens = self.default_llm_model 
        summarization_llm_model = self.summarizer.default_model_name

        tools_description_str = self._format_available_tools_for_prompt()

        # 1. System Prompt
        system_message_content = self.react_system_prompt_template.format(
            available_tools_description_string=tools_description_str
        )
        system_tokens = count_tokens(system_message_content, target_ollama_model_for_tokens)
        current_total_tokens = system_tokens
        print(f"    Tokens - System message (ReAct): {system_tokens}")
        messages = [{"role": "system", "content": system_message_content}]

        knowledge_context_str_parts = []
        
        # --- ADD OBSERVATION TO CONTEXT if present ---
        if observation:
            observation_str = f"OBSERVATION:\n{observation}"
            observation_tokens = count_tokens(observation_str, target_ollama_model_for_tokens)
            # Add observation before LTM, as it's more immediate context from tool use
            if current_total_tokens + observation_tokens < self.target_max_prompt_tokens:
                knowledge_context_str_parts.append(observation_str)
                current_total_tokens += observation_tokens
            else:
                print(f"    WARNING: Observation was too long to fit in prompt ({observation_tokens} tokens).")
        # --- END OBSERVATION ---

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
                print(f"    STM Condensation: Attempting 'summarize' strategy.")
                print(f"      DEBUG: Initial stm_history_turns_original: {json.dumps(stm_history_turns_original, indent=2)}") # DEBUG PRINT
                
                managed_stm_turns_for_prompt = [] 
                temp_turns_to_keep_verbatim = []
                temp_turns_to_summarize = [] 
                accumulated_verbatim_tokens = 0

                # Only operate on actual historical turns
                if stm_history_turns_original:
                    #print(f"      DEBUG: Processing {len(stm_history_turns_original)} historical STM turns.") # DEBUG PRINT
                    # Iterate from newest to oldest to decide what fits verbatim
                    for i in range(len(stm_history_turns_original) - 1, -1, -1):
                        turn_to_check = stm_history_turns_original[i]
                        turn_str_temp = f"{turn_to_check['role'].capitalize()}: {turn_to_check['content']}"
                        turn_tokens_temp = count_tokens(turn_str_temp, target_ollama_model_for_tokens)
                        
                        #print(f"        DEBUG: Checking turn (idx {i}): '{turn_str_temp[:50]}...' ({turn_tokens_temp} tokens)") # DEBUG PRINT
                        #print(f"        DEBUG: Current verbatim tokens: {accumulated_verbatim_tokens}, STM budget: {stm_token_budget}") # DEBUG PRINT

                        if (accumulated_verbatim_tokens + turn_tokens_temp) <= stm_token_budget:
                            temp_turns_to_keep_verbatim.insert(0, turn_to_check) 
                            accumulated_verbatim_tokens += turn_tokens_temp
                            #print(f"          DEBUG: Added to verbatim. New verbatim tokens: {accumulated_verbatim_tokens}") # DEBUG PRINT
                        else:
                            #print(f"          DEBUG: Turn too large. Moving this and older to 'to_summarize'.") # DEBUG PRINT
                            temp_turns_to_summarize = list(stm_history_turns_original[0 : i + 1])
                            break 
                    
                    if not temp_turns_to_summarize and temp_turns_to_keep_verbatim: # All historical turns fit verbatim
                        managed_stm_turns_for_prompt = list(temp_turns_to_keep_verbatim) # Assign if all fit
                
                # --- At this point:
                # temp_turns_to_keep_verbatim contains newest historical turns that fit the budget.
                # temp_turns_to_summarize contains oldest historical turns that did not fit verbatim.
                # If stm_history_turns_original was empty, both lists are empty.
                
                print(f"      STM Summarize: Kept {len(temp_turns_to_keep_verbatim)} recent turn(s) verbatim ({accumulated_verbatim_tokens} tokens).")


                # --- Summarization block: ONLY execute if temp_turns_to_summarize was populated by historical processing ---
                if temp_turns_to_summarize: # This list is ONLY populated from stm_history_turns_original processing block
                    print(f"      STM Summarize: {len(temp_turns_to_summarize)} older turn(s) are candidates for summarization.")
                    # ... (the rest of your summarization logic using temp_turns_to_summarize, text_block_to_summarize, etc.)
                    # ... (assign to managed_stm_turns_for_prompt based on summary success/failure)
                    # Make sure that if summarization is skipped/fails, managed_stm_turns_for_prompt is set to temp_turns_to_keep_verbatim
                    # Example from previous correct logic:
                    text_block_to_summarize_parts = [f"{t['role'].capitalize()}: {t['content']}" for t in temp_turns_to_summarize]
                    text_block_to_summarize = "\n".join(text_block_to_summarize_parts)
                    original_block_tokens = count_tokens(text_block_to_summarize, target_ollama_model_for_tokens)
                    print(f"      STM Summarize: Original token count of block to summarize: {original_block_tokens}")

                    remaining_budget_for_summary = stm_token_budget - accumulated_verbatim_tokens

                    if original_block_tokens > self.stm_summary_target_tokens + 50 and \
                    remaining_budget_for_summary > self.stm_summary_target_tokens // 2:
                        print(f"      STM Summarize: Calling SummarizationAgent for older STM turns (target tokens: {self.stm_summary_target_tokens}).")
                        summary_of_older_stm = self.summarizer.summarize_text(
                            text_to_summarize=text_block_to_summarize,
                            model_name=self.summarizer.default_model_name, 
                            max_summary_length=self.stm_summary_target_tokens,
                            temperature=0.3 
                        )
                        if summary_of_older_stm:
                            summary_tokens = count_tokens(summary_of_older_stm, target_ollama_model_for_tokens)
                            print(f"      STM Summarize: Generated summary ({summary_tokens} tokens): \"{summary_of_older_stm[:100].strip()}...\"")
                            if summary_tokens <= remaining_budget_for_summary:
                                summary_turn = {"role": "system", "content": f"[Summary of earlier conversation]: {summary_of_older_stm}", "timestamp": datetime.now(timezone.utc).isoformat()}
                                managed_stm_turns_for_prompt = [summary_turn] + temp_turns_to_keep_verbatim 
                            else:
                                print(f"      STM Summarize: Generated summary ({summary_tokens} tokens) is too long for remaining budget ({remaining_budget_for_summary}). Discarding summary, keeping verbatim.")
                                managed_stm_turns_for_prompt = list(temp_turns_to_keep_verbatim) 
                        else:
                            print(f"      STM Summarize: Summarization failed. Discarding older turns, keeping verbatim.")
                            managed_stm_turns_for_prompt = list(temp_turns_to_keep_verbatim) 
                    else: 
                        print(f"      STM Summarize: Block of older turns not summarized (too small or budget insufficient). Keeping verbatim turns only.")
                        managed_stm_turns_for_prompt = list(temp_turns_to_keep_verbatim)
                
                elif not stm_history_turns_original: # No historical STM at all
                    print(f"      STM Summarize: No historical STM to process.")
                    managed_stm_turns_for_prompt = [] 
                else: # All historical STM fit verbatim (temp_turns_to_summarize is empty, but stm_history_turns_original was not)
                    print(f"      STM Summarize: All historical STM turns fit verbatim, no summarization needed.")
                    managed_stm_turns_for_prompt = list(temp_turns_to_keep_verbatim)


                current_stm_tokens_in_prompt = 0
                for turn in managed_stm_turns_for_prompt:
                    current_stm_tokens_in_prompt += count_tokens(f"{turn['role'].capitalize()}: {turn['content']}", target_ollama_model_for_tokens)
                
                print(f"    Tokens - STM after summarization attempt: {current_stm_tokens_in_prompt} from {len(managed_stm_turns_for_prompt)} effective turns.")
            
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
        llm_messages = None # Initialize
        try:
            #print("CO_DEBUG: About to call _build_prompt_with_context...") # New debug
            llm_messages = self._build_prompt_with_context(
                user_query=user_message, 
                retrieved_knowledge=retrieved_knowledge
                # observation is not passed yet
            )
            #print("CO_DEBUG: _build_prompt_with_context call successful.") # New debug
        except Exception as build_prompt_error:
            print(f"CO_CRITICAL_ERROR: Exception during _build_prompt_with_context!")
            print(f"  Error Type: {type(build_prompt_error)}")
            print(f"  Error Args: {build_prompt_error.args}")
            print(f"  Error String: {str(build_prompt_error)}")
            import traceback
            traceback.print_exc()
            # If prompt building fails, we can't proceed to LSW call for main response
            yield "I'm sorry, I had trouble preparing my thoughts for a response."
            # Log this critical failure (simplified logging for now)
            # ... (logging logic would go here if we want to record this type of error) ...
            return # Exit the handle_user_message generator

        # If llm_messages is None after try-except (though it shouldn't be if no error, but as a safeguard)
        if llm_messages is None:
            print("CO_ERROR: llm_messages is None after _build_prompt_with_context, even without explicit error.")
            yield "I seem to be having a problem formulating a response."
            return

        #print(f"CO_DEBUG: Messages being sent to LSW for main response: {json.dumps(llm_messages, indent=2)}")
        
        assistant_response_stream = self.lsw.generate_chat_completion( # This call is for the main response
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

        # --- MODIFIED STREAM HANDLING FOR TOOL CALL DETECTION ---
        accumulated_response_for_ltm = ""
        leading_buffer = "" # Buffer to check for TOOL_CALL:
        # Max buffer size before deciding if it's a direct answer or tool call.
        # Needs to be large enough to contain "TOOL_CALL:\n```json\n"
        # "TOOL_CALL:" is 10 chars. Let's say ~50 chars to be safe.
        MAX_LEAD_BUFFER_SIZE = 64 
        
        is_tool_call_detected = False
        json_block_started = False
        json_content_buffer = ""
        
        # This will be the generator we return, which wraps the logic
        def response_processor_generator():
            nonlocal accumulated_response_for_ltm, leading_buffer, is_tool_call_detected, json_block_started, json_content_buffer
            
            TOOL_CALL_PREFIX = "TOOL_CALL:"
            JSON_BLOCK_START = "```json"
            JSON_BLOCK_END = "```"

            try:
                for chunk in assistant_response_stream:
                    if not chunk: # Skip empty chunks if any
                        continue

                    accumulated_response_for_ltm += chunk # Always accumulate for LTM logging

                    if is_tool_call_detected:
                        # We are in tool call accumulation mode
                        json_content_buffer += chunk
                        if JSON_BLOCK_END in json_content_buffer: # A good signal to try parsing
                            # --- Call a new helper method to parse json_content_buffer ---
                            # This helper returns (parsed_tool_call_dict, error_message_if_any)
                            parsed_data, error_msg = self._parse_tool_call_json(json_content_buffer)
                            if parsed_data:
                                tool_name = parsed_data.get("tool_name")
                                tool_args = parsed_data.get("arguments", {})
                                print(f"CO_SUCCESS: Parsed Tool Call - Name: {tool_name}, Args: {tool_args}")
                                yield f"[Debug: Tool call parsed: {tool_name} with args {tool_args}. No user output.]"
                            else:
                                yield error_msg # Yield the error message from parsing
                            return # Exit generator as tool call attempt is processed
                    else: # Not yet detected as a tool call
                        leading_buffer += chunk
                        if TOOL_CALL_PREFIX in leading_buffer:
                            is_tool_call_detected = True
                            print(f"CO_DEBUG: '{TOOL_CALL_PREFIX}' detected in stream.")
                            json_content_buffer = leading_buffer[leading_buffer.find(TOOL_CALL_PREFIX) + len(TOOL_CALL_PREFIX):]
                            leading_buffer = ""
                            # Check if the *initial* part already contains the end block
                            if JSON_BLOCK_END in json_content_buffer:
                                parsed_data, error_msg = self._parse_tool_call_json(json_content_buffer)
                                # ... (handle parsed_data/error_msg and return as above) ...
                                if parsed_data:
                                    tool_name = parsed_data.get("tool_name"); tool_args = parsed_data.get("arguments", {})
                                    print(f"CO_SUCCESS (Immediate): Parsed Tool Call - Name: {tool_name}, Args: {tool_args}")
                                    yield f"[Debug: Immediate Tool call parsed: {tool_name}. No user output.]"
                                else: yield error_msg
                                return
                        elif len(leading_buffer) >= MAX_LEAD_BUFFER_SIZE or "\n" in chunk:
                            # ... (yield leading_buffer, clear it, and then yield current chunk if it wasn't part of buffer flush) ...
                            # This part for direct answer needs to be careful not to double-yield
                            print("CO_DEBUG: No TOOL_CALL prefix detected in lead buffer. Assuming direct answer.")
                            yield leading_buffer 
                            leading_buffer = "" # Buffer has been yielded
                            # The current chunk *was* added to leading_buffer. So if we yielded leading_buffer,
                            # we don't need to yield chunk separately here for this iteration.
                            # The next iteration will handle the next chunk IF is_tool_call_detected is still false.
                    
                    # If loop continues after buffer flush (is_tool_call_detected is False and leading_buffer is empty)
                    if not is_tool_call_detected and not leading_buffer:
                        yield chunk
                
                # AFTER THE LOOP (stream ended)
                if is_tool_call_detected and json_content_buffer and JSON_BLOCK_END not in json_content_buffer:
                    # Stream ended, we were in tool call mode, but no closing ``` was found.
                    # Try to parse what we have in json_content_buffer anyway.
                    print("CO_DEBUG_JSON: Stream ended while accumulating tool call. Attempting parse of incomplete buffer.")
                    parsed_data, error_msg = self._parse_tool_call_json(json_content_buffer)
                    if parsed_data:
                        tool_name = parsed_data.get("tool_name"); tool_args = parsed_data.get("arguments", {})
                        print(f"CO_SUCCESS (End of Stream): Parsed Tool Call - Name: {tool_name}, Args: {tool_args}")
                        yield f"[Debug: End of Stream Tool call parsed: {tool_name}. No user output.]"
                    else: yield error_msg
                    # No return here, let the generator finish.

                elif not is_tool_call_detected and leading_buffer: # Flush any remaining leading_buffer
                    print("CO_DEBUG: Flushing remaining leading_buffer as direct answer at end of stream.")
                    yield leading_buffer

            except Exception as e_stream_consume:
                print(f"  CO: Error consuming LLM stream (in response_processor_generator). Type: {type(e_stream_consume)}, Error: {e_stream_consume}")
                import traceback
                traceback.print_exc()
                yield "Sorry, there was an error while I was generating my response."


        # Immediately call the generator and return its iterator
        # The actual LTM logging will be moved after this loop in handle_user_message
        processed_stream = response_processor_generator()
        
        # Consume the processed_stream to build the final response for LTM
        # and yield chunks to the CIL.
        # If it was a tool call, the generator might yield a debug message and then stop.
        final_bot_output_chunks = []
        for processed_chunk in processed_stream:
            final_bot_output_chunks.append(processed_chunk)
            yield processed_chunk # Yield to CIL
        
        # Update accumulated_response_for_ltm based on what was actually yielded or processed
        # If a tool call was detected and handled, final_bot_output_chunks might just be a debug message.
        # The original accumulated_response_for_ltm has the raw LLM output.
        # For LTM, we want to log what the LLM *attempted* (the raw accumulated_response_for_ltm),
        # and then separately log the tool call details or the final user-facing message.

        # --- LTM Logging (Moved to after stream consumption/processing) ---
        print(f"  CO: Preparing to log to LTM for conversation '{self.active_conversation_id}'.")
        # Log the raw user turn
        # ... (existing user turn logging logic) ...
        current_ltm_history_len = len(self.mmu.get_ltm_conversation_history(self.active_conversation_id)) # Recalculate
        user_turn_seq_id = current_ltm_history_len # If user turn not logged yet
        # Check if user turn was already logged for this interaction.
        # This needs careful handling of turn sequence.
        # Let's assume it's simpler: log user turn, then assistant's response/action.

        # If first turn in LTM log is user, then current_ltm_history_len is okay for user turn seq.
        # If LTM is empty, user_turn_seq_id will be 0. Sequence should be 1-based.
        
        # Simpler LTM logic:
        # The user message was added to STM already.
        # Now, log the user message to LTM.
        # Then, log what the assistant did (direct response or tool call attempt).

        # Ensure user turn is logged once per handle_user_message call
        # We can use a flag or check if STM has more than one turn where last is not assistant.
        # For now, this basic sequence should be okay for a single interaction.
        # The `turn_sequence_id` management needs to be robust across calls.
        
        # Let's retrieve the latest turn_sequence_id for this conversation_id from LTM to increment it.
        # This is a simplified approach for robust turn sequencing.
        ltm_history_for_seq = self.mmu.get_ltm_conversation_history(
            self.active_conversation_id, 
            limit=1, 
            order_desc=True # <--- ADD THIS
        )
        last_seq_id = 0
        if ltm_history_for_seq:
            last_seq_id = ltm_history_for_seq[0].get('turn_sequence_id', 0) # Get from the first item
        print(f"  CO_LTM_DEBUG: Last LTM seq_id found for this convo: {last_seq_id}")
        user_log_seq_id = last_seq_id + 1
        print(f"  CO_LTM_DEBUG: User turn to be logged with seq_id: {user_log_seq_id}")
        user_turn_ltm_entry_id = self.mmu.log_ltm_interaction(
            self.active_conversation_id, user_log_seq_id, "user", user_message
        )

        # Now log the assistant's full raw response (which might be a tool call)
        assistant_action_seq_id = user_log_seq_id + 1
        print(f"  CO_LTM_DEBUG: Assistant raw output to be logged with seq_id: {assistant_action_seq_id}")
        self.mmu.log_ltm_interaction(
            conversation_id=self.active_conversation_id,
            turn_sequence_id=assistant_action_seq_id,
            role="assistant_raw_output", # New role to distinguish raw LLM output
            content=accumulated_response_for_ltm, # This has the full, possibly tool-calling, LLM response
            llm_model_used=self.default_llm_model,
            metadata={
                "retrieved_vector_ids": [res['id'] for res in retrieved_knowledge.get("vector_results",[]) if 'id' in res],
                "retrieved_fact_ids": [fact['fact_id'] for fact in retrieved_knowledge.get("skb_fact_results",[]) if 'fact_id' in fact],
                "llm_streamed": True,
                "is_tool_call_detected_by_co": is_tool_call_detected # Log if CO thought it was a tool call
            }
        )
        print(f"  CO_LTM_DEBUG: Attempted to log user turn with seq_id {user_log_seq_id} (LTM_entry_id: {user_turn_ltm_entry_id}) and assistant_raw_output with seq_id {assistant_action_seq_id}.")

        # Add the *final user-facing message* (if any) or a tool status to STM
        # `"".join(final_bot_output_chunks)` is what was actually yielded to the user.
        final_stm_message_for_assistant = "".join(final_bot_output_chunks)
        if not final_stm_message_for_assistant and is_tool_call_detected:
            # If tool call was detected and nothing was yielded to user (e.g. only debug message)
            final_stm_message_for_assistant = f"[CO attempting tool call based on LLM output. Waiting for tool result.]"
        
        self.mmu.add_stm_turn(role="assistant", content=final_stm_message_for_assistant)


        # Vector Store Learning for user message (existing logic)
        if user_message and not is_likely_question:
            # ... (existing vector store learning logic, ensure user_turn_ltm_entry_id is correct)
            user_message_tokens = count_tokens(user_message, self.default_llm_model)
            if user_message_tokens >= self.min_tokens_for_vs_learn:
                print(f"  CO: User message is significant ({user_message_tokens} tokens). Storing to Vector Store.")
                doc_id_vs = f"user_stmt_{self.active_conversation_id}_{user_log_seq_id}"
                metadata_vs = {
                    "type": "user_statement", "conversation_id": self.active_conversation_id,
                    "turn_sequence_id": user_log_seq_id, # Use user's log sequence ID
                    "ltm_raw_log_entry_id": user_turn_ltm_entry_id if user_turn_ltm_entry_id else "unknown",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                vector_store_add_id = self.mmu.add_document_to_ltm_vector_store(
                    text_chunk=user_message, metadata=metadata_vs, doc_id=doc_id_vs
                )
                if vector_store_add_id: print(f"    CO: User statement stored in Vector Store with ID: {vector_store_add_id}")
                else: print(f"    CO: Failed to store user statement in Vector Store.")
            else:
                print(f"  CO: User message not stored in Vector Store (tokens: {user_message_tokens} < {self.min_tokens_for_vs_learn}).")
        # --- End Vector Store Learning for user message ---

    def _parse_tool_call_json(self, json_content_after_tool_call_prefix: str) -> tuple[dict | None, str | None]:
        JSON_BLOCK_START_TAG = "```json"
        JSON_BLOCK_END_TAG = "```"
        actual_json_str = None

        print(f"CO_PARSE_DEBUG: Attempting to parse (content after TOOL_CALL: prefix): '''{json_content_after_tool_call_prefix[:300]}...'''")
        
        buffer_to_parse_stripped = json_content_after_tool_call_prefix.strip()

        # Priority 1: Check for raw JSON object (starts with { and ends with })
        if buffer_to_parse_stripped.startswith("{") and buffer_to_parse_stripped.endswith("}"):
            print(f"CO_PARSE_DEBUG: Buffer starts with '{{' and ends with '}}'. Assuming it's a raw JSON object.")
            actual_json_str = buffer_to_parse_stripped 
            # No need for decoder.decode() and idx here if we assume the whole stripped buffer is the JSON

        # Priority 2: Check for markdown block if raw JSON wasn't identified
        if actual_json_str is None: 
            md_start_idx = json_content_after_tool_call_prefix.find(JSON_BLOCK_START_TAG) # Use original buffer for find
            if md_start_idx != -1:
                md_end_idx = json_content_after_tool_call_prefix.rfind(JSON_BLOCK_END_TAG, md_start_idx + len(JSON_BLOCK_START_TAG))
                if md_end_idx != -1: # md_end_idx > (md_start_idx + len(JSON_BLOCK_START_TAG) -1) implied by search start
                    actual_json_str = json_content_after_tool_call_prefix[md_start_idx + len(JSON_BLOCK_START_TAG) : md_end_idx].strip()
                    print(f"CO_PARSE_DEBUG: Extracted from markdown block: '''{actual_json_str}'''")
                else:
                    # Found ```json but no valid closing ```.
                    # Could the content after ```json be a raw JSON object itself?
                    possible_json_content = json_content_after_tool_call_prefix[md_start_idx + len(JSON_BLOCK_START_TAG):].strip()
                    if possible_json_content.startswith("{") and possible_json_content.endswith("}"):
                        actual_json_str = possible_json_content
                        print(f"CO_PARSE_DEBUG: Fallback - extracted raw JSON after '```json' (no closing '```' found): '''{actual_json_str}'''")
                    else:
                        print(f"CO_PARSE_DEBUG: Found '{JSON_BLOCK_START_TAG}' but no valid closing '{JSON_BLOCK_END_TAG}' and no clear raw JSON after. Buffer: '''{json_content_after_tool_call_prefix[:200]}...'''")
                        actual_json_str = None
            # else: No ```json found at all. actual_json_str remains None by this path.

        # If after all attempts, actual_json_str is still None and the buffer_to_parse_stripped looked like it started with {
        # (but maybe didn't end with }), we can do one last very speculative attempt with decoder.decode
        # This handles cases like { "foo": "bar" } followed by unexpected text from LLM.
        if actual_json_str is None and buffer_to_parse_stripped.startswith("{"):
            print(f"CO_PARSE_DEBUG: Last attempt - buffer started with '{{' but other methods failed. Trying json.JSONDecoder().decode.")
            decoder = json.JSONDecoder()
            try:
                parsed_obj, idx = decoder.decode(buffer_to_parse_stripped)
                print(f"CO_PARSE_DEBUG: Last attempt decoder.decode succeeded. idx = {idx}, type(idx) = {type(idx)}")
                if isinstance(idx, int): # Ensure idx is an integer
                    actual_json_str = buffer_to_parse_stripped[:idx]
                    print(f"CO_PARSE_DEBUG: Last attempt - Found potential raw JSON object: '''{actual_json_str}'''")
                else:
                    print(f"CO_PARSE_DEBUG: Last attempt decoder.decode returned non-integer idx. Cannot use.")
                    actual_json_str = None # Should not happen if decode doesn't raise error
            except json.JSONDecodeError:
                print(f"CO_PARSE_DEBUG: Last attempt decoder.decode failed with JSONDecodeError.")
                actual_json_str = None

        # Now, try to load 'actual_json_str' if it was populated by either method
        if actual_json_str:
            try:
                parsed_tool_call = json.loads(actual_json_str.strip()) # Strip again for safety
                if not isinstance(parsed_tool_call, dict):
                    return None, "[Error: Parsed tool call JSON is not a dictionary.]"
                tool_name = parsed_tool_call.get("tool_name")
                arguments = parsed_tool_call.get("arguments")
                if arguments is not None and not isinstance(arguments, dict):
                    return None, "[Error: 'arguments' field in tool call JSON is not a dictionary.]"

                if tool_name:
                    return parsed_tool_call, None
                else:
                    return None, "[Error: Parsed tool call JSON missing 'tool_name'.]"
            except json.JSONDecodeError as json_e:
                return None, f"[Error: Could not parse tool call JSON. Detail: {json_e}. Content: '''{actual_json_str[:100].strip()}...''']"
            except Exception as tool_parse_e:
                return None, f"[Error: Unexpected problem processing tool call JSON. Detail: {tool_parse_e}]"
        else:
            return None, "[Error: Malformed tool call structure from LLM - JSON content not found or clear.]"

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