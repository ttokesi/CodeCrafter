import os
import sys
import shutil 
import time   
import gc     # For test cleanup
import json   # For parsing LLM's structured output
import re

# Conditional import for MMU and LSW based on execution context
if __name__ == '__main__' and __package__ is None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from mmu.mmu_manager import MemoryManagementUnit
    from core.llm_service_wrapper import LLMServiceWrapper # For test setup
else:
    from mmu.mmu_manager import MemoryManagementUnit
    from .llm_service_wrapper import LLMServiceWrapper # Relative import for package use


class KnowledgeRetrieverAgent:
    """
    An agent responsible for retrieving relevant knowledge from Long-Term Memory (LTM),
    including the Vector Store and Structured Knowledge Base (SKB).
    """
    def __init__(self, mmu: MemoryManagementUnit):
        """
        Initializes the KnowledgeRetrieverAgent.

        Args:
            mmu (MemoryManagementUnit): An instance of the MemoryManagementUnit
                                        to access LTM components.
        """
        if not isinstance(mmu, MemoryManagementUnit):
            raise TypeError("KnowledgeRetrieverAgent requires an instance of MemoryManagementUnit.")
        self.mmu = mmu
        #print("KnowledgeRetrieverAgent initialized.")

    def search_knowledge(self,
                         query_text: str,
                         search_vector_store: bool = True,
                         search_skb_facts: bool = True,
                         top_k_vector: int = 3,
                         # Allow CO to pass an explicit user_id if we support multi-user in future
                         # For now, it's not used by KRA but good for API consistency.
                         user_id: str = "default_user", 
                         skb_subject: str = None,
                         skb_predicate: str = None,
                         skb_object: str = None) -> dict:
        results = {
            "vector_results": [],
            "skb_fact_results": []
        }
        
        if search_vector_store:
            # ... (vector search logic as before - no changes here) ...
            try:
                vector_search_results = self.mmu.semantic_search_ltm_vector_store(query_text=query_text, top_k=top_k_vector)
                if vector_search_results: results["vector_results"] = vector_search_results
            except Exception as e: print(f"    Error during vector store search: {e}")

        if search_skb_facts:
            found_skb_facts_dict = {} 

            # --- STRATEGY 1: Targeted S-P-O Search (if provided) ---
            if skb_subject or skb_predicate or skb_object:
                print(f"  KRA_DEBUG: Performing targeted SKB search: S='{skb_subject}', P='{skb_predicate}', O='{skb_object}'")
                try:
                    targeted_facts = self.mmu.get_ltm_facts(
                        subject=skb_subject, predicate=skb_predicate, object_value=skb_object
                    )
                    print(f"  KRA_DEBUG: Targeted search found {len(targeted_facts)} fact(s).")
                    for fact in targeted_facts:
                        if 'fact_id' in fact: found_skb_facts_dict[fact['fact_id']] = fact
                except Exception as e:
                    print(f"    Error during targeted SKB fact search: {e}")

            # --- STRATEGY 2: Pattern-Based Search for Common Questions (if no specific S-P-O above) ---
            # This runs if specific skb_subject/predicate/object were NOT provided, or can be additive.
            # For now, let's make it run if specific S,P,O are not set by the caller.
            # We might want to make this logic more sophisticated later, e.g. always try patterns.
            if query_text and not (skb_subject or skb_predicate or skb_object):
                print(f"  KRA_DEBUG: Attempting pattern-based SKB search for query: '{query_text}'")
                # Pattern 1: "What is my X?" -> search for subject "user X" or "user's X"
            match_my_x = re.search(r"what is my ([\w\s]+)\??", query_text.lower())
            if match_my_x:
                full_entity_match = match_my_x.group(1).strip()
                print(f"    KRA_DEBUG: Pattern 'What is my X?' - Full matched entity: '{full_entity_match}'")

                entities_to_query = []
                if " and " in full_entity_match:
                    print(f"    KRA_DEBUG: Detected 'and' in entity. Splitting: '{full_entity_match}'")
                    parts = [part.strip() for part in full_entity_match.split(" and ") if part.strip()]
                    entities_to_query.extend(parts)
                else:
                    entities_to_query.append(full_entity_match)
                
                print(f"    KRA_DEBUG: Entities to query SKB for: {entities_to_query}")

                for entity in entities_to_query: # Now loop through potentially multiple entities
                    print(f"    KRA_DEBUG: Processing split entity: '{entity}'") # Optional more debug
                    potential_subjects = [f"user {entity}", f"user's {entity}", f"user {entity.rstrip('s')}", f"user's {entity.rstrip('s')}"]
                    for potential_subject in potential_subjects:
                        print(f"    KRA_DEBUG: Pattern 'What is my X?' -> Querying SKB for subject LIKE '%{potential_subject}%'")
                        # Make sure your mmu.get_ltm_facts can handle LIKE if you use it, 
                        # or use exact match if your subjects are canonical.
                        # For now, let's assume exact match for "user <entity>" is better.
                        facts_pattern = self.mmu.get_ltm_facts(subject=potential_subject, predicate="is") # Often "is"
                        for fact in facts_pattern:
                            if 'fact_id' in fact: found_skb_facts_dict[fact['fact_id']] = fact
                        if found_skb_facts_dict and entity in full_entity_match: # If we found something for this part, maybe stop for this *part*
                            # This break is tricky with multiple entities. We want to find all parts.
                            # So, the break should probably be inside the potential_subjects loop if a match is found FOR THAT potential_subject
                            pass # Let it try all potential_subjects for the current entity part

                # Pattern 2: "Do you remember my X?" or "Do you know my X?"
                # -> search for subject "user X" or "user's X"
                match_remember_my_x = re.search(r"do you (?:remember|know) my ([\w\s]+)\??", query_text.lower())
                if not found_skb_facts_dict and match_remember_my_x: # Only if previous pattern didn't yield
                    entity = match_remember_my_x.group(1).strip()
                    potential_subjects = [f"user {entity}", f"user's {entity}", f"user {entity.rstrip('s')}", f"user's {entity.rstrip('s')}"]
                    for potential_subject in potential_subjects:
                        #print(f"    KRA_DEBUG: Pattern 'Do you remember my X?' -> Querying SKB for subject LIKE '%{potential_subject}%'")
                        facts_pattern = self.mmu.get_ltm_facts(subject=f"%{potential_subject}%", predicate="is")
                        for fact in facts_pattern:
                            if 'fact_id' in fact: found_skb_facts_dict[fact['fact_id']] = fact
                        if found_skb_facts_dict: break
                
                # Add more patterns here as needed for common question types...
                # Example: "Where is X?" -> search subject "X", predicate "location is" OR "is located in"

            # --- STRATEGY 3: General Keyword-Based Search (if previous strategies found little or as a fallback) ---
            # Run this if pattern search found nothing or to augment it.
            if query_text: # and not found_skb_facts_dict: # Optional: only run if patterns failed
                stop_words = ["what", "is", "my", "me", "i", "do", "you", "the", "a", "and", "tell", "about", "can", "does", "was", "are", "were", "of", "for", "on", "in", "at", "it"]
                punctuation_to_strip = str.maketrans('', '', '.,?!;:"()[]{}')
                raw_keywords = query_text.lower().split()
                keywords = []
                for kw_raw in raw_keywords:
                    kw_cleaned = kw_raw.translate(punctuation_to_strip)
                    if len(kw_cleaned) > 1 and kw_cleaned not in stop_words: # Allow 2-letter keywords if not stop words
                        keywords.append(kw_cleaned)
                
                if keywords: # Only print if there are actual keywords to search
                    #print(f"  KRA_DEBUG: Performing general keyword-based SKB search. Cleaned Keywords: {keywords}")
                    for keyword in set(keywords):
                        if not keyword: continue
                        # print(f"    KRA_DEBUG: Querying SKB with general keyword '{keyword}'...") # Can be verbose
                        try:
                            for field_to_search in ["subject", "predicate", "object_value"]:
                                query_params = {field_to_search: f"%{keyword}%"}
                                # print(f"      KRA_DEBUG: ...in {field_to_search} LIKE '%{keyword}%'")
                                facts_kw = self.mmu.get_ltm_facts(**query_params)
                                # if facts_kw: print(f"        KRA_DEBUG: Found {len(facts_kw)} in {field_to_search} for '{keyword}'.")
                                for fact in facts_kw:
                                    if 'fact_id' in fact: found_skb_facts_dict[fact['fact_id']] = fact
                        except Exception as e:
                            print(f"    Error during general keyword-based SKB fact search for '{keyword}': {e}")
            
            results["skb_fact_results"] = list(found_skb_facts_dict.values())
            #if results["skb_fact_results"]:
                #print(f"  KRA: Found {len(results['skb_fact_results'])} unique fact(s) from SKB based on query/params.")
            #else:
                #print(f"  KRA: No unique facts found in SKB for this query/params.")
        
        return results


class SummarizationAgent:
    """
    An agent responsible for generating concise summaries of text using an LLM.
    """
    def __init__(self, lsw: LLMServiceWrapper, default_model_name: str = None):
        """
        Initializes the SummarizationAgent.

        Args:
            lsw (LLMServiceWrapper): An instance of the LLMServiceWrapper to interact with LLMs.
            default_model_name (str, optional): The default Ollama model to use for summarization.
                                                If None, LSW's default chat model will be used.
        """
        if not isinstance(lsw, LLMServiceWrapper):
            raise TypeError("SummarizationAgent requires an instance of LLMServiceWrapper.")
        self.lsw = lsw
        self.default_model_name = default_model_name if default_model_name else self.lsw.default_chat_model
        #print(f"SummarizationAgent initialized. Default summarization model: {self.default_model_name}")

    def summarize_text(self,
                       text_to_summarize: str,
                       model_name: str = None,
                       max_summary_length: int = 150, # Approximate target length in tokens for the summary
                       temperature: float = 0.2, # Lower temperature for more factual summaries
                       custom_prompt_template: str = None) -> str or None:
        """
        Generates a summary for the given text.

        Args:
            text_to_summarize (str): The text content to be summarized.
            model_name (str, optional): Specific Ollama model to use for this summarization.
                                        Defaults to agent's default_model_name.
            max_summary_length (int, optional): Approximate maximum number of tokens for the summary.
                                               This will be passed as 'num_predict' or similar to the LLM.
            temperature (float, optional): Temperature for LLM generation.
            custom_prompt_template (str, optional): A custom prompt template string.
                                                    Must include a placeholder for the text, e.g., "{text_to_summarize}".
                                                    If None, a default summarization prompt is used.

        Returns:
            str or None: The generated summary string, or None if summarization failed.
        """
        if not text_to_summarize.strip():
            print("SummarizationAgent: Input text is empty or whitespace. Cannot summarize.")
            return None

        model_to_use = model_name if model_name else self.default_model_name

        if custom_prompt_template:
            if "{text_to_summarize}" not in custom_prompt_template:
                print("SummarizationAgent Error: Custom prompt template must include '{text_to_summarize}'.")
                return None
            prompt_content = custom_prompt_template.format(text_to_summarize=text_to_summarize)
        else:
            # Default prompt - can be refined
            prompt_content = (
                f"Please provide a concise summary of the following text. "
                f"The summary should capture the main points and be approximately {max_summary_length} words or fewer.\n\n"
                f"Text to summarize:\n\"\"\"\n{text_to_summarize}\n\"\"\""
            )
        
        messages = [
            {"role": "system", "content": "You are an expert at summarizing text concisely and accurately."},
            {"role": "user", "content": prompt_content}
        ]

        # print(f"\nSummarizationAgent: Requesting summary from model '{model_to_use}' (max_length ~{max_summary_length} tokens, temp={temperature}).")
        # print(f"  Prompt content being sent (first 100 chars): {prompt_content[:100]}...")


        summary_response = self.lsw.generate_chat_completion(
            messages=messages,
            model_name=model_to_use,
            temperature=temperature,
            max_tokens=max_summary_length, # LSW maps this to num_predict
            # We might want to add other options like top_p for summarization
        )

        if summary_response:
            #print(f"  Summary generated: \"{summary_response[:100].strip()}...\"")
            return summary_response.strip()
        else:
            print("  Summarization failed or LLM returned no response.")
            return None

class FactExtractionAgent:
    """
    An agent responsible for extracting structured facts (e.g., subject-predicate-object triples)
    from a given piece of text using an LLM.
    """
    def __init__(self, lsw: LLMServiceWrapper, default_model_name: str = None):
        """
        Initializes the FactExtractionAgent.

        Args:
            lsw (LLMServiceWrapper): An instance of the LLMServiceWrapper.
            default_model_name (str, optional): Default Ollama model for fact extraction.
                                                If None, LSW's default chat model is used.
        """
        if not isinstance(lsw, LLMServiceWrapper):
            raise TypeError("FactExtractionAgent requires an instance of LLMServiceWrapper.")
        self.lsw = lsw
        self.default_model_name = default_model_name if default_model_name else self.lsw.default_chat_model
        #print(f"FactExtractionAgent initialized. Default extraction model: {self.default_model_name}")

    def extract_facts(self,
                      text_to_process: str,
                      model_name: str = None,
                      temperature: float = 0.0, # Low temperature for factual extraction
                      max_facts_to_extract: int = 5 # Limit number of facts to ask for
                     ) -> list[dict] or None:
        """
        Attempts to extract structured facts (S-P-O triples) from the given text.

        Args:
            text_to_process (str): The input text from which to extract facts.
            model_name (str, optional): Specific Ollama model to use.
            temperature (float, optional): LLM temperature.
            max_facts_to_extract (int, optional): A hint to the LLM about how many distinct facts to try to find.

        Returns:
            list[dict] or None: A list of dictionaries, where each dictionary represents a fact
                                (e.g., {"subject": "...", "predicate": "...", "object": "..."}).
                                Returns None if extraction fails or no facts are found, or an empty list.
        """
        if not text_to_process.strip():
            print("FactExtractionAgent: Input text is empty. Cannot extract facts.")
            return [] # Return empty list for no input

        model_to_use = model_name if model_name else self.default_model_name

        # Prompt Engineering is CRITICAL here.
        # We need to instruct the LLM to output in a specific JSON format.
        # This prompt might need significant iteration for reliability.
        prompt_content = (
            f"Your primary task is to analyze the USER'S STATEMENT provided below and extract key factual claims made by the user about themselves, their preferences, named entities, or important events they describe. "
            f"Focus on identifying clear relationships between distinct entities or concepts. "
            f"For each distinct fact, provide the subject, a meaningful predicate (verb or relational phrase), and the object.\n"
            f"Important Guidelines for Subjects:\n"
            f"  - If the user refers to themselves using 'I' or 'My' (e.g., 'My name is...', 'I like...'), try to make the subject more general and queryable, like 'user's name', 'user's preference', 'user's hobby', 'user's statement about X'.\n"
            f"  - For other entities, use the specific name of the entity as the subject (e.g., 'Project Alpha', 'The AI Conference').\n"
            f"  - The subject should be as complete and specific as possible to capture the core entity the fact is about.\n"
            f"Extract up to {max_facts_to_extract} of the most important and clearly stated facts from the USER'S STATEMENT.\n"
            f"Format your output STRICTLY as a JSON list of objects. Each object must have 'subject', 'predicate', and 'object' keys, and their values must be strings.\n"
            f"If the USER'S STATEMENT contains no clear, distinct factual claims or is a question, return an empty JSON list [].\n"
            f"Avoid extracting trivial, overly fragmented, or nonsensical statements. Focus on information that seems intended to be remembered.\n\n"
            f"CRITICAL: Do NOT extract any information from the examples provided below; only extract from the actual USER'S STATEMENT at the end.\n\n"
            f"Example of GOOD extractions for USER STATEMENTS:\n"
            f"USER'S STATEMENT: \"My name is Alex and I work as a software developer in New York.\"\n"
            f"Output:\n"
            f"[\n"
            f'  {{"subject": "user name", "predicate": "is", "object": "Alex"}},\n'
            f'  {{"subject": "user job", "predicate": "is", "object": "software developer"}},\n'
            f'  {{"subject": "user location", "predicate": "is", "object": "New York"}}\n'
            f"]\n\n"
            f"USER'S STATEMENT: \"The upcoming Innovatech summit is in Berlin.\"\n"
            f"Output:\n"
            f"[\n"
            f'  {{"subject": "Innovatech summit", "predicate": "location is", "object": "Berlin"}}\n'
            f"]\n\n"
            f"USER'S STATEMENT: \"What time is it?\"\n" # Example of a question
            f"Output:\n"
            f"[]\n\n"
            f"USER'S STATEMENT:\n\"\"\"\n{text_to_process}\n\"\"\"\n\n"
            f"Extracted facts (JSON list only, ensure valid JSON format):\n"
        )

        messages = [
            {"role": "system", "content": "You are an AI assistant specialized in extracting structured information as JSON."},
            {"role": "user", "content": prompt_content}
        ]

        #print(f"\nFactExtractionAgent: Requesting fact extraction from model '{model_to_use}' (temp={temperature}).")
        #print(f"  Prompt content being sent (first 150 chars for brevity): {prompt_content[:150]}...")

        # Requesting JSON output from Ollama model
        # Some models support a 'format: json' parameter in options for more reliable JSON.
        # We'll try with the prompt first, then consider 'format: json' if LSW supports passing it.
        # For now, LSW's generate_chat_completion just takes basic options.
        # We can add 'format' to kwargs if LSW is updated to pass it through `options`.
        llm_response_str = self.lsw.generate_chat_completion(
            messages=messages,
            model_name=model_to_use,
            temperature=temperature,
            # max_tokens might need to be generous enough to allow for JSON structure
            # This is for the length of the JSON string itself.
            max_tokens=500 # Adjust as needed based on expected number of facts and their length
        )

        if not llm_response_str:
            print("  FactExtractionAgent: LLM returned no response for fact extraction.")
            return None # Or [] if preferred for "no facts found" vs "error"

        print(f"  LLM Raw Response for facts: \"{llm_response_str.strip()}\"")

        # Attempt to parse the LLM's response as JSON
        try:
            # Clean up the response: LLMs sometimes add leading/trailing text or markdown backticks
            cleaned_response = llm_response_str.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith("```"):
                 cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()

            extracted_data = json.loads(cleaned_response)
            
            if not isinstance(extracted_data, list):
                print(f"  FactExtractionAgent: LLM output was valid JSON but not a list as expected. Output: {extracted_data}")
                return [] # Expecting a list of facts

            # Validate structure of each fact object (optional but good)
            valid_facts = []
            for fact_item in extracted_data:
                if isinstance(fact_item, dict) and \
                   'subject' in fact_item and \
                   'predicate' in fact_item and \
                   'object' in fact_item:
                    valid_facts.append({
                        "subject": str(fact_item['subject']),
                        "predicate": str(fact_item['predicate']),
                        "object": str(fact_item['object'])
                    })
                else:
                    print(f"  FactExtractionAgent: Skipping malformed fact item: {fact_item}")
            
            #print(f"  Successfully extracted {len(valid_facts)} facts.")
            return valid_facts

        except json.JSONDecodeError as e:
            print(f"  FactExtractionAgent: Failed to decode LLM response as JSON. Error: {e}")
            print(f"  LLM response that failed parsing: {llm_response_str}")
            return None # Indicate parsing failure
        except Exception as e:
            print(f"  FactExtractionAgent: An unexpected error occurred during fact processing: {e}")
            return None

# --- Update Test Block ---
if __name__ == "__main__":
    import gc 
   
    # Test Block for KnowledgeRetrieverAgent
    print("--- Testing KnowledgeRetrieverAgent ---")
    # (Code for testing KnowledgeRetrieverAgent as it was)
    # ... (ensure it's self-contained or dependencies are met by the overall __main__ setup) ...
    test_mmu_ltm_sqlite_db_path_kra = 'test_agent_kra_ltm_sqlite.db' # kra specific
    test_mmu_ltm_chroma_dir_kra = 'test_agent_kra_ltm_chroma'
    #test_mmu_mtm_db_path_kra = 'test_agent_kra_mtm_store.json'

    #if os.path.exists(test_mmu_mtm_db_path_kra): os.remove(test_mmu_mtm_db_path_kra)
    if os.path.exists(test_mmu_ltm_sqlite_db_path_kra): os.remove(test_mmu_ltm_sqlite_db_path_kra)
    if os.path.exists(test_mmu_ltm_chroma_dir_kra): shutil.rmtree(test_mmu_ltm_chroma_dir_kra)
    
    print("Initializing a test MMU for KRA...")
    test_mmu_kra = None
    try:
        test_mmu_kra = MemoryManagementUnit(
            ltm_sqlite_db_path=test_mmu_ltm_sqlite_db_path_kra,
            ltm_chroma_persist_dir=test_mmu_ltm_chroma_dir_kra
        )
    except Exception as e:
        print(f"Could not initialize test MMU for KRA: {e}. Aborting KRA test.")
        # Decide if you want to exit() or just skip KRA tests
    
    if test_mmu_kra:
        # Populate data for KRA
        test_mmu_kra.store_ltm_fact("KRA Test Fact", "is_for", "Knowledge Retriever")
        # ... (other KRA test data population and agent testing) ...
        retriever_agent = KnowledgeRetrieverAgent(mmu=test_mmu_kra)
        kra_results = retriever_agent.search_knowledge("KRA Test Fact")
        print(f"KRA search for 'KRA Test Fact' (SKB): {kra_results['skb_fact_results']}")
        print("KnowledgeRetrieverAgent tests finished.")


    # --- Test Block for SummarizationAgent ---
    print("\n\n--- Testing SummarizationAgent ---")
    test_lsw_for_summary = None # Unique name for this LSW instance
    try:
        test_lsw_for_summary = LLMServiceWrapper(default_chat_model="gemma3:1b-it-fp16") 
        if not test_lsw_for_summary.client: 
            print("Ollama client in LSW failed for Summarizer. Skipping SummarizationAgent tests.")
            test_lsw_for_summary = None 
    except Exception as e:
        print(f"Could not initialize LSW for SummarizationAgent test: {e}")
        test_lsw_for_summary = None

    if test_lsw_for_summary:
        summarizer = SummarizationAgent(lsw=test_lsw_for_summary) 
        # ... (one summarization test case for brevity here) ...
        text_to_sum = "The quick brown fox jumps over the lazy dog near the bank of the river. This sentence contains all letters of the alphabet."
        print("\nSummarizing a short text:")
        summary_res = summarizer.summarize_text(text_to_sum, max_summary_length=15)
        if summary_res: print(f"  Summary: {summary_res}")
        else: print("  Summarization failed.")
    else:
        print("Skipping SummarizationAgent tests as LSW could not be initialized.")
    print("SummarizationAgent tests finished.")


    # --- NEW Test Block for FactExtractionAgent ---
    print("\n\n--- Testing FactExtractionAgent ---")
    test_lsw_for_facts = None # Unique name
    try:
        # Using a slightly more capable model if available might be better for structured JSON output.
        # For now, stick to the small one for consistency.
        test_lsw_for_facts = LLMServiceWrapper(default_chat_model="gemma3:1b-it-fp16") # Or "mistral:latest" if you have it and it's good at JSON
        if not test_lsw_for_facts.client:
            print("Ollama client in LSW failed for FactExtractor. Skipping FactExtractionAgent tests.")
            test_lsw_for_facts = None
    except Exception as e:
        print(f"Could not initialize LSW for FactExtractionAgent test: {e}")
        test_lsw_for_facts = None

    if test_lsw_for_facts:
        fact_extractor = FactExtractionAgent(lsw=test_lsw_for_facts)

        text1 = "The company 'Innovatech Solutions' announced a new CEO, Alice Wonderland, yesterday. Their headquarters are in Neo-Tokyo."
        print(f"\nExtracting facts from: \"{text1}\"")
        facts1 = fact_extractor.extract_facts(text1, max_facts_to_extract=3)
        if facts1 is not None: # Check for None (error) vs empty list (no facts found)
            print(f"  Extracted Facts 1 ({len(facts1)}):")
            for fact in facts1:
                print(f"    - S: {fact['subject']}, P: {fact['predicate']}, O: {fact['object']}")
        else:
            print("  Fact extraction 1 failed or returned None.")

        text2 = "The weather is pleasant today, and the birds are singing. My cat is named Whiskers."
        print(f"\nExtracting facts from: \"{text2}\"")
        facts2 = fact_extractor.extract_facts(text2, max_facts_to_extract=2)
        if facts2 is not None:
            print(f"  Extracted Facts 2 ({len(facts2)}):")
            for fact in facts2:
                print(f"    - S: {fact['subject']}, P: {fact['predicate']}, O: {fact['object']}")
        else:
            print("  Fact extraction 2 failed or returned None.")
            
        text3 = "This sentence probably contains no extractable factual claims."
        print(f"\nExtracting facts from: \"{text3}\" (expecting empty list or few facts)")
        facts3 = fact_extractor.extract_facts(text3)
        if facts3 is not None: # Could be an empty list [] if LLM correctly finds no facts
            print(f"  Extracted Facts 3 ({len(facts3)}):")
            for fact in facts3:
                print(f"    - S: {fact['subject']}, P: {fact['predicate']}, O: {fact['object']}")
        else:
            print("  Fact extraction 3 failed or returned None.")

        text_empty = "  "
        print(f"\nExtracting facts from empty text:")
        facts_empty = fact_extractor.extract_facts(text_empty) # Should return []
        if facts_empty is not None:
            print(f"  Extracted Facts from empty ({len(facts_empty)}).")


    else:
        print("Skipping FactExtractionAgent tests as LSW could not be initialized.")
    print("FactExtractionAgent tests finished.")


    print("\nAll agent tests in agents.py finished.")

    # Final cleanup of KRA test files (if they were created)
    if 'test_mmu_kra' in locals() and test_mmu_kra is not None:
        print("\nAttempting final cleanup of KRA test files...")
        if hasattr(test_mmu_kra.mtm, 'is_persistent') and test_mmu_kra.mtm.is_persistent and \
           hasattr(test_mmu_kra.mtm, 'db') and test_mmu_kra.mtm.db:
            test_mmu_kra.mtm.db.close()
        del test_mmu_kra
        gc.collect() 
        time.sleep(0.1)
        # (Simplified cleanup for KRA test files, ensure paths are correct)
        if os.path.exists(test_mmu_ltm_sqlite_db_path_kra): 
            try: os.remove(test_mmu_ltm_sqlite_db_path_kra)
            except Exception as e: print(f"  KRA Cleanup: Could not remove {test_mmu_ltm_sqlite_db_path_kra}: {e}")
        if os.path.exists(test_mmu_ltm_chroma_dir_kra): 
            try: shutil.rmtree(test_mmu_ltm_chroma_dir_kra, ignore_errors=False)
            except Exception as e: print(f"  KRA Cleanup: Could not remove dir {test_mmu_ltm_chroma_dir_kra}: {e}")
        print("KRA test file cleanup attempt finished.")
    else:
        print("\nSkipping KRA test file cleanup as KRA test MMU might not have been initialized.")