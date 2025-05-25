import os
import sys
import shutil 
import time   
import gc     # For test cleanup
import json   # For parsing LLM's structured output

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
        print("KnowledgeRetrieverAgent initialized.")

    def search_knowledge(self,
                         query_text: str,
                         search_vector_store: bool = True,
                         search_skb_facts: bool = True,
                         top_k_vector: int = 3,
                         skb_subject: str = None,
                         skb_predicate: str = None,
                         skb_object: str = None) -> dict:
        """
        Searches for knowledge relevant to the query_text from configured LTM sources.

        Args:
            query_text (str): The primary text query for semantic search in the vector store.
                              Also used to derive terms for SKB search if specific skb args are not given.
            search_vector_store (bool): Whether to search the LTM Vector Store.
            search_skb_facts (bool): Whether to search the LTM Structured Knowledge Base (facts table).
            top_k_vector (int): Number of results to retrieve from the vector store.
            skb_subject (str, optional): Specific subject to search for in SKB facts.
                                         If None and search_skb_facts is True, might try to infer from query_text (simplistic for now).
            skb_predicate (str, optional): Specific predicate for SKB facts.
            skb_object (str, optional): Specific object for SKB facts.

        Returns:
            dict: A dictionary containing results from different sources.
                  Example: {
                      "vector_results": [
                          {"text_chunk": "...", "metadata": {...}, "distance": 0.X}, ...
                      ],
                      "skb_fact_results": [
                          {"subject": "...", "predicate": "...", "object": "...", ...}, ...
                      ]
                  }
        """
        results = {
            "vector_results": [],
            "skb_fact_results": []
        }
        print(f"\nKnowledgeRetrieverAgent: Searching for query: '{query_text}'")

        # 1. Search Vector Store (Semantic Search)
        if search_vector_store:
            print(f"  - Performing semantic search in LTM Vector Store (top_k={top_k_vector})...")
            try:
                vector_search_results = self.mmu.semantic_search_ltm_vector_store(
                    query_text=query_text,
                    top_k=top_k_vector
                )
                if vector_search_results:
                    results["vector_results"] = vector_search_results
                    print(f"    Found {len(vector_search_results)} item(s) from vector store.")
                else:
                    print("    No items found in vector store for this query.")
            except Exception as e:
                print(f"    Error during vector store search: {e}")

        # 2. Search Structured Knowledge Base (Facts)
        if search_skb_facts:
            # Use specific SKB parameters if provided, otherwise, we could try a simple keyword search
            # based on query_text for subject/object as a basic heuristic.
            # For this version, we'll prioritize explicit params or a general query against object.
            # A more advanced version might use NLP to parse query_text into S-P-O.
            
            effective_subject = skb_subject
            effective_predicate = skb_predicate
            effective_object = skb_object

            # Simplistic heuristic: if no specific SKB fields, use query_text for object search
            if not skb_subject and not skb_predicate and not skb_object and query_text:
                effective_object = query_text # Search query_text against the 'object' field
                print(f"  - Performing SKB fact search (query_text='{query_text}' against 'object' field)...")
            else:
                 print(f"  - Performing SKB fact search (S:'{effective_subject}', P:'{effective_predicate}', O:'{effective_object}')...")
            
            try:
                skb_fact_search_results = self.mmu.get_ltm_facts(
                    subject=effective_subject,
                    predicate=effective_predicate,
                    object_value=effective_object # Note: LTM method uses 'object_value'
                )
                if skb_fact_search_results:
                    results["skb_fact_results"] = skb_fact_search_results
                    print(f"    Found {len(skb_fact_search_results)} fact(s) from SKB.")
                else:
                    print("    No facts found in SKB for these criteria.")
            except Exception as e:
                print(f"    Error during SKB fact search: {e}")
        
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
        print(f"SummarizationAgent initialized. Default summarization model: {self.default_model_name}")

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

        print(f"\nSummarizationAgent: Requesting summary from model '{model_to_use}' (max_length ~{max_summary_length} tokens, temp={temperature}).")
        # print(f"  Prompt content being sent (first 100 chars): {prompt_content[:100]}...")


        summary_response = self.lsw.generate_chat_completion(
            messages=messages,
            model_name=model_to_use,
            temperature=temperature,
            max_tokens=max_summary_length, # LSW maps this to num_predict
            # We might want to add other options like top_p for summarization
        )

        if summary_response:
            print(f"  Summary generated: \"{summary_response[:100].strip()}...\"")
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
        print(f"FactExtractionAgent initialized. Default extraction model: {self.default_model_name}")

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
            f"Analyze the following text and extract up to {max_facts_to_extract} distinct factual statements "
            f"from it. For each fact, identify the subject, predicate, and object.\n"
            f"Present these facts as a JSON list of objects, where each object has 'subject', 'predicate', and 'object' keys.\n"
            f"If no clear facts can be extracted, return an empty JSON list [].\n\n"
            f"Example of desired output format:\n"
            f"[\n"
            f'  {{"subject": "The sky", "predicate": "is", "object": "blue"}},\n'
            f'  {{"subject": "Paris", "predicate": "is the capital of", "object": "France"}}\n'
            f"]\n\n"
            f"Text to analyze:\n\"\"\"\n{text_to_process}\n\"\"\"\n\n"
            f"Extracted facts (JSON list only):\n"
        )

        messages = [
            {"role": "system", "content": "You are an AI assistant specialized in extracting structured information as JSON."},
            {"role": "user", "content": prompt_content}
        ]

        print(f"\nFactExtractionAgent: Requesting fact extraction from model '{model_to_use}' (temp={temperature}).")
        print(f"  Prompt content being sent (first 150 chars for brevity): {prompt_content[:150]}...")

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
            
            print(f"  Successfully extracted {len(valid_facts)} facts.")
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