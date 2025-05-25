import os
import sys
import shutil 
import time   

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


# --- Update Test Block ---
if __name__ == "__main__":
    import gc 
    # ... (Keep existing KnowledgeRetrieverAgent test setup and tests) ...
    # ... (Make sure the imports for os, shutil, time, MMU are handled by the conditional block at the top)
    
    # Test Block for KnowledgeRetrieverAgent
    print("--- Testing KnowledgeRetrieverAgent ---")
    # (Code for testing KnowledgeRetrieverAgent as it was)
    # ... (ensure it's self-contained or dependencies are met by the overall __main__ setup) ...
    test_mmu_ltm_sqlite_db_path_kra = 'test_agent_kra_ltm_sqlite.db' # kra specific
    test_mmu_ltm_chroma_dir_kra = 'test_agent_kra_ltm_chroma'
    test_mmu_mtm_db_path_kra = 'test_agent_kra_mtm_store.json'

    if os.path.exists(test_mmu_mtm_db_path_kra): os.remove(test_mmu_mtm_db_path_kra)
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
    
    # Initialize LSW (make sure Ollama is running and model is available)
    # Adjust default_chat_model if 'gemma3:1b-it-fp16' is not what you want for summarization.
    # A slightly larger model might give better summaries but will be slower.
    test_lsw = None
    try:
        test_lsw = LLMServiceWrapper(default_chat_model="gemma3:1b-it-fp16") # Or your preferred small model
        if not test_lsw.client: # Check if Ollama client initialized
            print("Ollama client in LSW failed to initialize. Skipping SummarizationAgent tests.")
            test_lsw = None # Ensure it's None so tests are skipped
    except Exception as e:
        print(f"Could not initialize LSW for SummarizationAgent test: {e}")
        test_lsw = None

    if test_lsw:
        # Use LSW's default model for summarization unless overridden
        summarizer = SummarizationAgent(lsw=test_lsw) 

        long_text_1 = (
            "The James Webb Space Telescope (JWST) is a space telescope designed primarily to conduct infrared astronomy. "
            "As the largest optical telescope in space, its high infrared resolution and sensitivity allow it to view objects "
            "too old, distant, or faint for the Hubble Space Telescope. This is expected to enable a broad range of "
            "investigations across the fields of astronomy and cosmology, such as observation of the first stars and "
            "the formation of the first galaxies, and detailed atmospheric characterization of potentially habitable exoplanets."
        )
        print("\nSummarizing JWST text (default prompt):")
        summary1 = summarizer.summarize_text(long_text_1, max_summary_length=50) # Target ~50 tokens
        if summary1:
            print(f"  Original length: {len(long_text_1.split())} words. Summary: {summary1}")
        else:
            print("  Summarization 1 failed.")

        long_text_2 = (
            "Python is an interpreted, high-level, general-purpose programming language. Created by Guido van Rossum and "
            "first released in 1991, Python's design philosophy emphasizes code readability with its notable use of "
            "significant whitespace. Its language constructs and object-oriented approach aim to help programmers write "
            "clear, logical code for small and large-scale projects. Python is dynamically typed and garbage-collected. "
            "It supports multiple programming paradigms, including structured (particularly, procedural), object-oriented, "
            "and functional programming."
        )
        custom_template = "Condense this information about Python into a single sentence: {text_to_summarize}"
        print("\nSummarizing Python text (custom prompt - single sentence):")
        summary2 = summarizer.summarize_text(
            long_text_2, 
            max_summary_length=30, # Shorter target
            custom_prompt_template=custom_template,
            temperature=0.1
        )
        if summary2:
            print(f"  Original length: {len(long_text_2.split())} words. Summary: {summary2}")
        else:
            print("  Summarization 2 failed.")
            
        print("\nTesting with empty text:")
        summary_empty = summarizer.summarize_text("")
        # Should print "Input text is empty..." and return None

        print("\nTesting with custom prompt missing placeholder:")
        summary_bad_prompt = summarizer.summarize_text("Some text.", custom_prompt_template="Summarize this.")
        # Should print error and return None

    else:
        print("Skipping SummarizationAgent tests as LSW could not be initialized.")

    print("\nAll agent tests finished.")


    # Final cleanup of KRA test files (if they were created)
    # We can make this more robust by checking if test_mmu_kra was successfully created
    if 'test_mmu_kra' in locals() and test_mmu_kra is not None:
        print("\nAttempting final cleanup of KRA test files...")
        if hasattr(test_mmu_kra.mtm, 'is_persistent') and test_mmu_kra.mtm.is_persistent and \
           hasattr(test_mmu_kra.mtm, 'db') and test_mmu_kra.mtm.db:
            test_mmu_kra.mtm.db.close()
        del test_mmu_kra
        gc.collect() # gc was imported earlier for mmu_manager tests, ensure it's available
        time.sleep(0.1)
        if os.path.exists(test_mmu_mtm_db_path_kra): os.remove(test_mmu_mtm_db_path_kra)
        if os.path.exists(test_mmu_ltm_sqlite_db_path_kra): os.remove(test_mmu_ltm_sqlite_db_path_kra)
        if os.path.exists(test_mmu_ltm_chroma_dir_kra): shutil.rmtree(test_mmu_ltm_chroma_dir_kra, ignore_errors=True)
        print("KRA test file cleanup attempt finished.")
    else:
        print("\nSkipping KRA test file cleanup as KRA test MMU was not initialized.")