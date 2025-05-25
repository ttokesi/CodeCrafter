# offline_chat_bot/core/agents.py

# We need to import the MemoryManagementUnit to interact with LTM.
# We need to adjust the import path depending on how this agent module will be used.
# Assuming 'core' and 'mmu' are sibling directories under the project root.

# --- For direct execution/testing of this file ---
if __name__ == '__main__' and __package__ is None:
    import sys
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Goes up to offline_chat_bot
    sys.path.insert(0, project_root)
    # Now we can use absolute-like imports from the project root
    from mmu.mmu_manager import MemoryManagementUnit
    from mmu.ltm import LTMManager # For test setup LTM if needed directly
else:
    # When imported as part of the 'core' package
    from mmu.mmu_manager import MemoryManagementUnit
    import os
# --- End of testing specific import logic ---


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

# --- Test Block ---
if __name__ == "__main__":
    print("--- Testing KnowledgeRetrieverAgent ---")

    # Setup a mock/test MMU for the agent.
    # For this test, we'll use real MMU components but with test-specific paths.
    test_mmu_ltm_sqlite_db_path = 'test_agent_ltm_sqlite.db'
    test_mmu_ltm_chroma_dir = 'test_agent_ltm_chroma'
    test_mmu_mtm_db_path = 'test_agent_mtm_store.json' # MTM not directly used by this agent but MMU init needs it

    # Cleanup previous test files
    if os.path.exists(test_mmu_mtm_db_path): os.remove(test_mmu_mtm_db_path)
    if os.path.exists(test_mmu_ltm_sqlite_db_path): os.remove(test_mmu_ltm_sqlite_db_path)
    import shutil
    import time
    if os.path.exists(test_mmu_ltm_chroma_dir): shutil.rmtree(test_mmu_ltm_chroma_dir)

    print("Initializing a test MMU for the agent...")
    # Assuming LTMManager.SENTENCE_TRANSFORMER_AVAILABLE is True for this test to run fully.
    # If it's not, vector store operations will be skipped or might fail depending on LTMManager's robustness.
    # We need to ensure the LTM's default embedding function is available.
    # The VectorStoreChroma in LTMManager will print warnings if sentence-transformers isn't found.
    
    test_mmu = None
    try:
        test_mmu = MemoryManagementUnit(
            mtm_use_tinydb=False, # Keep MTM in-memory for this agent test
            ltm_sqlite_db_path=test_mmu_ltm_sqlite_db_path,
            ltm_chroma_persist_dir=test_mmu_ltm_chroma_dir
        )
        print("Test MMU initialized.")
    except Exception as e:
        print(f"Could not initialize test MMU: {e}")
        print("Aborting KnowledgeRetrieverAgent test.")
        exit()


    # Populate some data into LTM via MMU for testing
    print("\nPopulating LTM with test data...")
    # LTM Raw Log (not directly searched by this agent, but good practice)
    test_mmu.log_ltm_interaction("conv1", 1, "user", "What is the Project Alpha deadline?")
    test_mmu.log_ltm_interaction("conv1", 2, "assistant", "Project Alpha is due by Q4 2024.")

    # LTM SKB Facts
    test_mmu.store_ltm_fact("Project Alpha", "has_deadline", "Q4 2024", confidence=0.9)
    test_mmu.store_ltm_fact("Paris", "is_capital_of", "France", confidence=1.0)
    test_mmu.store_ltm_fact("The sky", "is_color", "blue", confidence=0.7)
    
    # LTM Vector Store Documents
    # Check if LTM's vector store is usable (embedding function available)
    ltm_vector_store_ready = False
    if hasattr(test_mmu.ltm, 'vector_store') and test_mmu.ltm.vector_store and \
       hasattr(test_mmu.ltm.vector_store, 'collection') and test_mmu.ltm.vector_store.collection and \
       hasattr(test_mmu.ltm.vector_store.collection, '_embedding_function') and \
       test_mmu.ltm.vector_store.collection._embedding_function is not None:
        ltm_vector_store_ready = True

    if ltm_vector_store_ready:
        print("  Populating LTM Vector Store...")
        test_mmu.add_document_to_ltm_vector_store(
            text_chunk="The Project Alpha initiative aims to deliver innovative solutions by the end of the year.",
            metadata={"document_id": "doc1", "topic": "project_alpha"}
        )
        test_mmu.add_document_to_ltm_vector_store(
            text_chunk="Paris is known for the Eiffel Tower and its rich history.",
            metadata={"document_id": "doc2", "topic": "cities"}
        )
        test_mmu.add_document_to_ltm_vector_store(
            text_chunk="The sky appears blue due to Rayleigh scattering.",
            metadata={"document_id": "doc3", "topic": "science"}
        )
        import time
        time.sleep(1) # Give Chroma a moment to process embeddings
    else:
        print("  LTM Vector Store embedding function not available. Skipping vector store population for test.")

    # Initialize the agent
    retriever_agent = KnowledgeRetrieverAgent(mmu=test_mmu)

    # Test Case 1: General query, search both vector store and SKB
    print("\n--- Test Case 1: General query for 'Project Alpha deadline' ---")
    results1 = retriever_agent.search_knowledge(query_text="Project Alpha deadline")
    print("Results1 (Vector Store):")
    for res in results1["vector_results"]:
        print(f"  - Text: '{res['text_chunk'][:50]}...', Dist: {res['distance']:.4f}, Meta: {res['metadata']}")
    print("Results1 (SKB Facts):")
    for fact in results1["skb_fact_results"]:
        print(f"  - Fact: {fact['subject']} {fact['predicate']} {fact['object']}") # LTM uses object_value

    # Test Case 2: Query more related to SKB fact structure
    print("\n--- Test Case 2: Query for 'capital of France' (SKB focus) ---")
    results2 = retriever_agent.search_knowledge(
        query_text="capital of France",
        skb_object="France", # More specific SKB query part
        skb_predicate="is_capital_of"
    )
    print("Results2 (Vector Store):") # Vector store will still search "capital of France"
    for res in results2["vector_results"]:
        print(f"  - Text: '{res['text_chunk'][:50]}...', Dist: {res['distance']:.4f}, Meta: {res['metadata']}")
    print("Results2 (SKB Facts):")
    for fact in results2["skb_fact_results"]:
        print(f"  - Fact: {fact['subject']} {fact['predicate']} {fact['object']}")

    # Test Case 3: Query only vector store
    print("\n--- Test Case 3: Query for 'Eiffel Tower' (Vector Store only) ---")
    results3 = retriever_agent.search_knowledge(query_text="Eiffel Tower", search_skb_facts=False)
    print("Results3 (Vector Store):")
    for res in results3["vector_results"]:
        print(f"  - Text: '{res['text_chunk'][:50]}...', Dist: {res['distance']:.4f}, Meta: {res['metadata']}")
    print(f"Results3 (SKB Facts should be empty): {results3['skb_fact_results']}")
    
    # Test Case 4: Query that might not find much
    print("\n--- Test Case 4: Query for 'latest TPS reports' (likely no results) ---")
    results4 = retriever_agent.search_knowledge(query_text="latest TPS reports")
    print(f"Results4 (Vector Store count): {len(results4['vector_results'])}")
    print(f"Results4 (SKB Facts count): {len(results4['skb_fact_results'])}")

    print("\nKnowledgeRetrieverAgent tests finished.")

    # Final cleanup
    print("\nAttempting final cleanup of agent test files...")
    del retriever_agent
    if hasattr(test_mmu, 'mtm') and test_mmu.mtm.is_persistent and hasattr(test_mmu.mtm, 'db') and test_mmu.mtm.db:
        test_mmu.mtm.db.close()
    del test_mmu
    import gc
    gc.collect()
    time.sleep(0.1)

    if os.path.exists(test_mmu_mtm_db_path): os.remove(test_mmu_mtm_db_path)
    if os.path.exists(test_mmu_ltm_sqlite_db_path): os.remove(test_mmu_ltm_sqlite_db_path)
    if os.path.exists(test_mmu_ltm_chroma_dir): shutil.rmtree(test_mmu_ltm_chroma_dir, ignore_errors=True)
    print("Agent test file cleanup attempt finished.")