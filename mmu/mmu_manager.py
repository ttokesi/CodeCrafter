# offline_chat_bot/mmu/mmu_manager.py

import datetime
from collections import deque
import json
import time 
import os
import shutil 
import gc 


# Conditional import for LTMManager
if __name__ == '__main__' and __package__ is None:
    import sys
    project_root_for_direct_run = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root_for_direct_run not in sys.path:
        sys.path.insert(0, project_root_for_direct_run)
    from mmu.ltm import LTMManager
    # For the __main__ test block in this file, we'll also need LSW and its embedding function (or a mock)
    from core.llm_service_wrapper import LLMServiceWrapper # For testing MMU init
    from core.config_loader import get_config, get_project_root
    from chromadb import Documents, EmbeddingFunction, Embeddings # For mock embedding class in __main__

else: 
    from .ltm import LTMManager 
    from core.config_loader import get_config, get_project_root
    # When MMU is imported as a module, it will expect an embedding_function to be passed to its __init__
    # It won't initialize LSW itself.

# TinyDB imports (as before)
try:
    from tinydb import TinyDB, Query, where
    from tinydb.storages import JSONStorage
    from tinydb.middlewares import CachingMiddleware
    TINYDB_AVAILABLE = True
except ImportError:
    TINYDB_AVAILABLE = False
    # ... (dummy TinyDB classes as before) ...
    class TinyDB: pass 
    class Query: pass
    def where(key): return None 

class ShortTermMemory:
    """
    Manages Short-Term Memory (STM) for the immediate conversation context.
    Uses a deque to store recent conversation turns.
    """
    def __init__(self, max_turns: int): # max_turns is now required
        self.max_turns = max_turns
        self.history = deque(maxlen=max_turns)
        self.scratchpad_content = "" 

    def add_turn(self, role: str, content: str):
        """
        Adds a new turn to the STM.
        If the history exceeds max_turns, the oldest turn is automatically dropped by the deque.

        Args:
            role (str): "user" or "assistant" (or "system", "tool_call", "tool_response").
            content (str): The text content of the turn.
        """
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        turn_data = {"role": role, "content": content, "timestamp": timestamp}
        self.history.append(turn_data)
        # print(f"STM: Added turn from '{role}'. History size: {len(self.history)}")

    def get_history(self) -> list:
        """
        Returns the current STM history as a list of turns.

        Returns:
            list: A list of turn dictionaries, from oldest to newest in the current STM window.
        """
        return list(self.history)

    def get_formatted_history(self, include_timestamps: bool = False) -> str:
        """
        Returns the STM history formatted as a simple string, suitable for an LLM prompt.

        Args:
            include_timestamps (bool): Whether to include timestamps in the formatted string.

        Returns:
            str: A multi-line string representing the conversation history.
        """
        formatted_lines = []
        for turn in self.history:
            prefix = f"{turn['role'].capitalize()}"
            if include_timestamps:
                # Basic time formatting, can be improved
                dt_obj = datetime.datetime.fromisoformat(turn['timestamp'].replace('Z', '+00:00'))
                time_str = dt_obj.strftime("%H:%M:%S")
                prefix += f" ({time_str})"
            formatted_lines.append(f"{prefix}: {turn['content']}")
        return "\n".join(formatted_lines)

    def clear(self):
        """Clears the STM history and scratchpad."""
        self.history.clear()
        self.scratchpad_content = ""
        # print("STM: Cleared history and scratchpad.")

    def update_scratchpad(self, content: str):
        """Updates the scratchpad content. Overwrites previous content."""
        self.scratchpad_content = content

    def append_to_scratchpad(self, content_to_append: str, separator: str = "\n"):
        """Appends content to the scratchpad."""
        if self.scratchpad_content:
            self.scratchpad_content += separator + content_to_append
        else:
            self.scratchpad_content = content_to_append
            
    def get_scratchpad(self) -> str:
        """Returns the current scratchpad content."""
        return self.scratchpad_content

class MediumTermMemory:
    """
    Manages Medium-Term Memory (MTM) for session-specific summaries, entities, etc.
    Can operate in-memory or with TinyDB for simple file-based persistence.
    """
    def __init__(self, use_tinydb: bool, db_path: str): # Parameters are now required
        self.is_persistent = False
        self.db = None 
        self.summaries_table = None
        self.entities_table = None
        self.context_table = None

        if use_tinydb and TINYDB_AVAILABLE:
            try:
                # Using CachingMiddleware can improve performance for frequent reads
                self.db = TinyDB(db_path, storage=CachingMiddleware(JSONStorage)) # Use passed db_path
                self.summaries_table = self.db.table('summaries')
                self.entities_table = self.db.table('entities')
                self.context_table = self.db.table('task_context')
                self.is_persistent = True
                print(f"MTM: Using TinyDB for persistence at '{db_path}'.")
            except Exception as e:
                print(f"MTM: Error initializing TinyDB at '{db_path}'. Falling back to in-memory. Error: {e}")
                self._initialize_in_memory_stores()
        else:
            if use_tinydb and not TINYDB_AVAILABLE:
                print("MTM: TinyDB requested but not available. Using in-memory MTM.")
            self._initialize_in_memory_stores()

    def _initialize_in_memory_stores(self):
        """Initializes MTM stores as simple Python dictionaries."""
        self.in_memory_summaries = {} # key: summary_id, value: {"text": "...", "metadata": {...}}
        self.in_memory_entities = {}  # key: category, value: {entity_key: value}
        self.in_memory_context = {}   # key: context_key, value: data
        self.is_persistent = False
        # print("MTM: Using in-memory stores.")

    # --- Summaries ---
    def store_summary(self, summary_id: str, summary_text: str, metadata: dict = None):
        """Stores or updates a conversation summary."""
        if not metadata: metadata = {}
        metadata['last_updated'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        if self.is_persistent and self.summaries_table is not None:
            self.summaries_table.upsert({'summary_id': summary_id, 'text': summary_text, 'metadata': metadata}, 
                                        Query().summary_id == summary_id)
        else:
            self.in_memory_summaries[summary_id] = {'text': summary_text, 'metadata': metadata}

    def get_summary(self, summary_id: str) -> dict or None:
        """Retrieves a specific summary by its ID."""
        if self.is_persistent and self.summaries_table is not None:
            result = self.summaries_table.get(Query().summary_id == summary_id)
            return result if result else None
        else:
            return self.in_memory_summaries.get(summary_id)

    def get_recent_summaries(self, max_count: int = 5) -> list:
        """Retrieves the most recent summaries (if timestamps are in metadata)."""
        if self.is_persistent and self.summaries_table is not None:
            # TinyDB sorting can be a bit basic. For complex sorting, might fetch all and sort in Python.
            all_summaries = self.summaries_table.all()
            # Sort by 'last_updated' in metadata, descending.
            # Ensure 'last_updated' exists in metadata for reliable sorting.
            all_summaries.sort(key=lambda s: s.get('metadata', {}).get('last_updated', ''), reverse=True)
            return all_summaries[:max_count]
        else:
            # In-memory sorting
            sorted_summaries = sorted(
                self.in_memory_summaries.values(),
                key=lambda s: s.get('metadata', {}).get('last_updated', ''),
                reverse=True
            )
            return sorted_summaries[:max_count]

    # --- Entities ---
    def store_entity(self, category: str, entity_key: str, entity_value: any):
        """Stores or updates an entity."""
        if self.is_persistent and self.entities_table is not None:
            # For entities, we might store them nested under a category document
            category_doc = self.entities_table.get(Query().category == category)
            if category_doc:
                category_doc['entities'][entity_key] = entity_value
                self.entities_table.update({'entities': category_doc['entities']}, Query().category == category)
            else:
                self.entities_table.insert({'category': category, 'entities': {entity_key: entity_value}})
        else:
            if category not in self.in_memory_entities:
                self.in_memory_entities[category] = {}
            self.in_memory_entities[category][entity_key] = entity_value
            
    def get_entity(self, category: str, entity_key: str) -> any or None:
        """Retrieves a specific entity."""
        if self.is_persistent and self.entities_table is not None:
            category_doc = self.entities_table.get(Query().category == category)
            return category_doc['entities'].get(entity_key) if category_doc and 'entities' in category_doc else None
        else:
            return self.in_memory_entities.get(category, {}).get(entity_key)

    def get_entities_by_category(self, category: str) -> dict or None:
        """Retrieves all entities for a given category."""
        if self.is_persistent and self.entities_table is not None:
            category_doc = self.entities_table.get(Query().category == category)
            return category_doc['entities'] if category_doc and 'entities' in category_doc else None
        else:
            return self.in_memory_entities.get(category)

    # --- General Task Context ---
    def store_task_context(self, context_key: str, data: any):
        """Stores or updates general task-specific context data."""
        if self.is_persistent and self.context_table is not None:
            self.context_table.upsert({'key': context_key, 'data': data}, Query().key == context_key)
        else:
            self.in_memory_context[context_key] = data

    def get_task_context(self, context_key: str) -> any or None:
        """Retrieves task-specific context data."""
        if self.is_persistent and self.context_table is not None:
            result = self.context_table.get(Query().key == context_key)
            return result['data'] if result and 'data' in result else None
        else:
            return self.in_memory_context.get(context_key)
            
    def clear_session_data(self):
        """Clears all MTM data for the current session."""
        if self.is_persistent and self.db is not None:
            # For TinyDB, clearing means truncating tables
            if self.summaries_table: self.summaries_table.truncate()
            if self.entities_table: self.entities_table.truncate()
            if self.context_table: self.context_table.truncate()
            print("MTM: Cleared all persistent session data (tables truncated).")
        else:
            self._initialize_in_memory_stores() # Re-initialize in-memory dicts
            # print("MTM: Cleared all in-memory session data.")

class MemoryManagementUnit:
    """
    Central controller for all memory tiers: STM, MTM, and LTM.
    Provides a unified API for memory operations.
    """
    def __init__(self, 
                 embedding_function, # Required: for LTM VectorStore
                 config: dict = None  # Optional: for passing a specific config (e.g., for tests)
                ):
        """
        Initializes all memory components using global config or a provided config dict.

        Args:
            embedding_function (callable): The function to be used for generating embeddings
                                           for the LTM Vector Store.
            config (dict, optional): A configuration dictionary. If None, loads global config.
        """
        if not callable(embedding_function):
            raise TypeError("MemoryManagementUnit requires a callable embedding_function for LTM.")

        if config is None:
            print("MMU: Loading global configuration...")
            config = get_config()
        else:
            print("MMU: Using provided configuration dictionary.")
            
        mmu_config = config.get('mmu', {})
        project_r = get_project_root()

        # STM Config
        stm_max_turns = mmu_config.get('stm_max_turns', 10) # Default if not in config
        
        # MTM Config
        mtm_use_tinydb = mmu_config.get('mtm_use_tinydb', False)
        mtm_db_path_relative = mmu_config.get('mtm_db_path', 'data/mtm_store.json') # Default path
        mtm_db_path_abs = os.path.join(project_r, mtm_db_path_relative)
        # Ensure directory for MTM db exists if it's persistent
        if mtm_use_tinydb:
            os.makedirs(os.path.dirname(mtm_db_path_abs), exist_ok=True)

        # LTM Config (paths for LTMManager)
        ltm_sqlite_db_path_relative = mmu_config.get('ltm_sqlite_db_path', 'data/ltm_database.db')
        ltm_sqlite_db_path_abs = os.path.join(project_r, ltm_sqlite_db_path_relative)
        
        ltm_chroma_persist_dir_relative = mmu_config.get('ltm_chroma_persist_directory', 'data/ltm_vector_store')
        ltm_chroma_persist_dir_abs = os.path.join(project_r, ltm_chroma_persist_dir_relative)
        # Ensure directories for LTM dbs exist
        os.makedirs(os.path.dirname(ltm_sqlite_db_path_abs), exist_ok=True)
        os.makedirs(ltm_chroma_persist_dir_abs, exist_ok=True)


        print("Initializing MemoryManagementUnit components...")
        self.stm = ShortTermMemory(max_turns=stm_max_turns)
        print(f"  STM initialized (max_turns={stm_max_turns}).")

        self.mtm = MediumTermMemory(use_tinydb=mtm_use_tinydb, db_path=mtm_db_path_abs)
        print(f"  MTM initialized (persistent={self.mtm.is_persistent}, path='{mtm_db_path_abs if mtm_use_tinydb else 'in-memory'}').")
        
        self.ltm = LTMManager(
            db_path=ltm_sqlite_db_path_abs,
            chroma_persist_dir=ltm_chroma_persist_dir_abs,
            embedding_function=embedding_function # Pass the provided embedding function
        )
        print(f"  LTM initialized (SQLite='{ltm_sqlite_db_path_abs}', Chroma_dir='{ltm_chroma_persist_dir_abs}').")
        print("MemoryManagementUnit initialization complete.")

    # --- STM Facade Methods ---
    def add_stm_turn(self, role: str, content: str):
        self.stm.add_turn(role, content)

    def get_stm_history(self) -> list:
        return self.stm.get_history()

    def get_stm_formatted_history(self, include_timestamps: bool = False) -> str:
        return self.stm.get_formatted_history(include_timestamps=include_timestamps)

    def clear_stm(self):
        self.stm.clear()

    def update_stm_scratchpad(self, content: str):
        self.stm.update_scratchpad(content)
    
    def append_stm_scratchpad(self, content_to_append: str, separator: str = "\n"):
        self.stm.append_to_scratchpad(content_to_append, separator)

    def get_stm_scratchpad(self) -> str:
        return self.stm.get_scratchpad()

    # --- MTM Facade Methods ---
    def store_mtm_summary(self, summary_id: str, summary_text: str, metadata: dict = None):
        self.mtm.store_summary(summary_id, summary_text, metadata)

    def get_mtm_summary(self, summary_id: str) -> dict or None:
        return self.mtm.get_summary(summary_id)

    def get_mtm_recent_summaries(self, max_count: int = 5) -> list:
        return self.mtm.get_recent_summaries(max_count)

    def store_mtm_entity(self, category: str, entity_key: str, entity_value: any):
        self.mtm.store_entity(category, entity_key, entity_value)

    def get_mtm_entity(self, category: str, entity_key: str) -> any or None:
        return self.mtm.get_entity(category, entity_key)
    
    def get_mtm_entities_by_category(self, category: str) -> dict or None:
        return self.mtm.get_entities_by_category(category)

    def store_mtm_task_context(self, context_key: str, data: any):
        self.mtm.store_task_context(context_key, data)

    def get_mtm_task_context(self, context_key: str) -> any or None:
        return self.mtm.get_task_context(context_key)

    def clear_mtm_session_data(self):
        self.mtm.clear_session_data()

    # --- LTM Facade Methods (from LTMManager) ---
    def log_ltm_interaction(self, conversation_id: str, turn_sequence_id: int, role: str, content: str, **kwargs):
        # Pass through kwargs to LTMManager's log_interaction
        return self.ltm.log_interaction(conversation_id, turn_sequence_id, role, content, **kwargs)

    def get_ltm_conversation_history(self, conversation_id: str, limit: int = None, offset: int = 0) -> list:
        return self.ltm.get_conversation_history(conversation_id, limit, offset)
    
    def get_ltm_all_conversation_ids(self) -> list:
        return self.ltm.get_all_conversation_ids()

    def store_ltm_fact(self, subject: str, predicate: str, object_value: str, **kwargs):
        return self.ltm.store_fact(subject, predicate, object_value, **kwargs)

    def get_ltm_facts(self, subject: str = None, predicate: str = None, object_value: str = None) -> list:
        return self.ltm.get_facts(subject, predicate, object_value)

    def store_ltm_preference(self, category: str, key: str, value: str, user_id: str = "default_user") -> str:
        return self.ltm.store_preference(category, key, value, user_id)

    def get_ltm_preference(self, category: str, key: str, user_id: str = "default_user") -> dict:
        return self.ltm.get_preference(category, key, user_id)

    def add_document_to_ltm_vector_store(self, text_chunk: str, metadata: dict, doc_id: str = None) -> str:
        return self.ltm.add_document_to_vector_store(text_chunk, metadata, doc_id)

    def semantic_search_ltm_vector_store(self, query_text: str, top_k: int = 5, metadata_filter: dict = None) -> list:
        return self.ltm.semantic_search_vector_store(query_text, top_k, metadata_filter)

    def reset_all_ltm(self, confirm_reset: bool = False) -> bool:
        """Resets all Long-Term Memory components (SQLite and ChromaDB)."""
        return self.ltm.reset_ltm(confirm_reset=confirm_reset)

    # --- Overall MMU Reset (optional, could combine MTM clear with LTM reset) ---
    def reset_all_memory(self, confirm_reset: bool = False):
        """
        Resets STM, MTM, and LTM.
        USE WITH CAUTION.
        """
        if not confirm_reset:
            print("Full MMU reset aborted. `confirm_reset` must be True.")
            return False
        
        print("Initiating full MMU reset...")
        self.clear_stm()
        print("  STM cleared.")
        self.clear_mtm_session_data()
        print("  MTM cleared.")
        ltm_reset_success = self.reset_all_ltm(confirm_reset=True) # LTM reset already has its own confirmation
        if ltm_reset_success:
            print("Full MMU reset completed successfully.")
            return True
        else:
            print("Full MMU reset completed, but LTM reset reported issues.")
            return False

# In offline_chat_bot/mmu/mmu_manager.py
if __name__ == "__main__":
    print("--- Testing MemoryManagementUnit (with Config and Mock Embedding) ---")

    # Load the main application config for this test
    # This ensures MMU initializes with paths that should exist based on config.
    # The get_config() in config_loader already creates the base 'data_dir'
    app_config = get_config() # Loads from config.yaml
    from chromadb import Documents, EmbeddingFunction, Embeddings
    # --- Mock LSW and Embedding Function for MMU Test ---
    # We need this because MMU now requires an embedding_function for LTMManager
    class MockMMUTestEmbeddingFunction(EmbeddingFunction): # Inherit from Chroma's base
        def __init__(self, dim: int = 10):
            self.dim = dim
            print(f"  MMU_TEST_MOCK_EMBEDDER_CLASS: Initialized with dim={self.dim}")
        def __call__(self, input: Documents) -> Embeddings:
            if not input: return []
            print(f"  MMU_TEST_MOCK_EMBEDDER_CLASS: __call__ received {len(input)} doc(s).")
            return [[0.0] * self.dim for _ in range(len(input))]
    
    mock_embed_func_for_mmu = MockMMUTestEmbeddingFunction(dim=100) # Instance
    # --- End Mock ---

    # Define unique test paths to avoid conflict with production data or other tests
    test_mmu_main_mtm_db_path = os.path.join(get_project_root(), 'test_data/mmu_test_mtm_store.json')
    test_mmu_main_ltm_sqlite_db_path = os.path.join(get_project_root(), 'test_data/mmu_test_ltm_sqlite.db')
    test_mmu_main_ltm_chroma_dir = os.path.join(get_project_root(), 'test_data/mmu_test_ltm_chroma')

    # Create a specific test configuration dictionary to override default paths
    test_specific_mmu_config_override = {
        "mmu": {
            "stm_max_turns": 5, # Test with a different value
            "mtm_use_tinydb": True, # Test MTM persistence
            "mtm_db_path": os.path.relpath(test_mmu_main_mtm_db_path, get_project_root()),
            "ltm_sqlite_db_path": os.path.relpath(test_mmu_main_ltm_sqlite_db_path, get_project_root()),
            "ltm_chroma_persist_directory": os.path.relpath(test_mmu_main_ltm_chroma_dir, get_project_root())
        },
        "data_dir": "test_data" # Ensure test_data is used as base by get_project_root() + path in MMU init
                                # Or make all paths absolute above and don't rely on data_dir here for testing.
                                # Let's make paths absolute for clarity in test setup.
    }
    # Re-assign absolute paths for clarity, MMU will resolve from project_root if paths in config are relative
    test_specific_mmu_config_override["mmu"]["mtm_db_path"] = test_mmu_main_mtm_db_path
    test_specific_mmu_config_override["mmu"]["ltm_sqlite_db_path"] = test_mmu_main_ltm_sqlite_db_path
    test_specific_mmu_config_override["mmu"]["ltm_chroma_persist_directory"] = test_mmu_main_ltm_chroma_dir
    
    # Ensure test_data directory and its subdirectories for Chroma exist for this specific test
    os.makedirs(os.path.dirname(test_mmu_main_mtm_db_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_mmu_main_ltm_sqlite_db_path), exist_ok=True)
    os.makedirs(test_mmu_main_ltm_chroma_dir, exist_ok=True)


    # --- Cleanup previous test files for MMU's own test ---
    if os.path.exists(test_mmu_main_mtm_db_path): os.remove(test_mmu_main_mtm_db_path)
    if os.path.exists(test_mmu_main_ltm_sqlite_db_path): os.remove(test_mmu_main_ltm_sqlite_db_path)
    if os.path.exists(test_mmu_main_ltm_chroma_dir): shutil.rmtree(test_mmu_main_ltm_chroma_dir)
    # Recreate Chroma dir after rmtree
    os.makedirs(test_mmu_main_ltm_chroma_dir, exist_ok=True)
    # --- End MMU test file cleanup ---

    mmu_instance = None
    try:
        print(f"\nInitializing MemoryManagementUnit with specific test config and mock embedder...")
        mmu_instance = MemoryManagementUnit(
            embedding_function=mock_embed_func_for_mmu,
            config=test_specific_mmu_config_override # Pass our test-specific config
        )
    except Exception as e:
        print(f"FATAL: Could not initialize MMU for testing: {e}")
        raise # Reraise to see traceback during testing

    print("\n--- Testing STM via MMU (with config values) ---")
    mmu_instance.add_stm_turn("user", "MMU Test (Config): User message 1")
    mmu_instance.add_stm_turn("assistant", "MMU Test (Config): Assistant response 1")
    print(f"STM History via MMU: {mmu_instance.get_stm_history()}")
    assert len(mmu_instance.get_stm_history()) <= test_specific_mmu_config_override["mmu"]["stm_max_turns"]
    print(f"STM max_turns from config ({test_specific_mmu_config_override['mmu']['stm_max_turns']}) respected.")


    print("\n--- Testing MTM via MMU (with config values) ---")
    mmu_instance.store_mtm_summary("mmu_conf_sum_01", "MMU config test summary.", {"source": "mmu_config_test"})
    retrieved_mtm_sum = mmu_instance.get_mtm_summary('mmu_conf_sum_01')
    print(f"MTM Summary via MMU: {retrieved_mtm_sum}")
    assert retrieved_mtm_sum is not None and retrieved_mtm_sum['text'] == "MMU config test summary."
    print(f"MTM using TinyDB: {mmu_instance.mtm.is_persistent}, Path: {test_mmu_main_mtm_db_path}")
    assert mmu_instance.mtm.is_persistent == test_specific_mmu_config_override["mmu"]["mtm_use_tinydb"]

    print("\n--- Testing LTM via MMU (with config values & mock embedder) ---")
    conf_conv_id_mmu = "mmu_config_conv_001"
    mmu_instance.log_ltm_interaction(conf_conv_id_mmu, 1, "user", "LTM log via MMU (config).")
    print(f"LTM History for {conf_conv_id_mmu} via MMU: {len(mmu_instance.get_ltm_conversation_history(conf_conv_id_mmu))} turns")
    
    # Test adding to vector store via MMU (will use mock embedder)
    vs_doc_id = mmu_instance.add_document_to_ltm_vector_store("LTM vector search test via MMU (config).", {"tag": "mmu_config_test"})
    print(f"Document added to VS via MMU with ID: {vs_doc_id}")
    assert vs_doc_id is not None
    time.sleep(0.5)
    search_results_vs = mmu_instance.semantic_search_ltm_vector_store("MMU vector test config")
    print(f"LTM Vector Search results via MMU: {len(search_results_vs)} found.")
    assert len(search_results_vs) > 0 if vs_doc_id else True # If add failed, search might be empty

    print("\n--- Testing Full MMU Reset (via MMU instance) ---")
    mmu_instance.reset_all_memory(confirm_reset=True)
    print(f"STM History after reset: {mmu_instance.get_stm_history()}")
    assert not mmu_instance.get_stm_history() 
    print(f"LTM Vector store count after reset: {mmu_instance.ltm.vector_store.collection.count()}")
    assert mmu_instance.ltm.vector_store.collection.count() == 0
    
    print("\nMemoryManagementUnit (with config) direct test finished.")

    # Final Cleanup
    print("\nAttempting final cleanup of MMU direct test files...")
    # ... (del mmu_instance, gc.collect, os.remove, shutil.rmtree for test_mmu_main_* paths) ...
    del mmu_instance
    gc.collect()
    time.sleep(0.1)
    if os.path.exists(test_mmu_main_mtm_db_path): os.remove(test_mmu_main_mtm_db_path)
    if os.path.exists(test_mmu_main_ltm_sqlite_db_path): os.remove(test_mmu_main_ltm_sqlite_db_path)
    if os.path.exists(test_mmu_main_ltm_chroma_dir): shutil.rmtree(test_mmu_main_ltm_chroma_dir, ignore_errors=True)
    # Remove the test_data directory if it's empty (optional)
    try:
        if os.path.exists(os.path.join(get_project_root(), "test_data")) and not os.listdir(os.path.join(get_project_root(), "test_data")):
            os.rmdir(os.path.join(get_project_root(), "test_data"))
            print("  Removed empty test_data directory.")
    except OSError as e:
        print(f"  Note: Could not remove test_data directory (it might not be empty or access issues): {e}")

    print("MMU direct test file cleanup attempt finished.")