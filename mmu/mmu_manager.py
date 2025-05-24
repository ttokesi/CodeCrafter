# offline_chat_bot/mmu/mmu_manager.py

import datetime
from collections import deque # Efficient for adding/removing from both ends
import json # For TinyDB if we use it for complex objects
import time 
import os
import shutil

# --- For direct execution/testing of this file ---
if __name__ == '__main__' and __package__ is None: # Only run if executed directly AND not as part of a package
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    try:
        from .ltm import LTMManager # Try relative first (for package use)
    except ImportError:
        from mmu.ltm import LTMManager # Fallback for direct run after path mod
else: # If imported as part of a package
    from .ltm import LTMManager # Standard relative import
# --- End of testing specific import logic ---

try:
    from tinydb import TinyDB, Query, where
    from tinydb.storages import JSONStorage
    from tinydb.middlewares import CachingMiddleware
    TINYDB_AVAILABLE = True
except ImportError:
    TINYDB_AVAILABLE = False
    print("Warning: tinydb library not found. MTM will be in-memory only.")
    class TinyDB: pass 
    class Query: pass
    def where(key): return None 

DEFAULT_STM_MAX_TURNS = 10 # Default number of recent turns to keep in STM
MTM_DATABASE_PATH = 'mtm_store.json' # Default path if TinyDB is used

LTM_SQLITE_DB_PATH = 'ltm_main_database.db'
LTM_CHROMA_PERSIST_DIRECTORY = "ltm_main_vector_store"

class ShortTermMemory:
    """
    Manages Short-Term Memory (STM) for the immediate conversation context.
    Uses a deque to store recent conversation turns.
    """
    def __init__(self, max_turns: int = DEFAULT_STM_MAX_TURNS):
        """
        Initializes the ShortTermMemory.

        Args:
            max_turns (int): The maximum number of recent turns to keep in STM.
        """
        self.max_turns = max_turns
        # A deque is a double-ended queue, efficient for appending and popping from either end.
        # We'll store dictionaries representing turns: {"role": "user/assistant", "content": "...", "timestamp": "..."}
        self.history = deque(maxlen=max_turns)
        self.scratchpad_content = "" # For ReAct style agent thoughts

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
    def __init__(self, use_tinydb: bool = False, db_path: str = MTM_DATABASE_PATH):
        """
        Initializes the MediumTermMemory.

        Args:
            use_tinydb (bool): If True and TinyDB is available, use TinyDB for persistence.
                               Otherwise, MTM is in-memory only.
            db_path (str): Path to the TinyDB JSON file if use_tinydb is True.
        """
        self.is_persistent = False
        self.db = None # For TinyDB instance
        self.summaries_table = None
        self.entities_table = None
        self.context_table = None

        if use_tinydb and TINYDB_AVAILABLE:
            try:
                # Using CachingMiddleware can improve performance for frequent reads
                self.db = TinyDB(db_path, storage=CachingMiddleware(JSONStorage))
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
                 stm_max_turns: int = DEFAULT_STM_MAX_TURNS,
                 mtm_use_tinydb: bool = False,
                 mtm_db_path: str = MTM_DATABASE_PATH,
                 ltm_sqlite_db_path: str = LTM_SQLITE_DB_PATH,
                 ltm_chroma_persist_dir: str = LTM_CHROMA_PERSIST_DIRECTORY,
                 #ltm_embedding_function = None # Allow passing custom embedding func to LTM
                ):
        """
        Initializes all memory components.

        Args:
            stm_max_turns (int): Max turns for Short-Term Memory.
            mtm_use_tinydb (bool): Whether Medium-Term Memory should use TinyDB for persistence.
            mtm_db_path (str): Path for MTM's TinyDB file (if used).
            ltm_sqlite_db_path (str): Path for LTM's SQLite database file.
            ltm_chroma_persist_dir (str): Directory for LTM's ChromaDB persistence.
            ltm_embedding_function: Embedding function to pass to LTM's VectorStore.
                                   If None, LTMManager will use its default.
        """
        print("Initializing MemoryManagementUnit...")
        self.stm = ShortTermMemory(max_turns=stm_max_turns)
        print(f"  STM initialized (max_turns={stm_max_turns}).")

        self.mtm = MediumTermMemory(use_tinydb=mtm_use_tinydb, db_path=mtm_db_path)
        print(f"  MTM initialized (persistent={self.mtm.is_persistent}, path='{mtm_db_path if mtm_use_tinydb else 'in-memory'}').")
        
        # LTMManager's __init__ already has defaults for its paths if not provided,
        # but we pass them explicitly here from MMU's config.
        self.ltm = LTMManager(
            db_path=ltm_sqlite_db_path,
            chroma_persist_dir=ltm_chroma_persist_dir,
            #embedding_function=ltm_embedding_function
        )
        print(f"  LTM initialized (SQLite='{ltm_sqlite_db_path}', Chroma_dir='{ltm_chroma_persist_dir}').")
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

# --- Updated __main__ to test MemoryManagementUnit ---
if __name__ == "__main__":
    print("--- Testing MemoryManagementUnit ---")

    # Define paths for test databases for MMU context
    test_mmu_mtm_db_path = 'test_mmu_mtm_store.json'
    test_mmu_ltm_sqlite_db_path = 'test_mmu_ltm_sqlite.db'
    test_mmu_ltm_chroma_dir = 'test_mmu_ltm_chroma'

    # --- Cleanup previous test files for MMU ---
    if os.path.exists(test_mmu_mtm_db_path): os.remove(test_mmu_mtm_db_path)
    if os.path.exists(test_mmu_ltm_sqlite_db_path): os.remove(test_mmu_ltm_sqlite_db_path)
    import shutil # For rmtree
    if os.path.exists(test_mmu_ltm_chroma_dir): shutil.rmtree(test_mmu_ltm_chroma_dir)
    # --- End MMU test file cleanup ---

    # Initialize MMU
    # For testing, we'll enable TinyDB for MTM to see its path logging.
    # LTM will use its default embedding function (sentence-transformer).
    mmu_instance = MemoryManagementUnit(
        mtm_use_tinydb=TINYDB_AVAILABLE, # Only use TinyDB if available for test
        mtm_db_path=test_mmu_mtm_db_path,
        ltm_sqlite_db_path=test_mmu_ltm_sqlite_db_path,
        ltm_chroma_persist_dir=test_mmu_ltm_chroma_dir
    )

    print("\n--- Testing STM via MMU ---")
    mmu_instance.add_stm_turn("user", "MMU Test: User message 1")
    mmu_instance.add_stm_turn("assistant", "MMU Test: Assistant response 1")
    print(f"STM History via MMU: {mmu_instance.get_stm_history()}")
    mmu_instance.update_stm_scratchpad("MMU STM scratchpad test.")
    print(f"STM Scratchpad via MMU: {mmu_instance.get_stm_scratchpad()}")

    print("\n--- Testing MTM via MMU ---")
    mmu_instance.store_mtm_summary("mmu_sum_01", "MMU test summary.", {"source": "mmu_test"})
    print(f"MTM Summary via MMU: {mmu_instance.get_mtm_summary('mmu_sum_01')}")
    mmu_instance.store_mtm_entity("mmu_cat", "mmu_key", "mmu_value")
    print(f"MTM Entity via MMU: {mmu_instance.get_mtm_entity('mmu_cat', 'mmu_key')}")

    print("\n--- Testing LTM via MMU ---")
    conv_id_mmu = "mmu_conv_001"
    mmu_instance.log_ltm_interaction(conv_id_mmu, 1, "user", "LTM log via MMU.")
    print(f"LTM History for {conv_id_mmu} via MMU: {len(mmu_instance.get_ltm_conversation_history(conv_id_mmu))} turns")
    
    # Check if the vector_store component of LTM is usable
    # LTMManager's vector_store is an instance of VectorStoreChroma
    # VectorStoreChroma's __init__ would raise an error or set self.collection to None if embedding_function failed.
    # A more direct check is if the embedding function on the collection is set.
    can_test_vector_store = False
    if hasattr(mmu_instance.ltm, 'vector_store') and mmu_instance.ltm.vector_store and \
    hasattr(mmu_instance.ltm.vector_store, 'collection') and mmu_instance.ltm.vector_store.collection and \
    hasattr(mmu_instance.ltm.vector_store.collection, '_embedding_function') and \
    mmu_instance.ltm.vector_store.collection._embedding_function is not None:
        can_test_vector_store = True

    if can_test_vector_store:
        print("  LTM Vector Store seems available for testing via MMU.")
        mmu_instance.add_document_to_ltm_vector_store("LTM vector search test via MMU.", {"tag": "mmu_test"})
        time.sleep(0.5) # Give chroma a moment
        search_results = mmu_instance.semantic_search_ltm_vector_store("MMU vector test")
        print(f"LTM Vector Search results via MMU: {len(search_results)} found.")
        if search_results:
            print(f"  Top result text: '{search_results[0]['text_chunk']}'")
    else:
        print("  Skipping LTM vector store tests via MMU as its embedding function seems unavailable (check ltm.py output for warnings).")


    print("\n--- Testing Full MMU Reset ---")
    # Add a bit more data before full reset
    mmu_instance.add_stm_turn("user", "Another STM turn before reset.")
    mmu_instance.store_mtm_summary("mmu_sum_02", "Another MTM summary before reset.")
    mmu_instance.log_ltm_interaction(conv_id_mmu, 2, "assistant", "Another LTM log before reset.")

    # Attempt reset without confirmation
    print("\nAttempting full MMU reset (no confirm):")
    mmu_instance.reset_all_memory(confirm_reset=False)

    # Attempt reset WITH confirmation
    print("\nAttempting full MMU reset (WITH confirm):")
    reset_success = mmu_instance.reset_all_memory(confirm_reset=True)
    print(f"Full MMU reset status: {reset_success}")

    if reset_success:
        print(f"STM History after reset: {mmu_instance.get_stm_history()}") # Should be empty
        print(f"MTM Summary 'mmu_sum_01' after reset: {mmu_instance.get_mtm_summary('mmu_sum_01')}") # Should be None
        print(f"LTM History for {conv_id_mmu} after reset: {len(mmu_instance.get_ltm_conversation_history(conv_id_mmu))} turns") # Should be 0
        if can_test_vector_store: # Use the same check for post-reset verification
            # After reset, the collection is re-created, so we re-check its count.
            # The LTMManager.reset_ltm calls vector_store.clear_all_documents which recreates the collection.
            if hasattr(mmu_instance.ltm.vector_store, 'collection') and mmu_instance.ltm.vector_store.collection:
                print(f"LTM Chroma collection count after reset: {mmu_instance.ltm.vector_store.collection.count()}") # Should be 0
            else:
                print("LTM Chroma collection seems unavailable after reset.")
    
    print("\nMemoryManagementUnit test finished.")

    # --- Final Cleanup of MMU test files (handles potential TinyDB lock) ---
    if TINYDB_AVAILABLE and os.path.exists(test_mmu_mtm_db_path):
        if hasattr(mmu_instance.mtm, 'db') and mmu_instance.mtm.db:
             mmu_instance.mtm.db.close() # Close MTM's TinyDB if it was used
        del mmu_instance.mtm # Help release MTM
    
    # LTM's ChromaDB might also hold locks, though its reset is more robust.
    # Explicitly delete the MMU instance to help release all its components.
    del mmu_instance 
    # import gc
    # gc.collect() # Optional garbage collect

    print("\nAttempting final cleanup of MMU test files...")
    files_to_remove = [test_mmu_mtm_db_path, test_mmu_ltm_sqlite_db_path]
    dirs_to_remove = [test_mmu_ltm_chroma_dir]

    for f_path in files_to_remove:
        if os.path.exists(f_path):
            try:
                os.remove(f_path)
                print(f"  Removed {f_path}")
            except Exception as e:
                print(f"  Could not remove {f_path}: {e}")
    
    for d_path in dirs_to_remove:
        if os.path.exists(d_path):
            try:
                shutil.rmtree(d_path)
                print(f"  Removed directory {d_path}")
            except Exception as e:
                print(f"  Could not remove directory {d_path}: {e}")
    print("Final cleanup attempt finished.")