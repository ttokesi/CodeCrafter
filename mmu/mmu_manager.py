# offline_chat_bot/mmu/mmu_manager.py

import datetime
from collections import deque # Efficient for adding/removing from both ends
import json # For TinyDB if we use it for complex objects

# --- For direct execution/testing of this file ---
if __name__ == '__main__' and __package__ is None: # Only run if executed directly AND not as part of a package
    import sys
    import os
    # Get the parent directory of the current file's directory (e.g., 'offline_chat_bot')
    # This allows finding the 'mmu' package.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    # If we are here, it means mmu_manager is being run directly.
    # We need to make sure the import can resolve.
    # For direct run, we might need to temporarily adjust __package__ or use absolute import after path mod.
    # Let's try making the relative import work by ensuring the context is right.
    # OR, more simply for direct run, we can condition the import:
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

# --- Update main test block ---
if __name__ == "__main__":
    print("--- Testing ShortTermMemory ---")
    stm = ShortTermMemory(max_turns=3)
    stm.add_turn("user", "Hello there!")
    stm.add_turn("assistant", "General Kenobi!")
    stm.add_turn("user", "You are a bold one.")
    print("\nFormatted STM History:")
    print(stm.get_formatted_history(include_timestamps=False)) # Simpler output for this test
    stm.clear()

    print("\n--- Testing MediumTermMemory (In-Memory) ---")
    mtm_in_memory = MediumTermMemory(use_tinydb=False)
    mtm_in_memory.store_summary("sum_001", "User asked about Star Wars.", {"convo_id": "c1"})
    mtm_in_memory.store_entity("user_prefs", "language", "english")
    mtm_in_memory.store_task_context("current_project", {"name": "Death Star", "status": "operational"})

    print(f"Retrieved summary: {mtm_in_memory.get_summary('sum_001')['text']}")
    print(f"Retrieved language pref: {mtm_in_memory.get_entity('user_prefs', 'language')}")
    print(f"Retrieved project status: {mtm_in_memory.get_task_context('current_project')['status']}")
    
    print("\nRecent summaries (in-memory):")
    for summ in mtm_in_memory.get_recent_summaries(max_count=1): # Test with 1 for brevity
        print(f"  - {summ['text']} (Updated: {summ.get('metadata', {}).get('last_updated')})")

    mtm_in_memory.clear_session_data()
    print(f"Summary after clear: {mtm_in_memory.get_summary('sum_001')}")


    if TINYDB_AVAILABLE:
        print("\n--- Testing MediumTermMemory (TinyDB) ---")
        test_mtm_db_path = 'test_mtm_store.json'
        # Clean up previous test MTM store if it exists
        import os
        if os.path.exists(test_mtm_db_path):
            # Attempt to close any lingering handles before removing
            # This is a bit of a guess if we don't have the instance.
            # For a clean test, ensuring instances are closed is better.
            try:
                temp_db_to_close = TinyDB(test_mtm_db_path) # Re-open to get a handle
                temp_db_to_close.close()
            except Exception: # If it fails (e.g. file corrupted or already locked)
                pass
            os.remove(test_mtm_db_path)

        mtm_persistent = MediumTermMemory(use_tinydb=True, db_path=test_mtm_db_path)
        mtm_persistent.store_summary("sum_t001", "Persistent summary about project requirements.", 
                                     {"project_id": "p789", "version": 1})
        mtm_persistent.store_summary("sum_t002", "User preferences for notifications discussed.",
                                     {"user_id": "user123"})
        mtm_persistent.store_entity("project_alpha", "status", "pending_review")
        mtm_persistent.store_task_context("active_tool", {"name": "calculator", "last_used": "10:30"})
        
        # Explicitly close the first instance if it's no longer needed after writing
        if hasattr(mtm_persistent, 'db') and mtm_persistent.db:
            print(f"Closing initial TinyDB instance for {test_mtm_db_path}")
            mtm_persistent.db.close()

        # Re-initialize to test persistence (data should load from file)
        print("Re-initializing MTM from TinyDB file to test persistence...")
        mtm_persistent_load_test = MediumTermMemory(use_tinydb=True, db_path=test_mtm_db_path)
        
        retrieved_pers_sum = mtm_persistent_load_test.get_summary("sum_t001")
        print(f"Retrieved persistent summary: {retrieved_pers_sum['text'] if retrieved_pers_sum else 'Not found'}")
        
        retrieved_pers_entity = mtm_persistent_load_test.get_entity("project_alpha", "status")
        print(f"Retrieved persistent entity: {retrieved_pers_entity if retrieved_pers_entity else 'Not found'}")

        print("\nRecent summaries (TinyDB):")
        for summ in mtm_persistent_load_test.get_recent_summaries(max_count=2):
            print(f"  - {summ['text']} (Updated: {summ.get('metadata', {}).get('last_updated')})")

        mtm_persistent_load_test.clear_session_data() # This will truncate the tables in the JSON file
        print(f"Summary after persistent clear: {mtm_persistent_load_test.get_summary('sum_t001')}")
        
        # Check if file is empty or tables are empty
        if os.path.exists(test_mtm_db_path):
            content_to_print = "File is empty or not valid JSON after clear."
            try:
                with open(test_mtm_db_path, 'r') as f:
                    file_content_str = f.read()
                    if file_content_str.strip(): # Check if the string is not empty after stripping whitespace
                        content = json.loads(file_content_str) # Use json.loads for a string
                        content_to_print = content
                    else: # File was empty or only whitespace
                        content_to_print = "{}" # Represent as empty JSON object string or dict
            except json.JSONDecodeError:
                # This will catch if the file has content but it's not valid JSON
                print(f"Warning: {test_mtm_db_path} contains non-JSON data after clear, or was unexpectedly formatted.")
                pass # content_to_print remains the default error message
            except Exception as e:
                print(f"An unexpected error occurred reading {test_mtm_db_path}: {e}")
                pass


            print(f"Content of {test_mtm_db_path} after clear: {content_to_print}")
            
            # It's good practice to close the MTM TinyDB instance before removing the file,
            # though Python's garbage collection usually handles it.
            if hasattr(mtm_persistent_load_test, 'db') and mtm_persistent_load_test.db:
                mtm_persistent_load_test.db.close()
            
            try:
                os.remove(test_mtm_db_path) 
                print(f"Successfully removed {test_mtm_db_path}")
            except PermissionError as e:
                print(f"Still could not remove {test_mtm_db_path}: {e}. File might still be locked.")
            except Exception as e:
                print(f"Error removing {test_mtm_db_path}: {e}")

    print("\nShortTermMemory and MediumTermMemory tests finished.")