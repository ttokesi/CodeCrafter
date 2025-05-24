# offline_chat_bot/mmu/ltm.py

import sqlite3
import uuid  # For generating unique IDs
import datetime # For timestamps
import json # For storing metadata as JSON strings

# Define the path for our database file.
# This will create 'ltm_database.db' in the main project directory.
# You might want to place it inside an 'data' directory later.
DATABASE_PATH = 'ltm_database.db'

class RawConversationLog:
    """
    Manages the raw conversation log stored in an SQLite database.
    Stores every interaction turn by turn.
    """
    def __init__(self, db_path=DATABASE_PATH):
        """
        Initializes the RawConversationLog.
        Connects to the SQLite database and creates the necessary table if it doesn't exist.

        Args:
            db_path (str): The path to the SQLite database file.
        """
        self.db_path = db_path
        self._create_table()

    def _get_connection(self):
        """Helper method to get a database connection."""
        # `check_same_thread=False` is used here for simplicity.
        # For more complex applications, especially with multi-threading,
        # you might need a more robust connection management strategy.
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _create_table(self):
        """
        Creates the 'conversation_log' table if it doesn't already exist.
        This table will store individual turns of a conversation.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS conversation_log (
                        entry_id TEXT PRIMARY KEY,
                        conversation_id TEXT NOT NULL,
                        turn_sequence_id INTEGER NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        llm_model_used TEXT,
                        tokens_prompt INTEGER,
                        tokens_completion INTEGER,
                        metadata TEXT, 
                        UNIQUE (conversation_id, turn_sequence_id)
                    )
                ''')
                # Creating indexes for faster queries
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_conversation_id 
                    ON conversation_log (conversation_id)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_timestamp 
                    ON conversation_log (timestamp)
                ''')
                conn.commit()
        except sqlite3.Error as e:
            print(f"SQLite error during table creation for conversation_log: {e}")
            # In a real app, you might want to raise the exception or handle it more gracefully

    def log_interaction(self,
                        conversation_id: str,
                        turn_sequence_id: int,
                        role: str,
                        content: str,
                        llm_model_used: str = None,
                        tokens_prompt: int = None,
                        tokens_completion: int = None,
                        metadata: dict = None) -> str:
        """
        Logs a single interaction turn to the database.

        Args:
            conversation_id (str): The ID of the overall conversation.
            turn_sequence_id (int): The sequential order of this turn within the conversation.
            role (str): The role of the speaker (e.g., "user", "assistant", "system", "tool_call", "tool_response").
            content (str): The actual text or structured data of the interaction.
            llm_model_used (str, optional): The LLM model used for this turn, if applicable.
            tokens_prompt (int, optional): Number of prompt tokens, if applicable.
            tokens_completion (int, optional): Number of completion tokens, if applicable.
            metadata (dict, optional): Any additional JSON-serializable metadata.

        Returns:
            str: The unique entry_id for this log entry, or None if logging failed.
        """
        entry_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        # Convert metadata dict to a JSON string for storage
        metadata_json = json.dumps(metadata) if metadata else None

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO conversation_log (
                        entry_id, conversation_id, turn_sequence_id, role, content, 
                        timestamp, llm_model_used, tokens_prompt, tokens_completion, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (entry_id, conversation_id, turn_sequence_id, role, content,
                      timestamp, llm_model_used, tokens_prompt, tokens_completion, metadata_json))
                conn.commit()
            return entry_id
        except sqlite3.IntegrityError as e:
            # This can happen if (conversation_id, turn_sequence_id) is not unique
            print(f"SQLite integrity error (likely duplicate turn_sequence_id for conversation_id): {e}")
            return None
        except sqlite3.Error as e:
            print(f"SQLite error during log_interaction: {e}")
            return None

    def get_conversation_history(self, conversation_id: str, limit: int = None, offset: int = 0) -> list:
        """
        Retrieves the history for a specific conversation, ordered by turn sequence.

        Args:
            conversation_id (str): The ID of the conversation to retrieve.
            limit (int, optional): Maximum number of turns to retrieve.
            offset (int, optional): Number of turns to skip from the beginning.

        Returns:
            list: A list of dictionaries, where each dictionary represents a turn.
                  Returns an empty list if no history is found or an error occurs.
        """
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row # This allows accessing columns by name
                cursor = conn.cursor()
                
                query = '''
                    SELECT * FROM conversation_log
                    WHERE conversation_id = ?
                    ORDER BY turn_sequence_id ASC
                '''
                params = [conversation_id]

                if limit is not None:
                    query += " LIMIT ?"
                    params.append(limit)
                if offset > 0 : # Add offset only if it is greater than 0, SQLite LIMIT takes offset as second param in LIMIT X, Y
                    if limit is None: # If no limit, we need a large number for limit before offset
                        query += " LIMIT -1" # SQLite specific way to say no limit
                    query += " OFFSET ?"
                    params.append(offset)
                
                cursor.execute(query, tuple(params))
                rows = cursor.fetchall()
                
                # Convert rows to list of dicts, parsing metadata back to dict
                history = []
                for row in rows:
                    row_dict = dict(row)
                    if row_dict.get('metadata'):
                        try:
                            row_dict['metadata'] = json.loads(row_dict['metadata'])
                        except json.JSONDecodeError:
                            # If metadata is not valid JSON, keep it as is or log an error
                            print(f"Warning: Could not decode metadata for entry_id {row_dict.get('entry_id')}")
                            pass 
                    history.append(row_dict)
                return history
        except sqlite3.Error as e:
            print(f"SQLite error during get_conversation_history: {e}")
            return []

    def get_all_conversation_ids(self) -> list:
        """
        Retrieves a list of all unique conversation_ids from the log.

        Returns:
            list: A list of unique conversation_id strings.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT DISTINCT conversation_id FROM conversation_log
                    ORDER BY conversation_id ASC
                ''')
                rows = cursor.fetchall()
                return [row[0] for row in rows] # Each row is a tuple with one element
        except sqlite3.Error as e:
            print(f"SQLite error during get_all_conversation_ids: {e}")
            return []

# --- Quick Test (Example of how to use it) ---
if __name__ == "__main__":
    # This block runs only when you execute ltm.py directly (e.g., python mmu/ltm.py)
    print("Testing RawConversationLog...")
    
    # Create an instance of the log
    raw_log = RawConversationLog(db_path='test_raw_log.db') # Use a test database

    # Log some interactions for conversation "conv_001"
    conv1_id = "conv_001"
    raw_log.log_interaction(conv1_id, 1, "user", "Hello bot!")
    raw_log.log_interaction(conv1_id, 2, "assistant", "Hello user! How can I help you today?", 
                            llm_model_used="test_model_v1", tokens_prompt=10, tokens_completion=15,
                            metadata={"source": "greeting_module", "confidence": 0.98})
    raw_log.log_interaction(conv1_id, 3, "user", "What's the weather like?")
    
    # Log some interactions for conversation "conv_002"
    conv2_id = "conv_002"
    raw_log.log_interaction(conv2_id, 1, "user", "Tell me a joke.")
    raw_log.log_interaction(conv2_id, 2, "assistant", "Why don't scientists trust atoms? Because they make up everything!",
                            metadata={"tags": ["joke", "pun"]})

    print(f"\nAll conversation IDs: {raw_log.get_all_conversation_ids()}")

    print(f"\nHistory for {conv1_id}:")
    history1 = raw_log.get_conversation_history(conv1_id)
    for turn in history1:
        print(f"  {turn['role']} (Seq {turn['turn_sequence_id']}): {turn['content']} [Model: {turn.get('llm_model_used', 'N/A')}, Meta: {turn.get('metadata')}]")

    print(f"\nHistory for {conv2_id} (limit 1):")
    history2_limited = raw_log.get_conversation_history(conv2_id, limit=1)
    for turn in history2_limited:
        print(f"  {turn['role']}: {turn['content']}")
        
    print(f"\nHistory for {conv1_id} (limit 1, offset 1):")
    history1_offset = raw_log.get_conversation_history(conv1_id, limit=1, offset=1)
    for turn in history1_offset:
        print(f"  {turn['role']}: {turn['content']}")

    print("\nRawConversationLog test finished.")
    # You can inspect 'test_raw_log.db' with a SQLite browser to see the data.