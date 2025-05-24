# offline_chat_bot/mmu/ltm.py

import sqlite3
import uuid  # For generating unique IDs
import datetime # For timestamps
import json # For storing metadata as JSON strings
import os # For path joining
import chromadb
from chromadb.utils import embedding_functions


try:
    # Using a specific, known small model for consistency
    # If you have issues with this model, 'all-MiniLM-L6-v2' is another popular choice.
    DEFAULT_EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    # The embedding_functions.SentenceTransformerEmbeddingFunction handles model download
    default_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=DEFAULT_EMBEDDING_MODEL_NAME)
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    print("Warning: sentence-transformers library not found. VectorStoreChroma may not function with default embeddings.")
    print("Please install it: pip install sentence-transformers")
    default_ef = None
    SENTENCE_TRANSFORMER_AVAILABLE = False
except Exception as e: # Catch other potential errors during model loading
    print(f"Warning: Could not load default sentence-transformer model '{DEFAULT_EMBEDDING_MODEL_NAME}'. Error: {e}")
    print("VectorStoreChroma may not function with default embeddings.")
    default_ef = None
    SENTENCE_TRANSFORMER_AVAILABLE = False

# This will create 'ltm_database.db' in the main project directory.
# You might want to place it inside an 'data' directory later.
DATABASE_PATH = 'ltm_database.db'

CHROMA_PERSIST_DIRECTORY = "ltm_vector_store_chroma"

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

class StructuredKnowledgeBase:
    """
    Manages structured knowledge (facts, preferences, entities) in SQLite.
    """
    def __init__(self, db_path=DATABASE_PATH):
        self.db_path = db_path
        self._create_tables()

    def _get_connection(self):
        """Helper method to get a database connection."""
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _create_tables(self):
        """
        Creates tables for structured knowledge if they don't exist.
        - 'facts': For storing subject-predicate-object style facts.
        - 'user_preferences': For storing user-specific preferences.
        - 'entities': For storing general entities and their attributes.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # Facts table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS facts (
                        fact_id TEXT PRIMARY KEY,
                        subject TEXT NOT NULL,
                        predicate TEXT NOT NULL,
                        object TEXT NOT NULL,
                        source_turn_ids TEXT, -- JSON list of entry_ids from conversation_log
                        created_at TEXT NOT NULL,
                        last_accessed TEXT,
                        confidence REAL, -- e.g., 0.0 to 1.0
                        UNIQUE (subject, predicate, object) -- Basic uniqueness for a fact
                    )
                ''')
                # User Preferences table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_preferences (
                        preference_id TEXT PRIMARY KEY,
                        user_id TEXT DEFAULT 'default_user', -- Can be extended for multi-user
                        category TEXT NOT NULL,
                        key TEXT NOT NULL,
                        value TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        UNIQUE (user_id, category, key)
                    )
                ''')
                # Entities table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS entities (
                        entity_id TEXT PRIMARY KEY,
                        entity_text TEXT NOT NULL UNIQUE, -- The canonical text of the entity
                        entity_type TEXT, -- e.g., "PERSON", "LOCATION", "PROJECT"
                        attributes TEXT, -- JSON string for additional attributes
                        created_at TEXT NOT NULL
                    )
                ''')
                conn.commit()
        except sqlite3.Error as e:
            print(f"SQLite error during SKB table creation: {e}")

    def store_fact(self, subject: str, predicate: str, object_value: str,
                   source_turn_ids: list = None, confidence: float = 1.0) -> str:
        """Stores a fact (subject-predicate-object)."""
        fact_id = str(uuid.uuid4())
        created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        source_ids_json = json.dumps(source_turn_ids) if source_turn_ids else None

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO facts (fact_id, subject, predicate, object, source_turn_ids, created_at, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(subject, predicate, object) DO UPDATE SET
                        source_turn_ids = excluded.source_turn_ids, -- Or append logic
                        last_accessed = datetime('now', 'utc'),
                        confidence = excluded.confidence -- Or update based on new evidence
                ''', (fact_id, subject, predicate, object_value, source_ids_json, created_at, confidence))
                conn.commit()
                # If ON CONFLICT happened, we might want the existing fact_id.
                # For simplicity, we return the new one or the one that would have been inserted.
                # A more robust way would be to SELECT after INSERT/UPDATE.
                if cursor.lastrowid == 0: # No insert happened due to conflict (for some drivers)
                    # Try to fetch the existing one
                    cursor.execute("SELECT fact_id FROM facts WHERE subject=? AND predicate=? AND object=?",
                                   (subject, predicate, object_value))
                    row = cursor.fetchone()
                    if row: return row[0]
                return fact_id
        except sqlite3.Error as e:
            print(f"SQLite error during store_fact: {e}")
            return None

    def get_facts(self, subject: str = None, predicate: str = None, object_value: str = None) -> list:
        """Retrieves facts matching the given criteria."""
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = "SELECT * FROM facts WHERE 1=1" # Start with a true condition
                params = []
                if subject:
                    query += " AND subject LIKE ?" # Using LIKE for flexibility
                    params.append(f"%{subject}%")
                if predicate:
                    query += " AND predicate LIKE ?"
                    params.append(f"%{predicate}%")
                if object_value:
                    query += " AND object LIKE ?"
                    params.append(f"%{object_value}%")
                
                cursor.execute(query, tuple(params))
                rows = cursor.fetchall()
                # Convert to list of dicts, parsing source_turn_ids
                facts_list = []
                for row in rows:
                    row_dict = dict(row)
                    if row_dict.get('source_turn_ids'):
                        try:
                            row_dict['source_turn_ids'] = json.loads(row_dict['source_turn_ids'])
                        except json.JSONDecodeError:
                             print(f"Warning: Could not decode source_turn_ids for fact_id {row_dict.get('fact_id')}")
                             pass
                    facts_list.append(row_dict)
                return facts_list
        except sqlite3.Error as e:
            print(f"SQLite error during get_facts: {e}")
            return []

    def store_preference(self, category: str, key: str, value: str, user_id: str = "default_user") -> str:
        """Stores a user preference."""
        preference_id = str(uuid.uuid4())
        created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO user_preferences (preference_id, user_id, category, key, value, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, category, key) DO UPDATE SET
                        value = excluded.value,
                        created_at = excluded.created_at -- Or keep original created_at
                ''', (preference_id, user_id, category, key, value, created_at))
                conn.commit()
                if cursor.lastrowid == 0:
                     cursor.execute("SELECT preference_id FROM user_preferences WHERE user_id=? AND category=? AND key=?",
                                   (user_id, category, key))
                     row = cursor.fetchone()
                     if row: return row[0]
                return preference_id
        except sqlite3.Error as e:
            print(f"SQLite error during store_preference: {e}")
            return None

    def get_preference(self, category: str, key: str, user_id: str = "default_user") -> dict:
        """Retrieves a specific user preference."""
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM user_preferences
                    WHERE user_id = ? AND category = ? AND key = ?
                ''', (user_id, category, key))
                row = cursor.fetchone()
                return dict(row) if row else None
        except sqlite3.Error as e:
            print(f"SQLite error during get_preference: {e}")
            return None

    # We can add methods for entities (store_entity, get_entity) similarly later if needed.

class VectorStoreChroma:
    """
    Manages a vector store using ChromaDB for semantic search.
    """
    def __init__(self, persist_directory=CHROMA_PERSIST_DIRECTORY, collection_name="conversation_store", embedding_function=None):
        """
        Initializes the ChromaDB vector store.

        Args:
            persist_directory (str): Directory to store ChromaDB data.
            collection_name (str): Name of the collection within ChromaDB.
            embedding_function (chromadb.EmbeddingFunction, optional): 
                The embedding function to use. Defaults to a pre-configured SentenceTransformer.
                If None and SENTENCE_TRANSFORMER_AVAILABLE is False, an error will be raised.
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        if embedding_function is None:
            if SENTENCE_TRANSFORMER_AVAILABLE and default_ef is not None:
                self.embedding_function = default_ef
                print(f"VectorStoreChroma initialized with default SentenceTransformer embedding function: {DEFAULT_EMBEDDING_MODEL_NAME}")
            else:
                raise ValueError("No embedding function provided and default SentenceTransformer is not available.")
        else:
            self.embedding_function = embedding_function
            print("VectorStoreChroma initialized with provided embedding function.")

        # Ensure the persist directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            # Get or create the collection. This is idempotent.
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function 
                # metadata={"hnsw:space": "cosine"} # Optional: specify distance metric if needed
            )
            print(f"ChromaDB collection '{self.collection_name}' loaded/created successfully from '{self.persist_directory}'.")
            print(f"Current item count in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error initializing ChromaDB client or collection: {e}")
            # In a real app, might want to raise this or handle more gracefully
            self.client = None
            self.collection = None
            raise # Reraise for now to make it clear initialization failed

    def add_document(self, text_chunk: str, metadata: dict, doc_id: str = None) -> str:
        """
        Adds a text chunk (document) and its metadata to the vector store.
        The text chunk will be automatically embedded by the collection's embedding function.

        Args:
            text_chunk (str): The text content to store and embed.
            metadata (dict): A dictionary of metadata associated with the text chunk. 
                             ChromaDB allows filtering by this metadata.
                             Example: {"conversation_id": "conv123", "turn_id": "turn_abc", "type": "user_message"}
            doc_id (str, optional): A unique ID for this document. If None, a UUID will be generated.

        Returns:
            str: The ID of the added document, or None if an error occurred.
        """
        if not self.collection:
            print("Error: ChromaDB collection is not initialized.")
            return None
        
        if not doc_id:
            doc_id = str(uuid.uuid4())

        try:
            # ChromaDB's add method can take lists for batching. Here we add one.
            # It expects 'documents', 'metadatas', and 'ids' to be lists.
            self.collection.add(
                documents=[text_chunk],
                metadatas=[metadata],
                ids=[doc_id]
            )
            # print(f"Document added to Chroma: ID='{doc_id}', Text='{text_chunk[:50]}...'")
            return doc_id
        except Exception as e:
            print(f"Error adding document to ChromaDB: {e}")
            # Consider if the document ID might already exist, though `add` usually overwrites/updates.
            # `upsert` is also an option if you want to explicitly update if ID exists, or insert if not.
            # self.collection.upsert(ids=[doc_id], documents=[text_chunk], metadatas=[metadata])
            return None

    def semantic_search(self, query_text: str, top_k: int = 5, metadata_filter: dict = None) -> list:
        """
        Performs a semantic search for text chunks relevant to the query_text.

        Args:
            query_text (str): The text to search for.
            top_k (int): The number of top results to return.
            metadata_filter (dict, optional): A dictionary to filter results by metadata.
                                            Example: {"conversation_id": "conv123"}
                                            See ChromaDB docs for $eq, $ne, $in, $nin operators.
                                            Example: {"type": {"$eq": "user_message"}}

        Returns:
            list: A list of dictionaries, each containing 'id', 'text_chunk', 'metadata', and 'distance'.
                  Returns an empty list if no results or an error occurs.
        """
        if not self.collection:
            print("Error: ChromaDB collection is not initialized.")
            return []
        
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=top_k,
                where=metadata_filter,  # 'where' clause for metadata filtering
                include=['metadatas', 'documents', 'distances'] # Specify what to include in results
            )
            
            # Process results into a more usable format
            # ChromaDB query results are structured with lists for each field, corresponding to each query_text.
            # Since we send one query_text, we access the first element of these lists.
            output_results = []
            if results and results.get('ids') and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    output_results.append({
                        "id": results['ids'][0][i],
                        "text_chunk": results['documents'][0][i] if results['documents'] and results['documents'][0] else None,
                        "metadata": results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else None,
                        "distance": results['distances'][0][i] if results['distances'] and results['distances'][0] else None,
                    })
            return output_results
        except Exception as e:
            print(f"Error during semantic search in ChromaDB: {e}")
            return []

    def delete_document(self, doc_id: str) -> bool:
        """Deletes a document from the vector store by its ID."""
        if not self.collection:
            print("Error: ChromaDB collection is not initialized.")
            return False
        try:
            self.collection.delete(ids=[doc_id])
            return True
        except Exception as e:
            print(f"Error deleting document '{doc_id}' from ChromaDB: {e}")
            return False
    
    def get_document_by_id(self, doc_id: str) -> dict or None:
        """Retrieves a document by its ID."""
        if not self.collection:
            print("Error: ChromaDB collection is not initialized.")
            return None
        try:
            result = self.collection.get(
                ids=[doc_id],
                include=['metadatas', 'documents']
            )
            if result and result.get('ids') and result['ids']:
                 return {
                        "id": result['ids'][0],
                        "text_chunk": result['documents'][0] if result['documents'] else None,
                        "metadata": result['metadatas'][0] if result['metadatas'] else None,
                    }
            return None
        except Exception as e:
            print(f"Error getting document '{doc_id}' from ChromaDB: {e}")
            return None

    def clear_all_documents(self):
        """
        DANGER: Deletes all documents from the collection.
        This is useful for testing or resetting.
        ChromaDB currently doesn't have a simple `collection.clear()` method.
        One way is to delete the collection and recreate it.
        """
        if not self.client or not self.collection_name:
            print("Error: Client or collection name not available for clearing.")
            return False
        try:
            print(f"Attempting to delete and recreate collection '{self.collection_name}' to clear all documents...")
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            print(f"Collection '{self.collection_name}' cleared and recreated. Count: {self.collection.count()}")
            return True
        except Exception as e:
            print(f"Error clearing collection '{self.collection_name}': {e}")
            return False


# --- Updated LTMManager ---
class LTMManager:
    """
    Manages all Long-Term Memory components: Raw Conversation Log, Structured Knowledge Base,
    and Vector Store.
    """
    def __init__(self, db_path=DATABASE_PATH, chroma_persist_dir=CHROMA_PERSIST_DIRECTORY, embedding_function=None):
        self.db_path = db_path
        self.chroma_persist_dir = chroma_persist_dir # Store for reset
        self.embedding_function_ref = embedding_function # Store for reset

        self.raw_log = RawConversationLog(db_path)
        self.skb = StructuredKnowledgeBase(db_path)
        
        self.vector_store = VectorStoreChroma(
            persist_directory=self.chroma_persist_dir,
            embedding_function=self.embedding_function_ref
        )

    # Expose methods from raw_log
    def log_interaction(self, *args, **kwargs): return self.raw_log.log_interaction(*args, **kwargs)
    def get_conversation_history(self, *args, **kwargs): return self.raw_log.get_conversation_history(*args, **kwargs)
    def get_all_conversation_ids(self, *args, **kwargs): return self.raw_log.get_all_conversation_ids(*args, **kwargs)

    # Expose methods from skb
    def store_fact(self, *args, **kwargs): return self.skb.store_fact(*args, **kwargs)
    def get_facts(self, *args, **kwargs): return self.skb.get_facts(*args, **kwargs)
    def store_preference(self, *args, **kwargs): return self.skb.store_preference(*args, **kwargs)
    def get_preference(self, *args, **kwargs): return self.skb.get_preference(*args, **kwargs)

    # Expose methods from vector_store
    def add_document_to_vector_store(self, *args, **kwargs): return self.vector_store.add_document(*args, **kwargs)
    def semantic_search_vector_store(self, *args, **kwargs): return self.vector_store.semantic_search(*args, **kwargs)
    def delete_document_from_vector_store(self, *args, **kwargs): return self.vector_store.delete_document(*args, **kwargs)
    def get_document_from_vector_store_by_id(self, *args, **kwargs): return self.vector_store.get_document_by_id(*args, **kwargs)
    def clear_vector_store(self, *args, **kwargs): return self.vector_store.clear_all_documents(*args, **kwargs)

    def reset_ltm(self, confirm_reset=False):
        """
        DANGER: Resets all Long-Term Memory components.
        This involves deleting all data from SQLite tables used by RawConversationLog and SKB,
        and clearing the ChromaDB vector store.

        Args:
            confirm_reset (bool): Must be True to proceed with the reset.
                                  This is a safety measure.
        Returns:
            bool: True if reset was successful, False otherwise.
        """
        if not confirm_reset:
            print("LTM reset aborted. `confirm_reset` must be True.")
            return False

        print(f"WARNING: Proceeding with full LTM reset for SQLite DB: '{self.db_path}' and Chroma dir: '{self.chroma_persist_dir}'")
        
        success_sqlite = False
        success_chroma = False

        # 1. Reset SQLite tables (delete all rows)
        try:
            with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
                cursor = conn.cursor()
                print("Deleting data from 'conversation_log' table...")
                cursor.execute("DELETE FROM conversation_log")
                print("Deleting data from 'facts' table...")
                cursor.execute("DELETE FROM facts")
                print("Deleting data from 'user_preferences' table...")
                cursor.execute("DELETE FROM user_preferences")
                # Add DELETE FROM entities if you implement it fully
                # cursor.execute("DELETE FROM entities") 
                conn.commit()
                print("SQLite tables cleared.")
                success_sqlite = True
        except sqlite3.Error as e:
            print(f"SQLite error during LTM reset: {e}")
        
        # 2. Reset ChromaDB vector store
        # We can use the existing clear_all_documents method in VectorStoreChroma
        # which deletes and recreates the collection.
        print("Clearing ChromaDB vector store...")
        if self.vector_store:
            success_chroma = self.vector_store.clear_all_documents()
        else:
            print("Vector store not initialized, skipping Chroma clear.")
            # If vector_store might not be initialized, this check is good.
            # For our current LTMManager, it should always be initialized.

        if success_sqlite and success_chroma:
            print("LTM reset completed successfully.")
            return True
        else:
            print("LTM reset failed or was partial.")
            return False


if __name__ == "__main__":
    print("Testing LTMManager (including VectorStoreChroma and reset)...")
    
    test_sqlite_db = 'test_ltm_manager_full.db'
    test_chroma_dir = 'test_ltm_chroma_store' 

    import shutil
    if os.path.exists(test_chroma_dir):
        print(f"Removing previous test Chroma store: {test_chroma_dir}")
        shutil.rmtree(test_chroma_dir)
    if os.path.exists(test_sqlite_db):
        print(f"Removing previous test SQLite DB: {test_sqlite_db}")
        os.remove(test_sqlite_db)

    if not SENTENCE_TRANSFORMER_AVAILABLE:
        print("Skipping VectorStoreChroma tests as sentence-transformers is not available.")
        ltm_for_reset_test = None # Cannot test reset fully
    else:
        print(f"Initializing LTMManager with SQLite: '{test_sqlite_db}' and Chroma: '{test_chroma_dir}'")
        try:
            ltm = LTMManager(db_path=test_sqlite_db, chroma_persist_dir=test_chroma_dir)
            ltm_for_reset_test = ltm # Use this instance for reset test too
        except Exception as e:
            print(f"Failed to initialize LTMManager for testing: {e}")
            print("Skipping LTMManager tests.")
            ltm_for_reset_test = None
            exit() # Exit if LTM can't be initialized for tests

        # ... (existing vector store tests - keep them here) ...
        print("\n--- Testing Vector Store (via LTMManager) ---")
        doc1_id = ltm.add_document_to_vector_store(
            text_chunk="The weather today is sunny and warm.",
            metadata={"source": "weather_update", "day": "monday", "conversation_id": "conv_vec_001"}
        )
        print(f"Added doc1: {doc1_id}")
        doc2_id = ltm.add_document_to_vector_store(
            text_chunk="Project Phoenix is due by end of next month.",
            metadata={"source": "project_meeting", "project": "Phoenix", "conversation_id": "conv_vec_001"}
        )
        print(f"Added doc2: {doc2_id}")
        # ... more vector tests ...
        import time
        time.sleep(1)
        print("\nSearching for 'current weather':")
        results1 = ltm.semantic_search_vector_store(query_text="current weather", top_k=2)
        for res in results1: print(f"  ID: {res['id']}, Dist: {res['distance']:.4f}, Text: '{res['text_chunk']}', Meta: {res['metadata']}")


        # --- Test LTM Reset ---
        if ltm_for_reset_test: # Check if ltm was initialized successfully
            print("\n--- Testing LTM Reset ---")
            # Add some data to SKB and RawLog first
            ltm_for_reset_test.log_interaction("conv_reset_test", 1, "user", "Test log before reset")
            ltm_for_reset_test.store_fact("TestSubject", "TestPredicate", "TestObject")
            
            print(f"Items in vector store before reset: {ltm_for_reset_test.vector_store.collection.count()}")
            history_before_reset = ltm_for_reset_test.get_conversation_history("conv_reset_test")
            print(f"Log entries for 'conv_reset_test' before reset: {len(history_before_reset)}")
            facts_before_reset = ltm_for_reset_test.get_facts(subject="TestSubject")
            print(f"Facts for 'TestSubject' before reset: {len(facts_before_reset)}")

            # Attempt reset without confirmation
            print("\nAttempting reset without confirmation:")
            ltm_for_reset_test.reset_ltm(confirm_reset=False) # Should abort

            # Attempt reset WITH confirmation
            print("\nAttempting reset WITH confirmation:")
            reset_status = ltm_for_reset_test.reset_ltm(confirm_reset=True)
            print(f"Reset status: {reset_status}")

            if reset_status:
                print(f"Items in vector store after reset: {ltm_for_reset_test.vector_store.collection.count()}")
                history_after_reset = ltm_for_reset_test.get_conversation_history("conv_reset_test")
                print(f"Log entries for 'conv_reset_test' after reset: {len(history_after_reset)}")
                facts_after_reset = ltm_for_reset_test.get_facts(subject="TestSubject")
                print(f"Facts for 'TestSubject' after reset: {len(facts_after_reset)}")
            else:
                print("Reset failed, cannot verify post-reset state accurately.")
        else:
            print("\nSkipping LTM Reset test as LTMManager was not initialized.")


    print("\nLTMManager full test finished.")