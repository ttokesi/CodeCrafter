# offline_chat_bot/mmu/ltm.py

import sqlite3
import uuid  # For generating unique IDs
from datetime import datetime, timezone # Ensure timezone is imported here
import json # For storing metadata as JSON strings
import os # For path joining
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings 
import time # <--- ADD THIS IMPORT
import gc

class RawConversationLog:
    # ... (init as confirmed, with self._conn for :memory:) ...
    def __init__(self, db_path: str):
        if not isinstance(db_path, str) or not db_path:
            raise ValueError("RawConversationLog requires a valid db_path string.")
        self.db_path = db_path
        self._conn = None 
        if self.db_path == ":memory:":
            # For in-memory, create and hold the connection.
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            #print(f"DEBUG: RawConversationLog (:memory:) created persistent connection: {self._conn}")
        #else:
            #print(f"DEBUG: RawConversationLog (file: {self.db_path}) will use per-operation connections.")
        self._create_table()

    def _get_connection(self):
        if self._conn: # If we have a persistent connection (for :memory:)
            return self._conn
        else: # For file-based DBs, create a new connection each time
            return sqlite3.connect(self.db_path, check_same_thread=False)

    def _create_table(self):
        # print(f"DEBUG (_create_table): Attempting to create table in {self.db_path}")
        try:
            # Use 'with' for transaction management. 
            # If self._conn exists, it's used. If not, a new one is made and closed by 'with'.
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
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_conversation_id 
                    ON conversation_log (conversation_id)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_timestamp 
                    ON conversation_log (timestamp)
                ''')
                conn.commit()
                #print(f"DEBUG (_create_table): Table creation/check successful for {self.db_path}") # Temporary debug
        except sqlite3.Error as e:
            # This print is important. If it appears, table creation failed.
            #print(f"CRITICAL SQLITE ERROR in _create_table for {self.db_path}: {e}") 
            raise # Re-raise the exception to make the test fail clearly if table creation fails

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
        timestamp = datetime.now(timezone.utc).isoformat()
        
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
    
    def close(self): # Add a close method
        """Closes the persistent connection if one is held (for :memory: databases)."""
        if self._conn:
            #print(f"DEBUG: Closing RawConversationLog persistent connection for {self.db_path}")
            self._conn.close()
            self._conn = None

class StructuredKnowledgeBase:
    """
    Manages structured knowledge (facts, preferences, entities) in SQLite.
    """
    def __init__(self, db_path: str):
        if not isinstance(db_path, str) or not db_path:
            raise ValueError("StructuredKnowledgeBase requires a valid db_path string.")
        self.db_path = db_path
        self._conn = None # Attribute to hold the connection if :memory:
        
        if self.db_path == ":memory:":
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            #print(f"DEBUG: SKB (:memory:) created persistent connection: {self._conn}") # Optional
        #else:
            #print(f"DEBUG: SKB (file: {self.db_path}) will use per-operation connections.") # Optional
            
        self._create_tables()

    def _get_connection(self):
        """Helper method to get a database connection."""
        if self._conn: # If we have a persistent connection (for :memory:)
            return self._conn
        else: # For file-based DBs, create a new connection each time
            return sqlite3.connect(self.db_path, check_same_thread=False)

    def _create_tables(self):
        """
        Creates tables for structured knowledge if they don't exist.
        """
        #print(f"DEBUG (SKB _create_tables): Attempting for {self.db_path}") # Optional debug
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # Facts table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS facts (
                        fact_id TEXT PRIMARY KEY,
                        subject TEXT NOT NULL,
                        predicate TEXT NOT NULL,
                        object TEXT NOT NULL, -- Changed from object_value to object to match Design Doc & common use
                        source_turn_ids TEXT, 
                        created_at TEXT NOT NULL,
                        last_accessed TEXT,
                        confidence REAL, 
                        UNIQUE (subject, predicate, object) 
                    )
                ''')
                # User Preferences table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_preferences (
                        preference_id TEXT PRIMARY KEY,
                        user_id TEXT DEFAULT 'default_user', 
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
                        entity_text TEXT NOT NULL UNIQUE, 
                        entity_type TEXT, 
                        attributes TEXT, 
                        created_at TEXT NOT NULL
                    )
                ''')
                conn.commit() # Not strictly needed with 'with conn:', but ensures it for all cases
                #print(f"DEBUG (SKB _create_tables): Successful for {self.db_path}") # Optional debug
        except sqlite3.Error as e:
            print(f"CRITICAL SQLITE ERROR in SKB _create_tables for {self.db_path}: {e}")
            raise

    def store_fact(self, subject: str, predicate: str, object_value: str, # Param name object_value
                   source_turn_ids: list = None, confidence: float = 1.0) -> str:
        fact_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        source_ids_json = json.dumps(source_turn_ids) if source_turn_ids else None

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # SQL uses 'object' column, method param is 'object_value'
                cursor.execute(''' 
                    INSERT INTO facts (fact_id, subject, predicate, object, source_turn_ids, created_at, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(subject, predicate, object) DO UPDATE SET
                        source_turn_ids = excluded.source_turn_ids, 
                        last_accessed = datetime('now', 'utc'), 
                        confidence = excluded.confidence 
                ''', (fact_id, subject, predicate, object_value, source_ids_json, created_at, confidence))
                # conn.commit() is handled by 'with conn:' for file DBs
                if self.db_path == ":memory:" and self._conn: # Explicit commit for persistent :memory:
                    self._conn.commit()
                
                # Determine the ID that was actually used (new or existing due to conflict)
                # This logic might need adjustment if fact_id is not the one kept on conflict.
                # A robust way is to SELECT after the operation.
                if cursor.lastrowid == 0: 
                    cursor.execute("SELECT fact_id FROM facts WHERE subject=? AND predicate=? AND object=?",
                                   (subject, predicate, object_value))
                    row = cursor.fetchone()
                    if row: return row[0]
                return fact_id # Return initially generated fact_id or the one from SELECT
        except sqlite3.Error as e:
            print(f"SQLite error during store_fact: {e}")
            return None

    def get_facts(self, subject: str = None, predicate: str = None, object_value: str = None) -> list: # Param name object_value
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = "SELECT * FROM facts WHERE 1=1" 
                params = []
                if subject:
                    query += " AND subject LIKE ?" 
                    params.append(f"%{subject}%")
                if predicate:
                    query += " AND predicate LIKE ?"
                    params.append(f"%{predicate}%")
                if object_value: # Query against 'object' column in DB
                    query += " AND object LIKE ?" 
                    params.append(f"%{object_value}%")
                
                cursor.execute(query, tuple(params))
                rows = cursor.fetchall()
                facts_list = []
                for row in rows:
                    row_dict = dict(row)
                    if row_dict.get('source_turn_ids'):
                        try:
                            row_dict['source_turn_ids'] = json.loads(row_dict['source_turn_ids'])
                        except json.JSONDecodeError:
                             print(f"Warning: Could not decode source_turn_ids for fact_id {row_dict.get('fact_id')}")
                             pass # Keep as string if decode fails
                    facts_list.append(row_dict)
                return facts_list
        except sqlite3.Error as e:
            print(f"SQLite error during get_facts: {e}")
            return []

    # --- Methods for user_preferences (ensure they also use 'with self._get_connection()') ---
    def store_preference(self, category: str, key: str, value: str, user_id: str = "default_user") -> str:
        preference_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO user_preferences (preference_id, user_id, category, key, value, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, category, key) DO UPDATE SET
                        value = excluded.value,
                        created_at = excluded.created_at 
                ''', (preference_id, user_id, category, key, value, created_at))
                if self.db_path == ":memory:" and self._conn:
                    self._conn.commit()
                
                if cursor.lastrowid == 0: # Check if conflict occurred
                     cursor.execute("SELECT preference_id FROM user_preferences WHERE user_id=? AND category=? AND key=?",
                                   (user_id, category, key))
                     row = cursor.fetchone()
                     if row: return row[0]
                return preference_id
        except sqlite3.Error as e:
            print(f"SQLite error during store_preference: {e}")
            return None

    def get_preference(self, category: str, key: str, user_id: str = "default_user") -> dict:
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

    def close(self): # Add a close method
        """Closes the persistent connection if one is held (for :memory: databases)."""
        if self._conn:
            #print(f"DEBUG: Closing SKB persistent :memory: connection.") # Optional
            self._conn.close()
            self._conn = None

    # We can add methods for entities (store_entity, get_entity) similarly later if needed.

class VectorStoreChroma:
    """
    Manages a vector store using ChromaDB for semantic search.
    Requires an externally provided embedding function.
    """
    def __init__(self, 
                 persist_directory: str, 
                 embedding_function, # Now a required argument
                 collection_name: str = "conversation_store"):
        """
        Initializes the ChromaDB vector store.

        Args:
            persist_directory (str): Directory to store ChromaDB data.
            embedding_function (callable): A function that takes a list of texts
                                           and returns a list of embeddings.
                                           ChromaDB will use this for new documents.
            collection_name (str): Name of the collection within ChromaDB.
        """
        if not callable(embedding_function):
            raise TypeError("embedding_function must be a callable (e.g., a function or method).")

        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_function = embedding_function # Store it, Chroma needs it

        os.makedirs(self.persist_directory, exist_ok=True)
        
        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            # get_or_create_collection will use the provided embedding_function
            # for new documents. For querying with text, it will also use it.
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function 
                # If your embedding_function is a custom class that Chroma needs to instantiate,
                # you might need to pass it as embedding_function=MyEmbeddingFunction()
                # But if it's just a function like lsw.generate_embedding, passing the function itself is fine.
                # ChromaDB's default SentenceTransformerEmbeddingFunction is a class.
                # If lsw.generate_embedding needs to be wrapped in a class that Chroma expects,
                # we'll do that when creating the function to pass in.
                # For now, assume Chroma can use the passed callable directly.
            )
            print(f"VectorStoreChroma: ChromaDB collection '{self.collection_name}' loaded/created successfully from '{self.persist_directory}'.")
            #print(f"  Using provided embedding function: {embedding_function.__name__ if hasattr(embedding_function, '__name__') else type(embedding_function)}")
            print(f"  Current item count in collection: {self.collection.count()}")
        except Exception as e:
            print(f"VectorStoreChroma: Error initializing ChromaDB client or collection: {e}")
            self.client = None
            self.collection = None
            raise

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
    def __init__(self, 
                 db_path: str, # Required SQLite DB path
                 chroma_persist_dir: str, # Required ChromaDB persistence directory
                 embedding_function # Required embedding function for VectorStoreChroma
                ):
        """
        Initializes all Long-Term Memory components.

        Args:
            db_path (str): Path for LTM's SQLite database file.
            chroma_persist_dir (str): Directory for LTM's ChromaDB persistence.
            embedding_function (callable): Embedding function to be used by VectorStoreChroma.
        """
        if not db_path or not isinstance(db_path, str):
            raise ValueError("LTMManager requires a valid db_path string.")
        if not chroma_persist_dir or not isinstance(chroma_persist_dir, str):
            raise ValueError("LTMManager requires a valid chroma_persist_dir string.")
        if not callable(embedding_function):
            raise TypeError("LTMManager requires a callable embedding_function for its VectorStore.")

        self.db_path = db_path
        # self.chroma_persist_dir = chroma_persist_dir # Not needed to store if passed directly
        
        #print(f"LTMManager: Initializing with DB: '{db_path}', Chroma Dir: '{chroma_persist_dir}'")

        self.raw_log = RawConversationLog(db_path=self.db_path)
        self.skb = StructuredKnowledgeBase(db_path=self.db_path)
        
        self.vector_store = VectorStoreChroma(
            persist_directory=chroma_persist_dir, # Pass it through
            embedding_function=embedding_function   # Pass it through
        )
        #print("LTMManager: All LTM components initialized.")

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
        DANGER: Resets all Long-Term Memory components for this LTMManager instance.
        This involves deleting all data from SQLite tables used by RawConversationLog and SKB,
        and clearing the ChromaDB vector store associated with this LTMManager.

        Args:
            confirm_reset (bool): Must be True to proceed with the reset.
        Returns:
            bool: True if reset was successful, False otherwise.
        """
        if not confirm_reset:
            print("LTM reset aborted. `confirm_reset` must be True.")
            return False

        # Get chroma_persist_dir from the vector_store instance for the warning message
        chroma_dir_for_warning = "unknown (vector_store not initialized)"
        if hasattr(self, 'vector_store') and self.vector_store and hasattr(self.vector_store, 'persist_directory'):
            chroma_dir_for_warning = self.vector_store.persist_directory
            
        print(f"WARNING: Proceeding with full LTM reset for SQLite DB: '{self.db_path}' and Chroma dir: '{chroma_dir_for_warning}'")
        
        success_sqlite = False
        success_chroma = False

        # 1. Reset SQLite tables (delete all rows)
        try:
            # Ensure raw_log and skb are initialized
            if hasattr(self, 'raw_log') and self.raw_log and hasattr(self, 'skb') and self.skb:
                with sqlite3.connect(self.db_path, check_same_thread=False) as conn: # Assuming check_same_thread for simplicity
                    cursor = conn.cursor()
                    # Check if tables exist before trying to delete from them (more robust)
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='conversation_log'")
                    if cursor.fetchone():
                        print("  LTM Reset: Deleting data from 'conversation_log' table...")
                        cursor.execute("DELETE FROM conversation_log")
                    
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='facts'")
                    if cursor.fetchone():
                        print("  LTM Reset: Deleting data from 'facts' table...")
                        cursor.execute("DELETE FROM facts")

                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_preferences'")
                    if cursor.fetchone():
                        print("  LTM Reset: Deleting data from 'user_preferences' table...")
                        cursor.execute("DELETE FROM user_preferences")
                    
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='entities'")
                    if cursor.fetchone(): # If entities table exists
                        print("  LTM Reset: Deleting data from 'entities' table...")
                        cursor.execute("DELETE FROM entities")
                    conn.commit()
                print("  LTM Reset: SQLite tables cleared.")
                success_sqlite = True
            else:
                print("  LTM Reset: Raw log or SKB not initialized, skipping SQLite table clear.")
        except sqlite3.Error as e:
            print(f"  LTM Reset: SQLite error during LTM reset: {e}")
        
        # 2. Reset ChromaDB vector store
        print("  LTM Reset: Clearing ChromaDB vector store...")
        if hasattr(self, 'vector_store') and self.vector_store:
            success_chroma = self.vector_store.clear_all_documents() # This method deletes and recreates collection
        else:
            print("  LTM Reset: Vector store not initialized, skipping Chroma clear.")
            success_chroma = True # Consider it a success if no VS to clear

        if success_sqlite and success_chroma:
            print("LTM reset completed successfully for this LTMManager instance.")
            return True
        else:
            print("LTM reset failed or was partial for this LTMManager instance.")
            return False


# In offline_chat_bot/mmu/ltm.py

if __name__ == "__main__":
    print("--- Testing LTMManager (RawConversationLog, SKB, VectorStoreChroma with mock embedding) ---")
    
    # Define paths for test databases/stores for this ltm.py direct test
    test_ltm_sqlite_db = 'test_ltm_direct_sqlite.db'
    test_ltm_chroma_dir = 'test_ltm_direct_chroma_store'

    # Cleanup previous test files for this specific test run
    import shutil # Ensure shutil is imported if not already global in this file
    if os.path.exists(test_ltm_chroma_dir):
        print(f"Removing previous test Chroma store: {test_ltm_chroma_dir}")
        shutil.rmtree(test_ltm_chroma_dir)
    if os.path.exists(test_ltm_sqlite_db):
        print(f"Removing previous test SQLite DB: {test_ltm_sqlite_db}")
        os.remove(test_ltm_sqlite_db)

    # --- Mock Embedding Function for testing LTMManager directly ---
    # This simulates what an LSW embedding function would provide.
    # It needs to return a list of embeddings (list of lists of floats).
    # For simplicity, our mock will return fixed-size lists of zeros.
    # The actual embedding values don't matter for testing VectorStoreChroma's mechanics.
	
    MOCK_EMBEDDING_DIM = 10 

    class MockLtmTestEmbeddingFunction(EmbeddingFunction):
        def __init__(self, dim: int = 10):
            self.dim = dim
            #print(f"  LTM_TEST_MOCK_EMBEDDER_CLASS: Initialized with dim={self.dim}")

        # This is the method ChromaDB will call.
        # It MUST be named __call__ and take 'self' and 'input'.
        def __call__(self, input: Documents) -> Embeddings:
            # 'input' will be a list of strings (Documents is type alias for Sequence[Document])
            # 'Embeddings' is a type alias for Sequence[Embedding] (list of lists of floats)
            if not input: # Handle empty input list
                return []
                
            #print(f"  LTM_TEST_MOCK_EMBEDDER_CLASS: __call__ received {len(input)} document(s). Example: '{str(input[0])[:30] if input else 'N/A'}'")
            
            embeddings_list: Embeddings = []
            for _ in range(len(input)):
                embeddings_list.append([0.0] * self.dim)
            return embeddings_list

    # --- End Mock Embedding Function CLASS ---

    ltm = None
    try:
        # Instantiate the mock embedding function class
        mock_embedder_instance = MockLtmTestEmbeddingFunction(dim=10)

        ltm = LTMManager(
            db_path=test_ltm_sqlite_db, 
            chroma_persist_dir=test_ltm_chroma_dir,
            embedding_function=mock_embedder_instance # Pass the INSTANCE of the class
        )
    except Exception as e:
        print(f"Failed to initialize LTMManager for testing: {e}")
        print("Aborting LTMManager direct tests.")
        exit()

    # --- Test RawConversationLog via LTMManager (as before) ---
    print("\n--- Testing Raw Log (via LTMManager) ---")
    conv1_id_ltm_test = "conv_ltm_test_001"
    ltm.log_interaction(conv1_id_ltm_test, 1, "user", "Hello LTM test bot!")
    entry2_id_ltm_test = ltm.log_interaction(conv1_id_ltm_test, 2, "assistant", "Hello! LTM is working.",
                                             metadata={"module": "ltm_test"})
    # ... (other raw log tests if you have them)

    # --- Test StructuredKnowledgeBase via LTMManager (as before) ---
    print("\n--- Testing SKB (via LTMManager) ---")
    ltm.store_fact("LTM Test Subject", "is_testing", "SKB component", source_turn_ids=[entry2_id_ltm_test])
    skb_facts_ltm_test = ltm.get_facts(subject="LTM Test Subject")
    print(f"SKB facts found: {len(skb_facts_ltm_test)}")
    # ... (other SKB tests)

    # --- Test VectorStoreChroma via LTMManager (now uses mock embedder) ---
    print("\n--- Testing Vector Store (via LTMManager with mock embedder) ---")
    doc1_id_vs_ltm_test = ltm.add_document_to_vector_store(
        text_chunk="This is a test document for the LTM vector store using a mock embedder.",
        metadata={"source": "ltm_direct_test", "id": "vs_doc1"}
    )
    print(f"Added doc1 to VS: {doc1_id_vs_ltm_test}")
    if doc1_id_vs_ltm_test: # Only search if add was successful
        time.sleep(0.5) # Give Chroma a moment
        vs_results_ltm_test = ltm.semantic_search_vector_store(query_text="test document mock", top_k=1)
        print("Vector Store Search Results (mock embeddings):")
        for res in vs_results_ltm_test:
            print(f"  - ID: {res['id']}, Text: '{res['text_chunk'][:50]}...', Meta: {res['metadata']}")
    # ... (other vector store tests) ...

    # Test LTM Reset
    print("\n--- Testing LTM Reset (via LTMManager) ---")
    ltm.reset_ltm(confirm_reset=True) # This will clear SQLite and recreate Chroma collection
    print(f"LTM Reset complete. SKB facts after reset: {len(ltm.get_facts(subject='LTM Test Subject'))}")
    print(f"LTM Vector store count after reset: {ltm.vector_store.collection.count()}")


    print("\nLTMManager direct test finished.")
    # Cleanup for ltm.py's own test
    print("\nAttempting cleanup of ltm.py direct test files...")
    del ltm
    gc.collect() # Ensure gc is imported if used: import gc
    time.sleep(0.1)
    if os.path.exists(test_ltm_sqlite_db): os.remove(test_ltm_sqlite_db)
    if os.path.exists(test_ltm_chroma_dir): shutil.rmtree(test_ltm_chroma_dir, ignore_errors=True) # ignore_errors for robustness
    print("ltm.py direct test file cleanup attempt finished.")