import pytest
import os
import sys
import sqlite3 # For checking IntegrityError directly if needed
import uuid
import json # For metadata testing
from datetime import datetime, timezone

# Adjust sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mmu.ltm import RawConversationLog

# Pytest fixture for a temporary SQLite DB path for RawConversationLog
@pytest.fixture
def temp_sqlite_db_path(tmp_path):
    """Provides a temporary path for an SQLite DB file and ensures cleanup."""
    db_file = tmp_path / "test_raw_log.db"
    yield str(db_file)
    if db_file.exists():
        db_file.unlink()

# Pytest fixture for an in-memory SQLite DB for RawConversationLog
# Most tests will use this for speed and isolation.
@pytest.fixture
def in_memory_raw_log():
    """Provides a RawConversationLog instance using an in-memory SQLite DB."""
    # Using ":memory:" creates an in-memory database
    log = RawConversationLog(db_path=":memory:")
    yield log
    log.close() # Call the new close method

# Pytest fixture for a file-based (temporary) RawConversationLog instance
# Useful for tests that might need to inspect the file or test persistence across connections (though less common for unit tests)
@pytest.fixture
def file_based_raw_log(temp_sqlite_db_path):
    """Provides a RawConversationLog instance using a temporary file-based SQLite DB."""
    log = RawConversationLog(db_path=temp_sqlite_db_path)
    return log
    # The temp_sqlite_db_path fixture will handle cleanup of the file

def test_raw_log_initialization_creates_table(in_memory_raw_log):
    """Test that initialization creates the conversation_log table."""
    log = in_memory_raw_log
    # Try to query the table structure or insert/select a dummy row
    # to ensure the table exists.
    try:
        with log._get_connection() as conn:
            cursor = conn.cursor()
            # This query will fail if the table or columns don't exist
            cursor.execute("SELECT entry_id, conversation_id FROM conversation_log LIMIT 1")
    except sqlite3.Error as e:
        pytest.fail(f"Table 'conversation_log' likely not created or schema is wrong: {e}")

def test_raw_log_log_single_interaction(in_memory_raw_log):
    """Test logging a single interaction."""
    log = in_memory_raw_log
    conv_id = "conv_test_001"
    entry_id = log.log_interaction(
        conversation_id=conv_id,
        turn_sequence_id=1,
        role="user",
        content="Hello Raw Log!",
        llm_model_used="test_model",
        tokens_prompt=10,
        tokens_completion=20,
        metadata={"tag": "test_event", "source_ip": "127.0.0.1"}
    )
    
    assert entry_id is not None
    assert isinstance(entry_id, str)
    
    history = log.get_conversation_history(conv_id)
    assert len(history) == 1
    turn = history[0]
    
    assert turn["entry_id"] == entry_id
    assert turn["conversation_id"] == conv_id
    assert turn["turn_sequence_id"] == 1
    assert turn["role"] == "user"
    assert turn["content"] == "Hello Raw Log!"
    assert "timestamp" in turn and isinstance(turn["timestamp"], str)
    assert turn["llm_model_used"] == "test_model"
    assert turn["tokens_prompt"] == 10
    assert turn["tokens_completion"] == 20
    assert isinstance(turn["metadata"], dict) # Check it was deserialized
    assert turn["metadata"]["tag"] == "test_event"
    assert turn["metadata"]["source_ip"] == "127.0.0.1"

def test_raw_log_log_interaction_minimal_args(in_memory_raw_log):
    """Test logging with only required arguments."""
    log = in_memory_raw_log
    conv_id = "conv_minimal_001"
    entry_id = log.log_interaction(
        conversation_id=conv_id,
        turn_sequence_id=1,
        role="system",
        content="System initialized."
    )
    assert entry_id is not None
    history = log.get_conversation_history(conv_id)
    assert len(history) == 1
    turn = history[0]
    assert turn["role"] == "system"
    assert turn["content"] == "System initialized."
    assert turn["llm_model_used"] is None
    assert turn["tokens_prompt"] is None
    assert turn["tokens_completion"] is None
    assert turn["metadata"] is None # Metadata is None if not provided