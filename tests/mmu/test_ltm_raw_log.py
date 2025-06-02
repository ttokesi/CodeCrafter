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

def test_raw_log_multiple_turns_and_conversations(in_memory_raw_log):
    """Test logging multiple turns for multiple conversations and history retrieval order."""
    log = in_memory_raw_log
    conv1_id = "conv1"
    conv2_id = "conv2"

    log.log_interaction(conv1_id, 1, "user", "C1: User 1")
    log.log_interaction(conv1_id, 2, "assistant", "C1: Assistant 1")
    log.log_interaction(conv2_id, 1, "user", "C2: User 1")
    log.log_interaction(conv1_id, 3, "user", "C1: User 2")

    # Test history for conv1
    history1 = log.get_conversation_history(conv1_id)
    assert len(history1) == 3
    assert history1[0]["content"] == "C1: User 1"
    assert history1[0]["turn_sequence_id"] == 1
    assert history1[1]["content"] == "C1: Assistant 1"
    assert history1[1]["turn_sequence_id"] == 2
    assert history1[2]["content"] == "C1: User 2"
    assert history1[2]["turn_sequence_id"] == 3
    
    # Test history for conv2
    history2 = log.get_conversation_history(conv2_id)
    assert len(history2) == 1
    assert history2[0]["content"] == "C2: User 1"
    
def test_raw_log_get_history_limit_offset(in_memory_raw_log):
    """Test get_conversation_history with limit and offset."""
    log = in_memory_raw_log
    conv_id = "conv_limit_offset"
    for i in range(1, 6): # Log 5 turns
        log.log_interaction(conv_id, i, "user", f"Turn {i}")

    # Test limit
    history_limit_2 = log.get_conversation_history(conv_id, limit=2)
    assert len(history_limit_2) == 2
    assert history_limit_2[0]["content"] == "Turn 1"
    assert history_limit_2[1]["content"] == "Turn 2"
    
    # Test offset
    history_offset_2 = log.get_conversation_history(conv_id, offset=2)
    assert len(history_offset_2) == 3 # Should get turns 3, 4, 5
    assert history_offset_2[0]["content"] == "Turn 3"
    
    # Test limit and offset
    history_limit_2_offset_1 = log.get_conversation_history(conv_id, limit=2, offset=1)
    assert len(history_limit_2_offset_1) == 2
    assert history_limit_2_offset_1[0]["content"] == "Turn 2"
    assert history_limit_2_offset_1[1]["content"] == "Turn 3"
    
    # Test limit larger than available items with offset
    history_limit_5_offset_3 = log.get_conversation_history(conv_id, limit=5, offset=3)
    assert len(history_limit_5_offset_3) == 2 # Should get turns 4, 5
    assert history_limit_5_offset_3[0]["content"] == "Turn 4"

def test_raw_log_get_history_non_existent_conv(in_memory_raw_log):
    """Test get_conversation_history for a non-existent conversation_id."""
    log = in_memory_raw_log
    history = log.get_conversation_history("non_existent_conv_id")
    assert len(history) == 0
    assert isinstance(history, list)

def test_raw_log_get_all_conversation_ids(in_memory_raw_log):
    """Test retrieving all unique conversation IDs."""
    log = in_memory_raw_log
    assert log.get_all_conversation_ids() == [] # Initially empty

    log.log_interaction("conv_a", 1, "user", "Test")
    log.log_interaction("conv_b", 1, "user", "Test")
    log.log_interaction("conv_a", 2, "user", "Test again")
    
    all_ids = log.get_all_conversation_ids()
    assert len(all_ids) == 2
    assert "conv_a" in all_ids
    assert "conv_b" in all_ids
    # The order is defined by "ORDER BY conversation_id ASC" in the SQL query
    assert all_ids == ["conv_a", "conv_b"]


def test_raw_log_unique_constraint_violation(in_memory_raw_log):
    """Test UNIQUE constraint (conversation_id, turn_sequence_id) behavior."""
    log = in_memory_raw_log
    conv_id = "conv_unique"
    
    entry_id1 = log.log_interaction(conv_id, 1, "user", "First turn")
    assert entry_id1 is not None
    
    # Try to log another turn with the same conversation_id and turn_sequence_id
    entry_id2 = log.log_interaction(conv_id, 1, "user", "Attempted duplicate turn")
    
    # The log_interaction method catches sqlite3.IntegrityError and returns None
    assert entry_id2 is None
    
    # Verify that only the first turn was actually logged
    history = log.get_conversation_history(conv_id)
    assert len(history) == 1
    assert history[0]["content"] == "First turn"