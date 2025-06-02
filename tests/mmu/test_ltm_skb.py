import pytest
import os
import sys
import sqlite3
import uuid
import json
from datetime import datetime, timezone

# Adjust sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mmu.ltm import StructuredKnowledgeBase

# Pytest fixture for an in-memory SQLite DB for StructuredKnowledgeBase
@pytest.fixture
def in_memory_skb():
    """Provides a StructuredKnowledgeBase instance using an in-memory SQLite DB."""
    skb = StructuredKnowledgeBase(db_path=":memory:")
    yield skb
    skb.close() # Ensure the persistent :memory: connection is closed after the test

def test_skb_initialization_creates_tables(in_memory_skb):
    """Test that SKB initialization creates all necessary tables."""
    skb = in_memory_skb
    table_names_to_check = ["facts", "user_preferences", "entities"]
    try:
        with skb._get_connection() as conn: # Use internal _get_connection for this check
            cursor = conn.cursor()
            for table_name in table_names_to_check:
                cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
                assert cursor.fetchone() is not None, f"Table '{table_name}' was not created."
    except sqlite3.Error as e:
        pytest.fail(f"SQLite error during table check: {e}")

def test_skb_store_and_get_single_fact(in_memory_skb):
    """Test storing and retrieving a single fact."""
    skb = in_memory_skb
    subject = "user_color_pref"
    predicate = "is"
    obj_val = "blue"
    source_ids = [str(uuid.uuid4())]
    confidence = 0.95
    
    fact_id = skb.store_fact(subject, predicate, obj_val, source_ids, confidence)
    assert fact_id is not None
    
    facts = skb.get_facts(subject=subject, predicate=predicate, object_value=obj_val)
    assert len(facts) == 1
    fact = facts[0]
    
    assert fact["fact_id"] == fact_id
    assert fact["subject"] == subject
    assert fact["predicate"] == predicate
    assert fact["object"] == obj_val # Note: SKB stores it as 'object' in DB, retrieves as 'object'
    assert fact["source_turn_ids"] == source_ids # Deserialized from JSON
    assert fact["confidence"] == confidence
    assert "created_at" in fact
    assert fact["last_accessed"] is None # Not accessed via get_facts directly updating this yet

def test_skb_store_fact_updates_existing_on_conflict(in_memory_skb):
    """Test that storing a fact with the same S-P-O updates it (e.g., confidence, source_ids)."""
    skb = in_memory_skb
    subject = "project_status"
    predicate = "is"
    obj_val = "active"
    
    fact_id1 = skb.store_fact(subject, predicate, obj_val, source_turn_ids=["id1"], confidence=0.8)
    
    # Store again with different source_ids and confidence
    fact_id2 = skb.store_fact(subject, predicate, obj_val, source_turn_ids=["id2"], confidence=0.9)
    
    # The ON CONFLICT clause should update. The returned fact_id might be the original or new 
    # depending on SQLite version and exact ON CONFLICT behavior.
    # What's important is that the data is updated.
    # Let's assume for now fact_id1 is the one that persists if subject,predicate,object are the same.
    # We need to fetch the fact_id after an ON CONFLICT to be sure.
    # For simplicity, let's retrieve by S-P-O and check updated fields.
    
    facts = skb.get_facts(subject=subject, predicate=predicate, object_value=obj_val)
    assert len(facts) == 1
    updated_fact = facts[0]
    
    # Check that original fact_id is likely retained due to ON CONFLICT not re-inserting primary key
    # but this depends on implementation details. A safer check is on content.
    # assert updated_fact["fact_id"] == fact_id1 
    
    assert updated_fact["source_turn_ids"] == ["id2"] # Updated
    assert updated_fact["confidence"] == 0.9        # Updated
    assert updated_fact["last_accessed"] is not None # Should be updated by ON CONFLICT clause

def test_skb_get_facts_with_criteria(in_memory_skb):
    """Test get_facts with various querying criteria (LIKE behavior)."""
    skb = in_memory_skb
    skb.store_fact("user_name", "is", "Tibi")
    skb.store_fact("user_location", "is", "Switzerland")
    skb.store_fact("user_hobby", "is", "Python programming")
    skb.store_fact("project_alpha", "status", "active")

    # Get all facts
    all_facts = skb.get_facts()
    assert len(all_facts) == 4

    # Get by subject (exact part, using LIKE)
    user_facts = skb.get_facts(subject="user")
    assert len(user_facts) == 3 
    
    # Get by predicate
    is_facts = skb.get_facts(predicate="is")
    assert len(is_facts) == 3
    
    # Get by object_value (exact part, using LIKE)
    python_facts = skb.get_facts(object_value="Python")
    assert len(python_facts) == 1
    assert python_facts[0]["subject"] == "user_hobby"

    # Get by specific S-P-O
    name_fact = skb.get_facts(subject="user_name", predicate="is", object_value="Tibi")
    assert len(name_fact) == 1
    
    # Get by partial S-P-O (subject and predicate)
    user_is_facts = skb.get_facts(subject="user", predicate="is")
    assert len(user_is_facts) == 3

def test_skb_get_facts_no_match(in_memory_skb):
    """Test get_facts returns an empty list if no facts match criteria."""
    skb = in_memory_skb
    skb.store_fact("test_subject", "test_predicate", "test_object")
    
    no_match_facts = skb.get_facts(subject="non_existent")
    assert len(no_match_facts) == 0
    assert isinstance(no_match_facts, list)

def test_skb_store_and_get_preference(in_memory_skb):
    """Test storing and retrieving a user preference."""
    skb = in_memory_skb
    category = "ui_settings"
    key = "theme"
    value = "dark"
    
    pref_id = skb.store_preference(category=category, key=key, value=value)
    assert pref_id is not None
    
    retrieved_pref = skb.get_preference(category=category, key=key)
    assert retrieved_pref is not None
    assert retrieved_pref["preference_id"] == pref_id
    assert retrieved_pref["user_id"] == "default_user" # Default user_id
    assert retrieved_pref["category"] == category
    assert retrieved_pref["key"] == key
    assert retrieved_pref["value"] == value
    assert "created_at" in retrieved_pref

def test_skb_store_preference_updates_existing(in_memory_skb):
    """Test storing a preference with an existing user_id, category, key updates it."""
    skb = in_memory_skb
    user_id = "user123"
    category = "notifications"
    key = "email_enabled"
    
    pref_id1 = skb.store_preference(category=category, key=key, value="true", user_id=user_id)
    
    # Store again with a different value
    pref_id2 = skb.store_preference(category=category, key=key, value="false", user_id=user_id)
    
    # The ON CONFLICT clause should update. 
    # Similar to facts, the exact returned ID might vary, focus on data.
    retrieved_pref = skb.get_preference(category=category, key=key, user_id=user_id)
    assert retrieved_pref is not None
    assert retrieved_pref["value"] == "false" # Value should be updated
    # The created_at timestamp should also be updated by the ON CONFLICT clause
    # To test this reliably, we'd need to mock datetime like in MTM tests or check it's different.
    # For now, verifying the value update is key.

def test_skb_get_non_existent_preference(in_memory_skb):
    """Test retrieving a non-existent preference returns None."""
    skb = in_memory_skb
    assert skb.get_preference("ui_settings", "non_existent_key") is None
    assert skb.get_preference("non_existent_category", "any_key") is None
    assert skb.get_preference("ui_settings", "theme", user_id="other_user") is None

def test_skb_store_preference_different_users(in_memory_skb):
    """Test storing preferences for different users with the same category/key."""
    skb = in_memory_skb
    category = "ui_settings"
    key = "font_size"
    
    skb.store_preference(category, key, "12px", user_id="user_A")
    skb.store_preference(category, key, "14px", user_id="user_B")
    
    pref_A = skb.get_preference(category, key, user_id="user_A")
    pref_B = skb.get_preference(category, key, user_id="user_B")
    
    assert pref_A is not None
    assert pref_A["value"] == "12px"
    
    assert pref_B is not None
    assert pref_B["value"] == "14px"