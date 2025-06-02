import pytest
import os
import sys
from datetime import datetime, timezone

# Adjust sys.path to allow importing from the 'mmu' package
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mmu.mmu_manager import ShortTermMemory # Import the class we want to test

# --- Test Functions will go here ---
def test_stm_initialization():
    """Test ShortTermMemory initialization."""
    stm = ShortTermMemory(max_turns=5)
    assert stm.max_turns == 5
    assert len(stm.history) == 0
    assert stm.scratchpad_content == ""

def test_stm_add_single_turn():
    """Test adding a single turn to STM."""
    stm = ShortTermMemory(max_turns=3)
    stm.add_turn(role="user", content="Hello")
    
    history = stm.get_history()
    assert len(history) == 1
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Hello"
    assert "timestamp" in history[0]

def test_stm_add_multiple_turns_within_limit():
    """Test adding multiple turns that do not exceed max_turns."""
    stm = ShortTermMemory(max_turns=3)
    stm.add_turn(role="user", content="Turn 1")
    stm.add_turn(role="assistant", content="Turn 2")
    
    history = stm.get_history()
    assert len(history) == 2
    assert history[0]["content"] == "Turn 1"
    assert history[1]["content"] == "Turn 2"

def test_stm_exceeding_max_turns():
    """Test that STM correctly drops the oldest turn when max_turns is exceeded."""
    stm = ShortTermMemory(max_turns=2)
    stm.add_turn(role="user", content="Turn 1 (oldest)")
    stm.add_turn(role="assistant", content="Turn 2")
    stm.add_turn(role="user", content="Turn 3 (newest)") # This should push out Turn 1
    
    history = stm.get_history()
    assert len(history) == 2
    assert history[0]["content"] == "Turn 2"       # Turn 1 should be gone
    assert history[1]["content"] == "Turn 3 (newest)"
    
def test_stm_clear():
    """Test clearing the STM history and scratchpad."""
    stm = ShortTermMemory(max_turns=3)
    stm.add_turn(role="user", content="Some content")
    stm.update_scratchpad("Some scratchpad text")
    
    assert len(stm.get_history()) == 1
    assert stm.get_scratchpad() == "Some scratchpad text"
    
    stm.clear()
    
    assert len(stm.get_history()) == 0
    assert stm.get_scratchpad() == ""
    assert len(stm.history) == 0 # Also check the internal deque

def test_stm_get_formatted_history_empty():
    """Test get_formatted_history with no turns."""
    stm = ShortTermMemory(max_turns=3)
    assert stm.get_formatted_history() == ""
    assert stm.get_formatted_history(include_timestamps=True) == ""

def test_stm_get_formatted_history_with_turns():
    """Test get_formatted_history with some turns."""
    stm = ShortTermMemory(max_turns=3)
    stm.add_turn(role="user", content="Hello")
    # We need to mock datetime.now for predictable timestamps if testing include_timestamps=True
    # For now, let's test without timestamps for simplicity in this test.
    stm.add_turn(role="assistant", content="Hi there")
    
    expected_output = "User: Hello\nAssistant: Hi there"
    assert stm.get_formatted_history(include_timestamps=False) == expected_output

def test_stm_get_formatted_history_with_timestamps(monkeypatch):
    """Test get_formatted_history with timestamps using a mocked datetime."""
    stm = ShortTermMemory(max_turns=3)

    # Mock datetime.datetime.now to return a fixed timestamp
    mock_now_counter = 0
    fixed_timestamps_iso = [
        "2024-01-01T10:00:00+00:00",
        "2024-01-01T10:00:05+00:00"
    ]
    fixed_timestamps_formatted = [ # Expected format H:M:S
        "10:00:00",
        "10:00:05"
    ]

    # Our MockDateTime is intended to behave like the original datetime.datetime class
    class MockDateTimeClass: # Renamed for clarity
        @classmethod
        def now(cls, tz=None): # tz=None to match signature of datetime.datetime.now()
            nonlocal mock_now_counter
            # Original datetime.datetime.now() returns a datetime object.
            # Our fixed_timestamps_iso are strings, so we need to parse them into actual datetime objects here.
            # The original datetime.fromisoformat should be used inside the mock.
            dt_obj = datetime.fromisoformat(fixed_timestamps_iso[mock_now_counter])
            mock_now_counter += 1
            return dt_obj
        
        @staticmethod
        def fromisoformat(date_string): # This method is on the datetime.datetime class
            if date_string.endswith('Z'):
                date_string = date_string[:-1] + '+00:00'
            # Use the *original* datetime.fromisoformat for parsing within the mock
            return datetime.fromisoformat(date_string)

    # We need to patch 'datetime.datetime' as seen by mmu_manager.
    # The 'datetime' module is imported at the top of mmu_manager.py.
    # So we target 'mmu.mmu_manager.datetime.datetime'
    # This replaces the `datetime` class within the `datetime` module that `mmu_manager` uses.
    monkeypatch.setattr('mmu.mmu_manager.datetime.datetime', MockDateTimeClass)

    stm.add_turn(role="user", content="Hello")
    stm.add_turn(role="assistant", content="Hi there")
    
    expected_output = (
        f"User ({fixed_timestamps_formatted[0]}): Hello\n"
        f"Assistant ({fixed_timestamps_formatted[1]}): Hi there"
    )
    actual_formatted_history = stm.get_formatted_history(include_timestamps=True)
    print(f"Expected: '''{expected_output}'''") # Debug print
    print(f"Actual:   '''{actual_formatted_history}'''") # Debug print
    assert actual_formatted_history == expected_output


def test_stm_scratchpad_initial_state():
    """Test initial scratchpad state."""
    stm = ShortTermMemory(max_turns=3)
    assert stm.get_scratchpad() == ""

def test_stm_update_scratchpad():
    """Test updating the scratchpad."""
    stm = ShortTermMemory(max_turns=3)
    stm.update_scratchpad("Initial thought.")
    assert stm.get_scratchpad() == "Initial thought."
    
    stm.update_scratchpad("New thought, overwrites old.")
    assert stm.get_scratchpad() == "New thought, overwrites old."

def test_stm_append_to_scratchpad():
    """Test appending to the scratchpad."""
    stm = ShortTermMemory(max_turns=3)
    stm.append_to_scratchpad("Part 1.")
    assert stm.get_scratchpad() == "Part 1."
    
    stm.append_to_scratchpad("Part 2.", separator=" ")
    assert stm.get_scratchpad() == "Part 1. Part 2."
    
    stm.append_to_scratchpad("Part 3.") # Default separator is newline
    assert stm.get_scratchpad() == "Part 1. Part 2.\nPart 3."
    
def test_stm_append_to_empty_scratchpad():
    """Test appending to an initially empty scratchpad."""
    stm = ShortTermMemory(max_turns=3)
    stm.append_to_scratchpad("First line")
    assert stm.get_scratchpad() == "First line"