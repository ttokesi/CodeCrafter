import pytest
import os
import sys
import shutil # For cleaning up TinyDB files
from datetime import datetime, timezone, timedelta
import time 
import gc   
import tinydb # Make sure this is imported for isinstance checks

# Adjust sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mmu.mmu_manager import MediumTermMemory

# Pytest fixture for a temporary TinyDB path
@pytest.fixture
def temp_tinydb_path(tmp_path):
    """Provides a temporary path for a TinyDB file and ensures cleanup."""
    db_file = tmp_path / "test_mtm_db.json"
    yield str(db_file) 
    
    # Teardown:
    import time 
    import gc   
    time.sleep(0.1) 
    gc.collect()    

    if db_file.exists():
        try:
            db_file.unlink()
            print(f"DEBUG (temp_tinydb_path teardown): Successfully unlinked {db_file}")
        except PermissionError as e:
            print(f"DEBUG (temp_tinydb_path teardown): PermissionError unlinking {db_file}. Error: {e}")
        except Exception as e: # Catch any other potential errors during unlink
            print(f"DEBUG (temp_tinydb_path teardown): Unexpected error unlinking {db_file}. Error: {e}")

@pytest.fixture(params=["in_memory", "persistent"])
def mtm_instance(request, temp_tinydb_path):
    db_to_close = None
    if request.param == "in_memory":
        mtm = MediumTermMemory(use_tinydb=False, db_path="")
        yield mtm 
    elif request.param == "persistent":
        os.makedirs(os.path.dirname(temp_tinydb_path), exist_ok=True)
        mtm = MediumTermMemory(use_tinydb=True, db_path=temp_tinydb_path)
        
        assert mtm.db is not None, "TinyDB instance (mtm.db) should not be None for persistent MTM after init."

        # --- Path retrieval and assertion ---
        path_to_assert = None
        try:
            # Most specific check first for CachingMiddleware -> JSONStorage -> _handle.name
            if (hasattr(mtm.db, 'storage') and mtm.db.storage and # Ensure storage object exists
                hasattr(mtm.db.storage, 'storage') and mtm.db.storage.storage and # Ensure underlying storage object exists
                isinstance(mtm.db.storage.storage, tinydb.storages.JSONStorage) and
                hasattr(mtm.db.storage.storage, '_handle') and mtm.db.storage.storage._handle and
                hasattr(mtm.db.storage.storage._handle, 'name') and
                isinstance(mtm.db.storage.storage._handle.name, str)):
                path_to_assert = mtm.db.storage.storage._handle.name
                print(f"DEBUG: Path successfully retrieved via mtm.db.storage.storage._handle.name: {path_to_assert}")
            
            # Fallback 1: Directly on TinyDB object (less common with middleware for path)
            elif hasattr(mtm.db, 'filepath') and mtm.db.filepath and isinstance(mtm.db.filepath, str):
                path_to_assert = mtm.db.filepath
                print(f"DEBUG: Path successfully retrieved via mtm.db.filepath: {path_to_assert}")
            
            # Fallback 2: Another common internal name
            elif hasattr(mtm.db, '_path') and mtm.db._path and isinstance(mtm.db._path, str):
                path_to_assert = mtm.db._path
                print(f"DEBUG: Path successfully retrieved via mtm.db._path: {path_to_assert}")
            
            else:
                # If none of the above worked, print detailed attributes for debugging
                print(f"DEBUG: Could not find path directly. mtm.db type: {type(mtm.db)}")
                if hasattr(mtm.db, 'storage'):
                    print(f"DEBUG: mtm.db.storage type: {type(mtm.db.storage)}")
                    if hasattr(mtm.db.storage, 'storage'):
                        print(f"DEBUG: mtm.db.storage.storage type: {type(mtm.db.storage.storage)}")
                        print(f"DEBUG: Attributes of mtm.db.storage.storage: {dir(mtm.db.storage.storage)}")
                        if hasattr(mtm.db.storage.storage, '_handle'):
                             print(f"DEBUG: mtm.db.storage.storage._handle type: {type(mtm.db.storage.storage._handle)}")
                             print(f"DEBUG: Attributes of mtm.db.storage.storage._handle: {dir(mtm.db.storage.storage._handle)}")
                    else:
                        print(f"DEBUG: mtm.db.storage has no 'storage' attribute. Attributes: {dir(mtm.db.storage)}")
                else:
                    print(f"DEBUG: mtm.db has no 'storage' attribute. Attributes: {dir(mtm.db)}")
        
        except Exception as e_path:
            print(f"DEBUG: Exception during path retrieval: {e_path}")
            # This ensures path_to_assert remains None if an unexpected error occurs
            # during the attribute checks.

        assert path_to_assert is not None, "Could not resolve actual DB path from TinyDB object/storage. Path remains None."
        assert os.path.normpath(str(path_to_assert)) == os.path.normpath(temp_tinydb_path), \
            f"Failed to verify DB path. Expected '{temp_tinydb_path}', got '{path_to_assert}'"

        if mtm.db:
            db_to_close = mtm.db 
        yield mtm

        # Teardown for persistent mode
        if db_to_close:
            print(f"DEBUG (Teardown): Attempting to close TinyDB: {db_to_close}")
            if hasattr(db_to_close.storage, '_storage') and hasattr(db_to_close.storage._storage, 'flush') and callable(db_to_close.storage._storage.flush): # Try flushing underlying JSONStorage if CachingMiddleware
                try:
                    db_to_close.storage._storage.flush()
                    print("DEBUG (Teardown): Flushed underlying JSONStorage via CachingMiddleware's storage.")
                except Exception as e:
                    print(f"DEBUG (Teardown): Error flushing underlying JSONStorage: {e}")
            elif hasattr(db_to_close.storage, 'flush') and callable(db_to_close.storage.flush): # Try flushing CachingMiddleware itself
                try:
                    db_to_close.storage.flush()
                    print("DEBUG (Teardown): Flushed TinyDB CachingMiddleware")
                except Exception as e:
                    print(f"DEBUG (Teardown): Error flushing TinyDB CachingMiddleware: {e}")
            db_to_close.close()
            print(f"DEBUG (Teardown): Closed TinyDB for {temp_tinydb_path}")
            db_to_close = None 
        else:
            print(f"DEBUG (Teardown): No db_to_close object for {temp_tinydb_path}.")

def test_mtm_initialization(mtm_instance):
    """Test MTM initialization for both modes."""
    if not mtm_instance.is_persistent: 
        assert mtm_instance.db is None
        print("DEBUG (test_mtm_initialization): Testing in-memory MTM initialization: PASSED.")
    else: 
        assert mtm_instance.is_persistent 
        assert mtm_instance.db is not None
        
        path_to_check = None
        try:
            if (hasattr(mtm_instance.db.storage, 'storage') and 
                isinstance(mtm_instance.db.storage.storage, tinydb.storages.JSONStorage) and 
                hasattr(mtm_instance.db.storage.storage, '_handle') and
                hasattr(mtm_instance.db.storage.storage._handle, 'name') and
                isinstance(mtm_instance.db.storage.storage._handle.name, str)):
                path_to_check = mtm_instance.db.storage.storage._handle.name
            elif hasattr(mtm_instance.db, 'filepath') and isinstance(mtm_instance.db.filepath, str): # Added isinstance
                path_to_check = mtm_instance.db.filepath
            elif hasattr(mtm_instance.db, '_path') and isinstance(mtm_instance.db._path, str): # Added isinstance
                path_to_check = mtm_instance.db._path
        except Exception as e:
            print(f"DEBUG (test_mtm_initialization): Error during path check: {e}")


        print(f"DEBUG (test_mtm_initialization): Testing persistent MTM initialization. Path resolved for check: {path_to_check}")
        assert path_to_check is not None, "Could not retrieve DB path from TinyDB object or its storage for test_mtm_initialization."
        assert os.path.exists(str(path_to_check)), f"DB file {path_to_check} does not exist for test_mtm_initialization."
        print("DEBUG (test_mtm_initialization): Testing persistent MTM initialization: PASSED path checks.")


def test_mtm_store_and_get_summary(mtm_instance):
    """Test storing and retrieving a summary."""
    summary_id = "sum_001"
    summary_text = "This is a test summary."
    metadata = {"source": "test"} 
    
    mtm_instance.store_summary(summary_id, summary_text, metadata)
    retrieved = mtm_instance.get_summary(summary_id)
    
    assert retrieved is not None
    assert retrieved["text"] == summary_text
    assert retrieved["metadata"]["source"] == "test"
    assert "last_updated" in retrieved["metadata"]

    if mtm_instance.is_persistent: 
        assert retrieved["summary_id"] == summary_id

def test_mtm_get_non_existent_summary(mtm_instance):
    """Test retrieving a non-existent summary returns None."""
    assert mtm_instance.get_summary("non_existent_id") is None

def test_mtm_get_recent_summaries_sorted(mtm_instance, monkeypatch):
    """Test retrieving recent summaries, checking sorting by 'last_updated' (auto-added)."""
    mtm_instance.clear_session_data() 

    mock_now_time = datetime.now(timezone.utc)
    
    fixed_iso_timestamps = [
        (mock_now_time - timedelta(seconds=1)).isoformat(), 
        (mock_now_time - timedelta(seconds=5)).isoformat(), 
        (mock_now_time - timedelta(seconds=10)).isoformat() 
    ]
    
    call_count = 0
    def mock_datetime_now_for_store(tz=None):
        nonlocal call_count
        dt_to_return = datetime.fromisoformat(fixed_iso_timestamps[call_count])
        # Ensure we don't go out of bounds if more than 3 calls are made, though test expects 3
        current_call_idx = call_count 
        call_count = (call_count + 1) % len(fixed_iso_timestamps) 
        return dt_to_return

    # Patch datetime.datetime.now, assuming mmu_manager.py imports 'datetime'
    # and store_summary uses 'datetime.datetime.now(...)'
    monkeypatch.setattr('mmu.mmu_manager.datetime.datetime', type('MockDateTimeForMTM', (object,), {'now': mock_datetime_now_for_store}))

    mtm_instance.store_summary("s1", "Summary Oldest", {"id_check": "s1"}) 
    mtm_instance.store_summary("s3", "Summary Middle", {"id_check": "s3"}) 
    mtm_instance.store_summary("s2", "Summary Newest", {"id_check": "s2"}) 
    
    recent_summaries = mtm_instance.get_recent_summaries(max_count=3)
    assert len(recent_summaries) == 3
    
    # Based on mocked timestamps: s1 got fixed_iso_timestamps[0] (newest by mock logic), 
    # s3 got fixed_iso_timestamps[1] (middle), s2 got fixed_iso_timestamps[2] (oldest)
    # So, order should be s1, s3, s2.
    if mtm_instance.is_persistent:
        assert recent_summaries[0]["summary_id"] == "s1" 
        assert recent_summaries[1]["summary_id"] == "s3"
        assert recent_summaries[2]["summary_id"] == "s2"
    else: 
        assert recent_summaries[0]["metadata"]["id_check"] == "s1"
        assert recent_summaries[1]["metadata"]["id_check"] == "s3"
        assert recent_summaries[2]["metadata"]["id_check"] == "s2"

    recent_one = mtm_instance.get_recent_summaries(max_count=1)
    assert len(recent_one) == 1
    if mtm_instance.is_persistent:
        assert recent_one[0]["summary_id"] == "s1"
    else:
        assert recent_one[0]["metadata"]["id_check"] == "s1"