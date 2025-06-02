import pytest
import yaml # For creating a malformed YAML
import os
import sys # <--- ADDED

# --- ADD THIS BLOCK ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --- END BLOCK ---

# To test the module, we need to import it.
# Since 'tests' is at the same level as 'core', and we're in 'tests/core',
# we need to adjust the Python path for the import to work correctly when running pytest
# from the project root, or use relative imports if pytest is run as a module.
# A common way is to ensure the project root is discoverable.
# Pytest often handles this well if run from the project root, but let's be explicit
# or rely on Python's import mechanisms with __init__.py files.

# Assuming pytest is run from the project root:
from core import config_loader 

# In tests/core/test_config_loader.py

# Helper function to reset the config_loader's internal cache for isolated tests
def reset_config_loader_cache():
    config_loader._config = None
    config_loader._project_root = None # Reset project root cache too

def test_get_project_root_returns_correct_path():
    """Test that get_project_root returns the expected project root directory."""
    reset_config_loader_cache()
    # Expected root is one level up from 'core' where config_loader.py is,
    # or two levels up from 'tests/core' where this test file is.
    expected_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    actual_root = config_loader.get_project_root()
    assert actual_root == expected_root
    assert os.path.basename(actual_root) == "CodeCrafter" # Assuming your project folder is CodeCrafter

def test_load_default_config_successful(tmp_path):
    """Test successfully loading the default config.yaml from the project root."""
    reset_config_loader_cache()
    
    project_root = config_loader.get_project_root()
    original_config_path = os.path.join(project_root, 'config.yaml')

    # Ensure the actual config.yaml exists for this test
    assert os.path.exists(original_config_path), "Default config.yaml must exist for this test"

    # We'll call get_config which should load the default one.
    # We need to handle the data_dir creation carefully if it's defined relative.
    # For this test, we just want to ensure it loads.
    # Create a dummy data_dir in tmp_path if config specifies one,
    # to avoid modifying the actual project structure during test.
    
    # To make this test more isolated and not depend on the actual config.yaml's data_dir value,
    # it's better to mock or control config_loader.get_project_root() or the path to config.
    # However, for a first pass, let's test against the actual default config.
    
    try:
        cfg = config_loader.get_config() # This will try to load PROJECT_ROOT/config.yaml
        assert cfg is not None
        assert isinstance(cfg, dict)
        # Check for a known key from your actual config.yaml
        assert 'lsw' in cfg 
        assert 'mmu' in cfg
        assert 'orchestrator' in cfg
    finally:
        reset_config_loader_cache() # Clean up for other tests

def test_get_config_uses_cache(monkeypatch):
    """Test that get_config uses the cached configuration on subsequent calls."""
    reset_config_loader_cache()
    
    # Mock the load_config function to see if it's called multiple times
    # For simplicity here, we'll check the _config global variable directly.
    
    # First call should load
    cfg1 = config_loader.get_config() 
    assert config_loader._config is not None # Internal check that it's loaded
    
    # Second call should return the same cached object
    cfg2 = config_loader.get_config()
    assert cfg1 is cfg2 # Check for object identity (same object in memory)
    
    reset_config_loader_cache()


def test_data_dir_creation(tmp_path):
    """Test that the data directory is created if specified and doesn't exist."""
    reset_config_loader_cache()
    
    # Create a temporary project root structure for this test
    test_project_root = tmp_path / "test_proj"
    test_project_root.mkdir()
    
    # Path for a dummy config file within this temp project root
    dummy_config_content = {
        "data_dir": "my_test_data", # Relative path for data directory
        "some_other_key": "value"
    }
    dummy_config_file = test_project_root / "config.yaml"
    with open(dummy_config_file, 'w') as f:
        yaml.dump(dummy_config_content, f)
        
    expected_data_dir = test_project_root / "my_test_data"
    assert not expected_data_dir.exists() # Ensure it doesn't exist yet

    # Monkeypatch get_project_root to return our temporary project root
    # and clear config cache so load_config uses our dummy config
    original_get_project_root = config_loader.get_project_root
    config_loader.get_project_root = lambda: str(test_project_root) # Mock it
    config_loader._config = None # Clear cache

    try:
        cfg = config_loader.load_config() # This should use the dummy config and create data_dir
        assert cfg is not None
        assert expected_data_dir.exists()
        assert expected_data_dir.is_dir()
    finally:
        # Restore original get_project_root and clear config
        config_loader.get_project_root = original_get_project_root
        reset_config_loader_cache()


def test_load_config_file_not_found(tmp_path):
    """Test FileNotFoundError is raised if config file doesn't exist."""
    reset_config_loader_cache()
    non_existent_config_path = tmp_path / "non_existent_config.yaml"
    
    with pytest.raises(FileNotFoundError):
        config_loader.load_config(config_path=str(non_existent_config_path))
    
    reset_config_loader_cache()

def test_load_config_malformed_yaml(tmp_path):
    """Test yaml.YAMLError is raised if config file is malformed."""
    reset_config_loader_cache()
    
    malformed_config_file = tmp_path / "malformed_config.yaml"
    with open(malformed_config_file, 'w') as f: # Corrected to one underscore
        f.write("lsw: {ollama_host: 'localhost:11434'\n  bad_indent: true") # Malformed YAML

    with pytest.raises(yaml.YAMLError):
        config_loader.load_config(config_path=str(malformed_config_file))
        
    reset_config_loader_cache()