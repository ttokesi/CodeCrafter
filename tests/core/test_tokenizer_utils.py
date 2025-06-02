import pytest
import os
import sys

# Adjust sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core import tokenizer_utils
from core import config_loader # We'll need to mock get_config

# We need to mock the Hugging Face Tokenizer class itself for many tests
# to avoid actual downloads and external dependencies during unit testing.
# We can define a simple mock class here.
class MockHFTokenizer:
    def __init__(self, model_name_or_path="mock_tokenizer"):
        self.model_name_or_path = model_name_or_path
        self.encode_call_count = 0
        self.mock_encoded_ids = []

    def encode(self, text):
        self.encode_call_count += 1
        # Simulate encoding: for testing, let's say length of words
        # or a fixed list if set by the test
        if self.mock_encoded_ids:
            return MockEncodedOutput(self.mock_encoded_ids)
        return MockEncodedOutput(text.split()) # Returns a list of "tokens" (words)

    @staticmethod
    def from_pretrained(name_or_path):
        # This method will be mocked by specific tests using monkeypatch
        # to control its behavior (e.g., return an instance of MockHFTokenizer or raise an error)
        print(f"MockHFTokenizer.from_pretrained called with: {name_or_path}") # For debug
        return MockHFTokenizer(name_or_path)

class MockEncodedOutput:
    def __init__(self, ids_list):
        self.ids = ids_list
        
# Helper to reset tokenizer_utils internal caches for test isolation
def reset_tokenizer_caches():
    tokenizer_utils._tokenizer_cache_hf.clear()

@pytest.fixture(autouse=True) # This fixture will run for every test in this file
def clear_caches_before_each_test():
    reset_tokenizer_caches()
    config_loader._config = None # Also reset config_loader cache

def test_get_hf_tokenizer_loads_and_caches(monkeypatch):
    """Test get_hf_tokenizer loads a tokenizer and caches it."""
    
    # Mock config_loader.get_config to return a controlled config
    mock_config_data = {
        "tokenizer": {
            "default_hf_tokenizer": "mock-default-tokenizer"
        }
    }
    monkeypatch.setattr(tokenizer_utils, 'get_config', lambda: mock_config_data)

    # Mock Tokenizer.from_pretrained to return our MockHFTokenizer instance
    # We want to patch it where it's *imported* in tokenizer_utils.py
    # tokenizer_utils.py does: from tokenizers import Tokenizer
    # So we patch 'tokenizer_utils.Tokenizer.from_pretrained'
    
    # Keep track of how many times from_pretrained is called
    from_pretrained_call_args = []
    def mock_from_pretrained(name_or_path):
        from_pretrained_call_args.append(name_or_path)
        return MockHFTokenizer(name_or_path)
    
    monkeypatch.setattr(tokenizer_utils.Tokenizer, 'from_pretrained', mock_from_pretrained)

    # First call: should load "test-tokenizer-name"
    tokenizer1 = tokenizer_utils.get_hf_tokenizer("test-tokenizer-name")
    assert isinstance(tokenizer1, MockHFTokenizer)
    assert tokenizer1.model_name_or_path == "test-tokenizer-name"
    assert len(from_pretrained_call_args) == 1
    assert from_pretrained_call_args[0] == "test-tokenizer-name"
    
    # Second call for same name: should use cache, from_pretrained not called again
    tokenizer2 = tokenizer_utils.get_hf_tokenizer("test-tokenizer-name")
    assert tokenizer2 is tokenizer1 # Should be the same cached object
    assert len(from_pretrained_call_args) == 1 # Still 1, not called again

    # Call with no name: should use default_hf_tokenizer from mock_config
    from_pretrained_call_args.clear()
    tokenizer3 = tokenizer_utils.get_hf_tokenizer() # No name, should use default
    assert isinstance(tokenizer3, MockHFTokenizer)
    assert tokenizer3.model_name_or_path == "mock-default-tokenizer"
    assert len(from_pretrained_call_args) == 1
    assert from_pretrained_call_args[0] == "mock-default-tokenizer"

def test_get_hf_tokenizer_load_failure(monkeypatch):
    """Test get_hf_tokenizer when Tokenizer.from_pretrained fails."""
    mock_config_data = {"tokenizer": {"default_hf_tokenizer": "fail-tokenizer"}}
    monkeypatch.setattr(tokenizer_utils, 'get_config', lambda: mock_config_data)

    def mock_from_pretrained_raises_error(name_or_path):
        raise ValueError("Simulated load failure")
    
    monkeypatch.setattr(tokenizer_utils.Tokenizer, 'from_pretrained', mock_from_pretrained_raises_error)

    tokenizer = tokenizer_utils.get_hf_tokenizer("some-tokenizer")
    assert tokenizer is None

def test_get_hf_tokenizer_lib_not_available(monkeypatch):
    """Test get_hf_tokenizer when HF_TOKENIZERS_LIB_AVAILABLE is False."""
    mock_config_data = {"tokenizer": {"default_hf_tokenizer": "any-tokenizer"}}
    monkeypatch.setattr(tokenizer_utils, 'get_config', lambda: mock_config_data)
    
    # Temporarily set HF_TOKENIZERS_LIB_AVAILABLE to False within tokenizer_utils
    monkeypatch.setattr(tokenizer_utils, 'HF_TOKENIZERS_LIB_AVAILABLE', False)
    
    tokenizer = tokenizer_utils.get_hf_tokenizer("some-tokenizer")
    assert tokenizer is None