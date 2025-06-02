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

def test_count_tokens_hf_success(monkeypatch):
    """Test count_tokens_hf successfully counts tokens using a mock tokenizer."""
    
    # Mock get_hf_tokenizer to return a controllable MockHFTokenizer instance
    mock_tokenizer_instance = MockHFTokenizer("test-hf-model")
    # We want to control the output of encode().ids for this test
    mock_tokenizer_instance.mock_encoded_ids = [1, 2, 3, 4, 5] # Simulate 5 tokens
    
    monkeypatch.setattr(tokenizer_utils, 'get_hf_tokenizer', lambda name=None: mock_tokenizer_instance)
    
    count = tokenizer_utils.count_tokens_hf("Some example text", hf_tokenizer_name="test-hf-model")
    assert count == 5
    assert mock_tokenizer_instance.encode_call_count == 1 # Ensure encode was called

def test_count_tokens_hf_empty_text(monkeypatch):
    """Test count_tokens_hf with empty text."""
    # No need to mock get_hf_tokenizer for this, as it should return 0 before even trying to get a tokenizer
    count = tokenizer_utils.count_tokens_hf("", hf_tokenizer_name="any-model")
    assert count == 0

def test_count_tokens_hf_get_tokenizer_fails(monkeypatch):
    """Test count_tokens_hf when get_hf_tokenizer returns None."""
    monkeypatch.setattr(tokenizer_utils, 'get_hf_tokenizer', lambda name=None: None)
    
    count = tokenizer_utils.count_tokens_hf("Some text", hf_tokenizer_name="fail-model")
    assert count is None

def test_count_tokens_hf_encode_fails(monkeypatch):
    """Test count_tokens_hf when tokenizer.encode() raises an error."""
    
    mock_tokenizer_instance = MockHFTokenizer("test-hf-model")
    # Make the encode method raise an error
    def mock_encode_raises_error(text):
        raise ValueError("Simulated encoding error")
    mock_tokenizer_instance.encode = mock_encode_raises_error
        
    monkeypatch.setattr(tokenizer_utils, 'get_hf_tokenizer', lambda name=None: mock_tokenizer_instance)
    
    count = tokenizer_utils.count_tokens_hf("Some text", hf_tokenizer_name="test-hf-model")
    assert count is None # Expect None if encoding fails

def test_count_tokens_main_function_direct_map(monkeypatch):
    """Test main count_tokens uses direct map from config."""
    mock_config_data = {
        "tokenizer": {
            "hf_tokenizer_map": {
                "ollama-model-exact:7b": "hf-exact-tokenizer-for-ollama-model" 
            },
            "default_hf_tokenizer": "hf-overall-default"
        }
    }
    monkeypatch.setattr(tokenizer_utils, 'get_config', lambda: mock_config_data)

    # Mock count_tokens_hf to see what hf_tokenizer_name it was called with
    called_with_hf_tokenizer_name = None
    def mock_ct_hf(text, hf_tokenizer_name=None):
        nonlocal called_with_hf_tokenizer_name
        called_with_hf_tokenizer_name = hf_tokenizer_name
        return len(text.split()) # Simulate token count
    monkeypatch.setattr(tokenizer_utils, 'count_tokens_hf', mock_ct_hf)
    
    tokenizer_utils.count_tokens("Test text", ollama_model_name="ollama-model-exact:7b")
    assert called_with_hf_tokenizer_name == "hf-exact-tokenizer-for-ollama-model"

def test_count_tokens_main_function_family_fallback(monkeypatch):
    """Test main count_tokens uses family fallback from config."""
    mock_config_data = {
        "tokenizer": {
            "hf_tokenizer_map": {
                "default_gemma": "hf-gemma-family-default"
            },
            "default_hf_tokenizer": "hf-overall-default"
        }
    }
    monkeypatch.setattr(tokenizer_utils, 'get_config', lambda: mock_config_data)
    
    called_with_hf_tokenizer_name = None
    def mock_ct_hf(text, hf_tokenizer_name=None):
        nonlocal called_with_hf_tokenizer_name
        called_with_hf_tokenizer_name = hf_tokenizer_name
        return 1 # Dummy count
    monkeypatch.setattr(tokenizer_utils, 'count_tokens_hf', mock_ct_hf)

    tokenizer_utils.count_tokens("Test text", ollama_model_name="gemma:some-variant")
    assert called_with_hf_tokenizer_name == "hf-gemma-family-default"

def test_count_tokens_main_function_overall_default(monkeypatch):
    """Test main count_tokens uses overall default hf_tokenizer from config."""
    mock_config_data = {
        "tokenizer": {
            "hf_tokenizer_map": {}, # Empty map
            "default_hf_tokenizer": "hf-overall-default-for-this-test"
        }
    }
    monkeypatch.setattr(tokenizer_utils, 'get_config', lambda: mock_config_data)

    called_with_hf_tokenizer_name = None
    def mock_ct_hf(text, hf_tokenizer_name=None):
        nonlocal called_with_hf_tokenizer_name
        called_with_hf_tokenizer_name = hf_tokenizer_name
        return 1 # Dummy count
    monkeypatch.setattr(tokenizer_utils, 'count_tokens_hf', mock_ct_hf)

    tokenizer_utils.count_tokens("Test text", ollama_model_name="unknown-model:latest")
    assert called_with_hf_tokenizer_name == "hf-overall-default-for-this-test"
    
def test_count_tokens_main_function_override(monkeypatch):
    """Test main count_tokens uses hf_tokenizer_name_override."""
    mock_config_data = {"tokenizer": {"hf_tokenizer_map": {}, "default_hf_tokenizer": "any"}}
    monkeypatch.setattr(tokenizer_utils, 'get_config', lambda: mock_config_data)
    
    called_with_hf_tokenizer_name = None
    def mock_ct_hf(text, hf_tokenizer_name=None):
        nonlocal called_with_hf_tokenizer_name
        called_with_hf_tokenizer_name = hf_tokenizer_name
        return 1
    monkeypatch.setattr(tokenizer_utils, 'count_tokens_hf', mock_ct_hf)

    tokenizer_utils.count_tokens(
        "Test text", 
        ollama_model_name="any-ollama-model",
        hf_tokenizer_name_override="explicit-override-tokenizer"
    )
    assert called_with_hf_tokenizer_name == "explicit-override-tokenizer"

def test_count_tokens_main_function_hf_fails_fallback_to_word_count(monkeypatch):
    """Test main count_tokens falls back to word count if count_tokens_hf fails."""
    mock_config_data = {"tokenizer": {"default_hf_tokenizer": "any"}} # Minimal config
    monkeypatch.setattr(tokenizer_utils, 'get_config', lambda: mock_config_data)
    
    # Mock count_tokens_hf to return None (simulating failure)
    monkeypatch.setattr(tokenizer_utils, 'count_tokens_hf', lambda text, hf_tokenizer_name=None: None)
    
    text_to_test = "This is a test sentence." # 6 words
    count = tokenizer_utils.count_tokens(text_to_test, ollama_model_name="any-model")
    assert count == 5

def test_count_tokens_main_function_empty_text():
    """Test main count_tokens with empty text returns 0."""
    # No mocking needed, should short-circuit
    count = tokenizer_utils.count_tokens("", ollama_model_name="any-model")
    assert count == 0