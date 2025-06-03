import pytest
import os
import sys

# --- Path Adjustment ---
# This block ensures that the 'core' and 'mmu' (and other project) modules
# can be imported correctly when running pytest from the project root.
# It adds the project's root directory ('CodeCrafter') to Python's import search path.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --- End Path Adjustment ---

# Import the module we want to test
from core.llm_service_wrapper import LLMServiceWrapper

# We will also need to import the ollama library to mock its Client later
import ollama

# And potentially SentenceTransformer if we are testing that functionality specifically
# We can conditionally import it or mock it entirely. Let's prepare for mocking it.
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE_FOR_TEST = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE_FOR_TEST = False
    # Define a dummy class if not available, so type hints don't break
    # and tests can proceed by mocking this dummy.
    class SentenceTransformer:
        def __init__(self, model_name_or_path):
            self.model_name_or_path = model_name_or_path
        def encode(self, sentences, **kwargs):
            # This mock should probably raise an error or return something
            # specific if called unexpectedly during a test.
            # For now, let's make it return a fixed-shape dummy embedding.
            print(f"DUMMY SentenceTransformer.encode called with {len(sentences)} sentences for model {self.model_name_or_path}")
            return [[0.1] * 10 for _ in sentences] # Dummy 10-dim embedding

# We'll need to mock config_loader.get_config to control the configuration
# LSW uses during tests.
from core import config_loader

# --- Helper Fixtures and Mock Classes (We will add these progressively) ---

# Example of a basic test function structure (we will replace this later)
def test_lsw_placeholder():
    """A placeholder test to ensure the file can be discovered by pytest."""
    assert True

@pytest.fixture(autouse=True) # autouse=True means this fixture runs for every test in this file
def isolated_config_for_lsw(monkeypatch):
    """
    Provides an isolated, mock configuration for LSW tests and resets
    the config_loader cache before each test.
    It also allows tests to further customize the config via monkeypatching
    the 'mock_config_data' within the test itself if needed.
    """
    # Reset config_loader's global cache before each test using this fixture
    config_loader._config = None
    config_loader._project_root = None # In case any test inadvertently sets it

    # Define a default mock configuration for LSW
    default_mock_lsw_config = {
        "lsw": {
            "ollama_host": "http://testhost:12345",
            "default_chat_model": "test-default-chat-model:latest",
            "default_embedding_model_ollama": "test-default-ollama-embed:latest",
            "default_embedding_model_st": "test-default-st-embed-model"
        },
        # Include other sections if LSW init indirectly accesses them, though it shouldn't.
        # For now, 'lsw' section is primary.
    }

    # Mock config_loader.get_config() to return our default_mock_lsw_config
    # This ensures that when LSW calls get_config(), it receives our controlled version.
    monkeypatch.setattr('core.llm_service_wrapper.get_config', lambda: default_mock_lsw_config)

    # Yield the config in case a test wants to inspect it, though typically
    # LSW will pick it up automatically via the mocked get_config.
    yield default_mock_lsw_config

    # Teardown (after test runs): Reset config_loader cache again just in case
    config_loader._config = None
    config_loader._project_root = None

class MockOllamaClient:
    """A mock for the ollama.Client."""
    def __init__(self, host=None):
        self.host = host
        self.init_called = True
        self.init_host_arg = host
        print(f"MockOllamaClient initialized with host: {host}")
        self.chat_called_with = None
        self.mock_chat_response_non_stream = None
        self.mock_chat_response_stream = None
        self.chat_should_raise_error = None
        # --- New attributes for embeddings ---
        self.embeddings_called_with = None
        self.mock_embeddings_response = None
        self.embeddings_should_raise_error = None

    # We'll add more mock methods (chat, embeddings, list) as we test LSW's other functions
    def list(self): # Add a dummy list method for now as LSW might call it in future
        print("MockOllamaClient.list called")
        return {"models": []}

    def chat(self, model, messages, stream=False, options=None, **kwargs): # Match ollama.Client.chat signature
        print(f"MockOllamaClient.chat called with: model='{model}', stream={stream}, options={options}, messages_count={len(messages)}")
        self.chat_called_with = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": options,
            "kwargs": kwargs
        }
        if self.chat_should_raise_error:
            raise self.chat_should_raise_error

        if stream:
            if self.mock_chat_response_stream:
                # This should be an iterator or a function that returns one
                return self.mock_chat_response_stream() if callable(self.mock_chat_response_stream) else iter(self.mock_chat_response_stream)
            else: # Default mock stream behavior
                def default_mock_stream():
                    yield {"message": {"content": "Default "}, "done": False}
                    yield {"message": {"content": "streamed "}, "done": False}
                    yield {"message": {"content": "response."}, "done": True}
                return default_mock_stream()
        else: # Non-streamed
            if self.mock_chat_response_non_stream:
                return self.mock_chat_response_non_stream
            else: # Default mock non-streamed response
                return {
                    "model": model,
                    "created_at": "mock_time",
                    "message": {
                        "role": "assistant",
                        "content": "Default mock non-streamed response."
                    },
                    "done": True
                }
    def embeddings(self, model, prompt, **kwargs): # Match ollama.Client.embeddings signature
        print(f"MockOllamaClient.embeddings called with: model='{model}', prompt_len={len(prompt)}")
        self.embeddings_called_with = {
            "model": model,
            "prompt": prompt,
            "kwargs": kwargs
        }
        if self.embeddings_should_raise_error:
            raise self.embeddings_should_raise_error
        
        return self.mock_embeddings_response or {"embedding": [0.1, 0.2, 0.3]} # Default mock            

class MockSentenceTransformer:
    """A mock for the sentence_transformers.SentenceTransformer."""
    def __init__(self, model_name_or_path):
        self.model_name_or_path = model_name_or_path
        self.init_called = True
        self.init_model_arg = model_name_or_path
        print(f"MockSentenceTransformer initialized with model: {self.model_name_or_path}")
        # --- New attributes for encode ---
        self.encode_called_with = None
        self.mock_encode_response = None
        self.encode_should_raise_error = None

    def encode(self, sentences, **kwargs):
        # `sentences` here is usually a string or list of strings
        print(f"MockSentenceTransformer.encode called with sentences: '{str(sentences)[:50]}...'")
        self.encode_called_with = {
            "sentences": sentences,
            "kwargs": kwargs
        }
        if self.encode_should_raise_error:
            raise self.encode_should_raise_error
        
        # LSW expects a list[float] from st_model.encode().tolist()
        # So, encode should return something that can be .tolist()-ed, like a numpy array,
        # or just directly the list if we simplify the mock. Let's return the list directly.
        if self.mock_encode_response:
            return self.mock_encode_response # Test should set this as list[float]
        
        # Default mock: if sentences is a list, return list of embeddings
        if isinstance(sentences, list):
            return [[0.01 * i] * 10 for i in range(1, len(sentences) + 1)]
        # If sentences is a single string
        return [0.05] * 10 # Default single dummy 10-dim embedding

class MockChatResponseChunk: # Helper for streaming tests
    def __init__(self, content, done, role="assistant"):
        class Message:
            def __init__(self, content, role):
                self.content = content
                self.role = role
        self.message = Message(content=content, role=role)
        self.done = done
        # Add other attributes if LSW starts using them from the chunk
        self.model = "mock_chunk_model" 
        self.created_at = "mock_chunk_time"
        # ... any other fields that might be accessed by LSW, even if just for logging
        self.done_reason = 'stop' if done else None
        self.total_duration = 0
        self.load_duration = 0
        self.prompt_eval_count = 0
        self.prompt_eval_duration = 0
        self.eval_count = 0
        self.eval_duration = 0
# --- End Helper Fixtures and Mock Classes ---

def test_lsw_init_uses_mocked_global_config_and_initializes_clients(monkeypatch, isolated_config_for_lsw):
    """
    Tests LSW initialization when it relies on the (mocked) global config.
    Verifies that default attributes are set from this config and client mocks are called.
    """
    # isolated_config_for_lsw fixture is autoused, so config_loader.get_config is already mocked.
    # The `isolated_config_for_lsw` argument here allows us to access the config if needed,
    # but the primary effect is that it ensures the mock is active.

    # Mock ollama.Client and SentenceTransformer constructors
    mock_ollama_client_instance = None
    def mock_ollama_client_constructor(host=None):
        nonlocal mock_ollama_client_instance
        mock_ollama_client_instance = MockOllamaClient(host=host)
        return mock_ollama_client_instance

    mock_st_client_instance = None
    def mock_st_constructor(model_name_or_path):
        nonlocal mock_st_client_instance
        mock_st_client_instance = MockSentenceTransformer(model_name_or_path=model_name_or_path)
        return mock_st_client_instance

    monkeypatch.setattr(ollama, 'Client', mock_ollama_client_constructor)
    # Patch SentenceTransformer where it's imported in llm_service_wrapper
    # Assuming: from sentence_transformers import SentenceTransformer
    # So, we patch 'core.llm_service_wrapper.SentenceTransformer'
    monkeypatch.setattr('core.llm_service_wrapper.SentenceTransformer', mock_st_constructor)
    # Also ensure SENTENCE_TRANSFORMERS_AVAILABLE is True for this test path in LSW
    monkeypatch.setattr('core.llm_service_wrapper.SENTENCE_TRANSFORMERS_AVAILABLE', True)

    # Initialize LSW - it should use the mocked config_loader.get_config()
    lsw = LLMServiceWrapper()

    # Assertions:
    # 1. Check attributes loaded from the mock config provided by isolated_config_for_lsw
    mock_cfg_data = isolated_config_for_lsw # Get the config from the fixture
    assert lsw.ollama_host == mock_cfg_data['lsw']['ollama_host']
    assert lsw.default_chat_model == mock_cfg_data['lsw']['default_chat_model']
    assert lsw.default_embedding_model_ollama == mock_cfg_data['lsw']['default_embedding_model_ollama']
    assert lsw.default_embedding_model_st == mock_cfg_data['lsw']['default_embedding_model_st']

    # 2. Check that our mock Ollama client was initialized
    assert mock_ollama_client_instance is not None
    assert mock_ollama_client_instance.init_called
    assert mock_ollama_client_instance.init_host_arg == mock_cfg_data['lsw']['ollama_host']
    assert lsw.client is mock_ollama_client_instance # LSW should hold our mock instance

    # 3. Check that our mock SentenceTransformer was initialized
    assert mock_st_client_instance is not None
    assert mock_st_client_instance.init_called
    assert mock_st_client_instance.init_model_arg == mock_cfg_data['lsw']['default_embedding_model_st']
    assert lsw.st_model is mock_st_client_instance # LSW should hold our mock instance

def test_lsw_init_with_direct_config_parameter(monkeypatch):
    """
    Tests LSW initialization when a config dict is passed directly to its constructor.
    """
    direct_test_config = {
        "lsw": {
            "ollama_host": "http://direct-config-host:8080",
            "default_chat_model": "direct-config-chat:v1",
            "default_embedding_model_ollama": "direct-config-ollama-embed:v1",
            "default_embedding_model_st": "direct-config-st-embed"
        }
    }

    # Mock client constructors as before
    mock_ollama_client_instance = None
    def mock_ollama_client_constructor(host=None):
        nonlocal mock_ollama_client_instance
        mock_ollama_client_instance = MockOllamaClient(host=host)
        return mock_ollama_client_instance

    mock_st_client_instance = None
    def mock_st_constructor(model_name_or_path):
        nonlocal mock_st_client_instance
        mock_st_client_instance = MockSentenceTransformer(model_name_or_path=model_name_or_path)
        return mock_st_client_instance

    monkeypatch.setattr(ollama, 'Client', mock_ollama_client_constructor)
    monkeypatch.setattr('core.llm_service_wrapper.SentenceTransformer', mock_st_constructor)
    monkeypatch.setattr('core.llm_service_wrapper.SENTENCE_TRANSFORMERS_AVAILABLE', True)

    # Initialize LSW with the direct_test_config
    # The isolated_config_for_lsw fixture is still active and mocks get_config(),
    # but LSW should prioritize the directly passed config.
    lsw = LLMServiceWrapper(config=direct_test_config)

    # Assertions:
    assert lsw.ollama_host == direct_test_config['lsw']['ollama_host']
    assert lsw.default_chat_model == direct_test_config['lsw']['default_chat_model']
    # ... (assert other attributes match direct_test_config) ...

    assert mock_ollama_client_instance is not None
    assert mock_ollama_client_instance.init_host_arg == direct_test_config['lsw']['ollama_host']
    assert lsw.client is mock_ollama_client_instance

    assert mock_st_client_instance is not None
    assert mock_st_client_instance.init_model_arg == direct_test_config['lsw']['default_embedding_model_st']
    assert lsw.st_model is mock_st_client_instance

def test_lsw_init_ollama_client_init_fails(monkeypatch, isolated_config_for_lsw):
    """
    Tests LSW initialization when ollama.Client constructor raises an exception.
    LSW should still initialize but self.client should be None.
    """
    def mock_ollama_client_constructor_raises_error(host=None):
        print(f"MockOllamaClient constructor called for host {host}, raising simulated error.")
        raise Exception("Simulated Ollama connection error")

    monkeypatch.setattr(ollama, 'Client', mock_ollama_client_constructor_raises_error)
    # We don't need to mock SentenceTransformer here as Ollama client init comes first

    # The isolated_config_for_lsw fixture will provide the config.
    # LSW should catch the exception from ollama.Client and set self.client to None.
    lsw = LLMServiceWrapper()

    assert lsw.client is None
    # Check that other attributes are still set from config
    mock_cfg_data = isolated_config_for_lsw
    assert lsw.ollama_host == mock_cfg_data['lsw']['ollama_host']
    # st_model might or might not be initialized depending on its own success/failure,
    # which is fine for this specific test focusing on Ollama client failure.
    # We can assert it tries to load ST if SENTENCE_TRANSFORMERS_AVAILABLE is True.

def test_lsw_init_sentence_transformer_load_fails(monkeypatch, isolated_config_for_lsw):
    """
    Tests LSW initialization when SentenceTransformer model loading fails.
    LSW should initialize, self.client (Ollama) should be fine, but self.st_model should be None.
    """
    # Mock Ollama client to succeed
    mock_ollama_client_instance = MockOllamaClient() # Use our existing mock
    monkeypatch.setattr(ollama, 'Client', lambda host=None: mock_ollama_client_instance)

    # Mock SentenceTransformer constructor to raise an error
    def mock_st_constructor_raises_error(model_name_or_path):
        print(f"MockSentenceTransformer constructor called for model {model_name_or_path}, raising simulated error.")
        raise Exception("Simulated ST model load failure")

    monkeypatch.setattr('core.llm_service_wrapper.SentenceTransformer', mock_st_constructor_raises_error)
    monkeypatch.setattr('core.llm_service_wrapper.SENTENCE_TRANSFORMERS_AVAILABLE', True)

    lsw = LLMServiceWrapper()

    assert lsw.client is mock_ollama_client_instance # Ollama client should be fine
    assert lsw.st_model is None
    # Check that other attributes are still set
    mock_cfg_data = isolated_config_for_lsw
    assert lsw.default_embedding_model_st == mock_cfg_data['lsw']['default_embedding_model_st']

def test_lsw_init_sentence_transformer_library_not_available(monkeypatch, isolated_config_for_lsw):
    """
    Tests LSW initialization when the sentence_transformers library is not available.
    LSW should initialize, self.client (Ollama) fine, self.st_model should be None.
    """
    mock_ollama_client_instance = MockOllamaClient()
    monkeypatch.setattr(ollama, 'Client', lambda host=None: mock_ollama_client_instance)

    # Simulate library not being available by setting the flag in LSW's context
    monkeypatch.setattr('core.llm_service_wrapper.SENTENCE_TRANSFORMERS_AVAILABLE', False)
    
    # We don't need to mock 'core.llm_service_wrapper.SentenceTransformer' itself here because
    # LSW should check SENTENCE_TRANSFORMERS_AVAILABLE before trying to use it.

    lsw = LLMServiceWrapper()

    assert lsw.client is mock_ollama_client_instance
    assert lsw.st_model is None
    mock_cfg_data = isolated_config_for_lsw
    assert lsw.default_embedding_model_st == mock_cfg_data['lsw']['default_embedding_model_st']

def test_lsw_list_local_models_success(monkeypatch, isolated_config_for_lsw):
    """Tests list_local_models successfully retrieves and returns models."""
    # Mock ollama.Client and its list method
    mock_ollama_client_instance = MockOllamaClient() # Our existing mock
    expected_models_data = {
        "models": [
            {"name": "test-model-1:latest", "size": 12345},
            {"name": "another-model:7b", "size": 67890}
        ]
    }
    # Define what the mock's list() method should return
    mock_ollama_client_instance.list = lambda: expected_models_data

    monkeypatch.setattr(ollama, 'Client', lambda host=None: mock_ollama_client_instance)
    # For this test, ST availability doesn't matter much, but let's be consistent
    monkeypatch.setattr('core.llm_service_wrapper.SENTENCE_TRANSFORMERS_AVAILABLE', False)

    lsw = LLMServiceWrapper() # Uses isolated_config_for_lsw
    assert lsw.client is mock_ollama_client_instance # Ensure our mock client is set

    models = lsw.list_local_models()

    assert models == expected_models_data["models"]

def test_lsw_list_local_models_client_not_initialized(monkeypatch, isolated_config_for_lsw):
    """Tests list_local_models when LSW's Ollama client is None."""
    # Mock ollama.Client constructor to raise an error, so lsw.client will be None
    def mock_ollama_client_constructor_raises_error(host=None):
        raise Exception("Simulated Ollama connection error for list_local_models test")
    monkeypatch.setattr(ollama, 'Client', mock_ollama_client_constructor_raises_error)

    lsw = LLMServiceWrapper()
    assert lsw.client is None # Pre-condition for the test

    models = lsw.list_local_models()

    assert models == [] # Expected to return an empty list on error

def test_lsw_list_local_models_ollama_api_error(monkeypatch, isolated_config_for_lsw):
    """Tests list_local_models when the ollama client's list() method raises an exception."""
    mock_ollama_client_instance = MockOllamaClient()
    # Make the mock's list() method raise an error
    def mock_list_raises_error():
        raise Exception("Simulated Ollama API error during list()")
    mock_ollama_client_instance.list = mock_list_raises_error

    monkeypatch.setattr(ollama, 'Client', lambda host=None: mock_ollama_client_instance)
    monkeypatch.setattr('core.llm_service_wrapper.SENTENCE_TRANSFORMERS_AVAILABLE', False)

    lsw = LLMServiceWrapper()
    assert lsw.client is mock_ollama_client_instance

    models = lsw.list_local_models()

    assert models == [] # Expected to return an empty list on error

def test_lsw_gcc_non_stream_success_default_model(monkeypatch, isolated_config_for_lsw):
    """
    Tests generate_chat_completion (non-streamed) using LSW's default chat model.
    GCC stands for Generate Chat Completion.
    """
    mock_ollama_client = MockOllamaClient()
    expected_response_content = "Test response from default model."
    mock_ollama_client.mock_chat_response_non_stream = { # Ollama's typical non-streamed structure
        "message": {"role": "assistant", "content": expected_response_content}
    }
    monkeypatch.setattr(ollama, 'Client', lambda host=None: mock_ollama_client)
    monkeypatch.setattr('core.llm_service_wrapper.SENTENCE_TRANSFORMERS_AVAILABLE', False)

    lsw = LLMServiceWrapper() # Uses config from isolated_config_for_lsw
    test_messages = [{"role": "user", "content": "Hello"}]

    response = lsw.generate_chat_completion(messages=test_messages, stream=False)

    assert response == expected_response_content
    assert mock_ollama_client.chat_called_with is not None
    # LSW's default_chat_model comes from the mocked config
    assert mock_ollama_client.chat_called_with["model"] == isolated_config_for_lsw['lsw']['default_chat_model']
    assert mock_ollama_client.chat_called_with["messages"] == test_messages
    assert mock_ollama_client.chat_called_with["stream"] is False
    assert "temperature" in mock_ollama_client.chat_called_with["options"] # Default temperature

def test_lsw_gcc_non_stream_success_explicit_model_and_options(monkeypatch, isolated_config_for_lsw):
    """
    Tests generate_chat_completion (non-streamed) with an explicit model and options.
    """
    mock_ollama_client = MockOllamaClient()
    expected_response_content = "Response from explicit model."
    mock_ollama_client.mock_chat_response_non_stream = {
        "message": {"content": expected_response_content}
    }
    monkeypatch.setattr(ollama, 'Client', lambda host=None: mock_ollama_client)
    monkeypatch.setattr('core.llm_service_wrapper.SENTENCE_TRANSFORMERS_AVAILABLE', False)

    lsw = LLMServiceWrapper()
    test_messages = [{"role": "user", "content": "Query"}]
    explicit_model = "explicit-test-model:v1"
    explicit_temp = 0.2
    explicit_max_tokens = 100

    response = lsw.generate_chat_completion(
        messages=test_messages,
        model_name=explicit_model,
        temperature=explicit_temp,
        max_tokens=explicit_max_tokens,
        stream=False,
        top_p=0.9 # Example of an additional option
    )

    assert response == expected_response_content
    assert mock_ollama_client.chat_called_with["model"] == explicit_model
    assert mock_ollama_client.chat_called_with["options"]["temperature"] == explicit_temp
    assert mock_ollama_client.chat_called_with["options"]["num_predict"] == explicit_max_tokens
    assert mock_ollama_client.chat_called_with["options"]["top_p"] == 0.9 # Check other options pass through

def test_lsw_gcc_non_stream_client_not_initialized(monkeypatch, isolated_config_for_lsw):
    """Tests GCC (non-streamed) when LSW's Ollama client is None."""
    def mock_ollama_client_constructor_raises_error(host=None):
        raise Exception("Simulated Ollama client init error for GCC test")
    monkeypatch.setattr(ollama, 'Client', mock_ollama_client_constructor_raises_error)

    lsw = LLMServiceWrapper()
    assert lsw.client is None

    response = lsw.generate_chat_completion(messages=[{"role": "user", "content": "Hi"}], stream=False)
    assert response is None # Expect None on error

def test_lsw_gcc_non_stream_ollama_api_error(monkeypatch, isolated_config_for_lsw):
    """Tests GCC (non-streamed) when the ollama client's chat() method raises an exception."""
    mock_ollama_client = MockOllamaClient()
    mock_ollama_client.chat_should_raise_error = Exception("Simulated Ollama API error during chat")
    monkeypatch.setattr(ollama, 'Client', lambda host=None: mock_ollama_client)
    monkeypatch.setattr('core.llm_service_wrapper.SENTENCE_TRANSFORMERS_AVAILABLE', False)

    lsw = LLMServiceWrapper()
    response = lsw.generate_chat_completion(messages=[{"role": "user", "content": "Test"}], stream=False)
    assert response is None # Expect None on error

def test_lsw_gcc_stream_success(monkeypatch, isolated_config_for_lsw):
    """Tests generate_chat_completion (streamed) successfully yields content."""
    mock_ollama_client = MockOllamaClient()
    # Define the sequence of chunks our mock client's chat() method should yield
    # test_lsw_gcc_stream_success
    mock_stream_chunks = [
        MockChatResponseChunk(content="Hel", done=False),
        MockChatResponseChunk(content="lo ", done=False),
        MockChatResponseChunk(content="World!", done=False),
        MockChatResponseChunk(content="", done=True) # Final chunk
    ]
    mock_ollama_client.mock_chat_response_stream = iter(mock_stream_chunks)

    monkeypatch.setattr(ollama, 'Client', lambda host=None: mock_ollama_client)
    monkeypatch.setattr('core.llm_service_wrapper.SENTENCE_TRANSFORMERS_AVAILABLE', False)

    lsw = LLMServiceWrapper()
    test_messages = [{"role": "user", "content": "Tell me a story"}]

    response_generator = lsw.generate_chat_completion(messages=test_messages, stream=True)

    assert hasattr(response_generator, '__iter__') and hasattr(response_generator, '__next__') # Check it's a generator/iterator

    accumulated_content = "".join(list(response_generator)) # Consume the generator

    assert accumulated_content == "Hello World!"
    assert mock_ollama_client.chat_called_with is not None
    assert mock_ollama_client.chat_called_with["stream"] is True
    assert mock_ollama_client.chat_called_with["model"] == isolated_config_for_lsw['lsw']['default_chat_model']

def test_lsw_gcc_stream_empty_content_chunks(monkeypatch, isolated_config_for_lsw):
    """Tests GCC (streamed) when some chunks have no content but are not 'done'."""
    mock_ollama_client = MockOllamaClient()
    mock_stream_chunks = [
        MockChatResponseChunk(content="Data", done=False),
        MockChatResponseChunk(content=None, done=False), # Handled by LSW if content is None
        MockChatResponseChunk(content="More", done=False),
        MockChatResponseChunk(content="", done=True)
    ]
    mock_ollama_client.mock_chat_response_stream = iter(mock_stream_chunks)
    monkeypatch.setattr(ollama, 'Client', lambda host=None: mock_ollama_client)
    lsw = LLMServiceWrapper()
    response_generator = lsw.generate_chat_completion(messages=[{"role": "user", "content": "Q"}], stream=True)
    accumulated_content = "".join(list(response_generator))
    assert accumulated_content == "DataMore" # LSW should filter out empty/None content yields

def test_lsw_gcc_stream_ollama_api_error_during_stream(monkeypatch, isolated_config_for_lsw):
    """Tests GCC (streamed) when the client's stream itself raises an error."""
    mock_ollama_client = MockOllamaClient()

    def erroring_stream_generator():
        yield MockChatResponseChunk(content="First part...", done=False)
        raise Exception("Simulated network error during stream")

    mock_ollama_client.mock_chat_response_stream = erroring_stream_generator
    # Note: erroring_stream_generator is a function that returns a generator.
    # Our MockOllamaClient.chat will call it if callable.

    monkeypatch.setattr(ollama, 'Client', lambda host=None: mock_ollama_client)
    lsw = LLMServiceWrapper()

    response_generator = lsw.generate_chat_completion(messages=[{"role": "user", "content": "Q"}], stream=True)
    
    # Consume the generator and expect an error
    # LSW's current stream error handling is to print and yield nothing further.
    # The CIL or CO would then handle the empty/aborted stream.
    # For this unit test, we verify that iteration stops and what was yielded before error.
    collected_chunks = []
    try:
        for chunk in response_generator:
            collected_chunks.append(chunk)
    except Exception as e:
        # If LSW's generator re-raises, we could catch it here.
        # Current LSW stream error handling might just stop yielding.
        print(f"Caught error while consuming generator: {e}") # For debugging
        pass 
    
    assert "".join(collected_chunks) == "First part..."
    # The key is that the LSW's wrapper around the stream generator
    # should handle the exception from the underlying ollama stream gracefully.
    # The LSW's content_generator currently catches Exception and breaks.

def test_lsw_gcc_stream_client_not_initialized(monkeypatch, isolated_config_for_lsw):
    """Tests GCC (streamed) when LSW's Ollama client is None."""
    def mock_ollama_client_constructor_raises_error(host=None):
        raise Exception("Simulated Ollama client init error for GCC stream test")
    monkeypatch.setattr(ollama, 'Client', mock_ollama_client_constructor_raises_error)

    lsw = LLMServiceWrapper() # LSW __init__ will set self.client to None
    assert lsw.client is None

    response_generator = lsw.generate_chat_completion(messages=[{"role": "user", "content": "Hi"}], stream=True)
    assert hasattr(response_generator, '__iter__') # Should be an iterator
    assert list(response_generator) == [] # Consuming it should yield an empty list


def test_lsw_ge_ollama_success_default_model(monkeypatch, isolated_config_for_lsw):
    """Tests generate_embedding (source='ollama') with LSW's default Ollama embedding model. GE = Generate Embedding."""
    mock_ollama_client = MockOllamaClient()
    expected_embedding = [0.1, 0.2, 0.3, 0.4]
    mock_ollama_client.mock_embeddings_response = {"embedding": expected_embedding}
    monkeypatch.setattr(ollama, 'Client', lambda host=None: mock_ollama_client)
    lsw = LLMServiceWrapper()

    embedding = lsw.generate_embedding(text_to_embed="Test text", source="ollama")

    assert embedding == expected_embedding
    assert mock_ollama_client.embeddings_called_with is not None
    assert mock_ollama_client.embeddings_called_with["model"] == isolated_config_for_lsw['lsw']['default_embedding_model_ollama']
    assert mock_ollama_client.embeddings_called_with["prompt"] == "Test text"

def test_lsw_ge_ollama_success_explicit_model(monkeypatch, isolated_config_for_lsw):
    """Tests generate_embedding (source='ollama') with an explicit model name."""
    mock_ollama_client = MockOllamaClient()
    expected_embedding = [0.5, 0.6]
    mock_ollama_client.mock_embeddings_response = {"embedding": expected_embedding}
    monkeypatch.setattr(ollama, 'Client', lambda host=None: mock_ollama_client)
    lsw = LLMServiceWrapper()
    explicit_model = "explicit-ollama-embed:v2"

    embedding = lsw.generate_embedding(text_to_embed="Another text", source="ollama", model_name=explicit_model)

    assert embedding == expected_embedding
    assert mock_ollama_client.embeddings_called_with["model"] == explicit_model

def test_lsw_ge_ollama_client_not_initialized(monkeypatch, isolated_config_for_lsw):
    """Tests GE (source='ollama') when LSW's Ollama client is None."""
    monkeypatch.setattr(ollama, 'Client', lambda host=None: exec("raise Exception('Ollama client init failed for GE test')"))
    lsw = LLMServiceWrapper()
    assert lsw.client is None
    embedding = lsw.generate_embedding(text_to_embed="Text", source="ollama")
    assert embedding is None

def test_lsw_ge_ollama_api_error(monkeypatch, isolated_config_for_lsw):
    """Tests GE (source='ollama') when the ollama client's embeddings() method raises an exception."""
    mock_ollama_client = MockOllamaClient()
    mock_ollama_client.embeddings_should_raise_error = Exception("Simulated Ollama API error during embeddings")
    monkeypatch.setattr(ollama, 'Client', lambda host=None: mock_ollama_client)
    lsw = LLMServiceWrapper()
    embedding = lsw.generate_embedding(text_to_embed="Text", source="ollama")
    assert embedding is None

def test_lsw_ge_st_success_default_model(monkeypatch, isolated_config_for_lsw):
    """Tests generate_embedding (source='st') with LSW's default ST model."""
    
    # --- MODIFIED MOCKING ---
    # This will store the ST model instance created by the mock constructor
    created_mock_st_instance = None 
    
    def mock_st_constructor_that_records_arg(model_name_or_path):
        nonlocal created_mock_st_instance
        # Create a new MockSentenceTransformer instance each time,
        # initialized with the model_name_or_path that LSW passes.
        instance = MockSentenceTransformer(model_name_or_path=model_name_or_path)
        created_mock_st_instance = instance # Store it so we can assert against it
        return instance

    monkeypatch.setattr('core.llm_service_wrapper.SentenceTransformer', mock_st_constructor_that_records_arg)
    # --- END MODIFIED MOCKING ---

    monkeypatch.setattr('core.llm_service_wrapper.SENTENCE_TRANSFORMERS_AVAILABLE', True)
    monkeypatch.setattr(ollama, 'Client', MockOllamaClient)

    lsw = LLMServiceWrapper() # Uses mocked config for default_embedding_model_st
    
    # Now, lsw.st_model should be the instance created by mock_st_constructor_that_records_arg
    assert lsw.st_model is created_mock_st_instance 
    assert isinstance(lsw.st_model, MockSentenceTransformer) # Also good to check type

    # Set the mock response on the *actual instance LSW is holding*
    expected_st_embedding = [0.01, 0.02, 0.03]
    if lsw.st_model: # Check if st_model was successfully set (it should be)
        lsw.st_model.mock_encode_response = expected_st_embedding

    embedding = lsw.generate_embedding(text_to_embed="ST test text", source="st")

    assert embedding == expected_st_embedding
    
    # Check encode was called on the correct mock instance
    assert lsw.st_model.encode_called_with is not None
    assert lsw.st_model.encode_called_with["sentences"] == "ST test text"
    
    # Check that LSW initialized ST with the default model from config
    # This assertion should now pass because mock_st_constructor_that_records_arg
    # used the correct model_name_or_path to initialize the MockSentenceTransformer.
    assert lsw.st_model.init_model_arg == isolated_config_for_lsw['lsw']['default_embedding_model_st']

def test_lsw_ge_st_model_not_loaded(monkeypatch, isolated_config_for_lsw):
    """Tests GE (source='st') when LSW's ST model (self.st_model) is None."""
    # Make ST constructor fail so self.st_model is None
    monkeypatch.setattr('core.llm_service_wrapper.SentenceTransformer', lambda model_name_or_path: exec("raise Exception('ST load failed for GE test')"))
    monkeypatch.setattr('core.llm_service_wrapper.SENTENCE_TRANSFORMERS_AVAILABLE', True)
    monkeypatch.setattr(ollama, 'Client', MockOllamaClient)

    lsw = LLMServiceWrapper()
    assert lsw.st_model is None # Pre-condition for the test

    embedding = lsw.generate_embedding(text_to_embed="Text", source="st")
    assert embedding is None

def test_lsw_ge_st_encode_error(monkeypatch, isolated_config_for_lsw):
    """Tests GE (source='st') when st_model.encode() raises an error."""
    mock_st_model = MockSentenceTransformer("dummy_path")
    mock_st_model.encode_should_raise_error = Exception("Simulated ST encode error")
    monkeypatch.setattr('core.llm_service_wrapper.SentenceTransformer', lambda model_name_or_path: mock_st_model)
    monkeypatch.setattr('core.llm_service_wrapper.SENTENCE_TRANSFORMERS_AVAILABLE', True)
    monkeypatch.setattr(ollama, 'Client', MockOllamaClient)
    lsw = LLMServiceWrapper()
    assert lsw.st_model is mock_st_model

    embedding = lsw.generate_embedding(text_to_embed="Text", source="st")
    assert embedding is None
    
def test_lsw_ge_invalid_source(monkeypatch, isolated_config_for_lsw):
    """Tests generate_embedding with an invalid source."""
    monkeypatch.setattr(ollama, 'Client', MockOllamaClient) # Needs a valid client for LSW init
    lsw = LLMServiceWrapper()
    embedding = lsw.generate_embedding(text_to_embed="Text", source="invalid_source_name")
    assert embedding is None