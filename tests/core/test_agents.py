# tests/core/test_agents.py

import pytest
import os
import sys

# --- Path Adjustment ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --- End Path Adjustment ---

# Import the agent classes we want to test
from core.agents import KnowledgeRetrieverAgent, SummarizationAgent, FactExtractionAgent

# Import classes that Agents depend on, so we can mock them
from mmu.mmu_manager import MemoryManagementUnit
from core.llm_service_wrapper import LLMServiceWrapper # For Summarization & FactExtraction agents later

# We'll need to mock config_loader.get_config for agents that might use it indirectly
# or if their dependencies use it during initialization.
from core import config_loader

# --- Helper Mock Classes and Fixtures ---

# class MockMemoryManagementUnit:
#     """A mock for the MemoryManagementUnit."""
#     def __init__(self):
#         print("MockMemoryManagementUnit initialized for an Agent test.")
#         # --- Attributes to store call arguments and control responses ---
#         self.semantic_search_ltm_vector_store_called_with = None
#         self.mock_vector_store_results = [] # Default to empty list

#         self.get_ltm_facts_called_with = None
#         self.mock_skb_fact_results = [] # Default to empty list

#     def semantic_search_ltm_vector_store(self, query_text: str, top_k: int = 5, metadata_filter: dict = None):
#         print(f"MockMMU.semantic_search_ltm_vector_store called with query: '{query_text}', top_k: {top_k}")
#         self.semantic_search_ltm_vector_store_called_with = {
#             "query_text": query_text,
#             "top_k": top_k,
#             "metadata_filter": metadata_filter
#         }
#         return self.mock_vector_store_results # Return the mock results

#     def get_ltm_facts(self, subject: str = None, predicate: str = None, object_value: str = None) -> list:
#         print(f"MockMMU.get_ltm_facts called with S='{subject}', P='{predicate}', O='{object_value}'")
#         self.get_ltm_facts_called_with = {
#             "subject": subject,
#             "predicate": predicate,
#             "object_value": object_value
#         }
#         # Return a copy to prevent modification of the mock attribute by the agent if it mutates results
#         return list(self.mock_skb_fact_results)

#     # Add other MMU methods if KRA or other agents start using them.
#     # For now, these two are the primary ones KRA uses.

# Fixture to provide a clean MockMMU instance for each test that needs it
@pytest.fixture
def mock_mmu(mocker): # Add mocker fixture
    """
    Provides a mock MemoryManagementUnit object that passes isinstance checks
    and allows setting return values for its methods.
    """
    # Create a mock object that has the same specification as MemoryManagementUnit.
    # `instance=True` makes it behave like an instance for isinstance() checks.
    # `spec=True` (or just passing the class) ensures it only has attributes/methods of the real class.
    mmu_mock = mocker.create_autospec(MemoryManagementUnit, instance=True)
    
    # You can pre-configure some default behavior if needed, e.g.,
    mmu_mock.semantic_search_ltm_vector_store.return_value = []
    mmu_mock.get_ltm_facts.return_value = []
    # print("mock_mmu fixture: Created autospecced MemoryManagementUnit mock.")
    return mmu_mock

# Placeholder for future mock LSW
class MockLLMServiceWrapper:
    def __init__(self, config=None): # Add config to match real LSW
        self.config = config or {} # Store it if needed
        self.default_chat_model = self.config.get('lsw', {}).get('default_chat_model', 'mock_default_lsw_chat_model')
        print(f"MockLLMServiceWrapper initialized. Default chat model: {self.default_chat_model}")
        # Attributes for generate_chat_completion
        self.gcc_called_with = None
        self.mock_gcc_response = "Default mock LLM response."
        self.gcc_should_raise_error = None

    def generate_chat_completion(self, messages: list, model_name: str = None, **kwargs):
        print(f"MockLSW.generate_chat_completion called with model: '{model_name or self.default_chat_model}', messages_count: {len(messages)}")
        self.gcc_called_with = {"messages": messages, "model_name": model_name or self.default_chat_model, "kwargs": kwargs}
        if self.gcc_should_raise_error:
            raise self.gcc_should_raise_error
        return self.mock_gcc_response
    
    # Add other LSW methods as needed for agent tests (e.g., generate_embedding if an agent uses it)

@pytest.fixture
def mock_lsw(isolated_agent_config): # Depends on isolated_agent_config
    """Provides a fresh instance of MockLLMServiceWrapper, configured."""
    # Pass the 'lsw' part of the isolated_agent_config to MockLSW
    # to simulate LSW loading its specific configuration.
    lsw_specific_config = {"lsw": isolated_agent_config.get('lsw', {})}
    return MockLLMServiceWrapper(config=lsw_specific_config)


# Autouse fixture to ensure config_loader is reset for agent tests
# (Similar to the one in test_llm_service_wrapper.py but tailored for agents)
@pytest.fixture(autouse=True)
def isolated_agent_config(monkeypatch):
    """
    Provides an isolated, mock configuration for Agent tests and resets
    the config_loader cache. Specific for agent configurations.
    """
    config_loader._config = None
    config_loader._project_root = None

    # Define a default mock configuration relevant to Agents and their dependencies
    default_mock_agent_related_config = {
        "lsw": { # For LSW if used by agents (Summarizer, FactExtractor)
            "default_chat_model": "agent-test-lsw-chat-model",
            # Add other LSW defaults if agents depend on them indirectly
        },
        "agents": { # For agent-specific model configs
            "summarization_agent_model": "agent-test-summarizer-model",
            "fact_extraction_agent_model": "agent-test-extractor-model"
        }
        # Add other sections like 'mmu', 'orchestrator' if agents somehow
        # read from them via get_config() (they generally shouldn't directly)
    }

    # Mock get_config to be used by LSW (when agents initialize it) or other components
    # Patch it where it's most likely to be called by dependencies of agents.
    # If agents themselves call get_config(), patch 'core.agents.get_config'.
    # If LSW (a dependency) calls get_config(), that's already handled if LSW is tested.
    # For safety, let's assume agents or their direct init dependencies might call it.
    # LSW constructor calls get_config if no config is passed.
    # Agents get LSW passed in, LSW is initialized by test setup.
    # So, the main place get_config() matters here is when MockLSW is initialized
    # by the mock_lsw fixture.
    monkeypatch.setattr(config_loader, 'get_config', lambda: default_mock_agent_related_config)


    yield default_mock_agent_related_config # Provide the config to tests if needed

    config_loader._config = None
    config_loader._project_root = None


# --- Test Functions for KnowledgeRetrieverAgent will start here ---

def test_kra_init_success(mock_mmu): # mock_mmu now comes from the updated fixture
    """Tests successful initialization of KnowledgeRetrieverAgent."""
    try:
        # mock_mmu is now an autospecced mock that should pass isinstance(mmu, MemoryManagementUnit)
        kra = KnowledgeRetrieverAgent(mmu=mock_mmu)
        assert kra.mmu is mock_mmu, "KRA should store the provided MMU instance."
        # Verify that the mock MMU was indeed an instance that KRA accepted
        # This is implicitly tested by not raising a TypeError.
    except Exception as e:
        pytest.fail(f"KnowledgeRetrieverAgent initialization failed unexpectedly: {e}")

def test_kra_init_type_error_for_invalid_mmu(mock_lsw): # This test should still pass
    """
    Tests that KnowledgeRetrieverAgent raises a TypeError if not initialized
    with a MemoryManagementUnit instance.
    """
    with pytest.raises(TypeError) as excinfo:
        # Pass an object that is definitely not a MemoryManagementUnit (nor a mock of it)
        KnowledgeRetrieverAgent(mmu=mock_lsw) 
    
    assert "KnowledgeRetrieverAgent requires an instance of MemoryManagementUnit" in str(excinfo.value)

def test_kra_search_knowledge_vector_store_only(mock_mmu): # Uses the mock_mmu fixture
    """Tests search_knowledge when only vector store search is enabled."""
    kra = KnowledgeRetrieverAgent(mmu=mock_mmu)
    
    test_query = "find in vector store"
    expected_vector_results = [{"id": "vec1", "text_chunk": "vector data", "score": 0.9}]
    mock_mmu.semantic_search_ltm_vector_store.return_value = expected_vector_results
    # We don't need to set mock_mmu.get_ltm_facts.return_value as it shouldn't be called.

    results = kra.search_knowledge(
        query_text=test_query,
        search_vector_store=True,
        search_skb_facts=False, # Explicitly disable SKB search
        top_k_vector=3
    )

    # Check assertions
    assert results["vector_results"] == expected_vector_results
    assert results["skb_fact_results"] == [] # Should be empty as SKB search was off

    # Verify MMU method calls
    mock_mmu.semantic_search_ltm_vector_store.assert_called_once_with(
        query_text=test_query, top_k=3 # Check arguments
    )
    mock_mmu.get_ltm_facts.assert_not_called() # SKB search should not have been called

def test_kra_search_knowledge_skb_facts_only_targeted(mock_mmu):
    """Tests search_knowledge with targeted SKB fact search only."""
    kra = KnowledgeRetrieverAgent(mmu=mock_mmu)

    expected_skb_results = [{"fact_id": "fact1", "subject": "test_S", "predicate": "is", "object": "test_O"}]
    mock_mmu.get_ltm_facts.return_value = expected_skb_results

    results = kra.search_knowledge(
        query_text="any query text, not used by targeted SKB", # Query text might not be used if S,P,O are given
        search_vector_store=False, # Explicitly disable vector search
        search_skb_facts=True,
        skb_subject="test_S",
        skb_predicate="is",
        skb_object="test_O"
    )

    assert results["skb_fact_results"] == expected_skb_results
    assert results["vector_results"] == []

    mock_mmu.get_ltm_facts.assert_called_once_with(
        subject="test_S", predicate="is", object_value="test_O" # KRA passes object_value
    )
    mock_mmu.semantic_search_ltm_vector_store.assert_not_called()

def test_kra_search_knowledge_both_sources_no_results(mock_mmu):
    """Tests search_knowledge when both sources are searched but yield no results."""
    kra = KnowledgeRetrieverAgent(mmu=mock_mmu)

    # Configure mocks to return empty lists
    mock_mmu.semantic_search_ltm_vector_store.return_value = []
    mock_mmu.get_ltm_facts.return_value = []

    results = kra.search_knowledge(
        query_text="find nothing",
        search_vector_store=True,
        search_skb_facts=True, # KRA will try pattern/keyword for SKB if no S,P,O
        top_k_vector=5
    )

    assert results["vector_results"] == []
    assert results["skb_fact_results"] == []

    mock_mmu.semantic_search_ltm_vector_store.assert_called_once()
    # KRA's search_skb_facts logic when no explicit S,P,O are given can be complex.
    # It will try pattern matching first on query_text, then keyword search.
    # For this "no results" test, it's enough that get_ltm_facts was called.
    # The number of calls might vary depending on keyword extraction.
    # We can assert it was called at least once if search_skb_facts is True and query_text is given.
    assert mock_mmu.get_ltm_facts.call_count > 0 # Expect it to be called for pattern/keyword search

def test_kra_search_knowledge_both_sources_with_results(mock_mmu):
    """Tests search_knowledge when both sources yield results."""
    kra = KnowledgeRetrieverAgent(mmu=mock_mmu)

    expected_vector_res = [{"id": "v1", "text_chunk": "from vector"}]
    # For SKB, KRA first tries targeted (not used here), then pattern, then keyword.
    # Let's assume the query "user name" triggers a pattern that finds a fact.
    # We need to set up get_ltm_facts to return something when called by the pattern.
    expected_skb_res_for_pattern = [{"fact_id": "f_name", "subject": "user name", "predicate": "is", "object": "TestUser"}]

    mock_mmu.semantic_search_ltm_vector_store.return_value = expected_vector_res
    
    # This mock setup is a bit simplistic for KRA's multi-stage SKB search.
    # We'll make get_ltm_facts return our expected result if called with subject="user name".
    # More refined tests later will target KRA's internal SKB search strategies.
    def mock_get_facts_for_pattern(subject=None, predicate=None, object_value=None):
        # KRA generates subjects like "user <entity>" or "user's <entity>"
        # For entity "user name", it will query for "user user name" or "user's user name"
        expected_subjects_from_kra_pattern = ["user user name", "user's user name", "user user name".rstrip('s'), "user's user name".rstrip('s')]
        if subject in expected_subjects_from_kra_pattern and predicate == "is":
            print(f"Mock_get_facts_for_pattern: Matched subject '{subject}' for 'user name' entity. Returning expected facts.")
            return expected_skb_res_for_pattern
        # Optional: print if no match for debugging
        # print(f"Mock_get_facts_for_pattern: No match for S='{subject}', P='{predicate}'. KRA keywords might be S='{subject and '%' in subject}'.")

        if subject and "%" in subject: # A crude way to detect keyword search by KRA
            # print(f"Mock_get_facts_for_pattern: Detected keyword search for subject '{subject}'. Returning [].")
            return []
        return [] # Default empty for other calls
       
    mock_mmu.get_ltm_facts.side_effect = mock_get_facts_for_pattern

    results = kra.search_knowledge(
        query_text="what is my user name and some vector stuff", # This query should trigger both
        search_vector_store=True,
        search_skb_facts=True
    )

    assert results["vector_results"] == expected_vector_res
    assert results["skb_fact_results"] == expected_skb_res_for_pattern
    
    mock_mmu.semantic_search_ltm_vector_store.assert_called_once()
    # Assert get_ltm_facts was called for one of the expected pattern-generated subjects
    pattern_call_found = False
    entity_for_pattern = "user name" # The entity KRA extracts from "what is my user name..."
    expected_subjects_for_pattern_call_check = list(set([
        f"user {entity_for_pattern}",
        f"user's {entity_for_pattern}",
        f"user {entity_for_pattern.rstrip('s')}",
        f"user's {entity_for_pattern.rstrip('s')}"
    ]))

    for call_args_tuple in mock_mmu.get_ltm_facts.call_args_list:
        if hasattr(call_args_tuple, 'kwargs'): # For mock.call objects
            kwargs_of_call = call_args_tuple.kwargs
        else: # For (args, kwargs) tuples (less common with modern mocker)
            _, kwargs_of_call = call_args_tuple
        
        if kwargs_of_call.get("subject") in expected_subjects_for_pattern_call_check and \
            kwargs_of_call.get("predicate") == "is":
            pattern_call_found = True
            break
            
    assert pattern_call_found, \
        f"get_ltm_facts was not called for any of the expected '{entity_for_pattern}' pattern subjects. Expected one of: {expected_subjects_for_pattern_call_check}. Calls: {mock_mmu.get_ltm_facts.call_args_list}"