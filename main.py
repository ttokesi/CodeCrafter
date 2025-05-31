# offline_chat_bot/main.py

import os
import sys
import uuid # For conversation IDs
import shutil # For potential cleanup of test files during full reset
import time   # For potential delays

# Ensure project root is in path for imports if running main.py directly
# (though ideally, we might run this as a module later if the project grows)
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mmu.mmu_manager import MemoryManagementUnit
from core.llm_service_wrapper import LLMServiceWrapper
from core.agents import KnowledgeRetrieverAgent, SummarizationAgent, FactExtractionAgent
from core.orchestrator import ConversationOrchestrator
from core.config_loader import get_config, get_project_root # <--- IMPORT FROM CONFIG_LOADER

# Ensure data directory exists
os.makedirs("data", exist_ok=True)
# --- End Configuration ---


def initialize_chatbot_components(app_config: dict): 
    """Initializes and returns all core components of the chatbot using provided config."""
    print("Initializing chatbot components from configuration...")
     
    # LSW initialization
    # LSW's __init__ now accepts an optional 'config' argument.
    # If not provided, it calls get_config() itself.
    # So, we can pass app_config or let it load. Passing it is slightly more explicit here.
    lsw = LLMServiceWrapper(config=app_config)
    if not lsw.client:
        print("FATAL: Ollama client in LSW could not be initialized. Ensure Ollama is running.")
        # Access config safely for error messages
        lsw_cfg_for_error = app_config.get('lsw', {})
        default_ollama_host = lsw_cfg_for_error.get('ollama_host', 'http://localhost:11434')
        default_chat_model = lsw_cfg_for_error.get('default_chat_model', 'unavailable_model_for_error_msg')
        print(f"  Attempted to connect to: {default_ollama_host}")
        print(f"  Ensure model '{default_chat_model}' and necessary embedding models are pulled in Ollama.")
        return None 

    # MMU initialization requires an embedding_function.
    # We define a wrapper that uses our LSW instance.
    try:
        from chromadb import Documents, EmbeddingFunction, Embeddings # For type hints in wrapper
    except ImportError:
        print("CRITICAL: chromadb library not found. Cannot create LSW embedding wrapper for MMU.")
        return None

    class LSWEmbeddingFunctionForMMU(EmbeddingFunction):
        def __init__(self, lsw_instance: LLMServiceWrapper, 
                     embedding_source: str, # e.g., "st" or "ollama"
                     model_name: str = None): # Specific model for this embedder instance
            self.lsw = lsw_instance
            self.source = embedding_source
            self.model_name = model_name # LSW will use its default for the source if this is None
            
            # Determine expected dimension for dummy embeddings if needed (for error case)
            # This logic can be improved by trying to get dim from a test embedding call
            self.dim = 384 # Default (e.g., for all-MiniLM-L6-v2)
            temp_model_for_dim_check = self.model_name
            if not temp_model_for_dim_check: # If model_name is None, use LSW's default for the source
                if self.source == "st": temp_model_for_dim_check = self.lsw.default_embedding_model_st
                elif self.source == "ollama": temp_model_for_dim_check = self.lsw.default_embedding_model_ollama
            
            if temp_model_for_dim_check:
                if "nomic-embed-text" in temp_model_for_dim_check: self.dim = 768
                elif "all-MiniLM-L6-v2" in temp_model_for_dim_check: self.dim = 384
                # Add other known model dimensions here
            effective_model_name_for_print = self.model_name if self.model_name else f"LSW default for source '{self.source}' (likely {self.lsw.default_embedding_model_st if self.source == 'st' else self.lsw.default_embedding_model_ollama})"
            print(f"  LSWEmbeddingFunctionForMMU: Initialized for source '{self.source}', model '{effective_model_name_for_print}', dim {self.dim}")

        def __call__(self, input_texts: Documents) -> Embeddings:
            embeddings_list: Embeddings = []
            if not input_texts: return embeddings_list
            
            for text_doc in input_texts:
                emb = self.lsw.generate_embedding(
                    text_to_embed=str(text_doc), 
                    source=self.source,
                    model_name=self.model_name # Pass the specific model, or LSW uses its default for source
                ) 
                if emb:
                    embeddings_list.append(emb)
                else: 
                    print(f"    WARNING: Embedding failed via LSW for text: {str(text_doc)[:50]}... Appending dummy.")
                    embeddings_list.append([0.0] * self.dim) 
            return embeddings_list

    # Get LTM embedding settings from config (mmu section)
    mmu_cfg = app_config.get('mmu', {})
    ltm_embedding_source = mmu_cfg.get('ltm_vector_store_embedding_source', 'st') # Default to 'st'
    # Optional: allow specifying a particular model for LTM embeddings in config
    ltm_embedding_model_name_config = mmu_cfg.get('ltm_vector_store_embedding_model_name', None)

    lsw_embedder_for_mmu = LSWEmbeddingFunctionForMMU(
        lsw_instance=lsw, 
        embedding_source=ltm_embedding_source,
        model_name=ltm_embedding_model_name_config
    )

    # MMU's __init__ now also takes an optional 'config' argument.
    mmu = MemoryManagementUnit(
        embedding_function=lsw_embedder_for_mmu,
        config=app_config 
    )

    # Agents initialization
    agents_cfg = app_config.get('agents', {})
    lsw_default_chat_model_for_agents = app_config.get('lsw',{}).get('default_chat_model') # Fallback

    # Get agent-specific models from config, or fallback to LSW's default chat model
    summarizer_model = agents_cfg.get('summarization_agent_model', lsw_default_chat_model_for_agents)
    fact_extractor_model = agents_cfg.get('fact_extraction_agent_model', lsw_default_chat_model_for_agents)

    knowledge_retriever = KnowledgeRetrieverAgent(mmu=mmu) # KRA doesn't take model config directly
    summarizer = SummarizationAgent(lsw=lsw, default_model_name=summarizer_model)
    fact_extractor = FactExtractionAgent(lsw=lsw, default_model_name=fact_extractor_model)

    # ConversationOrchestrator's __init__ also takes an optional 'config' argument.
    orchestrator = ConversationOrchestrator(
        mmu=mmu, lsw=lsw,
        knowledge_retriever=knowledge_retriever,
        summarizer=summarizer, fact_extractor=fact_extractor,
        config=app_config
    )
    
    print("\nChatbot components initialized successfully using master configuration.")
    return orchestrator, mmu


def run_cli():
    print("Loading application configuration...")
    try:
        app_cfg = get_config() 
    except Exception as e:
        print(f"FATAL: Could not load configuration. Error: {e}")
        print("Please ensure 'config.yaml' exists in the project root and is correctly formatted.")
        return

    # data_dir is created by get_config() if it doesn't exist and is specified in config
    data_dir_from_config = app_cfg.get('data_dir', 'data')
    abs_data_dir = os.path.join(get_project_root(), data_dir_from_config)
    if not os.path.exists(abs_data_dir): # Double check, though get_config should handle it
        try:
            os.makedirs(abs_data_dir, exist_ok=True)
            print(f"Ensured data directory exists: {abs_data_dir}")
        except Exception as e:
            print(f"Warning: Could not create data directory '{abs_data_dir}'. Error: {e}")
    
    print("Starting chatbot CLI...")
    components = initialize_chatbot_components(app_cfg)
    if components is None:
        print("Failed to initialize chatbot. Exiting.")
        return 
    
    orchestrator, mmu_instance = components
    
    print("\nWelcome to Offline ChatBot!")
    co_default_model = app_cfg.get('orchestrator', {}).get('default_llm_model', 
                          app_cfg.get('lsw', {}).get('default_chat_model', 'N/A_MODEL_IN_CONFIG'))
    print(f"Using LLM: {co_default_model} for responses.")
    print("Type your message, or use commands: /quit, /new, /reset_memory")
    
    # ... (rest of CLI loop as before, it's already robust) ...
    current_conversation_id = str(uuid.uuid4())
    if orchestrator: 
        orchestrator.active_conversation_id = current_conversation_id 
        orchestrator.mmu.clear_stm() 
    print(f"Starting new conversation: {current_conversation_id}")
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input: continue
            if user_input.lower() == "/quit":
                print("Exiting chatbot. Goodbye!")
                if hasattr(mmu_instance, 'mtm') and mmu_instance.mtm.is_persistent and hasattr(mmu_instance.mtm.db, 'close'):
                    mmu_instance.mtm.db.close()
                break
            # ... (other commands: /new, /reset_memory) ...
            elif user_input.lower() == "/new":
                current_conversation_id = str(uuid.uuid4())
                print(f"\nStarting new conversation: {current_conversation_id}")
                if orchestrator:
                    orchestrator.active_conversation_id = current_conversation_id 
                    orchestrator.mmu.clear_stm() 
                print("  STM cleared for new conversation.")
                continue
            elif user_input.lower() == "/reset_memory":
                confirm = input("WARNING: This will erase ALL LTM (logs, facts, vector store) and MTM. STM will be cleared. Are you sure? (yes/no): ").strip().lower()
                if confirm == "yes": # ... (reset logic) ...
                    if mmu_instance: mmu_instance.reset_all_memory(confirm_reset=True)
                    current_conversation_id = str(uuid.uuid4()) 
                    if orchestrator: orchestrator.active_conversation_id = current_conversation_id; orchestrator.mmu.clear_stm()
                    print(f"\nMemory reset. Starting new conversation: {current_conversation_id}")
                else: print("Memory reset aborted.")
                continue
            
            print("Bot: ", end="", flush=True)
            if orchestrator:
                assistant_response_generator = orchestrator.handle_user_message(
                    user_message=user_input, conversation_id=current_conversation_id )
                if assistant_response_generator:
                    try:
                        for chunk in assistant_response_generator: print(chunk, end="", flush=True)
                        print() 
                    except Exception as e: print(f"\nError processing streamed response: {e}\n")
                else: print("I'm sorry, I couldn't process that.")
            else: print("Orchestrator not available.")
        except KeyboardInterrupt: # ... 
            print("\nExiting chatbot (KeyboardInterrupt)...")
            if hasattr(mmu_instance, 'mtm') and mmu_instance.mtm.is_persistent and hasattr(mmu_instance.mtm.db, 'close'):
                mmu_instance.mtm.db.close()
            break
        except Exception as e: # ... 
            print(f"\nAn unexpected error occurred in CLI: {e}")
            import traceback; traceback.print_exc()
            if hasattr(mmu_instance, 'mtm') and mmu_instance.mtm.is_persistent and hasattr(mmu_instance.mtm.db, 'close'):
                mmu_instance.mtm.db.close()
            break

if __name__ == "__main__":
    run_cli()