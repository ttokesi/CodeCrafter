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
     
    lsw = LLMServiceWrapper(config=app_config)
    if not lsw.client:
        print("FATAL: Ollama client in LSW could not be initialized. Ensure Ollama is running.")
        default_ollama_host = app_config.get('lsw', {}).get('ollama_host', 'http://localhost:11434')
        default_chat_model = app_config.get('lsw', {}).get('default_chat_model', 'a_default_model')
        print(f"  Attempted to connect to: {default_ollama_host}")
        print(f"  Ensure model '{default_chat_model}' and necessary embedding models are pulled in Ollama.")
        return None # Indicate failure

    # Import ChromaDB types for the wrapper class definition
    try:
        from chromadb import Documents, EmbeddingFunction, Embeddings
    except ImportError:
        print("CRITICAL: chromadb library not found. Cannot create LSW embedding wrapper for MMU.")
        print("Please install it: pip install chromadb")
        return None # Cannot proceed without chromadb types for the wrapper

    class LSWEmbeddingFunctionForMMU(EmbeddingFunction):
        def __init__(self, lsw_instance: LLMServiceWrapper, 
                     embedding_source: str = "st", # Default to ST via LSW
                     model_name: str = None): # Specific model for this embedder instance
            self.lsw = lsw_instance
            self.source = embedding_source
            
            if model_name:
                self.model_name = model_name
            elif self.source == "st":
                self.model_name = self.lsw.default_embedding_model_st
            elif self.source == "ollama":
                self.model_name = self.lsw.default_embedding_model_ollama
            else:
                raise ValueError(f"Unsupported embedding source '{self.source}' for LSWEmbeddingFunctionForMMU")

            # Determine expected dimension for dummy embeddings if needed
            self.dim = 384 # Default, common for ST models like all-MiniLM-L6-v2
            if self.source == "ollama" and "nomic-embed-text" in self.model_name:
                self.dim = 768
            elif self.source == "st": # Add more specific ST model dimension checks if needed
                if "all-MiniLM-L6-v2" in self.model_name: self.dim = 384
                # Add other ST model dimensions here
            print(f"  LSWEmbeddingFunctionForMMU: Initialized for source '{self.source}', model '{self.model_name}', dim {self.dim}")

        def __call__(self, input_texts: Documents) -> Embeddings: # Parameter name must be 'input' for Chroma, but our wrapper can be flexible
                                                                # Let's keep it 'input_texts' here and adapt if Chroma complains later for main app
                                                                # Or stick to 'input' for direct compatibility
            actual_input: Documents = input_texts # If Chroma passes it as 'input'
            
            embeddings_list: Embeddings = []
            if not actual_input: return embeddings_list
            
            # print(f"  LSWEmbeddingFunctionForMMU: __call__ received {len(actual_input)} document(s). Using LSW source '{self.source}'.")
            for text_doc in actual_input:
                emb = self.lsw.generate_embedding(
                    text_to_embed=str(text_doc), 
                    source=self.source,
                    model_name=self.model_name
                ) 
                if emb:
                    embeddings_list.append(emb)
                else: 
                    print(f"    WARNING: Embedding failed via LSW for text: {str(text_doc)[:50]}... Appending dummy.")
                    embeddings_list.append([0.0] * self.dim) 
            return embeddings_list

    # Decide which embedding source/model to use for LTM Vector Store from config
    # Default to SentenceTransformer via LSW for good local performance and control
    ltm_embedding_source = app_config.get('mmu', {}).get('ltm_vector_store_embedding_source', 'st') # 'st' or 'ollama'
    ltm_embedding_model_name = None # Let LSWEmbeddingFunctionForMMU pick LSW's default for the source

    lsw_embedder_for_mmu = LSWEmbeddingFunctionForMMU(
        lsw_instance=lsw, 
        embedding_source=ltm_embedding_source,
        model_name=ltm_embedding_model_name # Optional, will use LSW default for source if None
    )

    # MemoryManagementUnit's __init__ is already designed to take an optional config dict and required embedding_function
    mmu = MemoryManagementUnit(
        embedding_function=lsw_embedder_for_mmu,
        config=app_config 
    )

    agents_config = app_config.get('agents', {})
    # LSW's default_chat_model serves as a fallback if agent-specific models aren't in config
    lsw_default_chat = app_config.get('lsw',{}).get('default_chat_model')

    summarizer_model = agents_config.get('summarization_agent_model', lsw_default_chat)
    fact_extractor_model = agents_config.get('fact_extraction_agent_model', lsw_default_chat)

    knowledge_retriever = KnowledgeRetrieverAgent(mmu=mmu)
    summarizer = SummarizationAgent(lsw=lsw, default_model_name=summarizer_model)
    fact_extractor = FactExtractionAgent(lsw=lsw, default_model_name=fact_extractor_model)

    # ConversationOrchestrator's __init__ is already designed to take an optional config dict
    orchestrator = ConversationOrchestrator(
        mmu=mmu, lsw=lsw,
        knowledge_retriever=knowledge_retriever,
        summarizer=summarizer, fact_extractor=fact_extractor,
        config=app_config # Pass the main app_config
    )
    
    print("\nChatbot components initialized successfully using master configuration.")
    return orchestrator, mmu

def run_cli():
    print("Loading application configuration...")
    app_cfg = get_config() # Load config once at the start
    
    # The data_dir is created by get_config() if it doesn't exist.
    
    print("Starting chatbot CLI...")
    components = initialize_chatbot_components(app_cfg) # Pass loaded config
    if components is None:
        print("Failed to initialize chatbot. Exiting.")
        return 
    
    orchestrator, mmu_instance = components # mmu_instance needed for /reset_memory command
    
    print("\nWelcome to Offline ChatBot!")
    print(f"Using LLM: {orchestrator.default_llm_model if orchestrator else 'N/A'}") # Show which LLM is primary for CO
    print("Type your message, or use commands: /quit, /new, /reset_memory")
    
    current_conversation_id = str(uuid.uuid4())
    if orchestrator: # Ensure orchestrator was initialized
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
                    print("Closing MTM database...")
                    mmu_instance.mtm.db.close()
                break
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
                if confirm == "yes":
                    print("Resetting all memory...")
                    if mmu_instance:
                        reset_success = mmu_instance.reset_all_memory(confirm_reset=True)
                        if reset_success: print("All memory has been reset.")
                        else: print("Memory reset encountered issues.")
                    current_conversation_id = str(uuid.uuid4()) # Start fresh
                    if orchestrator:
                        orchestrator.active_conversation_id = current_conversation_id
                        orchestrator.mmu.clear_stm()
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

        except KeyboardInterrupt: # ... (interrupt handling)
            print("\nExiting chatbot (KeyboardInterrupt)...")
            if hasattr(mmu_instance, 'mtm') and mmu_instance.mtm.is_persistent and hasattr(mmu_instance.mtm.db, 'close'):
                mmu_instance.mtm.db.close()
            break
        except Exception as e: # ... (general exception handling)
            print(f"\nAn unexpected error occurred in CLI: {e}")
            import traceback
            traceback.print_exc()
            if hasattr(mmu_instance, 'mtm') and mmu_instance.mtm.is_persistent and hasattr(mmu_instance.mtm.db, 'close'):
                mmu_instance.mtm.db.close()
            break

if __name__ == "__main__":
    run_cli()