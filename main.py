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
from core.agents import KnowledgeRetrieverAgent, SummarizationAgent
from core.orchestrator import ConversationOrchestrator

# --- Configuration (can be moved to a config file/class later) ---
# Paths for MMU components (these are the "production" paths, not test paths)
# These should match the defaults in mmu_manager.py or be overridden as desired.
MMU_MTM_USE_TINYDB = True # Let's try persistent MTM for the actual app
MMU_MTM_DB_PATH = 'data/mtm_store.json'
MMU_LTM_SQLITE_DB_PATH = 'data/ltm_database.db'
MMU_LTM_CHROMA_DIR = 'data/ltm_vector_store'

# LSW Configuration
LSW_OLLAMA_HOST = "http://localhost:11434"
LSW_DEFAULT_CHAT_MODEL = "gemma3:1b-it-fp16" # Ensure this model is pulled in Ollama
LSW_DEFAULT_EMBEDDING_OLLAMA = "nomic-embed-text" # Ensure pulled
LSW_DEFAULT_EMBEDDING_ST = "all-MiniLM-L6-v2"

# Ensure data directory exists
os.makedirs("data", exist_ok=True)
# --- End Configuration ---


def initialize_chatbot_components():
    """Initializes and returns all core components of the chatbot."""
    print("Initializing chatbot components...")
    
    mmu = MemoryManagementUnit(
        mtm_use_tinydb=MMU_MTM_USE_TINYDB,
        mtm_db_path=MMU_MTM_DB_PATH,
        ltm_sqlite_db_path=MMU_LTM_SQLITE_DB_PATH,
        ltm_chroma_persist_dir=MMU_LTM_CHROMA_DIR
    )
    
    lsw = LLMServiceWrapper(
        ollama_host=LSW_OLLAMA_HOST,
        default_chat_model=LSW_DEFAULT_CHAT_MODEL,
        default_embedding_model_ollama=LSW_DEFAULT_EMBEDDING_OLLAMA,
        default_embedding_model_st=LSW_DEFAULT_EMBEDDING_ST
    )
    if not lsw.client:
        print("FATAL: Ollama client in LSW could not be initialized. Ensure Ollama is running.")
        print("You may need to run 'ollama serve' in a separate terminal.")
        print(f"Attempted to connect to: {LSW_OLLAMA_HOST}")
        print(f"Make sure model '{LSW_DEFAULT_CHAT_MODEL}' and embedding model '{LSW_DEFAULT_EMBEDDING_OLLAMA}' are pulled.")
        return None # Indicate failure

    knowledge_retriever = KnowledgeRetrieverAgent(mmu=mmu)
    summarizer = SummarizationAgent(lsw=lsw) # Uses LSW's default chat model for summaries

    orchestrator = ConversationOrchestrator(
        mmu=mmu,
        lsw=lsw,
        knowledge_retriever=knowledge_retriever,
        summarizer=summarizer,
        default_llm_model=LSW_DEFAULT_CHAT_MODEL # CO uses this for its main responses
    )
    
    print("\nChatbot components initialized successfully.")
    return orchestrator, mmu # Return MMU for potential reset command


def run_cli():
    """Runs the command-line interface for the chatbot."""
    
    components = initialize_chatbot_components()
    if components is None:
        return # Exit if initialization failed
    
    orchestrator, mmu_instance = components
    
    print("\nWelcome to Offline ChatBot!")
    print("Type your message, or use commands: /quit, /new, /reset_memory")
    
    current_conversation_id = str(uuid.uuid4())
    orchestrator.active_conversation_id = current_conversation_id # Initialize CO's active ID
    orchestrator.mmu.clear_stm() # Clear STM for the very first session
    print(f"Starting new conversation: {current_conversation_id}")

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "/quit":
                print("Exiting chatbot. Goodbye!")
                # Clean up TinyDB MTM file lock if persistent MTM was used by closing the db
                if hasattr(mmu_instance, 'mtm') and mmu_instance.mtm.is_persistent and \
                   hasattr(mmu_instance.mtm, 'db') and mmu_instance.mtm.db:
                    print("Closing MTM database...")
                    mmu_instance.mtm.db.close()
                break
            
            elif user_input.lower() == "/new":
                current_conversation_id = str(uuid.uuid4())
                # The CO's handle_user_message will detect the new ID and clear STM
                # or we can explicitly call a method on CO if we add one for new sessions.
                # For now, CO's existing logic for changing conversation_id will clear STM.
                print(f"\nStarting new conversation: {current_conversation_id}")
                # We need to ensure the CO's active_conversation_id is updated BEFORE the next message
                # The handle_user_message already does this, and clears STM if it detects a change.
                # To be super explicit and ensure STM is cleared *before* first message of new convo:
                orchestrator.active_conversation_id = current_conversation_id # Update CO's tracker
                orchestrator.mmu.clear_stm() # Manually clear STM for /new command
                print("  STM cleared for new conversation.")
                continue

            elif user_input.lower() == "/reset_memory":
                confirm = input("WARNING: This will erase all long-term memory (facts, vector store, logs) and medium-term memory. "
                                "Short-term memory for the current session will also be cleared. "
                                "Are you sure? (yes/no): ").strip().lower()
                if confirm == "yes":
                    print("Resetting all memory...")
                    # We need a way to also clear MTM persistent store if used.
                    # MMU's reset_all_memory clears STM and LTM. MTM part needs to be handled.
                    
                    # Call MMU's full reset
                    reset_success = mmu_instance.reset_all_memory(confirm_reset=True) # MMU handles STM and LTM
                    
                    if reset_success:
                        print("All memory has been reset.")
                    else:
                        print("Memory reset encountered issues (LTM might be partially reset).")
                    
                    # Start a fresh conversation state after reset
                    current_conversation_id = str(uuid.uuid4())
                    orchestrator.active_conversation_id = current_conversation_id
                    orchestrator.mmu.clear_stm() # Ensure STM is clean
                    print(f"\nMemory reset. Starting new conversation: {current_conversation_id}")
                else:
                    print("Memory reset aborted.")
                continue

            # Process regular message through the orchestrator
            print("Bot: Thinking...") # Give some feedback
            assistant_response = orchestrator.handle_user_message(
                user_message=user_input,
                conversation_id=current_conversation_id
            )

            if assistant_response:
                print(f"Bot: {assistant_response}")
            else:
                # This case should ideally be handled by CO returning a fallback message.
                print("Bot: I'm sorry, I couldn't process that.")

        except KeyboardInterrupt:
            print("\nExiting chatbot due to KeyboardInterrupt. Goodbye!")
            if hasattr(mmu_instance, 'mtm') and mmu_instance.mtm.is_persistent and \
               hasattr(mmu_instance.mtm, 'db') and mmu_instance.mtm.db:
                print("Closing MTM database...")
                mmu_instance.mtm.db.close()
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            # Depending on the error, you might want to break or try to continue
            # For simplicity, we'll break on unexpected errors for now.
            # Consider logging such errors.
            if hasattr(mmu_instance, 'mtm') and mmu_instance.mtm.is_persistent and \
               hasattr(mmu_instance.mtm, 'db') and mmu_instance.mtm.db:
                print("Closing MTM database due to error...")
                mmu_instance.mtm.db.close()
            break


if __name__ == "__main__":
    run_cli()