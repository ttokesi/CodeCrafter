# offline_chat_bot/core/llm_service_wrapper.py

import ollama
import time

# Optional: For using sentence-transformers as an alternative embedding source
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    # Define a dummy class if not available so type hints don't break
    class SentenceTransformer: 
        def __init__(self, model_name_or_path): pass
        def encode(self, sentences, **kwargs): return [] 

DEFAULT_OLLAMA_HOST = "http://localhost:11434" # Default Ollama API host

# Default model names - these should be models you have pulled into Ollama
DEFAULT_CHAT_MODEL = "gemma3:1b-it-fp16" # A smaller, faster model for general chat
DEFAULT_EMBEDDING_MODEL_OLLAMA = "nomic-embed-text" # Ollama's recommended default
DEFAULT_EMBEDDING_MODEL_ST = "all-MiniLM-L6-v2" # A good default SentenceTransformer

class LLMServiceWrapper:
    """
    Wraps interactions with the Ollama LLM service and local SentenceTransformer models
    for text generation and embedding.
    """
    def __init__(self, ollama_host: str = DEFAULT_OLLAMA_HOST, 
                 default_chat_model: str = DEFAULT_CHAT_MODEL,
                 default_embedding_model_ollama: str = DEFAULT_EMBEDDING_MODEL_OLLAMA,
                 default_embedding_model_st: str = DEFAULT_EMBEDDING_MODEL_ST):
        """
        Initializes the LLMServiceWrapper.

        Args:
            ollama_host (str): The host address of the Ollama API.
            default_chat_model (str): Default Ollama model to use for chat.
            default_embedding_model_ollama (str): Default Ollama model for embeddings.
            default_embedding_model_st (str): Default SentenceTransformer model for embeddings.
        """
        print(f"Initializing LLMServiceWrapper with Ollama host: {ollama_host}")
        self.ollama_host = ollama_host
        self.default_chat_model = default_chat_model
        self.default_embedding_model_ollama = default_embedding_model_ollama
        self.default_embedding_model_st = default_embedding_model_st
        
        try:
            self.client = ollama.Client(host=self.ollama_host)
            # Test connection by listing models (can be slow if many models)
            # self.list_local_models() # Optional: test connection on init
            print("  Ollama client initialized.")
        except Exception as e:
            print(f"  Error initializing Ollama client: {e}. Ensure Ollama service is running at {ollama_host}.")
            self.client = None # Mark client as unusable
            # raise # Optionally re-raise to halt if Ollama is critical at startup

        # Initialize SentenceTransformer model for embeddings if available and needed
        self.st_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                print(f"  Attempting to load SentenceTransformer model: {self.default_embedding_model_st}")
                # This might download the model on first use if not cached
                self.st_model = SentenceTransformer(self.default_embedding_model_st)
                print(f"  SentenceTransformer model '{self.default_embedding_model_st}' loaded successfully.")
            except Exception as e:
                print(f"  Warning: Could not load SentenceTransformer model '{self.default_embedding_model_st}'. Error: {e}")
                self.st_model = None
        else:
            print("  SentenceTransformer library not available. Local ST embeddings disabled.")


    def list_local_models(self) -> list:
        """
        Lists all models available locally in the Ollama service.
        Returns:
            list: A list of model details dictionaries, or an empty list if error.
        """
        if not self.client:
            print("Error: Ollama client not initialized.")
            return []
        try:
            models_info = self.client.list()
            return models_info.get('models', []) # The actual list is under the 'models' key
        except Exception as e:
            print(f"Error listing Ollama models: {e}")
            return []

    def generate_chat_completion(self,
                                 messages: list,
                                 model_name: str = None,
                                 temperature: float = 0.7,
                                 max_tokens: int = None, # Ollama's 'num_predict'
                                 stream: bool = False,
                                 **kwargs) -> str or iter: # Returns str or iterator if streaming
        """
        Generates a chat completion response from Ollama.

        Args:
            messages (list): A list of message dictionaries, e.g.,
                             [{"role": "user", "content": "Hello"}].
            model_name (str, optional): The Ollama model to use. Defaults to self.default_chat_model.
            temperature (float, optional): Controls randomness. Lower is more deterministic.
            max_tokens (int, optional): Max tokens to generate. Maps to Ollama's 'num_predict'.
            stream (bool, optional): Whether to stream the response.
            **kwargs: Additional parameters to pass to ollama.chat API (e.g., top_p, top_k).
                      See ollama.Client.chat documentation or Ollama REST API docs.

        Returns:
            str or iter: The assistant's response content string if stream=False,
                         or an iterator of response chunks if stream=True.
                         Returns None if an error occurs.
        """
        if not self.client:
            print("Error: Ollama client not initialized.")
            return None
        
        model_to_use = model_name if model_name else self.default_chat_model
        options = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens # Ollama uses num_predict
        
        # Add any other kwargs to options
        options.update(kwargs.get('options', {})) # If options are passed nested
        for key, value in kwargs.items(): # For flat kwargs
            if key not in ['options', 'model', 'messages', 'stream', 'format', 'keep_alive']: # Avoid overriding core args
                 options[key] = value

        try:
            response_stream = self.client.chat( # Renamed to response_stream for clarity
                model=model_to_use,
                messages=messages,
                stream=stream, # This will be True for streaming calls
                options=options
            )

            if stream:
                # print("LSW: Streaming response...") # Optional debug
                def content_generator():
                    full_response_for_debug = "" # For debugging if needed
                    for chunk in response_stream:
                        # The structure of a chunk from ollama.Client.chat (stream=True) is like:
                        # {'model': 'gemma3:1b-it-fp16', 'created_at': '...', 'message': {'role': 'assistant', 'content': 'some text'}, 'done': False/True}
                        # Sometimes, 'content' might be missing or empty in intermediate chunks if only metadata changes.
                        # Or the very last chunk might have 'done': True and empty 'content'.
                        
                        message_part = chunk.get('message', {})
                        content_piece = message_part.get('content', '')
                        
                        if content_piece: # Only yield if there's actual text content
                            # full_response_for_debug += content_piece # Optional: accumulate for logging
                            yield content_piece
                        
                        # Check if this is the final chunk and if there was no content in it
                        if chunk.get('done') is True and not content_piece:
                            # print(f"LSW: Stream finished. Full response for debug: {full_response_for_debug}") # Optional
                            break # Stop generation
                return content_generator() # Return the generator
            else: # Non-streaming
                # When stream=False, response_stream is actually the complete response dictionary
                full_response_content = response_stream['message']['content']
                return full_response_content
        except Exception as e:
            print(f"Error during Ollama chat completion with model '{model_to_use}': {e}")
            if stream: return iter([]) # Return an empty iterator on error for stream
            return None # Return None on error for non-stream

    def generate_embedding(self,
                           text_to_embed: str,
                           source: str = "ollama", # "ollama" or "st" (SentenceTransformer)
                           model_name: str = None) -> list[float] or None:
        """
        Generates an embedding for the given text.

        Args:
            text_to_embed (str): The text to embed.
            source (str, optional): "ollama" to use an Ollama embedding model,
                                    "st" to use the local SentenceTransformer model.
                                    Defaults to "ollama".
            model_name (str, optional): The specific model to use.
                                       If source="ollama", defaults to self.default_embedding_model_ollama.
                                       If source="st", defaults to self.default_embedding_model_st (which is loaded at init).

        Returns:
            list[float] or None: The embedding vector, or None if an error occurs.
        """
        if source == "ollama":
            if not self.client:
                print("Error: Ollama client not initialized for embeddings.")
                return None
            
            model_to_use = model_name if model_name else self.default_embedding_model_ollama
            try:
                # print(f"LSW: Requesting Ollama embedding for model '{model_to_use}'")
                response = self.client.embeddings(
                    model=model_to_use,
                    prompt=text_to_embed
                )
                return response.get('embedding') # The vector is under the 'embedding' key
            except Exception as e:
                print(f"Error generating Ollama embedding with model '{model_to_use}': {e}")
                return None
        elif source == "st":
            if not self.st_model:
                print("Error: SentenceTransformer model not loaded for embeddings.")
                return None
            # If a different ST model is requested via model_name, we'd need to load it here.
            # For simplicity, this LSW currently only uses the one loaded at __init__.
            if model_name and model_name != self.default_embedding_model_st:
                print(f"Warning: LSW currently only supports the pre-loaded ST model ('{self.default_embedding_model_st}'). "
                      f"Ignoring requested ST model '{model_name}'.")

            try:
                # print(f"LSW: Generating SentenceTransformer embedding with model '{self.default_embedding_model_st}'")
                # .tolist() converts numpy array to Python list
                embedding = self.st_model.encode(text_to_embed, convert_to_tensor=False).tolist()
                return embedding
            except Exception as e:
                print(f"Error generating SentenceTransformer embedding: {e}")
                return None
        else:
            print(f"Error: Unknown embedding source '{source}'. Choose 'ollama' or 'st'.")
            return None

# --- Test Block ---
if __name__ == "__main__":
    print("--- Testing LLMServiceWrapper ---")

    # Ensure Ollama is running and you have pulled some models, e.g.:
    # ollama pull gemma:2b
    # ollama pull nomic-embed-text
    # ollama pull all-minilm (this might pull all-minilm-l6-v2)
    
    # You might need to adjust model names based on what you have locally.
    # For ST, the first run will download the model if not cached.

    lsw = LLMServiceWrapper(
        default_chat_model="gemma3:1b-it-fp16", # Make sure you have this model in Ollama
        default_embedding_model_ollama="nomic-embed-text", # Or another Ollama embed model
        default_embedding_model_st="all-MiniLM-L6-v2" # A common ST model
    )

    if not lsw.client:
        print("Ollama client failed to initialize. Exiting LSW tests.")
    else:
        print("\n--- Listing Local Ollama Models ---")
        local_models = lsw.list_local_models()
        if local_models:
            print(f"Found {len(local_models)} models:")
            for model_obj in local_models[:3]: # Rename to model_obj to emphasize it's an object
                try:
                    # Accessing as attributes
                    model_tag = model_obj.model
                    size_gb = model_obj.size / (1024**3)
                    modified_at = model_obj.modified_at
                    print(f"  - Model Tag: {model_tag}, Size: {size_gb:.2f} GB, Modified: {modified_at}")
                except AttributeError as e:
                    print(f"  - Error accessing attributes for a model: {e}. Raw model info: {model_obj}")
                except Exception as e: # Catch any other unexpected errors for a model entry
                    print(f"  - Unexpected error processing a model entry: {e}. Raw model info: {model_obj}")
        else:
            print("No local Ollama models found or error listing them.")

        print("\n--- Testing Chat Completion (Non-Streamed) ---")
        chat_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
        # Using the default_chat_model specified in LSW init
        chat_response = lsw.generate_chat_completion(messages=chat_messages, temperature=0.1) 
        if chat_response:
            print(f"Chatbot Response: {chat_response}")
        else:
            print("Chat completion failed.")

        print("\n--- Testing Chat Completion (Streamed) ---")
        # Using a different model if available and desired for testing
        # stream_model = "mistral:latest" # Example, ensure you have it
        # print(f"Attempting streamed response from {lsw.default_chat_model}:")
        streamed_response_iter = lsw.generate_chat_completion(
            messages=[{"role": "user", "content": "Tell me a very short story about a robot."}],
            model_name=lsw.default_chat_model, # Explicitly using default
            stream=True,
            max_tokens=50 # Example option
        )
        if streamed_response_iter:
            print("Streamed Story: ", end='')
            full_story = ""
            for chunk in streamed_response_iter:
                print(chunk, end='', flush=True)
                full_story += chunk
            print("\n(End of stream)")
            # print(f"Full streamed story collected: {full_story}")
        else:
            print("Streamed chat completion failed.")

        print("\n--- Testing Ollama Embeddings ---")
        ollama_emb = lsw.generate_embedding("Hello from Ollama embeddings!", source="ollama")
        if ollama_emb:
            print(f"Ollama Embedding (first 5 dims): {ollama_emb[:5]}... (Length: {len(ollama_emb)})")
        else:
            print("Ollama embedding generation failed.")

    # Test SentenceTransformer embeddings separately, as Ollama client isn't needed
    if lsw.st_model: # Check if ST model was loaded
        print("\n--- Testing SentenceTransformer Embeddings ---")
        st_emb = lsw.generate_embedding("Hello from SentenceTransformer!", source="st")
        if st_emb:
            print(f"SentenceTransformer Embedding (first 5 dims): {st_emb[:5]}... (Length: {len(st_emb)})")
        else:
            print("SentenceTransformer embedding generation failed.")
    else:
        print("\nSkipping SentenceTransformer Embeddings test as model did not load.")

    print("\nLLMServiceWrapper tests finished.")