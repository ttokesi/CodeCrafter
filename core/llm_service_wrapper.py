# offline_chat_bot/core/llm_service_wrapper.py

import ollama
import time # Keep if used, otherwise can remove

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    class SentenceTransformer: 
        def __init__(self, model_name_or_path): pass
        def encode(self, sentences, **kwargs): return [] 

from .config_loader import get_config # <--- ADD IMPORT

class LLMServiceWrapper:
    def __init__(self, config: dict = None): # <--- ADD optional config parameter
        """
        Initializes the LLMServiceWrapper using settings from the global configuration.
        """
        if config is None:
            # print("LSW: Loading global configuration...") # Optional debug
            config = get_config() # Load global config if specific one not provided
        # else:
            # print("LSW: Using provided configuration dictionary.") # Optional debug
        lsw_config = config.get('lsw', {}) # Get the 'lsw' section, or empty dict if not found
        
        self.ollama_host = lsw_config.get('ollama_host', "http://localhost:11434")
        self.default_chat_model = lsw_config.get('default_chat_model', "gemma3:1b-it-fp16") # Sensible fallback
        self.default_embedding_model_ollama = lsw_config.get('default_embedding_model_ollama', "nomic-embed-text")
        self.default_embedding_model_st = lsw_config.get('default_embedding_model_st', "all-MiniLM-L6-v2")
        
        print(f"Initializing LLMServiceWrapper with Ollama host: {self.ollama_host}")
        print(f"  Default chat model: {self.default_chat_model}")
        print(f"  Default Ollama embedding model: {self.default_embedding_model_ollama}")
        print(f"  Default ST embedding model: {self.default_embedding_model_st}")
        
        try:
            self.client = ollama.Client(host=self.ollama_host)
            print("  Ollama client initialized.")
        except Exception as e:
            print(f"  Error initializing Ollama client: {e}. Ensure Ollama service is running at {self.ollama_host}.")
            self.client = None

        self.st_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # The ST model name is now self.default_embedding_model_st
                print(f"  Attempting to load SentenceTransformer model: {self.default_embedding_model_st}")
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
            print("LSW_ERROR: Ollama client not initialized.") # Changed to LSW_ERROR
            return None if not stream else iter([]) # Return empty iter if stream expected
        
        model_to_use = model_name if model_name else self.default_chat_model
        options = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens # Ollama uses num_predict
        
        # Add any other kwargs to options
        options.update(kwargs.get('options', {})) # If options are passed nested
        for key, value in kwargs.items(): # For flat kwargs
            if key not in ['options', 'model', 'messages', 'stream', 'format', 'keep_alive']: # Avoid overriding core args
                 options[key] = value
        
        response_stream = None # Initialize to None
        try:
            print(f"LSW_DEBUG: About to call self.client.chat with model='{model_to_use}', stream={stream}, options={options}") # New
            response_stream = self.client.chat(
                model=model_to_use,
                messages=messages,
                stream=stream,
                options=options
            )
            print(f"LSW_DEBUG: self.client.chat call successful. Type of response_stream: {type(response_stream)}")
            if stream:
                print(f"LSW_DEBUG: response_stream (for stream=True) raw object: {str(response_stream)[:500]}")

        except Exception as client_chat_error: # Catch error from self.client.chat()
            print(f"LSW_CRITICAL_ERROR: Exception directly from self.client.chat() for model '{model_to_use}', stream={stream}.")
            print(f"  Error Type: {type(client_chat_error)}")
            print(f"  Error Args: {client_chat_error.args}")
            print(f"  Error String: {str(client_chat_error)}")
            import traceback
            traceback.print_exc()
            return None if not stream else iter([]) # Consistent error return

        # If self.client.chat() succeeded, proceed to define and return content_generator for streams
        if stream:
            if response_stream is None: # Should not happen if no exception, but a safeguard
                print("LSW_ERROR: response_stream is None after client.chat call for streaming, but no exception was caught.")
                return iter([])

            def content_generator():
                full_response_for_debug = "" 
                print("LSW_STREAM_DEBUG: content_generator started.")
                try:
                    for i, chunk in enumerate(response_stream): # chunk is ollama._types.ChatResponse
                        #print(f"LSW_STREAM_DEBUG: Raw chunk {i}: {chunk}")
                        #print(f"LSW_STREAM_DEBUG: Type of chunk {i}: {type(chunk)}")

                        # --- MODIFIED ACCESS TO CHUNK DATA ---
                        content_piece = ""
                        if hasattr(chunk, 'message') and chunk.message and hasattr(chunk.message, 'content'):
                            content_piece = chunk.message.content
                        # --- END MODIFIED ACCESS ---
                        
                        if content_piece: # Only yield if there's actual content
                            #print(f"LSW_STREAM_DEBUG: Yielding content_piece: '{content_piece[:50]}...'")
                            yield content_piece
                            full_response_for_debug += content_piece
                        
                        # Access 'done' attribute
                        done_flag = False
                        if hasattr(chunk, 'done'):
                            done_flag = chunk.done
                        
                        if done_flag is True:
                            print(f"LSW_STREAM_DEBUG: Done flag is True for chunk {i}.")
                            # If the 'done' chunk also had content, it was yielded above.
                            # If it's a final 'done' chunk with no new content, just break.
                            if not content_piece:
                                 print(f"LSW_STREAM_DEBUG: Final 'done' chunk {i} had no new content. Stopping.")
                            break 
                    print(f"LSW_STREAM_DEBUG: content_generator loop finished. Full response: {full_response_for_debug[:200]}...")
                except Exception as stream_gen_ex:
                    print(f"LSW_STREAM_ERROR: Error within LSW's content_generator stream: Type {type(stream_gen_ex)}, {stream_gen_ex}")
                    import traceback
                    traceback.print_exc()
            return content_generator()
        
        else: # Non-streaming (this part seems to work fine)
            if response_stream is None: # Should not happen if no exception
                 print("LSW_ERROR: response_stream is None after client.chat call for non-streaming, but no exception was caught.")
                 return None
            full_response_content = response_stream['message']['content']
            return full_response_content

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
                encoded_output = self.st_model.encode(text_to_embed, convert_to_tensor=False)
                if hasattr(encoded_output, 'tolist'): # Check if it has tolist() method (like numpy arrays)
                    embedding = encoded_output.tolist()
                elif isinstance(encoded_output, list): # Check if it's already a list
                    embedding = encoded_output
                else:
                    # If it's neither, this is unexpected, or we need to handle other types
                    print(f"Warning: SentenceTransformer encode() returned an unexpected type: {type(encoded_output)}")
                    # Attempt to convert to list if it's some other iterable, or handle error
                    try:
                        embedding = list(encoded_output) # Fallback attempt
                        if not embedding or not isinstance(embedding[0], (float, int)): # Basic check
                            raise TypeError("Fallback list conversion did not yield list of numbers.")
                    except TypeError as te:
                        print(f"Error: Could not convert ST output to list of floats. {te}")
                        return None
                return embedding
            except Exception as e:
                print(f"Error generating SentenceTransformer embedding: {e}")
                return None
        else:
            print(f"Error: Unknown embedding source '{source}'. Choose 'ollama' or 'st'.")
            return None

# In offline_chat_bot/core/llm_service_wrapper.py
if __name__ == "__main__":
    print("--- Testing LLMServiceWrapper (using values from config.yaml) ---")

    # Ensure config.yaml exists and Ollama is running with configured default models
    # Models needed for test:
    # - lsw.default_chat_model (e.g., gemma3:4b-it-fp16 from config)
    # - lsw.default_embedding_model_ollama (e.g., nomic-embed-text from config)
    # - lsw.default_embedding_model_st (e.g., all-MiniLM-L6-v2 from config)
    
    # The conditional import for config_loader should handle sys.path if needed for direct run
    # (assuming config_loader.py is in the same 'core' directory or path is set up)
    # If config_loader itself has issues being found when running this file directly,
    # the get_config() call in LSW.__init__ will fail.
    # Let's add the sys.path modification here too for robustness of direct execution of this file.
    if __package__ is None: # Running file directly
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # Now from .config_loader import get_config should work, or it's already imported globally

    lsw = LLMServiceWrapper() # Now uses config internally

    if not lsw.client:
        print("Ollama client failed to initialize. Exiting LSW tests.")
    else:
        print("\n--- Listing Local Ollama Models ---")
        local_models = lsw.list_local_models()
        # ... (rest of the model listing print logic, no changes needed here) ...
        if local_models:
            print(f"Found {len(local_models)} models:")
            for model_obj in local_models[:3]: 
                try:
                    model_tag = model_obj.model 
                    size_gb = model_obj.size / (1024**3)
                    modified_at = model_obj.modified_at
                    print(f"  - Model Tag: {model_tag}, Size: {size_gb:.2f} GB, Modified: {modified_at}")
                except AttributeError as e: print(f"  - Error accessing attributes for a model: {e}. Raw model info: {model_obj}")
                except Exception as e: print(f"  - Unexpected error processing a model entry: {e}. Raw model info: {model_obj}")
        else: print("No local Ollama models found or error listing them.")


        print("\n--- Testing Chat Completion (Non-Streamed) ---")
        chat_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
        # Uses lsw.default_chat_model from config
        chat_response = lsw.generate_chat_completion(messages=chat_messages, temperature=0.1) 
        if chat_response: print(f"Chatbot Response: {chat_response}")
        else: print("Chat completion failed.")

        print("\n--- Testing Chat Completion (Streamed) ---")
        # Uses lsw.default_chat_model from config
        streamed_response_iter = lsw.generate_chat_completion(
            messages=[{"role": "user", "content": "Tell me a very short story about a robot."}],
            stream=True, max_tokens=50
        )
        # ... (rest of streaming print logic, no changes needed here) ...
        if streamed_response_iter:
            print("Streamed Story: ", end='')
            full_story = ""
            try:
                for chunk in streamed_response_iter:
                    print(chunk, end='', flush=True)
                    full_story += chunk
                print("\n(End of stream)")
            except Exception as e:
                print(f"\nError during stream consumption: {e}")
        else: print("Streamed chat completion failed.")


        print("\n--- Testing Ollama Embeddings ---")
        # Uses lsw.default_embedding_model_ollama from config
        ollama_emb = lsw.generate_embedding("Hello from Ollama embeddings!", source="ollama")
        # ... (rest of Ollama embedding print logic) ...
        if ollama_emb: print(f"Ollama Embedding (first 5 dims): {ollama_emb[:5]}... (Length: {len(ollama_emb)})")
        else: print("Ollama embedding generation failed.")

    # Test SentenceTransformer embeddings separately
    if lsw.st_model: 
        print("\n--- Testing SentenceTransformer Embeddings ---")
        # Uses lsw.default_embedding_model_st from config
        st_emb = lsw.generate_embedding("Hello from SentenceTransformer!", source="st")
        # ... (rest of ST embedding print logic) ...
        if st_emb: print(f"SentenceTransformer Embedding (first 5 dims): {st_emb[:5]}... (Length: {len(st_emb)})")
        else: print("SentenceTransformer embedding generation failed.")
    else:
        print("\nSkipping SentenceTransformer Embeddings test as model did not load.")

    print("\nLLMServiceWrapper tests finished.")