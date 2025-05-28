# offline_chat_bot/core/tokenizer_utils.py

import ollama # Still useful for LSW, not directly for token counting here

# --- Hugging Face Tokenizer Logic ---
_tokenizer_cache_hf = {}

# --- THIS IS THE KEY CHANGE: Set default to your target model's tokenizer ---
# For Ollama model "gemma3:4b-it-fp16", the corresponding HF tokenizer is "google/gemma-3-4b-it"
DEFAULT_HF_TOKENIZER_MODEL = "google/gemma-3-4b-it" 

try:
    from tokenizers import Tokenizer
    HF_TOKENIZERS_LIB_AVAILABLE = True
except ImportError:
    print("TokenizerUtils: Hugging Face 'tokenizers' library not found. Please run: `pip install tokenizers`")
    HF_TOKENIZERS_LIB_AVAILABLE = False
    # Define a dummy Tokenizer class if the library is not available, so type hints don't break
    # and the rest of the code can have a placeholder.
    class Tokenizer:
        @staticmethod
        def from_pretrained(name_or_path):
            print(f"TokenizerUtils: Dummy Tokenizer.from_pretrained called for {name_or_path} because 'tokenizers' lib is missing.")
            return None # Indicate failure to load

        def encode(self, text):
            # This is a dummy encode, should not be relied upon if Tokenizer is None
            print("TokenizerUtils: Dummy Tokenizer.encode called.")
            # Return a dummy Encoding object that has an 'ids' attribute
            class DummyEncoding:
                def __init__(self):
                    self.ids = []
            return DummyEncoding()


def get_hf_tokenizer(tokenizer_name_or_path: str = DEFAULT_HF_TOKENIZER_MODEL) -> Tokenizer or None:
    """
    Loads a Hugging Face tokenizer from its name or path.
    Caches loaded tokenizers.

    Args:
        tokenizer_name_or_path (str): The Hugging Face model name/path for the tokenizer.

    Returns:
        Tokenizer or None: The loaded tokenizer instance, or None if loading fails.
    """
    if not HF_TOKENIZERS_LIB_AVAILABLE:
        print(f"TokenizerUtils: Cannot load HF tokenizer '{tokenizer_name_or_path}' because 'tokenizers' library is not installed.")
        return None
        
    if tokenizer_name_or_path in _tokenizer_cache_hf:
        return _tokenizer_cache_hf[tokenizer_name_or_path]

    try:
        print(f"TokenizerUtils: Attempting to load HF tokenizer '{tokenizer_name_or_path}' from Hugging Face Hub...")
        # This will download tokenizer files on first use for a given name.
        # It MIGHT require the 'transformers' library to be installed as well,
        # if the tokenizer_config.json for this model specifies custom Python classes
        # or relies on AutoTokenizer logic from the transformers library.
        tokenizer = Tokenizer.from_pretrained(tokenizer_name_or_path)
        _tokenizer_cache_hf[tokenizer_name_or_path] = tokenizer
        print(f"TokenizerUtils: HF Tokenizer '{tokenizer_name_or_path}' loaded and cached successfully.")
        return tokenizer
    except Exception as e:
        print(f"TokenizerUtils: Error loading HF tokenizer '{tokenizer_name_or_path}': {e}")
        print("  Recommendations:")
        print("  1. Ensure the tokenizer name is correct and you have an active internet connection for the first download.")
        print("  2. Try installing/upgrading the 'transformers' library: `pip install --upgrade transformers`")
        print("  3. Check the Hugging Face Hub page for the tokenizer for any specific loading instructions.")
        return None

def count_tokens_hf(text: str, hf_tokenizer_name: str = DEFAULT_HF_TOKENIZER_MODEL) -> int or None:
    """
    Counts tokens using a specified Hugging Face tokenizer.

    Args:
        text (str): The text to tokenize.
        hf_tokenizer_name (str): The Hugging Face model name/path for the tokenizer.

    Returns:
        int or None: The number of tokens, or None if tokenization fails.
    """
    if not text: # Treat empty string as 0 tokens for practical purposes
        return 0
        
    tokenizer = get_hf_tokenizer(hf_tokenizer_name)
    if tokenizer:
        try:
            encoded = tokenizer.encode(text) # text is sequence, not text.encode()
            return len(encoded.ids)
        except Exception as e:
            print(f"TokenizerUtils: Error encoding text with HF tokenizer '{hf_tokenizer_name}': {e}")
            return None # Indicate failure to count with this specific tokenizer
    return None # Tokenizer itself could not be loaded


# --- Primary function to be used by the CO ---
def count_tokens(text: str, 
                 ollama_model_name: str, # Used to help select the appropriate HF tokenizer
                 hf_tokenizer_name_override: str = None 
                ) -> int:
    """
    Counts tokens, primarily using a Hugging Face tokenizer mapped from the Ollama model name.
    Falls back to word count if HF tokenization fails.

    Args:
        text (str): The text to count tokens for.
        ollama_model_name (str): The Ollama model tag (e.g., "gemma3:4b-it-fp16").
                                 Used to select an appropriate HF tokenizer.
        hf_tokenizer_name_override (str, optional): Explicitly specify an HF tokenizer name,
                                                    bypassing the mapping from ollama_model_name.
    Returns:
        int: The number of tokens. Defaults to a rough word count if all methods fail.
    """
    if not text:
        return 0

    hf_tokenizer_to_use = DEFAULT_HF_TOKENIZER_MODEL # Fallback default

    if hf_tokenizer_name_override:
        hf_tokenizer_to_use = hf_tokenizer_name_override
    else:
        # Simple mapping from Ollama model name to likely HF tokenizer
        # This can be expanded and made more robust.
        # Prioritize direct match for your primary model
        if "gemma-3-4b-it" in ollama_model_name.lower() or "gemma3:4b-it" in ollama_model_name.lower(): # Check against common Ollama tag format
            hf_tokenizer_to_use = "google/gemma-3-4b-it"
        elif "gemma" in ollama_model_name.lower(): # General gemma
             hf_tokenizer_to_use = "google/gemma-2b" # Or another general Gemma tokenizer as fallback
        elif "llama" in ollama_model_name.lower():
            # For Llama models, e.g., "meta-llama/Llama-2-7b-hf" or "meta-llama/Llama-3-8B"
            # This part needs adjustment based on specific Llama versions you might use.
            if "llama-3" in ollama_model_name.lower() or "llama3" in ollama_model_name.lower() :
                 hf_tokenizer_to_use = "meta-llama/Llama-3-8B" # Llama 3 tokenizer
            elif "llama-2" in ollama_model_name.lower() or "llama2" in ollama_model_name.lower():
                 hf_tokenizer_to_use = "meta-llama/Llama-2-7b-hf" # Llama 2 tokenizer
            else:
                hf_tokenizer_to_use = "hf-internal-testing/llama-tokenizer" 
                print(f"TokenizerUtils: Detected Llama model ('{ollama_model_name}'), attempting generic Llama tokenizer. For specific versions, update mapping.")
        elif "mistral" in ollama_model_name.lower():
            hf_tokenizer_to_use = "mistralai/Mistral-7B-v0.1" # Example
        # Add more mappings as needed for other model families you use

    # print(f"TokenizerUtils: Attempting token count for Ollama model '{ollama_model_name}' using mapped HF tokenizer '{hf_tokenizer_to_use}'.")
    hf_token_count = count_tokens_hf(text, hf_tokenizer_name=hf_tokenizer_to_use)
    
    if hf_token_count is not None:
        return hf_token_count
    
    # Final fallback: rough word count
    print(f"TokenizerUtils: HF token counting failed for Ollama model '{ollama_model_name}' (mapped to HF '{hf_tokenizer_to_use}'). Using rough word count as fallback.")
    return len(text.split())


# --- Test Block for tokenizer_utils (Focusing on HF tokenizers) ---
if __name__ == "__main__":
    print("--- Testing Tokenizer Utilities (Hugging Face Focus) ---")

    # Ensure you have `pip install tokenizers transformers`
    # And internet access for the first download of tokenizer files.

    # Test with your primary target model's tokenizer
    your_ollama_model_tag = "gemma3:4b-it-fp16" 
    # This should map to "google/gemma-3-4b-it" in the count_tokens function.

    print(f"\nTesting for Ollama model: {your_ollama_model_tag} (will use best guess HF tokenizer)")
    text1 = "Hello, world! This is a test."
    count1 = count_tokens(text1, ollama_model_name=your_ollama_model_tag)
    print(f"Text: \"{text1}\" | Approx Words: {len(text1.split())} | Tokens (HF for {your_ollama_model_tag}): {count1}")

    text2 = "Large Language Models are powerful."
    count2 = count_tokens(text2, ollama_model_name=your_ollama_model_tag)
    print(f"Text: \"{text2}\" | Approx Words: {len(text2.split())} | Tokens (HF for {your_ollama_model_tag}): {count2}")

    text3_long = "Tokenization is the process of breaking down a sequence of text into smaller pieces called tokens. " \
                 "These tokens can then be mapped to numerical vectors that machine learning models can process. " \
                 "It's a fundamental step in Natural Language Processing."
    count3 = count_tokens(text3_long, ollama_model_name=your_ollama_model_tag)
    print(f"Text: \"{text3_long[:50]}...\" | Approx Words: {len(text3_long.split())} | Tokens (HF for {your_ollama_model_tag}): {count3}")

    # Test caching (of HF tokenizer)
    print("\nTesting HF tokenizer caching (second call for same tokenizer should be faster and use cache):")
    # The ollama_model_name will map to the same HF tokenizer, so it should be cached
    count1_again = count_tokens(text1, ollama_model_name=your_ollama_model_tag) 
    print(f"Text: \"{text1}\" | Tokens (HF for {your_ollama_model_tag}, cached): {count1_again}")
    
    # Test with an empty string
    count_empty = count_tokens("", ollama_model_name=your_ollama_model_tag)
    print(f"Text: \"\" | Tokens (HF for {your_ollama_model_tag}): {count_empty}")

    # Test with an explicit HF tokenizer override that might be different
    print("\nTesting with explicit HF tokenizer override (e.g., a generic gemma-2b tokenizer):")
    gemma2b_tokenizer = "google/gemma-2b"
    count_gemma2b = count_tokens("Test with explicit Gemma 2b tokenizer.", 
                              ollama_model_name="gemma3:4b-it-fp16", # This will be overridden
                              hf_tokenizer_name_override=gemma2b_tokenizer)
    print(f"Text: \"Test with explicit Gemma 2b tokenizer.\" | Tokens (HF {gemma2b_tokenizer}): {count_gemma2b}")
    
    print("\nTesting with a non-existent HF tokenizer (to see fallback):")
    count_fail_hf = count_tokens("Test fallback.", 
                                 ollama_model_name="any-model", 
                                 hf_tokenizer_name_override="this-hf-tokenizer-is-fake-123")
    print(f"Text: \"Test fallback.\" | Tokens (failed HF, fallback to words): {count_fail_hf}")


    print("\nTokenizer utility tests finished.")