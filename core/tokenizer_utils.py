# offline_chat_bot/core/tokenizer_utils.py

import ollama # Still useful for LSW, not directly for token counting here
from .config_loader import get_config

try:
    from tokenizers import Tokenizer
    HF_TOKENIZERS_LIB_AVAILABLE = True
except ImportError:
    print("TokenizerUtils: Hugging Face 'tokenizers' library not found. Please run: `pip install tokenizers`")
    HF_TOKENIZERS_LIB_AVAILABLE = False
    # Define a dummy Tokenizer class if the library is not available, so type hints don't break
    # and the rest of the code can have a placeholder.
    class Tokenizer: # Dummy
        @staticmethod
        def from_pretrained(name_or_path): return None
        def encode(self, text): return type('obj', (object,), {'ids': []})()

if __name__ == '__main__' and __package__ is None:
    import sys
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from core.config_loader import get_config # Absolute-like path after sys.path mod
else:
    from .config_loader import get_config # Relative path for package use

_tokenizer_cache_hf = {}

def get_hf_tokenizer(tokenizer_name_or_path: str = None) -> Tokenizer or None:
    if not HF_TOKENIZERS_LIB_AVAILABLE:
        print("TokenizerUtils: Cannot load HF tokenizer because 'tokenizers' library is not installed.")
        return None
    
    cfg = get_config()
    # Use provided name, or fallback to config default, or finally a hardcoded absolute fallback
    effective_tokenizer_name = tokenizer_name_or_path if tokenizer_name_or_path \
                               else cfg.get('tokenizer', {}).get('default_hf_tokenizer', "google/gemma-2b") # Last resort fallback

    if effective_tokenizer_name in _tokenizer_cache_hf:
        return _tokenizer_cache_hf[effective_tokenizer_name]

    try:
        #print(f"TokenizerUtils: Attempting to load HF tokenizer '{effective_tokenizer_name}' from Hugging Face Hub...")
        tokenizer = Tokenizer.from_pretrained(effective_tokenizer_name)
        _tokenizer_cache_hf[effective_tokenizer_name] = tokenizer
        #print(f"TokenizerUtils: HF Tokenizer '{effective_tokenizer_name}' loaded and cached successfully.")
        return tokenizer
    except Exception as e:
        print(f"TokenizerUtils: Error loading HF tokenizer '{effective_tokenizer_name}': {e}")
        print("  Recommendations:")
        print("  1. Ensure the tokenizer name is correct and you have an active internet connection for the first download.")
        print("  2. Try installing/upgrading the 'transformers' library: `pip install --upgrade transformers`")
        print("  3. Check the Hugging Face Hub page for the tokenizer for any specific loading instructions.")
        return None

def count_tokens_hf(text: str, hf_tokenizer_name: str = None) -> int or None: # Allow None for default
    if not text: return 0
    
    # If hf_tokenizer_name is not provided, get_hf_tokenizer will use its own default (from config)
    tokenizer = get_hf_tokenizer(hf_tokenizer_name) # Pass along None if that's what we got
    
    if tokenizer:
        try:
            encoded = tokenizer.encode(text)
            return len(encoded.ids)
        except Exception as e:
            print(f"TokenizerUtils: Error encoding text with HF tokenizer '{hf_tokenizer_name if hf_tokenizer_name else 'default'}': {e}")
            return None 
    return None

# Inside core/tokenizer_utils.py
def count_tokens(text: str, 
                 ollama_model_name: str, 
                 hf_tokenizer_name_override: str = None 
                ) -> int:
    if not text: return 0

    cfg = get_config()
    tokenizer_config = cfg.get('tokenizer', {})
    hf_tokenizer_map_from_config = tokenizer_config.get('hf_tokenizer_map', {})
    overall_default_hf_tokenizer = tokenizer_config.get('default_hf_tokenizer', "google/gemma-3-4b-it")

    hf_tokenizer_to_use = overall_default_hf_tokenizer # Start with this

    if hf_tokenizer_name_override:
        hf_tokenizer_to_use = hf_tokenizer_name_override
        # print(f"TokenizerUtils: Using HF tokenizer override: '{hf_tokenizer_to_use}'")
    else:
        # Normalize ollama_model_name for map lookup
        normalized_ollama_model_name = ollama_model_name.lower()
        
        # Check for an exact match in the map first (case-insensitive key comparison)
        exact_match_tokenizer = None
        for map_key, hf_name in hf_tokenizer_map_from_config.items():
            if map_key.lower() == normalized_ollama_model_name:
                exact_match_tokenizer = hf_name
                break
        
        if exact_match_tokenizer:
            hf_tokenizer_to_use = exact_match_tokenizer
            # print(f"TokenizerUtils: Found direct map for '{ollama_model_name}' -> HF Tokenizer: '{hf_tokenizer_to_use}'")
        else: 
            # Fallback for model families if no exact match from the map
            if "gemma" in normalized_ollama_model_name:
                hf_tokenizer_to_use = hf_tokenizer_map_from_config.get("default_gemma", overall_default_hf_tokenizer)
            elif "llama" in normalized_ollama_model_name:
                hf_tokenizer_to_use = hf_tokenizer_map_from_config.get("default_llama", overall_default_hf_tokenizer)
            elif "mistral" in normalized_ollama_model_name:
                hf_tokenizer_to_use = hf_tokenizer_map_from_config.get("default_mistral", overall_default_hf_tokenizer)
            # print(f"TokenizerUtils: No direct map for '{ollama_model_name}'. Using family/default HF Tokenizer: '{hf_tokenizer_to_use}'")
    
    # print(f"TokenizerUtils: Final selected HF tokenizer: '{hf_tokenizer_to_use}' for Ollama model '{ollama_model_name}'.")
    hf_token_count = count_tokens_hf(text, hf_tokenizer_name=hf_tokenizer_to_use)
    
    if hf_token_count is not None:
        return hf_token_count
    
    print(f"TokenizerUtils: HF token counting failed for Ollama model '{ollama_model_name}' (using HF '{hf_tokenizer_to_use}'). Using rough word count as fallback.")
    return len(text.split())


# In offline_chat_bot/core/tokenizer_utils.py
if __name__ == "__main__":
    print("--- Testing Tokenizer Utilities (Hugging Face Focus, from config) ---")

    # Ensure config.yaml is present in the project root for this test to run correctly.
    # The functions will call get_config() internally.
    
    # Test with your primary target model's name (should map via config)
    your_ollama_model_tag_from_config = "gemma3:4b-it-fp16" 

    print(f"\nTesting for Ollama model: {your_ollama_model_tag_from_config} (will use HF tokenizer from config map)")
    text1 = "Hello, world! This is a test."
    # The count_tokens function will look up "gemma3:4b-it-fp16" in config's hf_tokenizer_map
    count1 = count_tokens(text1, ollama_model_name=your_ollama_model_tag_from_config)
    print(f"Text: \"{text1}\" | Approx Words: {len(text1.split())} | Tokens (for {your_ollama_model_tag_from_config}): {count1}")

    text2 = "Large Language Models are powerful."
    count2 = count_tokens(text2, ollama_model_name=your_ollama_model_tag_from_config)
    print(f"Text: \"{text2}\" | Approx Words: {len(text2.split())} | Tokens (for {your_ollama_model_tag_from_config}): {count2}")

    # Test caching (of HF tokenizer)
    print("\nTesting HF tokenizer caching:")
    count1_again = count_tokens(text1, ollama_model_name=your_ollama_model_tag_from_config) 
    print(f"Text: \"{text1}\" | Tokens (for {your_ollama_model_tag_from_config}, cached): {count1_again}")
    
    # Test with an explicit HF tokenizer override
    print("\nTesting with explicit HF tokenizer override (e.g., a generic gemma-2b tokenizer):")
    gemma2b_tokenizer_from_config_map_or_default = "google/gemma-2b" # As an example
    count_gemma2b = count_tokens("Test with explicit Gemma 2b tokenizer.", 
                              ollama_model_name="any-ollama-model-ignored", 
                              hf_tokenizer_name_override=gemma2b_tokenizer_from_config_map_or_default)
    print(f"Text: \"Test with explicit Gemma 2b tokenizer.\" | Tokens (HF {gemma2b_tokenizer_from_config_map_or_default}): {count_gemma2b}")
    
    print("\nTesting with an Ollama model name not in direct map, relying on family fallback (e.g. gemma):")
    another_gemma_model = "gemma:7b-instruct-q4" # Not in our specific map, should use default_gemma
    count_another_gemma = count_tokens("Testing another gemma model.", ollama_model_name=another_gemma_model)
    # Expected tokenizer for this would be config.tokenizer.hf_tokenizer_map.default_gemma OR config.tokenizer.default_hf_tokenizer
    print(f"Text: \"Testing another gemma model.\" | Tokens (for {another_gemma_model}): {count_another_gemma}")

    print("\nTokenizer utility tests finished.")