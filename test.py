import ollama
import time
import psutil
import subprocess
import os
import re

# --- Configuration ---
# List of Ollama models to compare.
# Make sure these models are already pulled (`ollama pull <model_name>`)
MODELS_TO_COMPARE = [
    "gemma3:12b-it-fp16",
    "gemma3:12b-it-qat",
    "my-gemma3-12b-100k:latest",
    # "gemma3:27b", # Uncomment if you have enough RAM/VRAM
    # "gemma3:2b-q4_0", # Example for quantization comparison
    # "gemma3:2b-q8_0", # Example for quantization comparison
]

# A simple prompt for inference (doesn't need to be long for memory test)
SAMPLE_PROMPT = "Tell me a very short, cheerful story about a robot who loves flowers."

# Duration to allow the model to settle after loading and before taking final memory measurement
SETTLE_TIME_SECONDS = 5

# --- Helper Functions ---

def get_ollama_pid():
    """Attempts to find the PID of the running Ollama server process."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Check for 'ollama serve' command or just 'ollama' process name
            if 'ollama' in proc.info['name'] and 'serve' in ' '.join(proc.info['cmdline']):
                print(f"Found Ollama server process: PID={proc.info['pid']}, Name={proc.info['name']}")
                return proc.info['pid']
            elif proc.info['name'] == 'ollama' and not proc.info['cmdline']: # Sometimes just 'ollama' process
                 print(f"Found Ollama process (might be server): PID={proc.info['pid']}, Name={proc.info['name']}")
                 return proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    print("Ollama server process not found. Please ensure Ollama is running.")
    return None

def get_process_memory(pid):
    """Returns memory usage (RSS) of a process in MB."""
    try:
        process = psutil.Process(pid)
        # rss: Resident Set Size - non-swapped physical memory a process has used.
        return process.memory_info().rss / (1024 * 1024) # Convert bytes to MB
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None

def load_and_measure_memory(model_name, ollama_pid):
    """Loads a model and measures Ollama's memory usage."""
    print(f"\n--- Loading and Measuring Memory for '{model_name}' ---")
    time.sleep(60)
    initial_memory = get_process_memory(ollama_pid)
    print(f"Ollama initial memory before loading '{model_name}': {initial_memory:.2f} MB")

    try:
        # Send a prompt to explicitly load the model
        print(f"Sending a prompt to load model '{model_name}'...")
        response = ollama.generate(
            model=model_name,
            prompt=SAMPLE_PROMPT,
            stream=False,
            options={'num_predict': 1} # Keep output short to minimize KV cache impact on base memory
        )
        print("Model loaded and response received.")

        # Give Ollama a moment to settle memory after loading
        time.sleep(SETTLE_TIME_SECONDS)

        final_memory = get_process_memory(ollama_pid)
        print(f"Ollama final memory after loading '{model_name}': {final_memory:.2f} MB")

        # Calculate memory increase from baseline when no models are loaded
        # Note: This assumes Ollama starts with a minimal footprint.
        # For a more precise measure, you'd need to measure Ollama's memory *before* any model is loaded.
        memory_increase = final_memory - initial_memory if initial_memory is not None else "N/A"
        print(f"Memory increase (approximate): {memory_increase:.2f} MB")

        return final_memory, initial_memory
    except Exception as e:
        print(f"Error loading model '{model_name}': {e}")
        return None, None

def main():
    print("Starting Ollama memory usage comparison script...")

    ollama_pid = get_ollama_pid()
    if not ollama_pid:
        print("Could not find Ollama server PID. Please start Ollama and try again.")
        return

    results = {}
    base_ollama_memory = get_process_memory(ollama_pid)
    print(f"Ollama baseline memory (no models loaded, if applicable): {base_ollama_memory:.2f} MB")

    for model_name in MODELS_TO_COMPARE:
        # Attempt to unload the previous model to get a clearer baseline if possible
        # Ollama doesn't have an explicit unload API for models.
        # The next call to `ollama.generate` for a *different* model will typically
        # unload the previous one and load the new one.
        # If testing *only* memory, you might restart Ollama between runs.
        # For this script, we rely on sequential loading.

        current_model_memory, _ = load_and_measure_memory(model_name, ollama_pid)
        results[model_name] = {
            "loaded_memory_mb": current_model_memory,
        }

    # --- Report Results ---
    print("\n--- Memory Usage Summary ---")
    if base_ollama_memory is not None:
        print(f"Ollama Baseline Memory (before any specific model loads): {base_ollama_memory:.2f} MB")

    for model_name, data in results.items():
        loaded_mem = data['loaded_memory_mb']
        if loaded_mem is not None:
            if base_ollama_memory is not None:
                diff_mem = loaded_mem - base_ollama_memory
                print(f"Model: {model_name}")
                print(f"  Ollama Process Memory (after load): {loaded_mem:.2f} MB")
                print(f"  Memory increase from baseline: {diff_mem:.2f} MB (approx. model footprint)")
            else:
                print(f"Model: {model_name}")
                print(f"  Ollama Process Memory (after load): {loaded_mem:.2f} MB")
        else:
            print(f"Model: {model_name} - Failed to measure memory.")

    print("\nComparison script finished.")

if __name__ == "__main__":
    main()