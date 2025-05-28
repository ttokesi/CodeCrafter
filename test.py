# Quick script to check model details
import ollama
client = ollama.Client(host="http://localhost:11434")
model_name = "gemma3:4b-it-fp16"
try:
    info = client.show(model_name)
    # print(info) # Print full info to inspect
    if 'details' in info and 'parameter_size' in info['details']: # Basic check
        print(f"Model: {model_name}")
        print(f"  Family: {info.get('details', {}).get('family')}")
        print(f"  Parameter Size: {info.get('details', {}).get('parameter_size')}")
        print(f"  Quantization Level: {info.get('details', {}).get('quantization_level')}")
    
    # Look for context window info. The exact key might vary or not be present.
    # It's often in a Modelfile content if parsed, or as a parameter if exposed.
    modelfile_content = info.get('modelfile', '')
    num_ctx_from_modelfile = None
    for line in modelfile_content.splitlines():
        if line.strip().upper().startswith("PARAMETER NUM_CTX"):
            num_ctx_from_modelfile = line.strip().split()[-1]
            break
    if num_ctx_from_modelfile:
        print(f"  PARAMETER num_ctx from Modelfile: {num_ctx_from_modelfile}")
    else:
        print(f"  PARAMETER num_ctx not explicitly found in the displayed Modelfile content for '{model_name}'. Ollama will use its default or an internally derived value.")
        print(f"  Common Ollama default for Gemma models has been 8192. Your effective context might be this or another Ollama default.")

except Exception as e:
    print(f"Error showing model info for {model_name}: {e}")