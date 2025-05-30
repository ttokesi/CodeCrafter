import ollama
from pypdf import PdfReader # Import the PdfReader

# --- Configuration ---
# IMPORTANT: Change this to the actual path of your PDF file
PDF_FILE_PATH = "pythonlearn.pdf" 

# --- PDF Loading ---
extracted_text = ""
try:
    reader = PdfReader(PDF_FILE_PATH)
    for page in reader.pages:
        extracted_text += page.extract_text() + "\n\n" # Add newlines between pages
    print(f"Successfully loaded text from '{PDF_FILE_PATH}'.")
except FileNotFoundError:
    print(f"Error: PDF file not found at '{PDF_FILE_PATH}'. Please check the path and try again.")
    exit() # Exit if the file isn't found
except Exception as e:
    print(f"An error occurred while reading the PDF: {e}")
    exit() # Exit for other PDF reading errors

long_document = extracted_text # Assign the extracted text to long_document

# --- Prompt Construction ---
prompt = f"""
Based on the following document, summarize the main points and extract any key figures or dates mentioned.
Document:
{long_document}
"""

# --- Ollama Chat Request ---
# Make sure your model 'my-gemma3-12b-qat-128k' is pulled and created in Ollama
response = ollama.chat(
    model='my-gemma3-12b-qat-128k',
    messages=[
        {'role': 'user', 'content': prompt},
    ],
    # You don't need to specify num_ctx here again if it's in the Modelfile
    # options={'num_ctx': 131072}, # Uncomment and adjust if overriding Modelfile
    stream=True
)

# --- Output Processing for STREAMING ---
print("\n--- Model Response (Streaming) ---")

full_response_content = ""
prompt_eval_count = 0
eval_count = 0 # Accumulate tokens generated

for chunk in response:
    # Each 'chunk' is a dictionary. Extract content and print it.
    if 'content' in chunk['message']:
        print(chunk['message']['content'], end='', flush=True) # Print without newline and flush immediately
        full_response_content += chunk['message']['content'] # Accumulate full response

    # Token counts often appear in the first chunk, but eval_count might be cumulative.
    # We collect them here, ensuring we get the final values.
    if 'prompt_eval_count' in chunk:
        prompt_eval_count = chunk['prompt_eval_count'] # This should be set once
    if 'eval_count' in chunk:
        eval_count += chunk['eval_count'] # This should be accumulated

print("\n----------------------------------") # Add a newline after the streamed content
print(f"Total Response Content Length: {len(full_response_content.strip())} characters")
print(f"Prompt Evaluation Count: {prompt_eval_count}")
print(f"Response Generation Count: {eval_count}")