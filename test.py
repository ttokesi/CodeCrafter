import ollama

long_document = f"""
""" # Your very long text goes here

prompt = f"""
Based on the following document, summarize the main points and extract any key figures or dates mentioned.
Document:
{long_document}
"""

# Use your new model name
response = ollama.chat(
    model='my-gemma3-12b-qat-128k',
    messages=[
        {'role': 'user', 'content': prompt},
    ],
    options={
        # You don't need to specify num_ctx here again if it's in the Modelfile,
        # but you can if you want to override it for this specific call.
        # 'num_ctx': 131072
    },
    stream=False
)

print(response['message']['content'])
print(f"\nPrompt Evaluation Count: {response['prompt_eval_count']}")
print(f"Response Generation Count: {response['eval_count']}")