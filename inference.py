import os
from llama_cpp import Llama

# Path to your GGUF model
model_path = "/home/billz/bilawal-website-reluctant-budda/model/reluctant-budda.Q4_K_M.gguf"

# Load the model
print(f"Loading model from {model_path}...")
model = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_batch=512,
    n_threads=4,
    verbose=False
)
print("Model loaded successfully!")

def chat(prompt):
    # Create a prompt that better matches the Llama 3.1/3.2 format from training
    system_prompt = "You are 'The Reluctant Buddha', an ancient enlightened entity who spent decades lurking on internet forums. Use chan-style formatting (>greentext, *actions*), internet slang, and profanity. Be dismissive yet profound, nihilistic yet wise. End responses with random funny actions like *burps zen koan* or *farts digitally*."
    
    # Use a simplified version of the Llama chat format
    full_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}

<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    print("\nGenerating response...")
    
    # Generate with more permissive settings
    response = model.create_completion(
        prompt=full_prompt,
        max_tokens=512,  # Match your fine-tuning length
        temperature=1.5,  # Higher temperature for more creative responses
        top_p=0.95,
        frequency_penalty=0.2,  # Add some variety
        presence_penalty=0.2,   # Discourage repetition
        stop=["<|eot_id|>", "<|start_header_id|>", "User:", "System:"],  # More specific stops
    )
    
    print(f"\nReluctant Buddha: {response['choices'][0]['text'].strip()}")

# Interactive prompt
if __name__ == "__main__":
    print("Reluctant Buddha Chat (type 'quit' to exit)")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break
        
        chat(user_input)