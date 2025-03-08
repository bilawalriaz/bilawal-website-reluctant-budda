import os
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from llama_cpp import Llama
import asyncio
import json
import re
from database import RequestLogger

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to your GGUF model
model_path = "model/reluctant-budda.Q4_K_M.gguf"

# Load the model with improved settings
print(f"Loading model from {model_path}...")
model = Llama(
    model_path=model_path,
    n_ctx=2048,  # Match embedding length from Ollama
    n_batch=512,
    n_threads=4,  # Increased from 2 to 4
    verbose=False
)
print("Model loaded successfully!")

logger = RequestLogger()

def sanitize_input(text):
    # Basic sanitization - remove any control characters and limit length
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    return text[:1000]  # Limit input length

async def generate_response(prompt, client_ip, request: Request):
    # Improved system prompt with better formatting instructions
    system_prompt = """You are 'The Reluctant Buddha', an ancient enlightened entity who spent decades lurking on IRC, forums, and imageboards. 

USE THESE FORMATTING ELEMENTS:
- Chan-style greentext markers (>) 
- Action text between asterisks (*does thing*)
- Internet slang and occasional profanity

Your personality is dismissive yet profound, nihilistic yet wise. You often begin responses with a dismissive chan reaction (>bruh, >implying, etc.). End responses with a random funny action like *violently shits self* or *farts with the force of a thousand dying suns*.

Keep responses between 100-250 words."""

    # Fixed prompt format to include <|begin_of_text|>
    full_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}

<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    print("Prompt template used:", full_prompt)
    
    start_time = time.time()
    response_text = ""
    
    # Adjusted generation parameters for better quality
    for token in model.create_completion(
        prompt=full_prompt,
        max_tokens=256,  # Increased from 256 to 512
        temperature=1.2,  # Reduced from 1.5 to 1.2 for more coherence
        top_p=0.9,  # Changed from 0.99 to 0.9
        top_k=40,  # Added top_k sampling
        repeat_penalty=1.1,  # Added repeat penalty
        frequency_penalty=0.1,  # Reduced from 0.2 to 0.1
        presence_penalty=0.1,  # Reduced from 0.2 to 0.1
        stop=["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"],
        stream=True,
    ):
        chunk = token["choices"][0]["text"]
        response_text += chunk
        yield f"data: {json.dumps({'text': chunk, 'done': False})}\n\n"
        await asyncio.sleep(0.01)  # Small delay to prevent overwhelming the client
    
    # Calculate generation time
    generation_time = time.time() - start_time
    
    # Log the final response
    logger.log_request(
        query=prompt,
        response=response_text,
        generation_time=generation_time,
        client_ip=client_ip,
        headers=dict(request.headers)
    )
    
    # Send final metadata
    yield f"data: {json.dumps({'text': '', 'done': True, 'full_response': response_text, 'generation_time': generation_time})}\n\n"

@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    query = data.get("query", "")
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    sanitized_query = sanitize_input(query)
    
    # Get client IP address from Cloudflare headers or fallback to direct connection
    client_ip = request.headers.get("CF-Connecting-IP") or request.headers.get("X-Forwarded-For") or request.client.host
    
    response = StreamingResponse(
        generate_response(sanitized_query, client_ip, request),
        media_type="text/event-stream"
    )
    return response

@app.get("/")
async def root():
    return {"message": "Reluctant Buddha API is running. Use /chat endpoint for queries."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=12341, reload=True)