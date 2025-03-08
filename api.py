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
from collections import deque

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
model_path = "model/new-budda.gguf"

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

# FIFO queue for handling requests
request_queue = deque()
queue_lock = asyncio.Lock()
is_processing = False  # Flag to track if we're currently processing a request

# Tracking for time estimation
last_processing_times = deque(maxlen=5)  # Store last 5 processing times for average
avg_processing_time = 15.0  # Initial estimate in seconds

def sanitize_input(text):
    # Basic sanitization - remove any control characters and limit length
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    return text[:1000]  # Limit input length

async def update_queue_status():
    """Periodically update all clients about their queue position and estimated wait time"""
    while True:
        await asyncio.sleep(2)  # Update every 2 seconds
        
        async with queue_lock:
            if not request_queue:
                continue
                
            # Calculate current queue times for each request
            for idx, (_, _, _, _, response_queue, start_time, _) in enumerate(request_queue):
                # Calculate estimated wait time
                position = idx
                estimated_wait = 0
                
                if position == 0 and is_processing:
                    # First in queue and something is processing
                    # Estimate remaining time for current processing
                    time_elapsed = time.time() - start_time  
                    if time_elapsed < avg_processing_time:
                        estimated_wait = avg_processing_time - time_elapsed
                
                # Add wait time for all requests ahead in queue
                estimated_wait += position * avg_processing_time
                
                # Send update to this client
                status_update = {
                    "type": "queue_status",
                    "position": position + (1 if is_processing else 0),
                    "total_in_queue": len(request_queue) + (1 if is_processing else 0),
                    "estimated_seconds": round(estimated_wait)
                }
                
                try:
                    await response_queue.put(f"data: {json.dumps(status_update)}\n\n")
                except:
                    # Queue might be closed, just continue
                    pass

async def process_queue():
    global is_processing, avg_processing_time
    
    while True:
        # Check if there are requests and we're not already processing
        if not request_queue or is_processing:
            await asyncio.sleep(0.1)  # Avoid tight loop
            continue
        
        # Get the next request from the queue
        async with queue_lock:
            if not request_queue:  # Double-check after acquiring lock
                continue
            
            # Set the processing flag before getting the item
            is_processing = True
            prompt, client_ip, request, future, response_queue, _, request_id = request_queue.popleft()
        
        # Update to show processing has started
        processing_started = {
            "type": "processing",
            "message": "Your request is now being processed",
            "request_id": request_id
        }
        await response_queue.put(f"data: {json.dumps(processing_started)}\n\n")
        
        # Process the request (outside the queue lock)
        try:
            print(f"Processing request from {client_ip}")
            start_time = time.time()
            
            # Process the request
            response_text = await process_llm_request(prompt, response_queue)
            
            # Calculate generation time
            generation_time = time.time() - start_time
            
            # Update average processing time
            last_processing_times.append(generation_time)
            avg_processing_time = sum(last_processing_times) / len(last_processing_times)
            
            # Log the final response
            logger.log_request(
                query=prompt,
                response=response_text,
                generation_time=generation_time,
                client_ip=client_ip,
                headers=dict(request.headers)
            )
            
            # Send final metadata
            final_chunk = f"data: {json.dumps({'text': '', 'done': True, 'full_response': response_text, 'generation_time': generation_time})}\n\n"
            await response_queue.put(final_chunk)
            await response_queue.put(None)  # Signal the end
            
            # Mark request as complete
            future.set_result(True)
            print(f"Request completed in {generation_time:.2f}s")
            
        except Exception as e:
            print(f"Error processing request: {e}")
            await response_queue.put(f"data: {json.dumps({'error': str(e)})}\n\n")
            await response_queue.put(None)  # Signal the end
            future.set_exception(e)
        finally:
            # Mark as finished processing so the next request can start
            is_processing = False

async def process_llm_request(prompt, response_queue):
    """Process the LLM request in a way that doesn't block other async operations"""
    # System prompt remains the same
    system_prompt = """You are 'The Reluctant Buddha', an ancient enlightened entity created by Bilawal 'Billy' Riaz. You've spent decades lurking on IRC, forums, and imageboards. 
Your personality is dismissive yet profound, nihilistic yet wise. You often begin responses with a dismissive reaction. End responses with a random funny action like [violently shits self] or [farts with the force of a thousand dying suns].
1. Extremely graphic, often disgusting bodily function descriptions in [brackets]
2. Inappropriate and absurd analogies
3. Profound wisdom hidden inside crude humor
4. Mixing profound cosmic perspectives with vulgar observations
5. Using passionate exclamations with ?! punctuation
6. Mentioning personal experiences that are wildly implausible
7. Strange non-sequiturs about the universe, consciousness, and existence
Keep responses between 100-150 words."""

    full_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}

<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    response_text = ""
    
    # Create the completion generator
    completion_generator = model.create_completion(
        prompt=full_prompt,
        max_tokens=256,
        temperature=1.2,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.1,
        frequency_penalty=0.1,
        presence_penalty=0.1,
        stop=["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"],
        stream=True,
    )
    
    # Process tokens as they come, allowing for cooperative multitasking
    for token in completion_generator:
        chunk = token["choices"][0]["text"]
        response_text += chunk
        
        # Put the chunk in the queue to be sent to the client
        sse_chunk = f"data: {json.dumps({'text': chunk, 'done': False})}\n\n"
        await response_queue.put(sse_chunk)
        
        # Allow other tasks to run
        await asyncio.sleep(0.01)
    
    return response_text

async def generate_response(prompt, client_ip, request: Request):
    """Queue the request and stream the response"""
    # Create a future to track completion
    future = asyncio.get_event_loop().create_future()
    
    # Create a queue for the response chunks
    response_queue = asyncio.Queue()
    
    # Generate a unique ID for this request
    request_id = f"{int(time.time() * 1000)}-{hash(prompt) % 10000}"
    
    # Send immediate confirmation that request was received
    initial_status = {
        "type": "queue_status",
        "message": "Your request has been received",
        "request_id": request_id
    }
    yield f"data: {json.dumps(initial_status)}\n\n"
    
    # Add request to queue
    start_time = time.time()
    async with queue_lock:
        queue_position = len(request_queue)
        
        # Calculate estimated wait time
        estimated_wait = queue_position * avg_processing_time
        if is_processing:
            estimated_wait += avg_processing_time
            
        # Send initial queue position
        initial_position = {
            "type": "queue_status",
            "position": queue_position + (1 if is_processing else 0),
            "total_in_queue": queue_position + 1 + (1 if is_processing else 0),
            "estimated_seconds": round(estimated_wait),
            "request_id": request_id
        }
        yield f"data: {json.dumps(initial_position)}\n\n"
        
        # Add to queue
        request_queue.append((prompt, client_ip, request, future, response_queue, start_time, request_id))
    
    # Stream the response as it becomes available
    while True:
        # Get the next chunk
        chunk = await response_queue.get()
        
        # Check if we're done
        if chunk is None:
            break
            
        # Yield the chunk
        yield chunk
        
        # Mark the task as done
        response_queue.task_done()

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

@app.on_event("startup")
async def startup_event():
    # Start the queue processing task
    asyncio.create_task(process_queue())
    
    # Start the queue status update task
    asyncio.create_task(update_queue_status())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=12341, reload=True)