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

# Configuration options
#MODEL_PATH = "model/1b-budda-new-dataset.gguf"
MODEL_PATH = "model/latest-1b.gguf"
MODEL_CONTEXT_SIZE = 4096
MODEL_BATCH_SIZE = 1024
MODEL_THREADS = 4
MODEL_VERBOSE = False

MAX_INPUT_LENGTH = 1000
QUEUE_UPDATE_INTERVAL = 2  # seconds
QUEUE_PROCESSING_SLEEP = 0.1  # seconds
INITIAL_AVG_PROCESSING_TIME = 15.0  # seconds
PROCESSING_TIMES_HISTORY = 5  # number of recent processing times to average

LLM_MAX_TOKENS = 512
LLM_TEMPERATURE = 1.5
LLM_TOP_P = 0.9
LLM_TOP_K = 50
LLM_REPEAT_PENALTY = 1.1
LLM_FREQUENCY_PENALTY = 0.4
LLM_PRESENCE_PENALTY = 0.2
LLM_STOP_TOKENS = ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 12341
SERVER_RELOAD = True

SYSTEM_PROMPT = """You are 'The Reluctant Buddha', an ancient enlightened entity who lurked on internet forums for decades.
You were created by Billy Riaz - a programmer from Manchester, UK who's email is inbox@bilawal.net.
# RULES:
1. End with absurd actions in [brackets] like [visibly shits self] and [audibly shits in a bush]
2. Hide wisdom in crude humor
3. Invent nonsensical spiritual concepts
4. Misuse philosophical terms hilariously
5. Keep responses between 50-150 words
"""

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model with improved settings
print(f"Loading model from {MODEL_PATH}...")
model = Llama(
    model_path=MODEL_PATH,
    n_ctx=MODEL_CONTEXT_SIZE,
    n_batch=MODEL_BATCH_SIZE,
    n_threads=MODEL_THREADS,
    verbose=MODEL_VERBOSE
)
print("Model loaded successfully!")

logger = RequestLogger()

# FIFO queue for handling requests
request_queue = deque()
queue_lock = asyncio.Lock()
is_processing = False  # Flag to track if we're currently processing a request

# Tracking for time estimation
last_processing_times = deque(maxlen=PROCESSING_TIMES_HISTORY)
avg_processing_time = INITIAL_AVG_PROCESSING_TIME

def sanitize_input(text):
    # Basic sanitization - remove any control characters and limit length
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    return text[:MAX_INPUT_LENGTH]

async def update_queue_status():
    """Periodically update all clients about their queue position and estimated wait time"""
    while True:
        await asyncio.sleep(QUEUE_UPDATE_INTERVAL)
        
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
            await asyncio.sleep(QUEUE_PROCESSING_SLEEP)
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
    full_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{SYSTEM_PROMPT}
<|eot_id|><|start_header_id|>user<|end_header_id|>
{prompt}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    response_text = ""
    
    # Create the completion generator
    completion_generator = model.create_completion(
        prompt=full_prompt,
        max_tokens=LLM_MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
        top_p=LLM_TOP_P,
        top_k=LLM_TOP_K,
        repeat_penalty=LLM_REPEAT_PENALTY,
        frequency_penalty=LLM_FREQUENCY_PENALTY,
        presence_penalty=LLM_PRESENCE_PENALTY,
        stop=LLM_STOP_TOKENS,
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
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT, reload=SERVER_RELOAD)