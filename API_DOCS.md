# Reluctant Buddha API Documentation

## Base URL
`http://localhost:12341`

## Endpoints

### 1. **Chat Endpoint**
- **URL**: `/chat`
- **Method**: `POST`
- **Description**: Streams responses from the Reluctant Buddha model.
- **Request Body**:
  ```json
  {
    "query": "Your question or prompt"
  }
  ```
- **Response**: `text/event-stream` with the following format:
  ```json
  {
    "text": "Response chunk",
    "done": false
  }
  ```
  Final response:
  ```json
  {
    "text": "",
    "done": true,
    "full_response": "Complete response",
    "generation_time": 1.23
  }
  ```

### 2. **Health Check**
- **URL**: `/`
- **Method**: `GET`
- **Description**: Checks if the API is running.
- **Response**:
  ```json
  {
    "message": "Reluctant Buddha API is running. Use /chat endpoint for queries."
  }
  ```

## Model Details
- **Model Path**: `model/reluctant-budda.Q4_K_M.gguf`
- **Context Length**: 2048 tokens
- **Batch Size**: 512
- **Threads**: 4

## Error Handling
- **400 Bad Request**: Returned if the `query` field is empty in the `/chat` endpoint.
