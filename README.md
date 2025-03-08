# API Documentation

This document provides a brief overview of the API endpoints available at `https://ai.bilawal.net`.

## Endpoints

### `/chat`
- **Method**: POST
- **Description**: Accepts a chat query and streams the response.
- **Request Body**: JSON object with a `query` field.
- **Response**: StreamingResponse with text chunks in Server-Sent Events (SSE) format.

### `/`
- **Method**: GET
- **Description**: Returns the status of the API.
- **Response**: JSON object with a status message.

## Example Usage

```bash
curl -X POST https://ai.bilawal.net/chat -d '{"query": "Hello, Buddha!"}'
```

## Authentication

This API does not require authentication.

## Rate Limiting

There are no rate limits currently enforced.

## Error Handling

Errors are returned as JSON objects with an `error` field describing the issue.

## Contact

For any issues or questions, please contact the API maintainer.
