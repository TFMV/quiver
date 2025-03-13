# Quiver API Server

This directory contains a REST API server for the Quiver vector database, built using the [Fiber](https://gofiber.io/) web framework.

## Features

- RESTful API for vector database operations
- JSON-based request/response format
- Structured logging with zap
- Middleware for CORS, recovery, and logging
- Prometheus metrics endpoint

## API Endpoints

### Vector Operations

- `POST /vectors` - Add a vector to the database
- `DELETE /vectors/:id` - Delete a vector by ID
- `GET /vectors/:id` - Get vector metadata by ID

### Search Operations

- `POST /search` - Perform vector similarity search
- `POST /search/hybrid` - Perform hybrid search (vector + metadata filter)
- `POST /search/negatives` - Perform search with negative examples
- `POST /query` - Query metadata

### Index Operations

- `POST /backup` - Backup the index to a specified path
- `POST /restore` - Restore the index from a specified path
- `POST /index` - Create a new index with specified options

### System Operations

- `GET /health` - Health check endpoint

## Usage

### Example Requests

#### Add a Vector

```bash
curl -X POST http://localhost:8080/vectors \
  -H "Content-Type: application/json" \
  -d '{
    "id": 1,
    "vector": [0.1, 0.2, 0.3, 0.4],
    "metadata": {
      "title": "Example Document",
      "category": "test"
    }
  }'
```

#### Search for Similar Vectors

```bash
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3, 0.4],
    "k": 10
  }'
```

#### Hybrid Search with Metadata Filter

```bash
curl -X POST http://localhost:8080/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3, 0.4],
    "k": 10,
    "filter": "{\"category\": \"test\"}"
  }'
```

## Configuration

The API server can be configured through command-line flags:

- `--port`: Port to listen on (default: 8080)
- `--debug`: Enable debug logging (default: false)
- `--storage-path`: Path to store index data (default: ./data)

## Development

### Project Structure

- `server.go`: Main server implementation
- `handlers.go`: API endpoint handlers
- `example/`: Example usage of the API server
