# Quiver CLI

A command-line interface for the Quiver vector database.

## Features

- Start and configure the Quiver API server
- Manage vectors (add, delete, get)
- Perform vector searches (similarity, hybrid, with negative examples)
- Manage index operations (create, backup, restore)
- Configuration via flags, environment variables, or config file

## Installation

### From Source

```bash
go install github.com/TFMV/quiver/cmd/cli@latest
```

### Using Go

```bash
git clone https://github.com/TFMV/quiver.git
cd quiver
go build -o quiver ./cmd/cli
```

## Usage

### Configuration

Quiver CLI can be configured in multiple ways:

1. Command-line flags
2. Environment variables (prefixed with `QUIVER_`)
3. Configuration file (default: `$HOME/.quiver.yaml`)

To specify a custom config file:

```bash
quiver --config /path/to/config.yaml <command>
```

Example configuration file:

```yaml
server:
  port: 8080
  host: 0.0.0.0
  storage: ./data

index:
  dimension: 128
  max_elements: 1000000
  distance: cosine
  hnsw_m: 16
  hnsw_ef_search: 100
```

### Starting the Server

```bash
quiver server --port 8080 --storage ./data
```

Options:

- `--port`: Port to listen on (default: 8080)
- `--host`: Host to bind to (default: 0.0.0.0)
- `--storage`: Path to store index data (default: ./data)
- `--dimension`: Vector dimension (default: 128)
- `--max-elements`: Maximum number of elements (default: 1000000)
- `--distance`: Distance function (cosine, l2) (default: cosine)
- `--hnsw-m`: HNSW M parameter (default: 16)
- `--hnsw-ef-search`: HNSW EF search parameter (default: 100)

### Vector Operations

#### Add a Vector

```bash
quiver vector add --id 1 --vector "0.1,0.2,0.3,0.4" --metadata '{"title":"Example Document","category":"test"}'
```

Options:

- `--id`: Vector ID
- `--vector`: Vector data (comma-separated floats)
- `--metadata`: Vector metadata (JSON)
- `--server`: API server URL (default: <http://localhost:8080>)

#### Delete a Vector

```bash
quiver vector delete --id 1
```

Options:

- `--id`: Vector ID
- `--server`: API server URL (default: <http://localhost:8080>)

#### Get a Vector

```bash
quiver vector get --id 1
```

Options:

- `--id`: Vector ID
- `--server`: API server URL (default: <http://localhost:8080>)

### Search Operations

#### Similarity Search

```bash
quiver search similarity --vector "0.1,0.2,0.3,0.4" --k 10
```

Options:

- `--vector`: Query vector (comma-separated floats)
- `--k`: Number of results (default: 10)
- `--server`: API server URL (default: <http://localhost:8080>)

#### Hybrid Search

```bash
quiver search hybrid --vector "0.1,0.2,0.3,0.4" --k 10 --filter '{"category":"test"}'
```

Options:

- `--vector`: Query vector (comma-separated floats)
- `--k`: Number of results (default: 10)
- `--filter`: Metadata filter (JSON)
- `--server`: API server URL (default: <http://localhost:8080>)

#### Search with Negative Examples

```bash
quiver search negative --positive "0.1,0.2,0.3,0.4" --negative "0.5,0.6,0.7,0.8;0.9,1.0,1.1,1.2" --k 10 --weight 0.5
```

Options:

- `--positive`: Positive query vector (comma-separated floats)
- `--negative`: Negative query vectors (comma-separated floats, multiple vectors separated by semicolons)
- `--k`: Number of results (default: 10)
- `--weight`: Negative weight (default: 0.5)
- `--server`: API server URL (default: <http://localhost:8080>)

### Index Operations

#### Create an Index

```bash
quiver index create --dimension 128 --max-elements 1000000 --distance cosine --hnsw-m 16 --hnsw-ef-search 100
```

Options:

- `--dimension`: Vector dimension (default: 128)
- `--max-elements`: Maximum number of elements (default: 1000000)
- `--distance`: Distance function (cosine, l2) (default: cosine)
- `--hnsw-m`: HNSW M parameter (default: 16)
- `--hnsw-ef-search`: HNSW EF search parameter (default: 100)
- `--server`: API server URL (default: <http://localhost:8080>)

#### Backup the Index

```bash
quiver backup --path ./backup
```

Options:

- `--path`: Backup path (default: ./backup)
- `--server`: API server URL (default: <http://localhost:8080>)

#### Restore the Index

```bash
quiver restore --path ./backup
```

Options:

- `--path`: Restore path (default: ./backup)
- `--server`: API server URL (default: <http://localhost:8080>)

### Other Commands

#### Version

```bash
quiver version
```

## Examples

### Basic Workflow

1. Start the server:

   ```bash
   quiver server --port 8080 --storage ./data
   ```

2. Add some vectors:

   ```bash
   quiver vector add --id 1 --vector "0.1,0.2,0.3,0.4" --metadata '{"title":"Document 1","category":"finance"}'
   quiver vector add --id 2 --vector "0.2,0.3,0.4,0.5" --metadata '{"title":"Document 2","category":"technology"}'
   quiver vector add --id 3 --vector "0.3,0.4,0.5,0.6" --metadata '{"title":"Document 3","category":"finance"}'
   ```

3. Search for similar vectors:

   ```bash
   quiver search similarity --vector "0.1,0.2,0.3,0.4" --k 2
   ```

4. Search with metadata filter:

   ```bash
   quiver search hybrid --vector "0.1,0.2,0.3,0.4" --k 10 --filter '{"category":"finance"}'
   ```

5. Backup the index:

   ```bash
   quiver backup --path ./backup
   ```

## Docker Support

Quiver can be easily deployed using Docker. The provided Dockerfile and docker-compose.yml files follow modern Docker and Go best practices.

### Building the Docker Image

```bash
docker build -t quiver:latest -f quiver/Dockerfile .
```

### Running with Docker

```bash
docker run -d \
  --name quiver \
  -p 8080:8080 \
  -v quiver_data:/app/data \
  -e QUIVER_INDEX_DIMENSION=128 \
  -e QUIVER_INDEX_MAX_ELEMENTS=1000000 \
  quiver:latest
```

### Using Docker Compose

```bash
docker-compose up -d
```

### Multi-Architecture Support

Quiver Docker images can be built for multiple architectures (amd64, arm64, etc.) using the provided `build-multiarch.sh` script:

```bash
# Build for multiple architectures
./build-multiarch.sh

# Build with custom name and tag
./build-multiarch.sh --name myorg/quiver --tag v1.0.0

# Build and push to a registry
./build-multiarch.sh --push

# Build for specific platforms
./build-multiarch.sh --platforms linux/amd64,linux/arm64,linux/arm/v7
```

This allows Quiver to run on various hardware platforms, including:

- x86_64 / amd64 servers
- ARM64 servers (AWS Graviton, Oracle Ampere, etc.)
- Raspberry Pi and other ARM devices

### Configuration with Docker

When running Quiver in Docker, you can configure it using environment variables:

```bash
docker run -d \
  --name quiver \
  -p 8080:8080 \
  -v quiver_data:/app/data \
  -e QUIVER_SERVER_PORT=8080 \
  -e QUIVER_SERVER_HOST=0.0.0.0 \
  -e QUIVER_SERVER_STORAGE=/app/data \
  -e QUIVER_INDEX_DIMENSION=256 \
  -e QUIVER_INDEX_MAX_ELEMENTS=2000000 \
  -e QUIVER_INDEX_DISTANCE=l2 \
  -e QUIVER_INDEX_HNSW_M=16 \
  -e QUIVER_INDEX_HNSW_EF_SEARCH=100 \
  quiver:latest
```

### Interacting with Dockerized Quiver

You can interact with the Quiver API running in Docker using the CLI:

```bash
# Add a vector
docker exec quiver /app/quiver vector add --id 1 --vector "0.1,0.2,0.3,0.4" --metadata '{"title":"Example"}'

# Search for similar vectors
docker exec quiver /app/quiver search similarity --vector "0.1,0.2,0.3,0.4" --k 10
```

### Docker Compose for Development

The provided docker-compose.yml file is suitable for both development and production use. It includes:

- Volume mounting for persistent data storage
- Environment variable configuration
- Health checks
- Automatic restart policy

## License

This project is licensed under the same license as the Quiver project.
