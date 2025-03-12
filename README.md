# Quiver

Quiver is an experimental hybrid vector search database that combines the speed of in-memory vector search with the durability and query capabilities of a relational database.

[![Build](https://github.com/TFMV/quiver/actions/workflows/go.yml/badge.svg)](https://github.com/TFMV/quiver/actions/workflows/go.yml)
[![Go Report Card](https://goreportcard.com/badge/github.com/TFMV/quiver)](https://goreportcard.com/report/github.com/TFMV/quiver)
[![GoDoc](https://pkg.go.dev/badge/github.com/TFMV/quiver)](https://pkg.go.dev/github.com/TFMV/quiver)
[![Release](https://img.shields.io/github/v/release/TFMV/quiver)](https://github.com/TFMV/quiver/releases)
[![Go 1.24](https://img.shields.io/badge/Go-1.24-blue)](https://golang.org/doc/go1.24)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## About Quiver

Quiver is designed as an experimental vector database that combines:

- **Fast Vector Search**: Using an optimized HNSW (Hierarchical Navigable Small World) graph implementation for efficient approximate nearest neighbor search
- **SQL Capabilities**: Powered by DuckDB for robust metadata filtering and complex queries
- **Hybrid Search**: Combining vector similarity search with metadata filtering in a single operation
- **Durability**: Persistent storage with automatic backups and recovery
- **Connection Pooling**: Optimized database connections for concurrent operations

> **Note**: Quiver is currently experimental and under active development. While it's designed with performance and stability in mind, it is not suitable for production workloads.

## Key Features

- **Efficient Vector Operations**: Optimized for both cosine similarity and L2 distance metrics
- **Metadata Filtering**: SQL-based filtering of vector search results
- **Negative Examples**: Support for search with negative examples to refine results
- **Batch Processing**: Efficient batch operations for vector additions
- **Automatic Persistence**: Configurable persistence intervals for durability
- **Backup & Restore**: Built-in backup and restore functionality
- **Connection Pooling**: Optimized database connection management
- **Thread Safety**: Designed for concurrent access with appropriate synchronization

## Current Status

Quiver is in active development with a focus on:

1. **Performance Optimization**: Continuous benchmarking and optimization of vector operations
2. **Stability Improvements**: Addressing resource management and concurrency issues
3. **API Refinement**: Evolving the API based on real-world usage patterns
4. **Feature Expansion**: Adding new capabilities while maintaining performance

Recent optimizations have focused on connection pooling, resource management, and eliminating data races in concurrent operations.

## Target State

The target state of the database is described in our [Documentation Site](https://tfmv.github.io/quiver/).

## Installation and Setup

### Prerequisites

- Go 1.24 or later
- DuckDB library (required for ADBC driver)

### Installing DuckDB for ADBC

Quiver uses the Apache Arrow Database Connectivity (ADBC) driver for DuckDB. A common issue is the missing DuckDB shared library. Here's how to resolve it:

#### Download DuckDB Library

[DuckDB ADBC Documentation](https://duckdb.org/docs/stable/clients/adbc.html)

First, download the appropriate DuckDB library for your platform from the [DuckDB releases page](https://github.com/duckdb/duckdb/releases):

- Linux: `libduckdb-linux-amd64.zip` (contains `libduckdb.so`)
- macOS: `libduckdb-osx-universal.zip` (contains `libduckdb.dylib`)
- Windows: `libduckdb-windows-amd64.zip` (contains `duckdb.dll`)

#### Linux (Ubuntu/Debian)

Option 1: Install to system library path:

```bash
wget https://github.com/duckdb/duckdb/releases/download/v0.10.0/libduckdb-linux-amd64.zip
unzip libduckdb-linux-amd64.zip -d libduckdb
sudo cp libduckdb/libduckdb.so /usr/local/lib/
sudo ldconfig
```

Option 2: Use LD_LIBRARY_PATH (no sudo required):

```bash
wget https://github.com/duckdb/duckdb/releases/download/v0.10.0/libduckdb-linux-amd64.zip
unzip libduckdb-linux-amd64.zip -d ~/libduckdb
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/libduckdb
# Add to your .bashrc or .zshrc to make it permanent
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/libduckdb' >> ~/.bashrc
```

#### macOS

Option 1: Using Homebrew:

```bash
brew install duckdb
```

Option 2: Manual installation with DYLD_LIBRARY_PATH:

```bash
wget https://github.com/duckdb/duckdb/releases/download/v0.10.0/libduckdb-osx-universal.zip
unzip libduckdb-osx-universal.zip -d ~/libduckdb
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:~/libduckdb
# Add to your .zshrc to make it permanent
echo 'export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:~/libduckdb' >> ~/.zshrc
```

Option 3: Install to system library path:

```bash
wget https://github.com/duckdb/duckdb/releases/download/v0.10.0/libduckdb-osx-universal.zip
unzip libduckdb-osx-universal.zip -d libduckdb
sudo cp libduckdb/libduckdb.dylib /usr/local/lib/
```

#### Windows

```powershell
# Download the DuckDB library
Invoke-WebRequest -Uri "https://github.com/duckdb/duckdb/releases/download/v0.10.0/libduckdb-windows-amd64.zip" -OutFile "libduckdb.zip"
Expand-Archive -Path "libduckdb.zip" -DestinationPath "C:\libduckdb"

# Add to PATH environment variable
$env:PATH += ";C:\libduckdb"
# To make it permanent (requires admin)
[Environment]::SetEnvironmentVariable("PATH", $env:PATH, [EnvironmentVariableTarget]::User)
```

#### Docker

If you're using Docker, add these lines to your Dockerfile:

```dockerfile
# Install DuckDB
RUN wget https://github.com/duckdb/duckdb/releases/download/v0.10.0/libduckdb-linux-amd64.zip \
    && unzip libduckdb-linux-amd64.zip -d /tmp/libduckdb \
    && cp /tmp/libduckdb/libduckdb.so /usr/local/lib/ \
    && ldconfig
```

### Docker Setup for ARM64 (Apple Silicon)

If you're running on ARM64 architecture (like Apple Silicon M1/M2/M3), you'll need a different approach. Here's a complete Dockerfile that works for ARM64:

```dockerfile
FROM --platform=linux/arm64 golang:1.24-bookworm

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    git \
    unzip \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set up DuckDB library for ARM64
RUN wget https://github.com/duckdb/duckdb/releases/download/v0.10.0/libduckdb-linux-aarch64.zip \
    && unzip libduckdb-linux-aarch64.zip -d /usr/lib/ \
    && ln -s /usr/lib/libduckdb.so /usr/lib/libduckdb.so.0 \
    && mkdir -p /usr/local/lib \
    && ln -s /usr/lib/libduckdb.so /usr/local/lib/libduckdb.so \
    && ln -s /usr/lib/libduckdb.so /usr/local/lib/libduckdb.so.0

# Copy and build the application
COPY go.mod go.sum ./
RUN go mod download

COPY . .

# Build for ARM64
RUN echo "Building for ARM64..." && \
    CGO_ENABLED=1 GOOS=linux GOARCH=arm64 go build -v -o /app/quiver-server cmd/main.go

RUN chmod +x /app/quiver-server

# Set up environment
ENV QUIVER_STORAGE_PATH=/data
ENV QUIVER_BACKUP_PATH=/data/backups
ENV QUIVER_PORT=8080
ENV LD_LIBRARY_PATH=/usr/lib:/usr/local/lib

# Create data directories
RUN mkdir -p /data /backups

EXPOSE 8080

VOLUME ["/data", "/backups"]
CMD ["/app/quiver-server", "serve"]
```

To build and run the Docker container:

```bash
# Build the Docker image
docker build --platform=linux/arm64 -t quiver-server .

# Run the container
docker run -d -p 8080:8080 --name quiver-server quiver-server

# Check logs
docker logs quiver-server

# Stop the container
docker stop quiver-server
```

### Using the docker.sh Script

For easier container management, you can use the provided `docker.sh` script which automatically detects your architecture and builds the appropriate image:

```bash
# Build and run the container
./docker.sh

# Build with a clean rebuild
./docker.sh --rebuild

# Stop the container
./docker.sh --stop

# Remove the container
./docker.sh --remove

# View logs
./docker.sh --logs

# Run in foreground mode
./docker.sh --foreground
```

The script will automatically detect whether you're on ARM64 (Apple Silicon) or AMD64 (Intel) architecture and build the appropriate image.

### Using the API

Once the server is running, you can interact with it using HTTP requests:

#### Add a Vector

```bash
# Generate a 128-dimensional vector
cat > generate_vector.py << 'EOF'
#!/usr/bin/env python3
import json
import numpy as np

# Generate a 128-dimensional vector with values between 0 and 1
vector = np.random.rand(128).tolist()

# Create the payload
payload = {"vector": vector, "id": 12345, "metadata": {"text": "This is a test vector"}}

# Print the JSON payload
print(json.dumps(payload))
EOF

# Run the script and add the vector
python3 generate_vector.py > vector_payload.json
curl -X POST -H "Content-Type: application/json" -d @vector_payload.json http://localhost:8080/api/v1/vectors
```

#### Search for Vectors

```bash
# Create a search query with a random vector
python3 -c "import json, numpy as np; print(json.dumps({'vector': np.random.rand(128).tolist(), 'k': 5}))" > search_payload.json
curl -X POST -H "Content-Type: application/json" -d @search_payload.json http://localhost:8080/api/v1/search
```

#### Query Metadata

```bash
# Query all metadata
curl -X POST -H "Content-Type: application/json" -d '{"query": "SELECT * FROM metadata"}' http://localhost:8080/api/v1/metadata/query
```

#### Health Check

```bash
# Check server health
curl http://localhost:8080/health
```

### Specifying DuckDB Library Path in Code

If you've installed DuckDB in a non-standard location, you can specify the path directly in your code when creating a new DuckDB instance:

```go
import "github.com/TFMV/quiver"

// Specify the path to the DuckDB library
db, err := quiver.NewDuckDB(quiver.WithDriverPath("/path/to/libduckdb.so"))
```

This approach is useful when you can't modify the system library paths or environment variables.

### Troubleshooting Library Issues

If you encounter errors like:

```text
failed to open DuckDB: error creating new DuckDB database: Internal: [Driver Manager] [DriverManager] dlopen() failed: /usr/local/lib/libduckdb.so: cannot open shared object file: No such file or directory
```

Try one of these solutions:

1. Ensure the DuckDB library is installed in a location on your library path
2. Set the appropriate environment variable (LD_LIBRARY_PATH on Linux, DYLD_LIBRARY_PATH on macOS)
3. Specify the exact path to the library using `WithDriverPath` as shown above
4. For Docker on ARM64, ensure you're creating symlinks in both `/usr/lib/` and `/usr/local/lib/` as shown in the ARM64 Dockerfile example

## License

[MIT License](LICENSE)
