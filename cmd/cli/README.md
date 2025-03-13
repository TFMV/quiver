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
