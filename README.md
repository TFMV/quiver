# quiver

Vectorized Search for Structured Data

## Overview

Quiver is a lightweight vector database optimized for structured datasets. It integrates Apache Arrow Tables with Approximate Nearest Neighbors (ANN) search, leveraging Hierarchical Navigable Small World (HNSW) graphs for efficient high-dimensional vector retrieval.

DuckDB is used as the underlying database to store and retrieve vector data.

## Installation

```bash
go get github.com/TFMV/quiver
```

## Usage

### Initializing an Index

```go
index := quiver.NewVectorIndex(10000, 128) // maxElements, vectorDim
```

### Adding Vectors

```go
vector := []float32{0.1, 0.2, 0.3, ..., 0.128}
index.Add(1, vector) // ID, vector
```

### Searching for Nearest Neighbors

```go
query := []float32{0.15, 0.25, 0.35, ..., 0.128}
k := 5
results := index.Search(query, k)
```

### Saving and Loading the Index

```go
index.Save()
loadedIndex := quiver.Load("index.hnsw")
```

## Roadmap

- [ ] Hybrid Search: Combine structured queries with ANN search.
- [ ] Multi-Threaded Indexing: Improve insertion performance.
- [ ] Arrow IPC & Flight Support: Stream vectors efficiently.
- [ ] Metadata Filtering: Structured queries alongside vectors.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Authors

- [@TFMV](https://github.com/TFMV)
