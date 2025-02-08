# Quiver: Blazing-Fast, Embeddable Vector Search in Go

[![Go Report Card](https://goreportcard.com/badge/github.com/TFMV/quiver)](https://goreportcard.com/report/github.com/TFMV/quiver)
[![GoDoc](https://pkg.go.dev/badge/github.com/TFMV/quiver)](https://pkg.go.dev/github.com/TFMV/quiver)

## 🚀 Overview

Quiver is a lightweight, high-performance vector search engine designed for structured datasets.
It seamlessly integrates Apache Arrow Tables with Approximate Nearest Neighbors (ANN) search, leveraging Hierarchical Navigable Small World (HNSW) graphs for ultra-fast high-dimensional vector retrieval.

- DuckDB-powered storage ensures persistence without sacrificing speed
- Thread-safe vector indexing ensures high concurrency with minimal overhead
- Embedded, no external dependencies – pure Go, blazing fast

## 📦 Installation

```bash
go get github.com/TFMV/quiver
```

## 🔧 Usage

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

### Using the Arrow Appender

The ArrowAppender is a zero-allocation, high-performance appender for Apache Arrow tables. It is the fastest way to get your vector data into Quiver.

```go
// Create a schema for your data
schema := arrow.NewSchema([]arrow.Field{
    {Name: "id", Type: arrow.PrimitiveTypes.Int64},
    {Name: "vector", Type: arrow.ListOf(arrow.PrimitiveTypes.Float32)},
}, nil)

// Initialize the appender
appender, err := appender.NewArrowAppender(schema, logger)
if err != nil {
    log.Fatal(err)
}
defer appender.Close()

// Append rows efficiently
appender.AppendRow([]interface{}{
    int64(1),
    []float32{0.1, 0.2, 0.3},
})

// Flush batch when ready
rows, err := appender.FlushBatch()
if err != nil {
    log.Fatal(err)
}

// Get final Arrow buffer
data, err := appender.Flush()
if err != nil {
    log.Fatal(err)
}
```

> Zero-allocation batch operations with Arrow's efficient memory layout

## ⚡ Performance Benchmarks

All benchmarks were run on an Apple M2 Pro.

### 🔥 Vector Search Performance

| Operation | Speed | Memory | Allocations |
|-----------|-------|---------|-------------|
| Vector Search | **354μs** | 3.2KB | 104 |
| Index Search | **337μs** | 3.1KB | 90 |

> Blazing-fast vector search with minimal memory footprint – built for performance-critical workloads.

### 📝 Arrow Appender Performance

| Operation | Speed | Memory | Allocations |
|-----------|-------|---------|-------------|
| Append Row | **33ns** | 72B | 0 |
| Flush Batch | **14ns** | 0B | 0 |
| Full Flush | **56ns** | 40B | 2 |

> Zero-allocation appending with microsecond-level batch operations.

## 💡 Why Quiver?

✅ Embedded & lightweight – No external database required  
✅ HNSW-powered ANN search – No brute-force nonsense  
✅ DuckDB-backed storage – Persistence without performance overhead  
✅ Optimized for high throughput – Millions of queries per second  
✅ Designed for performance-critical workloads – 3KB memory footprint per search

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🔗 Get Started

Clone the repo and start experimenting with Quiver today:

```bash
git clone https://github.com/TFMV/quiver.git
cd quiver
go run example.go
```

## Author

Quiver is developed by [TFMV](https://github.com/TFMV).
