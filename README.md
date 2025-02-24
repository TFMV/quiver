# Quiver

Blazing-Fast, Embeddable, Structured Vector Search in Go

[![Go Report Card](https://goreportcard.com/badge/github.com/TFMV/quiver)](https://goreportcard.com/report/github.com/TFMV/quiver)
[![GoDoc](https://pkg.go.dev/badge/github.com/TFMV/quiver)](https://pkg.go.dev/github.com/TFMV/quiver)

## 🏗 Architecture

Quiver is a lightweight, high-performance vector search engine designed for structured datasets that uses HNSW for efficient vector indexing and DuckDB for metadata storage.

```mermaid
flowchart TD
  A["Start: Initialize Quiver"]
  B["Validate Config"]
  C["Initialize HNSW Index"]
  D["Open DuckDB"]
  E["Create Metadata Table"]
  F["Index Object\n(config, hnsw, db, metadata)"]
  G["Start Batch Processor"]

  A --> B --> C --> D --> E --> F --> G

  %% Insert Flow
  H["Add Vector"]
  I["Validate Dimension & Append to Batch"]
  J{"Batch Full?"}
  K["Trigger Async Flush"]
  L["Flush Batch:\n- Store Vectors\n- Save Metadata\n- Insert into HNSW"]
  M["Wait for More Data"]

  F --> H --> I --> J
  J -- Yes --> K --> L
  J -- No --> M

  %% Search Flow
  N["Search"]
  O["Call HNSW.SearchKNN()"]
  P["Retrieve Metadata"]
  Q["Return Search Results"]

  F --> N --> O --> P --> Q

  %% Hybrid Search Flow
  R["Search with Filter"]
  S["Filter Metadata in DuckDB"]
  T{"Few Matches?"}
  U["Vector Search on Filtered IDs"]
  V["Return Results"]
  W["Full Vector Search + Metadata Filter"]

  F --> R --> S --> T
  T -- Yes --> U --> V
  T -- No --> W --> V
```

## 📦 Example Usage

```go
// Initialize Quiver
index, _ := quiver.New(quiver.Config{
    Dimension: 128, StoragePath: "data.db", MaxElements: 10000,
})

// Insert a vector
vector := []float32{0.1, 0.2, 0.3, ...}
index.Add(1, vector, map[string]interface{}{"category": "science"})

// Perform a search
results, _ := index.Search(vector, 5)
fmt.Println("Closest match:", results[0].ID, results[0].Metadata)

// Save the index to disk
index.Save("index.quiver")

// Load the index from disk
index, _ = quiver.Load("index.quiver")

// Perform a hybrid search with metadata filter
filteredResults, _ := index.SearchWithFilter(vector, 5, "category = 'science'")
fmt.Println("Filtered results:", filteredResults)

// Close the index
index.Close()
```

## 🌟 Features

- **High-Performance**: Utilizes HNSW for efficient vector indexing
- **Structured Data**: Supports metadata storage for structured datasets
- **Lightweight**: Minimal dependencies, easy to embed in your Go applications
- **Flexible**: Supports various distance metrics (e.g., Euclidean, Cosine)
- **Configurable**: Customize HNSW parameters for optimal performance

## :small_airplane: Performance

The following benchmarks were performed on a 2024 MacBook Pro with an M2 Pro CPU.

| Operation | Operations/sec | Latency | Memory (B/op) | Allocations (allocs/op) |
|-----------|---------------|--------------|---------------|------------------------|
| Add | 4,872 | 3.178ms | 1,389 | 21 |
| Search | 30,246 | 44.338µs | 1,520 | 18 |
| Hybrid Search | 2,395 | 431.622µs | 7,533 | 278 |
| Add Parallel | 2,850 | 6.088ms | 1,282 | 19 |
| Arrow Append | 100 | 2.770s | 2,336,470 | 32377 |

Benchmark details:

- Vector dimension: 128
- Dataset size: 10,000 vectors
- HNSW M: 32
- HNSW efConstruction: 200
- HNSW efSearch: 200
- Batch size: 1,000

Benchmark notes:

- Arrow Append is a one-time operation that pre-allocates memory for the entire dataset.
- The other benchmarks are for typical insert/search operations.
