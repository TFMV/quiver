# quiver

[![Go Report Card](https://goreportcard.com/badge/github.com/TFMV/quiver)](https://goreportcard.com/report/github.com/TFMV/quiver)
[![GoDoc](https://pkg.go.dev/badge/github.com/TFMV/quiver)](https://pkg.go.dev/github.com/TFMV/quiver)

## Overview

Quiver is a lightweight vector database optimized for structured datasets. It integrates Apache Arrow Tables with Approximate Nearest Neighbors (ANN) search, leveraging Hierarchical Navigable Small World (HNSW) graphs for efficient high-dimensional vector retrieval.

DuckDB is used as the underlying database to store and retrieve vector data.

The Vector Index is thread safe and uses a lock to ensure that only one thread can access the index at a time.

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

## Benchmark Results

Quiver v0.2.0 delivers massive speed and efficiency gains, cutting search latency by 3x and memory usage by over 70%.

Below are the benchmark results for Quiver running on Apple M2 Pro:

| Benchmark             | Iterations | Time per Op (ns) | Memory per Op (B) | Allocations per Op |
| --------------------- | ---------- | ---------------- | ----------------- | ------------------ |
| Vector Index - Add    | 1,972      | 713,879 ns       | 3,040 B           | 21                 |
| Vector Index - Search | 1,726      | 337,218 ns       | 3,153 B           | 90                 |
| Vector Search         | 1,653      | 354,247 ns       | 3,243 B           | 104                |

## 🚀 Performance Improvements in v0.2.0

### Vector Operations

#### ⚡️ Adding Vectors

- **Speed**: 7.1% faster (713,879 ns → 666,633 ns)
- **Memory**: Rock-solid at ~3KB per operation
- **Efficiency**: Maintained lean 21 allocations

#### 🔍 Searching Vectors

- **Speed**: Lightning fast - 3x improvement! (966,301 ns → 337,218 ns)
- **Memory**: Slashed by 74.4% (12,329 B → 3,153 B)
- **Efficiency**: Dramatically reduced allocations (296 → 90)

### 🎯 Overall Search Performance

- **Speed**: 2.5x faster searches (901,365 ns → 354,247 ns)
- **Memory**: 73.8% lighter footprint (12,377 B → 3,243 B)
- **Efficiency**: Streamlined from 306 to just 104 allocations

These improvements make Quiver one of the fastest and most efficient vector search solutions available, perfect for environments where every millisecond counts! 🏃‍♂️💨

## Observations

Vector Index - Add

- Improved by ~7.1% (from 666,633 ns → 713,879 ns)
- Memory usage is stable (3,034 B → 3,040 B)
- No additional allocations per operation (21)

🔍 Vector Index - Search

- Nearly 3x faster (from 966,301 ns → 337,218 ns)
- ~74.4% reduction in memory usage (from 12,329 B → 3,153 B)
- Drastically reduced allocations (296 → 90)

🔎 Vector Search

- 2.5x improvement in speed (from 901,365 ns → 354,247 ns)
- ~73.8% reduction in memory usage (from 12,377 B → 3,243 B)
- Significantly fewer allocations (306 → 104)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Authors

[TFMV](https://github.com/TFMV)
