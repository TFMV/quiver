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

## Benchmark Results

Below are the benchmark results for **Quiver** running on **Apple M2 Pro**:

| Benchmark                     | Iterations  | Time per Op (ns) | Memory per Op (B) | Allocations per Op |
|--------------------------------|-------------|------------------|--------------------|---------------------|
| **Vector Index - Add**         | 10,000      | 1,963,062 ns     | 4,476 B            | 23                  |
| **Vector Index - Search**      | 496,915     | 25,667 ns        | 856 B              | 14                  |
| **Vector Search**              | 1,576,177   | 7,590 ns         | 856 B              | 14                  |

## Observations

- **Vector Insertion:** Each vector addition takes **~2ms** with **23 allocations**. Optimizations might include **batch inserts** or **parallelization**.
- **Vector Search:** Achieves **~7.6µs per search**, which is **highly efficient** for ANN lookups. Memory allocation is low (856 B/op).
- **Overall Performance:** With nearly **1.5M+ queries per second**, **Quiver** performs well in high-throughput vector search scenarios.
- **Future Optimizations:** Exploring **lower memory overhead** and **faster inserts** using more efficient memory allocation strategies.

---

## Roadmap

- [ ] Hybrid Search: Combine structured queries with ANN search.
- [ ] Multi-Threaded Indexing: Improve insertion performance.
- [ ] Arrow IPC & Flight Support: Stream vectors efficiently.
- [ ] Metadata Filtering: Structured queries alongside vectors.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Authors

- [@TFMV](https://github.com/TFMV)
