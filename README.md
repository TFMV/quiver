# quiver

Vectorized Search for Structured Data

It is thread safe and uses a lock to ensure that only one thread can access the index at a time.

![logo](assets/quiver.png)

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

Below are the benchmark results for Quiver running on Apple M2 Pro:

| Benchmark                     | Iterations  | Time per Op (ns) | Memory per Op (B) | Allocations per Op |
|--------------------------------|-------------|------------------|--------------------|---------------------|
| Vector Index - Add             | 1,797       | 666,633 ns       | 3,034 B            | 21                  |
| Vector Index - Search           | 561         | 966,301 ns       | 12,329 B           | 296                 |
| Vector Search                  | 660         | 901,365 ns       | 12,377 B           | 306                 |

## Observations

### Vector Insertion:

- Now ~0.67ms per vector, down from ~2ms, with lower memory usage (3,034 B/op).
- Optimizations in batching and memory efficiency have significantly reduced overhead.

### Vector Search:

- Query times remain under 1ms, making Quiver highly efficient for high-throughput ANN lookups.
- Memory allocations remain stable, but there is potential for additional optimizations in search caching.

---

## Roadmap

- [ ] Hybrid Search: Combine structured queries with ANN search.
- [ ] Multi-Threaded Indexing: Improve insertion performance.
- [ ] Arrow IPC & Flight Support: Stream vectors efficiently.
- [ ] Metadata Filtering: Structured queries alongside vectors.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Authors

[TFMV](https://github.com/TFMV)
