# quiver

Vectorized Search for Structured Data

It is thread safe and uses a lock to ensure that only one thread can access the index at a time.

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
| Vector Index - Add            | 2,115       | 739,556 ns       | 4,430 B           | 22                |
| Vector Index - Search          | 645         | 945,968 ns       | 12,328 B          | 297               |
| Vector Search                 | 663         | 976,710 ns       | 12,378 B          | 306               |

## Observations

Vector Insertion:

- Each vector addition takes ~740µs, with 22 allocations per op.
- Optimization Potential: Further batch inserts or parallelized writes could improve throughput.

Vector Search:

- Search operations take ~945µs per lookup, with 297+ allocations per op.
- The relatively high memory usage suggests optimization opportunities in query execution, possibly via pre-fetching or caching.

Overall Performance:

- With search speeds averaging ~1ms per query, Quiver is highly efficient for Approximate Nearest Neighbor (ANN) search.
- Memory footprint and allocation count indicate areas where further tuning could yield faster search times.

Future Optimizations:

- Reducing memory overhead via improved buffer reuse.
- Index build time optimizations, possibly by precomputing hierarchical navigable structures.
- Fine-tuning concurrency settings to balance index updates vs. search latency.

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
