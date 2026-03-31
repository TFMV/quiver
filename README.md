# 🏹 Quiver

## What is Quiver?

Quiver is a Go-based vector database that combines the best of HNSW (Hierarchical Navigable Small World) graphs with other cool search techniques. It provides efficient similarity search capabilities while maintaining a clean, easy-to-use API.

## 🙋 Why I Built Quiver

I didn't create Quiver for production use. It's a learning project—my way of exploring the internals of vector databases and sharing what I've learned. It's also a toy.

If you're curious about how vector search works under the hood, or if you want a foundation to build your own system, feel free to fork or clone it. I've kept it small and modular to make that easier.

Accompanying write-ups are available on Medium.

## Supported Index Types

Quiver offers three powerful index types, each of which can be backed by durable storage:

1. **HNSW Index**: The classic HNSW (Hierarchical Navigable Small World) graph implementation. This in-memory index offers a great balance of speed and recall for most use cases. It's fast, memory-efficient, and perfect for medium-sized datasets.

2. **Hybrid Index**: Our most advanced index type that combines multiple search strategies to optimize for both speed and recall. It can automatically select between exact search (for small datasets) and approximate search (for larger datasets), and includes optimizations for different query patterns. The hybrid index is particularly effective for datasets with varying sizes and query patterns.

3. **Arrow HNSW Index**: An Apache Arrow backed variant of HNSW for zero-copy columnar storage and optional Parquet persistence.

All index types can be backed by **Parquet Storage**, which efficiently persists vectors to disk in Parquet format. This makes them suitable for larger datasets that need durability while maintaining good performance characteristics.

All index types support metadata filtering and negative examples. Choose the right index type for your needs and let APT optimize your parameters automatically!

## Why Choose Quiver?

- **🚀 Performance**: Quiver is built for speed without sacrificing accuracy
- **🔍 Smart Search Strategy**: Quiver doesn't just use one search method - it combines HNSW with exact search to find the best results
- **😌 Easy to Use**: Our fluent API just makes sense
- **🔗 Fluent Query API**: Write queries that read like plain English
- **🏷️ Rich Metadata**: Attach JSON metadata to your vectors and filter search results based on it
- **🏎️ Faceted Search**: High-performance categorical filtering for your vectors
- **👎 Negative Examples**: Tell Quiver what you don't want to see in your results
- **⚡ Batch Operations**: Add, update, and delete vectors in batches for lightning speed
- **💾 Durability**: Your data stays safe with Parquet-based storage with WAL
- **📊 Observability**: Structured logging, Prometheus metrics (p50/p95/p99 latency), tracing
- **📦 Backup & Restore**: Create snapshots of your database and bring them back when needed

## Quick Start

```go
package main

import (
    "fmt"
    "log"

    "github.com/TFMV/quiver/pkg/core"
    "github.com/TFMV/quiver/pkg/vectortypes"
)

func main() {
    // Create database with default options
    opts := core.DefaultDBOptions()
    opts.StoragePath = "./data"
    opts.EnablePersistence = true

    db, err := core.NewDB(opts)
    if err != nil {
        log.Fatalf("Failed to create DB: %v", err)
    }
    defer db.Close()

    // Create collection with cosine distance
    collection, err := db.CreateCollection("products", 128, vectortypes.CosineDistance)
    if err != nil {
        log.Fatalf("Failed to create collection: %v", err)
    }

    // Add vectors with metadata
    err = collection.Add("product-1", []float32{0.1, 0.2, /* ... */}, map[string]string{
        "category": "electronics",
        "price": "299.99",
    })
    if err != nil {
        log.Fatalf("Failed to add vector: %v", err)
    }

    // Or use batch insert for better performance
    vectors := []vectortypes.Vector{
        {ID: "p2", Values: []float32{0.5, 0.6, /* ... */}},
        {ID: "p3", Values: []float32{0.7, 0.8, /* ... */}},
    }
    err = collection.AddBatch(vectors)
    if err != nil {
        log.Fatalf("Failed to batch add: %v", err)
    }

    // Search with the fluent API
    results, err := collection.FluentSearch(queryVector).
        WithK(10).
        Filter("category", "electronics").
        Execute()
    if err != nil {
        log.Fatalf("Search failed: %v", err)
    }

    fmt.Printf("Found %d results\n", len(results.Results))
}
```

## Fluent API

The fluent API provides chainable methods for building complex queries:

```go
results, err := collection.FluentSearch(queryVector).
    WithK(10).
    Filter("category", "electronics").
    FilterIn("tags", []interface{}{"smartphone", "5G"}).
    FilterGreaterThan("price", 100.0).
    FilterLessThan("price", 500.0).
    IncludeVectors(true).
    Execute()
```

The fluent API validates inputs early:
- Query vector dimension is checked at creation time
- K values are validated before execution
- Empty filter fields are rejected during building

## Public API Safety

Quiver's public API is designed to prevent invalid states:

- **Input Validation**: All operations validate inputs (dimensions, IDs, K values)
- **Clear Errors**: Returns specific errors instead of silent failures
- **Immutable Builders**: FluentSearch validates at build time and fails fast

### Error Types

```go
var (
    ErrCollectionNotFound   = errors.New("collection not found")
    ErrCollectionExists      = errors.New("collection already exists")
    ErrInvalidDimension     = errors.New("invalid vector dimension")
    ErrVectorNotFound       = errors.New("vector not found")
    ErrVectorAlreadyExist   = errors.New("vector with the same ID already exists")
    ErrInvalidMetadata      = errors.New("invalid metadata format")
)
```

## Observability

Quiver includes production-grade observability:

### Metrics (Prometheus)

```go
import "github.com/TFMV/quiver/pkg/observability"

// Enable metrics
metrics := observability.GlobalMetrics()
metrics.SetEnabled(true)

// Access Prometheus registry
reg := metrics.Registry()
```

Available metrics:
- `quiver_search_latency_ms` - Search latency by collection/stage (traversal/filter/total)
- `quiver_insert_latency_ms` - Insert latency by collection/operation
- `quiver_batch_latency_ms` - Batch operation latency
- `quiver_search_total` - Total searches by collection
- `quiver_index_vectors` - Vector count by collection

### In-Memory Percentiles

```go
// Get p50/p95/p99 for any operation
p99 := metrics.GetLatencyPercentiles("search")
fmt.Printf("Search p99: %.2fms\n", p99.P99)
```

### Structured Logging

```go
import "github.com/TFMV/quiver/pkg/observability"

// Configure logging
logger := observability.NewLogger(os.Stdout, observability.LevelInfo)
observability.SetDefault(logger)

// Use structured logging
observability.Info("search completed", 
    "collection", "products",
    "results", 10,
    "duration_ms", 45.2)
```

### Tracing

```go
// Enable tracing
tracer := observability.DefaultTracer()
tracer.SetEnabled(true)

// Track query stages
span := observability.StartSpan(ctx, "search")
defer observability.EndSpan(span)
span.SetAttr("collection", "products")
```

## Filtering with Metadata

Quiver supports powerful filtering capabilities through metadata:

```go
// Search with metadata filters
results, err := collection.Search(types.SearchRequest{
    Vector: queryVector,
    TopK: 10,
    Filters: []types.Filter{
        {Field: "category", Operator: "=", Value: "electronics"},
        {Field: "price", Operator: "<", Value: 500},
    },
})
```

## Faceted Search

For high-performance categorical filtering:

```go
// Set up facet fields
collection.SetFacetFields([]string{"category", "price_range"})

// Search with facets
results, err := collection.SearchWithFacets(queryVector, 10, []facets.Filter{
    facets.NewEqualityFilter("category", "electronics"),
})
```

## Tips for Best Performance

1. **Choose the Right Index Type**:
   - Small dataset (<1000 vectors)? Use the Hybrid index with a low ExactThreshold for perfect recall
   - Medium dataset? Use the HNSW index for a good balance of speed and recall
   - Large dataset with durability needs? Use HNSW with Parquet storage
   - Complex workloads with varying query patterns? Use the Hybrid index

2. **Tune Your Parameters** (or let APT do it for you):
   - M: Controls the number of connections per node (higher = more accurate but more memory)
   - EfSearch: Controls search depth (higher = more accurate but slower)
   - EfConstruction: Controls index build quality (higher = better index but slower construction)

3. **Use Batch Operations**:
   - Always prefer `BatchAdd` over multiple `Add` calls
   - Same goes for `BatchDelete` vs multiple `Delete` calls

4. **Enable Observability**:
   - Track latency percentiles to identify bottlenecks
   - Use structured logs for debugging production issues

5. **Delete with Caution**:
   - Deleting vectors can degrade graph quality
   - Consider marking vectors as inactive instead of deleting them

## License

Quiver is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Happy vector searching. 🏹