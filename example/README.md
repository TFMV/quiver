# 🏹 Quiver Example: The Ultimate Guide

Welcome to the **Quiver Example** - your comprehensive guide to using Quiver, the blazingly fast vector database for Go! This example demonstrates all the powerful features that make Quiver the perfect choice for your vector search needs.

## 🚀 What's Inside?

This example is a **complete showcase** of Quiver's capabilities, including:

- **All three index types**: HNSW, Parquet, and Hybrid
- **Advanced search techniques**: Standard similarity search, negative examples, facet filtering, and metadata filtering
- **Batch operations**: Adding and deleting vectors in batches for maximum performance
- **Backup and restore**: Keeping your vector data safe
- **Analytics**: Understanding your vector database's performance
- **Adaptive Parameter Tuning (APT)**: Automatic optimization of HNSW parameters

## 🧠 Index Types Explained

Quiver offers three powerful index types, each with its own strengths:

### 1. HNSW Index

The classic HNSW (Hierarchical Navigable Small World) graph implementation. It's fast, memory-efficient, and offers a great balance of speed and recall.

```go
config := quiver.DefaultDBConfig()
config.Hybrid.Type = hybrid.HNSWIndexType
config.Hybrid.M = 16         // Number of connections per node
config.Hybrid.EfSearch = 100 // Query time search depth
```

### 2. Parquet Index

A persistent storage-backed HNSW implementation that efficiently stores vectors in Parquet format. Perfect for larger datasets that need durability.

```go
config := quiver.DefaultDBConfig()
config.Hybrid.Type = hybrid.HNSWIndexType
config.Parquet.Directory = "/path/to/parquet/data"
```

### 3. Hybrid Index

Our most advanced index type that combines multiple search strategies (HNSW, exact search, LSH) to optimize for both speed and recall, automatically selecting the best approach based on your data.

```go
config := quiver.DefaultDBConfig()
config.Hybrid.Type = hybrid.HybridIndexType
config.Hybrid.ExactThreshold = 1000 // Use exact search for datasets smaller than 1000 vectors
```

## 🔍 Search Techniques

### Basic Search

```go
queryVector := []float32{0.2, 0.3, 0.4, 0.5, 0.6}
options := db.DefaultQueryOptions().WithK(3) // Return top 3 results
results, err := db.Search(queryVector, options)
```

### Search with Negative Examples

Tell Quiver what you don't want to see in your results:

```go
options := db.DefaultQueryOptions().
    WithK(3).
    WithNegativeExample(negativeVector).
    WithNegativeWeight(0.7) // Higher weight gives more importance to avoiding negative examples
```

### Search with Facet Filters

Filter results based on facet attributes:

```go
options := db.DefaultQueryOptions().
    WithK(3).
    WithFacetFilters(facets.NewEqualityFilter("category", "finance"))
```

### Search with Metadata Filters

Filter results based on JSON metadata:

```go
metadataFilter := []byte(`{"year": 2023}`) // Only return documents from 2023
options := db.DefaultQueryOptions().
    WithK(3).
    WithMetadataFilter(metadataFilter)
```

## 📦 Batch Operations

Add vectors in batches for maximum performance:

```go
keys := []uint64{101, 102, 103}
vectors := [][]float32{...}
metadataList := []interface{}{...}
facetsList := [][]facets.Facet{...}

err := db.BatchAdd(keys, vectors, metadataList, facetsList)
```

Delete vectors in batches:

```go
keysToDelete := []uint64{101, 103}
deleteResults := db.BatchDelete(keysToDelete)
```

## 💾 Backup and Restore

Create a backup of your database:

```go
err := db.Backup("/path/to/backup")
```

Restore from a backup:

```go
err := restoredDB.Restore("/path/to/backup")
```

## 📊 Analytics

Get insights into your database's performance:

```go
stats := db.GetStats()
metrics, err := db.Analyze()
```

## 🧠 Adaptive Parameter Tuning (APT)

Quiver's APT system automatically optimizes HNSW parameters based on your workload patterns and performance metrics:

```go
// Check if APT is enabled
enabled := adaptive.IsEnabled()

// Get current parameters
params := adaptive.GetCurrentParameters()

// Get workload analysis
analysis := adaptive.GetWorkloadAnalysis()

// Get performance report
report := adaptive.GetPerformanceReport()

// Enable or disable APT
adaptive.SetEnabled(true)
```

## 🏃‍♂️ Running the Example

To run this example, simply execute:

```bash
go run main.go
```

The example will:

1. Initialize Quiver with APT enabled
2. Demonstrate all three index types
3. Show various search techniques
4. Demonstrate batch operations
5. Show backup and restore functionality
6. Display analytics
7. Demonstrate APT features

## 🔮 What's Next?

After exploring this example, you'll have a solid understanding of Quiver's capabilities. Ready to use Quiver in your own project? Check out our [documentation](../README.md) for more information!

Happy vector searching! 🏹
