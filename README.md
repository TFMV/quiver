# 🏹 Quiver

> **Note:** Quiver is an experimental vector database. While it's packed with cool features, it's still finding its feet in the world. Feel free to play with it, but maybe don't bet your production system on it just yet!

## What is Quiver?

Quiver is a Go-based vector database that combines the best of HNSW (Hierarchical Navigable Small World) graphs with other cool search techniques.

## Why Choose Quiver?

- **🔍 Smart Search Strategy**: Quiver doesn't just use one search method - it combines HNSW with exact search, LSH, and data partitioning to find the best results
- **🏷️ Rich Metadata**: Attach JSON metadata to your vectors and filter search results based on it
- **🧩 Faceted Search**: Filter results based on attributes (we call them facets)
- **👎 Negative Examples**: Tell Quiver what you don't want to see in your results
- **📊 Analytics**: Peek under the hood with graph quality and performance metrics
- **💾 Durability**: Your data stays safe with Parquet-based storage
- **📦 Backup & Restore**: Create snapshots of your database and bring them back when needed
- **⚡ Batch Operations**: Add, update, and delete vectors in batches for lightning speed
- **🔗 Fluent Query API**: Write queries that read like plain English
- **🚀 Performance**: Quiver is built for speed without sacrificing accuracy
- **😌 Easy to Use**: Our fluent API just makes sense

## Getting Started

Let's create a simple vector database with Quiver:

```go
package main

import (
 "fmt"
 "log"

 "github.com/TFMV/hnsw"
 "github.com/TFMV/hnsw/hnsw-extensions/db"
 "github.com/TFMV/hnsw/hnsw-extensions/facets"
 "github.com/TFMV/hnsw/hnsw-extensions/hybrid"
)

func main() {
 // Create a database configuration
 config := db.DefaultDBConfig()
 config.BaseDir = "my_quiver_db"
 config.Hybrid.Type = hybrid.HybridIndexType
 config.Hybrid.ExactThreshold = 1000 // Use exact search for datasets smaller than 1000 vectors

 // Create a Quiver database
 quiverDB, err := db.NewVectorDB[string](config)
 if err != nil {
  log.Fatalf("Failed to create Quiver database: %v", err)
 }
 defer quiverDB.Close()

 // Add vectors with metadata and facets
 vector1 := []float32{0.1, 0.2, 0.3}
 metadata1 := map[string]interface{}{
  "name": "Document 1",
  "tags": []string{"finance", "report"},
  "year": 2023,
 }
 facets1 := []facets.Facet{
  facets.NewBasicFacet("category", "finance"),
  facets.NewBasicFacet("year", 2023),
 }

 if err := quiverDB.Add("doc1", vector1, metadata1, facets1); err != nil {
  log.Fatalf("Failed to add vector: %v", err)
 }

 // Add more vectors
 vector2 := []float32{0.2, 0.3, 0.4}
 metadata2 := map[string]interface{}{
  "name": "Document 2",
  "tags": []string{"technology", "report"},
  "year": 2022,
 }
 facets2 := []facets.Facet{
  facets.NewBasicFacet("category", "technology"),
  facets.NewBasicFacet("year", 2022),
 }

 if err := quiverDB.Add("doc2", vector2, metadata2, facets2); err != nil {
  log.Fatalf("Failed to add vector: %v", err)
 }

 fmt.Println("Added vectors to Quiver! 🎯")
}
```

## Searching with Quiver

Quiver makes searching a breeze with its fluent API:

```go
// Create query options using the fluent API
options := quiverDB.DefaultQueryOptions().
    WithK(5).                                                // Return top 5 results
    WithFacetFilters(facets.NewEqualityFilter("category", "finance")).  // Add facet filter
    WithNegativeExample([]float32{0.9, 0.8, 0.7}).          // Example of what we don't want
    WithNegativeWeight(0.7).                                 // Higher weight gives more importance to avoiding negative examples
    WithContext(context.Background())                        // Add context for cancellation

// Perform search
query := []float32{0.15, 0.25, 0.35}
results, err := quiverDB.Search(query, options)
if err != nil {
 log.Fatalf("Search failed: %v", err)
}

// Process results
fmt.Printf("Found %d results! 🎉\n", len(results))
for i, result := range results {
 fmt.Printf("Result %d: Key=%v, Distance=%f\n", i+1, result.Key, result.Distance)
 
 // Access metadata
 var metadata map[string]interface{}
 if err := json.Unmarshal(result.Metadata, &metadata); err == nil {
  fmt.Printf("  Name: %s\n", metadata["name"])
  fmt.Printf("  Tags: %v\n", metadata["tags"])
 }
 
 // Access facets
 for _, facet := range result.Facets {
  fmt.Printf("  %s: %v\n", facet.Name(), facet.Value())
 }
}
```

## Batch Operations

Quiver loves efficiency. Use batch operations to speed things up:

```go
// Batch add vectors
keys := []string{"doc3", "doc4", "doc5"}
vectors := [][]float32{
 {0.3, 0.4, 0.5},
 {0.4, 0.5, 0.6},
 {0.5, 0.6, 0.7},
}
metadataList := []interface{}{
 map[string]interface{}{"name": "Document 3", "category": "finance"},
 map[string]interface{}{"name": "Document 4", "category": "technology"},
 map[string]interface{}{"name": "Document 5", "category": "healthcare"},
}

// Create facets for each vector
facetsList := [][]facets.Facet{
 {facets.NewBasicFacet("category", "finance")},
 {facets.NewBasicFacet("category", "technology")},
 {facets.NewBasicFacet("category", "healthcare")},
}

// Add vectors in batch
if err := quiverDB.BatchAdd(keys, vectors, metadataList, facetsList); err != nil {
 log.Fatalf("Batch add failed: %v", err)
}

// Batch delete vectors
keysToDelete := []string{"doc1", "doc3"}
results := quiverDB.BatchDelete(keysToDelete)
for i, deleted := range results {
 fmt.Printf("Deleted %s: %v\n", keysToDelete[i], deleted)
}

// Optimize storage after batch operations
if err := quiverDB.OptimizeStorage(); err != nil {
 log.Fatalf("Failed to optimize storage: %v", err)
}
```

## Backup and Restore

Keep your data safe with Quiver's backup and restore features:

```go
// Create a backup of the database
backupDir := "./quiver_backup"
if err := quiverDB.Backup(backupDir); err != nil {
 log.Fatalf("Failed to create backup: %v", err)
}
fmt.Printf("Backup created at %s! 📦\n", backupDir)

// Create a new database instance for restoration
restoredDB, err := db.NewVectorDB[string](db.DefaultDBConfig())
if err != nil {
 log.Fatalf("Failed to create new database: %v", err)
}
defer restoredDB.Close()

// Restore from backup
if err := restoredDB.Restore(backupDir); err != nil {
 log.Fatalf("Failed to restore from backup: %v", err)
}
fmt.Println("Database restored successfully! 🎉")

// Verify restoration by performing a search
query := []float32{0.15, 0.25, 0.35}
options := restoredDB.DefaultQueryOptions().WithK(5)
results, err := restoredDB.Search(query, options)
if err != nil {
 log.Fatalf("Search failed: %v", err)
}
fmt.Printf("Found %d results in restored database! 🔍\n", len(results))
```

## Analytics

Curious about how your database is performing? Quiver has you covered:

```go
// Get database statistics
stats := quiverDB.GetStats()
fmt.Printf("Total vectors: %d\n", stats.VectorCount)
fmt.Printf("Total queries: %d\n", stats.TotalQueries)
fmt.Printf("Successful queries: %d\n", stats.SuccessfulQueries)
fmt.Printf("Failed queries: %d\n", stats.FailedQueries)
fmt.Printf("Average query time: %v\n", stats.AverageQueryTime)

// Get graph quality metrics (only available for HNSW graphs)
metrics, err := quiverDB.Analyze()
if err != nil {
 log.Printf("Analytics not available: %v", err)
} else {
 fmt.Printf("Node count: %d\n", metrics.NodeCount)
 fmt.Printf("Average connectivity: %.2f\n", metrics.AvgConnectivity)
 fmt.Printf("Connectivity std dev: %.2f\n", metrics.ConnectivityStdDev)
 fmt.Printf("Distortion ratio: %.2f\n", metrics.DistortionRatio)
 fmt.Printf("Layer balance: %.2f\n", metrics.LayerBalance)
 fmt.Printf("Graph height: %d\n", metrics.GraphHeight)
}
```

## Configuration Options

Quiver is flexible. Customize it to fit your needs:

```go
type DBConfig struct {
 // Base directory for storage
 BaseDir string

 // Hybrid index configuration
 Hybrid hybrid.IndexConfig

 // Parquet storage configuration (if using parquet)
 Parquet parquet.ParquetStorageConfig

 // Cache configuration
 CacheSize int // Maximum number of vectors to cache in memory

 // Query execution configuration
 MaxConcurrentQueries int // Maximum number of concurrent queries
 QueryTimeout         time.Duration

 // Analytics configuration
 EnableAnalytics bool // Whether to collect and store analytics data
}
```

## Tips for Best Performance

1. **Choose the Right Index Type**:
   - Small dataset (<1000 vectors)? Use `ExactIndexType` for perfect recall
   - Medium dataset? Use `HNSWIndexType` for a good balance
   - Large dataset? Use `HybridIndexType` for optimal performance

2. **Tune Your EfSearch Parameter**:
   - Higher values = more accurate but slower
   - Lower values = faster but potentially less accurate
   - Start with the default and adjust based on your needs

3. **Use Batch Operations**:
   - Always prefer `BatchAdd` over multiple `Add` calls
   - Same goes for `BatchDelete` vs multiple `Delete` calls
   - Run `OptimizeStorage` after large batch operations

4. **Be Smart About Backups**:
   - Schedule backups during quiet times
   - Keep backups in a separate location
   - Test your restore process regularly

5. **Delete with Caution**:
   - Deleting vectors can degrade graph quality
   - Consider marking vectors as inactive instead of deleting them
   - If you must delete, run `OptimizeStorage` afterward

## What's Coming Next?

Quiver is just getting started! Here's what we're working on:

1. **Distributed Search**: Search across multiple nodes
2. **Query Caching**: Speed up repeated queries
3. **Incremental Updates**: Update your index more efficiently
4. **Advanced Filtering**: More powerful filtering capabilities
5. **Vector Compression**: Reduce memory usage
6. **Cloud Storage**: Store your vectors in the cloud
7. **Incremental Backups**: Save space with smarter backups
8. **Streaming Updates**: Update your index in real-time
9. **DuckDB Integration**: Persist your vector data in DuckDB.

## License

Quiver is licensed under the MIT License - see the LICENSE file for details.

Happy vector searching. 🏹
