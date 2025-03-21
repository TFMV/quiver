# 🏹 Quiver

> **Note:** Quiver is an experimental vector database. While it's packed with cool features, it's still finding its feet in the world. Feel free to play with it, but maybe don't bet your production system on it just yet!

## 🎉 Announcing APT 🎉

We're thrilled to introduce Adaptive Parameter Tuning **(APT)**, a feature that automatically optimizes HNSW parameters based on your workload patterns and performance metrics. No more manual tuning!

APT continuously analyzes your query patterns, monitors performance, and intelligently adjusts parameters to deliver optimal performance for your specific use case. It's like having a database expert constantly fine-tuning your vector index.

Read all about it [here](adaptive/README.md)

## What is Quiver?

Quiver is a Go-based vector database that combines the best of HNSW (Hierarchical Navigable Small World) graphs with other cool search techniques.

## Supported Index Types

Quiver offers three powerful index types to suit different use cases:

1. **HNSW Index**: The classic HNSW (Hierarchical Navigable Small World) graph implementation. This in-memory index offers a great balance of speed and recall for most use cases. It's fast, memory-efficient, and perfect for medium-sized datasets.

2. **HNSW with Parquet Storage**: This is not a separate index type, but rather an HNSW index backed by Parquet storage. It efficiently persists vectors to disk in Parquet format, making it ideal for larger datasets that need durability while maintaining the performance characteristics of HNSW.

3. **Hybrid Index**: Our most advanced index type that combines multiple search strategies to optimize for both speed and recall. It can automatically select between exact search (for small datasets) and approximate search (for larger datasets), and includes optimizations for different query patterns. The hybrid index is particularly effective for datasets with varying sizes and query patterns.

All index types support the full range of Quiver features, including faceted search, metadata filtering, and negative examples. Choose the right index type for your needs and let APT optimize your parameters automatically!

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
- **🧠 Adaptive Tuning**: APT automatically optimizes parameters based on your workload

## What can you do with Quiver?

Quiver makes it easy to form complex queries for any of our index types.

```go
func SearchWithComplexOptions(database *db.VectorDB[uint64]) {
   log.Println("\nPerforming search with complex options...")

    // Create a query vector
    queryVector := []float32{0.2, 0.3, 0.4, 0.5, 0.6}

    // Create a negative example vector
    negativeVector := []float32{0.9, 0.8, 0.7, 0.6, 0.5}

    // Create a facet filter for active items
    activeFilter := facets.NewEqualityFilter("is_active", true)

    // Create a metadata filter for vectors with score > 0.5
    metadataFilter := []byte(`{"score": {"$gt": 0.5}}`)

    // Create search options with all filters
    options := database.DefaultQueryOptions().
        WithK(5).
        WithNegativeExample(negativeVector).
        WithNegativeWeight(0.5).
        WithFacetFilters(activeFilter).
        WithMetadataFilter(metadataFilter)

    // Perform the search
    results, err := database.Search(queryVector, options)
    if err != nil {
       log.Printf("Search with complex options failed: %v", err)
       return
    }

    // Display results
   log.Printf("Found %d results (with complex options):", len(results))
   for i, result := range results {
     log.Printf("  Result %d: Key=%d, Distance=%f", i+1, result.Key, result.Distance)
   }
}
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
   - Run `OptimizeStorage` after large batch operations

4. **Be Smart About Backups**:
   - Schedule backups during quiet times
   - Keep backups in a separate location
   - Test your restore process regularly

5. **Delete with Caution**:
   - Deleting vectors can degrade graph quality
   - Consider marking vectors as inactive instead of deleting them
   - If you must delete, run `OptimizeStorage` afterward

## License

Quiver is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Happy vector searching. 🏹
