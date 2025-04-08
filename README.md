# 🏹 Quiver

> **Note:** Quiver is an experimental vector database. While it's packed with cool features, it's still finding its feet in the world. Feel free to play with it, but maybe don't bet your production system on it just yet!

## What is Quiver?

Quiver is a Go-based vector database that combines the best of HNSW (Hierarchical Navigable Small World) graphs with other cool search techniques. It provides efficient similarity search capabilities while maintaining a clean, easy-to-use API.

## Supported Index Types

Quiver offers two powerful index types, both of which can be backed by durable storage:

1. **HNSW Index**: The classic HNSW (Hierarchical Navigable Small World) graph implementation. This in-memory index offers a great balance of speed and recall for most use cases. It's fast, memory-efficient, and perfect for medium-sized datasets.

2. **Hybrid Index**: Our most advanced index type that combines multiple search strategies to optimize for both speed and recall. It can automatically select between exact search (for small datasets) and approximate search (for larger datasets), and includes optimizations for different query patterns. The hybrid index is particularly effective for datasets with varying sizes and query patterns.

Both index types can be backed by **Parquet Storage**, which efficiently persists vectors to disk in Parquet format. This makes them suitable for larger datasets that need durability while maintaining good performance characteristics.

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
- **💾 Durability**: Your data stays safe with Parquet-based storage
- **📊 Analytics**: Peek under the hood with graph quality and performance metrics
- **📦 Backup & Restore**: Create snapshots of your database and bring them back when needed

## What can you do with Quiver?

Quiver makes it easy to form complex queries for any of our index types.

```go
func SearchWithComplexOptions(db *quiver.DB) {
    log.Println("\nPerforming search with complex options...")

    // Get a collection
    collection, err := db.GetCollection("products")
    if err != nil {
        log.Fatalf("Failed to get collection: %v", err)
    }

    // Create a query vector
    queryVector := []float32{0.2, 0.3, 0.4, 0.5, 0.6}
    
    // Use the fluent API for building a complex search query
    // with both facets and negative examples
    results, err := collection.FluentSearch(queryVector).
        WithK(10).
        WithNegativeExample([]float32{0.9, 0.8, 0.7, 0.6, 0.5}).
        WithNegativeWeight(0.3).
        Filter("category", "electronics").
        FilterIn("tags", []interface{}{"smartphone", "5G"}).
        FilterGreaterThan("price", 100.0).
        FilterLessThan("price", 500.0).
        Execute()
    
    if err != nil {
        log.Printf("Search with complex options failed: %v", err)
        return
    }

    // Display results
    log.Printf("Found %d results (with complex options):", len(results.Results))
    for i, result := range results.Results {
        log.Printf("  Result %d: ID=%s, Distance=%f", i+1, result.ID, result.Distance)
        if result.Metadata != nil {
            log.Printf("    Metadata: %s", string(result.Metadata))
        }
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

6. **Use Facets for High-Performance Filtering**:
   - For categorical data that you frequently filter on, use facets instead of metadata filtering
   - Set up facet fields early when creating your collections
   - Use facet filtering for categories, tags, and other discrete attributes

## Filtering with Metadata

Quiver supports powerful filtering capabilities through metadata. You can attach arbitrary JSON metadata to your vectors and then filter search results based on this metadata.

```go
// Add a vector with metadata
metadata := map[string]interface{}{
    "category": "electronics",
    "price": 299.99,
    "in_stock": true,
    "tags": []string{"smartphone", "android", "5G"}
}
collection.Add(id, vector, metadata)

// Later, search with a metadata filter
filter := []byte(`{
    "category": "electronics",
    "price": {"$lt": 500},
    "in_stock": true,
    "tags": {"$contains": "5G"}
}`)

results, err := collection.Search(types.SearchRequest{
    Vector: queryVector,
    TopK: 10,
    Filters: []types.Filter{
        {Field: "category", Operator: "=", Value: "electronics"},
        {Field: "price", Operator: "<", Value: 500},
        {Field: "in_stock", Operator: "=", Value: true},
        {Field: "tags", Operator: "contains", Value: "5G"},
    },
})
```

## Faceted Search

Quiver offers an optimized filtering mechanism called faceted search. Facets enable high-performance filtering for categorical data, which is much faster than regular metadata filtering.

### What are Facets?

Facets are precomputed, indexed categorical attributes that allow for rapid filtering. They're ideal for attributes like:

- Product categories
- Tags
- Status values (active/inactive)
- Geographic regions
- Price ranges
- Content types

### Using Facets

```go
// Specify which fields should be indexed as facets
collection.SetFacetFields([]string{"category", "price_range", "tags"})

// Add vectors with metadata that includes facet fields
metadata := map[string]interface{}{
    "category": "electronics",
    "price_range": "200-500",
    "tags": ["smartphone", "android", "5G"],
    "other_field": "this won't be indexed as a facet"
}
collection.Add(id, vector, metadata)

// Search with facet filters
filters := []facets.Filter{
    facets.NewEqualityFilter("category", "electronics"),
    facets.NewEqualityFilter("tags", "5G"),
}

results, err := collection.SearchWithFacets(queryVector, 10, filters)
```

### Benefits of Facets vs. Metadata Filtering

1. **Performance**: Facets are precomputed and indexed, making filtering operations much faster
2. **Memory Efficiency**: Facet values are stored in an optimized format
3. **Type Safety**: Facets provide type-aware filtering with proper comparisons
4. **Range Queries**: Built-in support for numeric range queries
5. **Set Operations**: Easily filter based on membership in sets

### When to Use Facets

Use facets when:

- You frequently filter on the same fields
- Your filtering needs are primarily categorical
- Performance is critical for your application
- You're working with large datasets

Use metadata filtering when:

- Your filtering needs are ad-hoc or unpredictable
- You need complex query expressions
- You're filtering on fields that change frequently

## Using Negative Examples

Negative examples allow you to steer search results away from specific concepts or characteristics. This is useful when you want to find vectors similar to your query but dissimilar from certain examples.

### How Negative Examples Work

When you provide a negative example vector, Quiver:

1. Performs the standard search to find candidates similar to your query vector
2. Calculates the similarity between each candidate and the negative example
3. Re-ranks results to prefer candidates that are less similar to the negative example
4. Returns the final, adjusted results

### Using Negative Examples with the Standard API

```go
// Create a search request with a negative example
request := types.SearchRequest{
    Vector: queryVector,
    TopK: 10,
    NegativeExample: negativeVector,
    NegativeWeight: 0.5, // 0.5 gives equal importance to positive and negative examples
}

// Execute the search
response, err := collection.Search(request)
```

### Using Negative Examples with the Fluent API

```go
// Using the fluent API for a search with a negative example
results, err := collection.FluentSearch(queryVector).
    WithK(10).
    WithNegativeExample(negativeVector).
    WithNegativeWeight(0.5).
    Execute()
```

### Advanced Example: Finding Similar But Different Items

```go
// Find products similar to smartphones but not tablets
queryVector := productEmbedding["smartphone"]
negativeVector := productEmbedding["tablet"]

results, err := collection.FluentSearch(queryVector).
    WithK(20).
    WithNegativeExample(negativeVector).
    WithNegativeWeight(0.7). // Strong preference against tablet-like results
    Filter("category", "electronics").
    FilterGreaterThan("rating", 4.0).
    Execute()
```

### Combining Multiple Approaches

For the most powerful queries, combine negative examples with facets and other filtering options:

```go
// Find articles about AI but not focused on robotics,
// published in the last year, with high engagement
results, err := collection.FluentSearch(aiTopicVector).
    WithK(50).
    WithNegativeExample(roboticsVector).
    WithNegativeWeight(0.6).
    Filter("published_date", map[string]interface{}{
        "$gt": time.Now().AddDate(-1, 0, 0),
    }).
    Filter("engagement_score", map[string]interface{}{
        "$gt": 75,
    }).
    FilterNotEquals("is_sponsored", true).
    Execute()
```

## License

Quiver is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Happy vector searching. 🏹
