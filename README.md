# 🏹 Quiver

> **Note:** Quiver is an experimental vector database. While it's packed with cool features, it's still finding its feet in the world. Feel free to play with it, but maybe don't bet your production system on it just yet!

## 🎉 Announcing APT 🎉

We're thrilled to introduce Adaptive Parameter Tuning **(APT)**, a feature that automatically optimizes HNSW parameters based on your workload patterns and performance metrics. No more manual tuning!

APT continuously analyzes your query patterns, monitors performance, and intelligently adjusts parameters to deliver optimal performance for your specific use case. It's like having a database expert constantly fine-tuning your vector index.

Read all about it [here](adaptive/README.md)

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
- **🧠 Adaptive Tuning**: APT automatically optimizes parameters based on your workload

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

Quiver is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Happy vector searching. 🏹
