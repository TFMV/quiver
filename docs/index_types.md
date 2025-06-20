# Index Types

## ArrowHNSW Index

The ArrowHNSW index is a variant of the HNSW index that stores vectors in Apache Arrow format. Vectors and metadata are kept in columnar structures to enable zero-copy access and efficient batch operations. The index can be persisted using Arrow IPC files and optionally converted to Parquet for long term storage.

### Use Cases
- Workloads that benefit from columnar processing
- Integration with Arrow based analytics pipelines

### Persistence Format
`Save` writes the vectors and identifiers as an Arrow record with a fixed size list column. `Load` restores the index from this file.

### Example
```go
idx := index.NewArrowHNSWIndex(128)
// build a query vector as Arrow Float32 array
b := array.NewFloat32Builder(memory.DefaultAllocator)
b.AppendValues(myVec, nil)
arr := b.NewArray()
res, _ := idx.Search(arr, 10)
```
