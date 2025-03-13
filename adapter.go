package db

import (
	"cmp"
	"fmt"

	"github.com/TFMV/hnsw"
	"github.com/TFMV/hnsw/hnsw-extensions/hybrid"
	"github.com/TFMV/hnsw/hnsw-extensions/parquet"
)

// IndexAdapter provides a common interface for different index implementations
type IndexAdapter[K cmp.Ordered] struct {
	// The underlying index
	index interface{}

	// Type of the index
	indexType string
}

// NewHNSWAdapter creates a new adapter for a standard HNSW graph
func NewHNSWAdapter[K cmp.Ordered](graph *hnsw.Graph[K]) *IndexAdapter[K] {
	return &IndexAdapter[K]{
		index:     graph,
		indexType: "hnsw",
	}
}

// NewParquetAdapter creates a new adapter for a Parquet-based HNSW graph
func NewParquetAdapter[K cmp.Ordered](graph *parquet.ParquetGraph[K]) *IndexAdapter[K] {
	return &IndexAdapter[K]{
		index:     graph,
		indexType: "parquet",
	}
}

// NewHybridAdapter creates a new adapter for a hybrid index
func NewHybridAdapter[K cmp.Ordered](index *hybrid.HybridIndex[K]) *IndexAdapter[K] {
	return &IndexAdapter[K]{
		index:     index,
		indexType: "hybrid",
	}
}

// Add adds a vector to the index
func (a *IndexAdapter[K]) Add(key K, vector []float32) error {
	switch a.indexType {
	case "hnsw":
		graph := a.index.(*hnsw.Graph[K])
		node := hnsw.MakeNode(key, vector)
		return graph.Add(node)
	case "parquet":
		graph := a.index.(*parquet.ParquetGraph[K])
		node := hnsw.MakeNode(key, vector)
		return graph.Add(node)
	case "hybrid":
		index := a.index.(*hybrid.HybridIndex[K])
		return index.Add(key, vector)
	default:
		return fmt.Errorf("unsupported index type: %s", a.indexType)
	}
}

// BatchAdd adds multiple vectors to the index
func (a *IndexAdapter[K]) BatchAdd(keys []K, vectors [][]float32) error {
	if len(keys) != len(vectors) {
		return fmt.Errorf("keys and vectors must have the same length")
	}

	switch a.indexType {
	case "hnsw":
		graph := a.index.(*hnsw.Graph[K])
		nodes := make([]hnsw.Node[K], len(keys))
		for i, key := range keys {
			nodes[i] = hnsw.MakeNode(key, vectors[i])
		}
		return graph.BatchAdd(nodes)
	case "parquet":
		// Optimize batch operations for ParquetGraph
		// Instead of adding vectors one by one, we'll use a more efficient approach
		graph := a.index.(*parquet.ParquetGraph[K])

		// Create a batch of nodes
		nodes := make([]hnsw.Node[K], len(keys))
		for i, key := range keys {
			nodes[i] = hnsw.MakeNode(key, vectors[i])
		}

		// Use a worker pool to add vectors in parallel for better performance
		const maxWorkers = 8
		workerCount := min(maxWorkers, len(nodes))

		if workerCount <= 1 || len(nodes) < 10 {
			// For small batches, just add sequentially
			for _, node := range nodes {
				if err := graph.Add(node); err != nil {
					return err
				}
			}
			return nil
		}

		// For larger batches, use parallel processing
		errChan := make(chan error, len(nodes))
		nodeChan := make(chan hnsw.Node[K], len(nodes))

		// Start worker goroutines
		for i := 0; i < workerCount; i++ {
			go func() {
				for node := range nodeChan {
					if err := graph.Add(node); err != nil {
						errChan <- err
						return
					}
				}
				errChan <- nil
			}()
		}

		// Send nodes to workers
		for _, node := range nodes {
			nodeChan <- node
		}
		close(nodeChan)

		// Wait for workers to finish and collect errors
		for i := 0; i < workerCount; i++ {
			if err := <-errChan; err != nil {
				return err
			}
		}

		return nil
	case "hybrid":
		index := a.index.(*hybrid.HybridIndex[K])
		return index.BatchAdd(keys, vectors)
	default:
		return fmt.Errorf("unsupported index type: %s", a.indexType)
	}
}

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Search finds the k nearest neighbors to the query vector
func (a *IndexAdapter[K]) Search(query []float32, k int) ([]hnsw.Node[K], error) {
	switch a.indexType {
	case "hnsw":
		graph := a.index.(*hnsw.Graph[K])
		return graph.Search(query, k)
	case "parquet":
		graph := a.index.(*parquet.ParquetGraph[K])
		return graph.Search(query, k)
	case "hybrid":
		index := a.index.(*hybrid.HybridIndex[K])
		nodes, err := index.Search(query, k)
		return nodes, err
	default:
		return nil, fmt.Errorf("unsupported index type: %s", a.indexType)
	}
}

// SearchWithNegative finds the k nearest neighbors while avoiding vectors similar to the negative example
func (a *IndexAdapter[K]) SearchWithNegative(query []float32, negative []float32, k int, negWeight float32) ([]hnsw.Node[K], error) {
	switch a.indexType {
	case "hnsw":
		graph := a.index.(*hnsw.Graph[K])
		return graph.SearchWithNegative(query, negative, k, negWeight)
	default:
		// Fallback to regular search for other index types
		return a.Search(query, k)
	}
}

// SearchWithNegatives finds the k nearest neighbors while avoiding vectors similar to multiple negative examples
func (a *IndexAdapter[K]) SearchWithNegatives(query []float32, negatives [][]float32, k int, negWeight float32) ([]hnsw.Node[K], error) {
	switch a.indexType {
	case "hnsw":
		graph := a.index.(*hnsw.Graph[K])
		return graph.SearchWithNegatives(query, negatives, k, negWeight)
	default:
		// Fallback to regular search for other index types
		return a.Search(query, k)
	}
}

// Delete removes a vector from the index
func (a *IndexAdapter[K]) Delete(key K) bool {
	switch a.indexType {
	case "hnsw":
		graph := a.index.(*hnsw.Graph[K])
		return graph.Delete(key)
	case "parquet":
		graph := a.index.(*parquet.ParquetGraph[K])
		return graph.Delete(key)
	case "hybrid":
		index := a.index.(*hybrid.HybridIndex[K])
		return index.Delete(key)
	default:
		return false
	}
}

// BatchDelete removes multiple vectors from the index
func (a *IndexAdapter[K]) BatchDelete(keys []K) []bool {
	switch a.indexType {
	case "hnsw":
		graph := a.index.(*hnsw.Graph[K])
		return graph.BatchDelete(keys)
	case "parquet":
		graph := a.index.(*parquet.ParquetGraph[K])
		return graph.BatchDelete(keys)
	case "hybrid":
		index := a.index.(*hybrid.HybridIndex[K])
		return index.BatchDelete(keys)
	default:
		results := make([]bool, len(keys))
		return results
	}
}

// Len returns the number of vectors in the index
func (a *IndexAdapter[K]) Len() int {
	switch a.indexType {
	case "hnsw":
		graph := a.index.(*hnsw.Graph[K])
		return graph.Len()
	case "parquet":
		graph := a.index.(*parquet.ParquetGraph[K])
		return graph.Len()
	case "hybrid":
		index := a.index.(*hybrid.HybridIndex[K])
		return index.Len()
	default:
		return 0
	}
}

// Close releases resources used by the index
func (a *IndexAdapter[K]) Close() error {
	switch a.indexType {
	case "hnsw":
		// Standard HNSW graph doesn't need to be closed
		return nil
	case "parquet":
		graph := a.index.(*parquet.ParquetGraph[K])
		return graph.Close()
	case "hybrid":
		index := a.index.(*hybrid.HybridIndex[K])
		return index.Close()
	default:
		return fmt.Errorf("unsupported index type: %s", a.indexType)
	}
}

// GetUnderlyingIndex returns the underlying index
func (a *IndexAdapter[K]) GetUnderlyingIndex() interface{} {
	return a.index
}

// GetIndexType returns the type of the index
func (a *IndexAdapter[K]) GetIndexType() string {
	return a.indexType
}
