package hybrid

import (
	"errors"
	"fmt"
	"sort"
	"sync"

	"github.com/TFMV/quiver/pkg/types"
	"github.com/TFMV/quiver/pkg/vectortypes"
)

// ExactIndex provides brute-force exact search for small datasets
type ExactIndex struct {
	// Map of vector IDs to vectors
	vectors map[string]vectortypes.F32

	// Distance function to use
	distFunc vectortypes.DistanceFunc

	// dimension of stored vectors
	vectorDim int

	// Mutex for thread safety
	mu sync.RWMutex
}

// NewExactIndex creates a new exact search index
func NewExactIndex(distFunc vectortypes.DistanceFunc) *ExactIndex {
	return &ExactIndex{
		vectors:   make(map[string]vectortypes.F32),
		distFunc:  distFunc,
		vectorDim: 0,
	}
}

// Insert adds a vector to the index
func (idx *ExactIndex) Insert(id string, vector vectortypes.F32) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Validate dimension consistency
	if idx.vectorDim == 0 {
		idx.vectorDim = len(vector)
	} else if len(vector) != idx.vectorDim {
		return fmt.Errorf("vector dimension mismatch: expected %d, got %d", idx.vectorDim, len(vector))
	}

	// Make a copy of the vector to prevent external modification
	vectorCopy := make(vectortypes.F32, len(vector))
	copy(vectorCopy, vector)

	idx.vectors[id] = vectorCopy
	return nil
}

// Delete removes a vector from the index
func (idx *ExactIndex) Delete(id string) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	delete(idx.vectors, id)
	if len(idx.vectors) == 0 {
		idx.vectorDim = 0
	}
	return nil
}

// resultHeap is a min-heap for search results based on distance
type resultHeap []types.BasicSearchResult

func (h resultHeap) Len() int           { return len(h) }
func (h resultHeap) Less(i, j int) bool { return h[i].Distance < h[j].Distance }
func (h resultHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *resultHeap) Push(x interface{}) {
	*h = append(*h, x.(types.BasicSearchResult))
}

func (h *resultHeap) Pop() interface{} {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[0 : n-1]
	return item
}

// Search finds the k nearest vectors to the query vector
func (idx *ExactIndex) Search(query vectortypes.F32, k int) ([]types.BasicSearchResult, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if len(idx.vectors) == 0 {
		return []types.BasicSearchResult{}, nil
	}

	if idx.vectorDim > 0 && len(query) != idx.vectorDim {
		return nil, fmt.Errorf("query dimension mismatch: expected %d, got %d", idx.vectorDim, len(query))
	}

	if k <= 0 {
		return nil, errors.New("k must be positive")
	}

	// Limit k to the number of vectors
	if k > len(idx.vectors) {
		k = len(idx.vectors)
	}

	// Calculate distances for all vectors
	results := make(resultHeap, 0, len(idx.vectors))
	for id, vec := range idx.vectors {
		distance := idx.distFunc(query, vec)
		results = append(results, types.BasicSearchResult{
			ID:       id,
			Distance: distance,
		})
	}

	// Sort by distance
	sort.Sort(&results)

	// Return only k nearest neighbors
	if k < len(results) {
		results = results[:k]
	}

	// Return results in ascending order of distance
	return results, nil
}

// Size returns the number of vectors in the index
func (idx *ExactIndex) Size() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	return len(idx.vectors)
}

// GetType returns the index type
func (idx *ExactIndex) GetType() IndexType {
	return ExactIndexType
}

// GetStats returns statistics about this index
func (idx *ExactIndex) GetStats() interface{} {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	return map[string]interface{}{
		"type":         string(ExactIndexType),
		"vector_count": len(idx.vectors),
	}
}
