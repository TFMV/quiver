package hybrid

import (
	"github.com/TFMV/quiver/pkg/hnsw"
	"github.com/TFMV/quiver/pkg/types"
	"github.com/TFMV/quiver/pkg/vectortypes"
)

// HNSWAdapter adapts the existing HNSW implementation to the hybrid Index interface
type HNSWAdapter struct {
	// The underlying HNSW adapter
	adapter *hnsw.HNSWAdapter

	// Configuration for optimization
	config HNSWConfig
}

// NewHNSWAdapter creates a new HNSW adapter for the hybrid index
func NewHNSWAdapter(distFunc vectortypes.DistanceFunc, config HNSWConfig) *HNSWAdapter {
	// Convert the distance function to HNSW format
	hnswDistFunc := func(a, b []float32) (float32, error) {
		return distFunc(a, b), nil
	}

	// Create HNSW config
	hnswConfig := hnsw.Config{
		M:              config.M,
		MaxM0:          config.MaxM0,
		EfConstruction: config.EfConstruction,
		EfSearch:       config.EfSearch,
		DistanceFunc:   hnswDistFunc,
	}

	// Create the adapter
	return &HNSWAdapter{
		adapter: hnsw.NewAdapter(hnswConfig),
		config:  config,
	}
}

// Insert adds a vector to the index
func (idx *HNSWAdapter) Insert(id string, vector vectortypes.F32) error {
	return idx.adapter.Insert(id, vector)
}

// Delete removes a vector from the index
func (idx *HNSWAdapter) Delete(id string) error {
	return idx.adapter.Delete(id)
}

// Search finds the k nearest vectors to the query vector
func (idx *HNSWAdapter) Search(query vectortypes.F32, k int) ([]types.BasicSearchResult, error) {
	return idx.adapter.Search(query, k)
}

// SearchWithNegative finds the k nearest vectors to the query vector,
// taking into account a negative example vector
func (idx *HNSWAdapter) SearchWithNegative(query vectortypes.F32, negativeExample vectortypes.F32, negativeWeight float32, k int) ([]types.BasicSearchResult, error) {
	// Delegate to the underlying HNSW adapter
	return idx.adapter.SearchWithNegativeExample(query, negativeExample, negativeWeight, k)
}

// Size returns the number of vectors in the index
func (idx *HNSWAdapter) Size() int {
	return idx.adapter.Size()
}

// GetType returns the index type
func (idx *HNSWAdapter) GetType() IndexType {
	return HNSWIndexType
}

// GetStats returns statistics about this index
func (idx *HNSWAdapter) GetStats() interface{} {
	params := idx.adapter.GetOptimizationParameters()
	metrics := idx.adapter.GetPerformanceMetrics()

	return map[string]interface{}{
		"type":         string(HNSWIndexType),
		"vector_count": idx.Size(),
		"parameters":   params,
		"metrics":      metrics,
	}
}

// SetSearchEf adjusts the EfSearch parameter which controls search accuracy
func (idx *HNSWAdapter) SetSearchEf(efSearch int) error {
	return idx.adapter.SetOptimizationParameters(map[string]float64{
		"EfSearch": float64(efSearch),
	})
}
