// Package hybrid implements a multi-strategy approach to vector indexing and search
// by combining multiple search strategies (exact search, HNSW, etc.)
package hybrid

import (
	"time"

	"github.com/TFMV/quiver/pkg/types"
	"github.com/TFMV/quiver/pkg/vectortypes"
)

// IndexType represents the type of index used
type IndexType string

const (
	// ExactIndexType represents an exact search index
	ExactIndexType IndexType = "exact"

	// HNSWIndexType represents an HNSW index
	HNSWIndexType IndexType = "hnsw"

	// HybridIndexType represents a hybrid index
	HybridIndexType IndexType = "hybrid"
)

// IndexConfig holds configuration options for the hybrid index
type IndexConfig struct {
	// Distance function to use
	DistanceFunc vectortypes.DistanceFunc

	// Configuration for the HNSW index
	HNSWConfig HNSWConfig

	// Threshold for switching to exact search (in number of vectors)
	ExactThreshold int
}

// DefaultIndexConfig returns a default configuration for the hybrid index
func DefaultIndexConfig() IndexConfig {
	return IndexConfig{
		DistanceFunc:   vectortypes.CosineDistance,
		HNSWConfig:     DefaultHNSWConfig(),
		ExactThreshold: 1000, // Use exact search for datasets smaller than 1000 vectors
	}
}

// HNSWConfig holds configuration options for the HNSW index
type HNSWConfig struct {
	// M is the number of connections per element
	M int

	// MaxM0 defines the maximum number of connections at layer 0
	MaxM0 int

	// EfConstruction is the size of the dynamic candidate list during index construction
	EfConstruction int

	// EfSearch is the size of the dynamic candidate list during search
	EfSearch int
}

// DefaultHNSWConfig returns a default configuration for the HNSW index
func DefaultHNSWConfig() HNSWConfig {
	return HNSWConfig{
		M:              16,
		MaxM0:          32, // Typically 2*M
		EfConstruction: 200,
		EfSearch:       100,
	}
}

// AdaptiveConfig holds configuration options for the adaptive strategy selector
type AdaptiveConfig struct {
	// Controls the exploration vs exploitation tradeoff (0-1)
	ExplorationFactor float64

	// Initial threshold for dataset size to switch from exact to HNSW (overrides IndexConfig)
	InitialExactThreshold int

	// Initial threshold for query dimension to prefer HNSW
	InitialDimThreshold int

	// Number of queries to keep metrics for
	MetricsWindowSize int

	// How aggressively to adapt thresholds (0-1)
	AdaptationRate float64
}

// DefaultAdaptiveConfig returns a default configuration for the adaptive strategy selector
func DefaultAdaptiveConfig() AdaptiveConfig {
	return AdaptiveConfig{
		ExplorationFactor:     0.1,  // 10% exploration
		InitialExactThreshold: 1000, // Use exact search for datasets smaller than 1000 vectors
		InitialDimThreshold:   100,  // Consider high-dimensional for dim > 100
		MetricsWindowSize:     1000, // Keep metrics for last 1000 queries
		AdaptationRate:        0.05, // Adapt thresholds by 5% each time
	}
}

// QueryMetrics holds performance metrics for a single query
type QueryMetrics struct {
	// The strategy used for the query
	Strategy IndexType

	// The dimension of the query vector
	QueryDimension int

	// The number of results requested
	K int

	// How long the query took
	Duration time.Duration

	// How many results were returned
	ResultCount int

	// When the query was executed
	Timestamp time.Time
}

// StrategyStats holds statistics for a specific strategy
type StrategyStats struct {
	// Number of times this strategy was selected
	UsageCount int

	// Sum of query durations for this strategy
	TotalDuration time.Duration

	// Average query duration for this strategy
	AvgDuration time.Duration
}

// HybridStats holds statistics for the hybrid index
type HybridStats struct {
	// Number of vectors in the index
	VectorCount int

	// Average dimension of vectors in the index
	AvgDimension int

	// Statistics for each strategy
	StrategyStats map[IndexType]*StrategyStats
}

// HybridSearchRequest holds parameters for a hybrid search
type HybridSearchRequest struct {
	// Query vector
	Query vectortypes.F32

	// Number of results to return
	K int

	// Force use of a specific strategy (empty means use adaptive selection)
	ForceStrategy IndexType

	// Whether to include detailed stats in the response
	IncludeStats bool

	// Negative example vector - results should be dissimilar from this
	NegativeExample vectortypes.F32

	// Weight to apply to the negative example (0.0-1.0)
	// Higher weight means stronger influence of the negative example
	NegativeWeight float32
}

// HybridSearchResponse contains the results of a hybrid search
type HybridSearchResponse struct {
	// The search results
	Results []types.BasicSearchResult

	// The strategy that was used
	StrategyUsed IndexType

	// How long the search took
	SearchTime time.Duration

	// Detailed stats (if requested)
	Stats *QueryMetrics
}

// VectorWithID represents a vector with its ID
type VectorWithID struct {
	ID     string
	Vector vectortypes.F32
}

// SearchResult represents a search result with ID and distance
type SearchResult struct {
	ID       string
	Distance float32
}

// Index is the interface that must be implemented by all indexes
type Index interface {
	// Insert adds a vector to the index
	Insert(id string, vector vectortypes.F32) error

	// Delete removes a vector from the index
	Delete(id string) error

	// Search finds the k nearest vectors to the query vector
	Search(query vectortypes.F32, k int) ([]types.BasicSearchResult, error)

	// Size returns the number of vectors in the index
	Size() int

	// GetType returns the type of this index
	GetType() IndexType

	// GetStats returns statistics about this index
	GetStats() interface{}
}

// StrategySelector is the interface for components that select search strategies
type StrategySelector interface {
	// SelectStrategy chooses the best strategy for a query
	SelectStrategy(query vectortypes.F32, k int) IndexType

	// RecordQueryMetrics records metrics for a completed query
	RecordQueryMetrics(metrics QueryMetrics)

	// GetStats returns statistics about strategy selection
	GetStats() map[string]interface{}
}
