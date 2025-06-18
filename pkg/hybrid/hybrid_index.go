package hybrid

import (
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/TFMV/quiver/pkg/types"
	"github.com/TFMV/quiver/pkg/vectortypes"
)

// HybridIndex is the main index implementation that combines multiple search strategies
type HybridIndex struct {
	// Configuration
	config IndexConfig

	// Component indexes
	exactIndex *ExactIndex
	hnswIndex  *HNSWAdapter

	// Adaptive strategy selector
	selector *AdaptiveStrategySelector

	// Distance function
	distFunc vectortypes.DistanceFunc

	// Storage for vectors (needed for exact search)
	vectors map[string]vectortypes.F32

	// Track the dimensions of stored vectors
	dimensions []int

	// Stats tracking
	stats HybridStats

	// Dimension of vectors stored in the index
	vectorDim int

	// Mutex for thread safety
	mu sync.RWMutex
}

// NewHybridIndex creates a new hybrid index with the specified configuration
func NewHybridIndex(config IndexConfig) *HybridIndex {
	// Ensure we have a distance function
	if config.DistanceFunc == nil {
		config.DistanceFunc = vectortypes.CosineDistance
	}

	// Create component indexes
	exactIndex := NewExactIndex(config.DistanceFunc)
	hnswIndex := NewHNSWAdapter(config.DistanceFunc, config.HNSWConfig)

	// Create adaptive selector
	adaptiveConfig := DefaultAdaptiveConfig()
	adaptiveConfig.InitialExactThreshold = config.ExactThreshold
	selector := NewAdaptiveStrategySelector(adaptiveConfig)

	// Create the index
	index := &HybridIndex{
		config:     config,
		exactIndex: exactIndex,
		hnswIndex:  hnswIndex,
		selector:   selector,
		distFunc:   config.DistanceFunc,
		vectors:    make(map[string]vectortypes.F32),
		dimensions: make([]int, 0),
		stats: HybridStats{
			StrategyStats: make(map[IndexType]*StrategyStats),
		},
	}

	// Initialize strategy stats
	index.stats.StrategyStats[ExactIndexType] = &StrategyStats{}
	index.stats.StrategyStats[HNSWIndexType] = &StrategyStats{}
	index.stats.StrategyStats[HybridIndexType] = &StrategyStats{}

	index.vectorDim = 0

	return index
}

// Insert adds a vector to the index
func (idx *HybridIndex) Insert(id string, vector vectortypes.F32) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if idx.vectorDim == 0 {
		idx.vectorDim = len(vector)
	} else if len(vector) != idx.vectorDim {
		return fmt.Errorf("vector dimension mismatch: expected %d, got %d", idx.vectorDim, len(vector))
	}

	// Make a copy of the vector to prevent external modification
	vectorCopy := make(vectortypes.F32, len(vector))
	copy(vectorCopy, vector)

	// Store the vector in our map
	idx.vectors[id] = vectorCopy

	// Track dimensions for statistics
	idx.dimensions = append(idx.dimensions, len(vector))

	// Update statistics
	idx.stats.VectorCount++
	idx.stats.AvgDimension = calculateAvgDimension(idx.dimensions)

	// Update adaptive selector thresholds
	idx.selector.UpdateThresholds(idx.stats.VectorCount, idx.stats.AvgDimension)

	// Add to all component indexes
	if err := idx.exactIndex.Insert(id, vectorCopy); err != nil {
		return err
	}

	if err := idx.hnswIndex.Insert(id, vectorCopy); err != nil {
		// If HNSW insertion fails, remove from exact index to maintain consistency
		if deleteErr := idx.exactIndex.Delete(id); deleteErr != nil {
			// Log that rollback failed but return the original error
			return fmt.Errorf("insert failed (%w) and rollback also failed: %v", err, deleteErr)
		}
		return err
	}

	return nil
}

// InsertBatch adds multiple vectors to the index in a single operation
func (idx *HybridIndex) InsertBatch(vectors map[string]vectortypes.F32) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if len(vectors) == 0 {
		return nil
	}

	if idx.vectorDim == 0 {
		for _, v := range vectors {
			idx.vectorDim = len(v)
			break
		}
	}
	for _, v := range vectors {
		if len(v) != idx.vectorDim {
			return fmt.Errorf("vector dimension mismatch: expected %d, got %d", idx.vectorDim, len(v))
		}
	}

	// Make copies and collect dimensions
	vectorsCopy := make(map[string]vectortypes.F32, len(vectors))
	newDimensions := make([]int, 0, len(vectors))

	for id, vector := range vectors {
		// Copy each vector
		vectorCopy := make(vectortypes.F32, len(vector))
		copy(vectorCopy, vector)
		vectorsCopy[id] = vectorCopy

		// Track dimension
		newDimensions = append(newDimensions, len(vector))
	}

	// Perform insertions into both indexes first
	// exact index insertions
	for id, vector := range vectorsCopy {
		if err := idx.exactIndex.Insert(id, vector); err != nil {
			// Rollback any insertions made so far
			rollbackErrs := []string{}
			for rollbackID := range vectorsCopy {
				if rollbackID == id {
					break // Stop at the failed ID
				}
				if deleteErr := idx.exactIndex.Delete(rollbackID); deleteErr != nil {
					rollbackErrs = append(rollbackErrs, fmt.Sprintf("failed to rollback %s: %v", rollbackID, deleteErr))
				}
			}
			if len(rollbackErrs) > 0 {
				return fmt.Errorf("batch insert failed at ID %s: %w (rollback errors: %v)", id, err, rollbackErrs)
			}
			return fmt.Errorf("batch insert failed at ID %s: %w", id, err)
		}
	}

	// HNSW index insertions
	for id, vector := range vectorsCopy {
		if err := idx.hnswIndex.Insert(id, vector); err != nil {
			// Rollback all insertions
			rollbackErrs := []string{}
			for rollbackID := range vectorsCopy {
				if deleteErr := idx.exactIndex.Delete(rollbackID); deleteErr != nil {
					rollbackErrs = append(rollbackErrs, fmt.Sprintf("failed to rollback exact %s: %v", rollbackID, deleteErr))
				}
				// Only try to delete from HNSW if we've already inserted it
				if rollbackID == id {
					break
				}
				if deleteErr := idx.hnswIndex.Delete(rollbackID); deleteErr != nil {
					rollbackErrs = append(rollbackErrs, fmt.Sprintf("failed to rollback hnsw %s: %v", rollbackID, deleteErr))
				}
			}
			if len(rollbackErrs) > 0 {
				return fmt.Errorf("batch insert failed at ID %s: %w (rollback errors: %v)", id, err, rollbackErrs)
			}
			return fmt.Errorf("batch insert failed at ID %s: %w", id, err)
		}
	}

	// If all successful, update our internal vectors map
	for id, vector := range vectorsCopy {
		idx.vectors[id] = vector
	}

	// Update dimensions tracking
	idx.dimensions = append(idx.dimensions, newDimensions...)

	// Update statistics
	idx.stats.VectorCount += len(vectors)
	idx.stats.AvgDimension = calculateAvgDimension(idx.dimensions)

	// Update adaptive selector thresholds
	idx.selector.UpdateThresholds(idx.stats.VectorCount, idx.stats.AvgDimension)

	return nil
}

// Delete removes a vector from the index
func (idx *HybridIndex) Delete(id string) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Check if the vector exists
	if _, exists := idx.vectors[id]; !exists {
		return fmt.Errorf("vector with ID %s not found", id)
	}

	// Remove from all component indexes
	if err := idx.exactIndex.Delete(id); err != nil {
		return err
	}

	if err := idx.hnswIndex.Delete(id); err != nil {
		return err
	}

	// Remove from our map
	delete(idx.vectors, id)

	// Update statistics
	if idx.stats.VectorCount > 0 {
		idx.stats.VectorCount--
	}

	// Recalculate average dimension
	if len(idx.dimensions) > 0 {
		// Remove one dimension entry (this is approximate since we don't track which vector had which dimension)
		idx.dimensions = idx.dimensions[:len(idx.dimensions)-1]
		if len(idx.dimensions) > 0 {
			idx.stats.AvgDimension = calculateAvgDimension(idx.dimensions)
		} else {
			idx.stats.AvgDimension = 0
		}
	}

	if len(idx.vectors) == 0 {
		idx.vectorDim = 0
	}

	idx.selector.UpdateThresholds(idx.stats.VectorCount, idx.stats.AvgDimension)

	return nil
}

// DeleteBatch removes multiple vectors from the index in a single operation
func (idx *HybridIndex) DeleteBatch(ids []string) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if len(ids) == 0 {
		return nil
	}

	// Verify all vectors exist first
	nonExistingIDs := make([]string, 0)
	for _, id := range ids {
		if _, exists := idx.vectors[id]; !exists {
			nonExistingIDs = append(nonExistingIDs, id)
		}
	}

	if len(nonExistingIDs) > 0 {
		return fmt.Errorf("some vectors not found: %v", nonExistingIDs)
	}

	// Collect dimensions for later statistics updates
	dimensionsToRemove := len(ids)

	// Delete from both indexes
	errorsOccurred := false
	errorMsgs := make([]string, 0)

	// First delete from exact index
	for _, id := range ids {
		if err := idx.exactIndex.Delete(id); err != nil {
			errorsOccurred = true
			errorMsgs = append(errorMsgs, fmt.Sprintf("failed to delete %s from exact index: %v", id, err))
		}
	}

	// Then delete from HNSW index
	for _, id := range ids {
		if err := idx.hnswIndex.Delete(id); err != nil {
			errorsOccurred = true
			errorMsgs = append(errorMsgs, fmt.Sprintf("failed to delete %s from HNSW index: %v", id, err))
		}
	}

	// Delete from internal map
	for _, id := range ids {
		delete(idx.vectors, id)
	}

	// Update statistics
	vectorCountDelta := idx.stats.VectorCount - dimensionsToRemove
	if vectorCountDelta < 0 {
		vectorCountDelta = 0
	}
	idx.stats.VectorCount = vectorCountDelta

	// Update dimensions tracking (approximate since we don't know exactly which ones to remove)
	if len(idx.dimensions) > dimensionsToRemove {
		idx.dimensions = idx.dimensions[:len(idx.dimensions)-dimensionsToRemove]
	} else {
		idx.dimensions = make([]int, 0)
	}

	// Recalculate average dimension
	if len(idx.dimensions) > 0 {
		idx.stats.AvgDimension = calculateAvgDimension(idx.dimensions)
	} else {
		idx.stats.AvgDimension = 0
	}

	if errorsOccurred {
		return fmt.Errorf("errors during batch delete: %v", errorMsgs)
	}

	if len(idx.vectors) == 0 {
		idx.vectorDim = 0
	}

	idx.selector.UpdateThresholds(idx.stats.VectorCount, idx.stats.AvgDimension)

	return nil
}

// Search finds the k nearest vectors to the query vector
func (idx *HybridIndex) Search(query vectortypes.F32, k int) ([]types.BasicSearchResult, error) {
	return idx.searchWithStrategy(query, k, "")
}

// SearchWithRequest performs a search using the provided request parameters
func (idx *HybridIndex) SearchWithRequest(req HybridSearchRequest) (HybridSearchResponse, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	startTime := time.Now()

	if idx.vectorDim > 0 && len(req.Query) != idx.vectorDim {
		return HybridSearchResponse{}, fmt.Errorf("query dimension mismatch: expected %d, got %d", idx.vectorDim, len(req.Query))
	}
	if len(req.NegativeExample) > 0 && idx.vectorDim > 0 && len(req.NegativeExample) != idx.vectorDim {
		return HybridSearchResponse{}, fmt.Errorf("negative example dimension mismatch: expected %d, got %d", idx.vectorDim, len(req.NegativeExample))
	}

	// Ensure valid K
	if req.K <= 0 {
		return HybridSearchResponse{}, errors.New("k must be positive")
	}

	// Use the forced strategy if specified, otherwise use adaptive selection
	var strategy IndexType
	if req.ForceStrategy != "" {
		strategy = req.ForceStrategy
	} else {
		// Use vector count and dimension from the index stats
		strategy = idx.selector.SelectStrategy(idx.stats.VectorCount, idx.stats.AvgDimension, req.K)
	}

	// Execute search with selected strategy, including negative example if provided
	var results []types.BasicSearchResult
	var err error

	if len(req.NegativeExample) > 0 && req.NegativeWeight > 0 {
		// Create a single-element "vector" to hold the negative weight
		negWeight := vectortypes.F32{req.NegativeWeight}
		results, err = idx.searchWithStrategy(req.Query, req.K, strategy, req.NegativeExample, negWeight)
	} else {
		results, err = idx.searchWithStrategy(req.Query, req.K, strategy)
	}

	if err != nil {
		return HybridSearchResponse{}, err
	}

	// Calculate search time
	duration := time.Since(startTime)

	// Create metrics
	metrics := QueryMetrics{
		Strategy:       strategy,
		QueryDimension: len(req.Query),
		K:              req.K,
		Duration:       duration,
		ResultCount:    len(results),
		Timestamp:      time.Now(),
	}

	// Record metrics
	idx.selector.RecordQueryMetrics(metrics)

	// Update strategy stats
	if strategyStats, ok := idx.stats.StrategyStats[strategy]; ok {
		strategyStats.UsageCount++
		strategyStats.TotalDuration += duration
		if strategyStats.UsageCount > 0 {
			strategyStats.AvgDuration = strategyStats.TotalDuration / time.Duration(strategyStats.UsageCount)
		}
	}

	// Build response
	response := HybridSearchResponse{
		Results:      results,
		StrategyUsed: strategy,
		SearchTime:   duration,
	}

	// Include detailed stats if requested
	if req.IncludeStats {
		response.Stats = &metrics
	}

	return response, nil
}

// Helper function to get the name of a function (used for distance function comparison)
func getFunctionName(i interface{}) string {
	return fmt.Sprintf("%v", i)
}

// helper function
func min(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

// searchWithStrategy performs a search using the specified strategy
// If negativeExample is not empty and negativeWeight > 0, it will influence the search
func (idx *HybridIndex) searchWithStrategy(query vectortypes.F32, k int, strategy IndexType, negativeExample ...vectortypes.F32) ([]types.BasicSearchResult, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if idx.vectorDim > 0 && len(query) != idx.vectorDim {
		return nil, fmt.Errorf("query dimension mismatch: expected %d, got %d", idx.vectorDim, len(query))
	}

	// If no specific strategy is provided, select one
	if strategy == "" {
		// Pass proper parameters to SelectStrategy: vectorCount, dimension, k
		strategy = idx.selector.SelectStrategy(idx.stats.VectorCount, idx.stats.AvgDimension, k)
	}

	// Check if we have a negative example
	var negExample vectortypes.F32
	var negWeight float32 = 0.5 // default weight
	hasNegative := false

	if len(negativeExample) > 0 {
		negExample = negativeExample[0]
		hasNegative = len(negExample) > 0

		// If we have more than one parameter, the second one is the negative weight
		if len(negativeExample) > 1 && len(negativeExample[1]) == 1 {
			negWeight = negativeExample[1][0]
		}
		if hasNegative && idx.vectorDim > 0 && len(negExample) != idx.vectorDim {
			return nil, fmt.Errorf("negative example dimension mismatch: expected %d, got %d", idx.vectorDim, len(negExample))
		}
	}

	// Execute search with selected strategy
	var (
		results []types.BasicSearchResult
		err     error
	)

	switch strategy {
	case ExactIndexType:
		// For exact search, we'll do a standard search and then re-rank using negative examples
		results, err = idx.exactIndex.Search(query, k)
		if err != nil {
			return nil, err
		}

		if hasNegative {
			// Re-rank exact search results using negative example
			// This is done by explicitly calculating distances
			for i, result := range results {
				vector, exists := idx.vectors[result.ID]
				if !exists {
					continue
				}

				// Calculate distance to negative example
				negDist := idx.distFunc(vector, negExample)

				// For our distance function, lower is closer (more similar)
				// We want to boost items that are LESS similar to the negative example
				// So we combine the distances:
				// - Keep the original distance as the main part
				// - Subtract a factor for the negative example (inverted, since higher values are better)

				// Get distance function signature
				distFuncName := getFunctionName(idx.distFunc)

				// Normalize negative distance based on distance function
				var normNegDist float32
				if strings.Contains(distFuncName, "Cosine") {
					// For cosine, range is 0-2, where 0 is identical, 2 is opposite
					// Normalize to [0,1] and invert
					normNegDist = 1.0 - (negDist / 2.0)
				} else {
					// For Euclidean, normalize using a heuristic max value (sqrt(2) for normalized vectors)
					normNegDist = min(negDist/1.414, 1.0)
				}

				// Combine distances
				combinedDist := (result.Distance * (1 - negWeight)) - ((1 - normNegDist) * negWeight)
				if combinedDist < 0 {
					combinedDist = 0
				}

				results[i].Distance = combinedDist
			}

			// Re-sort by adjusted distance
			sort.Slice(results, func(i, j int) bool {
				return results[i].Distance < results[j].Distance
			})
		}

	case HNSWIndexType:
		// HNSW adapter might have specialized support for negative examples
		if hasNegative {
			// If HNSWAdapter has SearchWithNegative method, use it
			results, err = idx.hnswIndex.SearchWithNegative(query, negExample, negWeight, k)
		} else {
			results, err = idx.hnswIndex.Search(query, k)
		}
	default:
		return nil, fmt.Errorf("invalid search strategy: %s", strategy)
	}

	return results, err
}

// Size returns the number of vectors in the index
func (idx *HybridIndex) Size() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	return len(idx.vectors)
}

// GetType returns the index type
func (idx *HybridIndex) GetType() IndexType {
	return HybridIndexType
}

// GetStats returns statistics about this index
func (idx *HybridIndex) GetStats() interface{} {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	// Get component index stats
	exactStats := idx.exactIndex.GetStats()
	hnswStats := idx.hnswIndex.GetStats()
	selectorStats := idx.selector.GetStats()

	return map[string]interface{}{
		"type":           string(HybridIndexType),
		"vector_count":   idx.stats.VectorCount,
		"avg_dimension":  idx.stats.AvgDimension,
		"exact_index":    exactStats,
		"hnsw_index":     hnswStats,
		"adaptive_stats": selectorStats,
	}
}

// calculateAvgDimension calculates the average dimension
func calculateAvgDimension(dimensions []int) int {
	if len(dimensions) == 0 {
		return 0
	}

	sum := 0
	for _, dim := range dimensions {
		sum += dim
	}

	return sum / len(dimensions)
}

// BatchSearchRequest holds parameters for multiple search requests
type BatchSearchRequest struct {
	// Queries contains multiple query vectors
	Queries []vectortypes.F32
	// K is the number of results to return for each query
	K int
	// ForceStrategy forces a specific search strategy for all queries
	ForceStrategy IndexType
	// IncludeStats determines if stats should be included in the response
	IncludeStats bool
	// NegativeExamples contains optional negative examples for each query (can be nil)
	NegativeExamples []vectortypes.F32
	// NegativeWeight is the weight to apply to negative examples (0.0-1.0)
	NegativeWeight float32
}

// BatchSearchResponse contains the results of multiple searches
type BatchSearchResponse struct {
	// Results for each query, in the same order as the requests
	Results [][]types.BasicSearchResult
	// StrategiesUsed records which strategy was used for each query
	StrategiesUsed []IndexType
	// SearchTimes records how long each search took
	SearchTimes []time.Duration
	// Stats are optional detailed metrics for each query
	Stats []*QueryMetrics
}

// BatchSearch performs multiple searches in parallel
func (idx *HybridIndex) BatchSearch(request BatchSearchRequest) (BatchSearchResponse, error) {
	if len(request.Queries) == 0 {
		return BatchSearchResponse{}, errors.New("no queries provided")
	}

	// Prepare response containers
	results := make([][]types.BasicSearchResult, len(request.Queries))
	strategiesUsed := make([]IndexType, len(request.Queries))
	searchTimes := make([]time.Duration, len(request.Queries))
	var stats []*QueryMetrics
	if request.IncludeStats {
		stats = make([]*QueryMetrics, len(request.Queries))
	}

	// Use a wait group to synchronize parallel searches
	var wg sync.WaitGroup
	var errMu sync.Mutex
	var firstErr error

	// Check if we have valid negative examples
	hasNegativeExamples := len(request.NegativeExamples) > 0 && request.NegativeWeight > 0
	if hasNegativeExamples && len(request.NegativeExamples) != len(request.Queries) {
		return BatchSearchResponse{}, errors.New("number of negative examples must match number of queries")
	}

	// Process each query in parallel
	for i, query := range request.Queries {
		wg.Add(1)
		go func(index int, queryVector vectortypes.F32) {
			defer wg.Done()

			if idx.vectorDim > 0 && len(queryVector) != idx.vectorDim {
				errMu.Lock()
				if firstErr == nil {
					firstErr = fmt.Errorf("query %d dimension mismatch: expected %d, got %d", index, idx.vectorDim, len(queryVector))
				}
				errMu.Unlock()
				return
			}
			if hasNegativeExamples && len(request.NegativeExamples[index]) != idx.vectorDim {
				errMu.Lock()
				if firstErr == nil {
					firstErr = fmt.Errorf("negative example %d dimension mismatch: expected %d, got %d", index, idx.vectorDim, len(request.NegativeExamples[index]))
				}
				errMu.Unlock()
				return
			}

			// Time this search
			startTime := time.Now()

			// Select strategy
			var strategy IndexType
			if request.ForceStrategy != "" {
				strategy = request.ForceStrategy
			} else {
				// Read lock scope
				idx.mu.RLock()
				strategy = idx.selector.SelectStrategy(idx.stats.VectorCount, idx.stats.AvgDimension, request.K)
				idx.mu.RUnlock()
			}

			// Execute search, with negative example if provided
			var searchResults []types.BasicSearchResult
			var err error

			if hasNegativeExamples {
				// Create a weight vector
				negWeight := vectortypes.F32{request.NegativeWeight}
				searchResults, err = idx.searchWithStrategy(queryVector, request.K, strategy,
					request.NegativeExamples[index], negWeight)
			} else {
				searchResults, err = idx.searchWithStrategy(queryVector, request.K, strategy)
			}

			if err != nil {
				errMu.Lock()
				if firstErr == nil {
					firstErr = fmt.Errorf("search %d failed: %w", index, err)
				}
				errMu.Unlock()
				return
			}

			// Record duration
			duration := time.Since(startTime)

			// Collect results
			results[index] = searchResults
			strategiesUsed[index] = strategy
			searchTimes[index] = duration

			// Create and store metrics
			if request.IncludeStats {
				metrics := &QueryMetrics{
					Strategy:       strategy,
					QueryDimension: len(queryVector),
					K:              request.K,
					Duration:       duration,
					ResultCount:    len(searchResults),
					Timestamp:      time.Now(),
				}
				stats[index] = metrics

				// Record metrics
				go idx.selector.RecordQueryMetrics(*metrics)
			} else {
				// Still record metrics in the background, but don't include in response
				go idx.selector.RecordQueryMetrics(QueryMetrics{
					Strategy:       strategy,
					QueryDimension: len(queryVector),
					K:              request.K,
					Duration:       duration,
					ResultCount:    len(searchResults),
					Timestamp:      time.Now(),
				})
			}
		}(i, query)
	}

	// Wait for all searches to complete
	wg.Wait()

	// Check if any errors occurred
	if firstErr != nil {
		return BatchSearchResponse{}, firstErr
	}

	return BatchSearchResponse{
		Results:        results,
		StrategiesUsed: strategiesUsed,
		SearchTimes:    searchTimes,
		Stats:          stats,
	}, nil
}

// FluentHybridSearch provides a fluent API for building hybrid search queries
type FluentHybridSearch struct {
	index           *HybridIndex
	query           vectortypes.F32
	k               int
	forceStrategy   IndexType
	includeStats    bool
	negativeExample vectortypes.F32
	negativeWeight  float32
}

// FluentSearch creates a new fluent hybrid search builder
func (idx *HybridIndex) FluentSearch(query vectortypes.F32) *FluentHybridSearch {
	return &FluentHybridSearch{
		index:          idx,
		query:          query,
		k:              10, // Default to 10 results
		includeStats:   false,
		negativeWeight: 0.5, // Default weight when using negative examples
	}
}

// WithK sets the number of results to return
func (fhs *FluentHybridSearch) WithK(k int) *FluentHybridSearch {
	fhs.k = k
	return fhs
}

// WithStrategy forces a specific search strategy
func (fhs *FluentHybridSearch) WithStrategy(strategy IndexType) *FluentHybridSearch {
	fhs.forceStrategy = strategy
	return fhs
}

// IncludeStats specifies whether to include detailed stats in the response
func (fhs *FluentHybridSearch) IncludeStats(include bool) *FluentHybridSearch {
	fhs.includeStats = include
	return fhs
}

// WithNegativeExample adds a negative example vector to the search
// Results will be less similar to this vector
func (fhs *FluentHybridSearch) WithNegativeExample(vector vectortypes.F32) *FluentHybridSearch {
	fhs.negativeExample = vector
	return fhs
}

// WithNegativeWeight sets the weight for the negative example (0.0-1.0)
// Higher values give more importance to dissimilarity with the negative example
func (fhs *FluentHybridSearch) WithNegativeWeight(weight float32) *FluentHybridSearch {
	fhs.negativeWeight = weight
	return fhs
}

// Execute runs the search with the configured parameters
func (fhs *FluentHybridSearch) Execute() (HybridSearchResponse, error) {
	// Create a search request from the fluent parameters
	request := HybridSearchRequest{
		Query:           fhs.query,
		K:               fhs.k,
		ForceStrategy:   fhs.forceStrategy,
		IncludeStats:    fhs.includeStats,
		NegativeExample: fhs.negativeExample,
		NegativeWeight:  fhs.negativeWeight,
	}

	// Execute the search
	return fhs.index.SearchWithRequest(request)
}
