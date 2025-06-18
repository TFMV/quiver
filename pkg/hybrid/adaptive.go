package hybrid

import (
	"fmt"
	"math/rand"
	"sync"
	"time"
)

// AdaptiveStrategySelector implements adaptive selection of search strategies
type AdaptiveStrategySelector struct {
	// Configuration for the adaptive selector
	config AdaptiveConfig

	// Current thresholds based on learning
	exactThreshold int
	dimThreshold   int

	// Performance metrics for strategies
	metrics map[IndexType]*StrategyStats

	// Recent query metrics for analysis
	recentQueries []QueryMetrics

	// Random number generator for exploration
	rng *rand.Rand

	// Lock for thread safety
	mu sync.RWMutex
}

// NewAdaptiveStrategySelector creates a new adaptive strategy selector
func NewAdaptiveStrategySelector(config AdaptiveConfig) *AdaptiveStrategySelector {
	return &AdaptiveStrategySelector{
		config:         config,
		exactThreshold: config.InitialExactThreshold,
		dimThreshold:   config.InitialDimThreshold,
		metrics:        make(map[IndexType]*StrategyStats),
		recentQueries:  make([]QueryMetrics, 0, config.MetricsWindowSize),
		rng:            rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// SelectStrategy chooses the best strategy for a given query context
func (a *AdaptiveStrategySelector) SelectStrategy(vectorCount, dimension, k int) IndexType {
	a.mu.RLock()
	defer a.mu.RUnlock()

	// Exploration: sometimes try a random strategy to gather performance data
	if a.rng.Float64() < a.config.ExplorationFactor {
		if a.rng.Float64() < 0.5 {
			return ExactIndexType
		}
		return HNSWIndexType
	}

	// Exploitation: use learned thresholds to select strategy

	// For small datasets, exact search is often faster
	if vectorCount < a.exactThreshold {
		return ExactIndexType
	}

	// For high-dimensional data, consider the dimension threshold
	if dimension > a.dimThreshold {
		// For high-dimensional data with small k, HNSW usually performs better
		if k < 50 {
			return HNSWIndexType
		}
		// For high-dimensional data with large k, exact search might be better
		return ExactIndexType
	}

	// Default to HNSW for large datasets with low-to-medium dimensions
	return HNSWIndexType
}

// RecordQueryMetrics records performance metrics for a query
func (a *AdaptiveStrategySelector) RecordQueryMetrics(metrics QueryMetrics) {
	a.mu.Lock()
	defer a.mu.Unlock()

	// Initialize strategy stats if needed
	if _, exists := a.metrics[metrics.Strategy]; !exists {
		a.metrics[metrics.Strategy] = &StrategyStats{
			UsageCount:    0,
			TotalDuration: 0,
			AvgDuration:   0,
		}
	}

	// Update strategy stats
	stats := a.metrics[metrics.Strategy]
	stats.UsageCount++
	stats.TotalDuration += metrics.Duration
	stats.AvgDuration = stats.TotalDuration / time.Duration(stats.UsageCount)

	// Add to recent queries
	a.recentQueries = append(a.recentQueries, metrics)
	if len(a.recentQueries) > a.config.MetricsWindowSize {
		// Remove oldest query when we exceed window size
		a.recentQueries = a.recentQueries[1:]
	}

	// Adapt thresholds periodically
	if stats.UsageCount%20 == 0 && len(a.recentQueries) >= 10 {
		a.adaptThresholds()
	}
}

// adaptThresholds adjusts the thresholds based on observed performance
func (a *AdaptiveStrategySelector) adaptThresholds() {
	// Only adapt if we have stats for both strategies
	exactStats, hasExact := a.metrics[ExactIndexType]
	hnswStats, hasHNSW := a.metrics[HNSWIndexType]

	if !hasExact || !hasHNSW || exactStats.UsageCount < 10 || hnswStats.UsageCount < 10 {
		// Not enough data to adapt yet
		return
	}

	// Compare average performance of strategies
	// If exact is faster for larger datasets than our current threshold,
	// increase the threshold
	_ = exactStats.AvgDuration < hnswStats.AvgDuration

	// Analyze recent queries to find patterns
	var (
		smallDatasetExactAvg   time.Duration
		smallDatasetHNSWAvg    time.Duration
		smallDatasetExactCount int
		smallDatasetHNSWCount  int

		largeDatasetExactAvg   time.Duration
		largeDatasetHNSWAvg    time.Duration
		largeDatasetExactCount int
		largeDatasetHNSWCount  int
	)

	// Analyze recent queries by dataset size
	for _, q := range a.recentQueries {
		isSmall := q.ResultCount < a.exactThreshold

		if q.Strategy == ExactIndexType {
			if isSmall {
				smallDatasetExactAvg += q.Duration
				smallDatasetExactCount++
			} else {
				largeDatasetExactAvg += q.Duration
				largeDatasetExactCount++
			}
		} else if q.Strategy == HNSWIndexType {
			if isSmall {
				smallDatasetHNSWAvg += q.Duration
				smallDatasetHNSWCount++
			} else {
				largeDatasetHNSWAvg += q.Duration
				largeDatasetHNSWCount++
			}
		}
	}

	// Calculate averages
	if smallDatasetExactCount > 0 {
		smallDatasetExactAvg /= time.Duration(smallDatasetExactCount)
	}
	if smallDatasetHNSWCount > 0 {
		smallDatasetHNSWAvg /= time.Duration(smallDatasetHNSWCount)
	}
	// Only calculate these if they will be used later
	// For now these are calculated but not used, so we'll comment them out
	/*
		if largeDatasetExactCount > 0 {
			largeDatasetExactAvg /= time.Duration(largeDatasetExactCount)
		}
		if largeDatasetHNSWCount > 0 {
			largeDatasetHNSWAvg /= time.Duration(largeDatasetHNSWCount)
		}
	*/

	// Adapt exact threshold based on performance for small vs large datasets
	if smallDatasetExactCount > 5 && smallDatasetHNSWCount > 5 {
		if smallDatasetExactAvg < smallDatasetHNSWAvg {
			// Exact is faster for small datasets, increase threshold
			delta := int(float64(a.exactThreshold) * a.config.AdaptationRate)
			if delta < 10 {
				delta = 10
			}
			a.exactThreshold += delta
		} else {
			// HNSW is faster for small datasets, decrease threshold
			delta := int(float64(a.exactThreshold) * a.config.AdaptationRate)
			if delta < 10 {
				delta = 10
			}
			a.exactThreshold -= delta

			// Ensure threshold doesn't go below a reasonable minimum
			if a.exactThreshold < 100 {
				a.exactThreshold = 100
			}
		}
	}

	// Similarly, adapt dimension threshold based on performance for different dimensions
	// This would require additional analysis of performance by dimension
	// For simplicity, we'll keep this part as a future enhancement
}

// GetStats returns statistics about the adaptive selector
func (a *AdaptiveStrategySelector) GetStats() map[string]interface{} {
	a.mu.RLock()
	defer a.mu.RUnlock()

	strategyStats := make(map[string]interface{})
	for k, v := range a.metrics {
		strategyStats[string(k)] = v
	}

	return map[string]interface{}{
		"thresholds": map[string]interface{}{
			"exact":     a.exactThreshold,
			"dimension": a.dimThreshold,
		},
		"strategies":           strategyStats,
		"config":               a.config,
		"recent_queries_count": len(a.recentQueries),
	}
}

// String provides a string representation of the adaptive selector's state
func (a *AdaptiveStrategySelector) String() string {
	a.mu.RLock()
	defer a.mu.RUnlock()

	exactStats, hasExact := a.metrics[ExactIndexType]
	hnswStats, hasHNSW := a.metrics[HNSWIndexType]

	exactAvg := time.Duration(0)
	hnswAvg := time.Duration(0)

	if hasExact && exactStats.UsageCount > 0 {
		exactAvg = exactStats.AvgDuration
	}

	if hasHNSW && hnswStats.UsageCount > 0 {
		hnswAvg = hnswStats.AvgDuration
	}

	return fmt.Sprintf(
		"AdaptiveStrategySelector{exactThreshold=%d, dimThreshold=%d, exactAvg=%v, hnswAvg=%v}",
		a.exactThreshold,
		a.dimThreshold,
		exactAvg,
		hnswAvg,
	)
}

// UpdateThresholds updates the internal thresholds based on index statistics.
func (a *AdaptiveStrategySelector) UpdateThresholds(exact, dim int) {
	a.mu.Lock()
	a.exactThreshold = exact
	a.dimThreshold = dim
	a.mu.Unlock()
}
