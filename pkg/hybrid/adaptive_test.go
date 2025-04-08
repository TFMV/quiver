package hybrid

import (
	"testing"
	"time"
)

func TestNewAdaptiveStrategySelector(t *testing.T) {
	config := DefaultAdaptiveConfig()

	// Modify some values to ensure they're properly set
	config.ExplorationFactor = 0.2
	config.InitialExactThreshold = 2000
	config.InitialDimThreshold = 150

	selector := NewAdaptiveStrategySelector(config)

	if selector == nil {
		t.Fatal("NewAdaptiveStrategySelector returned nil")
	}

	if selector.config.ExplorationFactor != 0.2 {
		t.Errorf("Expected ExplorationFactor 0.2, got %f", selector.config.ExplorationFactor)
	}

	if selector.exactThreshold != 2000 {
		t.Errorf("Expected exactThreshold 2000, got %d", selector.exactThreshold)
	}

	if selector.dimThreshold != 150 {
		t.Errorf("Expected dimThreshold 150, got %d", selector.dimThreshold)
	}

	if selector.metrics == nil {
		t.Error("metrics map not initialized")
	}

	if selector.recentQueries == nil {
		t.Error("recentQueries slice not initialized")
	}

	if selector.rng == nil {
		t.Error("random number generator not initialized")
	}
}

func TestSelectStrategy_SmallDataset(t *testing.T) {
	config := DefaultAdaptiveConfig()
	// Set exploration to 0 to make tests deterministic
	config.ExplorationFactor = 0
	config.InitialExactThreshold = 1000

	selector := NewAdaptiveStrategySelector(config)

	// For small datasets, should prefer exact search
	strategy := selector.SelectStrategy(500, 100, 10)
	if strategy != ExactIndexType {
		t.Errorf("Expected ExactIndexType for small dataset, got %s", strategy)
	}
}

func TestSelectStrategy_LargeDataset(t *testing.T) {
	config := DefaultAdaptiveConfig()
	// Set exploration to 0 to make tests deterministic
	config.ExplorationFactor = 0
	config.InitialExactThreshold = 1000

	selector := NewAdaptiveStrategySelector(config)

	// For large datasets with medium dimensions, should prefer HNSW
	strategy := selector.SelectStrategy(5000, 100, 10)
	if strategy != HNSWIndexType {
		t.Errorf("Expected HNSWIndexType for large dataset, got %s", strategy)
	}
}

func TestSelectStrategy_HighDimensionalSmallK(t *testing.T) {
	config := DefaultAdaptiveConfig()
	// Set exploration to 0 to make tests deterministic
	config.ExplorationFactor = 0
	config.InitialExactThreshold = 1000
	config.InitialDimThreshold = 100

	selector := NewAdaptiveStrategySelector(config)

	// For high dimensional data with small k, should prefer HNSW
	strategy := selector.SelectStrategy(5000, 150, 10)
	if strategy != HNSWIndexType {
		t.Errorf("Expected HNSWIndexType for high-dim small-k, got %s", strategy)
	}
}

func TestSelectStrategy_HighDimensionalLargeK(t *testing.T) {
	config := DefaultAdaptiveConfig()
	// Set exploration to 0 to make tests deterministic
	config.ExplorationFactor = 0
	config.InitialExactThreshold = 1000
	config.InitialDimThreshold = 100

	selector := NewAdaptiveStrategySelector(config)

	// For high dimensional data with large k, should prefer exact search
	strategy := selector.SelectStrategy(5000, 150, 100)
	if strategy != ExactIndexType {
		t.Errorf("Expected ExactIndexType for high-dim large-k, got %s", strategy)
	}
}

func TestRecordQueryMetrics_InitializeStats(t *testing.T) {
	selector := NewAdaptiveStrategySelector(DefaultAdaptiveConfig())

	// Record a metric for a strategy that hasn't been seen before
	metrics := QueryMetrics{
		Strategy:       ExactIndexType,
		QueryDimension: 128,
		K:              10,
		Duration:       100 * time.Millisecond,
		ResultCount:    5,
		Timestamp:      time.Now(),
	}

	selector.RecordQueryMetrics(metrics)

	// Check that stats were initialized
	if _, exists := selector.metrics[ExactIndexType]; !exists {
		t.Fatalf("Expected stats for %s to be initialized", ExactIndexType)
	}

	stats := selector.metrics[ExactIndexType]
	if stats.UsageCount != 1 {
		t.Errorf("Expected UsageCount 1, got %d", stats.UsageCount)
	}

	if stats.TotalDuration != 100*time.Millisecond {
		t.Errorf("Expected TotalDuration 100ms, got %v", stats.TotalDuration)
	}

	if stats.AvgDuration != 100*time.Millisecond {
		t.Errorf("Expected AvgDuration 100ms, got %v", stats.AvgDuration)
	}

	// Check that recent queries was updated
	if len(selector.recentQueries) != 1 {
		t.Errorf("Expected 1 recent query, got %d", len(selector.recentQueries))
	}
}

func TestRecordQueryMetrics_UpdateStats(t *testing.T) {
	selector := NewAdaptiveStrategySelector(DefaultAdaptiveConfig())

	// Record multiple metrics for the same strategy
	metrics1 := QueryMetrics{
		Strategy:       HNSWIndexType,
		QueryDimension: 128,
		K:              10,
		Duration:       100 * time.Millisecond,
		ResultCount:    5,
		Timestamp:      time.Now(),
	}

	metrics2 := QueryMetrics{
		Strategy:       HNSWIndexType,
		QueryDimension: 128,
		K:              10,
		Duration:       200 * time.Millisecond,
		ResultCount:    5,
		Timestamp:      time.Now(),
	}

	selector.RecordQueryMetrics(metrics1)
	selector.RecordQueryMetrics(metrics2)

	// Check that stats were updated
	stats := selector.metrics[HNSWIndexType]
	if stats.UsageCount != 2 {
		t.Errorf("Expected UsageCount 2, got %d", stats.UsageCount)
	}

	if stats.TotalDuration != 300*time.Millisecond {
		t.Errorf("Expected TotalDuration 300ms, got %v", stats.TotalDuration)
	}

	if stats.AvgDuration != 150*time.Millisecond {
		t.Errorf("Expected AvgDuration 150ms, got %v", stats.AvgDuration)
	}

	// Check that recent queries contains both
	if len(selector.recentQueries) != 2 {
		t.Errorf("Expected 2 recent queries, got %d", len(selector.recentQueries))
	}
}

func TestRecordQueryMetrics_LimitRecentQueries(t *testing.T) {
	config := DefaultAdaptiveConfig()
	config.MetricsWindowSize = 3 // Small window for testing
	selector := NewAdaptiveStrategySelector(config)

	// Add more queries than the window size
	for i := 0; i < 5; i++ {
		metrics := QueryMetrics{
			Strategy:       ExactIndexType,
			QueryDimension: 128,
			K:              10,
			Duration:       time.Duration(i+1) * 10 * time.Millisecond,
			ResultCount:    5,
			Timestamp:      time.Now(),
		}
		selector.RecordQueryMetrics(metrics)
	}

	// Check that only the most recent window size queries are kept
	if len(selector.recentQueries) != 3 {
		t.Errorf("Expected %d recent queries, got %d", config.MetricsWindowSize, len(selector.recentQueries))
	}

	// The oldest queries should be dropped, so the first duration should be 30ms (4th query)
	if selector.recentQueries[0].Duration != 30*time.Millisecond {
		t.Errorf("Expected first query duration 30ms, got %v", selector.recentQueries[0].Duration)
	}

	// The newest query should be 50ms (5th query)
	if selector.recentQueries[2].Duration != 50*time.Millisecond {
		t.Errorf("Expected last query duration 50ms, got %v", selector.recentQueries[2].Duration)
	}
}

func TestGetStats(t *testing.T) {
	selector := NewAdaptiveStrategySelector(DefaultAdaptiveConfig())

	// Add some test metrics
	exactMetrics := QueryMetrics{
		Strategy:       ExactIndexType,
		QueryDimension: 128,
		K:              10,
		Duration:       50 * time.Millisecond,
		ResultCount:    5,
		Timestamp:      time.Now(),
	}

	hnswMetrics := QueryMetrics{
		Strategy:       HNSWIndexType,
		QueryDimension: 256,
		K:              50,
		Duration:       30 * time.Millisecond,
		ResultCount:    10,
		Timestamp:      time.Now(),
	}

	selector.RecordQueryMetrics(exactMetrics)
	selector.RecordQueryMetrics(hnswMetrics)

	// Get stats
	stats := selector.GetStats()

	// Check basic stats
	if thresholds, ok := stats["thresholds"].(map[string]interface{}); ok {
		if exactThreshold, exists := thresholds["exact"]; !exists || exactThreshold != selector.exactThreshold {
			t.Errorf("Expected exact threshold %d, got %v", selector.exactThreshold, exactThreshold)
		}

		if dimThreshold, exists := thresholds["dimension"]; !exists || dimThreshold != selector.dimThreshold {
			t.Errorf("Expected dimension threshold %d, got %v", selector.dimThreshold, dimThreshold)
		}
	} else {
		t.Error("Expected thresholds in stats")
	}

	// Check strategies stats
	if strategyStats, ok := stats["strategies"].(map[string]interface{}); ok {
		if _, exists := strategyStats[string(ExactIndexType)]; !exists {
			t.Errorf("Expected stats for %s", ExactIndexType)
		}

		if _, exists := strategyStats[string(HNSWIndexType)]; !exists {
			t.Errorf("Expected stats for %s", HNSWIndexType)
		}
	} else {
		t.Error("Expected strategies in stats")
	}

	// Check config
	if _, ok := stats["config"]; !ok {
		t.Error("Expected config in stats")
	}

	// Check recent queries
	if recentCount, ok := stats["recent_queries_count"].(int); !ok || recentCount != 2 {
		t.Errorf("Expected recent_queries_count 2, got %v", stats["recent_queries_count"])
	}
}

func TestString(t *testing.T) {
	selector := NewAdaptiveStrategySelector(DefaultAdaptiveConfig())

	// Add some test metrics
	selector.RecordQueryMetrics(QueryMetrics{
		Strategy:       ExactIndexType,
		QueryDimension: 128,
		K:              10,
		Duration:       50 * time.Millisecond,
		ResultCount:    5,
		Timestamp:      time.Now(),
	})

	// Just check that String() doesn't panic and returns something
	str := selector.String()
	if str == "" {
		t.Error("Expected non-empty string representation")
	}
}
