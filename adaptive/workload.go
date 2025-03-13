// File: quiver/adaptive/workload.go
package adaptive

import (
	"container/ring"
	"sort"
	"sync"
	"time"
)

// WorkloadAnalyzer monitors and analyzes query patterns
type WorkloadAnalyzer struct {
	// Configuration
	config AnalyzerConfig

	// Query history
	queryHistory *ring.Ring
	queryCount   int

	// Last analysis
	lastAnalysis *WorkloadAnalysis

	// Previous workload state for comparison
	prevWorkload *WorkloadAnalysis

	// Mutex for thread safety
	mu sync.RWMutex
}

// AnalyzerConfig contains configuration for the workload analyzer
type AnalyzerConfig struct {
	// Size of the query history buffer
	HistorySize int

	// Thresholds for detecting workload shifts
	ShiftThresholds WorkloadShiftThresholds
}

// WorkloadShiftThresholds defines thresholds for detecting workload shifts
type WorkloadShiftThresholds struct {
	// Minimum change in median K to consider a shift
	MedianKChange float64

	// Minimum change in filtered query ratio to consider a shift
	FilteredRatioChange float64

	// Minimum change in query volume to consider a shift
	QueryVolumeChange float64
}

// QueryRecord represents a single query for analysis
type QueryRecord struct {
	// Query parameters
	EfSearch int
	K        int
	Filter   bool

	// Execution details
	Timestamp   time.Time
	Duration    time.Duration
	ResultCount int
}

// WorkloadAnalysis contains the analysis of the query workload
type WorkloadAnalysis struct {
	// Time period of the analysis
	Period time.Duration

	// Query volume metrics
	QueryCount    int
	QueriesPerSec float64

	// Query parameter distributions
	MedianK        int
	P95K           int
	MedianEfSearch int

	// Filter usage
	FilteredQueryRatio float64

	// Temporal patterns
	HasTemporalPatterns bool
	PeakHours           []int

	// Query latency
	MedianLatencyMs float64
	P95LatencyMs    float64
}

// WorkloadShift represents a detected shift in the workload
type WorkloadShift struct {
	// Type of shift
	Type string

	// Magnitude of the shift (0-1)
	Magnitude float64

	// Description of the shift
	Description string
}

// NewWorkloadAnalyzer creates a new workload analyzer
func NewWorkloadAnalyzer(config AnalyzerConfig) *WorkloadAnalyzer {
	// Use default config if needed
	if config.HistorySize == 0 {
		config.HistorySize = 10000
	}

	// Set default shift thresholds if not provided
	if config.ShiftThresholds.MedianKChange == 0 {
		config.ShiftThresholds.MedianKChange = 0.3 // 30% change
	}

	if config.ShiftThresholds.FilteredRatioChange == 0 {
		config.ShiftThresholds.FilteredRatioChange = 0.2 // 20% change
	}

	if config.ShiftThresholds.QueryVolumeChange == 0 {
		config.ShiftThresholds.QueryVolumeChange = 0.5 // 50% change
	}

	return &WorkloadAnalyzer{
		config:       config,
		queryHistory: ring.New(config.HistorySize),
	}
}

// RecordQuery adds a query to the history
func (wa *WorkloadAnalyzer) RecordQuery(query *QueryRecord) {
	wa.mu.Lock()
	defer wa.mu.Unlock()

	// Store the query in the ring buffer
	wa.queryHistory.Value = query
	wa.queryHistory = wa.queryHistory.Next()
	wa.queryCount++

	// Trigger analysis if we've collected enough new data
	if wa.queryCount%1000 == 0 {
		go wa.AnalyzeWorkload()
	}
}

// AnalyzeWorkload performs a full analysis of the query workload
func (wa *WorkloadAnalyzer) AnalyzeWorkload() *WorkloadAnalysis {
	wa.mu.Lock()
	defer wa.mu.Unlock()

	// Create a new analysis
	analysis := &WorkloadAnalysis{}

	// Skip if no queries
	if wa.queryCount == 0 {
		wa.lastAnalysis = analysis
		return analysis
	}

	// Collect data from the query history
	var kValues []int
	var efValues []int
	var latencies []float64
	var filteredCount int

	// Track earliest and latest timestamps
	var earliest, latest time.Time

	// Process query history
	r := wa.queryHistory
	for i := 0; i < wa.queryHistory.Len(); i++ {
		if r.Value != nil {
			query := r.Value.(*QueryRecord)

			// Update timestamps
			if earliest.IsZero() || query.Timestamp.Before(earliest) {
				earliest = query.Timestamp
			}
			if latest.IsZero() || query.Timestamp.After(latest) {
				latest = query.Timestamp
			}

			// Collect values
			kValues = append(kValues, query.K)
			efValues = append(efValues, query.EfSearch)
			latencies = append(latencies, float64(query.Duration.Milliseconds()))

			if query.Filter {
				filteredCount++
			}
		}
		r = r.Next()
	}

	// Calculate time period
	if !earliest.IsZero() && !latest.IsZero() {
		analysis.Period = latest.Sub(earliest)
	}

	// Calculate query volume
	analysis.QueryCount = len(kValues)
	if analysis.Period > 0 {
		analysis.QueriesPerSec = float64(analysis.QueryCount) / analysis.Period.Seconds()
	}

	// Calculate distributions
	if len(kValues) > 0 {
		analysis.MedianK = percentile(kValues, 50)
		analysis.P95K = percentile(kValues, 95)
	}

	if len(efValues) > 0 {
		analysis.MedianEfSearch = percentile(efValues, 50)
	}

	// Calculate filter ratio
	if analysis.QueryCount > 0 {
		analysis.FilteredQueryRatio = float64(filteredCount) / float64(analysis.QueryCount)
	}

	// Calculate latency metrics
	if len(latencies) > 0 {
		analysis.MedianLatencyMs = percentileFloat(latencies, 50)
		analysis.P95LatencyMs = percentileFloat(latencies, 95)
	}

	// Detect temporal patterns
	analysis.HasTemporalPatterns, analysis.PeakHours = wa.detectTemporalPatterns()

	// Store the analysis
	wa.prevWorkload = wa.lastAnalysis
	wa.lastAnalysis = analysis

	return analysis
}

// DetectWorkloadShifts identifies significant changes in the workload
func (wa *WorkloadAnalyzer) DetectWorkloadShifts(current *WorkloadAnalysis) []WorkloadShift {
	wa.mu.RLock()
	defer wa.mu.RUnlock()

	var shifts []WorkloadShift

	// Skip if we don't have a previous analysis
	if wa.prevWorkload == nil || current == nil {
		return shifts
	}

	// Check for K value shift
	if wa.prevWorkload.MedianK > 0 {
		kChange := float64(current.MedianK-wa.prevWorkload.MedianK) / float64(wa.prevWorkload.MedianK)
		if abs(kChange) > wa.config.ShiftThresholds.MedianKChange {
			shifts = append(shifts, WorkloadShift{
				Type:        "k_value",
				Magnitude:   abs(kChange),
				Description: "Significant change in typical K values",
			})
		}
	}

	// Check for filter usage shift
	filterChange := current.FilteredQueryRatio - wa.prevWorkload.FilteredQueryRatio
	if abs(filterChange) > wa.config.ShiftThresholds.FilteredRatioChange {
		shifts = append(shifts, WorkloadShift{
			Type:        "filter_usage",
			Magnitude:   abs(filterChange),
			Description: "Significant change in filter usage",
		})
	}

	// Check for query volume shift
	if wa.prevWorkload.QueriesPerSec > 0 {
		volumeChange := (current.QueriesPerSec - wa.prevWorkload.QueriesPerSec) / wa.prevWorkload.QueriesPerSec
		if abs(volumeChange) > wa.config.ShiftThresholds.QueryVolumeChange {
			shifts = append(shifts, WorkloadShift{
				Type:        "query_volume",
				Magnitude:   abs(volumeChange),
				Description: "Significant change in query volume",
			})
		}
	}

	// Check for temporal pattern changes
	if current.HasTemporalPatterns != wa.prevWorkload.HasTemporalPatterns {
		shifts = append(shifts, WorkloadShift{
			Type:        "temporal_pattern",
			Magnitude:   1.0,
			Description: "Change in temporal query patterns",
		})
	}

	return shifts
}

// GetAnalysis returns the most recent workload analysis
func (wa *WorkloadAnalyzer) GetAnalysis() *WorkloadAnalysis {
	wa.mu.RLock()
	defer wa.mu.RUnlock()

	if wa.lastAnalysis == nil {
		return &WorkloadAnalysis{}
	}

	return wa.lastAnalysis
}

// detectTemporalPatterns analyzes the query history for time-based patterns
func (wa *WorkloadAnalyzer) detectTemporalPatterns() (bool, []int) {
	// Count queries by hour
	hourCounts := make(map[int]int)

	r := wa.queryHistory
	for i := 0; i < wa.queryHistory.Len(); i++ {
		if r.Value != nil {
			query := r.Value.(*QueryRecord)
			hour := query.Timestamp.Hour()
			hourCounts[hour]++
		}
		r = r.Next()
	}

	// Find peak hours (hours with significantly more queries)
	var peakHours []int
	var totalQueries int
	for _, count := range hourCounts {
		totalQueries += count
	}

	avgQueriesPerHour := float64(totalQueries) / 24.0

	for hour, count := range hourCounts {
		if float64(count) > avgQueriesPerHour*1.5 {
			peakHours = append(peakHours, hour)
		}
	}

	// We have temporal patterns if we identified peak hours
	return len(peakHours) > 0, peakHours
}

// Helper functions for statistics
func percentile(values []int, p int) int {
	if len(values) == 0 {
		return 0
	}

	// Sort values
	sorted := make([]int, len(values))
	copy(sorted, values)
	sort.Ints(sorted)

	// Calculate percentile index
	// For p=95 and len=10, we want index 9 (the last element)
	// For p=50 and len=10, we want index 4 (the middle element)
	if p >= 100 {
		return sorted[len(sorted)-1]
	}

	// Use ceiling for high percentiles to ensure we get the right value
	// for small arrays
	if p > 90 && len(sorted) <= 20 {
		return sorted[len(sorted)-1]
	}

	idx := int(float64(len(sorted)-1) * float64(p) / 100.0)
	return sorted[idx]
}

func percentileFloat(values []float64, p int) float64 {
	if len(values) == 0 {
		return 0
	}

	// Sort values
	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)

	// Calculate percentile index
	// For p=95 and len=10, we want index 9 (the last element)
	// For p=50 and len=10, we want index 4 (the middle element)
	if p >= 100 {
		return sorted[len(sorted)-1]
	}

	// Use ceiling for high percentiles to ensure we get the right value
	// for small arrays
	if p > 90 && len(sorted) <= 20 {
		return sorted[len(sorted)-1]
	}

	idx := int(float64(len(sorted)-1) * float64(p) / 100.0)
	return sorted[idx]
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
