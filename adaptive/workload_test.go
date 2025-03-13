package adaptive

import (
	"testing"
	"time"
)

func TestWorkloadAnalyzer(t *testing.T) {
	// Create analyzer with default config
	analyzer := NewWorkloadAnalyzer(AnalyzerConfig{})

	// Test that default config is applied
	if analyzer.config.HistorySize != 10000 {
		t.Errorf("Expected default history size to be 10000, got %d", analyzer.config.HistorySize)
	}

	// Test recording queries
	for i := 0; i < 100; i++ {
		analyzer.RecordQuery(&QueryRecord{
			EfSearch:    100,
			K:           10,
			Filter:      i%2 == 0, // Every other query has a filter
			Timestamp:   time.Now().Add(-time.Duration(i) * time.Minute),
			Duration:    time.Duration(i+10) * time.Millisecond,
			ResultCount: i + 1,
		})
	}

	// Test analyzing workload
	analysis := analyzer.AnalyzeWorkload()

	// Verify analysis results
	if analysis.QueryCount != 100 {
		t.Errorf("Expected query count to be 100, got %d", analysis.QueryCount)
	}

	if analysis.MedianK != 10 {
		t.Errorf("Expected median K to be 10, got %d", analysis.MedianK)
	}

	if analysis.FilteredQueryRatio != 0.5 {
		t.Errorf("Expected filtered query ratio to be 0.5, got %f", analysis.FilteredQueryRatio)
	}

	// Test detecting workload shifts
	// First create a different workload
	newAnalyzer := NewWorkloadAnalyzer(AnalyzerConfig{})
	for i := 0; i < 100; i++ {
		newAnalyzer.RecordQuery(&QueryRecord{
			EfSearch:    100,
			K:           50,        // Higher K values
			Filter:      i%10 == 0, // Fewer filters
			Timestamp:   time.Now().Add(-time.Duration(i) * time.Minute),
			Duration:    time.Duration(i+10) * time.Millisecond,
			ResultCount: i + 1,
		})
	}

	// Analyze the new workload
	newAnalysis := newAnalyzer.AnalyzeWorkload()

	// Set the previous workload for comparison
	newAnalyzer.prevWorkload = analysis

	// Detect shifts
	shifts := newAnalyzer.DetectWorkloadShifts(newAnalysis)

	// Verify shifts were detected
	if len(shifts) == 0 {
		t.Errorf("Expected to detect workload shifts, but none were found")
	}

	// Check for specific shift types
	foundKShift := false
	foundFilterShift := false
	for _, shift := range shifts {
		if shift.Type == "k_value" {
			foundKShift = true
		}
		if shift.Type == "filter_usage" {
			foundFilterShift = true
		}
	}

	if !foundKShift {
		t.Errorf("Expected to detect a shift in K values")
	}

	if !foundFilterShift {
		t.Errorf("Expected to detect a shift in filter usage")
	}
}

func TestPercentileCalculation(t *testing.T) {
	// Test integer percentile calculation
	values := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

	p50 := percentile(values, 50)
	if p50 != 5 {
		t.Errorf("Expected 50th percentile to be 5, got %d", p50)
	}

	p95 := percentile(values, 95)
	if p95 != 10 {
		t.Errorf("Expected 95th percentile to be 10, got %d", p95)
	}

	// Test float percentile calculation
	floatValues := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}

	fp50 := percentileFloat(floatValues, 50)
	if fp50 != 5.0 {
		t.Errorf("Expected 50th percentile to be 5.0, got %f", fp50)
	}

	fp95 := percentileFloat(floatValues, 95)
	if fp95 != 10.0 {
		t.Errorf("Expected 95th percentile to be 10.0, got %f", fp95)
	}
}

func TestTemporalPatternDetection(t *testing.T) {
	analyzer := NewWorkloadAnalyzer(AnalyzerConfig{})

	// Create a workload with temporal patterns
	// More queries during business hours (9-17)
	now := time.Now()
	for hour := 0; hour < 24; hour++ {
		// Number of queries for this hour
		queryCount := 10
		if hour >= 9 && hour <= 17 {
			queryCount = 50 // 5x more queries during business hours
		}

		for i := 0; i < queryCount; i++ {
			timestamp := time.Date(
				now.Year(), now.Month(), now.Day(),
				hour, i%60, 0, 0, now.Location(),
			)

			analyzer.RecordQuery(&QueryRecord{
				EfSearch:    100,
				K:           10,
				Filter:      false,
				Timestamp:   timestamp,
				Duration:    10 * time.Millisecond,
				ResultCount: 10,
			})
		}
	}

	// Analyze the workload
	analysis := analyzer.AnalyzeWorkload()

	// Verify temporal patterns were detected
	if !analysis.HasTemporalPatterns {
		t.Errorf("Expected to detect temporal patterns, but none were found")
	}

	// Verify peak hours
	foundBusinessHours := false
	for _, hour := range analysis.PeakHours {
		if hour >= 9 && hour <= 17 {
			foundBusinessHours = true
			break
		}
	}

	if !foundBusinessHours {
		t.Errorf("Expected to detect business hours (9-17) as peak hours")
	}
}

func TestGetAnalysis(t *testing.T) {
	analyzer := NewWorkloadAnalyzer(AnalyzerConfig{})

	// Initially should return empty analysis
	analysis := analyzer.GetAnalysis()
	if analysis.QueryCount != 0 {
		t.Errorf("Expected initial query count to be 0, got %d", analysis.QueryCount)
	}

	// Record some queries
	for i := 0; i < 10; i++ {
		analyzer.RecordQuery(&QueryRecord{
			EfSearch:    100,
			K:           10,
			Filter:      false,
			Timestamp:   time.Now(),
			Duration:    10 * time.Millisecond,
			ResultCount: 10,
		})
	}

	// Analyze the workload
	analyzer.AnalyzeWorkload()

	// Get the analysis
	analysis = analyzer.GetAnalysis()

	// Verify analysis results
	if analysis.QueryCount != 10 {
		t.Errorf("Expected query count to be 10, got %d", analysis.QueryCount)
	}
}
