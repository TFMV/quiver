package adaptive

import (
	"testing"
	"time"
)

func TestPerformanceMonitor(t *testing.T) {
	// Create monitor with default config
	monitor := NewPerformanceMonitor(MonitorConfig{})

	// Test that default config is applied
	if monitor.config.BufferSize != 1000 {
		t.Errorf("Expected default buffer size to be 1000, got %d", monitor.config.BufferSize)
	}

	if monitor.config.Thresholds.MaxSearchLatencyMs != 100.0 {
		t.Errorf("Expected default max search latency to be 100.0, got %f", monitor.config.Thresholds.MaxSearchLatencyMs)
	}

	// Test recording search metrics
	for i := 0; i < 100; i++ {
		monitor.RecordSearchMetrics(time.Duration(i+10)*time.Millisecond, 100, 10)
	}

	// Test recording index metrics
	for i := 0; i < 50; i++ {
		monitor.RecordIndexMetrics("add", time.Duration(i+20)*time.Millisecond)
	}

	// Test recording memory usage
	monitor.RecordMemoryUsage(1024 * 1024 * 100) // 100MB

	// Test recording CPU usage
	monitor.RecordCPUUsage(50.0)

	// Test generating report
	report := monitor.GenerateReport()

	// Verify report results
	if report.SearchLatency.P50 <= 0 {
		t.Errorf("Expected search latency p50 to be positive, got %f", report.SearchLatency.P50)
	}

	if report.SearchLatency.P95 <= 0 {
		t.Errorf("Expected search latency p95 to be positive, got %f", report.SearchLatency.P95)
	}

	if report.IndexLatency.P50 <= 0 {
		t.Errorf("Expected index latency p50 to be positive, got %f", report.IndexLatency.P50)
	}

	if report.MemoryUsage != 1024*1024*100 {
		t.Errorf("Expected memory usage to be 104857600, got %d", report.MemoryUsage)
	}

	if report.CPUUsage != 50.0 {
		t.Errorf("Expected CPU usage to be 50.0, got %f", report.CPUUsage)
	}

	// Test GetReport
	reportFromGet := monitor.GetReport()
	if reportFromGet.Timestamp.IsZero() {
		t.Errorf("Expected report timestamp to be non-zero")
	}
}

func TestPerformanceIssueDetection(t *testing.T) {
	// Create monitor with custom thresholds
	monitor := NewPerformanceMonitor(MonitorConfig{
		Thresholds: PerformanceThresholds{
			MaxSearchLatencyMs: 50.0,
			MaxMemoryBytes:     1024 * 1024 * 50, // 50MB
			MaxCPUPercent:      30.0,
		},
	})

	// Record metrics that exceed thresholds
	monitor.RecordSearchMetrics(60*time.Millisecond, 100, 10) // Exceeds latency threshold
	monitor.RecordMemoryUsage(1024 * 1024 * 100)              // Exceeds memory threshold
	monitor.RecordCPUUsage(40.0)                              // Exceeds CPU threshold

	// Generate report
	report := monitor.GenerateReport()

	// Verify issues were detected
	if !report.HasIssues() {
		t.Errorf("Expected to detect performance issues, but none were found")
	}

	// Check for specific issue types
	foundLatencyIssue := false
	foundMemoryIssue := false
	foundCPUIssue := false
	for _, issue := range report.Issues {
		if issue == "high_search_latency" {
			foundLatencyIssue = true
		}
		if issue == "high_memory_usage" {
			foundMemoryIssue = true
		}
		if issue == "high_cpu_usage" {
			foundCPUIssue = true
		}
	}

	if !foundLatencyIssue {
		t.Errorf("Expected to detect high search latency issue")
	}

	if !foundMemoryIssue {
		t.Errorf("Expected to detect high memory usage issue")
	}

	if !foundCPUIssue {
		t.Errorf("Expected to detect high CPU usage issue")
	}
}

func TestMetricBuffer(t *testing.T) {
	// Create a metric buffer
	buffer := newMetricBuffer(5)

	// Add values
	for i := 0; i < 10; i++ {
		buffer.buffer.Value = &MetricPoint{
			Timestamp: time.Now(),
			Value:     float64(i),
		}
		buffer.buffer = buffer.buffer.Next()
		buffer.count++
	}

	// Test that the buffer only keeps the last 5 values
	values := make([]float64, 0)
	r := buffer.buffer
	for i := 0; i < buffer.buffer.Len(); i++ {
		if r.Value != nil {
			point := r.Value.(*MetricPoint)
			values = append(values, point.Value)
		}
		r = r.Next()
	}

	if len(values) != 5 {
		t.Errorf("Expected buffer to contain 5 values, got %d", len(values))
	}

	// Verify the values are the last 5 (5-9)
	for i, value := range values {
		expected := float64(i + 5)
		if value != expected {
			t.Errorf("Expected value at index %d to be %f, got %f", i, expected, value)
		}
	}
}

func TestIntToString(t *testing.T) {
	testCases := []struct {
		input    int
		expected string
	}{
		{0, "0"},
		{123, "123"},
		{-456, "-456"},
		{1000000, "1000000"},
	}

	for _, tc := range testCases {
		result := intToString(tc.input)
		if result != tc.expected {
			t.Errorf("intToString(%d) = %s, expected %s", tc.input, result, tc.expected)
		}
	}
}
