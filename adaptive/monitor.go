// File: quiver/adaptive/monitor.go
package adaptive

import (
	"container/ring"
	"sync"
	"time"
)

// PerformanceMonitor tracks system performance metrics
type PerformanceMonitor struct {
	// Configuration
	config MonitorConfig

	// Metrics storage
	searchLatency *MetricBuffer
	indexLatency  *MetricBuffer
	memoryUsage   *MetricBuffer
	cpuUsage      *MetricBuffer

	// Last report
	lastReport *PerformanceReport

	// Mutex for thread safety
	mu sync.RWMutex
}

// MonitorConfig contains configuration for the performance monitor
type MonitorConfig struct {
	// Size of metric buffers
	BufferSize int

	// Performance thresholds
	Thresholds PerformanceThresholds
}

// PerformanceThresholds defines thresholds for performance metrics
type PerformanceThresholds struct {
	// Maximum acceptable search latency (p95) in milliseconds
	MaxSearchLatencyMs float64

	// Maximum acceptable memory usage in bytes
	MaxMemoryBytes uint64

	// Maximum acceptable CPU usage percentage
	MaxCPUPercent float64
}

// MetricBuffer stores time series data for a metric
type MetricBuffer struct {
	// Ring buffer for storing values
	buffer *ring.Ring

	// Count of values added
	count int
}

// PerformanceReport contains the analysis of performance metrics
type PerformanceReport struct {
	// Time of the report
	Timestamp time.Time

	// Search latency metrics
	SearchLatency struct {
		P50 float64
		P95 float64
		P99 float64
	}

	// Index latency metrics
	IndexLatency struct {
		P50 float64
		P95 float64
	}

	// Resource usage
	MemoryUsage uint64
	CPUUsage    float64

	// Performance issues detected
	Issues []string
}

// MetricPoint represents a single data point in a time series
type MetricPoint struct {
	Timestamp time.Time
	Value     float64
	Labels    map[string]string
}

// NewPerformanceMonitor creates a new performance monitor
func NewPerformanceMonitor(config MonitorConfig) *PerformanceMonitor {
	// Use default config if needed
	if config.BufferSize == 0 {
		config.BufferSize = 1000
	}

	// Set default thresholds if not provided
	if config.Thresholds.MaxSearchLatencyMs == 0 {
		config.Thresholds.MaxSearchLatencyMs = 100.0 // 100ms
	}

	if config.Thresholds.MaxMemoryBytes == 0 {
		config.Thresholds.MaxMemoryBytes = 1024 * 1024 * 1024 // 1GB
	}

	if config.Thresholds.MaxCPUPercent == 0 {
		config.Thresholds.MaxCPUPercent = 80.0 // 80%
	}

	return &PerformanceMonitor{
		config:        config,
		searchLatency: newMetricBuffer(config.BufferSize),
		indexLatency:  newMetricBuffer(config.BufferSize),
		memoryUsage:   newMetricBuffer(config.BufferSize),
		cpuUsage:      newMetricBuffer(config.BufferSize),
	}
}

// newMetricBuffer creates a new metric buffer
func newMetricBuffer(size int) *MetricBuffer {
	return &MetricBuffer{
		buffer: ring.New(size),
	}
}

// RecordSearchMetrics records performance metrics for a search operation
func (pm *PerformanceMonitor) RecordSearchMetrics(duration time.Duration, efSearch int, resultCount int) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	// Record search latency
	pm.searchLatency.buffer.Value = &MetricPoint{
		Timestamp: time.Now(),
		Value:     float64(duration.Milliseconds()),
		Labels: map[string]string{
			"ef_search":    intToString(efSearch),
			"result_count": intToString(resultCount),
		},
	}
	pm.searchLatency.buffer = pm.searchLatency.buffer.Next()
	pm.searchLatency.count++
}

// RecordIndexMetrics records performance metrics for an index operation
func (pm *PerformanceMonitor) RecordIndexMetrics(op string, duration time.Duration) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	// Record index operation latency
	pm.indexLatency.buffer.Value = &MetricPoint{
		Timestamp: time.Now(),
		Value:     float64(duration.Milliseconds()),
		Labels: map[string]string{
			"operation": op,
		},
	}
	pm.indexLatency.buffer = pm.indexLatency.buffer.Next()
	pm.indexLatency.count++
}

// RecordMemoryUsage records memory usage
func (pm *PerformanceMonitor) RecordMemoryUsage(bytes uint64) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	// Record memory usage
	pm.memoryUsage.buffer.Value = &MetricPoint{
		Timestamp: time.Now(),
		Value:     float64(bytes),
	}
	pm.memoryUsage.buffer = pm.memoryUsage.buffer.Next()
	pm.memoryUsage.count++
}

// RecordCPUUsage records CPU usage
func (pm *PerformanceMonitor) RecordCPUUsage(percent float64) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	// Record CPU usage
	pm.cpuUsage.buffer.Value = &MetricPoint{
		Timestamp: time.Now(),
		Value:     percent,
	}
	pm.cpuUsage.buffer = pm.cpuUsage.buffer.Next()
	pm.cpuUsage.count++
}

// GenerateReport creates a performance report from collected metrics
func (pm *PerformanceMonitor) GenerateReport() *PerformanceReport {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	report := &PerformanceReport{
		Timestamp: time.Now(),
	}

	// Calculate search latency percentiles
	searchLatencies := pm.collectMetricValues(pm.searchLatency)
	if len(searchLatencies) > 0 {
		report.SearchLatency.P50 = percentileFloat(searchLatencies, 50)
		report.SearchLatency.P95 = percentileFloat(searchLatencies, 95)
		report.SearchLatency.P99 = percentileFloat(searchLatencies, 99)
	}

	// Calculate index latency percentiles
	indexLatencies := pm.collectMetricValues(pm.indexLatency)
	if len(indexLatencies) > 0 {
		report.IndexLatency.P50 = percentileFloat(indexLatencies, 50)
		report.IndexLatency.P95 = percentileFloat(indexLatencies, 95)
	}

	// Get latest memory usage
	memoryValues := pm.collectMetricValues(pm.memoryUsage)
	if len(memoryValues) > 0 {
		report.MemoryUsage = uint64(memoryValues[len(memoryValues)-1])
	}

	// Get latest CPU usage
	cpuValues := pm.collectMetricValues(pm.cpuUsage)
	if len(cpuValues) > 0 {
		report.CPUUsage = cpuValues[len(cpuValues)-1]
	}

	// Detect performance issues
	report.Issues = pm.detectPerformanceIssues(report)

	// Store the report
	pm.lastReport = report

	return report
}

// GetReport returns the most recent performance report
func (pm *PerformanceMonitor) GetReport() *PerformanceReport {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	if pm.lastReport == nil {
		return &PerformanceReport{
			Timestamp: time.Now(),
		}
	}

	return pm.lastReport
}

// HasIssues returns true if the report has any performance issues
func (pr *PerformanceReport) HasIssues() bool {
	return len(pr.Issues) > 0
}

// collectMetricValues extracts values from a metric buffer
func (pm *PerformanceMonitor) collectMetricValues(buffer *MetricBuffer) []float64 {
	var values []float64

	if buffer.count == 0 {
		return values
	}

	r := buffer.buffer
	for i := 0; i < buffer.buffer.Len(); i++ {
		if r.Value != nil {
			point := r.Value.(*MetricPoint)
			values = append(values, point.Value)
		}
		r = r.Next()
	}

	return values
}

// detectPerformanceIssues identifies performance problems
func (pm *PerformanceMonitor) detectPerformanceIssues(report *PerformanceReport) []string {
	var issues []string

	// Check search latency
	if report.SearchLatency.P95 > pm.config.Thresholds.MaxSearchLatencyMs {
		issues = append(issues, "high_search_latency")
	}

	// Check memory usage
	if report.MemoryUsage > pm.config.Thresholds.MaxMemoryBytes {
		issues = append(issues, "high_memory_usage")
	}

	// Check CPU usage
	if report.CPUUsage > pm.config.Thresholds.MaxCPUPercent {
		issues = append(issues, "high_cpu_usage")
	}

	return issues
}

// Helper function to convert int to string
func intToString(i int) string {
	// Simple implementation
	if i == 0 {
		return "0"
	}

	var digits []byte
	negative := false

	if i < 0 {
		negative = true
		i = -i
	}

	for i > 0 {
		digits = append(digits, byte('0'+i%10))
		i /= 10
	}

	// Reverse the digits
	for i, j := 0, len(digits)-1; i < j; i, j = i+1, j-1 {
		digits[i], digits[j] = digits[j], digits[i]
	}

	if negative {
		return "-" + string(digits)
	}

	return string(digits)
}
