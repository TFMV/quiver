package metrics

import (
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

// MetricsType defines the type of metric to collect
type MetricsType string

const (
	// LatencyMetric tracks query execution time
	LatencyMetric MetricsType = "latency"
	// RecallMetric tracks search accuracy
	RecallMetric MetricsType = "recall"
	// ThroughputMetric tracks query throughput
	ThroughputMetric MetricsType = "throughput"
	// CPUUsageMetric tracks CPU utilization
	CPUUsageMetric MetricsType = "cpu_usage"
	// MemoryUsageMetric tracks memory utilization
	MemoryUsageMetric MetricsType = "memory_usage"
)

// PerformanceMetrics holds various performance metrics
type PerformanceMetrics struct {
	// Average query latency in milliseconds
	AvgLatencyMs float64
	// Query throughput (queries per second)
	QPS float64
	// CPU utilization percentage
	CPUPercent float64
	// Memory usage in megabytes
	MemoryMB float64
	// Recall rate (if available)
	Recall float64
	// Time when metrics were collected
	Timestamp time.Time
}

// Collector manages the collection of metrics
type Collector struct {
	// Prometheus registry
	registry *prometheus.Registry
	// Query latency histogram
	queryLatency *prometheus.HistogramVec
	// Query throughput counter
	queries *prometheus.CounterVec
	// CPU usage gauge
	cpuUsage prometheus.Gauge
	// Memory usage gauge
	memoryUsage prometheus.Gauge
	// Recall metric gauge (if available)
	recall prometheus.Gauge
	// Optimization score gauge
	optimizationScore prometheus.Gauge
	// Whether Prometheus metrics are enabled
	prometheusEnabled bool
	// Lock for concurrent access
	mu sync.RWMutex
	// Recent metrics
	recentMetrics PerformanceMetrics
}

// NewCollector creates a new metrics collector
func NewCollector(prometheusEnabled bool) *Collector {
	c := &Collector{
		prometheusEnabled: prometheusEnabled,
		recentMetrics: PerformanceMetrics{
			Timestamp: time.Now(),
		},
		mu: sync.RWMutex{},
	}

	if prometheusEnabled {
		c.registry = prometheus.NewRegistry()

		// Create Prometheus metrics
		c.queryLatency = prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "quiver_query_latency_ms",
				Help:    "Query latency in milliseconds",
				Buckets: prometheus.ExponentialBuckets(1, 2, 10), // 1-512ms
			},
			[]string{"collection", "query_type"},
		)

		c.queries = prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "quiver_queries_total",
				Help: "Total number of queries executed",
			},
			[]string{"collection", "query_type"},
		)

		c.cpuUsage = prometheus.NewGauge(
			prometheus.GaugeOpts{
				Name: "quiver_cpu_usage_percent",
				Help: "CPU usage percentage",
			},
		)

		c.memoryUsage = prometheus.NewGauge(
			prometheus.GaugeOpts{
				Name: "quiver_memory_usage_mb",
				Help: "Memory usage in megabytes",
			},
		)

		c.recall = prometheus.NewGauge(
			prometheus.GaugeOpts{
				Name: "quiver_search_recall",
				Help: "Search recall rate (0-1)",
			},
		)

		c.optimizationScore = prometheus.NewGauge(
			prometheus.GaugeOpts{
				Name: "quiver_optimization_score",
				Help: "APT optimization score",
			},
		)

		// Register metrics with Prometheus
		c.registry.MustRegister(c.queryLatency)
		c.registry.MustRegister(c.queries)
		c.registry.MustRegister(c.cpuUsage)
		c.registry.MustRegister(c.memoryUsage)
		c.registry.MustRegister(c.recall)
		c.registry.MustRegister(c.optimizationScore)
	}

	return c
}

// RecordLatency records a query latency metric
func (c *Collector) RecordLatency(collection, queryType string, latencyMs float64) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.recentMetrics.AvgLatencyMs = (c.recentMetrics.AvgLatencyMs + latencyMs) / 2
	c.recentMetrics.Timestamp = time.Now()

	if c.prometheusEnabled {
		c.queryLatency.WithLabelValues(collection, queryType).Observe(latencyMs)
		c.queries.WithLabelValues(collection, queryType).Inc()
	}
}

// RecordSystemMetrics records system resource usage metrics
func (c *Collector) RecordSystemMetrics(cpuPercent, memoryMB float64) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.recentMetrics.CPUPercent = cpuPercent
	c.recentMetrics.MemoryMB = memoryMB
	c.recentMetrics.Timestamp = time.Now()

	if c.prometheusEnabled {
		c.cpuUsage.Set(cpuPercent)
		c.memoryUsage.Set(memoryMB)
	}
}

// RecordRecall records a search recall metric
func (c *Collector) RecordRecall(recall float64) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.recentMetrics.Recall = recall
	c.recentMetrics.Timestamp = time.Now()

	if c.prometheusEnabled {
		c.recall.Set(recall)
	}
}

// RecordOptimization records an optimization event
func (c *Collector) RecordOptimization(score float64, oldEfSearch, newEfSearch int) {
	if c.prometheusEnabled {
		c.optimizationScore.Set(score)
	}
}

// GetRecentMetrics retrieves the most recent metrics
func (c *Collector) GetRecentMetrics() PerformanceMetrics {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.recentMetrics
}

// GetRegistry returns the Prometheus registry
func (c *Collector) GetRegistry() *prometheus.Registry {
	return c.registry
}
