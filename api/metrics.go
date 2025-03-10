package api

import (
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

var (
	// Define Prometheus metrics
	searchRequests = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "quiver_search_requests_total",
			Help: "Total number of search requests",
		},
		[]string{"status"},
	)

	searchLatency = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "quiver_search_latency_seconds",
			Help:    "Search request latency in seconds",
			Buckets: prometheus.ExponentialBuckets(0.001, 2, 15), // from 1ms to ~16s
		},
		[]string{"status"},
	)

	vectorCount = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Name: "quiver_vector_count",
			Help: "Number of vectors in the index",
		},
	)

	memoryUsage = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Name: "quiver_memory_usage_bytes",
			Help: "Memory usage in bytes",
		},
	)
)

func init() {
	// Register metrics with Prometheus
	prometheus.MustRegister(searchRequests)
	prometheus.MustRegister(searchLatency)
	prometheus.MustRegister(vectorCount)
	prometheus.MustRegister(memoryUsage)
}

// ObserveSearchLatency records the latency of a search request
func ObserveSearchLatency(start time.Time, status string) {
	duration := time.Since(start).Seconds()
	searchLatency.WithLabelValues(status).Observe(duration)
	searchRequests.WithLabelValues(status).Inc()
}
