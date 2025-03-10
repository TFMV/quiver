package api

import (
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/valyala/fasthttp/fasthttpadaptor"
	"go.uber.org/zap"
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

// metricsHandler returns a handler for the /metrics endpoint
func metricsHandler(log *zap.Logger) fiber.Handler {
	return func(c *fiber.Ctx) error {
		// Convert the Prometheus handler to a Fiber handler
		handler := fasthttpadaptor.NewFastHTTPHandler(promhttp.Handler())
		handler(c.Context())
		return nil
	}
}

// recordMetrics updates the metrics based on the index state
func (s *Server) recordMetrics() {
	// Update metrics periodically
	ticker := time.NewTicker(15 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			// Get metrics from the index
			metrics := s.index.CollectMetrics()

			// Update vector count
			if count, ok := metrics["vector_count"].(int); ok {
				vectorCount.Set(float64(count))
			}

			// Update memory usage
			if memory, ok := metrics["memory_usage"].(uint64); ok {
				memoryUsage.Set(float64(memory))
			}
		}
	}
}

// ObserveSearchLatency records the latency of a search request
func ObserveSearchLatency(start time.Time, status string) {
	duration := time.Since(start).Seconds()
	searchLatency.WithLabelValues(status).Observe(duration)
	searchRequests.WithLabelValues(status).Inc()
}
