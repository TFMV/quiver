package observability

import (
	"math"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

type Metrics struct {
	enabled atomic.Bool

	searchLatency *prometheus.HistogramVec
	searchTotal   *prometheus.CounterVec
	searchErrors  *prometheus.CounterVec

	insertLatency *prometheus.HistogramVec
	insertTotal   *prometheus.CounterVec
	insertErrors  *prometheus.CounterVec

	batchLatency *prometheus.HistogramVec
	batchTotal   *prometheus.CounterVec
	batchErrors  *prometheus.CounterVec

	filterLatency    *prometheus.HistogramVec
	traversalLatency *prometheus.HistogramVec
	rerankLatency    *prometheus.HistogramVec

	indexSize      *prometheus.GaugeVec
	indexVectors   *prometheus.GaugeVec
	indexSizeBytes *prometheus.GaugeVec

	reg *prometheus.Registry
	mu  sync.RWMutex

	inMemoryLatencies map[string][]time.Duration
	inMemoryMu        sync.RWMutex
}

var globalMetrics *Metrics

func init() {
	globalMetrics = NewMetrics()
}

func GlobalMetrics() *Metrics {
	return globalMetrics
}

func NewMetrics() *Metrics {
	m := &Metrics{
		inMemoryLatencies: make(map[string][]time.Duration),
		reg:               prometheus.NewRegistry(),
	}

	m.searchLatency = promauto.With(m.reg).NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "quiver_search_latency_ms",
			Help:    "Search operation latency in milliseconds",
			Buckets: []float64{0.1, 0.5, 1, 5, 10, 25, 50, 100, 250, 500, 1000},
		},
		[]string{"collection", "stage"},
	)

	m.searchTotal = promauto.With(m.reg).NewCounterVec(
		prometheus.CounterOpts{
			Name: "quiver_search_total",
			Help: "Total number of search operations",
		},
		[]string{"collection"},
	)

	m.searchErrors = promauto.With(m.reg).NewCounterVec(
		prometheus.CounterOpts{
			Name: "quiver_search_errors_total",
			Help: "Total number of search errors",
		},
		[]string{"collection", "error_type"},
	)

	m.insertLatency = promauto.With(m.reg).NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "quiver_insert_latency_ms",
			Help:    "Insert operation latency in milliseconds",
			Buckets: []float64{0.1, 0.5, 1, 5, 10, 25, 50, 100, 250, 500, 1000},
		},
		[]string{"collection", "operation"},
	)

	m.insertTotal = promauto.With(m.reg).NewCounterVec(
		prometheus.CounterOpts{
			Name: "quiver_insert_total",
			Help: "Total number of insert operations",
		},
		[]string{"collection", "operation"},
	)

	m.insertErrors = promauto.With(m.reg).NewCounterVec(
		prometheus.CounterOpts{
			Name: "quiver_insert_errors_total",
			Help: "Total number of insert errors",
		},
		[]string{"collection", "error_type"},
	)

	m.batchLatency = promauto.With(m.reg).NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "quiver_batch_latency_ms",
			Help:    "Batch operation latency in milliseconds",
			Buckets: []float64{1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000},
		},
		[]string{"collection", "operation"},
	)

	m.batchTotal = promauto.With(m.reg).NewCounterVec(
		prometheus.CounterOpts{
			Name: "quiver_batch_total",
			Help: "Total number of batch operations",
		},
		[]string{"collection", "operation"},
	)

	m.batchErrors = promauto.With(m.reg).NewCounterVec(
		prometheus.CounterOpts{
			Name: "quiver_batch_errors_total",
			Help: "Total number of batch errors",
		},
		[]string{"collection", "operation", "error_type"},
	)

	m.filterLatency = promauto.With(m.reg).NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "quiver_filter_latency_ms",
			Help:    "Filter stage latency in milliseconds",
			Buckets: []float64{0.1, 0.5, 1, 5, 10, 25, 50, 100},
		},
		[]string{"collection"},
	)

	m.traversalLatency = promauto.With(m.reg).NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "quiver_traversal_latency_ms",
			Help:    "Index traversal stage latency in milliseconds",
			Buckets: []float64{0.1, 0.5, 1, 5, 10, 25, 50, 100},
		},
		[]string{"collection"},
	)

	m.rerankLatency = promauto.With(m.reg).NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "quiver_rerank_latency_ms",
			Help:    "Re-ranking stage latency in milliseconds",
			Buckets: []float64{0.1, 0.5, 1, 5, 10, 25, 50, 100},
		},
		[]string{"collection"},
	)

	m.indexSize = promauto.With(m.reg).NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "quiver_index_size",
			Help: "Number of vectors in the index",
		},
		[]string{"collection"},
	)

	m.indexVectors = promauto.With(m.reg).NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "quiver_index_vectors",
			Help: "Number of vectors in the collection",
		},
		[]string{"collection"},
	)

	m.indexSizeBytes = promauto.With(m.reg).NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "quiver_index_size_bytes",
			Help: "Estimated size of index in bytes",
		},
		[]string{"collection"},
	)

	return m
}

func (m *Metrics) SetEnabled(enabled bool) {
	m.enabled.Store(enabled)
}

func (m *Metrics) Enabled() bool {
	return m.enabled.Load()
}

func (m *Metrics) Registry() *prometheus.Registry {
	return m.reg
}

func (m *Metrics) RecordSearchLatency(collection, stage string, duration time.Duration) {
	if !m.enabled.Load() {
		return
	}
	m.searchLatency.WithLabelValues(collection, stage).Observe(float64(duration.Milliseconds()))
}

func (m *Metrics) RecordSearch(collection string) {
	if !m.enabled.Load() {
		return
	}
	m.searchTotal.WithLabelValues(collection).Inc()
}

func (m *Metrics) RecordSearchError(collection, errorType string) {
	if !m.enabled.Load() {
		return
	}
	m.searchErrors.WithLabelValues(collection, errorType).Inc()
}

func (m *Metrics) RecordInsertLatency(collection, operation string, duration time.Duration) {
	if !m.enabled.Load() {
		return
	}
	m.insertLatency.WithLabelValues(collection, operation).Observe(float64(duration.Milliseconds()))
}

func (m *Metrics) RecordInsert(collection, operation string, count int) {
	if !m.enabled.Load() {
		return
	}
	m.insertTotal.WithLabelValues(collection, operation).Add(float64(count))
}

func (m *Metrics) RecordInsertError(collection, errorType string) {
	if !m.enabled.Load() {
		return
	}
	m.insertErrors.WithLabelValues(collection, errorType).Inc()
}

func (m *Metrics) RecordBatchLatency(collection, operation string, duration time.Duration) {
	if !m.enabled.Load() {
		return
	}
	m.batchLatency.WithLabelValues(collection, operation).Observe(float64(duration.Milliseconds()))
}

func (m *Metrics) RecordBatch(collection, operation string, batchSize int) {
	if !m.enabled.Load() {
		return
	}
	m.batchTotal.WithLabelValues(collection, operation).Inc()
	m.insertTotal.WithLabelValues(collection, operation).Add(float64(batchSize))
}

func (m *Metrics) RecordBatchError(collection, operation, errorType string) {
	if !m.enabled.Load() {
		return
	}
	m.batchErrors.WithLabelValues(collection, operation, errorType).Inc()
}

func (m *Metrics) RecordFilterLatency(collection string, duration time.Duration) {
	if !m.enabled.Load() {
		return
	}
	m.filterLatency.WithLabelValues(collection).Observe(float64(duration.Milliseconds()))
}

func (m *Metrics) RecordTraversalLatency(collection string, duration time.Duration) {
	if !m.enabled.Load() {
		return
	}
	m.traversalLatency.WithLabelValues(collection).Observe(float64(duration.Milliseconds()))
}

func (m *Metrics) RecordRerankLatency(collection string, duration time.Duration) {
	if !m.enabled.Load() {
		return
	}
	m.rerankLatency.WithLabelValues(collection).Observe(float64(duration.Milliseconds()))
}

func (m *Metrics) UpdateIndexSize(collection string, size int) {
	if !m.enabled.Load() {
		return
	}
	m.indexSize.WithLabelValues(collection).Set(float64(size))
}

func (m *Metrics) UpdateVectorCount(collection string, count int) {
	if !m.enabled.Load() {
		return
	}
	m.indexVectors.WithLabelValues(collection).Set(float64(count))
}

func (m *Metrics) UpdateIndexSizeBytes(collection string, bytes int64) {
	if !m.enabled.Load() {
		return
	}
	m.indexSizeBytes.WithLabelValues(collection).Set(float64(bytes))
}

type LatencyPercentiles struct {
	P50 float64
	P95 float64
	P99 float64
	Max float64
	Avg float64
	Min float64
}

func (m *Metrics) RecordInMemoryLatency(name string, duration time.Duration) {
	if !m.enabled.Load() {
		return
	}

	m.inMemoryMu.Lock()
	defer m.inMemoryMu.Unlock()

	latencies := m.inMemoryLatencies[name]
	latencies = append(latencies, duration)

	if len(latencies) > 10000 {
		latencies = latencies[len(latencies)-10000:]
	}

	m.inMemoryLatencies[name] = latencies
}

func (m *Metrics) GetLatencyPercentiles(name string) LatencyPercentiles {
	m.inMemoryMu.RLock()
	defer m.inMemoryMu.RUnlock()

	latencies, ok := m.inMemoryLatencies[name]
	if !ok || len(latencies) == 0 {
		return LatencyPercentiles{}
	}

	sorted := make([]time.Duration, len(latencies))
	copy(sorted, latencies)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })

	n := len(sorted)
	p50 := float64(sorted[n*50/100]) / 1e6
	p95 := float64(sorted[n*95/100]) / 1e6
	p99 := float64(sorted[n*99/100]) / 1e6
	max := float64(sorted[n-1]) / 1e6
	min := float64(sorted[0]) / 1e6

	var sum float64
	for _, d := range sorted {
		sum += float64(d)
	}
	avg := sum / float64(n) / 1e6

	return LatencyPercentiles{
		P50: math.Round(p50*100) / 100,
		P95: math.Round(p95*100) / 100,
		P99: math.Round(p99*100) / 100,
		Max: math.Round(max*100) / 100,
		Avg: math.Round(avg*100) / 100,
		Min: math.Round(min*100) / 100,
	}
}

type TimedObserver struct {
	start    time.Time
	metric   *Metrics
	name     string
	attrs    []string
	recorder func(string, time.Duration)
}

func Observe(m *Metrics, name string) *TimedObserver {
	return &TimedObserver{
		start:  time.Now(),
		metric: m,
		name:   name,
	}
}

func (t *TimedObserver) With(attrs ...string) *TimedObserver {
	t.attrs = attrs
	return t
}

func (t *TimedObserver) RecordLatency(fn func(string, time.Duration)) {
	t.recorder = fn
}

func (t *TimedObserver) End() {
	duration := time.Since(t.start)
	if t.metric != nil && t.metric.enabled.Load() {
		t.metric.RecordInMemoryLatency(t.name, duration)
		if t.recorder != nil {
			t.recorder(t.name, duration)
		}
	}
}

func StartTimer() func() {
	start := time.Now()
	return func() {
		duration := time.Since(start)
		GlobalMetrics().RecordInMemoryLatency("operation", duration)
	}
}
