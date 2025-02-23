package quiver

import (
	"errors"
	"math"
	"sort"
	"sync"
)

// Index represents the vector search index.
type Index struct {
	config  Config
	vectors map[int][]float32
	lock    sync.RWMutex
}

// Config holds the index configuration.
type Config struct {
	Dimension int
	Distance  DistanceMetric
}

// DistanceMetric defines the supported similarity metrics.
type DistanceMetric int

const (
	Cosine DistanceMetric = iota
	Euclidean
)

// New creates a new index.
func New(config Config) *Index {
	return &Index{
		config:  config,
		vectors: make(map[int][]float32),
	}
}

// Add inserts a vector into the index.
func (idx *Index) Add(id int, vector []float32) error {
	if len(vector) != idx.config.Dimension {
		return ErrDimensionMismatch
	}
	idx.lock.Lock()
	idx.vectors[id] = vector
	idx.lock.Unlock()
	return nil
}

// Search returns the top-k nearest vectors.
func (idx *Index) Search(query []float32, k int) ([]int, error) {
	if len(query) != idx.config.Dimension {
		return nil, ErrDimensionMismatch
	}
	candidates := idx.searchInternal(query, k)
	ids := make([]int, len(candidates))
	for i, c := range candidates {
		ids[i] = c.id
	}
	return ids, nil
}

type candidate struct {
	id       int
	distance float32
}

// searchInternal performs brute-force k-NN search.
func (idx *Index) searchInternal(query []float32, k int) []candidate {
	idx.lock.RLock()
	defer idx.lock.RUnlock()
	var results []candidate
	for id, vec := range idx.vectors {
		var d float32
		if idx.config.Distance == Cosine {
			d = cosineDistance(query, vec)
		} else {
			d = euclideanDistance(query, vec)
		}
		results = append(results, candidate{id: id, distance: d})
	}
	// Sort by ascending distance
	sort.Slice(results, func(i, j int) bool { return results[i].distance < results[j].distance })
	if len(results) > k {
		results = results[:k]
	}
	return results
}

// cosineDistance computes the cosine similarity distance.
func cosineDistance(a, b []float32) float32 {
	var dot, normA, normB float32
	for i := 0; i < len(a); i++ {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 1.0
	}
	return 1.0 - (dot / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB)))))
}

// euclideanDistance computes the Euclidean distance.
func euclideanDistance(a, b []float32) float32 {
	var sum float32
	for i := 0; i < len(a); i++ {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}

var ErrDimensionMismatch = errors.New("vector dimension mismatch")
