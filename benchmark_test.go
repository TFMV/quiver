package quiver

import (
	"math/rand"
	"testing"
)

func randomVector(dim int) []float32 {
	vec := make([]float32, dim)
	for i := range vec {
		vec[i] = rand.Float32()
	}
	return vec
}

func BenchmarkVectorIndex_Add(b *testing.B) {
	index, err := NewVectorIndex(128, "bench.db", "bench.hnsw")
	if err != nil {
		b.Fatalf("Failed to create vector index: %v", err)
	}
	for i := range b.N {
		index.AddVector(i, randomVector(128))
	}
}

func BenchmarkVectorIndex_Search(b *testing.B) {
	index, err := NewVectorIndex(128, "bench.db", "bench.hnsw")
	if err != nil {
		b.Fatalf("Failed to create vector index: %v", err)
	}
	for i := range 10000 {
		index.AddVector(i, randomVector(128))
	}

	query := randomVector(128)
	b.ResetTimer()
	for range b.N {
		index.Search(query, 10)
	}
}
