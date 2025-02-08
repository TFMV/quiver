package quiver_test

import (
	"math/rand"
	"os"
	"testing"

	quiver "github.com/TFMV/quiver"
)

func randomVector(dim int) []float32 {
	vec := make([]float32, dim)
	for i := range vec {
		vec[i] = rand.Float32()
	}
	return vec
}

// cleanupFiles removes old benchmark data before each run.
func cleanupFiles() {
	_ = os.Remove("bench.db")
	_ = os.Remove("bench.hnsw")
}

func cleanup(b *testing.B, paths ...string) {
	for _, path := range paths {
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			b.Logf("Failed to cleanup %s: %v", path, err)
		}
	}
}

// Benchmark AddVector Performance
func BenchmarkVectorIndex_Add(b *testing.B) {
	defer cleanup(b, "bench.db", "bench.hnsw")

	index, err := quiver.NewVectorIndex(128, "bench.db", "bench.hnsw", quiver.Cosine)
	if err != nil {
		b.Fatalf("Failed to create vector index: %v", err)
	}

	b.ResetTimer()
	for i := range b.N {
		if err := index.AddVector(i, randomVector(128)); err != nil {
			b.Fatalf("Failed to add vector: %v", err)
		}
	}

	// Save to disk to measure persistence time
	index.Save()
}

// Benchmark Search Performance
func BenchmarkVectorIndex_Search(b *testing.B) {
	defer cleanup(b, "bench.db", "bench.hnsw")

	index, err := quiver.NewVectorIndex(128, "bench.db", "bench.hnsw", quiver.Cosine)
	if err != nil {
		b.Fatalf("Failed to create vector index: %v", err)
	}

	for i := range 1000 {
		if err := index.AddVector(i, randomVector(128)); err != nil {
			b.Fatalf("Failed to add vector: %v", err)
		}
	}

	query := randomVector(128)
	b.ResetTimer()
	for range b.N {
		if _, err := index.Search(query, 10); err != nil {
			b.Fatalf("Search failed: %v", err)
		}
	}
}
