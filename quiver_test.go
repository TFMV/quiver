package quiver_test

import (
	"os"
	"testing"

	quiver "github.com/TFMV/quiver"
	"github.com/stretchr/testify/assert"
)

func cleanupTest(t testing.TB, paths ...string) {
	for _, path := range paths {
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			t.Logf("Failed to cleanup %s: %v", path, err)
		}
	}
}

func TestVectorIndex_AddAndSearch(t *testing.T) {
	defer cleanupTest(t, "test_index.hnsw", "test_index.duckdb")
	index, err := quiver.NewVectorIndex(3, "test_index.hnsw", "test_index.duckdb", quiver.Cosine)
	if err != nil {
		t.Fatalf("Failed to create vector index: %v", err)
	}

	vectors := [][]float32{
		{0.1, 0.2, 0.3},
		{0.9, 0.8, 0.7},
		{0.4, 0.5, 0.6},
	}

	for i, vec := range vectors {
		index.AddVector(i, vec)
	}

	query := []float32{0.5, 0.5, 0.5}
	neighbors, _ := index.Search(query, 2)

	assert.NotEmpty(t, neighbors, "Neighbors should not be empty")
	assert.Contains(t, neighbors, 2, "Expected vector 2 to be a neighbor")
}

func TestVectorIndex_SaveLoad(t *testing.T) {
	defer cleanupTest(t, "test_index.hnsw", "test.db")
	// Create and populate index
	index, err := quiver.NewVectorIndex(3, "test.db", "test_index.hnsw", quiver.Cosine)
	if err != nil {
		t.Fatalf("Failed to create vector index: %v", err)
	}
	index.AddVector(1, []float32{0.1, 0.2, 0.3})

	// Save index
	index.Save()

	// Load index
	loadedIndex, err := quiver.NewVectorIndex(3, "test.db", "test_index.hnsw", quiver.Cosine)
	if err != nil {
		t.Fatalf("Failed to load vector index: %v", err)
	}

	neighbors, _ := loadedIndex.Search([]float32{0.1, 0.2, 0.3}, 1)
	assert.Equal(t, []int{1}, neighbors, "Loaded index should return correct neighbor")
}

func TestVectorIndex_DistanceMetrics(t *testing.T) {
	defer cleanupTest(t, "test.db", "test_index.hnsw")
	indexCosine, err := quiver.NewVectorIndex(3, "test.db", "test_index.hnsw", quiver.Cosine)
	if err != nil {
		t.Fatalf("Failed to create vector index: %v", err)
	}
	indexEuclidean, err := quiver.NewVectorIndex(3, "test.db", "test_index.hnsw", quiver.Euclidean)
	if err != nil {
		t.Fatalf("Failed to create vector index: %v", err)
	}

	vectors := [][]float32{
		{0.1, 0.2, 0.3},
		{0.9, 0.8, 0.7},
		{0.4, 0.5, 0.6},
	}

	for i, vec := range vectors {
		if err := indexCosine.AddVector(i, vec); err != nil {
			t.Fatalf("Failed to add vector to cosine index: %v", err)
		}
		if err := indexEuclidean.AddVector(i, vec); err != nil {
			t.Fatalf("Failed to add vector to euclidean index: %v", err)
		}
	}

	query := []float32{0.5, 0.5, 0.5}
	cosineNeighbors, err := indexCosine.Search(query, 2)
	if err != nil {
		t.Fatalf("Failed to search cosine index: %v", err)
	}
	euclideanNeighbors, err := indexEuclidean.Search(query, 2)
	if err != nil {
		t.Fatalf("Failed to search euclidean index: %v", err)
	}

	assert.NotEmpty(t, cosineNeighbors, "Cosine neighbors should not be empty")
	assert.NotEmpty(t, euclideanNeighbors, "Euclidean neighbors should not be empty")
	assert.NotEqual(t, cosineNeighbors, euclideanNeighbors, "Different metrics should yield different neighbors")
}

func BenchmarkVectorSearch(b *testing.B) {
	defer cleanupTest(b, "test_index.hnsw", "test_index.duckdb")
	index, err := quiver.NewVectorIndex(128, "test_index.hnsw", "test_index.duckdb", quiver.Cosine)
	if err != nil {
		b.Fatalf("Failed to create vector index: %v", err)
	}
	for i := range 10000 {
		vec := make([]float32, 128)
		for j := range vec {
			vec[j] = float32(i % 10)
		}
		index.AddVector(i, vec)
	}

	query := make([]float32, 128)
	for i := range query {
		query[i] = 5.0
	}

	b.ResetTimer()
	for range b.N {
		index.Search(query, 10)
	}
}
