package quiver_test

import (
	"os"
	"testing"

	quiver "github.com/TFMV/quiver"
	"github.com/stretchr/testify/assert"
)

func TestVectorIndex_AddAndSearch(t *testing.T) {
	index, err := quiver.NewVectorIndex(3, "test_index.hnsw", "test_index.duckdb")
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
	neighbors := index.Search(query, 2)

	assert.NotEmpty(t, neighbors, "Neighbors should not be empty")
	assert.Contains(t, neighbors, 2, "Expected vector 2 to be a neighbor")
}

func TestVectorIndex_SaveLoad(t *testing.T) {
	// Clean up any existing files first
	indexPath := "test_index.hnsw"
	dbPath := "test.db"
	os.Remove(indexPath)
	os.Remove(dbPath)
	defer func() {
		os.Remove(indexPath)
		os.Remove(dbPath)
	}()

	// Create and populate index
	index, err := quiver.NewVectorIndex(3, dbPath, indexPath)
	if err != nil {
		t.Fatalf("Failed to create vector index: %v", err)
	}
	index.AddVector(1, []float32{0.1, 0.2, 0.3})

	// Save index
	index.Save()

	// Load index
	loadedIndex, err := quiver.LoadVectorIndex(3, indexPath)
	if err != nil {
		t.Fatalf("Failed to load vector index: %v", err)
	}

	neighbors := loadedIndex.Search([]float32{0.1, 0.2, 0.3}, 1)
	assert.Equal(t, []int{1}, neighbors, "Loaded index should return correct neighbor")
}

func TestVectorIndex_DistanceMetrics(t *testing.T) {
	indexCosine, err := quiver.NewVectorIndexWithMetric(3, "test.db", "test_index.hnsw", quiver.Cosine)
	if err != nil {
		t.Fatalf("Failed to create vector index: %v", err)
	}
	indexEuclidean, err := quiver.NewVectorIndexWithMetric(3, "test.db", "test_index.hnsw", quiver.Euclidean)
	if err != nil {
		t.Fatalf("Failed to create vector index: %v", err)
	}

	vectors := [][]float32{
		{0.1, 0.2, 0.3},
		{0.9, 0.8, 0.7},
		{0.4, 0.5, 0.6},
	}

	for i, vec := range vectors {
		indexCosine.AddVector(i, vec)
		indexEuclidean.AddVector(i, vec)
	}

	query := []float32{0.5, 0.5, 0.5}
	cosineNeighbors := indexCosine.Search(query, 2)
	euclideanNeighbors := indexEuclidean.Search(query, 2)

	assert.NotEqual(t, cosineNeighbors, euclideanNeighbors, "Different metrics should yield different neighbors")
}

func BenchmarkVectorSearch(b *testing.B) {
	index, err := quiver.NewVectorIndex(128, "test_index.hnsw", "test_index.duckdb")
	if err != nil {
		b.Fatalf("Failed to create vector index: %v", err)
	}
	for i := 0; i < 10000; i++ {
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
	for i := 0; i < b.N; i++ {
		index.Search(query, 10)
	}
}
