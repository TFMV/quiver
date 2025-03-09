package quiver

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

// generateRandomVector creates a random vector of the specified dimension
func generateRandomVector(dim int) []float32 {
	vec := make([]float32, dim)
	for i := range vec {
		vec[i] = rand.Float32()
	}
	return vec
}

// generateRandomMetadata creates random metadata for a vector
func generateRandomMetadata(id uint64) map[string]interface{} {
	return map[string]interface{}{
		"id":       id,
		"category": fmt.Sprintf("category-%d", id%5),
		"name":     fmt.Sprintf("vector-%d", id),
	}
}

// createTempDir creates a temporary directory for the benchmark
func createTempDir(b *testing.B) string {
	b.Helper()
	tempDir, err := os.MkdirTemp("", "quiver-benchmark-*")
	require.NoError(b, err)
	b.Cleanup(func() {
		os.RemoveAll(tempDir)
	})
	return tempDir
}

// createTempDBPath creates a temporary file path for the DuckDB database
func createTempDBPath(b *testing.B) string {
	b.Helper()
	tempDir := createTempDir(b)
	return filepath.Join(tempDir, "quiver.db")
}

// setupBenchmarkIndex creates and populates an index for benchmarking
func setupBenchmarkIndex(b *testing.B, dimension, numVectors int) *Index {
	b.Helper()

	// Create a temporary directory for the benchmark
	tempDir := createTempDir(b)
	dbPath := filepath.Join(tempDir, "quiver.db")

	// Create a no-op logger for benchmarks
	logger := zap.NewNop()

	// Create a new index
	config := Config{
		Dimension:       dimension,
		Distance:        Cosine,
		BatchSize:       1000,
		PersistInterval: 0,                      // Disable persistence for benchmarks
		MaxElements:     uint64(numVectors * 2), // Set max elements to twice the number of vectors
		StoragePath:     dbPath,                 // Use temporary file path for storage
	}

	idx, err := New(config, logger)
	require.NoError(b, err)

	// Add vectors to the index
	for i := 0; i < numVectors; i++ {
		id := uint64(i + 1)
		vector := generateRandomVector(dimension)
		metadata := generateRandomMetadata(id)

		err := idx.Add(id, vector, metadata)
		require.NoError(b, err)
	}

	// Flush the batch to ensure all vectors are indexed
	err = idx.flushBatch()
	require.NoError(b, err)

	return idx
}

// BenchmarkAdd tests the performance of adding vectors to the index
func BenchmarkAdd(b *testing.B) {
	dimension := 128

	// Create a temporary directory for the benchmark
	tempDir := createTempDir(b)
	dbPath := filepath.Join(tempDir, "quiver.db")

	// Create a no-op logger for benchmarks
	logger := zap.NewNop()

	// Create a new index
	config := Config{
		Dimension:       dimension,
		Distance:        Cosine,
		BatchSize:       1000,
		PersistInterval: 0,                  // Disable persistence for benchmarks
		MaxElements:     uint64(b.N + 1000), // Set max elements to accommodate all iterations
		StoragePath:     dbPath,             // Use temporary file path for storage
	}

	idx, err := New(config, logger)
	require.NoError(b, err)
	defer idx.Close()

	// Reset the timer before the benchmark loop
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		id := uint64(i + 1)
		vector := generateRandomVector(dimension)
		metadata := generateRandomMetadata(id)

		err := idx.Add(id, vector, metadata)
		if err != nil {
			b.Fatalf("Failed to add vector: %v", err)
		}
	}
}

// BenchmarkSearch tests the performance of vector search
func BenchmarkSearch(b *testing.B) {
	dimension := 128
	numVectors := 1000 // Reduced for faster benchmark setup

	idx := setupBenchmarkIndex(b, dimension, numVectors)
	defer idx.Close()

	// Generate a query vector
	query := generateRandomVector(dimension)

	// Reset the timer before the benchmark loop
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := idx.Search(query, 10, 1, 10)
		if err != nil {
			b.Fatalf("Search failed: %v", err)
		}
	}
}

// BenchmarkHybridSearch tests the performance of hybrid search (vector + metadata filter)
func BenchmarkHybridSearch(b *testing.B) {
	dimension := 128
	numVectors := 1000 // Reduced for faster benchmark setup

	idx := setupBenchmarkIndex(b, dimension, numVectors)
	defer idx.Close()

	// Generate a query vector
	query := generateRandomVector(dimension)

	// Reset the timer before the benchmark loop
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := idx.SearchWithFilter(query, 10, "SELECT * FROM metadata WHERE json LIKE '%category-1%'")
		if err != nil {
			b.Fatalf("Hybrid search failed: %v", err)
		}
	}
}

// BenchmarkSearchWithNegatives tests the performance of search with negative examples
func BenchmarkSearchWithNegatives(b *testing.B) {
	dimension := 128
	numVectors := 1000 // Reduced for faster benchmark setup

	idx := setupBenchmarkIndex(b, dimension, numVectors)
	defer idx.Close()

	// Generate query vectors
	positiveQuery := generateRandomVector(dimension)
	negativeQuery1 := generateRandomVector(dimension)
	negativeQuery2 := generateRandomVector(dimension)

	// Reset the timer before the benchmark loop
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := idx.SearchWithNegatives(positiveQuery, [][]float32{negativeQuery1, negativeQuery2}, 10, 1, 10)
		if err != nil {
			b.Fatalf("Search with negatives failed: %v", err)
		}
	}
}

// BenchmarkBatchAdd tests the performance of adding vectors in batch
func BenchmarkBatchAdd(b *testing.B) {
	// Run with different batch sizes
	batchSizes := []int{100, 1000, 10000}

	for _, batchSize := range batchSizes {
		b.Run(fmt.Sprintf("BatchSize-%d", batchSize), func(b *testing.B) {
			dimension := 128

			// Create a temporary directory for the benchmark
			tempDir := createTempDir(b)
			dbPath := filepath.Join(tempDir, "quiver.db")

			// Create a no-op logger for benchmarks
			logger := zap.NewNop()

			// Create a new index with the specified batch size
			config := Config{
				Dimension:       dimension,
				Distance:        Cosine,
				BatchSize:       batchSize,
				PersistInterval: 0,                            // Disable persistence for benchmarks
				MaxElements:     uint64(b.N*batchSize + 1000), // Set max elements to accommodate all iterations
				StoragePath:     dbPath,                       // Use temporary file path for storage
			}

			idx, err := New(config, logger)
			require.NoError(b, err)
			defer idx.Close()

			// Reset the timer before the benchmark loop
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				// Add batchSize vectors
				for j := 0; j < batchSize; j++ {
					id := uint64(i*batchSize + j + 1)
					vector := generateRandomVector(dimension)
					metadata := generateRandomMetadata(id)

					err := idx.Add(id, vector, metadata)
					if err != nil {
						b.Fatalf("Failed to add vector: %v", err)
					}
				}

				// Flush the batch
				err := idx.flushBatch()
				if err != nil {
					b.Fatalf("Failed to flush batch: %v", err)
				}
			}
		})
	}
}

// BenchmarkSearchWithDifferentK tests search performance with different k values
func BenchmarkSearchWithDifferentK(b *testing.B) {
	dimension := 128
	numVectors := 1000 // Reduced for faster benchmark setup

	idx := setupBenchmarkIndex(b, dimension, numVectors)
	defer idx.Close()

	// Generate a query vector
	query := generateRandomVector(dimension)

	// Test with different k values
	kValues := []int{10, 50, 100}

	for _, k := range kValues {
		b.Run(fmt.Sprintf("K-%d", k), func(b *testing.B) {
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				_, err := idx.Search(query, k, 1, k)
				if err != nil {
					b.Fatalf("Search failed: %v", err)
				}
			}
		})
	}
}

// BenchmarkSearchWithDifferentDimensions tests search performance with different vector dimensions
func BenchmarkSearchWithDifferentDimensions(b *testing.B) {
	dimensions := []int{32, 128, 512}
	numVectors := 500 // Reduced for faster benchmark setup

	for _, dim := range dimensions {
		b.Run(fmt.Sprintf("Dim-%d", dim), func(b *testing.B) {
			idx := setupBenchmarkIndex(b, dim, numVectors)
			defer idx.Close()

			// Generate a query vector
			query := generateRandomVector(dim)

			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				_, err := idx.Search(query, 10, 1, 10)
				if err != nil {
					b.Fatalf("Search failed: %v", err)
				}
			}
		})
	}
}
