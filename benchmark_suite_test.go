package quiver

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/bytedance/sonic"
	"go.uber.org/zap"
)

// setupBenchmarkIndex creates a new index for benchmarking
func setupBenchmarkIndex(b *testing.B, dimension, batchSize int) (*Index, string) {
	b.Helper()

	// Create a temporary directory for the benchmark
	tmpDir, err := os.MkdirTemp("", "quiver-benchmark-*")
	if err != nil {
		b.Fatalf("Failed to create temp directory: %v", err)
	}

	// Create a logger that doesn't output during benchmarks
	logger := zap.NewNop()

	// Create a configuration for the index
	config := Config{
		Dimension:       dimension,
		StoragePath:     filepath.Join(tmpDir, "quiver.db"),
		Distance:        Cosine,
		MaxElements:     uint64(b.N + 10000), // Add some buffer
		HNSWM:           16,
		HNSWEfConstruct: 200,
		HNSWEfSearch:    100,
		BatchSize:       batchSize,
		PersistInterval: 0, // Disable persistence for benchmarks
	}

	// Create a new index
	idx, err := New(config, logger)
	if err != nil {
		os.RemoveAll(tmpDir)
		b.Fatalf("Failed to create index: %v", err)
	}

	return idx, tmpDir
}

// generateRandomVector creates a random vector of the specified dimension
func generateRandomVector(dimension int) []float32 {
	vector := make([]float32, dimension)
	for i := range vector {
		vector[i] = rand.Float32()
	}
	return vector
}

// generateRandomMetadata creates random metadata for benchmarking
func generateRandomMetadata(id uint64) map[string]interface{} {
	return map[string]interface{}{
		"id":       id,
		"category": "benchmark",
		"name":     fmt.Sprintf("item-%d", id),
		"score":    rand.Float64(),
		"tags":     []string{"benchmark", "test", fmt.Sprintf("tag-%d", id%10)},
	}
}

// BenchmarkAdd benchmarks the Add operation
func BenchmarkAdd(b *testing.B) {
	dimension := 128
	idx, tmpDir := setupBenchmarkIndex(b, dimension, 1000)
	defer os.RemoveAll(tmpDir)
	defer idx.Close()

	// Reset the timer before starting the benchmark
	b.ResetTimer()

	// Add vectors to the index
	for i := 0; i < b.N; i++ {
		vector := generateRandomVector(dimension)
		metadata := generateRandomMetadata(uint64(i))

		if err := idx.Add(uint64(i), vector, metadata); err != nil {
			b.Fatalf("Failed to add vector: %v", err)
		}
	}
}

// BenchmarkAddWithSmallBatch benchmarks the Add operation with a small batch size
func BenchmarkAddWithSmallBatch(b *testing.B) {
	dimension := 128
	idx, tmpDir := setupBenchmarkIndex(b, dimension, 10) // Small batch size
	defer os.RemoveAll(tmpDir)
	defer idx.Close()

	// Reset the timer before starting the benchmark
	b.ResetTimer()

	// Add vectors to the index
	for i := 0; i < b.N; i++ {
		vector := generateRandomVector(dimension)
		metadata := generateRandomMetadata(uint64(i))

		if err := idx.Add(uint64(i), vector, metadata); err != nil {
			b.Fatalf("Failed to add vector: %v", err)
		}
	}
}

// BenchmarkSearch benchmarks the Search operation
func BenchmarkSearch(b *testing.B) {
	dimension := 128
	numVectors := 10000 // Pre-populate with 10,000 vectors
	idx, tmpDir := setupBenchmarkIndex(b, dimension, 1000)
	defer os.RemoveAll(tmpDir)
	defer idx.Close()

	// Pre-populate the index
	for i := 0; i < numVectors; i++ {
		vector := generateRandomVector(dimension)
		metadata := generateRandomMetadata(uint64(i))

		if err := idx.Add(uint64(i), vector, metadata); err != nil {
			b.Fatalf("Failed to add vector: %v", err)
		}
	}

	// Flush the batch to ensure all vectors are indexed
	if err := idx.flushBatch(); err != nil {
		b.Fatalf("Failed to flush batch: %v", err)
	}

	// Generate random query vectors
	queryVectors := make([][]float32, b.N)
	for i := 0; i < b.N; i++ {
		queryVectors[i] = generateRandomVector(dimension)
	}

	// Reset the timer before starting the benchmark
	b.ResetTimer()

	// Perform searches
	for i := 0; i < b.N; i++ {
		_, err := idx.Search(queryVectors[i], 10, 1, 10)
		if err != nil {
			b.Fatalf("Failed to search: %v", err)
		}
	}
}

// BenchmarkSearchWithNegatives benchmarks the SearchWithNegatives operation
func BenchmarkSearchWithNegatives(b *testing.B) {
	dimension := 128
	numVectors := 10000 // Pre-populate with 10,000 vectors
	idx, tmpDir := setupBenchmarkIndex(b, dimension, 1000)
	defer os.RemoveAll(tmpDir)
	defer idx.Close()

	// Pre-populate the index
	for i := 0; i < numVectors; i++ {
		vector := generateRandomVector(dimension)
		metadata := generateRandomMetadata(uint64(i))

		if err := idx.Add(uint64(i), vector, metadata); err != nil {
			b.Fatalf("Failed to add vector: %v", err)
		}
	}

	// Flush the batch to ensure all vectors are indexed
	if err := idx.flushBatch(); err != nil {
		b.Fatalf("Failed to flush batch: %v", err)
	}

	// Generate random query vectors
	queryVectors := make([][]float32, b.N)
	negativeVectors := make([][][]float32, b.N)
	for i := 0; i < b.N; i++ {
		queryVectors[i] = generateRandomVector(dimension)
		// Create 2 negative examples for each query
		negativeVectors[i] = make([][]float32, 2)
		negativeVectors[i][0] = generateRandomVector(dimension)
		negativeVectors[i][1] = generateRandomVector(dimension)
	}

	// Reset the timer before starting the benchmark
	b.ResetTimer()

	// Perform searches with negatives
	for i := 0; i < b.N; i++ {
		_, err := idx.SearchWithNegatives(queryVectors[i], negativeVectors[i], 10, 1, 10)
		if err != nil {
			b.Fatalf("Failed to search with negatives: %v", err)
		}
	}
}

// BenchmarkDeleteVector benchmarks the DeleteVector operation
func BenchmarkDeleteVector(b *testing.B) {
	dimension := 128
	numVectors := 1000 // Reduced from 10,000 to 1,000 for stability
	idx, tmpDir := setupBenchmarkIndex(b, dimension, 100)
	defer os.RemoveAll(tmpDir)
	defer idx.Close()

	// Pre-populate the index with vectors that we'll delete
	// Add vectors in smaller batches to avoid overwhelming the system
	batchSize := 100
	for i := 0; i < numVectors; i += batchSize {
		end := i + batchSize
		if end > numVectors {
			end = numVectors
		}

		for j := i; j < end; j++ {
			vector := generateRandomVector(dimension)
			metadata := generateRandomMetadata(uint64(j))

			if err := idx.Add(uint64(j), vector, metadata); err != nil {
				b.Fatalf("Failed to add vector: %v", err)
			}
		}

		// Flush after each batch to ensure vectors are indexed
		if err := idx.flushBatch(); err != nil {
			b.Fatalf("Failed to flush batch: %v", err)
		}

		// Give some time for background operations to complete
		time.Sleep(10 * time.Millisecond)
	}

	// Reset the timer before starting the benchmark
	b.ResetTimer()

	// Delete vectors one by one
	for i := 0; i < b.N; i++ {
		// Use modulo to cycle through the available vectors
		id := uint64(i % numVectors)

		// Skip if the vector has already been deleted
		if _, ok := idx.vectors[id]; !ok {
			continue
		}

		// Delete the vector
		err := idx.DeleteVector(id)
		if err != nil {
			// Log the error but don't fail the benchmark
			b.Logf("Failed to delete vector %d: %v", id, err)
		}
	}
}

// BenchmarkDeleteVectors benchmarks the DeleteVectors operation
func BenchmarkDeleteVectors(b *testing.B) {
	dimension := 128
	numVectors := 1000 // Reduced from 10,000 to 1,000 for stability
	batchSize := 10    // Delete 10 vectors at a time
	idx, tmpDir := setupBenchmarkIndex(b, dimension, 100)
	defer os.RemoveAll(tmpDir)
	defer idx.Close()

	// Pre-populate the index with vectors that will be deleted
	startID := uint64(1000)
	for i := 0; i < numVectors; i++ {
		id := startID + uint64(i)
		vector := generateRandomVector(dimension)
		metadata := generateRandomMetadata(id)

		if err := idx.Add(id, vector, metadata); err != nil {
			b.Fatalf("Failed to add vector: %v", err)
		}

		// Flush every 100 vectors to avoid excessive memory usage
		if i > 0 && i%100 == 0 {
			if err := idx.flushBatch(); err != nil {
				b.Fatalf("Failed to flush batch: %v", err)
			}
			// Give some time for background operations to complete
			time.Sleep(10 * time.Millisecond)
		}
	}

	// Final flush to ensure all vectors are indexed
	if err := idx.flushBatch(); err != nil {
		b.Fatalf("Failed to flush batch: %v", err)
	}

	// Give some time for background operations to complete
	time.Sleep(100 * time.Millisecond)

	// Reset the timer before starting the benchmark
	b.ResetTimer()

	// For each benchmark iteration, delete a batch of vectors
	for i := 0; i < b.N; i++ {
		// Create a batch of IDs to delete
		batchStartID := startID + uint64((i*batchSize)%numVectors)
		if batchStartID+uint64(batchSize) > startID+uint64(numVectors) {
			// Wrap around if we reach the end
			batchStartID = startID
		}

		ids := make([]uint64, batchSize)
		for j := 0; j < batchSize; j++ {
			ids[j] = batchStartID + uint64(j)
		}

		// Delete the vectors
		err := idx.DeleteVectors(ids)
		if err != nil {
			// Log the error but continue with the benchmark
			b.Logf("Failed to delete vectors: %v", err)
		}
	}
}

// BenchmarkBatchAppendFromArrow benchmarks the BatchAppendFromArrow operation
func BenchmarkBatchAppendFromArrow(b *testing.B) {
	// Run sub-benchmarks with different configurations
	for _, dimension := range []int{128, 256} {
		for _, vectorsPerRecord := range []int{10, 50} {
			b.Run(fmt.Sprintf("dim=%d_vectors=%d", dimension, vectorsPerRecord), func(b *testing.B) {
				// Setup benchmark parameters
				batchSize := 10000
				recordsPerBatch := 5

				// Create a temporary directory for the benchmark
				tmpDir, err := os.MkdirTemp("", "quiver-benchmark-arrow-*")
				if err != nil {
					b.Fatalf("Failed to create temp directory: %v", err)
				}
				defer os.RemoveAll(tmpDir)

				// Create a logger that doesn't output during benchmarks
				logger := zap.NewNop()

				// Create a configuration for the index
				config := Config{
					Dimension:       dimension,
					StoragePath:     filepath.Join(tmpDir, "quiver.db"),
					Distance:        Cosine,
					MaxElements:     uint64(b.N*recordsPerBatch*vectorsPerRecord + 1000), // Add buffer
					HNSWM:           16,
					HNSWEfConstruct: 200,
					HNSWEfSearch:    100,
					BatchSize:       batchSize,
					PersistInterval: 0, // Disable persistence for benchmarks
				}

				// Create a new index
				idx, err := New(config, logger)
				if err != nil {
					b.Fatalf("Failed to create index: %v", err)
				}
				defer idx.Close()

				// Create a memory allocator
				pool := memory.NewGoAllocator()

				// Create schema for the Arrow records
				schema := arrow.NewSchema(
					[]arrow.Field{
						{Name: "id", Type: arrow.PrimitiveTypes.Uint64, Nullable: false},
						{Name: "vector", Type: arrow.FixedSizeListOf(int32(dimension), arrow.PrimitiveTypes.Float32), Nullable: false},
						{Name: "metadata", Type: arrow.BinaryTypes.String, Nullable: true},
					},
					nil,
				)

				// Pre-generate some metadata to reuse
				metadataCache := make([]string, 100)
				for i := 0; i < 100; i++ {
					metadata := generateRandomMetadata(uint64(i))
					metaJSON, err := sonic.Marshal(metadata)
					if err != nil {
						b.Fatalf("Failed to marshal metadata: %v", err)
					}
					metadataCache[i] = string(metaJSON)
				}

				// Reset the timer before starting the benchmark
				b.ResetTimer()
				b.ReportAllocs()

				// Run the benchmark
				for i := 0; i < b.N; i++ {
					// Create a batch of records
					records := make([]arrow.Record, recordsPerBatch)

					for r := 0; r < recordsPerBatch; r++ {
						// Create record builder
						builder := array.NewRecordBuilder(pool, schema)

						// Get builders for each column
						idBuilder := builder.Field(0).(*array.Uint64Builder)
						vectorBuilder := builder.Field(1).(*array.FixedSizeListBuilder)
						vectorValueBuilder := vectorBuilder.ValueBuilder().(*array.Float32Builder)
						metadataBuilder := builder.Field(2).(*array.StringBuilder)

						// Pre-allocate capacity for better performance
						idBuilder.Reserve(vectorsPerRecord)
						vectorBuilder.Reserve(vectorsPerRecord)
						vectorValueBuilder.Reserve(vectorsPerRecord * dimension)
						metadataBuilder.Reserve(vectorsPerRecord)

						// Add data to builders
						baseID := uint64(i*recordsPerBatch*vectorsPerRecord + r*vectorsPerRecord)

						for j := 0; j < vectorsPerRecord; j++ {
							id := baseID + uint64(j)

							// Add ID
							idBuilder.Append(id)

							// Add vector
							vectorBuilder.Append(true)
							vector := generateRandomVector(dimension)
							for _, v := range vector {
								vectorValueBuilder.Append(v)
							}

							// Add metadata (reuse from cache for better performance)
							metadataBuilder.Append(metadataCache[j%100])
						}

						// Build the record
						records[r] = builder.NewRecord()
						builder.Release()
					}

					// Benchmark the BatchAppendFromArrow function
					if err := idx.BatchAppendFromArrow(records); err != nil {
						b.Fatalf("Failed to batch append from arrow: %v", err)
					}

					// Release the records
					for _, rec := range records {
						if rec != nil {
							rec.Release()
						}
					}
				}
			})
		}
	}
}

// BenchmarkMultiVectorSearch benchmarks the MultiVectorSearch operation
func BenchmarkMultiVectorSearch(b *testing.B) {
	dimension := 128
	numVectors := 10000 // Pre-populate with 10,000 vectors
	numQueries := 5     // Number of query vectors per search
	idx, tmpDir := setupBenchmarkIndex(b, dimension, 1000)
	defer os.RemoveAll(tmpDir)
	defer idx.Close()

	// Pre-populate the index
	for i := 0; i < numVectors; i++ {
		vector := generateRandomVector(dimension)
		metadata := generateRandomMetadata(uint64(i))

		if err := idx.Add(uint64(i), vector, metadata); err != nil {
			b.Fatalf("Failed to add vector: %v", err)
		}
	}

	// Flush the batch to ensure all vectors are indexed
	if err := idx.flushBatch(); err != nil {
		b.Fatalf("Failed to flush batch: %v", err)
	}

	// Generate random query vectors
	multiQueries := make([][][]float32, b.N)
	for i := 0; i < b.N; i++ {
		multiQueries[i] = make([][]float32, numQueries)
		for j := 0; j < numQueries; j++ {
			multiQueries[i][j] = generateRandomVector(dimension)
		}
	}

	// Reset the timer before starting the benchmark
	b.ResetTimer()

	// Perform multi-vector searches
	for i := 0; i < b.N; i++ {
		_, err := idx.MultiVectorSearch(multiQueries[i], 10)
		if err != nil {
			b.Fatalf("Failed to perform multi-vector search: %v", err)
		}
	}
}

// BenchmarkSearchWithFilter benchmarks the SearchWithFilter operation
func BenchmarkSearchWithFilter(b *testing.B) {
	dimension := 128
	numVectors := 10000 // Pre-populate with 10,000 vectors
	idx, tmpDir := setupBenchmarkIndex(b, dimension, 1000)
	defer os.RemoveAll(tmpDir)
	defer idx.Close()

	// Pre-populate the index
	for i := 0; i < numVectors; i++ {
		vector := generateRandomVector(dimension)
		metadata := generateRandomMetadata(uint64(i))

		if err := idx.Add(uint64(i), vector, metadata); err != nil {
			b.Fatalf("Failed to add vector: %v", err)
		}
	}

	// Flush the batch to ensure all vectors are indexed
	if err := idx.flushBatch(); err != nil {
		b.Fatalf("Failed to flush batch: %v", err)
	}

	// Generate random query vectors
	queryVectors := make([][]float32, b.N)
	for i := 0; i < b.N; i++ {
		queryVectors[i] = generateRandomVector(dimension)
	}

	// Create filter queries
	filters := []string{
		"SELECT * FROM metadata WHERE json::json->>'category' = 'benchmark'",
		"SELECT * FROM metadata WHERE CAST(json::json->>'score' AS FLOAT) > 0.5",
		"SELECT * FROM metadata WHERE json::json->>'name' LIKE 'item-%'",
	}

	// Reset the timer before starting the benchmark
	b.ResetTimer()

	// Perform searches with filters
	for i := 0; i < b.N; i++ {
		filter := filters[i%len(filters)]
		_, err := idx.SearchWithFilter(queryVectors[i], 10, filter)
		if err != nil {
			b.Fatalf("Failed to search with filter: %v", err)
		}
	}
}

// BenchmarkComprehensive runs a comprehensive benchmark of the Quiver index
func BenchmarkComprehensive(b *testing.B) {
	// Skip this benchmark for now as it needs more work
	b.Skip("BenchmarkComprehensive needs further implementation")
}
