package persistence

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// Fuzz tests - random vectors, filters, operations

func TestFuzz_CollectionOperations(t *testing.T) {
	seed := int64(12345)
	rng := rand.New(rand.NewSource(seed))

	for i := 0; i < 100; i++ {
		dim := rng.Intn(128) + 1
		collection := NewCollection("fuzz", dim, func(a, b []float32) float32 {
			sum := float32(0)
			for j := range a {
				d := a[j] - b[j]
				sum += d * d
			}
			return sum
		})

		numOps := rng.Intn(50) + 10
		addedVectors := make(map[string][]float32)

		for j := 0; j < numOps; j++ {
			op := rng.Intn(3)

			switch op {
			case 0: // Add
				id := fmt.Sprintf("v%d_%d", i, j)
				vec := randomFloat32Slice(dim, rng)
				err := collection.AddVector(id, vec, nil)
				if err != nil {
					t.Fatalf("Failed to add vector: %v", err)
				}
				addedVectors[id] = vec

			case 1: // Get (if we have vectors)
				if len(addedVectors) > 0 {
					var ids []string
					for id := range addedVectors {
						ids = append(ids, id)
					}
					randomID := ids[rng.Intn(len(ids))]
					_, _, err := collection.GetVector(randomID)
					if err != nil {
						t.Fatalf("Failed to get vector: %v", err)
					}
				}

			case 2: // Search (if we have vectors)
				if len(addedVectors) > 0 {
					query := randomFloat32Slice(dim, rng)
					_, err := collection.Search(query, 10)
					if err != nil {
						t.Fatalf("Failed to search: %v", err)
					}
				}
			}
		}
	}
}

func TestFuzz_ConcurrentAdds(t *testing.T) {
	seed := int64(54321)
	_ = rand.New(rand.NewSource(seed))

	numGoroutines := 10
	vectorsPerGoroutine := 20

	var wg sync.WaitGroup
	errChan := make(chan error, numGoroutines*vectorsPerGoroutine)

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(goroutineID int) {
			defer wg.Done()
			localRng := rand.New(rand.NewSource(int64(goroutineID) + seed))

			collection := NewCollection("fuzz-concurrent", 16, func(a, b []float32) float32 {
				sum := float32(0)
				for j := range a {
					d := a[j] - b[j]
					sum += d * d
				}
				return sum
			})

			for j := 0; j < vectorsPerGoroutine; j++ {
				id := fmt.Sprintf("g%d_v%d", goroutineID, j)
				vec := randomFloat32Slice(16, localRng)
				err := collection.AddVector(id, vec, nil)
				if err != nil {
					errChan <- fmt.Errorf("goroutine %d: %w", goroutineID, err)
				}
			}
		}(i)
	}

	wg.Wait()
	close(errChan)

	for err := range errChan {
		t.Errorf("Concurrent add error: %v", err)
	}
}

func TestFuzz_ParallelSearch(t *testing.T) {
	seed := int64(99999)
	rng := rand.New(rand.NewSource(seed))

	collection := NewCollection("fuzz-parallel-search", 32, func(a, b []float32) float32 {
		sum := float32(0)
		for j := range a {
			d := a[j] - b[j]
			sum += d * d
		}
		return sum
	})

	// Add vectors
	for i := 0; i < 100; i++ {
		vec := randomFloat32Slice(32, rng)
		_ = collection.AddVector(fmt.Sprintf("v%d", i), vec, nil)
	}

	// Concurrent searches
	var wg sync.WaitGroup
	searchErrors := make(chan error, 50)

	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			query := randomFloat32Slice(32, rng)
			results, err := collection.Search(query, 10)
			if err != nil {
				searchErrors <- err
				return
			}
			// Verify results are sorted
			for j := 0; j < len(results)-1; j++ {
				if results[j].Distance > results[j+1].Distance {
					searchErrors <- fmt.Errorf("results not sorted at index %d", j)
					return
				}
			}
		}(i)
	}

	wg.Wait()
	close(searchErrors)

	for err := range searchErrors {
		t.Errorf("Search error: %v", err)
	}
}

// Stress tests - large datasets, concurrent workloads

func TestStress_LargeDataset(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping stress test in short mode")
	}

	const numVectors = 10000
	dim := 128

	collection := NewCollection("stress-large", dim, func(a, b []float32) float32 {
		sum := float32(0)
		for j := range a {
			d := a[j] - b[j]
			sum += d * d
		}
		return sum
	})

	// Add vectors
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < numVectors; i++ {
		vec := randomFloat32Slice(dim, rng)
		err := collection.AddVector(fmt.Sprintf("v%d", i), vec, map[string]string{
			"index": fmt.Sprintf("%d", i),
			"batch": fmt.Sprintf("%d", i/1000),
		})
		if err != nil {
			t.Fatalf("Failed to add vector: %v", err)
		}
	}

	// Verify count
	if collection.Count() != numVectors {
		t.Errorf("Expected %d vectors, got %d", numVectors, collection.Count())
	}

	// Search
	query := randomFloat32Slice(dim, rng)
	results, err := collection.Search(query, 100)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) == 0 {
		t.Error("Search returned no results")
	}

	// Verify results sorted
	for i := 0; i < len(results)-1; i++ {
		if results[i].Distance > results[i+1].Distance {
			t.Error("Results not sorted")
			break
		}
	}
}

func TestStress_ConcurrentWrites(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping stress test in short mode")
	}

	const numGoroutines = 20
	const vectorsPerGoroutine = 500

	tempDir, err := os.MkdirTemp("", "quiver-stress-writes-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	manager, err := NewManager(tempDir, 0)
	if err != nil {
		t.Fatalf("Failed to create manager: %v", err)
	}
	defer manager.Stop()

	var wg sync.WaitGroup
	errorCount := atomic.Int32{}

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(goroutineID int) {
			defer wg.Done()

			collection := NewCollection(
				fmt.Sprintf("stress_%d", goroutineID),
				64,
				func(a, b []float32) float32 {
					sum := float32(0)
					for j := range a {
						d := a[j] - b[j]
						sum += d * d
					}
					return sum
				},
			)
			collection.SetManager(manager)

			rng := rand.New(rand.NewSource(int64(goroutineID)))
			for j := 0; j < vectorsPerGoroutine; j++ {
				vec := randomFloat32Slice(64, rng)
				err := collection.AddVector(
					fmt.Sprintf("g%d_v%d", goroutineID, j),
					vec,
					nil,
				)
				if err != nil {
					errorCount.Add(1)
					return
				}
			}

			// Flush
			path := filepath.Join(tempDir, collection.GetName())
			if err := manager.FlushCollection(collection, path); err != nil {
				errorCount.Add(1)
			}
		}(i)
	}

	wg.Wait()

	if errorCount.Load() > 0 {
		t.Errorf("%d errors during concurrent writes", errorCount.Load())
	}
}

func TestStress_LargeMetadata(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping stress test in short mode")
	}

	collection := NewCollection("stress-metadata", 8, func(a, b []float32) float32 {
		sum := float32(0)
		for i := range a {
			d := a[i] - b[i]
			sum += d * d
		}
		return sum
	})

	// Add vectors with large metadata
	for i := 0; i < 1000; i++ {
		metadata := make(map[string]string)
		// Add 50 metadata entries per vector
		for j := 0; j < 50; j++ {
			metadata[fmt.Sprintf("key_%d", j)] = fmt.Sprintf("value_%d_%d", i, j)
		}

		err := collection.AddVector(
			fmt.Sprintf("v%d", i),
			[]float32{1, 2, 3, 4, 5, 6, 7, 8},
			metadata,
		)
		if err != nil {
			t.Fatalf("Failed to add vector: %v", err)
		}
	}

	// Verify metadata can be retrieved
	for i := 0; i < 100; i++ {
		_, metadata, err := collection.GetVector(fmt.Sprintf("v%d", i))
		if err != nil {
			t.Fatalf("Failed to get vector: %v", err)
		}
		if len(metadata) != 50 {
			t.Errorf("Expected 50 metadata entries, got %d", len(metadata))
		}
	}
}

func TestStress_ManyCollections(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping stress test in short mode")
	}

	tempDir, err := os.MkdirTemp("", "quiver-stress-collections-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	manager, err := NewManager(tempDir, 0)
	if err != nil {
		t.Fatalf("Failed to create manager: %v", err)
	}
	defer manager.Stop()

	const numCollections = 50

	// Create and flush many collections
	for i := 0; i < numCollections; i++ {
		collection := NewCollection(
			fmt.Sprintf("collection_%d", i),
			16,
			func(a, b []float32) float32 { return 0 },
		)

		for j := 0; j < 100; j++ {
			_ = collection.AddVector(
				fmt.Sprintf("v%d", j),
				randomFloat32Slice(16, rand.New(rand.NewSource(int64(j)))),
				nil,
			)
		}

		path := filepath.Join(tempDir, collection.GetName())
		if err := manager.FlushCollection(collection, path); err != nil {
			t.Fatalf("Failed to flush collection %d: %v", i, err)
		}
	}

	// Load all collections
	for i := 0; i < numCollections; i++ {
		collection := NewCollection(
			fmt.Sprintf("collection_%d", i),
			16,
			func(a, b []float32) float32 { return 0 },
		)

		path := filepath.Join(tempDir, fmt.Sprintf("collection_%d", i))
		if err := manager.LoadCollection(collection, path); err != nil {
			t.Fatalf("Failed to load collection %d: %v", i, err)
		}

		if collection.Count() != 100 {
			t.Errorf("Collection %d: expected 100 vectors, got %d", i, collection.Count())
		}
	}
}

// Determinism tests - same query → same results

func TestDeterminism_SameVectorsSameSearchResults(t *testing.T) {
	seed := int64(12345)

	searchWithSeed := func(s int64) []SearchResult {
		collection := NewCollection("det", 16, func(a, b []float32) float32 {
			sum := float32(0)
			for i := range a {
				d := a[i] - b[i]
				sum += d * d
			}
			return sum
		})

		rng := rand.New(rand.NewSource(s))
		for i := 0; i < 50; i++ {
			vec := randomFloat32Slice(16, rng)
			_ = collection.AddVector(fmt.Sprintf("v%d", i), vec, nil)
		}

		query := randomFloat32Slice(16, rand.New(rand.NewSource(9999)))
		results, _ := collection.Search(query, 10)
		return results
	}

	results1 := searchWithSeed(seed)
	results2 := searchWithSeed(seed)

	if len(results1) != len(results2) {
		t.Fatalf("Result lengths differ: %d vs %d", len(results1), len(results2))
	}

	for i := range results1 {
		if results1[i].ID != results2[i].ID {
			t.Errorf("Result %d: got ID %s, want %s", i, results2[i].ID, results1[i].ID)
		}
		if math.Abs(float64(results1[i].Distance-results2[i].Distance)) > 0.0001 {
			t.Errorf("Result %d: got distance %v, want %v", i, results2[i].Distance, results1[i].Distance)
		}
	}
}

func TestDeterminism_CollectionSerialization(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "quiver-det-serialization-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	manager, err := NewManager(tempDir, 0)
	if err != nil {
		t.Fatalf("Failed to create manager: %v", err)
	}
	defer manager.Stop()

	// Create and save collection
	original := NewCollection("det-ser", 8, func(a, b []float32) float32 {
		sum := float32(0)
		for i := range a {
			d := a[i] - b[i]
			sum += d * d
		}
		return sum
	})

	rng := rand.New(rand.NewSource(42))
	for i := 0; i < 100; i++ {
		vec := randomFloat32Slice(8, rng)
		_ = original.AddVector(fmt.Sprintf("v%d", i), vec, map[string]string{
			"idx": fmt.Sprintf("%d", i),
		})
	}

	path := filepath.Join(tempDir, "det-ser")
	err = manager.FlushCollection(original, path)
	if err != nil {
		t.Fatalf("Failed to flush: %v", err)
	}

	// Load twice and compare
	loaded1 := NewCollection("det-ser", 8, func(a, b []float32) float32 {
		sum := float32(0)
		for i := range a {
			d := a[i] - b[i]
			sum += d * d
		}
		return sum
	})

	loaded2 := NewCollection("det-ser", 8, func(a, b []float32) float32 {
		sum := float32(0)
		for i := range a {
			d := a[i] - b[i]
			sum += d * d
		}
		return sum
	})

	err = manager.LoadCollection(loaded1, path)
	if err != nil {
		t.Fatalf("Failed to load 1: %v", err)
	}

	err = manager.LoadCollection(loaded2, path)
	if err != nil {
		t.Fatalf("Failed to load 2: %v", err)
	}

	// Both should have same count
	if loaded1.Count() != loaded2.Count() {
		t.Errorf("Counts differ: %d vs %d", loaded1.Count(), loaded2.Count())
	}

	// Same queries should produce same results
	query := randomFloat32Slice(8, rand.New(rand.NewSource(999)))
	results1, _ := loaded1.Search(query, 10)
	results2, _ := loaded2.Search(query, 10)

	if len(results1) != len(results2) {
		t.Fatalf("Result lengths differ: %d vs %d", len(results1), len(results2))
	}

	for i := range results1 {
		if results1[i].ID != results2[i].ID {
			t.Errorf("Results differ at index %d: %s vs %s", i, results1[i].ID, results2[i].ID)
		}
	}
}

// Edge cases

func TestEdgeCase_EmptyCollectionSearch(t *testing.T) {
	collection := NewCollection("empty", 4, func(a, b []float32) float32 { return 0 })

	results, err := collection.Search([]float32{1, 2, 3, 4}, 10)
	if err != nil {
		t.Fatalf("Search on empty failed: %v", err)
	}

	if len(results) != 0 {
		t.Errorf("Expected 0 results, got %d", len(results))
	}
}

func TestEdgeCase_SingleVectorSearch(t *testing.T) {
	collection := NewCollection("single", 3, func(a, b []float32) float32 {
		sum := float32(0)
		for i := range a {
			d := a[i] - b[i]
			sum += d * d
		}
		return sum
	})

	_ = collection.AddVector("v1", []float32{1, 2, 3}, nil)

	results, err := collection.Search([]float32{1, 2, 3}, 1)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) != 1 {
		t.Errorf("Expected 1 result, got %d", len(results))
	}

	if results[0].ID != "v1" {
		t.Errorf("Expected v1, got %s", results[0].ID)
	}

	// Distance should be 0 for exact match
	if results[0].Distance != 0 {
		t.Errorf("Expected distance 0, got %f", results[0].Distance)
	}
}

func TestEdgeCase_DuplicateVectorAdd(t *testing.T) {
	collection := NewCollection("dup", 3, func(a, b []float32) float32 { return 0 })

	err := collection.AddVector("v1", []float32{1, 2, 3}, nil)
	if err != nil {
		t.Fatalf("First add failed: %v", err)
	}

	// Duplicate ID should replace (or error depending on implementation)
	err = collection.AddVector("v1", []float32{4, 5, 6}, nil)
	// Check if count is still 1 or if it replaced
	if collection.Count() > 1 {
		t.Logf("Duplicate add: collection has %d vectors", collection.Count())
	}
}

func TestEdgeCase_DeleteNonExistent(t *testing.T) {
	collection := NewCollection("del", 3, func(a, b []float32) float32 { return 0 })

	err := collection.DeleteVector("nonexistent")
	if err == nil {
		t.Error("Expected error when deleting non-existent vector")
	}
}

func TestEdgeCase_SearchWithKGreaterThanCount(t *testing.T) {
	collection := NewCollection("k-greater", 3, func(a, b []float32) float32 {
		sum := float32(0)
		for i := range a {
			d := a[i] - b[i]
			sum += d * d
		}
		return sum
	})

	_ = collection.AddVector("v1", []float32{1, 2, 3}, nil)
	_ = collection.AddVector("v2", []float32{4, 5, 6}, nil)

	results, err := collection.Search([]float32{1, 2, 3}, 100)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// Should return at most 2 results
	if len(results) > 2 {
		t.Errorf("Expected at most 2 results, got %d", len(results))
	}
}

func TestEdgeCase_ZeroDimension(t *testing.T) {
	collection := NewCollection("zero-dim", 0, func(a, b []float32) float32 { return 0 })

	err := collection.AddVector("v1", []float32{}, nil)
	if err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	results, err := collection.Search([]float32{}, 1)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) != 1 {
		t.Errorf("Expected 1 result, got %d", len(results))
	}
}

func TestEdgeCase_MaxDimension(t *testing.T) {
	const maxDim = 2048
	collection := NewCollection("max-dim", maxDim, func(a, b []float32) float32 {
		sum := float32(0)
		for i := range a {
			d := a[i] - b[i]
			sum += d * d
		}
		return sum
	})

	vec := make([]float32, maxDim)
	for i := 0; i < maxDim; i++ {
		vec[i] = float32(i) / float32(maxDim)
	}

	err := collection.AddVector("v1", vec, nil)
	if err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	results, err := collection.Search(vec, 1)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) != 1 || results[0].Distance != 0 {
		t.Errorf("Expected exact match")
	}
}

func TestEdgeCase_VeryLargeK(t *testing.T) {
	collection := NewCollection("large-k", 3, func(a, b []float32) float32 {
		sum := float32(0)
		for i := range a {
			d := a[i] - b[i]
			sum += d * d
		}
		return sum
	})

	for i := 0; i < 10; i++ {
		_ = collection.AddVector(fmt.Sprintf("v%d", i), []float32{float32(i), 0, 0}, nil)
	}

	// Search with k=0 should error or return empty
	_, err := collection.Search([]float32{0, 0, 0}, 0)
	if err == nil {
		t.Log("k=0 returned no error")
	}
}

func TestEdgeCase_InvalidDimension(t *testing.T) {
	collection := NewCollection("inv-dim", 3, func(a, b []float32) float32 { return 0 })

	// Add vector with wrong dimension
	err := collection.AddVector("v1", []float32{1, 2}, nil)
	if err == nil {
		t.Error("Expected error for dimension mismatch")
	}
}

func TestEdgeCase_NilMetadata(t *testing.T) {
	collection := NewCollection("nil-meta", 3, func(a, b []float32) float32 { return 0 })

	// Add with nil metadata - should not panic
	err := collection.AddVector("v1", []float32{1, 2, 3}, nil)
	if err != nil {
		t.Fatalf("Add with nil metadata failed: %v", err)
	}

	// Retrieve and verify
	_, meta, err := collection.GetVector("v1")
	if err != nil {
		t.Fatalf("GetVector failed: %v", err)
	}

	if meta != nil && len(meta) != 0 {
		t.Logf("Got metadata: %v", meta)
	}
}

func TestEdgeCase_CollectionNotFound(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "quiver-edge-notfound-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	manager, err := NewManager(tempDir, 0)
	if err != nil {
		t.Fatalf("Failed to create manager: %v", err)
	}
	defer manager.Stop()

	// Try to load non-existent collection
	collection := NewCollection("nonexistent", 3, func(a, b []float32) float32 { return 0 })
	err = manager.LoadCollection(collection, filepath.Join(tempDir, "nonexistent"))

	// Should not error, just return empty collection
	if err != nil {
		t.Logf("LoadCollection error: %v", err)
	}

	if collection.Count() != 0 {
		t.Errorf("Expected empty collection, got %d vectors", collection.Count())
	}
}

func TestEdgeCase_ParquetFailureFallback(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "quiver-edge-parquet-fail-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	manager, err := NewManager(tempDir, 0)
	if err != nil {
		t.Fatalf("Failed to create manager: %v", err)
	}
	defer manager.Stop()

	collectionName := "test-parquet-fallback"
	collectionPath := filepath.Join(tempDir, collectionName)

	// Create collection directory
	os.MkdirAll(collectionPath, 0755)

	// Write config
	config := CollectionConfig{
		Name:         collectionName,
		Dimension:    3,
		DistanceFunc: "cosine",
		CreatedAt:    time.Now(),
	}
	_ = SaveCollectionConfig(config, filepath.Join(collectionPath, "config.json"))

	// Write vectors as JSON only (no parquet)
	vectors := []VectorRecord{
		{ID: "v1", Vector: []float32{1, 2, 3}},
		{ID: "v2", Vector: []float32{4, 5, 6}},
	}
	_ = WriteVectorsToFile(vectors, filepath.Join(collectionPath, "vectors.json"))

	// Load - should fallback to JSON
	collection := NewCollection(collectionName, 3, func(a, b []float32) float32 { return 0 })
	err = manager.LoadCollection(collection, collectionPath)
	if err != nil {
		t.Fatalf("LoadCollection failed: %v", err)
	}

	if collection.Count() != 2 {
		t.Errorf("Expected 2 vectors, got %d", collection.Count())
	}
}

// Helper functions

func randomFloat32Slice(dim int, rng *rand.Rand) []float32 {
	vec := make([]float32, dim)
	for i := 0; i < dim; i++ {
		vec[i] = rng.Float32()
	}
	return vec
}

// Benchmark tests

func Benchmark_AddVectors(b *testing.B) {
	dim := 128
	numVectors := 10000

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		collection := NewCollection("bench", dim, func(a, b []float32) float32 {
			sum := float32(0)
			for j := range a {
				d := a[j] - b[j]
				sum += d * d
			}
			return sum
		})

		rng := rand.New(rand.NewSource(int64(i)))
		for j := 0; j < numVectors; j++ {
			vec := randomFloat32Slice(dim, rng)
			_ = collection.AddVector(fmt.Sprintf("v%d", j), vec, nil)
		}
	}
}

func Benchmark_SearchVectors(b *testing.B) {
	dim := 128
	numVectors := 5000

	// Setup
	collection := NewCollection("bench-search", dim, func(a, b []float32) float32 {
		sum := float32(0)
		for j := range a {
			d := a[j] - b[j]
			sum += d * d
		}
		return sum
	})

	rng := rand.New(rand.NewSource(42))
	for i := 0; i < numVectors; i++ {
		vec := randomFloat32Slice(dim, rng)
		_ = collection.AddVector(fmt.Sprintf("v%d", i), vec, nil)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		query := randomFloat32Slice(dim, rand.New(rand.NewSource(int64(i))))
		_, _ = collection.Search(query, 10)
	}
}

func Benchmark_ConcurrentAdds(b *testing.B) {
	dim := 64
	numGoroutines := 10
	vectorsPerGoroutine := 1000

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var wg sync.WaitGroup
		for g := 0; g < numGoroutines; g++ {
			wg.Add(1)
			go func(goroutineID int) {
				defer wg.Done()
				collection := NewCollection("bench-concurrent", dim, func(a, b []float32) float32 {
					sum := float32(0)
					for j := range a {
						d := a[j] - b[j]
						sum += d * d
					}
					return sum
				})

				rng := rand.New(rand.NewSource(int64(goroutineID)))
				for j := 0; j < vectorsPerGoroutine; j++ {
					vec := randomFloat32Slice(dim, rng)
					_ = collection.AddVector(fmt.Sprintf("v%d", j), vec, nil)
				}
			}(g)
		}
		wg.Wait()
	}
}

func Benchmark_PersistenceFlush(b *testing.B) {
	tempDir, err := os.MkdirTemp("", "quiver-bench-flush-*")
	if err != nil {
		b.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	manager, err := NewManager(tempDir, 0)
	if err != nil {
		b.Fatalf("Failed to create manager: %v", err)
	}
	defer manager.Stop()

	numVectors := 1000

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		collection := NewCollection(fmt.Sprintf("bench-flush-%d", i), 64, func(a, b []float32) float32 { return 0 })

		rng := rand.New(rand.NewSource(int64(i)))
		for j := 0; j < numVectors; j++ {
			vec := randomFloat32Slice(64, rng)
			_ = collection.AddVector(fmt.Sprintf("v%d", j), vec, nil)
		}

		path := filepath.Join(tempDir, collection.GetName())
		_ = manager.FlushCollection(collection, path)
	}
}
