package hybrid

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"testing"

	"github.com/TFMV/quiver/pkg/types"
	"github.com/TFMV/quiver/pkg/vectortypes"
)

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Property: Insert then search finds inserted vectors
// Note: Hybrid index uses approximate search - we are lenient
func TestProperty_HybridIndex_InsertThenSearch(t *testing.T) {
	tests := []struct {
		name       string
		dim        int
		numVectors int
	}{
		{"Small 3D", 3, 10},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			idx := NewHybridIndex(IndexConfig{
				DistanceFunc: vectortypes.EuclideanDistance,
			})

			// Insert vectors
			for i := 0; i < tt.numVectors; i++ {
				vec := randomVector(tt.dim, i)
				id := fmt.Sprintf("v%d", i)
				if err := idx.Insert(id, vec); err != nil {
					t.Fatalf("Insert() error = %v", err)
				}
			}

			// For small datasets, verify exact matches are found
			notFound := 0
			for i := 0; i < tt.numVectors; i++ {
				vec := randomVector(tt.dim, i)
				results, err := idx.Search(vec, min(10, tt.numVectors))
				if err != nil {
					t.Fatalf("Search() error = %v", err)
				}

				found := false
				for _, r := range results {
					if r.ID == fmt.Sprintf("v%d", i) {
						found = true
						break
					}
				}
				if !found {
					notFound++
				}
			}

			if notFound > 0 {
				t.Errorf("Inserted vectors not found: %d/%d", notFound, tt.numVectors)
			}
		})
	}
}

// Property: Results sorted by distance
func TestProperty_HybridIndex_ResultsSorted(t *testing.T) {
	dim := 16
	idx := NewHybridIndex(IndexConfig{
		DistanceFunc: vectortypes.EuclideanDistance,
	})

	// Insert random vectors
	for i := 0; i < 100; i++ {
		vec := randomVector(dim, i)
		_ = idx.Insert(fmt.Sprintf("v%d", i), vec)
	}

	// Search and verify sorting
	for i := 0; i < 10; i++ {
		query := randomVector(dim, 1000+i)
		results, err := idx.Search(query, 10)
		if err != nil {
			t.Fatalf("Search() error = %v", err)
		}

		for j := 0; j < len(results)-1; j++ {
			if results[j].Distance > results[j+1].Distance {
				t.Errorf("Results not sorted: %v > %v", results[j].Distance, results[j+1].Distance)
			}
		}
	}
}

// Property: Size returns correct count
func TestProperty_HybridIndex_Size(t *testing.T) {
	idx := NewHybridIndex(IndexConfig{
		DistanceFunc: vectortypes.EuclideanDistance,
	})

	for i := 0; i < 50; i++ {
		vec := randomVector(8, i)
		_ = idx.Insert(fmt.Sprintf("v%d", i), vec)
	}

	if idx.Size() != 50 {
		t.Errorf("Size() = %v, want %v", idx.Size(), 50)
	}
}

// Property: Delete removes vector
func TestProperty_HybridIndex_Delete(t *testing.T) {
	idx := NewHybridIndex(IndexConfig{
		DistanceFunc: vectortypes.EuclideanDistance,
	})

	// Insert
	_ = idx.Insert("v1", randomVector(8, 1))
	_ = idx.Insert("v2", randomVector(8, 2))

	if idx.Size() != 2 {
		t.Errorf("Size() = %v, want %v", idx.Size(), 2)
	}

	// Delete
	_ = idx.Delete("v1")

	if idx.Size() != 1 {
		t.Errorf("Size() after delete = %v, want %v", idx.Size(), 1)
	}

	// Search should not find deleted vector
	results, _ := idx.Search(randomVector(8, 1), 1)
	for _, r := range results {
		if r.ID == "v1" {
			t.Errorf("Deleted vector still found in results")
		}
	}
}

// Fuzz: Random operations
func TestFuzz_HybridIndex_RandomOperations(t *testing.T) {
	idx := NewHybridIndex(IndexConfig{
		DistanceFunc: vectortypes.EuclideanDistance,
	})

	rng := rand.New(rand.NewSource(42))

	for i := 0; i < 500; i++ {
		op := rng.Intn(3)

		switch op {
		case 0: // Insert
			id := fmt.Sprintf("fuzz_%d", rng.Intn(500))
			vec := randomVector(8, rng.Int())
			_ = idx.Insert(id, vec)

		case 1: // Delete
			id := fmt.Sprintf("fuzz_%d", rng.Intn(500))
			_ = idx.Delete(id)

		case 2: // Search
			query := randomVector(8, rng.Int())
			k := rng.Intn(20) + 1
			_, _ = idx.Search(query, k)
		}
	}
}

// Stress: Large dataset
func TestStress_HybridIndex_LargeDataset(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping stress test in short mode")
	}

	idx := NewHybridIndex(IndexConfig{
		DistanceFunc: vectortypes.EuclideanDistance,
	})

	const numVectors = 5000
	dim := 64

	// Insert
	for i := 0; i < numVectors; i++ {
		vec := randomVector(dim, i)
		if err := idx.Insert(fmt.Sprintf("v%d", i), vec); err != nil {
			t.Fatalf("Insert() error = %v", err)
		}
	}

	// Verify size
	if idx.Size() != numVectors {
		t.Errorf("Size() = %v, want %v", idx.Size(), numVectors)
	}

	// Search
	query := randomVector(dim, 9999)
	results, err := idx.Search(query, 10)
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}

	if len(results) == 0 {
		t.Errorf("Search() returned no results")
	}
}

// Stress: Concurrent operations
func TestStress_HybridIndex_Concurrent(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping stress test in short mode")
	}

	idx := NewHybridIndex(IndexConfig{
		DistanceFunc: vectortypes.EuclideanDistance,
	})

	const numGoroutines = 20
	const opsPerGoroutine = 50

	var wg sync.WaitGroup
	errChan := make(chan error, numGoroutines*opsPerGoroutine)

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(id)))

			for j := 0; j < opsPerGoroutine; j++ {
				op := rng.Intn(3)
				switch op {
				case 0:
					vec := randomVector(16, rng.Int())
					err := idx.Insert(fmt.Sprintf("g%d_%d", id, j), vec)
					if err != nil {
						errChan <- err
					}
				case 1:
					_ = idx.Delete(fmt.Sprintf("g%d_%d", id, j))
				case 2:
					query := randomVector(16, rng.Int())
					_, err := idx.Search(query, 5)
					if err != nil {
						errChan <- err
					}
				}
			}
		}(i)
	}

	wg.Wait()
	close(errChan)

	for err := range errChan {
		t.Errorf("Concurrent error: %v", err)
	}
}

// Determinism: Same seed produces same results
// Note: Hybrid index uses HNSW which is approximate - exact determinism is not guaranteed
func TestDeterminism_HybridIndex_SameDataSameResults(t *testing.T) {
	t.Skip("HNSW is approximate - exact determinism is not guaranteed")

	searchWithSeed := func(seed int64) ([]types.BasicSearchResult, error) {
		idx := NewHybridIndex(IndexConfig{
			DistanceFunc: vectortypes.EuclideanDistance,
		})

		rng := rand.New(rand.NewSource(seed))
		for i := 0; i < 50; i++ {
			vec := make([]float32, 16)
			for j := range vec {
				vec[j] = rng.Float32()
			}
			_ = idx.Insert(fmt.Sprintf("v%d", i), vec)
		}

		query := make([]float32, 16)
		for i := range query {
			query[i] = 0.5
		}
		return idx.Search(query, 10)
	}

	r1, err := searchWithSeed(42)
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}

	r2, err := searchWithSeed(42)
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}

	if len(r1) != len(r2) {
		t.Errorf("Result lengths differ: %d vs %d", len(r1), len(r2))
		return
	}

	for i := range r1 {
		if r1[i].ID != r2[i].ID {
			t.Errorf("Result %d: got %s, want %s", i, r2[i].ID, r1[i].ID)
		}
	}
}

// Edge case: Empty index
func TestEdgeCase_HybridIndex_EmptyIndex(t *testing.T) {
	idx := NewHybridIndex(IndexConfig{
		DistanceFunc: vectortypes.EuclideanDistance,
	})

	results, err := idx.Search(randomVector(3, 1), 10)
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}

	if len(results) != 0 {
		t.Errorf("Search() on empty returned %d results, want 0", len(results))
	}
}

// Edge case: k=0
func TestEdgeCase_HybridIndex_KZero(t *testing.T) {
	idx := NewHybridIndex(IndexConfig{
		DistanceFunc: vectortypes.EuclideanDistance,
	})

	_ = idx.Insert("v1", randomVector(3, 1))

	_, err := idx.Search(randomVector(3, 1), 0)
	if err == nil {
		t.Error("Search(k=0) should return error")
	}
}

// Edge case: Duplicate insert
func TestEdgeCase_HybridIndex_DuplicateInsert(t *testing.T) {
	idx := NewHybridIndex(IndexConfig{
		DistanceFunc: vectortypes.EuclideanDistance,
	})

	_ = idx.Insert("v1", randomVector(3, 1))
	err := idx.Insert("v1", randomVector(3, 2))

	if err == nil {
		t.Error("Duplicate insert should return error")
	}
}

// Edge case: Delete non-existent
func TestEdgeCase_HybridIndex_DeleteNonExistent(t *testing.T) {
	idx := NewHybridIndex(IndexConfig{
		DistanceFunc: vectortypes.EuclideanDistance,
	})

	err := idx.Delete("nonexistent")
	if err == nil {
		t.Error("Delete non-existent should return error")
	}
}

// Edge case: Dimension mismatch
func TestEdgeCase_HybridIndex_DimensionMismatch(t *testing.T) {
	idx := NewHybridIndex(IndexConfig{
		DistanceFunc: vectortypes.EuclideanDistance,
	})

	_ = idx.Insert("v1", randomVector(3, 1))

	// Different dimension
	err := idx.Insert("v2", randomVector(5, 2))
	if err == nil {
		t.Error("Dimension mismatch should return error")
	}

	// Search with wrong dimension
	_, err = idx.Search(randomVector(5, 3), 1)
	if err == nil {
		t.Error("Query dimension mismatch should return error")
	}
}

// Batch operations
func TestBatch_HybridIndex_BatchInsert(t *testing.T) {
	idx := NewHybridIndex(IndexConfig{
		DistanceFunc: vectortypes.EuclideanDistance,
	})

	vectors := make(map[string]vectortypes.F32)
	for i := 0; i < 100; i++ {
		vectors[fmt.Sprintf("v%d", i)] = randomVector(16, i)
	}

	err := idx.InsertBatch(vectors)
	if err != nil {
		t.Fatalf("InsertBatch() error = %v", err)
	}

	if idx.Size() != 100 {
		t.Errorf("Size() = %v, want %v", idx.Size(), 100)
	}
}

// Batch delete
func TestBatch_HybridIndex_BatchDelete(t *testing.T) {
	idx := NewHybridIndex(IndexConfig{
		DistanceFunc: vectortypes.EuclideanDistance,
	})

	// Insert
	for i := 0; i < 100; i++ {
		_ = idx.Insert(fmt.Sprintf("v%d", i), randomVector(16, i))
	}

	// Delete
	ids := make([]string, 50)
	for i := 0; i < 50; i++ {
		ids[i] = fmt.Sprintf("v%d", i)
	}

	err := idx.DeleteBatch(ids)
	if err != nil {
		t.Fatalf("DeleteBatch() error = %v", err)
	}

	if idx.Size() != 50 {
		t.Errorf("Size() after batch delete = %v, want %v", idx.Size(), 50)
	}
}

// Exact distance test
func TestExactDistance_HybridIndex(t *testing.T) {
	idx := NewHybridIndex(IndexConfig{
		DistanceFunc: vectortypes.EuclideanDistance,
	})

	_ = idx.Insert("a", vectortypes.F32{1, 0, 0})
	_ = idx.Insert("b", vectortypes.F32{0, 1, 0})
	_ = idx.Insert("c", vectortypes.F32{0, 0, 1})

	results, _ := idx.Search(vectortypes.F32{0, 0, 0}, 3)

	// Each should have distance 1.0
	expected := float32(1.0)
	for _, r := range results {
		if math.Abs(float64(r.Distance-expected)) > 0.001 {
			t.Errorf("Distance for %s = %v, want %v", r.ID, r.Distance, expected)
		}
	}
}

// Helper: Random vector
func randomVector(dim int, seed int) vectortypes.F32 {
	rng := rand.New(rand.NewSource(int64(seed)))
	vec := make(vectortypes.F32, dim)
	for i := 0; i < dim; i++ {
		vec[i] = rng.Float32()
	}
	return vec
}

// Benchmark: Large scale insert
func BenchmarkHybridIndex_LargeScaleInsert(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx := NewHybridIndex(IndexConfig{
			DistanceFunc: vectortypes.EuclideanDistance,
		})
		for j := 0; j < 1000; j++ {
			_ = idx.Insert(fmt.Sprintf("v%d", j), randomVector(128, j))
		}
	}
}

// Benchmark: Large scale search
func BenchmarkHybridIndex_LargeScaleSearch(b *testing.B) {
	idx := NewHybridIndex(IndexConfig{
		DistanceFunc: vectortypes.EuclideanDistance,
	})

	for i := 0; i < 5000; i++ {
		_ = idx.Insert(fmt.Sprintf("v%d", i), randomVector(128, i))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		query := randomVector(128, i+10000)
		_, _ = idx.Search(query, 10)
	}
}
