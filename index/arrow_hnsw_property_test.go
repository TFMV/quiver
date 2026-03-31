package index

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"
)

// Property test: Insert then search should find inserted vectors
func TestProperty_ArrowHNSW_InsertThenSearch(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping Arrow HNSW test in short mode")
	}

	idx := NewArrowHNSWIndex(3)

	vec1 := array.NewFloat32Builder(memory.DefaultAllocator)
	vec1.AppendValues([]float32{1, 2, 3}, nil)
	arr1 := vec1.NewArray().(*array.Float32)

	// Ensure the array is released when the test ends
	defer arr1.Release()
	defer vec1.Release()

	err := idx.Add(arr1, "v1")
	if err != nil {
		t.Fatalf("Failed to add vector: %v", err)
	}

	results, err := idx.Search(arr1, 1)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) == 0 {
		t.Fatal("Inserted vector not found")
	}
	if results[0].ID != "v1" {
		t.Errorf("Expected ID 'v1', got '%s'", results[0].ID)
	}
}

// Helper: Create float32 array with seed for reproducibility
// Note: The caller is responsible for calling Release() on the returned array.
func makeFloat32Array(dim int, seed int) *array.Float32 {
	builder := array.NewFloat32Builder(memory.DefaultAllocator)
	// Only release the builder here. The array outlives this function.
	defer builder.Release()

	vec := make([]float32, dim)
	rng := rand.New(rand.NewSource(int64(seed)))
	for i := 0; i < dim; i++ {
		vec[i] = rng.Float32()
	}
	builder.AppendValues(vec, nil)
	return builder.NewArray().(*array.Float32)
}

// Helper: Create vector from values
// Note: The caller is responsible for calling Release() on the returned array.
func float32Vector(values ...float32) *array.Float32 {
	builder := array.NewFloat32Builder(memory.DefaultAllocator)
	defer builder.Release()

	builder.AppendValues(values, nil)
	return builder.NewArray().(*array.Float32)
}

// Benchmark: Large scale insert
func BenchmarkArrowHNSW_LargeScaleInsert(b *testing.B) {
	dim := 128
	numInserts := 1000

	// Pre-generate arrays to avoid timing Arrow allocations
	vectors := make([]*array.Float32, numInserts)
	ids := make([]string, numInserts)
	for j := 0; j < numInserts; j++ {
		vectors[j] = makeFloat32Array(dim, j)
		ids[j] = fmt.Sprintf("v%d", j)
	}

	// Cleanup pre-allocated arrays after benchmark
	defer func() {
		for _, v := range vectors {
			v.Release()
		}
	}()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Pause timer to avoid penalizing the benchmark for index creation
		b.StopTimer()
		idx := NewArrowHNSWIndex(dim)
		b.StartTimer()

		for j := 0; j < numInserts; j++ {
			_ = idx.Add(vectors[j], ids[j])
		}
	}
}

// Benchmark: Large scale search
func BenchmarkArrowHNSW_LargeScaleSearch(b *testing.B) {
	dim := 128
	numVectors := 5000

	idx := NewArrowHNSWIndex(dim)

	// Pre-populate the index
	for i := 0; i < numVectors; i++ {
		vec := makeFloat32Array(dim, i)
		_ = idx.Add(vec, fmt.Sprintf("v%d", i))
		vec.Release() // Safe to release immediately after Add
	}

	// Pre-generate queries so we only benchmark the Search algorithm
	numQueries := 100
	queries := make([]*array.Float32, numQueries)
	for i := 0; i < numQueries; i++ {
		queries[i] = makeFloat32Array(dim, i+10000)
	}

	defer func() {
		for _, q := range queries {
			q.Release()
		}
	}()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Use modulo to cycle through our pre-generated queries
		query := queries[i%numQueries]
		_, _ = idx.Search(query, 10)
	}
}

// Test exact Euclidean distance
func TestExactDistance_ArrowHNSW(t *testing.T) {
	idx := NewArrowHNSWIndex(3)

	vecA := float32Vector(1, 0, 0)
	vecB := float32Vector(0, 1, 0)
	vecC := float32Vector(0, 0, 1)
	query := float32Vector(0, 0, 0)

	defer vecA.Release()
	defer vecB.Release()
	defer vecC.Release()
	defer query.Release()

	_ = idx.Add(vecA, "a")
	_ = idx.Add(vecB, "b")
	_ = idx.Add(vecC, "c")

	results, err := idx.Search(query, 3)
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}

	// All distances should be 1.0 (distance from origin to unit vectors)
	for _, r := range results {
		expected := float32(1.0)
		if math.Abs(float64(r.Distance-expected)) > 0.001 {
			t.Errorf("Distance for %s = %v, want %v", r.ID, r.Distance, expected)
		}
	}
}
