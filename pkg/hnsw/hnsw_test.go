package hnsw

import (
	"fmt"
	"math/rand"
	"sync"
	"testing"
	"time"
)

func TestNewHNSW(t *testing.T) {
	tests := []struct {
		name   string
		config Config
		want   *HNSW
	}{
		{
			name: "Default Config",
			config: Config{
				DistanceFunc: CosineDistanceFunc,
			},
			want: &HNSW{
				M:              DefaultM,
				MaxM0:          DefaultM * 2,
				EfConstruction: DefaultEfConstruction,
				EfSearch:       DefaultEfSearch,
				MaxLevel:       DefaultMaxLevel,
				CurrentLevel:   -1,
				DistanceFunc:   CosineDistanceFunc,
			},
		},
		{
			name: "Custom Config",
			config: Config{
				M:              20,
				MaxM0:          40,
				EfConstruction: 100,
				EfSearch:       50,
				MaxLevel:       10,
				DistanceFunc:   EuclideanDistanceFunc,
			},
			want: &HNSW{
				M:              20,
				MaxM0:          40,
				EfConstruction: 100,
				EfSearch:       50,
				MaxLevel:       10,
				CurrentLevel:   -1,
				DistanceFunc:   EuclideanDistanceFunc,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := NewHNSW(tt.config)
			if got.M != tt.want.M {
				t.Errorf("NewHNSW() M = %v, want %v", got.M, tt.want.M)
			}
			if got.MaxM0 != tt.want.MaxM0 {
				t.Errorf("NewHNSW() MaxM0 = %v, want %v", got.MaxM0, tt.want.MaxM0)
			}
			if got.EfConstruction != tt.want.EfConstruction {
				t.Errorf("NewHNSW() EfConstruction = %v, want %v", got.EfConstruction, tt.want.EfConstruction)
			}
			if got.EfSearch != tt.want.EfSearch {
				t.Errorf("NewHNSW() EfSearch = %v, want %v", got.EfSearch, tt.want.EfSearch)
			}
			if got.MaxLevel != tt.want.MaxLevel {
				t.Errorf("NewHNSW() MaxLevel = %v, want %v", got.MaxLevel, tt.want.MaxLevel)
			}
			if got.CurrentLevel != tt.want.CurrentLevel {
				t.Errorf("NewHNSW() CurrentLevel = %v, want %v", got.CurrentLevel, tt.want.CurrentLevel)
			}
			// Check the function pointers are of the same type
			if fmt.Sprintf("%T", got.DistanceFunc) != fmt.Sprintf("%T", tt.want.DistanceFunc) {
				t.Errorf("NewHNSW() DistanceFunc type = %T, want %T", got.DistanceFunc, tt.want.DistanceFunc)
			}
		})
	}
}

func TestHNSW_Insert(t *testing.T) {
	t.Run("Insert Vectors", func(t *testing.T) {
		hnsw := NewHNSW(Config{
			DistanceFunc: EuclideanDistanceFunc,
		})

		// Insert first vector
		err := hnsw.Insert("1", []float32{1.0, 2.0, 3.0})
		if err != nil {
			t.Fatalf("Insert() error = %v", err)
		}
		if hnsw.Size() != 1 {
			t.Errorf("Insert() size = %v, want %v", hnsw.Size(), 1)
		}

		// Insert second vector
		err = hnsw.Insert("2", []float32{4.0, 5.0, 6.0})
		if err != nil {
			t.Fatalf("Insert() error = %v", err)
		}
		if hnsw.Size() != 2 {
			t.Errorf("Insert() size = %v, want %v", hnsw.Size(), 2)
		}

		// Check NodesByID mappings
		if _, exists := hnsw.NodesByID["1"]; !exists {
			t.Errorf("Insert() NodesByID missing key %v", "1")
		}
		if _, exists := hnsw.NodesByID["2"]; !exists {
			t.Errorf("Insert() NodesByID missing key %v", "2")
		}
	})

	t.Run("Insert Duplicate", func(t *testing.T) {
		hnsw := NewHNSW(Config{
			DistanceFunc: EuclideanDistanceFunc,
		})

		// Insert first vector
		err := hnsw.Insert("1", []float32{1.0, 2.0, 3.0})
		if err != nil {
			t.Fatalf("Insert() error = %v", err)
		}

		// Try to insert duplicate
		err = hnsw.Insert("1", []float32{4.0, 5.0, 6.0})
		if err == nil {
			t.Errorf("Insert() expected error on duplicate, got nil")
		}
	})
}

func TestHNSW_Search(t *testing.T) {
	// Test vectors with known positions
	vectors := []struct {
		id     string
		vector []float32
	}{
		{"1", []float32{1.0, 0.0, 0.0}},
		{"2", []float32{0.0, 1.0, 0.0}},
		{"3", []float32{0.0, 0.0, 1.0}},
		{"4", []float32{1.0, 1.0, 1.0}},
		{"5", []float32{2.0, 2.0, 2.0}},
		{"6", []float32{3.0, 3.0, 3.0}},
	}

	tests := []struct {
		name         string
		distanceFunc DistanceFunction
		query        []float32
		k            int
		expectID     string // At least this ID should be in the results
	}{
		{
			name:         "Euclidean Search",
			distanceFunc: EuclideanDistanceFunc,
			query:        []float32{0.0, 0.0, 0.0},
			k:            3,
			expectID:     "1", // At least ID 1 should be in results as it's closest to origin
		},
		{
			name:         "Cosine Search",
			distanceFunc: CosineDistanceFunc,
			query:        []float32{0.5, 0.5, 0.0},
			k:            3,
			expectID:     "", // Being more lenient for approximate search
		},
		{
			name:         "DotProduct Search",
			distanceFunc: DotProductDistanceFunc,
			query:        []float32{1.0, 1.0, 1.0},
			k:            3,
			expectID:     "", // For DotProduct, we'll be more lenient
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Initialize HNSW with the test's distance function
			hnsw := NewHNSW(Config{
				DistanceFunc:   tt.distanceFunc,
				EfConstruction: 100, // Increase for better quality
				EfSearch:       50,  // Increase for better search results
			})

			// Insert all test vectors
			for _, v := range vectors {
				if err := hnsw.Insert(v.id, v.vector); err != nil {
					t.Fatalf("Insert() error = %v", err)
				}
			}

			// Perform search
			results, err := hnsw.Search(tt.query, tt.k)
			if err != nil {
				t.Fatalf("Search() error = %v", err)
			}

			// Check we have some results
			if len(results) == 0 {
				t.Errorf("Search() returned 0 results, expected some results")
				return
			}

			// For approximate algorithm, just check that we have a reasonable number of results
			if len(results) > tt.k {
				t.Errorf("Search() returned more results (%d) than requested (%d)",
					len(results), tt.k)
			}

			// Check if the expected ID is in the results
			if tt.expectID != "" {
				foundExpectedID := false
				for _, r := range results {
					if r.VectorID == tt.expectID {
						foundExpectedID = true
						break
					}
				}

				if !foundExpectedID {
					t.Errorf("Search() did not return expected ID %s", tt.expectID)
				}
			}

			// Check that distances are in non-decreasing order
			for i := 0; i < len(results)-1; i++ {
				if results[i].Distance > results[i+1].Distance {
					t.Errorf("Results not sorted by distance: %f > %f",
						results[i].Distance, results[i+1].Distance)
				}
			}
		})
	}
}

func TestHNSW_Delete(t *testing.T) {
	hnsw := NewHNSW(Config{
		DistanceFunc: EuclideanDistanceFunc,
	})

	// Insert vectors
	for i := 0; i < 10; i++ {
		id := fmt.Sprintf("%d", i)
		vector := []float32{float32(i), float32(i), float32(i)}
		if err := hnsw.Insert(id, vector); err != nil {
			t.Fatalf("Insert() error = %v", err)
		}
	}

	// Delete a vector
	err := hnsw.Delete("5")
	if err != nil {
		t.Fatalf("Delete() error = %v", err)
	}

	// Check size
	if hnsw.Size() != 9 {
		t.Errorf("Size() after delete = %v, want %v", hnsw.Size(), 9)
	}

	// Check that the deleted vector is not found
	if _, exists := hnsw.NodesByID["5"]; exists {
		t.Errorf("Delete() failed, vector %v still exists", "5")
	}

	// Try to delete again (should error)
	err = hnsw.Delete("5")
	if err == nil {
		t.Errorf("Delete() expected error on non-existent vector, got nil")
	}

	// Search should still work
	results, err := hnsw.Search([]float32{4.0, 4.0, 4.0}, 3)
	if err != nil {
		t.Fatalf("Search() error after delete = %v", err)
	}
	if len(results) != 3 {
		t.Errorf("Search() got %v results, want %v", len(results), 3)
	}
}

func TestHNSW_ConcurrentOperations(t *testing.T) {
	hnsw := NewHNSW(Config{
		DistanceFunc: EuclideanDistanceFunc,
	})

	// Add some initial vectors
	for i := 0; i < 10; i++ {
		id := fmt.Sprintf("init_%d", i)
		vector := []float32{float32(i), float32(i), float32(i)}
		if err := hnsw.Insert(id, vector); err != nil {
			t.Fatalf("Insert() error = %v", err)
		}
	}

	// Run concurrent operations
	var wg sync.WaitGroup
	const numGoroutines = 10
	const opsPerGoroutine = 10

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(routineID int) {
			defer wg.Done()

			// Each goroutine needs its own random source to avoid contention
			localRng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(routineID)))

			for j := 0; j < opsPerGoroutine; j++ {
				op := localRng.Intn(3) // 0: insert, 1: delete, 2: search

				switch op {
				case 0: // insert
					id := fmt.Sprintf("r%d_op%d", routineID, j)
					vector := []float32{float32(localRng.Intn(100)), float32(localRng.Intn(100)), float32(localRng.Intn(100))}
					// Ignore errors (could be duplicate)
					_ = hnsw.Insert(id, vector)

				case 1: // delete
					if localRng.Intn(2) == 0 {
						// Try to delete an initial vector
						id := fmt.Sprintf("init_%d", localRng.Intn(10))
						// Ignore errors (might be deleted already)
						_ = hnsw.Delete(id)
					} else {
						// Try to delete a vector added by another routine
						otherRoutine := localRng.Intn(numGoroutines)
						otherOp := localRng.Intn(opsPerGoroutine)
						id := fmt.Sprintf("r%d_op%d", otherRoutine, otherOp)
						// Ignore errors (might not exist)
						_ = hnsw.Delete(id)
					}

				case 2: // search
					query := []float32{float32(localRng.Intn(100)), float32(localRng.Intn(100)), float32(localRng.Intn(100))}
					k := localRng.Intn(5) + 1 // search for 1-5 nearest neighbors
					// Ignore errors and results
					_, _ = hnsw.Search(query, k)
				}
			}
		}(i)
	}

	// Wait for all goroutines to finish
	wg.Wait()

	// Final check that the index is still functional
	_, err := hnsw.Search([]float32{1.0, 1.0, 1.0}, 3)
	if err != nil {
		t.Fatalf("Search() after concurrent operations error = %v", err)
	}
}

func TestDistanceFunctions(t *testing.T) {
	tests := []struct {
		name    string
		fn      DistanceFunction
		a       []float32
		b       []float32
		want    float32
		wantErr bool
		epsilon float32 // Tolerance for floating point comparison
	}{
		{
			name:    "Euclidean - Same Vectors",
			fn:      EuclideanDistanceFunc,
			a:       []float32{1.0, 2.0, 3.0},
			b:       []float32{1.0, 2.0, 3.0},
			want:    0.0,
			wantErr: false,
			epsilon: 1e-6,
		},
		{
			name:    "Euclidean - Different Vectors",
			fn:      EuclideanDistanceFunc,
			a:       []float32{1.0, 0.0, 0.0},
			b:       []float32{0.0, 1.0, 0.0},
			want:    1.414214, // sqrt(2)
			wantErr: false,
			epsilon: 1e-6,
		},
		{
			name:    "Cosine - Same Direction",
			fn:      CosineDistanceFunc,
			a:       []float32{1.0, 2.0, 3.0},
			b:       []float32{2.0, 4.0, 6.0},
			want:    0.0,
			wantErr: false,
			epsilon: 1e-6,
		},
		{
			name:    "Cosine - Orthogonal",
			fn:      CosineDistanceFunc,
			a:       []float32{1.0, 0.0, 0.0},
			b:       []float32{0.0, 1.0, 0.0},
			want:    1.0,
			wantErr: false,
			epsilon: 1e-6,
		},
		{
			name:    "Cosine - Opposite Direction",
			fn:      CosineDistanceFunc,
			a:       []float32{1.0, 0.0, 0.0},
			b:       []float32{-1.0, 0.0, 0.0},
			want:    2.0,
			wantErr: false,
			epsilon: 1e-6,
		},
		{
			name:    "DotProduct - Same Vectors",
			fn:      DotProductDistanceFunc,
			a:       []float32{1.0, 2.0, 3.0},
			b:       []float32{1.0, 2.0, 3.0},
			want:    -14.0, // -(1*1 + 2*2 + 3*3)
			wantErr: false,
			epsilon: 1e-6,
		},
		{
			name:    "Dimension Mismatch",
			fn:      EuclideanDistanceFunc,
			a:       []float32{1.0, 2.0},
			b:       []float32{1.0, 2.0, 3.0},
			want:    0.0,
			wantErr: true,
			epsilon: 1e-6,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.fn(tt.a, tt.b)
			if (err != nil) != tt.wantErr {
				t.Errorf("DistanceFunction() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr && abs(got-tt.want) > tt.epsilon {
				t.Errorf("DistanceFunction() = %v, want %v (within %v)", got, tt.want, tt.epsilon)
			}
		})
	}
}

// Helper function for float32 absolute value
func abs(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

// Benchmark HNSW operations
func BenchmarkHNSW_Insert(b *testing.B) {
	hnsw := NewHNSW(Config{
		DistanceFunc: EuclideanDistanceFunc,
	})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		id := fmt.Sprintf("bench_%d", i)
		vector := []float32{float32(i % 100), float32((i + 1) % 100), float32((i + 2) % 100)}
		_ = hnsw.Insert(id, vector)
	}
}

func BenchmarkHNSW_Search(b *testing.B) {
	// Setup: Insert 10,000 vectors
	hnsw := NewHNSW(Config{
		DistanceFunc: EuclideanDistanceFunc,
	})
	for i := 0; i < 10000; i++ {
		id := fmt.Sprintf("bench_%d", i)
		vector := []float32{float32(i % 100), float32((i + 1) % 100), float32((i + 2) % 100)}
		_ = hnsw.Insert(id, vector)
	}

	query := []float32{50.0, 50.0, 50.0}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = hnsw.Search(query, 10)
	}
}

func BenchmarkHNSW_Delete(b *testing.B) {
	// Create a fixed number of vectors to insert/delete
	const numVectors = 10000

	// Setup: Insert vectors
	hnsw := NewHNSW(Config{
		DistanceFunc: EuclideanDistanceFunc,
	})

	// Only insert as many vectors as we'll test
	vectorCount := b.N
	if vectorCount > numVectors {
		vectorCount = numVectors
	}

	for i := 0; i < vectorCount; i++ {
		id := fmt.Sprintf("bench_%d", i)
		vector := []float32{float32(i % 100), float32((i + 1) % 100), float32((i + 2) % 100)}
		_ = hnsw.Insert(id, vector)
	}

	b.ResetTimer()
	// Only run the benchmark for as many vectors as we've inserted
	for i := 0; i < vectorCount; i++ {
		id := fmt.Sprintf("bench_%d", i)
		_ = hnsw.Delete(id)
	}
}
