package hybrid

import (
	"reflect"
	"sort"
	"testing"

	"github.com/TFMV/quiver/pkg/types"
	"github.com/TFMV/quiver/pkg/vectortypes"
)

func TestNewExactIndex(t *testing.T) {
	distFunc := vectortypes.CosineDistance
	idx := NewExactIndex(distFunc)

	if idx.distFunc == nil {
		t.Error("Distance function not set correctly")
	}

	if idx.vectors == nil {
		t.Error("Vectors map not initialized")
	}

	if len(idx.vectors) != 0 {
		t.Errorf("Expected empty vectors map, got %d items", len(idx.vectors))
	}
}

func TestExactIndex_Insert(t *testing.T) {
	idx := NewExactIndex(vectortypes.CosineDistance)

	// Test inserting a vector
	id := "test1"
	vector := vectortypes.F32{0.1, 0.2, 0.3, 0.4}

	err := idx.Insert(id, vector)
	if err != nil {
		t.Errorf("Insert returned unexpected error: %v", err)
	}

	// Verify vector was inserted
	if len(idx.vectors) != 1 {
		t.Errorf("Expected 1 vector in index, got %d", len(idx.vectors))
	}

	// Verify the stored vector is a copy, not the original
	storedVector, exists := idx.vectors[id]
	if !exists {
		t.Fatalf("Vector with id %s not found in index", id)
	}

	if !reflect.DeepEqual(storedVector, vector) {
		t.Errorf("Stored vector %v does not match original %v", storedVector, vector)
	}

	// Modify the original vector and verify the stored one is unchanged
	vector[0] = 0.9
	if storedVector[0] == vector[0] {
		t.Error("Stored vector should be a copy, not a reference to the original")
	}
}

func TestExactIndex_Delete(t *testing.T) {
	idx := NewExactIndex(vectortypes.CosineDistance)

	// Insert a vector first
	id := "test1"
	vector := vectortypes.F32{0.1, 0.2, 0.3, 0.4}
	err := idx.Insert(id, vector)
	if err != nil {
		t.Fatalf("Insert failed: %v", err)
	}

	// Verify it exists
	if len(idx.vectors) != 1 {
		t.Fatalf("Expected 1 vector in index, got %d", len(idx.vectors))
	}

	// Delete it
	err = idx.Delete(id)
	if err != nil {
		t.Errorf("Delete returned unexpected error: %v", err)
	}

	// Verify it's gone
	if len(idx.vectors) != 0 {
		t.Errorf("Expected 0 vectors after deletion, got %d", len(idx.vectors))
	}

	// Deleting a non-existent ID should not error
	err = idx.Delete("nonexistent")
	if err != nil {
		t.Errorf("Delete of non-existent ID returned error: %v", err)
	}
}

func TestExactIndex_Search(t *testing.T) {
	idx := NewExactIndex(vectortypes.CosineDistance)

	// Insert some test vectors
	testVectors := map[string]vectortypes.F32{
		"vec1": {1.0, 0.0, 0.0}, // Aligned with x-axis
		"vec2": {0.0, 1.0, 0.0}, // Aligned with y-axis
		"vec3": {0.0, 0.0, 1.0}, // Aligned with z-axis
	}

	for id, vec := range testVectors {
		if err := idx.Insert(id, vec); err != nil {
			t.Fatalf("Failed to insert vector %s: %v", id, err)
		}
	}

	// Test cases
	tests := []struct {
		name       string
		query      vectortypes.F32
		k          int
		wantIDs    []string
		exactOrder bool
	}{
		{
			name:       "Query similar to vec1",
			query:      vectortypes.F32{0.9, 0.1, 0.0},
			k:          2,
			wantIDs:    []string{"vec1", "vec2"},
			exactOrder: true,
		},
		{
			name:       "Query similar to vec2",
			query:      vectortypes.F32{0.1, 0.9, 0.0},
			k:          2,
			wantIDs:    []string{"vec2", "vec1"},
			exactOrder: true,
		},
		{
			name:       "Query similar to vec3",
			query:      vectortypes.F32{0.1, 0.1, 0.9},
			k:          1,
			wantIDs:    []string{"vec3"},
			exactOrder: true,
		},
		{
			name:       "Query for all vectors",
			query:      vectortypes.F32{0.5, 0.5, 0.5},
			k:          3,
			wantIDs:    []string{"vec1", "vec2", "vec3"},
			exactOrder: false, // Don't care about order since distances may be similar
		},
		{
			name:       "K larger than number of vectors",
			query:      vectortypes.F32{1.0, 0.0, 0.0},
			k:          10,
			wantIDs:    []string{"vec1", "vec2", "vec3"},
			exactOrder: false, // Changed to false as vec2 and vec3 have equal distance
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			results, err := idx.Search(tt.query, tt.k)
			if err != nil {
				t.Fatalf("Search returned unexpected error: %v", err)
			}

			// Check result count
			expectedCount := tt.k
			if expectedCount > len(testVectors) {
				expectedCount = len(testVectors)
			}
			if len(results) != expectedCount {
				t.Errorf("Expected %d results, got %d", expectedCount, len(results))
			}

			// If exact order matters, check each position
			if tt.exactOrder {
				for i, wantID := range tt.wantIDs {
					if i < len(results) && results[i].ID != wantID {
						t.Errorf("Result at position %d: got ID %s, want ID %s",
							i, results[i].ID, wantID)
					}
				}
			} else {
				// Otherwise just check that all expected IDs are present
				foundIDs := make(map[string]bool)
				for _, result := range results {
					foundIDs[result.ID] = true
				}

				for _, wantID := range tt.wantIDs {
					if !foundIDs[wantID] {
						t.Errorf("Expected ID %s not found in results", wantID)
					}
				}
			}

			// Check that distances are sorted
			for i := 0; i < len(results)-1; i++ {
				if results[i].Distance > results[i+1].Distance {
					t.Errorf("Results not sorted by distance: %f > %f",
						results[i].Distance, results[i+1].Distance)
				}
			}
		})
	}
}

func TestExactIndex_Size(t *testing.T) {
	idx := NewExactIndex(vectortypes.CosineDistance)

	// Empty index should have size 0
	if size := idx.Size(); size != 0 {
		t.Errorf("Expected size 0 for empty index, got %d", size)
	}

	// Insert a vector
	err := idx.Insert("test1", vectortypes.F32{0.1, 0.2, 0.3})
	if err != nil {
		t.Fatalf("Insert failed: %v", err)
	}

	// Size should be 1
	if size := idx.Size(); size != 1 {
		t.Errorf("Expected size 1 after insertion, got %d", size)
	}

	// Insert another vector
	err = idx.Insert("test2", vectortypes.F32{0.4, 0.5, 0.6})
	if err != nil {
		t.Fatalf("Insert failed: %v", err)
	}

	// Size should be 2
	if size := idx.Size(); size != 2 {
		t.Errorf("Expected size 2 after second insertion, got %d", size)
	}

	// Delete a vector
	err = idx.Delete("test1")
	if err != nil {
		t.Fatalf("Delete failed: %v", err)
	}

	// Size should be 1 again
	if size := idx.Size(); size != 1 {
		t.Errorf("Expected size 1 after deletion, got %d", size)
	}
}

func TestExactIndex_GetType(t *testing.T) {
	idx := NewExactIndex(vectortypes.CosineDistance)
	if idx.GetType() != ExactIndexType {
		t.Errorf("Expected index type %s, got %s", ExactIndexType, idx.GetType())
	}
}

func TestExactIndex_GetStats(t *testing.T) {
	idx := NewExactIndex(vectortypes.CosineDistance)

	// Insert some vectors
	for i := 0; i < 5; i++ {
		id := "test" + string(rune('0'+i))
		vector := vectortypes.F32{float32(i) * 0.1, float32(i) * 0.2, float32(i) * 0.3}
		if err := idx.Insert(id, vector); err != nil {
			t.Fatalf("Insert failed: %v", err)
		}
	}

	// Get stats
	stats := idx.GetStats().(map[string]interface{})

	// Check type
	if typeStr, ok := stats["type"].(string); !ok || typeStr != string(ExactIndexType) {
		t.Errorf("Expected type %s, got %v", ExactIndexType, stats["type"])
	}

	// Check vector count
	if count, ok := stats["vector_count"].(int); !ok || count != 5 {
		t.Errorf("Expected vector_count 5, got %v", stats["vector_count"])
	}
}

func TestResultHeap(t *testing.T) {
	// Create a result heap for testing
	heap := &resultHeap{}

	// Add results in any order
	*heap = append(*heap, types.BasicSearchResult{ID: "vec1", Distance: 0.5})
	*heap = append(*heap, types.BasicSearchResult{ID: "vec2", Distance: 0.2})
	*heap = append(*heap, types.BasicSearchResult{ID: "vec3", Distance: 0.8})
	*heap = append(*heap, types.BasicSearchResult{ID: "vec4", Distance: 0.3})

	// Initialize the heap
	sort.Sort(heap)

	// Pop results off the heap
	var popped []types.BasicSearchResult
	for heap.Len() > 0 {
		popped = append(popped, (*heap)[0])
		*heap = (*heap)[1:]
		if heap.Len() > 0 {
			sort.Sort(heap)
		}
	}

	// Results should be in ascending order of distance
	expected := []string{"vec2", "vec4", "vec1", "vec3"}
	for i, id := range expected {
		if popped[i].ID != id {
			t.Errorf("Expected %s at position %d, got %s", id, i, popped[i].ID)
		}
	}
}

func TestExactIndex_DimensionMismatch(t *testing.T) {
	idx := NewExactIndex(vectortypes.CosineDistance)
	if err := idx.Insert("vec1", vectortypes.F32{0.1, 0.2}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := idx.Insert("vec2", vectortypes.F32{0.3}); err == nil {
		t.Errorf("expected dimension mismatch error")
	}
}

func TestExactIndex_SearchDimensionMismatch(t *testing.T) {
	idx := NewExactIndex(vectortypes.CosineDistance)
	_ = idx.Insert("vec1", vectortypes.F32{0.1, 0.2})
	if _, err := idx.Search(vectortypes.F32{0.3}, 1); err == nil {
		t.Errorf("expected dimension mismatch error")
	}
}
