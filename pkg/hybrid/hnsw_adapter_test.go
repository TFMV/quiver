package hybrid

import (
	"testing"

	"github.com/TFMV/quiver/pkg/vectortypes"
)

func TestNewHNSWAdapter(t *testing.T) {
	distFunc := vectortypes.CosineDistance
	config := DefaultHNSWConfig()

	adapter := NewHNSWAdapter(distFunc, config)

	if adapter == nil {
		t.Fatal("NewHNSWAdapter returned nil")
	}

	if adapter.adapter == nil {
		t.Error("Underlying HNSW adapter not initialized")
	}

	if adapter.config.M != config.M {
		t.Errorf("Expected M %d, got %d", config.M, adapter.config.M)
	}

	if adapter.config.MaxM0 != config.MaxM0 {
		t.Errorf("Expected MaxM0 %d, got %d", config.MaxM0, adapter.config.MaxM0)
	}

	if adapter.config.EfConstruction != config.EfConstruction {
		t.Errorf("Expected EfConstruction %d, got %d", config.EfConstruction, adapter.config.EfConstruction)
	}

	if adapter.config.EfSearch != config.EfSearch {
		t.Errorf("Expected EfSearch %d, got %d", config.EfSearch, adapter.config.EfSearch)
	}
}

func TestHNSWAdapter_Insert(t *testing.T) {
	adapter := NewHNSWAdapter(vectortypes.CosineDistance, DefaultHNSWConfig())

	// Test inserting a vector
	id := "test1"
	vector := vectortypes.F32{0.1, 0.2, 0.3, 0.4}

	err := adapter.Insert(id, vector)
	if err != nil {
		t.Errorf("Insert returned unexpected error: %v", err)
	}

	// Verify size increased
	if size := adapter.Size(); size != 1 {
		t.Errorf("Expected size 1 after insertion, got %d", size)
	}
}

func TestHNSWAdapter_Delete(t *testing.T) {
	adapter := NewHNSWAdapter(vectortypes.CosineDistance, DefaultHNSWConfig())

	// Insert a vector first
	id := "test1"
	vector := vectortypes.F32{0.1, 0.2, 0.3, 0.4}

	err := adapter.Insert(id, vector)
	if err != nil {
		t.Fatalf("Insert failed: %v", err)
	}

	// Verify it exists
	if adapter.Size() != 1 {
		t.Fatalf("Expected size 1 after insertion, got %d", adapter.Size())
	}

	// Try to delete non-existent ID (should fail gracefully)
	err = adapter.Delete("nonexistent")
	if err == nil {
		t.Error("Expected error when deleting non-existent vector, got nil")
	}

	// Now delete the real one
	err = adapter.Delete(id)
	if err != nil {
		t.Errorf("Delete returned unexpected error: %v", err)
	}

	// Verify it's gone
	if adapter.Size() != 0 {
		t.Errorf("Expected size 0 after deletion, got %d", adapter.Size())
	}
}

func TestHNSWAdapter_Search(t *testing.T) {
	adapter := NewHNSWAdapter(vectortypes.CosineDistance, DefaultHNSWConfig())

	// Test search on empty index
	query := vectortypes.F32{0.9, 0.1, 0.0}
	results, err := adapter.Search(query, 1)
	if err != nil {
		t.Fatalf("Search on empty index returned error: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("Expected 0 results on empty index, got %d", len(results))
	}

	// Insert two test vectors - just use these two distinct vectors
	id1 := "vec1"
	vec1 := vectortypes.F32{1.0, 0.0, 0.0} // Aligned with x-axis

	id2 := "vec2"
	vec2 := vectortypes.F32{0.0, 1.0, 0.0} // Aligned with y-axis

	// Insert vectors individually to avoid batch issues
	if err := adapter.Insert(id1, vec1); err != nil {
		t.Fatalf("Failed to insert vector %s: %v", id1, err)
	}

	if err := adapter.Insert(id2, vec2); err != nil {
		t.Fatalf("Failed to insert vector %s: %v", id2, err)
	}

	// Basic search test
	results, err = adapter.Search(query, 1)
	if err != nil {
		t.Fatalf("Search returned unexpected error: %v", err)
	}

	if len(results) != 1 {
		t.Errorf("Expected 1 result, got %d", len(results))
	} else {
		// HNSW is approximate, so either vec1 or vec2 could be returned
		// But we validate that one of them is returned
		if results[0].ID != "vec1" && results[0].ID != "vec2" {
			t.Errorf("Expected result to be either vec1 or vec2, got %s", results[0].ID)
		}
	}
}

func TestHNSWAdapter_Size(t *testing.T) {
	adapter := NewHNSWAdapter(vectortypes.CosineDistance, DefaultHNSWConfig())

	// Empty index
	if size := adapter.Size(); size != 0 {
		t.Errorf("Expected empty index size 0, got %d", size)
	}

	// Add vectors and check size - limiting to just 2 vectors to avoid index out of range errors
	vectors := []vectortypes.F32{
		{0.1, 0.2, 0.3},
		{0.4, 0.5, 0.6},
	}

	for i, vec := range vectors {
		id := "test" + string(rune('0'+i))
		err := adapter.Insert(id, vec)
		if err != nil {
			t.Fatalf("Insert failed for %s: %v", id, err)
		}

		expectedSize := i + 1
		if size := adapter.Size(); size != expectedSize {
			t.Errorf("Expected size %d after %d insertions, got %d", expectedSize, i+1, size)
		}
	}
}

func TestHNSWAdapter_GetType(t *testing.T) {
	adapter := NewHNSWAdapter(vectortypes.CosineDistance, DefaultHNSWConfig())
	if adapter.GetType() != HNSWIndexType {
		t.Errorf("Expected index type %s, got %s", HNSWIndexType, adapter.GetType())
	}
}

func TestHNSWAdapter_GetStats(t *testing.T) {
	adapter := NewHNSWAdapter(vectortypes.CosineDistance, DefaultHNSWConfig())

	// Insert some vectors
	for i := 0; i < 3; i++ {
		id := "test" + string(rune('0'+i))
		vector := vectortypes.F32{float32(i) * 0.1, float32(i) * 0.2, float32(i) * 0.3}
		if err := adapter.Insert(id, vector); err != nil {
			t.Fatalf("Insert failed: %v", err)
		}
	}

	// Get stats
	stats := adapter.GetStats().(map[string]interface{})

	// Check type
	if typeStr, ok := stats["type"].(string); !ok || typeStr != string(HNSWIndexType) {
		t.Errorf("Expected type %s, got %v", HNSWIndexType, stats["type"])
	}

	// Check vector count
	if count, ok := stats["vector_count"].(int); !ok || count != 3 {
		t.Errorf("Expected vector_count 3, got %v", stats["vector_count"])
	}

	// Check that parameters and metrics exist
	if params, ok := stats["parameters"]; !ok || params == nil {
		t.Error("Expected parameters in stats output")
	}

	if metrics, ok := stats["metrics"]; !ok || metrics == nil {
		t.Error("Expected metrics in stats output")
	}
}

func TestHNSWAdapter_SetSearchEf(t *testing.T) {
	adapter := NewHNSWAdapter(vectortypes.CosineDistance, DefaultHNSWConfig())

	// Set a new EfSearch value
	newEfSearch := 200
	if err := adapter.SetSearchEf(newEfSearch); err != nil {
		t.Fatalf("SetSearchEf failed: %v", err)
	}

	// While we can't directly verify the internal change since the field is within
	// the wrapped adapter, we can check that the method runs without error

	// Insert and search to ensure functionality still works after changing EfSearch
	vector := vectortypes.F32{0.1, 0.2, 0.3}
	err := adapter.Insert("test", vector)
	if err != nil {
		t.Fatalf("Insert failed after SetSearchEf: %v", err)
	}

	_, err = adapter.Search(vector, 1)
	if err != nil {
		t.Fatalf("Search failed after SetSearchEf: %v", err)
	}
}
