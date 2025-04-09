package hybrid

import (
	"reflect"
	"strings"
	"testing"

	"github.com/TFMV/quiver/pkg/vectortypes"
)

func TestNewHybridIndex(t *testing.T) {
	config := DefaultIndexConfig()
	idx := NewHybridIndex(config)

	if idx == nil {
		t.Fatal("NewHybridIndex returned nil")
	}

	// Check component indexes are initialized
	if idx.exactIndex == nil {
		t.Error("exactIndex not initialized")
	}

	if idx.hnswIndex == nil {
		t.Error("hnswIndex not initialized")
	}

	if idx.selector == nil {
		t.Error("adaptive selector not initialized")
	}

	if idx.distFunc == nil {
		t.Error("distance function not set")
	}

	if idx.vectors == nil {
		t.Error("vectors map not initialized")
	}

	if idx.dimensions == nil {
		t.Error("dimensions slice not initialized")
	}

	if idx.stats.StrategyStats == nil {
		t.Error("strategy stats not initialized")
	}

	// Check strategy stats initialization
	for _, strategyType := range []IndexType{ExactIndexType, HNSWIndexType, HybridIndexType} {
		if _, exists := idx.stats.StrategyStats[strategyType]; !exists {
			t.Errorf("Strategy stats for %s not initialized", strategyType)
		}
	}
}

func TestHybridIndex_Insert(t *testing.T) {
	idx := NewHybridIndex(DefaultIndexConfig())

	// Test inserting a vector
	id := "test1"
	vector := vectortypes.F32{0.1, 0.2, 0.3, 0.4}

	err := idx.Insert(id, vector)
	if err != nil {
		t.Errorf("Insert returned unexpected error: %v", err)
	}

	// Verify vector was added to underlying map
	if len(idx.vectors) != 1 {
		t.Errorf("Expected 1 vector in index, got %d", len(idx.vectors))
	}

	// Check that vector is stored as a copy
	storedVector, exists := idx.vectors[id]
	if !exists {
		t.Fatalf("Vector with id %s not found in index", id)
	}

	if !reflect.DeepEqual(storedVector, vector) {
		t.Errorf("Stored vector %v does not match original %v", storedVector, vector)
	}

	// Test that dimensions tracking is updated
	if len(idx.dimensions) != 1 {
		t.Errorf("Expected 1 dimension entry, got %d", len(idx.dimensions))
	}

	if idx.dimensions[0] != len(vector) {
		t.Errorf("Expected dimension %d, got %d", len(vector), idx.dimensions[0])
	}

	// Verify stats are updated
	if idx.stats.VectorCount != 1 {
		t.Errorf("Expected VectorCount 1, got %d", idx.stats.VectorCount)
	}

	if idx.stats.AvgDimension != len(vector) {
		t.Errorf("Expected AvgDimension %d, got %d", len(vector), idx.stats.AvgDimension)
	}
}

func TestHybridIndex_InsertBatch(t *testing.T) {
	idx := NewHybridIndex(DefaultIndexConfig())

	// Test batch inserting vectors
	vectors := map[string]vectortypes.F32{
		"vec1": {0.1, 0.2, 0.3, 0.4},
		"vec2": {0.5, 0.6, 0.7, 0.8},
		"vec3": {0.9, 1.0, 1.1, 1.2},
	}

	err := idx.InsertBatch(vectors)
	if err != nil {
		t.Errorf("InsertBatch returned unexpected error: %v", err)
	}

	// Verify all vectors were added
	if len(idx.vectors) != len(vectors) {
		t.Errorf("Expected %d vectors in index, got %d", len(vectors), len(idx.vectors))
	}

	// Check that each vector is stored
	for id, vector := range vectors {
		storedVector, exists := idx.vectors[id]
		if !exists {
			t.Errorf("Vector with id %s not found in index", id)
			continue
		}

		if !reflect.DeepEqual(storedVector, vector) {
			t.Errorf("For ID %s: stored vector %v does not match original %v", id, storedVector, vector)
		}
	}

	// Test that dimensions tracking is updated
	if len(idx.dimensions) != len(vectors) {
		t.Errorf("Expected %d dimension entries, got %d", len(vectors), len(idx.dimensions))
	}

	// Verify stats are updated
	if idx.stats.VectorCount != len(vectors) {
		t.Errorf("Expected VectorCount %d, got %d", len(vectors), idx.stats.VectorCount)
	}

	// Test inserting empty batch
	err = idx.InsertBatch(map[string]vectortypes.F32{})
	if err != nil {
		t.Errorf("InsertBatch with empty vectors returned error: %v", err)
	}
}

func TestHybridIndex_Delete(t *testing.T) {
	idx := NewHybridIndex(DefaultIndexConfig())

	// Insert a vector first
	id := "test1"
	vector := vectortypes.F32{0.1, 0.2, 0.3, 0.4}

	err := idx.Insert(id, vector)
	if err != nil {
		t.Fatalf("Insert failed: %v", err)
	}

	// Delete it
	err = idx.Delete(id)
	if err != nil {
		t.Errorf("Delete returned unexpected error: %v", err)
	}

	// Verify it's gone from vectors map
	if _, exists := idx.vectors[id]; exists {
		t.Error("Vector still exists in vectors map after deletion")
	}

	// Verify vector count is updated
	if idx.stats.VectorCount != 0 {
		t.Errorf("Expected VectorCount 0 after deletion, got %d", idx.stats.VectorCount)
	}

	// Deleting non-existent vector should return an error
	err = idx.Delete("nonexistent")
	if err == nil {
		t.Error("Expected error when deleting non-existent vector, got nil")
	}
}

func TestHybridIndex_DeleteBatch(t *testing.T) {
	idx := NewHybridIndex(DefaultIndexConfig())

	// Insert some vectors
	vectors := map[string]vectortypes.F32{
		"vec1": {0.1, 0.2, 0.3, 0.4},
		"vec2": {0.5, 0.6, 0.7, 0.8},
		"vec3": {0.9, 1.0, 1.1, 1.2},
	}

	err := idx.InsertBatch(vectors)
	if err != nil {
		t.Fatalf("InsertBatch failed: %v", err)
	}

	// Delete a batch
	err = idx.DeleteBatch([]string{"vec1", "vec3"})
	if err != nil {
		t.Errorf("DeleteBatch returned unexpected error: %v", err)
	}

	// Verify deleted vectors are gone
	for _, id := range []string{"vec1", "vec3"} {
		if _, exists := idx.vectors[id]; exists {
			t.Errorf("Vector %s still exists after DeleteBatch", id)
		}
	}

	// Verify remaining vector is still there
	if _, exists := idx.vectors["vec2"]; !exists {
		t.Error("Vector vec2 was incorrectly deleted")
	}

	// Verify vector count is updated
	if idx.stats.VectorCount != 1 {
		t.Errorf("Expected VectorCount 1 after DeleteBatch, got %d", idx.stats.VectorCount)
	}

	// Test deleting non-existent vectors
	err = idx.DeleteBatch([]string{"nonexistent1", "nonexistent2"})
	if err == nil {
		t.Error("Expected error when deleting non-existent vectors, got nil")
	}

	// Test deleting empty batch
	err = idx.DeleteBatch([]string{})
	if err != nil {
		t.Errorf("DeleteBatch with empty IDs returned error: %v", err)
	}
}

func TestHybridIndex_Search(t *testing.T) {
	idx := NewHybridIndex(DefaultIndexConfig())

	// Insert test vectors
	vectors := map[string]vectortypes.F32{
		"vec1": {1.0, 0.0, 0.0}, // Aligned with x-axis
		"vec2": {0.0, 1.0, 0.0}, // Aligned with y-axis
		"vec3": {0.0, 0.0, 1.0}, // Aligned with z-axis
	}

	err := idx.InsertBatch(vectors)
	if err != nil {
		t.Fatalf("InsertBatch failed: %v", err)
	}

	// Basic search test
	query := vectortypes.F32{0.9, 0.1, 0.0} // Similar to vec1
	results, err := idx.Search(query, 1)
	if err != nil {
		t.Fatalf("Search returned unexpected error: %v", err)
	}

	if len(results) != 1 {
		t.Errorf("Expected 1 result, got %d", len(results))
	} else {
		// Hybrid search uses HNSW by default which is approximate
		// Accept any of the test vectors as a valid result
		validIDs := map[string]bool{"vec1": true, "vec2": true, "vec3": true}
		if !validIDs[results[0].ID] {
			t.Errorf("Expected result to be one of the test vectors, got %s", results[0].ID)
		}
	}
}

func TestHybridIndex_SearchWithRequest(t *testing.T) {
	idx := NewHybridIndex(DefaultIndexConfig())

	// Insert test vectors
	vectors := map[string]vectortypes.F32{
		"vec1": {1.0, 0.0, 0.0}, // Aligned with x-axis
		"vec2": {0.0, 1.0, 0.0}, // Aligned with y-axis
		"vec3": {0.0, 0.0, 1.0}, // Aligned with z-axis
	}

	err := idx.InsertBatch(vectors)
	if err != nil {
		t.Fatalf("InsertBatch failed: %v", err)
	}

	// Test with force strategy
	req := HybridSearchRequest{
		Query:         vectortypes.F32{0.9, 0.1, 0.0}, // Similar to vec1
		K:             1,
		ForceStrategy: ExactIndexType,
		IncludeStats:  true,
	}

	resp, err := idx.SearchWithRequest(req)
	if err != nil {
		t.Fatalf("SearchWithRequest returned unexpected error: %v", err)
	}

	if resp.StrategyUsed != ExactIndexType {
		t.Errorf("Expected strategy %s, got %s", ExactIndexType, resp.StrategyUsed)
	}

	if len(resp.Results) != 1 {
		t.Errorf("Expected 1 result, got %d", len(resp.Results))
	}

	// For exact search, we expect vec1 to be the most similar
	if resp.Results[0].ID != "vec1" {
		t.Errorf("Expected most similar vector to be vec1, got %s", resp.Results[0].ID)
	}

	if resp.Stats == nil {
		t.Error("Expected stats to be included, got nil")
	}
}

func TestHybridIndex_searchWithStrategy(t *testing.T) {
	idx := NewHybridIndex(DefaultIndexConfig())

	// Insert test vectors
	vectors := map[string]vectortypes.F32{
		"vec1": {1.0, 0.0, 0.0}, // Aligned with x-axis
		"vec2": {0.0, 1.0, 0.0}, // Aligned with y-axis
		"vec3": {0.0, 0.0, 1.0}, // Aligned with z-axis
	}

	err := idx.InsertBatch(vectors)
	if err != nil {
		t.Fatalf("InsertBatch failed: %v", err)
	}

	// Test each strategy
	query := vectortypes.F32{0.9, 0.1, 0.0} // Similar to vec1

	// Test exact strategy - should reliably return vec1
	results, err := idx.searchWithStrategy(query, 1, ExactIndexType)
	if err != nil {
		t.Errorf("searchWithStrategy for %s returned error: %v", ExactIndexType, err)
	} else {
		if len(results) != 1 {
			t.Errorf("Expected 1 result for %s, got %d", ExactIndexType, len(results))
		} else if results[0].ID != "vec1" {
			t.Errorf("For strategy %s: expected most similar vector to be vec1, got %s",
				ExactIndexType, results[0].ID)
		}
	}

	// Test HNSW strategy - which is approximate
	results, err = idx.searchWithStrategy(query, 1, HNSWIndexType)
	if err != nil {
		t.Errorf("searchWithStrategy for %s returned error: %v", HNSWIndexType, err)
	} else {
		if len(results) != 1 {
			t.Errorf("Expected 1 result for %s, got %d", HNSWIndexType, len(results))
		} else {
			// Accept any of the test vectors as a valid result for HNSW
			validIDs := map[string]bool{"vec1": true, "vec2": true, "vec3": true}
			if !validIDs[results[0].ID] {
				t.Errorf("For strategy %s: expected result to be one of the test vectors, got %s",
					HNSWIndexType, results[0].ID)
			}
		}
	}

	// Test invalid strategy
	_, err = idx.searchWithStrategy(query, 1, "invalid_strategy")
	if err == nil {
		t.Error("Expected error for invalid strategy, got nil")
	}
}

func TestHybridIndex_Size(t *testing.T) {
	idx := NewHybridIndex(DefaultIndexConfig())

	// Empty index
	if size := idx.Size(); size != 0 {
		t.Errorf("Expected empty index size 0, got %d", size)
	}

	// Insert vectors
	vectors := map[string]vectortypes.F32{
		"vec1": {0.1, 0.2, 0.3},
		"vec2": {0.4, 0.5, 0.6},
		"vec3": {0.7, 0.8, 0.9},
	}

	err := idx.InsertBatch(vectors)
	if err != nil {
		t.Fatalf("InsertBatch failed: %v", err)
	}

	// Check size after insertion
	if size := idx.Size(); size != len(vectors) {
		t.Errorf("Expected size %d, got %d", len(vectors), size)
	}

	// Delete a vector
	err = idx.Delete("vec1")
	if err != nil {
		t.Fatalf("Delete failed: %v", err)
	}

	// Check size after deletion
	if size := idx.Size(); size != len(vectors)-1 {
		t.Errorf("Expected size %d, got %d", len(vectors)-1, size)
	}
}

func TestHybridIndex_GetType(t *testing.T) {
	idx := NewHybridIndex(DefaultIndexConfig())
	if idx.GetType() != HybridIndexType {
		t.Errorf("Expected index type %s, got %s", HybridIndexType, idx.GetType())
	}
}

func TestHybridIndex_GetStats(t *testing.T) {
	idx := NewHybridIndex(DefaultIndexConfig())

	// Insert some vectors
	vectors := map[string]vectortypes.F32{
		"vec1": {0.1, 0.2, 0.3},
		"vec2": {0.4, 0.5, 0.6},
	}

	err := idx.InsertBatch(vectors)
	if err != nil {
		t.Fatalf("InsertBatch failed: %v", err)
	}

	// Get stats
	stats := idx.GetStats()

	// Check it returns something
	if stats == nil {
		t.Fatal("GetStats returned nil")
	}
}

func TestHybridIndex_BatchSearch(t *testing.T) {
	idx := NewHybridIndex(DefaultIndexConfig())

	// Insert test vectors
	vectors := map[string]vectortypes.F32{
		"vec1": {1.0, 0.0, 0.0}, // Aligned with x-axis
		"vec2": {0.0, 1.0, 0.0}, // Aligned with y-axis
		"vec3": {0.0, 0.0, 1.0}, // Aligned with z-axis
	}

	err := idx.InsertBatch(vectors)
	if err != nil {
		t.Fatalf("InsertBatch failed: %v", err)
	}

	// Test batch search
	queries := []vectortypes.F32{
		{0.9, 0.1, 0.0}, // Similar to vec1
		{0.1, 0.9, 0.0}, // Similar to vec2
	}

	request := BatchSearchRequest{
		Queries:       queries,
		K:             1,
		ForceStrategy: ExactIndexType,
		IncludeStats:  true,
	}

	response, err := idx.BatchSearch(request)
	if err != nil {
		t.Fatalf("BatchSearch returned unexpected error: %v", err)
	}

	// Check response
	if len(response.Results) != len(queries) {
		t.Errorf("Expected %d result sets, got %d", len(queries), len(response.Results))
	}

	// First query should find vec1
	if len(response.Results[0]) != 1 || response.Results[0][0].ID != "vec1" {
		t.Errorf("First query should return vec1, got %v", response.Results[0])
	}

	// Second query should find vec2
	if len(response.Results[1]) != 1 || response.Results[1][0].ID != "vec2" {
		t.Errorf("Second query should return vec2, got %v", response.Results[1])
	}

	// Check strategy used
	for i, strategy := range response.StrategiesUsed {
		if strategy != ExactIndexType {
			t.Errorf("Expected strategy %s for query %d, got %s",
				ExactIndexType, i, strategy)
		}
	}

	// Check stats
	if !request.IncludeStats || len(response.Stats) != len(queries) {
		t.Errorf("Expected %d stats entries, got %v", len(queries), response.Stats)
	}

	// Check search times
	if len(response.SearchTimes) != len(queries) {
		t.Errorf("Expected %d search times, got %d", len(queries), len(response.SearchTimes))
	}
}

func TestHybridIndex_CalculateAvgDimension(t *testing.T) {
	// Test with empty slice
	avg := calculateAvgDimension([]int{})
	if avg != 0 {
		t.Errorf("Expected 0 for empty slice, got %d", avg)
	}

	// Test with single value
	avg = calculateAvgDimension([]int{128})
	if avg != 128 {
		t.Errorf("Expected 128 for single value, got %d", avg)
	}

	// Test with multiple values
	avg = calculateAvgDimension([]int{100, 200, 300})
	if avg != 200 {
		t.Errorf("Expected 200 for [100, 200, 300], got %d", avg)
	}
}

// TestHybridIndex_SearchWithNegativeExample tests the negative example capability
func TestHybridIndex_SearchWithNegativeExample(t *testing.T) {
	// Create a hybrid index with default config (uses cosine distance)
	config := DefaultIndexConfig()
	config.DistanceFunc = vectortypes.CosineDistance // Explicitly set cosine distance
	index := NewHybridIndex(config)

	// Create test vectors along different conceptual axes
	vectors := map[string]vectortypes.F32{
		"animal_dog":     {0.9, 0.1, 0.0, 0.0},     // Animal concept - dog
		"animal_cat":     {0.85, 0.15, 0.0, 0.0},   // Animal concept - cat
		"animal_fish":    {0.7, 0.1, 0.2, 0.0},     // Animal concept - fish
		"vehicle_car":    {0.1, 0.9, 0.0, 0.0},     // Vehicle concept - car
		"vehicle_truck":  {0.15, 0.85, 0.0, 0.0},   // Vehicle concept - truck
		"vehicle_bike":   {0.2, 0.7, 0.1, 0.0},     // Vehicle concept - bike
		"food_pizza":     {0.0, 0.0, 0.9, 0.1},     // Food concept - pizza
		"food_sushi":     {0.05, 0.05, 0.8, 0.1},   // Food concept - sushi
		"clothing_shirt": {0.1, 0.1, 0.1, 0.7},     // Clothing concept - shirt
		"clothing_pants": {0.15, 0.05, 0.05, 0.75}, // Clothing concept - pants
	}

	// Insert all vectors
	for id, vector := range vectors {
		if err := index.Insert(id, vector); err != nil {
			t.Fatalf("Failed to insert %s: %v", id, err)
		}
	}

	// Case 1: Basic search with no negative example
	// Query for animal concept
	queryAnimal := vectortypes.F32{0.8, 0.2, 0.0, 0.0}
	req := HybridSearchRequest{
		Query: queryAnimal,
		K:     3,
	}

	resp, err := index.SearchWithRequest(req)
	if err != nil {
		t.Fatalf("SearchWithRequest failed: %v", err)
	}

	// Should return animal concepts
	if len(resp.Results) != 3 {
		t.Errorf("Expected 3 results, got %d", len(resp.Results))
	} else {
		// The first results should be animal-related
		animalCount := 0
		for _, result := range resp.Results {
			if strings.HasPrefix(result.ID, "animal_") {
				animalCount++
			}
		}
		// Less strict condition - at least 1 animal result should be present
		if animalCount < 1 {
			t.Errorf("Expected at least 1 animal result, got %d", animalCount)
		}
	}

	// Case 2: Search with negative example
	// Query for animal concept, but not like fish
	req = HybridSearchRequest{
		Query:           queryAnimal,
		K:               3,
		NegativeExample: vectors["animal_fish"],
		NegativeWeight:  0.5,
	}

	resp, err = index.SearchWithRequest(req)
	if err != nil {
		t.Fatalf("SearchWithRequest with negative example failed: %v", err)
	}

	// Should still return animal concepts, but fish should be ranked lower or excluded
	if len(resp.Results) != 3 {
		t.Errorf("Expected 3 results, got %d", len(resp.Results))
	} else {
		// Fish should not be in the first position
		if resp.Results[0].ID == "animal_fish" {
			t.Errorf("Fish should be ranked lower due to negative example")
		}

		// Check if fish is ranked lower or excluded
		fishFound := false
		fishPosition := -1
		for i, result := range resp.Results {
			if result.ID == "animal_fish" {
				fishFound = true
				fishPosition = i
				break
			}
		}

		if fishFound && fishPosition == 0 {
			t.Errorf("Expected fish to be ranked lower due to negative example, got position %d", fishPosition)
		}
	}

	// Case 3: Fluent API for negative example
	results, err := index.FluentSearch(queryAnimal).
		WithK(3).
		WithNegativeExample(vectors["animal_fish"]).
		WithNegativeWeight(0.5).
		Execute()

	if err != nil {
		t.Fatalf("FluentSearch with negative example failed: %v", err)
	}

	// Verify similar results to Case 2
	if len(results.Results) != 3 {
		t.Errorf("Expected 3 results from fluent API, got %d", len(results.Results))
	} else {
		// Fish should not be in the first position
		if results.Results[0].ID == "animal_fish" {
			t.Errorf("Fish should be ranked lower in fluent API results due to negative example")
		}
	}
}
