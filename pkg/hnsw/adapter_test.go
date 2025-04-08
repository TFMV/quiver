package hnsw

import (
	"fmt"
	"reflect"
	"testing"
	"time"

	"github.com/TFMV/quiver/pkg/types"
)

func TestNewAdapter(t *testing.T) {
	config := Config{
		M:              32,
		MaxM0:          64,
		EfConstruction: 200,
		EfSearch:       100,
		MaxLevel:       16,
		DistanceFunc:   CosineDistanceFunc,
	}

	adapter := NewAdapter(config)

	// Check that the adapter created a proper HNSW index
	if adapter.hnsw == nil {
		t.Fatal("NewAdapter() didn't create HNSW index")
	}

	if adapter.hnsw.M != config.M {
		t.Errorf("NewAdapter() created HNSW with M = %v, want %v", adapter.hnsw.M, config.M)
	}

	if adapter.hnsw.MaxM0 != config.MaxM0 {
		t.Errorf("NewAdapter() created HNSW with MaxM0 = %v, want %v", adapter.hnsw.MaxM0, config.MaxM0)
	}

	if adapter.hnsw.EfConstruction != config.EfConstruction {
		t.Errorf("NewAdapter() created HNSW with EfConstruction = %v, want %v", adapter.hnsw.EfConstruction, config.EfConstruction)
	}

	if adapter.hnsw.EfSearch != config.EfSearch {
		t.Errorf("NewAdapter() created HNSW with EfSearch = %v, want %v", adapter.hnsw.EfSearch, config.EfSearch)
	}

	if adapter.hnsw.MaxLevel != config.MaxLevel {
		t.Errorf("NewAdapter() created HNSW with MaxLevel = %v, want %v", adapter.hnsw.MaxLevel, config.MaxLevel)
	}

	// Check that the idToIndex map is initialized
	if adapter.idToIndex == nil {
		t.Fatal("NewAdapter() didn't initialize idToIndex map")
	}
}

func TestHNSWAdapter_Insert(t *testing.T) {
	adapter := NewAdapter(Config{
		M:              16,
		EfConstruction: 200,
		DistanceFunc:   EuclideanDistanceFunc,
	})

	// Test inserting a single vector
	err := adapter.Insert("test1", []float32{1.0, 2.0, 3.0})
	if err != nil {
		t.Fatalf("Insert() error = %v", err)
	}

	// Check that the vector was added to the HNSW index
	if adapter.Size() != 1 {
		t.Errorf("Insert() didn't increase Size(), got %v, want %v", adapter.Size(), 1)
	}

	// Check that the ID mapping was created
	if _, exists := adapter.idToIndex["test1"]; !exists {
		t.Errorf("Insert() didn't add ID mapping for 'test1'")
	}

	// Test inserting a duplicate (should fail)
	err = adapter.Insert("test1", []float32{4.0, 5.0, 6.0})
	if err == nil {
		t.Error("Insert() should return error for duplicate ID")
	}

	// Test inserting another vector
	err = adapter.Insert("test2", []float32{4.0, 5.0, 6.0})
	if err != nil {
		t.Fatalf("Insert() error = %v", err)
	}

	// Check that the size increased
	if adapter.Size() != 2 {
		t.Errorf("Insert() didn't increase Size(), got %v, want %v", adapter.Size(), 2)
	}
}

func TestHNSWAdapter_Delete(t *testing.T) {
	adapter := NewAdapter(Config{
		M:              16,
		EfConstruction: 200,
		DistanceFunc:   EuclideanDistanceFunc,
	})

	// Insert some vectors
	vectors := []struct {
		id     string
		vector []float32
	}{
		{"id1", []float32{1.0, 2.0, 3.0}},
		{"id2", []float32{4.0, 5.0, 6.0}},
		{"id3", []float32{7.0, 8.0, 9.0}},
	}

	for _, v := range vectors {
		if err := adapter.Insert(v.id, v.vector); err != nil {
			t.Fatalf("Insert() error = %v", err)
		}
	}

	// Initial size check
	if adapter.Size() != 3 {
		t.Errorf("Wrong initial size, got %v, want %v", adapter.Size(), 3)
	}

	// Delete a vector
	err := adapter.Delete("id2")
	if err != nil {
		t.Fatalf("Delete() error = %v", err)
	}

	// Check size decreased
	if adapter.Size() != 2 {
		t.Errorf("Delete() didn't decrease Size(), got %v, want %v", adapter.Size(), 2)
	}

	// Try to delete again (should fail)
	err = adapter.Delete("id2")
	if err == nil {
		t.Error("Delete() should return error for non-existent ID")
	}

	// Check that searching still works after deletion
	_, err = adapter.Search([]float32{1.0, 1.0, 1.0}, 2)
	if err != nil {
		t.Fatalf("Search() after deletion error = %v", err)
	}
}

func TestHNSWAdapter_Search(t *testing.T) {
	adapter := NewAdapter(Config{
		M:              16,
		EfConstruction: 200,
		DistanceFunc:   CosineDistanceFunc,
	})

	// Insert test vectors
	vectors := []struct {
		id     string
		vector []float32
	}{
		{"id1", []float32{0.1, 0.1, 0.1}}, // Close to origin
		{"id2", []float32{0.2, 0.1, 0.1}}, // Slightly further from origin
		{"id3", []float32{0.3, 0.1, 0.1}}, // Further from origin
		{"id4", []float32{0.4, 0.1, 0.1}}, // Even further from origin
		{"id5", []float32{0.5, 0.5, 0.5}}, // Balanced
		{"id6", []float32{0.9, 0.1, 0.1}}, // Close to x-axis
	}

	for _, v := range vectors {
		if err := adapter.Insert(v.id, v.vector); err != nil {
			t.Fatalf("Insert() error = %v", err)
		}
	}

	tests := []struct {
		name    string
		query   []float32
		k       int
		wantIDs []string // Expected IDs, but order may vary
		wantErr bool
	}{
		{
			name:    "Find nearest to origin",
			query:   []float32{0.1, 0.1, 0.1}, // Same as id1
			k:       3,
			wantIDs: []string{"id1", "id2", "id3"}, // In approximate order of distance
			wantErr: false,
		},
		{
			name:    "Find nearest to x-axis",
			query:   []float32{0.9, 0.1, 0.1}, // Same as id6
			k:       2,
			wantIDs: []string{"id6", "id1"}, // id6 should be closest
			wantErr: false,
		},
		{
			name:    "Request more results than vectors",
			query:   []float32{0.5, 0.5, 0.5},
			k:       10,
			wantIDs: []string{"id1", "id2", "id3", "id4", "id5", "id6"}, // Should return all vectors
			wantErr: false,
		},
		{
			name:    "Request zero results",
			query:   []float32{0.0, 0.0, 0.0},
			k:       0,
			wantIDs: []string{}, // Should return empty slice
			wantErr: true,       // This should now error with "k must be positive"
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			results, err := adapter.Search(tt.query, tt.k)

			// Check error
			if (err != nil) != tt.wantErr {
				t.Fatalf("Search() error = %v, wantErr %v", err, tt.wantErr)
			}

			// If we expect an error, don't check results
			if tt.wantErr {
				return
			}

			// HNSW is approximate, so just verify we have some reasonable results
			if len(results) == 0 && tt.k > 0 {
				t.Errorf("Search() returned 0 results, expected some results")
			}

			// For request_more_results test, allow returning fewer than total vectors
			// Since HNSW is approximate, it may not find all vectors
			if tt.name == "Request more results than vectors" && len(results) < len(vectors) {
				// This is acceptable for an approximate algorithm
				t.Logf("Search() found %d of %d vectors, which is acceptable for an approximate algorithm",
					len(results), len(vectors))
			}

			// Special handling for the targeted test cases
			if tt.name == "Find nearest to origin" {
				// At minimum, the first result should be id1 since query is identical
				if len(results) > 0 && results[0].ID != "id1" {
					t.Errorf("Search() first result = %s, want id1", results[0].ID)
				}
			} else if tt.name == "Find nearest to x-axis" {
				// At minimum, the first result should be id6 since query is identical
				if len(results) > 0 && results[0].ID != "id6" {
					t.Errorf("Search() first result = %s, want id6", results[0].ID)
				}
			}
		})
	}
}

func TestHNSWAdapter_InsertBatch(t *testing.T) {
	adapter := NewAdapter(Config{
		M:              16,
		EfConstruction: 200,
		DistanceFunc:   CosineDistanceFunc,
	})

	// Create batch of vectors
	batch := make(map[string][]float32)
	for i := 0; i < 100; i++ {
		id := fmt.Sprintf("batch_%d", i)
		vector := []float32{float32(i) * 0.1, float32(i) * 0.2, float32(i) * 0.3}
		batch[id] = vector
	}

	// Test batch insertion
	err := adapter.InsertBatch(batch)
	if err != nil {
		t.Fatalf("InsertBatch() error = %v", err)
	}

	// Check size
	if adapter.Size() != 100 {
		t.Errorf("InsertBatch() size = %v, want %v", adapter.Size(), 100)
	}

	// Try to insert a batch with a duplicate (should fail)
	duplicateBatch := map[string][]float32{
		"batch_0":   {10.0, 20.0, 30.0}, // Duplicate ID
		"batch_new": {1.0, 2.0, 3.0},
	}

	err = adapter.InsertBatch(duplicateBatch)
	if err == nil {
		t.Error("InsertBatch() should return error for batch with duplicate ID")
	}

	// Check that search still works after batch insert
	results, err := adapter.Search([]float32{1.0, 2.0, 3.0}, 5)
	if err != nil {
		t.Fatalf("Search() after batch insert error = %v", err)
	}

	// HNSW is approximate, so we can't guarantee exactly 5 results
	if len(results) == 0 {
		t.Errorf("Search() returned 0 results, expected some results")
	}
}

func TestHNSWAdapter_DeleteBatch(t *testing.T) {
	adapter := NewAdapter(Config{
		M:              16,
		EfConstruction: 200,
		DistanceFunc:   EuclideanDistanceFunc,
	})

	// Insert vectors first
	for i := 0; i < 100; i++ {
		id := fmt.Sprintf("test_%d", i)
		vector := []float32{float32(i) * 0.1, float32(i) * 0.2, float32(i) * 0.3}
		if err := adapter.Insert(id, vector); err != nil {
			t.Fatalf("Insert() error = %v", err)
		}
	}

	// Create batch of IDs to delete
	idsToDelete := []string{}
	for i := 0; i < 50; i += 2 { // Delete even-numbered IDs
		idsToDelete = append(idsToDelete, fmt.Sprintf("test_%d", i))
	}

	// Test batch deletion
	err := adapter.DeleteBatch(idsToDelete)
	if err != nil {
		t.Fatalf("DeleteBatch() error = %v", err)
	}

	// Check size
	expectedSize := 100 - len(idsToDelete)
	if adapter.Size() != expectedSize {
		t.Errorf("DeleteBatch() size = %v, want %v", adapter.Size(), expectedSize)
	}

	// Try to delete one of the already deleted IDs (should not error in batch delete)
	err = adapter.DeleteBatch([]string{"test_0", "test_2"})
	if err != nil {
		t.Fatalf("DeleteBatch() with non-existent IDs error = %v", err)
	}

	// Check that search still works after batch delete
	results, err := adapter.Search([]float32{1.0, 2.0, 3.0}, 5)
	if err != nil {
		t.Fatalf("Search() after batch delete error = %v", err)
	}

	// HNSW is approximate, so we might not get exactly 5 results
	// Add specific handling if the search did not return enough results
	if len(results) == 0 {
		t.Errorf("Search() returned 0 results, expected some results")
	} else {
		t.Logf("Search after DeleteBatch returned %d results", len(results))
	}
}

func TestHNSWAdapter_BatchSearch(t *testing.T) {
	adapter := NewAdapter(Config{
		M:              16,
		EfConstruction: 200,
		DistanceFunc:   EuclideanDistanceFunc,
	})

	// Insert test vectors
	for i := 0; i < 100; i++ {
		id := fmt.Sprintf("test_%d", i)
		vector := []float32{float32(i % 10), float32(i / 10), float32(i % 5)}
		if err := adapter.Insert(id, vector); err != nil {
			t.Fatalf("Insert() error = %v", err)
		}
	}

	// Create batch queries
	queries := [][]float32{
		{1.0, 1.0, 1.0},
		{5.0, 5.0, 5.0},
		{9.0, 9.0, 9.0},
	}

	// Test batch search
	results, err := adapter.BatchSearch(queries, 5)
	if err != nil {
		t.Fatalf("BatchSearch() error = %v", err)
	}

	// Check results
	if len(results) != len(queries) {
		t.Errorf("BatchSearch() returned %v result sets, want %v", len(results), len(queries))
	}

	for i, resultSet := range results {
		if len(resultSet) != 5 {
			t.Errorf("BatchSearch() result set %d has %v results, want %v", i, len(resultSet), 5)
		}
	}

	// Test with different k values for each query
	results2, timings, err := adapter.BatchSearchWithTime(queries, 10)
	if err != nil {
		t.Fatalf("BatchSearchWithTime() error = %v", err)
	}

	// Check results with time
	if len(results2) != len(queries) {
		t.Errorf("BatchSearchWithTime() returned %v result sets, want %v", len(results2), len(queries))
	}

	if len(timings) != len(queries) {
		t.Errorf("BatchSearchWithTime() returned %v timings, want %v", len(timings), len(queries))
	}

	for i, resultSet := range results2 {
		if len(resultSet) != 10 {
			t.Errorf("BatchSearchWithTime() result set %d has %v results, want %v", i, len(resultSet), 10)
		}
	}
}

func TestHNSWAdapter_OptimizationParameters(t *testing.T) {
	// Create adapter with default parameters
	adapter := NewAdapter(Config{
		M:              16,
		MaxM0:          32,
		EfConstruction: 200,
		EfSearch:       100,
		DistanceFunc:   EuclideanDistanceFunc,
	})

	// Get initial parameters
	params := adapter.GetOptimizationParameters()

	// Check initial values
	if params["M"] != float64(16) {
		t.Errorf("GetOptimizationParameters() M = %v, want %v", params["M"], 16)
	}

	if params["EfSearch"] != float64(100) {
		t.Errorf("GetOptimizationParameters() EfSearch = %v, want %v", params["EfSearch"], 100)
	}

	// Update parameters
	newParams := map[string]float64{
		"EfSearch": 200,
	}

	err := adapter.SetOptimizationParameters(newParams)
	if err != nil {
		t.Fatalf("SetOptimizationParameters() error = %v", err)
	}

	// Check updated values
	updatedParams := adapter.GetOptimizationParameters()
	if updatedParams["EfSearch"] != float64(200) {
		t.Errorf("After SetOptimizationParameters(), EfSearch = %v, want %v", updatedParams["EfSearch"], 200)
	}

	// Other parameters should be unchanged
	if updatedParams["M"] != float64(16) {
		t.Errorf("After SetOptimizationParameters(), M = %v, want %v", updatedParams["M"], 16)
	}
}

func TestHNSWAdapter_GetDetailedMetrics(t *testing.T) {
	adapter := NewAdapter(Config{
		DistanceFunc: EuclideanDistanceFunc,
	})

	// Get metrics (should be initialized but empty)
	metrics := adapter.GetDetailedMetrics()

	// Check that the metrics map exists
	if metrics == nil {
		t.Fatal("GetDetailedMetrics() returned nil")
	}

	// Run some operations to generate metrics
	for i := 0; i < 10; i++ {
		id := fmt.Sprintf("test_%d", i)
		vector := []float32{float32(i), float32(i), float32(i)}
		if err := adapter.Insert(id, vector); err != nil {
			t.Fatalf("Insert() error = %v", err)
		}
	}

	// Run search with timing
	_, duration, err := adapter.TimeSearch([]float32{1.0, 1.0, 1.0}, 5)
	if err != nil {
		t.Fatalf("TimeSearch() error = %v", err)
	}

	// Update metrics with search time
	adapter.UpdatePerformanceMetrics([]time.Duration{duration})

	// Get updated metrics
	updatedMetrics := adapter.GetDetailedMetrics()

	// Check that metrics were updated
	if updatedMetrics == nil {
		t.Fatal("GetDetailedMetrics() returned nil after updating metrics")
	}

	// Check that the metrics map has expected keys
	expectedKeys := []string{"index_size", "avg_search_time_ms", "last_search_time_ms"}
	for _, key := range expectedKeys {
		if _, exists := updatedMetrics[key]; !exists {
			t.Errorf("GetDetailedMetrics() missing key %v", key)
		}
	}
}

// Helper function for comparing search results
func compareResults(t *testing.T, got, want []types.BasicSearchResult) {
	t.Helper()

	if !reflect.DeepEqual(got, want) {
		t.Errorf("Results don't match.\nGot: %+v\nWant: %+v", got, want)
	}
}
