package core

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/TFMV/quiver/pkg/types"
	"github.com/TFMV/quiver/pkg/vectortypes"
)

// MockIndex implements the Index interface for testing
type MockIndex struct {
	vectors       map[string]vectortypes.F32
	insertCalled  int
	deleteCalled  int
	searchCalled  int
	searchResults []types.BasicSearchResult
	searchError   error
}

func NewMockIndex() *MockIndex {
	return &MockIndex{
		vectors:       make(map[string]vectortypes.F32),
		searchResults: []types.BasicSearchResult{},
	}
}

func (m *MockIndex) Insert(id string, vector vectortypes.F32) error {
	m.insertCalled++
	m.vectors[id] = vector
	return nil
}

func (m *MockIndex) Delete(id string) error {
	m.deleteCalled++
	delete(m.vectors, id)
	return nil
}

func (m *MockIndex) Search(vector vectortypes.F32, k int) ([]types.BasicSearchResult, error) {
	m.searchCalled++
	if m.searchError != nil {
		return nil, m.searchError
	}
	return m.searchResults, nil
}

func (m *MockIndex) Size() int {
	return len(m.vectors)
}

// MockBatchIndex implements the BatchIndex interface for testing
type MockBatchIndex struct {
	*MockIndex
	insertBatchCalled int
	deleteBatchCalled int
}

func NewMockBatchIndex() *MockBatchIndex {
	return &MockBatchIndex{
		MockIndex: NewMockIndex(),
	}
}

func (m *MockBatchIndex) InsertBatch(vectors map[string]vectortypes.F32) error {
	m.insertBatchCalled++
	for id, vector := range vectors {
		m.vectors[id] = vector
	}
	return nil
}

func (m *MockBatchIndex) DeleteBatch(ids []string) error {
	m.deleteBatchCalled++
	for _, id := range ids {
		delete(m.vectors, id)
	}
	return nil
}

func TestNewCollection(t *testing.T) {
	mockIndex := NewMockIndex()
	collection := NewCollection("test_collection", 128, mockIndex)

	if collection.Name != "test_collection" {
		t.Errorf("NewCollection() Name = %v, want %v", collection.Name, "test_collection")
	}

	if collection.Dimension != 128 {
		t.Errorf("NewCollection() Dimension = %v, want %v", collection.Dimension, 128)
	}

	if collection.Index != mockIndex {
		t.Errorf("NewCollection() Index not set correctly")
	}

	if collection.Metadata == nil {
		t.Errorf("NewCollection() Metadata not initialized")
	}

	if collection.Vectors == nil {
		t.Errorf("NewCollection() Vectors not initialized")
	}

	if collection.CreatedAt.IsZero() {
		t.Errorf("NewCollection() CreatedAt not set")
	}
}

func TestCollection_Add(t *testing.T) {
	mockIndex := NewMockIndex()
	collection := NewCollection("test_collection", 3, mockIndex)

	t.Run("Add Valid Vector", func(t *testing.T) {
		vector := []float32{1.0, 2.0, 3.0}
		metadata := json.RawMessage(`{"key": "value"}`)

		err := collection.Add("id1", vector, metadata)
		if err != nil {
			t.Fatalf("Add() error = %v", err)
		}

		// Check that the index Insert was called
		if mockIndex.insertCalled != 1 {
			t.Errorf("Add() called Insert %v times, want %v", mockIndex.insertCalled, 1)
		}

		// Check that the vector was stored
		if !reflect.DeepEqual(collection.Vectors["id1"], vector) {
			t.Errorf("Add() didn't store vector correctly")
		}

		// Check that the metadata was stored
		if !reflect.DeepEqual(collection.Metadata["id1"], metadata) {
			t.Errorf("Add() didn't store metadata correctly")
		}
	})

	t.Run("Wrong Dimension", func(t *testing.T) {
		// Try to add a vector with wrong dimension
		vector := []float32{1.0, 2.0}
		err := collection.Add("id2", vector, nil)
		if err == nil {
			t.Error("Add() should return error for wrong dimension")
		}
	})

	t.Run("Invalid Metadata", func(t *testing.T) {
		// Try to add a vector with invalid JSON metadata
		vector := []float32{4.0, 5.0, 6.0}
		invalidMetadata := json.RawMessage(`{"key": value}`) // Missing quotes around value
		err := collection.Add("id3", vector, invalidMetadata)
		if err == nil {
			t.Error("Add() should return error for invalid metadata")
		}
	})
}

func TestCollection_AddBatch(t *testing.T) {
	t.Run("With BatchIndex", func(t *testing.T) {
		mockBatchIndex := NewMockBatchIndex()
		collection := NewCollection("test_collection", 3, mockBatchIndex)

		// Create test vectors
		vectors := []vectortypes.Vector{
			{ID: "id1", Values: []float32{1.0, 2.0, 3.0}, Metadata: json.RawMessage(`{"key": "value1"}`)},
			{ID: "id2", Values: []float32{4.0, 5.0, 6.0}, Metadata: json.RawMessage(`{"key": "value2"}`)},
			{ID: "id3", Values: []float32{7.0, 8.0, 9.0}, Metadata: json.RawMessage(`{"key": "value3"}`)},
		}

		// Add vectors in batch
		err := collection.AddBatch(vectors)
		if err != nil {
			t.Fatalf("AddBatch() error = %v", err)
		}

		// Check that InsertBatch was called
		if mockBatchIndex.insertBatchCalled != 1 {
			t.Errorf("AddBatch() called InsertBatch %v times, want %v", mockBatchIndex.insertBatchCalled, 1)
		}

		// Check that all vectors were stored
		if len(collection.Vectors) != 3 {
			t.Errorf("AddBatch() stored %v vectors, want %v", len(collection.Vectors), 3)
		}

		// Check that all metadata was stored
		if len(collection.Metadata) != 3 {
			t.Errorf("AddBatch() stored %v metadata entries, want %v", len(collection.Metadata), 3)
		}
	})

	t.Run("Without BatchIndex", func(t *testing.T) {
		mockIndex := NewMockIndex()
		collection := NewCollection("test_collection", 3, mockIndex)

		// Create test vectors
		vectors := []vectortypes.Vector{
			{ID: "id1", Values: []float32{1.0, 2.0, 3.0}, Metadata: json.RawMessage(`{"key": "value1"}`)},
			{ID: "id2", Values: []float32{4.0, 5.0, 6.0}, Metadata: json.RawMessage(`{"key": "value2"}`)},
		}

		// Add vectors in batch
		err := collection.AddBatch(vectors)
		if err != nil {
			t.Fatalf("AddBatch() error = %v", err)
		}

		// Check that Insert was called for each vector
		if mockIndex.insertCalled != 2 {
			t.Errorf("AddBatch() called Insert %v times, want %v", mockIndex.insertCalled, 2)
		}

		// Check that all vectors were stored
		if len(collection.Vectors) != 2 {
			t.Errorf("AddBatch() stored %v vectors, want %v", len(collection.Vectors), 2)
		}

		// Check that all metadata was stored
		if len(collection.Metadata) != 2 {
			t.Errorf("AddBatch() stored %v metadata entries, want %v", len(collection.Metadata), 2)
		}
	})

	t.Run("Invalid Vectors", func(t *testing.T) {
		mockIndex := NewMockIndex()
		collection := NewCollection("test_collection", 3, mockIndex)

		// Create test vectors with wrong dimension
		vectors := []vectortypes.Vector{
			{ID: "id1", Values: []float32{1.0, 2.0, 3.0}, Metadata: json.RawMessage(`{"key": "value1"}`)},
			{ID: "id2", Values: []float32{4.0, 5.0}, Metadata: json.RawMessage(`{"key": "value2"}`)}, // Wrong dimension
		}

		// Add vectors in batch (should fail)
		err := collection.AddBatch(vectors)
		if err == nil {
			t.Error("AddBatch() should return error for invalid vectors")
		}

		// Check that no vectors were stored
		if len(collection.Vectors) != 0 {
			t.Errorf("AddBatch() stored %v vectors despite error, want %v", len(collection.Vectors), 0)
		}
	})
}

func TestCollection_Get(t *testing.T) {
	mockIndex := NewMockIndex()
	collection := NewCollection("test_collection", 3, mockIndex)

	// Add a test vector
	vector := []float32{1.0, 2.0, 3.0}
	metadata := json.RawMessage(`{"key": "value"}`)
	collection.Add("id1", vector, metadata)

	t.Run("Get Existing Vector", func(t *testing.T) {
		// Get the vector
		result, err := collection.Get("id1")
		if err != nil {
			t.Fatalf("Get() error = %v", err)
		}

		// Check the result
		if result.ID != "id1" {
			t.Errorf("Get() ID = %v, want %v", result.ID, "id1")
		}
		if !reflect.DeepEqual(result.Values, vector) {
			t.Errorf("Get() Values = %v, want %v", result.Values, vector)
		}
		if !reflect.DeepEqual(result.Metadata, metadata) {
			t.Errorf("Get() Metadata = %v, want %v", result.Metadata, metadata)
		}
	})

	t.Run("Get Non-existent Vector", func(t *testing.T) {
		// Try to get a non-existent vector
		_, err := collection.Get("id2")
		if err == nil {
			t.Error("Get() should return error for non-existent vector")
		}
	})
}

func TestCollection_Delete(t *testing.T) {
	mockIndex := NewMockIndex()
	collection := NewCollection("test_collection", 3, mockIndex)

	// Add a test vector
	collection.Add("id1", []float32{1.0, 2.0, 3.0}, nil)

	t.Run("Delete Existing Vector", func(t *testing.T) {
		// Delete the vector
		err := collection.Delete("id1")
		if err != nil {
			t.Fatalf("Delete() error = %v", err)
		}

		// Check that the index Delete was called
		if mockIndex.deleteCalled != 1 {
			t.Errorf("Delete() called Delete %v times, want %v", mockIndex.deleteCalled, 1)
		}

		// Check that the vector was removed
		if _, exists := collection.Vectors["id1"]; exists {
			t.Errorf("Delete() didn't remove vector")
		}

		// Check that the metadata was removed
		if _, exists := collection.Metadata["id1"]; exists {
			t.Errorf("Delete() didn't remove metadata")
		}
	})

	t.Run("Delete Non-existent Vector", func(t *testing.T) {
		// Try to delete a non-existent vector
		err := collection.Delete("id2")
		if err == nil {
			t.Error("Delete() should return error for non-existent vector")
		}
	})
}

func TestCollection_DeleteBatch(t *testing.T) {
	t.Run("With BatchIndex", func(t *testing.T) {
		mockBatchIndex := NewMockBatchIndex()
		collection := NewCollection("test_collection", 3, mockBatchIndex)

		// Add test vectors
		collection.Add("id1", []float32{1.0, 2.0, 3.0}, nil)
		collection.Add("id2", []float32{4.0, 5.0, 6.0}, nil)
		collection.Add("id3", []float32{7.0, 8.0, 9.0}, nil)

		// Delete vectors in batch
		idsToDelete := []string{"id1", "id3"}
		err := collection.DeleteBatch(idsToDelete)
		if err != nil {
			t.Fatalf("DeleteBatch() error = %v", err)
		}

		// Check that DeleteBatch was called
		if mockBatchIndex.deleteBatchCalled != 1 {
			t.Errorf("DeleteBatch() called DeleteBatch %v times, want %v", mockBatchIndex.deleteBatchCalled, 1)
		}

		// Check that the vectors were removed
		if _, exists := collection.Vectors["id1"]; exists {
			t.Errorf("DeleteBatch() didn't remove vector id1")
		}
		if _, exists := collection.Vectors["id3"]; exists {
			t.Errorf("DeleteBatch() didn't remove vector id3")
		}
		if _, exists := collection.Vectors["id2"]; !exists {
			t.Errorf("DeleteBatch() removed vector id2 which wasn't in the batch")
		}
	})

	t.Run("Without BatchIndex", func(t *testing.T) {
		mockIndex := NewMockIndex()
		collection := NewCollection("test_collection", 3, mockIndex)

		// Add test vectors
		collection.Add("id1", []float32{1.0, 2.0, 3.0}, nil)
		collection.Add("id2", []float32{4.0, 5.0, 6.0}, nil)

		// Delete vectors in batch
		idsToDelete := []string{"id1", "id2"}
		err := collection.DeleteBatch(idsToDelete)
		if err != nil {
			t.Fatalf("DeleteBatch() error = %v", err)
		}

		// Check that Delete was called for each vector
		if mockIndex.deleteCalled != 2 {
			t.Errorf("DeleteBatch() called Delete %v times, want %v", mockIndex.deleteCalled, 2)
		}

		// Check that the vectors were removed
		if len(collection.Vectors) != 0 {
			t.Errorf("DeleteBatch() didn't remove all vectors")
		}
	})
}

func TestCollection_Search(t *testing.T) {
	mockIndex := NewMockIndex()
	collection := NewCollection("test_collection", 3, mockIndex)

	// Add test vectors with metadata
	collection.Add("id1", []float32{1.0, 2.0, 3.0}, json.RawMessage(`{"category": "A", "score": 10}`))
	collection.Add("id2", []float32{4.0, 5.0, 6.0}, json.RawMessage(`{"category": "B", "score": 20}`))
	collection.Add("id3", []float32{7.0, 8.0, 9.0}, json.RawMessage(`{"category": "A", "score": 30}`))

	// Set up mock search results
	mockIndex.searchResults = []types.BasicSearchResult{
		{ID: "id1", Distance: 0.1},
		{ID: "id3", Distance: 0.3},
		{ID: "id2", Distance: 0.5},
	}

	t.Run("Basic Search", func(t *testing.T) {
		request := types.SearchRequest{
			Vector: []float32{1.0, 1.0, 1.0},
			TopK:   3,
			Options: types.SearchOptions{
				IncludeVectors:  true,
				IncludeMetadata: true,
			},
		}

		// Perform search
		results, err := collection.Search(request)
		if err != nil {
			t.Fatalf("Search() error = %v", err)
		}

		// Check that the index Search was called
		if mockIndex.searchCalled != 1 {
			t.Errorf("Search() called Search %v times, want %v", mockIndex.searchCalled, 1)
		}

		// Check results
		if len(results.Results) != 3 {
			t.Fatalf("Search() returned %v results, want %v", len(results.Results), 3)
		}

		// Check that results include vectors and metadata
		for _, result := range results.Results {
			if result.Vector == nil {
				t.Errorf("Search() result vector is nil for ID %s", result.ID)
			}
			if result.Metadata == nil {
				t.Errorf("Search() result metadata is nil for ID %s", result.ID)
			}
			if result.Distance == 0 && result.ID != "id1" {
				t.Errorf("Search() result distance is 0 for non-exact match ID %s", result.ID)
			}
		}
	})

	t.Run("Search with Filters", func(t *testing.T) {
		request := types.SearchRequest{
			Vector: []float32{1.0, 1.0, 1.0},
			TopK:   3,
			Options: types.SearchOptions{
				IncludeVectors:  true,
				IncludeMetadata: true,
			},
			Filters: []types.Filter{
				{Field: "category", Value: "A", Operator: string(Equals)},
			},
		}

		// Perform search with filters
		results, err := collection.Search(request)
		if err != nil {
			t.Fatalf("Search() with filters error = %v", err)
		}

		// Check that only results matching the filter are returned
		if len(results.Results) != 2 {
			t.Fatalf("Search() with filters returned %v results, want %v", len(results.Results), 2)
		}

		// All results should have category A
		for _, result := range results.Results {
			var metadata map[string]interface{}
			if err := json.Unmarshal(result.Metadata, &metadata); err != nil {
				t.Fatalf("Failed to unmarshal metadata: %v", err)
			}
			if category, ok := metadata["category"].(string); !ok || category != "A" {
				t.Errorf("Search() with filters returned result with category %v, want %v", metadata["category"], "A")
			}
		}
	})
}

func TestCollection_LegacySearch(t *testing.T) {
	mockIndex := NewMockIndex()
	collection := NewCollection("test_collection", 3, mockIndex)

	// Add test vectors with metadata
	collection.Add("id1", []float32{1.0, 2.0, 3.0}, json.RawMessage(`{"category": "A", "score": 10}`))
	collection.Add("id2", []float32{4.0, 5.0, 6.0}, json.RawMessage(`{"category": "B", "score": 20}`))

	// Set up mock search results
	mockIndex.searchResults = []types.BasicSearchResult{
		{ID: "id1", Distance: 0.1},
		{ID: "id2", Distance: 0.5},
	}

	// Perform legacy search
	results, err := collection.LegacySearch([]float32{1.0, 1.0, 1.0}, 2, nil)
	if err != nil {
		t.Fatalf("LegacySearch() error = %v", err)
	}

	// Check results
	if len(results) != 2 {
		t.Fatalf("LegacySearch() returned %v results, want %v", len(results), 2)
	}

	// Check result order
	if results[0].ID != "id1" || results[1].ID != "id2" {
		t.Errorf("LegacySearch() incorrect result order")
	}

	// Check distances
	if results[0].Distance != 0.1 || results[1].Distance != 0.5 {
		t.Errorf("LegacySearch() incorrect distances")
	}
}

func TestCollection_Update(t *testing.T) {
	mockIndex := NewMockIndex()
	collection := NewCollection("test_collection", 3, mockIndex)

	// Add a test vector
	originalVector := []float32{1.0, 2.0, 3.0}
	originalMetadata := json.RawMessage(`{"key": "value"}`)
	collection.Add("id1", originalVector, originalMetadata)

	t.Run("Update Existing Vector", func(t *testing.T) {
		// Update the vector
		newVector := []float32{4.0, 5.0, 6.0}
		newMetadata := json.RawMessage(`{"key": "new_value"}`)
		err := collection.Update("id1", newVector, newMetadata)
		if err != nil {
			t.Fatalf("Update() error = %v", err)
		}

		// Check that the vector was updated
		if !reflect.DeepEqual(collection.Vectors["id1"], newVector) {
			t.Errorf("Update() didn't update vector correctly")
		}

		// Check that the metadata was updated
		if !reflect.DeepEqual(collection.Metadata["id1"], newMetadata) {
			t.Errorf("Update() didn't update metadata correctly")
		}
	})

	t.Run("Update Non-existent Vector", func(t *testing.T) {
		// Try to update a non-existent vector
		err := collection.Update("id2", []float32{7.0, 8.0, 9.0}, nil)
		if err == nil {
			t.Error("Update() should return error for non-existent vector")
		}
	})

	t.Run("Update with Wrong Dimension", func(t *testing.T) {
		// Try to update with a vector of wrong dimension
		err := collection.Update("id1", []float32{7.0, 8.0}, nil)
		if err == nil {
			t.Error("Update() should return error for wrong dimension")
		}
	})
}

func TestCollection_Stats(t *testing.T) {
	mockIndex := NewMockIndex()
	collection := NewCollection("test_collection", 3, mockIndex)

	// Add some vectors
	collection.Add("id1", []float32{1.0, 2.0, 3.0}, nil)
	collection.Add("id2", []float32{4.0, 5.0, 6.0}, nil)

	// Get stats
	stats := collection.Stats()

	// Check stats
	if stats.Name != "test_collection" {
		t.Errorf("Stats() Name = %v, want %v", stats.Name, "test_collection")
	}
	if stats.Dimension != 3 {
		t.Errorf("Stats() Dimension = %v, want %v", stats.Dimension, 3)
	}
	if stats.VectorCount != 2 {
		t.Errorf("Stats() VectorCount = %v, want %v", stats.VectorCount, 2)
	}
	if stats.CreatedAt.IsZero() {
		t.Errorf("Stats() CreatedAt is zero")
	}
}

func TestCollection_FluentSearch(t *testing.T) {
	mockIndex := NewMockIndex()
	collection := NewCollection("test_collection", 3, mockIndex)

	// Add test vectors with metadata
	collection.Add("id1", []float32{1.0, 2.0, 3.0}, json.RawMessage(`{"category": "A", "score": 10}`))
	collection.Add("id2", []float32{4.0, 5.0, 6.0}, json.RawMessage(`{"category": "B", "score": 20}`))
	collection.Add("id3", []float32{7.0, 8.0, 9.0}, json.RawMessage(`{"category": "A", "score": 30}`))

	// Set up mock search results
	mockIndex.searchResults = []types.BasicSearchResult{
		{ID: "id1", Distance: 0.1},
		{ID: "id3", Distance: 0.3},
		{ID: "id2", Distance: 0.5},
	}

	// Test fluent search builder
	results, err := collection.FluentSearch([]float32{1.0, 1.0, 1.0}).
		WithK(2).
		IncludeVectors(true).
		IncludeMetadata(true).
		Filter("category", "A").
		Execute()

	if err != nil {
		t.Fatalf("FluentSearch() error = %v", err)
	}

	// Check results
	if len(results.Results) != 2 {
		t.Fatalf("FluentSearch() returned %v results, want %v", len(results.Results), 2)
	}

	// Check that results are filtered
	for _, result := range results.Results {
		var metadata map[string]interface{}
		if err := json.Unmarshal(result.Metadata, &metadata); err != nil {
			t.Fatalf("Failed to unmarshal metadata: %v", err)
		}
		if category, ok := metadata["category"].(string); !ok || category != "A" {
			t.Errorf("FluentSearch() returned result with category %v, want %v", metadata["category"], "A")
		}
	}
}

// TestMatchesFilter tests the matchesFilter function directly
func TestMatchesFilter(t *testing.T) {
	testCases := []struct {
		name     string
		metadata map[string]interface{}
		filter   Filter
		want     bool
	}{
		{
			name:     "Equals - Match",
			metadata: map[string]interface{}{"key": "value"},
			filter:   Filter{Field: "key", Operator: Equals, Value: "value"},
			want:     true,
		},
		{
			name:     "Equals - No Match",
			metadata: map[string]interface{}{"key": "value"},
			filter:   Filter{Field: "key", Operator: Equals, Value: "other"},
			want:     false,
		},
		{
			name:     "NotEquals - Match",
			metadata: map[string]interface{}{"key": "value"},
			filter:   Filter{Field: "key", Operator: NotEquals, Value: "other"},
			want:     true,
		},
		{
			name:     "NotEquals - No Match",
			metadata: map[string]interface{}{"key": "value"},
			filter:   Filter{Field: "key", Operator: NotEquals, Value: "value"},
			want:     false,
		},
		{
			name:     "GreaterThan - Match",
			metadata: map[string]interface{}{"key": 10},
			filter:   Filter{Field: "key", Operator: GreaterThan, Value: 5},
			want:     true,
		},
		{
			name:     "GreaterThan - No Match",
			metadata: map[string]interface{}{"key": 10},
			filter:   Filter{Field: "key", Operator: GreaterThan, Value: 15},
			want:     false,
		},
		{
			name:     "In - Match",
			metadata: map[string]interface{}{"key": "value"},
			filter:   Filter{Field: "key", Operator: In, Value: []interface{}{"other", "value"}},
			want:     true,
		},
		{
			name:     "In - No Match",
			metadata: map[string]interface{}{"key": "value"},
			filter:   Filter{Field: "key", Operator: In, Value: []interface{}{"other", "another"}},
			want:     false,
		},
		{
			name:     "Field Not Found",
			metadata: map[string]interface{}{"key": "value"},
			filter:   Filter{Field: "missing", Operator: Equals, Value: "value"},
			want:     false,
		},
		{
			name:     "Invalid Operator",
			metadata: map[string]interface{}{"key": "value"},
			filter:   Filter{Field: "key", Operator: "invalid", Value: "value"},
			want:     false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := matchesFilter(tc.metadata, tc.filter)
			if got != tc.want {
				t.Errorf("matchesFilter() = %v, want %v", got, tc.want)
			}
		})
	}
}
