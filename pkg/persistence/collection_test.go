package persistence

import (
	"reflect"
	"testing"

	"github.com/TFMV/quiver/pkg/facets"
	"github.com/TFMV/quiver/pkg/vectortypes"
)

// TestNewCollection tests the creation of a new collection
func TestNewCollection(t *testing.T) {
	name := "test-collection"
	dimension := 4
	distanceFunc := vectortypes.CosineDistance

	// Create a new collection
	collection := NewCollection(name, dimension, distanceFunc)

	// Verify the collection was created correctly
	if collection.name != name {
		t.Errorf("Expected name %s, got %s", name, collection.name)
	}
	if collection.dimension != dimension {
		t.Errorf("Expected dimension %d, got %d", dimension, collection.dimension)
	}
	if collection.distanceFunc == nil {
		t.Error("Expected distanceFunc to be set")
	}
	if collection.vectors == nil {
		t.Error("Expected vectors map to be initialized")
	}
	if collection.metadata == nil {
		t.Error("Expected metadata map to be initialized")
	}
	if collection.dirty {
		t.Error("Expected new collection to not be dirty")
	}
}

// TestGetName tests the GetName method
func TestGetName(t *testing.T) {
	name := "test-collection"
	collection := NewCollection(name, 4, vectortypes.CosineDistance)

	if collection.GetName() != name {
		t.Errorf("Expected name %s, got %s", name, collection.GetName())
	}
}

// TestGetDimension tests the GetDimension method
func TestGetDimension(t *testing.T) {
	dimension := 128
	collection := NewCollection("test", dimension, vectortypes.CosineDistance)

	if collection.GetDimension() != dimension {
		t.Errorf("Expected dimension %d, got %d", dimension, collection.GetDimension())
	}
}

// TestAddVector tests the AddVector method
func TestAddVector(t *testing.T) {
	collection := NewCollection("test", 4, vectortypes.CosineDistance)

	// Test adding a vector with metadata
	id := "test-vector"
	vector := []float32{0.1, 0.2, 0.3, 0.4}
	metadata := map[string]string{"key1": "value1", "key2": "value2"}

	err := collection.AddVector(id, vector, metadata)
	if err != nil {
		t.Fatalf("Failed to add vector: %v", err)
	}

	// Verify the vector was added
	if !collection.dirty {
		t.Error("Expected collection to be marked dirty after adding a vector")
	}

	// Verify the vector is stored
	storedVector, ok := collection.vectors[id]
	if !ok {
		t.Fatalf("Vector with ID %s not found", id)
	}
	if !reflect.DeepEqual(storedVector, vector) {
		t.Errorf("Expected vector %v, got %v", vector, storedVector)
	}

	// Verify metadata was stored properly
	storedMetadata, ok := collection.metadata[id]
	if !ok {
		t.Fatalf("Metadata for ID %s not found", id)
	}

	// Check metadata values directly
	if !reflect.DeepEqual(storedMetadata, metadata) {
		t.Errorf("Expected metadata %v, got %v", metadata, storedMetadata)
	}

	// Test adding a vector with wrong dimension
	wrongVector := []float32{0.1, 0.2, 0.3}
	err = collection.AddVector("wrong-dim", wrongVector, nil)
	if err == nil {
		t.Error("Expected error when adding vector with wrong dimension")
	}
}

// TestDeleteVector tests the DeleteVector method
func TestDeleteVector(t *testing.T) {
	collection := NewCollection("test", 4, vectortypes.CosineDistance)

	// Add a vector
	id := "test-vector"
	vector := []float32{0.1, 0.2, 0.3, 0.4}
	metadata := map[string]string{"key1": "value1"}

	err := collection.AddVector(id, vector, metadata)
	if err != nil {
		t.Fatalf("Failed to add vector: %v", err)
	}

	// Reset dirty flag
	collection.MarkClean()
	if collection.dirty {
		t.Error("Expected collection to be clean after MarkClean")
	}

	// Delete the vector
	err = collection.DeleteVector(id)
	if err != nil {
		t.Fatalf("Failed to delete vector: %v", err)
	}

	// Verify the vector was deleted
	if !collection.dirty {
		t.Error("Expected collection to be marked dirty after deleting a vector")
	}
	if _, exists := collection.vectors[id]; exists {
		t.Error("Expected vector to be deleted")
	}
	if _, exists := collection.metadata[id]; exists {
		t.Error("Expected metadata to be deleted")
	}

	// Test deleting a non-existent vector
	err = collection.DeleteVector("non-existent")
	if err == nil {
		t.Error("Expected error when deleting non-existent vector")
	}
}

// TestGetVector tests the GetVector method
func TestGetVector(t *testing.T) {
	collection := NewCollection("test", 4, vectortypes.CosineDistance)

	// Add a vector
	id := "test-vector"
	vector := []float32{0.1, 0.2, 0.3, 0.4}
	metadata := map[string]string{"key1": "value1"}

	err := collection.AddVector(id, vector, metadata)
	if err != nil {
		t.Fatalf("Failed to add vector: %v", err)
	}

	// Get the vector
	retrievedVector, retrievedMetadata, err := collection.GetVector(id)
	if err != nil {
		t.Fatalf("Failed to get vector: %v", err)
	}

	// Verify the vector data
	if !reflect.DeepEqual(retrievedVector, vector) {
		t.Errorf("Expected vector %v, got %v", vector, retrievedVector)
	}

	// Verify metadata
	for k, v := range metadata {
		if retrievedMetadata[k] != v {
			t.Errorf("Expected metadata[%s] = %s, got %s", k, v, retrievedMetadata[k])
		}
	}

	// Test getting a non-existent vector
	_, _, err = collection.GetVector("non-existent")
	if err == nil {
		t.Error("Expected error when getting non-existent vector")
	}
}

// TestGetVectors tests the GetVectors method
func TestGetVectors(t *testing.T) {
	collection := NewCollection("test", 4, vectortypes.CosineDistance)

	// Add vectors
	vectors := []struct {
		id       string
		vector   []float32
		metadata map[string]string
	}{
		{"vec1", []float32{0.1, 0.2, 0.3, 0.4}, map[string]string{"key1": "value1"}},
		{"vec2", []float32{0.5, 0.6, 0.7, 0.8}, map[string]string{"key2": "value2"}},
		{"vec3", []float32{0.9, 1.0, 1.1, 1.2}, nil},
	}

	for _, v := range vectors {
		err := collection.AddVector(v.id, v.vector, v.metadata)
		if err != nil {
			t.Fatalf("Failed to add vector %s: %v", v.id, err)
		}
	}

	// Get all vectors
	records := collection.GetVectors()

	// Verify correct number of records
	if len(records) != len(vectors) {
		t.Fatalf("Expected %d records, got %d", len(vectors), len(records))
	}

	// Create a map for easier lookup
	recordMap := make(map[string]VectorRecord)
	for _, record := range records {
		recordMap[record.ID] = record
	}

	// Verify each vector and its metadata
	for _, v := range vectors {
		record, ok := recordMap[v.id]
		if !ok {
			t.Errorf("Expected record with ID %s not found", v.id)
			continue
		}

		if !reflect.DeepEqual(record.Vector, v.vector) {
			t.Errorf("Expected vector %v, got %v", v.vector, record.Vector)
		}

		if v.metadata == nil {
			// If original metadata was nil, expect empty map or nil
			if len(record.Metadata) > 0 {
				t.Errorf("Expected empty metadata for %s, got %v", v.id, record.Metadata)
			}
		} else {
			// Verify metadata keys and values
			for k, expectedVal := range v.metadata {
				if actualVal, ok := record.Metadata[k]; !ok {
					t.Errorf("Missing metadata key %s for vector %s", k, v.id)
				} else if actualVal != expectedVal {
					t.Errorf("For vector %s, expected metadata[%s] = %s, got %s",
						v.id, k, expectedVal, actualVal)
				}
			}
		}
	}
}

// TestSearch tests the Search method
func TestSearch(t *testing.T) {
	collection := NewCollection("test", 3, vectortypes.EuclideanDistance)

	// Add test vectors
	vectors := []struct {
		id     string
		vector []float32
	}{
		{"vec1", []float32{1.0, 0.0, 0.0}},
		{"vec2", []float32{0.0, 1.0, 0.0}},
		{"vec3", []float32{0.0, 0.0, 1.0}},
	}

	for _, v := range vectors {
		err := collection.AddVector(v.id, v.vector, nil)
		if err != nil {
			t.Fatalf("Failed to add vector %s: %v", v.id, err)
		}
	}

	// Test query vector - should be closest to vec1
	query := []float32{0.9, 0.1, 0.1}
	results, err := collection.Search(query, 2)
	if err != nil {
		t.Fatalf("Failed to search: %v", err)
	}

	// Verify results
	if len(results) != 2 {
		t.Fatalf("Expected 2 results, got %d", len(results))
	}

	// First result should be vec1
	if results[0].ID != "vec1" {
		t.Errorf("Expected first result to be vec1, got %s", results[0].ID)
	}

	// Verify results are sorted by distance
	if results[0].Distance > results[1].Distance {
		t.Errorf("Expected results to be sorted by distance")
	}

	// Test query with invalid dimension
	badQuery := []float32{0.9, 0.1}
	_, err = collection.Search(badQuery, 2)
	if err == nil {
		t.Error("Expected error for query with wrong dimension")
	}

	// Test with limit larger than result set
	allResults, err := collection.Search(query, 10)
	if err != nil {
		t.Fatalf("Failed to search with large limit: %v", err)
	}
	if len(allResults) != 3 {
		t.Errorf("Expected 3 results with large limit, got %d", len(allResults))
	}

	// Test with zero limit (should return all)
	zeroLimitResults, err := collection.Search(query, 0)
	if err != nil {
		t.Fatalf("Failed to search with zero limit: %v", err)
	}
	if len(zeroLimitResults) != 3 {
		t.Errorf("Expected 3 results with zero limit, got %d", len(zeroLimitResults))
	}
}

// TestIsDirty tests the IsDirty and MarkClean methods
func TestIsDirty(t *testing.T) {
	collection := NewCollection("test", 4, vectortypes.CosineDistance)

	// New collection should not be dirty
	if collection.IsDirty() {
		t.Error("Expected new collection to not be dirty")
	}

	// Add a vector to make the collection dirty
	err := collection.AddVector("test", []float32{0.1, 0.2, 0.3, 0.4}, nil)
	if err != nil {
		t.Fatalf("Failed to add vector: %v", err)
	}

	// Verify the collection is dirty
	if !collection.IsDirty() {
		t.Error("Expected collection to be dirty after adding a vector")
	}

	// Mark clean and verify
	collection.MarkClean()
	if collection.IsDirty() {
		t.Error("Expected collection to be clean after MarkClean")
	}
}

// TestSortSearchResults tests the SortSearchResults function
func TestSortSearchResults(t *testing.T) {
	results := []SearchResult{
		{ID: "vec1", Distance: 0.5},
		{ID: "vec2", Distance: 0.1},
		{ID: "vec3", Distance: 0.9},
		{ID: "vec4", Distance: 0.3},
	}

	// Sort the results
	SortSearchResults(results)

	// Verify sorting
	for i := 0; i < len(results)-1; i++ {
		if results[i].Distance > results[i+1].Distance {
			t.Errorf("Results not properly sorted at index %d: %f > %f",
				i, results[i].Distance, results[i+1].Distance)
		}
	}

	// Expected order after sorting
	expectedOrder := []string{"vec2", "vec4", "vec1", "vec3"}
	for i, id := range expectedOrder {
		if results[i].ID != id {
			t.Errorf("Expected %s at position %d, got %s", id, i, results[i].ID)
		}
	}
}

// TestDistanceFunctionCheck tests that the collection has a way to get distance function info
func TestDistanceFunctionCheck(t *testing.T) {
	collection := NewCollection("test", 4, vectortypes.CosineDistance)

	// Check if the Persistable interface is implemented
	persistable, ok := interface{}(collection).(Persistable)
	if !ok {
		t.Fatal("Collection does not implement Persistable interface")
	}

	// Try to get distance function using type assertion
	if distFunc, ok := persistable.(interface{ GetDistanceFunction() string }); ok {
		// If the method exists, verify it returns a non-empty string
		if distFunc.GetDistanceFunction() == "" {
			t.Error("Expected non-empty distance function name")
		}
	} else {
		// Not a critical error if the optional method isn't implemented
		t.Log("Collection does not implement GetDistanceFunction method")
	}
}

// TestPersistableImplementation tests that Collection properly implements Persistable
func TestPersistableImplementation(t *testing.T) {
	// Create a collection
	collection := NewCollection("test", 4, vectortypes.CosineDistance)

	// Verify it implements Persistable interface
	var _ Persistable = collection

	// Test Persistable interface methods directly
	persistable := Persistable(collection)

	// Test GetName method
	if persistable.GetName() != collection.name {
		t.Errorf("Persistable GetName() = %s, want %s",
			persistable.GetName(), collection.name)
	}

	// Test GetDimension method
	if persistable.GetDimension() != collection.dimension {
		t.Errorf("Persistable GetDimension() = %d, want %d",
			persistable.GetDimension(), collection.dimension)
	}

	// Test GetVectors method
	persistableVectors := persistable.GetVectors()
	collectionVectors := collection.GetVectors()
	if !reflect.DeepEqual(persistableVectors, collectionVectors) {
		t.Error("Persistable GetVectors() returned different vectors than collection.GetVectors()")
	}

	// Test AddVector method
	testVector := []float32{0.5, 0.5, 0.5, 0.5}
	testMetadata := map[string]string{"test": "value"}

	if err := persistable.AddVector("test-id", testVector, testMetadata); err != nil {
		t.Errorf("Persistable AddVector() failed: %v", err)
	}

	// Verify the vector was added through the Persistable interface
	retrievedVector, retrievedMetadata, err := collection.GetVector("test-id")
	if err != nil {
		t.Errorf("Failed to retrieve vector added through Persistable interface: %v", err)
	}

	if !reflect.DeepEqual(retrievedVector, testVector) {
		t.Errorf("Vector added through Persistable interface doesn't match: got %v, want %v",
			retrievedVector, testVector)
	}

	if retrievedMetadata["test"] != testMetadata["test"] {
		t.Errorf("Metadata added through Persistable interface doesn't match: got %v, want %v",
			retrievedMetadata, testMetadata)
	}
}

// TestFacetFields tests the facet fields functionality
func TestFacetFields(t *testing.T) {
	// Create a collection
	collection := NewCollection("test_facets", 3, func(a, b []float32) float32 {
		return 0.0 // Mock distance function for testing
	})

	// Test initial state
	if len(collection.GetFacetFields()) != 0 {
		t.Errorf("Initial facet fields should be empty, got %v", collection.GetFacetFields())
	}

	// Set facet fields
	facetFields := []string{"category", "price", "tags"}
	collection.SetFacetFields(facetFields)

	// Verify facet fields were set
	if !reflect.DeepEqual(collection.GetFacetFields(), facetFields) {
		t.Errorf("Expected facet fields %v, got %v", facetFields, collection.GetFacetFields())
	}

	// Add vectors with metadata that matches facet fields
	testVectors := []struct {
		id       string
		vector   []float32
		metadata map[string]string
	}{
		{
			id:     "v1",
			vector: []float32{0.1, 0.2, 0.3},
			metadata: map[string]string{
				"category": "electronics",
				"price":    "199.99",
				"tags":     "smartphone,android",
			},
		},
		{
			id:     "v2",
			vector: []float32{0.4, 0.5, 0.6},
			metadata: map[string]string{
				"category": "clothing",
				"price":    "49.99",
				"tags":     "shirt,cotton",
			},
		},
		{
			id:     "v3",
			vector: []float32{0.7, 0.8, 0.9},
			metadata: map[string]string{
				"category": "electronics",
				"price":    "899.99",
				"tags":     "laptop,gaming",
			},
		},
	}

	// Add vectors
	for _, v := range testVectors {
		err := collection.AddVector(v.id, v.vector, v.metadata)
		if err != nil {
			t.Errorf("Failed to add vector: %v", err)
		}
	}

	// Check that facets were extracted properly
	for _, v := range testVectors {
		facets, exists := collection.GetVectorFacets(v.id)
		if !exists {
			t.Errorf("Expected facets for vector %s to exist", v.id)
			continue
		}

		// Verify we have facets for each field
		facetMap := make(map[string]bool)
		for _, f := range facets {
			facetMap[f.Field] = true
		}

		for _, field := range facetFields {
			if !facetMap[field] {
				t.Errorf("Expected facet for field %s in vector %s, but not found", field, v.id)
			}
		}
	}

	// Test searching with facets
	filters := []facets.Filter{
		&facets.EqualityFilter{FieldName: "category", Value: "electronics"},
	}

	results, err := collection.SearchWithFacets([]float32{0.5, 0.5, 0.5}, 10, filters)
	if err != nil {
		t.Errorf("SearchWithFacets failed: %v", err)
	}

	// Should return v1 and v3 (electronics category)
	if len(results) != 2 {
		t.Errorf("Expected 2 results, got %d", len(results))
	}

	// Check result IDs
	resultIDs := make(map[string]bool)
	for _, r := range results {
		resultIDs[r.ID] = true
	}

	if !resultIDs["v1"] || !resultIDs["v3"] {
		t.Errorf("Expected results to include v1 and v3, got %v", resultIDs)
	}

	// Test with multiple filters
	filters = []facets.Filter{
		&facets.EqualityFilter{FieldName: "category", Value: "electronics"},
		&facets.EqualityFilter{FieldName: "tags", Value: "laptop,gaming"},
	}

	results, err = collection.SearchWithFacets([]float32{0.5, 0.5, 0.5}, 10, filters)
	if err != nil {
		t.Errorf("SearchWithFacets failed: %v", err)
	}

	// Should return only v3
	if len(results) != 1 || results[0].ID != "v3" {
		t.Errorf("Expected only v3 in results, got %v", results)
	}
}
