package persistence

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/TFMV/quiver/pkg/facets"
)

// TestNewManager tests the creation of a new persistence manager
func TestNewManager(t *testing.T) {
	// Create a temporary directory for testing
	tempDir, err := os.MkdirTemp("", "quiver-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Create a new manager
	flushInterval := 1 * time.Second
	manager, err := NewManager(tempDir, flushInterval)
	if err != nil {
		t.Fatalf("Failed to create manager: %v", err)
	}
	defer manager.Stop()

	// Verify the manager was created correctly
	if manager.rootDir != tempDir {
		t.Errorf("Expected rootDir %s, got %s", tempDir, manager.rootDir)
	}
	if manager.flushInterval != flushInterval {
		t.Errorf("Expected flushInterval %v, got %v", flushInterval, manager.flushInterval)
	}
	if !manager.running {
		t.Error("Expected manager to be running")
	}
	if manager.getCollectionCallback != nil {
		t.Error("Expected getCollectionCallback to be nil")
	}
	if len(manager.dirtyCollections) != 0 {
		t.Errorf("Expected empty dirtyCollections, got %d items", len(manager.dirtyCollections))
	}
}

// TestMarkCollectionDirty tests marking a collection as dirty
func TestMarkCollectionDirty(t *testing.T) {
	// Create a temporary directory for testing
	tempDir, err := os.MkdirTemp("", "quiver-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Create a new manager
	manager, err := NewManager(tempDir, 1*time.Second)
	if err != nil {
		t.Fatalf("Failed to create manager: %v", err)
	}
	defer manager.Stop()

	// Mark a collection as dirty
	collectionName := "test-collection"
	manager.MarkCollectionDirty(collectionName)

	// Verify the collection was marked as dirty
	if !manager.dirtyCollections[collectionName] {
		t.Errorf("Expected collection %s to be marked dirty", collectionName)
	}
}

// TestSetGetCollectionCallback tests setting the collection callback
func TestSetGetCollectionCallback(t *testing.T) {
	// Create a temporary directory for testing
	tempDir, err := os.MkdirTemp("", "quiver-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Create a new manager
	manager, err := NewManager(tempDir, 1*time.Second)
	if err != nil {
		t.Fatalf("Failed to create manager: %v", err)
	}
	defer manager.Stop()

	// Define a callback function
	callback := func(name string) (Persistable, error) {
		return nil, nil
	}

	// Set the callback
	manager.SetGetCollectionCallback(callback)

	// Verify the callback was set
	if manager.getCollectionCallback == nil {
		t.Error("Expected getCollectionCallback to be set")
	}
}

// TestStop tests stopping the manager
func TestStop(t *testing.T) {
	// Create a temporary directory for testing
	tempDir, err := os.MkdirTemp("", "quiver-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Create a new manager
	manager, err := NewManager(tempDir, 1*time.Second)
	if err != nil {
		t.Fatalf("Failed to create manager: %v", err)
	}

	// Stop the manager
	manager.Stop()

	// Verify the manager was stopped
	if manager.running {
		t.Error("Expected manager to be stopped")
	}

	// Stopping again should not panic
	manager.Stop()
}

// TestSaveLoadCollectionConfig tests saving and loading a collection's configuration
func TestSaveLoadCollectionConfig(t *testing.T) {
	// Create a temporary directory for testing
	tempDir, err := os.MkdirTemp("", "quiver-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Create a test config
	config := CollectionConfig{
		Name:         "test-collection",
		Dimension:    128,
		DistanceFunc: "cosine",
		CreatedAt:    time.Now().Truncate(time.Millisecond), // Truncate to avoid precision issues
	}

	// Path to save the config
	configPath := filepath.Join(tempDir, "config.json")

	// Save the config
	err = SaveCollectionConfig(config, configPath)
	if err != nil {
		t.Fatalf("Failed to save config: %v", err)
	}

	// Load the config
	loadedConfig, err := LoadCollectionConfig(configPath)
	if err != nil {
		t.Fatalf("Failed to load config: %v", err)
	}

	// Verify the loaded config matches the original
	if loadedConfig.Name != config.Name {
		t.Errorf("Expected name %s, got %s", config.Name, loadedConfig.Name)
	}
	if loadedConfig.Dimension != config.Dimension {
		t.Errorf("Expected dimension %d, got %d", config.Dimension, loadedConfig.Dimension)
	}
	if loadedConfig.DistanceFunc != config.DistanceFunc {
		t.Errorf("Expected distanceFunc %s, got %s", config.DistanceFunc, loadedConfig.DistanceFunc)
	}
	if !loadedConfig.CreatedAt.Equal(config.CreatedAt) {
		t.Errorf("Expected createdAt %v, got %v", config.CreatedAt, loadedConfig.CreatedAt)
	}
}

// TestWriteReadVectorsToFile tests writing and reading vectors to/from JSON
func TestWriteReadVectorsToFile(t *testing.T) {
	// Create a temporary directory for testing
	tempDir, err := os.MkdirTemp("", "quiver-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Create test vectors
	vectors := []VectorRecord{
		{
			ID:       "vec1",
			Vector:   []float32{0.1, 0.2, 0.3, 0.4},
			Metadata: map[string]string{"key1": "value1", "key2": "value2"},
		},
		{
			ID:       "vec2",
			Vector:   []float32{0.5, 0.6, 0.7, 0.8},
			Metadata: map[string]string{"key3": "value3"},
		},
		{
			ID:       "vec3",
			Vector:   []float32{0.9, 1.0, 1.1, 1.2},
			Metadata: nil,
		},
	}

	// Path to save the vectors
	vectorsPath := filepath.Join(tempDir, "vectors.json")

	// Write vectors to file
	err = WriteVectorsToFile(vectors, vectorsPath)
	if err != nil {
		t.Fatalf("Failed to write vectors: %v", err)
	}

	// Read vectors from file
	loadedVectors, err := ReadVectorsFromFile(vectorsPath)
	if err != nil {
		t.Fatalf("Failed to read vectors: %v", err)
	}

	// Verify the loaded vectors match the original
	if len(loadedVectors) != len(vectors) {
		t.Fatalf("Expected %d vectors, got %d", len(vectors), len(loadedVectors))
	}

	for i, vector := range vectors {
		loadedVector := loadedVectors[i]

		if loadedVector.ID != vector.ID {
			t.Errorf("Expected ID %s, got %s", vector.ID, loadedVector.ID)
		}

		if len(loadedVector.Vector) != len(vector.Vector) {
			t.Errorf("Expected vector length %d, got %d", len(vector.Vector), len(loadedVector.Vector))
		} else {
			for j, val := range vector.Vector {
				if loadedVector.Vector[j] != val {
					t.Errorf("Expected vector[%d] = %f, got %f", j, val, loadedVector.Vector[j])
				}
			}
		}

		if len(loadedVector.Metadata) != len(vector.Metadata) {
			t.Errorf("Expected metadata length %d, got %d", len(vector.Metadata), len(loadedVector.Metadata))
		} else if vector.Metadata != nil {
			for k, v := range vector.Metadata {
				if loadedVector.Metadata[k] != v {
					t.Errorf("Expected metadata[%s] = %s, got %s", k, v, loadedVector.Metadata[k])
				}
			}
		}
	}
}

// MockPersistable implements the Persistable interface for testing
type MockPersistable struct {
	name        string
	dimension   int
	vectors     []VectorRecord
	addCalled   bool
	facetFields []string
}

func (m *MockPersistable) GetName() string {
	return m.name
}

func (m *MockPersistable) GetDimension() int {
	return m.dimension
}

func (m *MockPersistable) GetVectors() []VectorRecord {
	return m.vectors
}

func (m *MockPersistable) AddVector(id string, vector []float32, metadata map[string]string) error {
	m.addCalled = true
	m.vectors = append(m.vectors, VectorRecord{
		ID:       id,
		Vector:   vector,
		Metadata: metadata,
	})
	return nil
}

func (m *MockPersistable) GetDistanceFunction() string {
	return "cosine"
}

// Implement the new required facet methods

func (m *MockPersistable) GetFacetFields() []string {
	return m.facetFields
}

func (m *MockPersistable) SetFacetFields(fields []string) {
	m.facetFields = fields
}

// TestFlushLoadCollection tests flushing and loading a collection
func TestFlushLoadCollection(t *testing.T) {
	// Create a temporary directory for testing
	tempDir, err := os.MkdirTemp("", "quiver-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Create a manager
	manager, err := NewManager(tempDir, 1*time.Second)
	if err != nil {
		t.Fatalf("Failed to create manager: %v", err)
	}
	defer manager.Stop()

	// Create a mock collection
	collection := &MockPersistable{
		name:      "test-collection",
		dimension: 4,
		vectors: []VectorRecord{
			{
				ID:       "vec1",
				Vector:   []float32{0.1, 0.2, 0.3, 0.4},
				Metadata: map[string]string{"key1": "value1"},
			},
			{
				ID:       "vec2",
				Vector:   []float32{0.5, 0.6, 0.7, 0.8},
				Metadata: map[string]string{"key2": "value2"},
			},
		},
	}

	// Collection path
	collectionPath := filepath.Join(tempDir, collection.name)

	// Flush the collection
	err = manager.FlushCollection(collection, collectionPath)
	if err != nil {
		t.Fatalf("Failed to flush collection: %v", err)
	}

	// Create a new collection to load into
	loadCollection := &MockPersistable{
		name:      "test-collection",
		dimension: 4,
		vectors:   []VectorRecord{},
	}

	// Load the collection
	err = manager.LoadCollection(loadCollection, collectionPath)
	if err != nil {
		t.Fatalf("Failed to load collection: %v", err)
	}

	// Verify the collection was loaded correctly
	if !loadCollection.addCalled {
		t.Error("Expected AddVector to be called")
	}
	if len(loadCollection.vectors) != len(collection.vectors) {
		t.Fatalf("Expected %d vectors, got %d", len(collection.vectors), len(loadCollection.vectors))
	}

	// Check vectors were loaded correctly
	for i, vector := range collection.vectors {
		loadedVector := loadCollection.vectors[i]

		if loadedVector.ID != vector.ID {
			t.Errorf("Expected ID %s, got %s", vector.ID, loadedVector.ID)
		}

		if len(loadedVector.Vector) != len(vector.Vector) {
			t.Errorf("Expected vector length %d, got %d", len(vector.Vector), len(loadedVector.Vector))
		} else {
			for j, val := range vector.Vector {
				if loadedVector.Vector[j] != val {
					t.Errorf("Expected vector[%d] = %f, got %f", j, val, loadedVector.Vector[j])
				}
			}
		}

		if len(loadedVector.Metadata) != len(vector.Metadata) {
			t.Errorf("Expected metadata length %d, got %d", len(vector.Metadata), len(loadedVector.Metadata))
		} else {
			for k, v := range vector.Metadata {
				if loadedVector.Metadata[k] != v {
					t.Errorf("Expected metadata[%s] = %s, got %s", k, v, loadedVector.Metadata[k])
				}
			}
		}
	}
}

// TestFlushDirtyCollections tests flushing dirty collections
func TestFlushDirtyCollections(t *testing.T) {
	// Create a temporary directory for testing
	tempDir, err := os.MkdirTemp("", "quiver-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Create a manager
	manager, err := NewManager(tempDir, 1*time.Second)
	if err != nil {
		t.Fatalf("Failed to create manager: %v", err)
	}
	defer manager.Stop()

	// Create a mock collection
	collection := &MockPersistable{
		name:      "test-collection",
		dimension: 4,
		vectors: []VectorRecord{
			{
				ID:       "vec1",
				Vector:   []float32{0.1, 0.2, 0.3, 0.4},
				Metadata: map[string]string{"key1": "value1"},
			},
		},
	}

	// Set up callback
	callbackCalled := false
	manager.SetGetCollectionCallback(func(name string) (Persistable, error) {
		callbackCalled = true
		if name == collection.name {
			return collection, nil
		}
		return nil, nil
	})

	// Mark collection as dirty
	manager.MarkCollectionDirty(collection.name)

	// Flush dirty collections
	manager.FlushDirtyCollections()

	// Verify callback was called
	if !callbackCalled {
		t.Error("Expected callback to be called")
	}

	// Verify collection was flushed (no longer dirty)
	if manager.dirtyCollections[collection.name] {
		t.Error("Expected collection to be no longer dirty")
	}

	// Check that files were created
	configPath := filepath.Join(tempDir, collection.name, "config.json")
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		t.Errorf("Expected config file %s to exist", configPath)
	}

	// Check for vectors file
	parquetPath := filepath.Join(tempDir, collection.name, "vectors.parquet")
	jsonPath := filepath.Join(tempDir, collection.name, "vectors.json")
	if _, err := os.Stat(parquetPath); os.IsNotExist(err) {
		// If Parquet file doesn't exist, check for JSON file
		if _, err := os.Stat(jsonPath); os.IsNotExist(err) {
			t.Error("Expected either Parquet or JSON vectors file to exist")
		}
	}
}

// TestCreateRestoreBackup tests creating and restoring a backup
func TestCreateRestoreBackup(t *testing.T) {
	// Create temporary directories for testing
	sourceDir, err := os.MkdirTemp("", "quiver-test-source-*")
	if err != nil {
		t.Fatalf("Failed to create source directory: %v", err)
	}
	defer os.RemoveAll(sourceDir)

	backupDir, err := os.MkdirTemp("", "quiver-test-backup-*")
	if err != nil {
		t.Fatalf("Failed to create backup directory: %v", err)
	}
	defer os.RemoveAll(backupDir)

	restoreDir, err := os.MkdirTemp("", "quiver-test-restore-*")
	if err != nil {
		t.Fatalf("Failed to create restore directory: %v", err)
	}
	defer os.RemoveAll(restoreDir)

	// Create a manager
	manager, err := NewManager(sourceDir, 1*time.Second)
	if err != nil {
		t.Fatalf("Failed to create manager: %v", err)
	}
	defer manager.Stop()

	// Create some test files in the source directory
	testFiles := []string{
		"file1.txt",
		"file2.json",
		filepath.Join("subdir", "file3.txt"),
	}

	for _, file := range testFiles {
		path := filepath.Join(sourceDir, file)
		dir := filepath.Dir(path)
		if err := os.MkdirAll(dir, 0755); err != nil {
			t.Fatalf("Failed to create directory %s: %v", dir, err)
		}
		if err := os.WriteFile(path, []byte("test content"), 0644); err != nil {
			t.Fatalf("Failed to create file %s: %v", path, err)
		}
	}

	// Create a backup
	err = manager.CreateBackup(sourceDir, backupDir)
	if err != nil {
		t.Fatalf("Failed to create backup: %v", err)
	}

	// Verify backup files exist
	for _, file := range testFiles {
		backupPath := filepath.Join(backupDir, file)
		if _, err := os.Stat(backupPath); os.IsNotExist(err) {
			t.Errorf("Expected backup file %s to exist", backupPath)
		}
	}

	// Restore the backup
	err = manager.RestoreBackup(backupDir, restoreDir)
	if err != nil {
		t.Fatalf("Failed to restore backup: %v", err)
	}

	// Verify restored files exist
	for _, file := range testFiles {
		restorePath := filepath.Join(restoreDir, file)
		if _, err := os.Stat(restorePath); os.IsNotExist(err) {
			t.Errorf("Expected restored file %s to exist", restorePath)
		}
	}
}

// TestManagerWithFacets tests the manager with facet fields
func TestManagerWithFacets(t *testing.T) {
	// Create a temporary directory for testing
	tempDir, err := os.MkdirTemp("", "quiver_test_manager_facets")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Create a manager
	manager, err := NewManager(tempDir, 0)
	if err != nil {
		t.Fatalf("Failed to create manager: %v", err)
	}

	// Create a collection with facet fields
	collection := NewCollection("test_facets", 3, func(a, b []float32) float32 {
		return 0.0 // Mock distance function for testing
	})

	// Set facet fields
	facetFields := []string{"category", "price", "tags"}
	collection.SetFacetFields(facetFields)

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
	}

	// Add vectors
	for _, v := range testVectors {
		err := collection.AddVector(v.id, v.vector, v.metadata)
		if err != nil {
			t.Errorf("Failed to add vector: %v", err)
		}
	}

	// Save the collection
	collectionPath := filepath.Join(tempDir, collection.GetName())
	err = manager.FlushCollection(collection, collectionPath)
	if err != nil {
		t.Fatalf("Failed to flush collection: %v", err)
	}

	// Create a new collection to load into
	loadedCollection := NewCollection("test_facets", 3, func(a, b []float32) float32 {
		return 0.0 // Mock distance function for testing
	})

	// Load the collection
	err = manager.LoadCollection(loadedCollection, collectionPath)
	if err != nil {
		t.Fatalf("Failed to load collection: %v", err)
	}

	// Verify facet fields were loaded
	if !reflect.DeepEqual(loadedCollection.GetFacetFields(), facetFields) {
		t.Errorf("Expected facet fields %v, got %v", facetFields, loadedCollection.GetFacetFields())
	}

	// Verify vectors were loaded
	for _, v := range testVectors {
		vector, metadata, err := loadedCollection.GetVector(v.id)
		if err != nil {
			t.Errorf("Failed to get vector %s: %v", v.id, err)
			continue
		}

		if !reflect.DeepEqual(vector, v.vector) {
			t.Errorf("Vector mismatch for %s: expected %v, got %v", v.id, v.vector, vector)
		}

		if !reflect.DeepEqual(metadata, v.metadata) {
			t.Errorf("Metadata mismatch for %s: expected %v, got %v", v.id, v.metadata, metadata)
		}
	}

	// Test searching with facets
	filters := []facets.Filter{
		&facets.EqualityFilter{FieldName: "category", Value: "electronics"},
	}

	results, err := loadedCollection.SearchWithFacets([]float32{0.5, 0.5, 0.5}, 10, filters)
	if err != nil {
		t.Errorf("SearchWithFacets failed: %v", err)
	}

	// Should return v1 (electronics category)
	if len(results) != 1 {
		t.Errorf("Expected 1 result, got %d", len(results))
	}

	if len(results) > 0 && results[0].ID != "v1" {
		t.Errorf("Expected result v1, got %s", results[0].ID)
	}
}
