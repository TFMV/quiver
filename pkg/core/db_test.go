package core

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/TFMV/quiver/pkg/hnsw"
	"github.com/TFMV/quiver/pkg/hybrid"
	"github.com/TFMV/quiver/pkg/types"
	"github.com/TFMV/quiver/pkg/vectortypes"
)

func TestDefaultDBOptions(t *testing.T) {
	options := DefaultDBOptions()

	// Check that the defaults are set correctly
	if options.StoragePath != "./data" {
		t.Errorf("DefaultDBOptions() StoragePath = %v, want %v", options.StoragePath, "./data")
	}

	if options.EnableMetrics != true {
		t.Errorf("DefaultDBOptions() EnableMetrics = %v, want %v", options.EnableMetrics, true)
	}

	if options.EnablePersistence != true {
		t.Errorf("DefaultDBOptions() EnablePersistence = %v, want %v", options.EnablePersistence, true)
	}

	if options.FlushInterval != 5*time.Minute {
		t.Errorf("DefaultDBOptions() FlushInterval = %v, want %v", options.FlushInterval, 5*time.Minute)
	}

	if options.DefaultHNSWConfig.M != 16 {
		t.Errorf("DefaultDBOptions() DefaultHNSWConfig.M = %v, want %v", options.DefaultHNSWConfig.M, 16)
	}

	if options.EnableHybridSearch != true {
		t.Errorf("DefaultDBOptions() EnableHybridSearch = %v, want %v", options.EnableHybridSearch, true)
	}
}

func TestNewDB(t *testing.T) {
	// Create a temporary directory for test storage
	tempDir, err := os.MkdirTemp("", "quiver-test")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Test cases
	tests := []struct {
		name        string
		options     DBOptions
		wantErr     bool
		errContains string
	}{
		{
			name: "Valid Options",
			options: DBOptions{
				StoragePath:       tempDir,
				EnableMetrics:     true,
				EnablePersistence: true,
				FlushInterval:     time.Second,
				DefaultHNSWConfig: hnsw.Config{
					M:              16,
					EfConstruction: 200,
					DistanceFunc:   hnsw.CosineDistanceFunc,
				},
				EnableHybridSearch: true,
				HybridConfig:       hybrid.DefaultIndexConfig(),
			},
			wantErr: false,
		},
		{
			name: "Empty Storage Path",
			options: DBOptions{
				StoragePath:       "",
				EnableMetrics:     true,
				EnablePersistence: true,
			},
			wantErr:     true,
			errContains: "invalid configuration",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			db, err := NewDB(tt.options)

			// Check error
			if (err != nil) != tt.wantErr {
				t.Errorf("NewDB() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.wantErr && tt.errContains != "" && err != nil {
				if err.Error() != tt.errContains {
					t.Errorf("NewDB() error = %v, want error containing %v", err, tt.errContains)
				}
				return
			}

			// If no error, check DB initialization
			if !tt.wantErr {
				if db == nil {
					t.Fatal("NewDB() returned nil DB without error")
				}

				if db.options.StoragePath != tt.options.StoragePath {
					t.Errorf("NewDB() StoragePath = %v, want %v", db.options.StoragePath, tt.options.StoragePath)
				}

				if db.options.EnableMetrics != tt.options.EnableMetrics {
					t.Errorf("NewDB() EnableMetrics = %v, want %v", db.options.EnableMetrics, tt.options.EnableMetrics)
				}

				if db.options.EnablePersistence != tt.options.EnablePersistence {
					t.Errorf("NewDB() EnablePersistence = %v, want %v", db.options.EnablePersistence, tt.options.EnablePersistence)
				}

				// Check collections map initialized
				if db.collections == nil {
					t.Error("NewDB() collections map not initialized")
				}

				// Check metrics initialized if enabled
				if tt.options.EnableMetrics && db.metrics == nil {
					t.Error("NewDB() metrics not initialized")
				}

				// Check persistence manager initialized if enabled
				if tt.options.EnablePersistence && db.persistenceManager == nil {
					t.Error("NewDB() persistence manager not initialized")
				}
			}
		})
	}
}

func TestDB_CreateCollection(t *testing.T) {
	// Create a temporary directory for test storage
	tempDir, err := os.MkdirTemp("", "quiver-test")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Create DB with minimal options
	db, err := NewDB(DBOptions{
		StoragePath:       tempDir,
		EnableMetrics:     false,
		EnablePersistence: true,
		FlushInterval:     time.Millisecond * 100,
		DefaultHNSWConfig: hnsw.Config{
			M:              16,
			EfConstruction: 100,
			DistanceFunc:   hnsw.CosineDistanceFunc,
		},
	})
	if err != nil {
		t.Fatalf("Failed to create DB: %v", err)
	}

	t.Run("Create New Collection", func(t *testing.T) {
		// Create a new collection
		collection, err := db.CreateCollection("test_collection", 128, vectortypes.GetSurfaceByType(vectortypes.Cosine))

		// Check error
		if err != nil {
			t.Fatalf("CreateCollection() error = %v", err)
		}

		// Check collection
		if collection == nil {
			t.Fatal("CreateCollection() returned nil collection without error")
		}

		if collection.Name != "test_collection" {
			t.Errorf("CreateCollection() Name = %v, want %v", collection.Name, "test_collection")
		}

		if collection.Dimension != 128 {
			t.Errorf("CreateCollection() Dimension = %v, want %v", collection.Dimension, 128)
		}

		// Check collection added to DB
		if _, exists := db.collections["test_collection"]; !exists {
			t.Error("CreateCollection() didn't add collection to DB")
		}

		// Check persistent storage created
		if db.options.EnablePersistence {
			collectionPath := filepath.Join(tempDir, "test_collection")
			if _, err := os.Stat(collectionPath); os.IsNotExist(err) {
				t.Errorf("CreateCollection() didn't create collection directory: %v", err)
			}
		}
	})

	t.Run("Create Duplicate Collection", func(t *testing.T) {
		// Try to create a collection with the same name
		_, err := db.CreateCollection("test_collection", 64, vectortypes.GetSurfaceByType(vectortypes.Euclidean))

		// Should return error
		if err == nil {
			t.Error("CreateCollection() should return error for duplicate collection")
		}

		// Error should contain ErrCollectionExists
		if !errors.Is(err, ErrCollectionExists) {
			t.Errorf("CreateCollection() error = %v, want error containing %v", err, ErrCollectionExists)
		}
	})

	t.Run("Create Collection with Different Distance Functions", func(t *testing.T) {
		tests := []struct {
			name           string
			distanceType   vectortypes.DistanceType
			wantErr        bool
			checkFieldName string
		}{
			{"Cosine", vectortypes.Cosine, false, ""},
			{"Euclidean", vectortypes.Euclidean, false, ""},
			{"DotProduct", vectortypes.DotProduct, false, ""},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				collName := "collection_" + string(tt.distanceType)
				collection, err := db.CreateCollection(collName, 32, vectortypes.GetSurfaceByType(tt.distanceType))

				if (err != nil) != tt.wantErr {
					t.Errorf("CreateCollection() error = %v, wantErr %v", err, tt.wantErr)
					return
				}

				if !tt.wantErr && collection == nil {
					t.Fatal("CreateCollection() returned nil collection without error")
				}
			})
		}
	})
}

func TestDB_GetCollection(t *testing.T) {
	// Create a temporary directory for test storage
	tempDir, err := os.MkdirTemp("", "quiver-test")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Create DB
	db, err := NewDB(DBOptions{
		StoragePath:       tempDir,
		EnableMetrics:     false,
		EnablePersistence: false,
	})
	if err != nil {
		t.Fatalf("Failed to create DB: %v", err)
	}

	// Create a test collection
	_, err = db.CreateCollection("test_collection", 128, vectortypes.GetSurfaceByType(vectortypes.Cosine))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	t.Run("Get Existing Collection", func(t *testing.T) {
		// Get the collection
		collection, err := db.GetCollection("test_collection")

		// Check error
		if err != nil {
			t.Fatalf("GetCollection() error = %v", err)
		}

		// Check collection
		if collection == nil {
			t.Fatal("GetCollection() returned nil collection without error")
		}

		if collection.Name != "test_collection" {
			t.Errorf("GetCollection() Name = %v, want %v", collection.Name, "test_collection")
		}
	})

	t.Run("Get Non-existent Collection", func(t *testing.T) {
		// Try to get a non-existent collection
		_, err := db.GetCollection("non_existent")

		// Should return error
		if err == nil {
			t.Error("GetCollection() should return error for non-existent collection")
		}

		// Error should contain ErrCollectionNotFound
		if !errors.Is(err, ErrCollectionNotFound) {
			t.Errorf("GetCollection() error = %v, want error containing %v", err, ErrCollectionNotFound)
		}
	})
}

func TestDB_DeleteCollection(t *testing.T) {
	// Create a temporary directory for test storage
	tempDir, err := os.MkdirTemp("", "quiver-test")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Create DB
	db, err := NewDB(DBOptions{
		StoragePath:       tempDir,
		EnableMetrics:     false,
		EnablePersistence: true,
	})
	if err != nil {
		t.Fatalf("Failed to create DB: %v", err)
	}

	// Create a test collection
	_, err = db.CreateCollection("test_collection", 128, vectortypes.GetSurfaceByType(vectortypes.Cosine))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	t.Run("Delete Existing Collection", func(t *testing.T) {
		// Delete the collection
		err := db.DeleteCollection("test_collection")

		// Check error
		if err != nil {
			t.Fatalf("DeleteCollection() error = %v", err)
		}

		// Check collection removed from DB
		if _, exists := db.collections["test_collection"]; exists {
			t.Error("DeleteCollection() didn't remove collection from DB")
		}

		// Check persistent storage deleted
		if db.options.EnablePersistence {
			collectionPath := filepath.Join(tempDir, "test_collection")
			if _, err := os.Stat(collectionPath); !os.IsNotExist(err) {
				t.Errorf("DeleteCollection() didn't delete collection directory: %v", err)
			}
		}
	})

	t.Run("Delete Non-existent Collection", func(t *testing.T) {
		// Try to delete a non-existent collection
		err := db.DeleteCollection("non_existent")

		// Should return error
		if err == nil {
			t.Error("DeleteCollection() should return error for non-existent collection")
		}

		// Error should contain ErrCollectionNotFound
		if !errors.Is(err, ErrCollectionNotFound) {
			t.Errorf("DeleteCollection() error = %v, want error containing %v", err, ErrCollectionNotFound)
		}
	})
}

func TestDB_ListCollections(t *testing.T) {
	// Create a temporary directory for test storage
	tempDir, err := os.MkdirTemp("", "quiver-test")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Create DB
	db, err := NewDB(DBOptions{
		StoragePath:       tempDir,
		EnableMetrics:     false,
		EnablePersistence: false,
	})
	if err != nil {
		t.Fatalf("Failed to create DB: %v", err)
	}

	// Initial list should be empty
	collections := db.ListCollections()
	if len(collections) != 0 {
		t.Errorf("ListCollections() returned %v collections, want 0", len(collections))
	}

	// Create some test collections
	collectionNames := []string{"collection1", "collection2", "collection3"}
	for _, name := range collectionNames {
		_, err := db.CreateCollection(name, 128, vectortypes.GetSurfaceByType(vectortypes.Cosine))
		if err != nil {
			t.Fatalf("Failed to create collection %s: %v", name, err)
		}
	}

	// List collections
	collections = db.ListCollections()

	// Check number of collections
	if len(collections) != len(collectionNames) {
		t.Errorf("ListCollections() returned %v collections, want %v", len(collections), len(collectionNames))
	}

	// Check collection names
	collectionMap := make(map[string]bool)
	for _, name := range collections {
		collectionMap[name] = true
	}

	for _, name := range collectionNames {
		if !collectionMap[name] {
			t.Errorf("ListCollections() missing collection %s", name)
		}
	}
}

func TestDB_BackupRestore(t *testing.T) {
	// Skip if testing.Short()
	if testing.Short() {
		t.Skip("Skipping backup/restore test in short mode")
	}

	// Create a temporary directory for test storage
	tempDir, err := os.MkdirTemp("", "quiver-test")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Create a temporary directory for backup
	backupDir, err := os.MkdirTemp("", "quiver-backup")
	if err != nil {
		t.Fatalf("Failed to create backup directory: %v", err)
	}
	defer os.RemoveAll(backupDir)

	// Create DB
	db, err := NewDB(DBOptions{
		StoragePath:       tempDir,
		EnableMetrics:     false,
		EnablePersistence: true,
		FlushInterval:     time.Millisecond * 100,
	})
	if err != nil {
		t.Fatalf("Failed to create DB: %v", err)
	}

	// Create a test collection
	collection, err := db.CreateCollection("test_collection", 3, vectortypes.GetSurfaceByType(vectortypes.Cosine))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Add some vectors
	vectors := []struct {
		id       string
		vector   []float32
		metadata json.RawMessage
	}{
		{"id1", []float32{1.0, 0.0, 0.0}, json.RawMessage(`{"key": "value1"}`)},
		{"id2", []float32{0.0, 1.0, 0.0}, json.RawMessage(`{"key": "value2"}`)},
		{"id3", []float32{0.0, 0.0, 1.0}, json.RawMessage(`{"key": "value3"}`)},
	}

	for _, v := range vectors {
		err := collection.Add(v.id, v.vector, v.metadata)
		if err != nil {
			t.Fatalf("Failed to add vector: %v", err)
		}
	}

	// Wait for flush to complete
	time.Sleep(time.Millisecond * 200)

	// Create backup
	err = db.BackupDatabase(backupDir)
	if err != nil {
		t.Fatalf("BackupDatabase() error = %v", err)
	}

	// Create a new DB with same options
	newDB, err := NewDB(DBOptions{
		StoragePath:       filepath.Join(tempDir, "new"),
		EnableMetrics:     false,
		EnablePersistence: true,
		FlushInterval:     time.Millisecond * 100,
	})
	if err != nil {
		t.Fatalf("Failed to create new DB: %v", err)
	}

	// Restore from backup
	err = newDB.RestoreDatabase(backupDir)
	if err != nil {
		t.Fatalf("RestoreDatabase() error = %v", err)
	}

	// Check collection restored
	restoredCollection, err := newDB.GetCollection("test_collection")
	if err != nil {
		t.Fatalf("Failed to get restored collection: %v", err)
	}

	// Check collection properties
	if restoredCollection.Name != "test_collection" {
		t.Errorf("Restored collection Name = %v, want %v", restoredCollection.Name, "test_collection")
	}

	if restoredCollection.Dimension != 3 {
		t.Errorf("Restored collection Dimension = %v, want %v", restoredCollection.Dimension, 3)
	}

	// Check vectors restored
	for _, v := range vectors {
		result, err := restoredCollection.Get(v.id)
		if err != nil {
			t.Fatalf("Failed to get vector %s from restored collection: %v", v.id, err)
		}

		if result.ID != v.id {
			t.Errorf("Restored vector ID = %v, want %v", result.ID, v.id)
		}

		if !vectorEqual(result.Values, v.vector) {
			t.Errorf("Restored vector Values = %v, want %v", result.Values, v.vector)
		}
	}

	// Test search on restored collection
	searchRequest := types.SearchRequest{
		Vector: []float32{1.0, 0.0, 0.0},
		TopK:   2,
		Options: types.SearchOptions{
			IncludeVectors:  true,
			IncludeMetadata: true,
		},
	}

	searchResponse, err := restoredCollection.Search(searchRequest)
	if err != nil {
		t.Fatalf("Search on restored collection error = %v", err)
	}

	if len(searchResponse.Results) != 2 {
		t.Errorf("Search on restored collection returned %v results, want %v", len(searchResponse.Results), 2)
	}
}

func TestHybridIndexWrapper(t *testing.T) {
	// Create a hybrid index with default config
	config := hybrid.DefaultIndexConfig()
	hybridIndex := hybrid.NewHybridIndex(config)

	// Create wrapper
	wrapper := &HybridIndexWrapper{
		hybridIndex: hybridIndex,
	}

	// Test Insert
	err := wrapper.Insert("id1", []float32{1.0, 0.0, 0.0})
	if err != nil {
		t.Fatalf("Insert() error = %v", err)
	}

	// Test Size
	if wrapper.Size() != 1 {
		t.Errorf("Size() = %v, want %v", wrapper.Size(), 1)
	}

	// Test Search
	results, err := wrapper.Search([]float32{1.0, 0.0, 0.0}, 1)
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(results) != 1 {
		t.Errorf("Search() returned %v results, want %v", len(results), 1)
	}
	if results[0].ID != "id1" {
		t.Errorf("Search() result ID = %v, want %v", results[0].ID, "id1")
	}

	// Test Delete
	err = wrapper.Delete("id1")
	if err != nil {
		t.Fatalf("Delete() error = %v", err)
	}

	// Check size after delete
	if wrapper.Size() != 0 {
		t.Errorf("Size() after delete = %v, want %v", wrapper.Size(), 0)
	}
}

func TestDB_SetHybridConfig(t *testing.T) {
	// Create DB
	db, err := NewDB(DBOptions{
		StoragePath:       "./test",
		EnableMetrics:     false,
		EnablePersistence: false,
		HybridConfig:      hybrid.DefaultIndexConfig(),
	})
	if err != nil {
		t.Fatalf("Failed to create DB: %v", err)
	}

	// Get initial config
	_ = db.GetHybridConfig() // Just to verify it works

	// Set new config
	newConfig := hybrid.DefaultIndexConfig()
	// Modify config as needed for testing
	newConfig.HNSWConfig.M = 32
	newConfig.HNSWConfig.EfConstruction = 200
	db.SetHybridConfig(newConfig)

	// Get updated config
	updatedConfig := db.GetHybridConfig()
	if updatedConfig.HNSWConfig.M != newConfig.HNSWConfig.M {
		t.Errorf("Updated config HNSWConfig.M = %v, want %v", updatedConfig.HNSWConfig.M, newConfig.HNSWConfig.M)
	}
	if updatedConfig.HNSWConfig.EfConstruction != newConfig.HNSWConfig.EfConstruction {
		t.Errorf("Updated config HNSWConfig.EfConstruction = %v, want %v",
			updatedConfig.HNSWConfig.EfConstruction, newConfig.HNSWConfig.EfConstruction)
	}
}

// Helper function to check if two vectors are equal
func vectorEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
