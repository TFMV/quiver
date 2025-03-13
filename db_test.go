package db

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/TFMV/hnsw"
	"github.com/TFMV/hnsw/hnsw-extensions/facets"
	"github.com/TFMV/hnsw/hnsw-extensions/hybrid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestVectorDBBasicOperations tests basic CRUD operations
func TestVectorDBBasicOperations(t *testing.T) {
	// Create a temporary directory for the test
	tempDir, err := os.MkdirTemp("", "vectordb-test")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)

	// Create a database configuration
	config := DefaultDBConfig()
	config.BaseDir = tempDir
	config.Hybrid.Type = hybrid.HybridIndexType
	config.Hybrid.Distance = hnsw.CosineDistance

	// Create a vector database
	db, err := NewVectorDB[string](config)
	require.NoError(t, err)
	defer db.Close()

	// Test Add
	t.Run("Add", func(t *testing.T) {
		vector := []float32{0.1, 0.2, 0.3}
		metadata := map[string]string{"name": "test1"}
		facetList := []facets.Facet{
			facets.NewBasicFacet("category", "test"),
			facets.NewBasicFacet("score", 0.95),
		}

		err := db.Add("key1", vector, metadata, facetList)
		assert.NoError(t, err)

		// Verify stats
		stats := db.GetStats()
		assert.Equal(t, 1, stats.VectorCount)
	})

	// Test Get (via Search)
	t.Run("Search", func(t *testing.T) {
		query := []float32{0.1, 0.2, 0.3}
		options := db.DefaultQueryOptions().WithK(1)

		results, err := db.Search(query, options)
		assert.NoError(t, err)
		assert.Len(t, results, 1)
		assert.Equal(t, "key1", results[0].Key)

		// Verify metadata
		var metadata map[string]string
		err = json.Unmarshal(results[0].Metadata, &metadata)
		assert.NoError(t, err)
		assert.Equal(t, "test1", metadata["name"])

		// Verify facets
		assert.Len(t, results[0].Facets, 2)
		assert.Equal(t, "category", results[0].Facets[0].Name())
		assert.Equal(t, "test", results[0].Facets[0].Value())
	})

	// Test Delete
	t.Run("Delete", func(t *testing.T) {
		deleted := db.Delete("key1")
		assert.True(t, deleted)

		// Verify stats
		stats := db.GetStats()
		assert.Equal(t, 0, stats.VectorCount)

		// Verify deletion via search
		query := []float32{0.1, 0.2, 0.3}
		options := db.DefaultQueryOptions().WithK(1)

		results, err := db.Search(query, options)
		assert.NoError(t, err)
		assert.Len(t, results, 0)
	})
}

// TestVectorDBBatchOperations tests batch operations
func TestVectorDBBatchOperations(t *testing.T) {
	// Create a temporary directory for the test
	tempDir, err := os.MkdirTemp("", "vectordb-test")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)

	// Create a database configuration
	config := DefaultDBConfig()
	config.BaseDir = tempDir
	config.Hybrid.Type = hybrid.HybridIndexType
	config.Hybrid.Distance = hnsw.CosineDistance

	// Create a vector database
	db, err := NewVectorDB[string](config)
	require.NoError(t, err)
	defer db.Close()

	// Test BatchAdd
	t.Run("BatchAdd", func(t *testing.T) {
		keys := []string{"key1", "key2", "key3"}
		vectors := [][]float32{
			{0.1, 0.2, 0.3},
			{0.2, 0.3, 0.4},
			{0.3, 0.4, 0.5},
		}
		metadataList := []interface{}{
			map[string]string{"name": "test1"},
			map[string]string{"name": "test2"},
			map[string]string{"name": "test3"},
		}
		facetsList := [][]facets.Facet{
			{facets.NewBasicFacet("category", "test1")},
			{facets.NewBasicFacet("category", "test2")},
			{facets.NewBasicFacet("category", "test3")},
		}

		err := db.BatchAdd(keys, vectors, metadataList, facetsList)
		assert.NoError(t, err)

		// Verify stats
		stats := db.GetStats()
		assert.Equal(t, 3, stats.VectorCount)
	})

	// Test BatchDelete
	t.Run("BatchDelete", func(t *testing.T) {
		keys := []string{"key1", "key3"}
		results := db.BatchDelete(keys)
		assert.Len(t, results, 2)
		assert.True(t, results[0])
		assert.True(t, results[1])

		// Verify stats
		stats := db.GetStats()
		assert.Equal(t, 1, stats.VectorCount)

		// Verify remaining key via search
		query := []float32{0.2, 0.3, 0.4}
		options := db.DefaultQueryOptions().WithK(1)

		searchResults, err := db.Search(query, options)
		assert.NoError(t, err)
		assert.Len(t, searchResults, 1)
		assert.Equal(t, "key2", searchResults[0].Key)
	})

	// Test OptimizeStorage
	t.Run("OptimizeStorage", func(t *testing.T) {
		err := db.OptimizeStorage()
		assert.NoError(t, err)
	})
}

// TestVectorDBSearch tests search functionality
func TestVectorDBSearch(t *testing.T) {
	// Create a temporary directory for the test
	tempDir, err := os.MkdirTemp("", "vectordb-test")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)

	// Create a database configuration
	config := DefaultDBConfig()
	config.BaseDir = tempDir
	config.Hybrid.Type = hybrid.HybridIndexType
	config.Hybrid.Distance = hnsw.CosineDistance

	// Create a vector database
	db, err := NewVectorDB[string](config)
	require.NoError(t, err)
	defer db.Close()

	// Add test vectors
	keys := []string{"key1", "key2", "key3", "key4", "key5"}
	vectors := [][]float32{
		{0.1, 0.2, 0.3},
		{0.2, 0.3, 0.4},
		{0.3, 0.4, 0.5},
		{0.4, 0.5, 0.6},
		{0.5, 0.6, 0.7},
	}
	metadataList := []interface{}{
		map[string]string{"name": "test1", "category": "finance"},
		map[string]string{"name": "test2", "category": "technology"},
		map[string]string{"name": "test3", "category": "finance"},
		map[string]string{"name": "test4", "category": "healthcare"},
		map[string]string{"name": "test5", "category": "technology"},
	}
	facetsList := [][]facets.Facet{
		{facets.NewBasicFacet("category", "finance"), facets.NewBasicFacet("score", 0.9)},
		{facets.NewBasicFacet("category", "technology"), facets.NewBasicFacet("score", 0.8)},
		{facets.NewBasicFacet("category", "finance"), facets.NewBasicFacet("score", 0.7)},
		{facets.NewBasicFacet("category", "healthcare"), facets.NewBasicFacet("score", 0.6)},
		{facets.NewBasicFacet("category", "technology"), facets.NewBasicFacet("score", 0.5)},
	}

	err = db.BatchAdd(keys, vectors, metadataList, facetsList)
	require.NoError(t, err)

	// Test basic search
	t.Run("BasicSearch", func(t *testing.T) {
		query := []float32{0.1, 0.2, 0.3}
		options := db.DefaultQueryOptions().WithK(3)

		results, err := db.Search(query, options)
		assert.NoError(t, err)
		assert.Len(t, results, 3)
		assert.Equal(t, "key1", results[0].Key) // Closest match should be first
	})

	// Test search with facet filters
	t.Run("SearchWithFacetFilters", func(t *testing.T) {
		query := []float32{0.1, 0.2, 0.3}
		options := db.DefaultQueryOptions().
			WithK(5).
			WithFacetFilters(facets.NewEqualityFilter("category", "technology"))

		results, err := db.Search(query, options)
		assert.NoError(t, err)
		assert.Len(t, results, 2)

		// Verify all results have technology category
		for _, result := range results {
			found := false
			for _, facet := range result.Facets {
				if facet.Name() == "category" && facet.Value() == "technology" {
					found = true
					break
				}
			}
			assert.True(t, found, "Result should have technology category")
		}
	})

	// Test search with negative examples
	t.Run("SearchWithNegativeExamples", func(t *testing.T) {
		query := []float32{0.3, 0.4, 0.5}
		negative := []float32{0.5, 0.6, 0.7} // This should push away from key5
		options := db.DefaultQueryOptions().
			WithK(3).
			WithNegativeExample(negative).
			WithNegativeWeight(0.5)

		results, err := db.Search(query, options)
		assert.NoError(t, err)
		assert.Len(t, results, 3)

		// The last result should not be key5 due to negative example
		for _, result := range results {
			assert.NotEqual(t, "key5", result.Key, "key5 should be pushed away by negative example")
		}
	})

	// Test fluent API
	t.Run("FluentAPI", func(t *testing.T) {
		query := []float32{0.3, 0.4, 0.5}
		options := db.DefaultQueryOptions().
			WithK(2).
			WithFacetFilters(facets.NewEqualityFilter("category", "finance")).
			WithNegativeExample([]float32{0.1, 0.2, 0.3}).
			WithNegativeWeight(0.3)

		results, err := db.Search(query, options)
		assert.NoError(t, err)
		assert.LessOrEqual(t, len(results), 2)

		// Verify all results have finance category
		for _, result := range results {
			found := false
			for _, facet := range result.Facets {
				if facet.Name() == "category" && facet.Value() == "finance" {
					found = true
					break
				}
			}
			assert.True(t, found, "Result should have finance category")
		}
	})
}

// TestVectorDBBackupRestore tests backup and restore functionality
func TestVectorDBBackupRestore(t *testing.T) {
	// Create temporary directories for the test
	dbDir, err := os.MkdirTemp("", "vectordb-test")
	require.NoError(t, err)
	defer os.RemoveAll(dbDir)

	backupDir, err := os.MkdirTemp("", "vectordb-backup")
	require.NoError(t, err)
	defer os.RemoveAll(backupDir)

	restoreDir, err := os.MkdirTemp("", "vectordb-restore")
	require.NoError(t, err)
	defer os.RemoveAll(restoreDir)

	// Create a database configuration
	config := DefaultDBConfig()
	config.BaseDir = dbDir
	config.Hybrid.Type = hybrid.HybridIndexType
	config.Hybrid.Distance = hnsw.CosineDistance

	// Create a vector database
	db, err := NewVectorDB[string](config)
	require.NoError(t, err)
	defer db.Close()

	// Add test vectors
	keys := []string{"key1", "key2", "key3"}
	vectors := [][]float32{
		{0.1, 0.2, 0.3},
		{0.2, 0.3, 0.4},
		{0.3, 0.4, 0.5},
	}
	metadataList := []interface{}{
		map[string]string{"name": "test1"},
		map[string]string{"name": "test2"},
		map[string]string{"name": "test3"},
	}
	facetsList := [][]facets.Facet{
		{facets.NewBasicFacet("category", "test1")},
		{facets.NewBasicFacet("category", "test2")},
		{facets.NewBasicFacet("category", "test3")},
	}

	err = db.BatchAdd(keys, vectors, metadataList, facetsList)
	require.NoError(t, err)

	// Test Backup
	t.Run("Backup", func(t *testing.T) {
		err := db.Backup(backupDir)
		assert.NoError(t, err)

		// Verify backup files exist
		assert.FileExists(t, filepath.Join(backupDir, "config.json"))
		assert.FileExists(t, filepath.Join(backupDir, "stats.json"))
		assert.FileExists(t, filepath.Join(backupDir, "metadata.json"))
		assert.FileExists(t, filepath.Join(backupDir, "facets.json"))
		assert.DirExists(t, filepath.Join(backupDir, "vectors"))
	})

	// Test Restore
	t.Run("Restore", func(t *testing.T) {
		// Create a new database for restoration
		restoreConfig := DefaultDBConfig()
		restoreConfig.BaseDir = restoreDir
		restoreConfig.Hybrid.Type = hybrid.HybridIndexType
		restoreConfig.Hybrid.Distance = hnsw.CosineDistance

		restoredDB, err := NewVectorDB[string](restoreConfig)
		require.NoError(t, err)
		defer restoredDB.Close()

		// Restore from backup
		err = restoredDB.Restore(backupDir)
		assert.NoError(t, err)

		// Verify stats
		stats := restoredDB.GetStats()
		assert.Equal(t, 3, stats.VectorCount)

		// Verify data via search
		query := []float32{0.1, 0.2, 0.3}
		options := restoredDB.DefaultQueryOptions().WithK(1)

		results, err := restoredDB.Search(query, options)
		assert.NoError(t, err)
		assert.Len(t, results, 1)
		assert.Equal(t, "key1", results[0].Key)

		// Verify metadata
		var metadata map[string]string
		err = json.Unmarshal(results[0].Metadata, &metadata)
		assert.NoError(t, err)
		assert.Equal(t, "test1", metadata["name"])

		// Verify facets
		assert.Len(t, results[0].Facets, 1)
		assert.Equal(t, "category", results[0].Facets[0].Name())
		assert.Equal(t, "test1", results[0].Facets[0].Value())
	})
}

// TestVectorDBConcurrency tests concurrent operations
func TestVectorDBConcurrency(t *testing.T) {
	// Create a temporary directory for the test
	tempDir, err := os.MkdirTemp("", "vectordb-test")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)

	// Create a database configuration
	config := DefaultDBConfig()
	config.BaseDir = tempDir
	config.Hybrid.Type = hybrid.HybridIndexType
	config.Hybrid.Distance = hnsw.CosineDistance
	config.MaxConcurrentQueries = 10

	// Create a vector database
	db, err := NewVectorDB[string](config)
	require.NoError(t, err)
	defer db.Close()

	// Add test vectors
	for i := 0; i < 100; i++ {
		key := "key" + string(rune(i))
		vector := []float32{float32(i) * 0.01, float32(i) * 0.02, float32(i) * 0.03}
		metadata := map[string]string{"name": "test" + string(rune(i))}
		facetList := []facets.Facet{
			facets.NewBasicFacet("category", "test"),
		}

		err := db.Add(key, vector, metadata, facetList)
		require.NoError(t, err)
	}

	// Test concurrent searches
	t.Run("ConcurrentSearches", func(t *testing.T) {
		const numGoroutines = 10
		const numSearches = 10

		done := make(chan bool)
		errors := make(chan error, numGoroutines*numSearches)

		for i := 0; i < numGoroutines; i++ {
			go func(id int) {
				for j := 0; j < numSearches; j++ {
					query := []float32{float32(id) * 0.01, float32(id) * 0.02, float32(id) * 0.03}
					options := db.DefaultQueryOptions().WithK(5)

					_, err := db.Search(query, options)
					if err != nil {
						errors <- err
					}

					// Small delay to simulate real-world usage
					time.Sleep(time.Millisecond * 10)
				}
				done <- true
			}(i)
		}

		// Wait for all goroutines to complete
		for i := 0; i < numGoroutines; i++ {
			<-done
		}

		// Check for errors
		close(errors)
		for err := range errors {
			t.Errorf("Concurrent search error: %v", err)
		}

		// Verify stats
		stats := db.GetStats()
		assert.Equal(t, int64(numGoroutines*numSearches), stats.TotalQueries)
		assert.Equal(t, int64(numGoroutines*numSearches), stats.SuccessfulQueries)
		assert.Equal(t, int64(0), stats.FailedQueries)
	})
}

// TestVectorDBDifferentKeyTypes tests using different key types
func TestVectorDBDifferentKeyTypes(t *testing.T) {
	// Test with int keys
	t.Run("IntKeys", func(t *testing.T) {
		// Create a temporary directory for the test
		tempDir, err := os.MkdirTemp("", "vectordb-test-int")
		require.NoError(t, err)
		defer os.RemoveAll(tempDir)

		// Create a database configuration
		config := DefaultDBConfig()
		config.BaseDir = tempDir
		config.Hybrid.Type = hybrid.HybridIndexType
		config.Hybrid.Distance = hnsw.CosineDistance

		// Create a vector database with int keys
		db, err := NewVectorDB[int](config)
		require.NoError(t, err)
		defer db.Close()

		// Add a vector
		vector := []float32{0.1, 0.2, 0.3}
		metadata := map[string]string{"name": "test1"}
		facetList := []facets.Facet{
			facets.NewBasicFacet("category", "test"),
		}

		err = db.Add(123, vector, metadata, facetList)
		assert.NoError(t, err)

		// Search for the vector
		query := []float32{0.1, 0.2, 0.3}
		options := db.DefaultQueryOptions().WithK(1)

		results, err := db.Search(query, options)
		assert.NoError(t, err)
		assert.Len(t, results, 1)
		assert.Equal(t, 123, results[0].Key)
	})
}

// TestVectorDBEdgeCases tests edge cases
func TestVectorDBEdgeCases(t *testing.T) {
	// Create a temporary directory for the test
	tempDir, err := os.MkdirTemp("", "vectordb-test")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)

	// Create a database configuration
	config := DefaultDBConfig()
	config.BaseDir = tempDir
	config.Hybrid.Type = hybrid.HybridIndexType
	config.Hybrid.Distance = hnsw.CosineDistance

	// Create a vector database
	db, err := NewVectorDB[string](config)
	require.NoError(t, err)
	defer db.Close()

	// Test empty batch add
	t.Run("EmptyBatchAdd", func(t *testing.T) {
		err := db.BatchAdd([]string{}, [][]float32{}, []interface{}{}, [][]facets.Facet{})
		assert.Error(t, err) // Should error due to empty keys and vectors
	})

	// Test deleting non-existent key
	t.Run("DeleteNonExistentKey", func(t *testing.T) {
		deleted := db.Delete("non-existent-key")
		assert.False(t, deleted)
	})

	// Test search with zero K
	t.Run("SearchWithZeroK", func(t *testing.T) {
		query := []float32{0.1, 0.2, 0.3}
		options := db.DefaultQueryOptions().WithK(0)

		results, err := db.Search(query, options)
		assert.NoError(t, err)
		assert.Len(t, results, 0)
	})

	// Test backup to non-existent directory
	t.Run("BackupToNonExistentDir", func(t *testing.T) {
		nonExistentDir := filepath.Join(tempDir, "non-existent-dir")
		err := db.Backup(nonExistentDir)
		assert.NoError(t, err) // Should create the directory

		// Verify directory was created
		_, err = os.Stat(nonExistentDir)
		assert.NoError(t, err)
	})

	// Test restore from non-existent directory
	t.Run("RestoreFromNonExistentDir", func(t *testing.T) {
		nonExistentDir := filepath.Join(tempDir, "another-non-existent-dir")
		err := db.Restore(nonExistentDir)
		assert.Error(t, err) // Should fail
	})
}
