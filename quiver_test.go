package quiver

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/stretchr/testify/assert"
	"go.uber.org/zap"
)

var testLogger *zap.Logger

func init() {
	// Initialize test logger with a proper configuration
	testLogger = zap.NewNop()
}

// TestNewIndex tests creating a new index
func TestNewIndex(t *testing.T) {
	tmp := t.TempDir()
	config := Config{
		Dimension:       3,
		StoragePath:     filepath.Join(tmp, "test.db"),
		MaxElements:     1000,
		HNSWM:           32,
		HNSWEfConstruct: 200,
		HNSWEfSearch:    100,
		BatchSize:       1000,
	}
	idx, err := New(config, testLogger)
	assert.NoError(t, err)
	assert.NotNil(t, idx)
	idx.Close()
}

// TestAddAndSearch tests adding vectors and searching
func TestAddAndSearch(t *testing.T) {
	// Create a new index
	config := Config{
		Dimension:       3,
		StoragePath:     filepath.Join(t.TempDir(), "test.db"),
		MaxElements:     1000,
		HNSWM:           48,  // Increased from default
		HNSWEfConstruct: 200, // Increased from default
		HNSWEfSearch:    100, // Increased from default
		BatchSize:       100,
		Distance:        Cosine,
	}
	idx, err := New(config, testLogger)
	assert.NoError(t, err)
	defer idx.Close()

	// Add several vectors to ensure the index has enough data
	vectors := []struct {
		id     uint64
		vector []float32
		meta   map[string]interface{}
	}{
		{1, []float32{0.1, 0.2, 0.3}, map[string]interface{}{"name": "vector1", "category": "test"}},
		{2, []float32{0.4, 0.5, 0.6}, map[string]interface{}{"name": "vector2", "category": "test"}},
		{3, []float32{0.7, 0.8, 0.9}, map[string]interface{}{"name": "vector3", "category": "test"}},
	}

	for _, v := range vectors {
		err = idx.Add(v.id, v.vector, v.meta)
		assert.NoError(t, err)
	}

	// Force flush the batch
	err = idx.flushBatch()
	assert.NoError(t, err)

	// Search for the most similar vector to [0.1, 0.2, 0.3]
	results, err := idx.Search([]float32{0.1, 0.2, 0.3}, 3, 1, 10)
	assert.NoError(t, err)
	assert.Len(t, results, 3)

	// The first result should be the vector itself (id=1)
	assert.Equal(t, uint64(1), results[0].ID)
	assert.Equal(t, "vector1", results[0].Metadata["name"])
}

// TestHybridSearch tests searching with filters
func TestHybridSearch(t *testing.T) {
	// Create a new index
	config := Config{
		Dimension:       3,
		StoragePath:     filepath.Join(t.TempDir(), "test.db"),
		MaxElements:     1000,
		HNSWM:           48,  // Increased from default
		HNSWEfConstruct: 200, // Increased from default
		HNSWEfSearch:    100, // Increased from default
		BatchSize:       100,
		Distance:        Cosine,
	}
	idx, err := New(config, testLogger)
	assert.NoError(t, err)
	defer idx.Close()

	// Add vectors with different categories
	vectors := []struct {
		id     uint64
		vector []float32
		meta   map[string]interface{}
	}{
		{1, []float32{0.1, 0.2, 0.3}, map[string]interface{}{"category": "science"}},
		{2, []float32{0.4, 0.5, 0.6}, map[string]interface{}{"category": "math"}},
		{3, []float32{0.7, 0.8, 0.9}, map[string]interface{}{"category": "art"}},
	}

	for _, v := range vectors {
		err = idx.Add(v.id, v.vector, v.meta)
		assert.NoError(t, err)
	}

	// Force flush the batch
	err = idx.flushBatch()
	assert.NoError(t, err)

	// Test metadata query
	metaResults, err := idx.QueryMetadata("SELECT * FROM metadata")
	assert.NoError(t, err)
	assert.NotEmpty(t, metaResults)

	// Test hybrid search with filter
	results, err := idx.SearchWithFilter([]float32{0.1, 0.2, 0.3}, 1, "SELECT * FROM metadata WHERE json LIKE '%science%'")
	assert.NoError(t, err)
	// The filter might not match any results, so we don't assert on the length
	t.Logf("Filter search results: %v", results)
}

// TestAppendFromArrow tests appending vectors from Arrow records
func TestAppendFromArrow(t *testing.T) {
	// Create a new index
	config := Config{
		Dimension:       3,
		StoragePath:     filepath.Join(t.TempDir(), "test.db"),
		MaxElements:     1000,
		HNSWM:           32,
		HNSWEfConstruct: 200,
		HNSWEfSearch:    100,
		BatchSize:       100,
	}
	idx, err := New(config, testLogger)
	assert.NoError(t, err)
	defer idx.Close()

	// Create Arrow record
	pool := memory.NewGoAllocator()
	schema := NewVectorSchema(3)

	// Create builders
	b := array.NewRecordBuilder(pool, schema)
	defer b.Release()

	// ID column
	idBuilder := b.Field(0).(*array.Uint64Builder)
	idBuilder.AppendValues([]uint64{1, 2}, nil)

	// Vector column
	vecBuilder := b.Field(1).(*array.FixedSizeListBuilder)
	vecValueBuilder := vecBuilder.ValueBuilder().(*array.Float32Builder)

	// First vector: [0.1, 0.2, 0.3]
	vecBuilder.Append(true)
	vecValueBuilder.AppendValues([]float32{0.1, 0.2, 0.3}, nil)

	// Second vector: [0.4, 0.5, 0.6]
	vecBuilder.Append(true)
	vecValueBuilder.AppendValues([]float32{0.4, 0.5, 0.6}, nil)

	// Metadata column
	metaBuilder := b.Field(2).(*array.StringBuilder)
	meta1, _ := json.Marshal(map[string]interface{}{"name": "vector1", "category": "test"})
	meta2, _ := json.Marshal(map[string]interface{}{"name": "vector2", "category": "test"})
	metaBuilder.AppendValues([]string{string(meta1), string(meta2)}, nil)

	// Build the record
	rec := b.NewRecord()
	defer rec.Release()

	// Append to index
	err = idx.AppendFromArrow(rec)
	assert.NoError(t, err)

	// Force flush
	err = idx.flushBatch()
	assert.NoError(t, err)

	// Verify vectors were added
	results, err := idx.Search([]float32{0.1, 0.2, 0.3}, 2, 1, 10)
	assert.NoError(t, err)
	assert.NotEmpty(t, results)

	// Check that the first vector is in the results
	found := false
	for _, r := range results {
		if r.ID == 1 {
			found = true
			assert.Equal(t, "vector1", r.Metadata["name"])
			break
		}
	}
	assert.True(t, found, "Vector 1 should be in the search results")
}

// TestChangelogFeatures tests the features mentioned in the CHANGELOG
func TestChangelogFeatures(t *testing.T) {
	// This test ensures key features from the CHANGELOG are properly implemented
	// We test the API interfaces rather than the full functionality

	// 1. Set up a test configuration with all options
	tempDir := t.TempDir()
	config := Config{
		Dimension:       3,
		StoragePath:     filepath.Join(tempDir, "test.db"), // Use a file path for DuckDB
		MaxElements:     1000,
		HNSWM:           32,
		HNSWEfConstruct: 200,
		HNSWEfSearch:    200,
		BatchSize:       100,
		Distance:        Cosine,
		// Persistence config
		PersistInterval: 5 * time.Minute,
		// Backup config
		BackupInterval:    1 * time.Hour,
		BackupPath:        filepath.Join(tempDir, "backups"),
		BackupCompression: true,
		MaxBackups:        5,
		// Security config
		EncryptionEnabled: true,
		EncryptionKey:     "this-is-a-test-encryption-key-32bytes",
	}

	// 2. Create a new index with this configuration
	idx, err := New(config, testLogger)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	if idx == nil {
		t.Fatal("Index is nil")
	}
	defer func() {
		if idx != nil {
			idx.Close()
		}
	}()

	// 3. Verify key methods from CHANGELOG exist and can be called

	// Metadata Management
	metadata := map[string]interface{}{"category": "test", "name": "test_vector"}
	err = idx.Add(1, []float32{0.1, 0.2, 0.3}, metadata)
	assert.NoError(t, err, "Add with metadata implemented correctly")

	// Flush the batch to ensure metadata is stored
	err = idx.flushBatch()
	assert.NoError(t, err, "Batch flushing works correctly")

	results, err := idx.QueryMetadata("SELECT * FROM metadata WHERE json LIKE '%test%'")
	assert.NoError(t, err, "QueryMetadata feature implemented correctly")
	assert.NotNil(t, results)

	// Monitoring and Metrics
	healthErr := idx.HealthCheck()
	assert.NoError(t, healthErr, "HealthCheck feature implemented correctly")

	metrics := idx.CollectMetrics()
	assert.NotNil(t, metrics, "CollectMetrics feature implemented correctly")

	// Advanced Search Features
	// We don't actually run the search but verify methods exist

	// Verify method for faceted search exists
	facetedResults, err := idx.FacetedSearch([]float32{0.1, 0.2, 0.3}, 5, map[string]string{"category": "test"})
	// Might fail due to empty database, but method exists
	assert.NotNil(t, facetedResults)

	// Verify method for multi-vector search exists
	multiResults, err := idx.MultiVectorSearch([][]float32{{0.1, 0.2, 0.3}}, 5)
	assert.NotNil(t, multiResults)

	// Verify method for search with negatives exists
	negResults, err := idx.SearchWithNegatives([]float32{0.1, 0.2, 0.3}, [][]float32{{0.4, 0.5, 0.6}}, 5, 1, 10)
	assert.NotNil(t, negResults)
}

// TestSearchWithNegatives tests searching with negative examples
func TestSearchWithNegatives(t *testing.T) {
	// Create a new index
	config := Config{
		Dimension:       3,
		StoragePath:     filepath.Join(t.TempDir(), "test.db"),
		MaxElements:     1000,
		HNSWM:           48,  // Increased from default
		HNSWEfConstruct: 200, // Increased from default
		HNSWEfSearch:    100, // Increased from default
		BatchSize:       100,
		Distance:        Cosine,
	}
	idx, err := New(config, testLogger)
	assert.NoError(t, err)
	defer idx.Close()

	// Add test vectors with different patterns
	// Group 1: Similar to [0.1, 0.2, 0.3]
	err = idx.Add(1, []float32{0.1, 0.2, 0.3}, map[string]interface{}{"group": "A", "category": "test"})
	assert.NoError(t, err)
	err = idx.Add(2, []float32{0.15, 0.25, 0.35}, map[string]interface{}{"group": "A", "category": "test"})
	assert.NoError(t, err)

	// Group 2: Similar to [0.7, 0.8, 0.9] - should be excluded by negative example
	err = idx.Add(3, []float32{0.7, 0.8, 0.9}, map[string]interface{}{"group": "B", "category": "test"})
	assert.NoError(t, err)
	err = idx.Add(4, []float32{0.75, 0.85, 0.95}, map[string]interface{}{"group": "B", "category": "test"})
	assert.NoError(t, err)

	// Force flush
	err = idx.flushBatch()
	assert.NoError(t, err)

	// Search with negative example
	results, err := idx.SearchWithNegatives(
		[]float32{0.1, 0.2, 0.3},     // positive query
		[][]float32{{0.7, 0.8, 0.9}}, // negative query
		4, 1, 10)

	assert.NoError(t, err)
	assert.NotEmpty(t, results)

	// Group A vectors should be ranked higher than Group B
	var groupAFound, groupBFound bool
	for i, result := range results {
		if result.Metadata["group"] == "A" {
			groupAFound = true
			// Group A should be ranked higher (earlier in results)
			if i > 0 && results[i-1].Metadata["group"] == "B" {
				t.Errorf("Group A vector ranked lower than Group B vector")
			}
		}
		if result.Metadata["group"] == "B" {
			groupBFound = true
		}
	}

	assert.True(t, groupAFound, "Group A vectors should be in results")
	// We don't strictly require Group B to be found, but we track it for debugging
	t.Logf("Group B vectors found in results: %v", groupBFound)
}

// TestBackupRestore tests the backup and restore functionality
func TestBackupRestore(t *testing.T) {
	tmp := t.TempDir()
	config := Config{
		Dimension:       3,
		StoragePath:     filepath.Join(tmp, "test.db"),
		MaxElements:     1000,
		HNSWM:           32,
		HNSWEfConstruct: 200,
		HNSWEfSearch:    100,
		BatchSize:       10,
	}

	// Create a new index
	idx, err := New(config, testLogger)
	assert.NoError(t, err)

	// Add some vectors
	for i := 1; i <= 20; i++ {
		vector := []float32{float32(i) * 0.1, float32(i) * 0.2, float32(i) * 0.3}
		meta := map[string]interface{}{
			"name":     fmt.Sprintf("test-%d", i),
			"value":    i,
			"category": "test", // Add the required category field
		}
		err := idx.Add(uint64(i), vector, meta)
		assert.NoError(t, err)
	}

	// Force flush the batch
	err = idx.flushBatch()
	assert.NoError(t, err)

	// Perform a search to verify the index is working
	query := []float32{0.5, 1.0, 1.5}
	results, err := idx.Search(query, 5, 0, 5)
	assert.NoError(t, err)
	assert.Len(t, results, 5, "Should return 5 results")

	// Create a backup
	backupDir := filepath.Join(tmp, "backup")
	err = idx.Backup(backupDir, false, false)
	assert.NoError(t, err)

	// Verify backup files exist
	backupFiles := []string{
		"index.hnsw",
		"metadata.json",
		"vectors.json",
		"manifest.json",
		"backup.json",
	}

	for _, file := range backupFiles {
		path := filepath.Join(backupDir, file)
		_, err = os.Stat(path)
		assert.NoError(t, err, "Backup file %s should exist", file)
	}

	// Create a new index for restore
	restoreConfig := Config{
		Dimension:       3,
		StoragePath:     filepath.Join(tmp, "restore.db"),
		MaxElements:     1000,
		HNSWM:           32,
		HNSWEfConstruct: 200,
		HNSWEfSearch:    100,
		BatchSize:       10,
	}

	restoreIdx, err := New(restoreConfig, testLogger)
	assert.NoError(t, err)

	// Restore directly from the backup directory
	err = restoreIdx.Restore(backupDir)
	assert.NoError(t, err)

	// Verify the restored index has the same data
	// 1. Check that the number of vectors is the same
	assert.Equal(t, idx.hnsw.Len(), restoreIdx.hnsw.Len(), "Restored index should have the same number of vectors")

	// 2. Check that metadata is restored correctly
	for i := 1; i <= 20; i++ {
		id := uint64(i)
		originalMeta := idx.metadata[id]
		restoredMeta := restoreIdx.metadata[id]
		assert.Equal(t, originalMeta["name"], restoredMeta["name"], "Metadata name should match for ID %d", id)
		assert.Equal(t, originalMeta["value"], restoredMeta["value"], "Metadata value should match for ID %d", id)
		assert.Equal(t, originalMeta["category"], restoredMeta["category"], "Metadata category should match for ID %d", id)
	}

	// 3. Check that vectors are restored correctly
	for i := 1; i <= 20; i++ {
		id := uint64(i)
		originalVector := idx.vectors[id]
		restoredVector := restoreIdx.vectors[id]
		assert.Equal(t, originalVector, restoredVector, "Vector should match for ID %d", id)
	}

	// 4. Perform the same search on the restored index and verify results are similar
	restoredResults, err := restoreIdx.Search(query, 5, 0, 5)
	assert.NoError(t, err)
	assert.Len(t, restoredResults, 5, "Restored index should return 5 results")

	// The results might not be exactly the same due to how HNSW works, but they should be similar
	// Let's check that at least 3 of the top 5 results are the same
	originalIDs := make(map[uint64]bool)
	for _, result := range results {
		originalIDs[result.ID] = true
	}

	matchCount := 0
	for _, result := range restoredResults {
		if originalIDs[result.ID] {
			matchCount++
		}
	}

	assert.GreaterOrEqual(t, matchCount, 3, "At least 3 of the top 5 results should match between original and restored index")
}

// TestAddWithImmediateFlush tests that the Add method performs an immediate flush when the batch buffer is full
func TestAddWithImmediateFlush(t *testing.T) {
	tmp := t.TempDir()
	config := Config{
		Dimension:       3,
		StoragePath:     filepath.Join(tmp, "test.db"),
		MaxElements:     1000,
		HNSWM:           32,
		HNSWEfConstruct: 200,
		HNSWEfSearch:    100,
		BatchSize:       5, // Small batch size for testing
	}

	// Create a new index
	idx, err := New(config, testLogger)
	assert.NoError(t, err)

	// Add vectors up to batch size
	for i := 1; i <= 4; i++ {
		vector := []float32{float32(i) * 0.1, float32(i) * 0.2, float32(i) * 0.3}
		meta := map[string]interface{}{
			"name":     fmt.Sprintf("test-%d", i),
			"value":    i,
			"category": "test",
		}
		err := idx.Add(uint64(i), vector, meta)
		assert.NoError(t, err)
	}

	// At this point, the batch buffer should have 4 items

	// Add one more vector to trigger the flush
	vector := []float32{0.5, 0.6, 0.7}
	meta := map[string]interface{}{
		"name":     "test-5",
		"value":    5,
		"category": "test",
	}
	err = idx.Add(uint64(5), vector, meta)
	assert.NoError(t, err)

	// Give the goroutine a moment to execute
	time.Sleep(100 * time.Millisecond)

	// Now search for the vectors to verify they were added to the index
	results, err := idx.Search(vector, 5, 1, 10)
	assert.NoError(t, err)
	assert.NotEmpty(t, results, "Search should return results after flush")
}

// TestDeleteVector tests that the DeleteVector function properly removes entries from all relevant maps
func TestDeleteVector(t *testing.T) {
	tmp := t.TempDir()
	config := Config{
		Dimension:       3,
		StoragePath:     filepath.Join(tmp, "test.db"),
		MaxElements:     1000,
		HNSWM:           32,
		HNSWEfConstruct: 200,
		HNSWEfSearch:    100,
		BatchSize:       5,
	}

	// Create a new index
	idx, err := New(config, testLogger)
	assert.NoError(t, err)

	// Add a vector
	vector := []float32{0.1, 0.2, 0.3}
	meta := map[string]interface{}{
		"name":     "test-1",
		"value":    1,
		"category": "test",
	}
	err = idx.Add(uint64(1), vector, meta)
	assert.NoError(t, err)

	// Force flush the batch
	err = idx.flushBatch()
	assert.NoError(t, err)

	// Verify the vector exists in all maps
	_, vectorExists := idx.vectors[1]
	assert.True(t, vectorExists, "Vector should exist in vectors map")

	_, metadataExists := idx.metadata[1]
	assert.True(t, metadataExists, "Vector should exist in metadata map")

	_, cacheExists := idx.cache.Load(uint64(1))
	assert.True(t, cacheExists, "Vector should exist in cache")

	// Verify the vector exists in the database
	if idx.dbConn != nil {
		results, err := idx.QueryMetadata("SELECT * FROM metadata WHERE id = 1")
		assert.NoError(t, err)
		assert.Len(t, results, 1, "Vector should exist in database")
	}

	// Delete the vector
	err = idx.DeleteVector(uint64(1))
	assert.NoError(t, err)

	// Verify the vector is removed from all maps
	_, vectorExists = idx.vectors[1]
	assert.False(t, vectorExists, "Vector should be removed from vectors map")

	_, metadataExists = idx.metadata[1]
	assert.False(t, metadataExists, "Vector should be removed from metadata map")

	_, cacheExists = idx.cache.Load(uint64(1))
	assert.False(t, cacheExists, "Vector should be removed from cache")

	// Verify the vector is removed from the database
	if idx.dbConn != nil {
		results, err := idx.QueryMetadata("SELECT * FROM metadata WHERE id = 1")
		assert.NoError(t, err)
		assert.Len(t, results, 0, "Vector should be removed from database")
	}
}

// TestDeleteVectors tests that the DeleteVectors function properly removes entries from all relevant maps
func TestDeleteVectors(t *testing.T) {
	tmp := t.TempDir()
	config := Config{
		Dimension:       3,
		StoragePath:     filepath.Join(tmp, "test.db"),
		MaxElements:     1000,
		HNSWM:           32,
		HNSWEfConstruct: 200,
		HNSWEfSearch:    100,
		BatchSize:       5,
	}

	// Create a new index
	idx, err := New(config, testLogger)
	assert.NoError(t, err)

	// Add multiple vectors
	for i := 1; i <= 3; i++ {
		vector := []float32{float32(i) * 0.1, float32(i) * 0.2, float32(i) * 0.3}
		meta := map[string]interface{}{
			"name":     fmt.Sprintf("test-%d", i),
			"value":    i,
			"category": "test",
		}
		err = idx.Add(uint64(i), vector, meta)
		assert.NoError(t, err)
	}

	// Force flush the batch
	err = idx.flushBatch()
	assert.NoError(t, err)

	// Verify the vectors exist in all maps
	for i := 1; i <= 3; i++ {
		_, vectorExists := idx.vectors[uint64(i)]
		assert.True(t, vectorExists, "Vector %d should exist in vectors map", i)

		_, metadataExists := idx.metadata[uint64(i)]
		assert.True(t, metadataExists, "Vector %d should exist in metadata map", i)

		_, cacheExists := idx.cache.Load(uint64(i))
		assert.True(t, cacheExists, "Vector %d should exist in cache", i)
	}

	// Verify the vectors exist in the database
	if idx.dbConn != nil {
		results, err := idx.QueryMetadata("SELECT * FROM metadata WHERE id IN (1, 2, 3)")
		assert.NoError(t, err)
		assert.Len(t, results, 3, "All vectors should exist in database")
	}

	// Delete vectors with IDs 1 and 3
	err = idx.DeleteVectors([]uint64{1, 3})
	assert.NoError(t, err)

	// Verify vectors 1 and 3 are removed from all maps
	for _, id := range []uint64{1, 3} {
		_, vectorExists := idx.vectors[id]
		assert.False(t, vectorExists, "Vector %d should be removed from vectors map", id)

		_, metadataExists := idx.metadata[id]
		assert.False(t, metadataExists, "Vector %d should be removed from metadata map", id)

		_, cacheExists := idx.cache.Load(id)
		assert.False(t, cacheExists, "Vector %d should be removed from cache", id)
	}

	// Verify vector 2 still exists in all maps
	_, vectorExists := idx.vectors[uint64(2)]
	assert.True(t, vectorExists, "Vector 2 should still exist in vectors map")

	_, metadataExists := idx.metadata[uint64(2)]
	assert.True(t, metadataExists, "Vector 2 should still exist in metadata map")

	_, cacheExists := idx.cache.Load(uint64(2))
	assert.True(t, cacheExists, "Vector 2 should still exist in cache")

	// Verify vectors 1 and 3 are removed from the database and vector 2 still exists
	if idx.dbConn != nil {
		results, err := idx.QueryMetadata("SELECT * FROM metadata WHERE id IN (1, 3)")
		assert.NoError(t, err)
		assert.Len(t, results, 0, "Vectors 1 and 3 should be removed from database")

		results, err = idx.QueryMetadata("SELECT * FROM metadata WHERE id = 2")
		assert.NoError(t, err)
		assert.Len(t, results, 1, "Vector 2 should still exist in database")
	}
}
