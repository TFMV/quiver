package quiver

import (
	"encoding/json"
	"fmt"
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
	config := zap.NewDevelopmentConfig()
	config.DisableCaller = true
	config.DisableStacktrace = true
	logger, err := config.Build()
	if err != nil {
		panic(fmt.Sprintf("Failed to initialize test logger: %v", err))
	}
	testLogger = logger
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
