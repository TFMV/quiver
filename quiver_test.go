package quiver

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
)

// Test New Index
func TestNewIndex(t *testing.T) {
	idx, err := New(Config{Dimension: 3, StoragePath: "test.db", MaxElements: 1000})
	assert.NoError(t, err)
	assert.NotNil(t, idx)
}

// Test Add and Search
func TestAddAndSearch(t *testing.T) {
	idx, _ := New(Config{Dimension: 3, StoragePath: "test.db", MaxElements: 1000})
	defer idx.Close()

	err := idx.Add(1, []float32{0.1, 0.2, 0.3}, map[string]interface{}{"category": "science"})
	assert.NoError(t, err)

	results, _ := idx.Search([]float32{0.1, 0.2, 0.3}, 1)
	assert.Equal(t, uint64(1), results[0].ID)
}

// Test Hybrid Search
func TestHybridSearch(t *testing.T) {
	tmpDB := t.TempDir() + "/test.db"
	idx, err := New(Config{
		Dimension:   3,
		StoragePath: tmpDB,
		MaxElements: 1000,
	})
	assert.NoError(t, err)
	defer func() {
		idx.Close()
		os.Remove(tmpDB)
	}()

	// Add test vectors with metadata
	err = idx.Add(1, []float32{1.0, 0.0, 0.0}, map[string]interface{}{"category": "math"})
	assert.NoError(t, err)
	err = idx.Add(2, []float32{0.0, 1.0, 0.0}, map[string]interface{}{"category": "science"})
	assert.NoError(t, err)

	// Search with filter - note we pass the value without quotes
	results, err := idx.SearchWithFilter([]float32{1.0, 0.0, 0.0}, 1, "math")
	assert.NoError(t, err)
	assert.NotEmpty(t, results, "Should return at least one result")
	if len(results) > 0 {
		assert.Equal(t, uint64(1), results[0].ID)
	}
}
