package quiver

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestNewIndex ensures that a new index is properly initialized.
func TestNewIndex(t *testing.T) {
	config := Config{Dimension: 3, Distance: Cosine}
	idx := New(config)

	assert.NotNil(t, idx)
	assert.Equal(t, 3, idx.config.Dimension)
	assert.Equal(t, Cosine, idx.config.Distance)
	assert.Empty(t, idx.vectors)
}

// TestAddVector checks if vectors are added correctly.
func TestAddVector(t *testing.T) {
	idx := New(Config{Dimension: 3, Distance: Cosine})

	err := idx.Add(1, []float32{0.1, 0.2, 0.3})
	assert.NoError(t, err)
	assert.Contains(t, idx.vectors, 1)

	// Test dimension mismatch
	err = idx.Add(2, []float32{0.1, 0.2})
	assert.Equal(t, ErrDimensionMismatch, err)
}

// TestSearch verifies k-NN search returns correct results.
func TestSearch(t *testing.T) {
	idx := New(Config{Dimension: 3, Distance: Cosine})

	// Add some sample vectors
	_ = idx.Add(1, []float32{1.0, 0.0, 0.0})
	_ = idx.Add(2, []float32{0.0, 1.0, 0.0})
	_ = idx.Add(3, []float32{0.0, 0.0, 1.0})
	_ = idx.Add(4, []float32{1.0, 1.0, 1.0})

	// Perform a search
	query := []float32{1.0, 0.0, 0.0}
	results, err := idx.Search(query, 2)

	assert.NoError(t, err)
	assert.Len(t, results, 2)
	assert.Contains(t, results, 1) // Expect the closest match to be ID 1
}

// TestSearchDimensionMismatch ensures error handling for incorrect query dimensions.
func TestSearchDimensionMismatch(t *testing.T) {
	idx := New(Config{Dimension: 3, Distance: Cosine})

	query := []float32{1.0, 0.0} // Incorrect dimension
	_, err := idx.Search(query, 2)

	assert.Equal(t, ErrDimensionMismatch, err)
}

// TestCosineDistance ensures correct cosine distance calculations.
func TestCosineDistance(t *testing.T) {
	a := []float32{1, 0, 0}
	b := []float32{0, 1, 0}
	c := []float32{1, 1, 0}

	assert.Equal(t, float32(1.0), cosineDistance(a, b)) // Perpendicular vectors (max distance)
	assert.Less(t, cosineDistance(a, c), float32(1.0))  // Closer than perpendicular
}

// TestEuclideanDistance ensures correct Euclidean distance calculations.
func TestEuclideanDistance(t *testing.T) {
	a := []float32{0, 0}
	b := []float32{3, 4}

	assert.Equal(t, float32(5.0), euclideanDistance(a, b)) // Pythagorean theorem
}
