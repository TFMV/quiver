package hybrid

import (
	"testing"

	"github.com/TFMV/quiver/pkg/vectortypes"
)

func TestSearchWithNegative_Stability(t *testing.T) {
	config := IndexConfig{
		ExactThreshold: 100, // force exact index for small data
		DistanceFunc:   vectortypes.EuclideanDistance,
	}
	idx := NewHybridIndex(config)

	// Add identical items at exactly the same distance to the query and negative example
	idx.Insert("1", []float32{1.0, 0.0})
	idx.Insert("2", []float32{1.0, 0.0})
	idx.Insert("3", []float32{1.0, 0.0})

	query := []float32{0.0, 0.0}
	negEx := []float32{-1.0, 0.0}

	// Should not panic or return NaNs, and distances should be stable
	res, err := idx.SearchWithRequest(HybridSearchRequest{
		Query:           query,
		K:               3,
		NegativeExample: negEx,
		NegativeWeight:  0.5,
	})
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(res.Results) != 3 {
		t.Fatalf("Expected 3 results")
	}

	// Because they are identical points, the relative scores should exact match safely
	d1 := res.Results[0].Distance
	d2 := res.Results[1].Distance
	d3 := res.Results[2].Distance

	if d1 != d2 || d2 != d3 {
		t.Errorf("Expected identical distances for identical vectors, got %f, %f, %f", d1, d2, d3)
	}
}
