package persistence

import (
	"fmt"
	"testing"

	"github.com/TFMV/quiver/pkg/vectortypes"
)

func setupBenchmarkCollection(numVectors, dim int) *Collection {
	c := NewCollection("bench", dim, vectortypes.EuclideanDistance)
	vec := make([]float32, dim)
	for i := 0; i < numVectors; i++ {
		id := fmt.Sprintf("v%d", i)
		for j := range vec {
			vec[j] = float32((i+j)%dim) / 100.0
		}
		c.AddVector(id, append([]float32(nil), vec...), nil)
	}
	return c
}

func BenchmarkCollectionSearch(b *testing.B) {
	const numVectors = 1000
	const dim = 32
	c := setupBenchmarkCollection(numVectors, dim)
	query := make([]float32, dim)
	for i := range query {
		query[i] = 0.5
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = c.Search(query, 10)
	}
}

func BenchmarkSortSearchResults(b *testing.B) {
	const numResults = 1000
	results := make([]SearchResult, numResults)
	for i := 0; i < numResults; i++ {
		results[i] = SearchResult{
			ID:       fmt.Sprintf("v%d", i),
			Distance: float32(numResults - i),
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SortSearchResults(results)
	}
}
