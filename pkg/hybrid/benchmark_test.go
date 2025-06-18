package hybrid

import (
	"fmt"
	"testing"

	"github.com/TFMV/quiver/pkg/vectortypes"
)

func buildExactIndex(n, dim int) *ExactIndex {
	idx := NewExactIndex(vectortypes.CosineDistance)
	vec := make(vectortypes.F32, dim)
	for i := 0; i < n; i++ {
		idx.Insert(fmt.Sprintf("id%d", i), vec)
	}
	return idx
}

func BenchmarkExactIndexSearch(b *testing.B) {
	idx := buildExactIndex(1000, 64)
	q := make(vectortypes.F32, 64)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.Search(q, 10)
	}
}

func buildHybridIndex(n, dim int) *HybridIndex {
	cfg := DefaultIndexConfig()
	idx := NewHybridIndex(cfg)
	vec := make(vectortypes.F32, dim)
	for i := 0; i < n; i++ {
		idx.Insert(fmt.Sprintf("id%d", i), vec)
	}
	return idx
}

func BenchmarkHybridIndexSearch(b *testing.B) {
	idx := buildHybridIndex(1000, 64)
	q := make(vectortypes.F32, 64)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.Search(q, 10)
	}
}
