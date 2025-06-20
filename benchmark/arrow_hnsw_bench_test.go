package benchmark

import (
	"fmt"
	"testing"

	"github.com/apache/arrow/go/arrow/array"
	"github.com/apache/arrow/go/arrow/memory"

	"github.com/TFMV/quiver/index"
)

func buildArrowIndex(n, dim int) *index.ArrowHNSWIndex { // arrow-hnsw
	idx := index.NewArrowHNSWIndex(dim)
	for i := 0; i < n; i++ {
		b := array.NewFloat32Builder(memory.DefaultAllocator)
		vals := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vals[j] = float32(i*j) / float32(dim)
		}
		b.AppendValues(vals, nil)
		arr := b.NewArray()
		idx.Add(arr, fmt.Sprintf("%d", i))
		arr.Release()
	}
	return idx
}

func BenchmarkArrowHNSWBuild(b *testing.B) { // arrow-hnsw
	for i := 0; i < b.N; i++ {
		_ = buildArrowIndex(1000, 32)
	}
}

func BenchmarkArrowHNSWSearch(b *testing.B) { // arrow-hnsw
	idx := buildArrowIndex(100000, 32)
	qb := array.NewFloat32Builder(memory.DefaultAllocator)
	qb.AppendValues(make([]float32, 32), nil)
	query := qb.NewArray()
	defer query.Release()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.Search(query, 10)
	}
}
