package arrowindex

import (
	"reflect"
	"testing"

	"github.com/apache/arrow-go/v18/arrow/memory"
)

func TestSearch_SmallGraphSorted(t *testing.T) {
	// m=16, graph size < 16, means it will use exhaustiveSearch
	g := NewGraph(2, 16, 200, 100, 1024, memory.DefaultAllocator)

	// Add points in unsorted distance order relative to {0,0}
	g.Add(1, []float64{0.0, 0.0}) // closest to {0,0}
	g.Add(2, []float64{2.0, 2.0}) // furthest
	g.Add(3, []float64{1.0, 1.0}) // middle

	res, err := g.Search([]float64{0.1, 0.1}, 3)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	expected := []int{1, 3, 2}
	if !reflect.DeepEqual(res, expected) {
		t.Errorf("Expected %v, got %v", expected, res)
	}
}
