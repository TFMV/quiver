package index

import (
	"os"
	"testing"

	"github.com/apache/arrow/go/arrow/array"
	"github.com/apache/arrow/go/arrow/memory"
)

func TestArrowHNSWIndex_AddSearch(t *testing.T) { // arrow-hnsw
	idx := NewArrowHNSWIndex(3)
	b := array.NewFloat32Builder(memory.DefaultAllocator)
	b.AppendValues([]float32{1, 0, 0}, nil)
	vec := b.NewArray()
	defer vec.Release()
	if err := idx.Add(vec, "v1"); err != nil {
		t.Fatalf("add: %v", err)
	}
	b2 := array.NewFloat32Builder(memory.DefaultAllocator)
	b2.AppendValues([]float32{0.9, 0, 0}, nil)
	q := b2.NewArray()
	defer q.Release()
	res, err := idx.Search(q, 1)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(res) != 1 || res[0].ID != "v1" {
		t.Fatalf("unexpected result: %+v", res)
	}
}

func TestArrowHNSWIndex_SaveLoad(t *testing.T) { // arrow-hnsw
	idx := NewArrowHNSWIndex(2)
	b := array.NewFloat32Builder(memory.DefaultAllocator)
	b.AppendValues([]float32{1, 2}, nil)
	vec := b.NewArray()
	defer vec.Release()
	if err := idx.Add(vec, "a"); err != nil {
		t.Fatalf("add: %v", err)
	}
	path := "test_arrow_hnsw.arrow"
	if err := idx.Save(path); err != nil {
		t.Fatalf("save: %v", err)
	}
	defer os.Remove(path)

	idx2 := NewArrowHNSWIndex(2)
	if err := idx2.Load(path); err != nil {
		t.Fatalf("load: %v", err)
	}
	b2 := array.NewFloat32Builder(memory.DefaultAllocator)
	b2.AppendValues([]float32{1, 2}, nil)
	q := b2.NewArray()
	defer q.Release()
	res, err := idx2.Search(q, 1)
	if err != nil || len(res) != 1 || res[0].ID != "0" { // loaded id string is index string
		t.Fatalf("unexpected search result after load: %v %+v", err, res)
	}
}
