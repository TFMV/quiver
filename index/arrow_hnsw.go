package index

// arrow-hnsw

import (
	"fmt"
	"os"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/ipc"
	"github.com/apache/arrow-go/v18/arrow/memory"

	"github.com/TFMV/quiver/pkg/arrowindex"
)

// Result represents a search result from ArrowHNSWIndex.
type Result struct {
	ID       string
	Distance float32
}

// ArrowHNSWIndex stores vectors in Arrow format backed by an HNSW graph.
type ArrowHNSWIndex struct { // arrow-hnsw
	graph     *arrowindex.Graph
	dim       int
	allocator memory.Allocator
	idToIdx   map[string]int
	idxToID   map[int]string
}

// NewArrowHNSWIndex creates a new index with default HNSW parameters.
func NewArrowHNSWIndex(dim int) *ArrowHNSWIndex { // arrow-hnsw
	g := arrowindex.NewGraph(dim, 16, 200, 100, 1024, memory.DefaultAllocator)
	return &ArrowHNSWIndex{
		graph:     g,
		dim:       dim,
		allocator: memory.DefaultAllocator,
		idToIdx:   make(map[string]int),
		idxToID:   make(map[int]string),
	}
}

// Add inserts a vector with the given ID into the index.
func (idx *ArrowHNSWIndex) Add(vec *array.Float32, id string) error { // arrow-hnsw
	if vec.Len() != idx.dim {
		return fmt.Errorf("dimension mismatch: got %d want %d", vec.Len(), idx.dim)
	}
	vals := make([]float64, idx.dim)
	for i := 0; i < idx.dim; i++ {
		vals[i] = float64(vec.Value(i))
	}
	internal := len(idx.idToIdx)
	idx.idToIdx[id] = internal
	idx.idxToID[internal] = id
	return idx.graph.Add(internal, vals)
}

// Search returns the k nearest results to the query vector.
func (idx *ArrowHNSWIndex) Search(query *array.Float32, k int) ([]Result, error) { // arrow-hnsw
	if query.Len() != idx.dim {
		return nil, fmt.Errorf("dimension mismatch: got %d want %d", query.Len(), idx.dim)
	}
	q := make([]float64, idx.dim)
	for i := 0; i < idx.dim; i++ {
		q[i] = float64(query.Value(i))
	}
	indices, err := idx.graph.Search(q, k)
	if err != nil {
		return nil, err
	}
	res := make([]Result, len(indices))
	for i, idxNum := range indices {
		vec := idx.graph.GetVector(idxNum)
		var dist float64
		for j := 0; j < idx.dim; j++ {
			d := q[j] - vec[j]
			dist += d * d
		}
		res[i] = Result{ID: idx.idxToID[idxNum], Distance: float32(dist)}
	}
	return res, nil
}

// Save writes the index to an Arrow IPC file.
func (idx *ArrowHNSWIndex) Save(path string) error { // arrow-hnsw
	ids := make([]int32, 0, idx.graph.Len())
	vecBuilder := array.NewFloat32Builder(idx.allocator)
	for id, internal := range idx.idToIdx {
		_ = id
		ids = append(ids, int32(internal))
		vec := idx.graph.GetVector(internal)
		vecBuilder.AppendValues(float32Slice(vec), nil)
	}
	vecArray := vecBuilder.NewArray().(*array.Float32)
	defer vecArray.Release()

	schema := arrow.NewSchema([]arrow.Field{
		{Name: "id", Type: arrow.PrimitiveTypes.Int32},
		{Name: "vector", Type: arrow.FixedSizeListOf(int32(idx.dim), arrow.PrimitiveTypes.Float32)},
	}, nil)

	rb := array.NewRecordBuilder(idx.allocator, schema)
	defer rb.Release()
	rb.Field(0).(*array.Int32Builder).AppendValues(ids, nil)
	listBuilder := rb.Field(1).(*array.FixedSizeListBuilder)
	fb := listBuilder.ValueBuilder().(*array.Float32Builder)
	offset := 0
	for i := 0; i < len(ids); i++ {
		listBuilder.Append(true)
		fb.AppendValues(vecArray.Float32Values()[offset:offset+idx.dim], nil)
		offset += idx.dim
	}
	rec := rb.NewRecord()
	defer rec.Release()

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w, err := ipc.NewFileWriter(f, ipc.WithSchema(schema))
	if err != nil {
		return err
	}
	if err := w.Write(rec); err != nil {
		w.Close()
		return err
	}
	return w.Close()
}

// Load restores the index from an Arrow IPC file.
func (idx *ArrowHNSWIndex) Load(path string) error { // arrow-hnsw
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	r, err := ipc.NewFileReader(f)
	if err != nil {
		return err
	}
	rec, err := r.Record(0)
	if err != nil {
		return err
	}
	ids := rec.Column(0).(*array.Int32)
	list := rec.Column(1).(*array.FixedSizeList)
	values := list.ListValues().(*array.Float32)
	offset := 0
	for i := 0; i < int(rec.NumRows()); i++ {
		vecSlice := values.Float32Values()[offset : offset+idx.dim]
		vb := array.NewFloat32Builder(idx.allocator)
		vb.AppendValues(vecSlice, nil)
		arr := vb.NewArray()
		idStr := fmt.Sprintf("%d", ids.Value(i))
		if err := idx.Add(arr.(*array.Float32), idStr); err != nil {
			arr.Release()
			vb.Release()
			return err
		}
		arr.Release()
		vb.Release()
		offset += idx.dim
	}
	return nil
}

func float32Slice(in []float64) []float32 { // arrow-hnsw
	out := make([]float32, len(in))
	for i, v := range in {
		out[i] = float32(v)
	}
	return out
}
