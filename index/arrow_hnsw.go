package index

// arrow-hnsw

import (
	"fmt"
	"os"
	"sync"

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
type ArrowHNSWIndex struct {
	graph     *arrowindex.Graph
	dim       int
	allocator memory.Allocator

	mu      sync.RWMutex
	idToIdx map[string]int
	idxToID map[int]string

	// Thread-safe reusable buffer pool to eliminate allocation during Add/Search
	vectorPool sync.Pool
}

// NewArrowHNSWIndex creates a new index with default HNSW parameters.
func NewArrowHNSWIndex(dim int) *ArrowHNSWIndex {
	g := arrowindex.NewGraph(dim, 16, 200, 100, 1024, memory.DefaultAllocator)

	return &ArrowHNSWIndex{
		graph:     g,
		dim:       dim,
		allocator: memory.DefaultAllocator,
		idToIdx:   make(map[string]int),
		idxToID:   make(map[int]string),
		vectorPool: sync.Pool{
			New: func() any {
				v := make([]float64, dim)
				return &v
			},
		},
	}
}

// addRaw bypasses Arrow array construction for high-speed internal loading.
func (idx *ArrowHNSWIndex) addRaw(f32vals []float32, id string) error {
	idx.mu.Lock()
	if _, exists := idx.idToIdx[id]; exists {
		idx.mu.Unlock()
		return fmt.Errorf("vector with ID %s already exists", id)
	}
	internal := len(idx.idxToID)
	idx.idToIdx[id] = internal
	idx.idxToID[internal] = id
	idx.mu.Unlock()

	// Use existing buffer to avoid allocation
	ptr := idx.vectorPool.Get().(*[]float64)
	vals := *ptr

	for i := 0; i < idx.dim; i++ {
		vals[i] = float64(f32vals[i])
	}

	err := idx.graph.Add(internal, vals)
	idx.vectorPool.Put(ptr)
	return err
}

// Add inserts a vector with the given ID into the index.
func (idx *ArrowHNSWIndex) Add(vec *array.Float32, id string) error {
	if vec == nil || vec.Data() == nil {
		return fmt.Errorf("invalid or nil vector provided")
	}
	if vec.Len() != idx.dim {
		return fmt.Errorf("dimension mismatch: got %d want %d", vec.Len(), idx.dim)
	}
	return idx.addRaw(vec.Float32Values(), id)
}

// Search returns the k nearest results to the query vector.
func (idx *ArrowHNSWIndex) Search(query *array.Float32, k int) ([]Result, error) {
	if k <= 0 {
		return nil, fmt.Errorf("k must be positive")
	}
	if query == nil || query.Data() == nil {
		return nil, fmt.Errorf("invalid or nil query vector")
	}
	if query.Len() != idx.dim {
		return nil, fmt.Errorf("dimension mismatch: got %d want %d", query.Len(), idx.dim)
	}

	ptr := idx.vectorPool.Get().(*[]float64)
	q := *ptr
	defer idx.vectorPool.Put(ptr)

	// Direct slice access is significantly faster than querying .Value(i)
	f32vals := query.Float32Values()
	for i := 0; i < idx.dim; i++ {
		q[i] = float64(f32vals[i])
	}

	indices, err := idx.graph.Search(q, k)
	if err != nil {
		return nil, err
	}

	res := make([]Result, len(indices))
	idx.mu.RLock()
	defer idx.mu.RUnlock()

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
func (idx *ArrowHNSWIndex) Save(path string) error {
	idx.mu.RLock()
	length := idx.graph.Len()
	ids := make([]string, 0, length)

	for internal := 0; internal < length; internal++ {
		id, exists := idx.idxToID[internal]
		if !exists {
			idx.mu.RUnlock()
			return fmt.Errorf("missing external ID for internal index %d", internal)
		}
		ids = append(ids, id)
	}
	idx.mu.RUnlock()

	schema := arrow.NewSchema([]arrow.Field{
		{Name: "id", Type: arrow.BinaryTypes.String},
		{Name: "vector", Type: arrow.FixedSizeListOf(int32(idx.dim), arrow.PrimitiveTypes.Float32)},
	}, nil)

	rb := array.NewRecordBuilder(idx.allocator, schema)
	defer rb.Release()

	idBuilder := rb.Field(0).(*array.StringBuilder)
	listBuilder := rb.Field(1).(*array.FixedSizeListBuilder)
	fb := listBuilder.ValueBuilder().(*array.Float32Builder)

	// Pre-allocate array capacity to prevent re-allocations during iteration
	idBuilder.Reserve(length)
	listBuilder.Reserve(length)
	fb.Reserve(length * idx.dim)

	f32vec := make([]float32, idx.dim)
	for internal := 0; internal < length; internal++ {
		idBuilder.Append(ids[internal])
		listBuilder.Append(true)

		vec := idx.graph.GetVector(internal)
		for j := 0; j < idx.dim; j++ {
			f32vec[j] = float32(vec[j])
		}
		fb.AppendValues(f32vec, nil)
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
	defer w.Close()

	return w.Write(rec)
}

// Load restores the index from an Arrow IPC file.
func (idx *ArrowHNSWIndex) Load(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	r, err := ipc.NewFileReader(f)
	if err != nil {
		return err
	}
	defer r.Close()

	// Arrow IPC files can contain multiple records; iterate through all of them
	for i := 0; i < r.NumRecords(); i++ {
		rec, err := r.Record(i)
		if err != nil {
			return err
		}

		ids := rec.Column(0).(*array.String)
		list := rec.Column(1).(*array.FixedSizeList)
		values := list.ListValues().(*array.Float32)

		f32vals := values.Float32Values()
		offset := 0

		numRows := int(rec.NumRows())
		for j := 0; j < numRows; j++ {
			vecSlice := f32vals[offset : offset+idx.dim]

			// Bypass expensive array builder overhead by calling addRaw directly
			if err := idx.addRaw(vecSlice, ids.Value(j)); err != nil {
				return err
			}
			offset += idx.dim
		}
	}

	return nil
}
