package quiver

import (
	"fmt"
	"math/rand"
	"os"
	"sync/atomic"
	"testing"

	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/stretchr/testify/assert"
	"go.uber.org/zap"
)

var (
	testLogger  *zap.Logger
	benchLogger *zap.Logger
)

func init() {
	testLogger, _ = zap.NewProduction(
		zap.Fields(zap.String("test", "quiver_test")),
	)

	// Create error-only logger for benchmarks
	cfg := zap.NewProductionConfig()
	cfg.Level = zap.NewAtomicLevelAt(zap.ErrorLevel)
	benchLogger, _ = cfg.Build(
		zap.Fields(zap.String("test", "quiver_bench")),
	)
}

// Test New Index
func TestNewIndex(t *testing.T) {
	idx, err := New(Config{Dimension: 3, StoragePath: "test.db", MaxElements: 1000}, testLogger)
	assert.NoError(t, err)
	assert.NotNil(t, idx)
}

// Test Add and Search
func TestAddAndSearch(t *testing.T) {
	idx, err := New(Config{
		Dimension:       3,
		StoragePath:     t.TempDir() + "/test.db",
		MaxElements:     1000,
		HNSWM:           32,
		HNSWEfConstruct: 200,
		HNSWEfSearch:    200,
		BatchSize:       100,
		Distance:        Cosine,
	}, testLogger)
	assert.NoError(t, err)
	defer idx.Close()

	// Add several vectors to ensure the index has enough data
	vectors := []struct {
		id   uint64
		vec  []float32
		meta map[string]interface{}
	}{
		{1, []float32{0.1, 0.2, 0.3}, map[string]interface{}{"category": "science"}},
		{2, []float32{0.4, 0.5, 0.6}, map[string]interface{}{"category": "math"}},
		{3, []float32{0.7, 0.8, 0.9}, map[string]interface{}{"category": "art"}},
	}

	for _, v := range vectors {
		err := idx.Add(v.id, v.vec, v.meta)
		assert.NoError(t, err)
	}

	// Force flush the batch
	idx.flushBatch()

	// Now search
	results, err := idx.Search([]float32{0.1, 0.2, 0.3}, 1)
	assert.NoError(t, err)
	assert.NotEmpty(t, results)
	assert.Equal(t, uint64(1), results[0].ID)
}

// Test Hybrid Search
func TestHybridSearch(t *testing.T) {
	tmpDB := t.TempDir() + "/test.db"
	idx, err := New(Config{
		Dimension:       3,
		StoragePath:     tmpDB,
		MaxElements:     1000,
		HNSWM:           16,
		HNSWEfConstruct: 40,
		HNSWEfSearch:    10,
		BatchSize:       1, // Set to 1 to force immediate flush
		Distance:        Cosine,
	}, testLogger)
	assert.NoError(t, err)
	defer func() {
		idx.Close()
		os.Remove(tmpDB)
	}()

	// Add test vectors
	vectors := []struct {
		id   uint64
		vec  []float32
		meta map[string]interface{}
	}{
		{1, []float32{1.0, 0.0, 0.0}, map[string]interface{}{"category": "math"}},
		{2, []float32{0.0, 1.0, 0.0}, map[string]interface{}{"category": "science"}},
		{3, []float32{0.0, 0.0, 1.0}, map[string]interface{}{"category": "math"}},
	}

	// Add vectors one by one and verify metadata
	for _, v := range vectors {
		err := idx.Add(v.id, v.vec, v.meta)
		assert.NoError(t, err)
		idx.flushBatch()

		// Verify metadata is stored
		results, err := idx.Search(v.vec, 1)
		assert.NoError(t, err)
		assert.NotEmpty(t, results)
		assert.Equal(t, v.meta["category"], results[0].Metadata["category"])
	}

	// Now try hybrid search
	results, err := idx.SearchWithFilter([]float32{1.0, 0.0, 0.0}, 1, "math")
	assert.NoError(t, err)
	if assert.NotEmpty(t, results, "Should return at least one result") {
		assert.Equal(t, uint64(1), results[0].ID)
		assert.Equal(t, "math", results[0].Metadata["category"])
	}
}

// Add these benchmark functions after the existing tests

func BenchmarkAdd(b *testing.B) {
	idx, err := New(Config{
		Dimension:       128,
		StoragePath:     "",
		MaxElements:     100000,
		HNSWM:           32,
		HNSWEfConstruct: 200,
		HNSWEfSearch:    200,
		BatchSize:       1000,
		Distance:        Cosine,
	}, benchLogger)
	if err != nil {
		b.Fatal(err)
	}
	defer idx.Close()

	// Generate random vector once
	vector := make([]float32, 128)
	for i := range vector {
		vector[i] = rand.Float32()
	}
	meta := map[string]interface{}{"category": "bench"}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := idx.Add(uint64(i), vector, meta); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSearch(b *testing.B) {
	// Setup index with test data
	idx, err := New(Config{
		Dimension:       128,
		StoragePath:     "",
		MaxElements:     100000,
		HNSWM:           32,
		HNSWEfConstruct: 200,
		HNSWEfSearch:    200,
		BatchSize:       1000,
		Distance:        Cosine,
	}, benchLogger)
	if err != nil {
		b.Fatal(err)
	}
	defer idx.Close()

	// Add 10k vectors
	for i := 0; i < 10000; i++ {
		vector := make([]float32, 128)
		for j := range vector {
			vector[j] = rand.Float32()
		}
		if err := idx.Add(uint64(i), vector, nil); err != nil {
			b.Fatal(err)
		}
	}
	idx.flushBatch()

	// Generate query vector
	query := make([]float32, 128)
	for i := range query {
		query[i] = rand.Float32()
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := idx.Search(query, 10); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkHybridSearch(b *testing.B) {
	idx, err := New(Config{
		Dimension:       128,
		StoragePath:     "",
		MaxElements:     100000,
		HNSWM:           32,
		HNSWEfConstruct: 200,
		HNSWEfSearch:    200,
		BatchSize:       1000,
		Distance:        Cosine,
	}, benchLogger)
	if err != nil {
		b.Fatal(err)
	}
	defer idx.Close()

	categories := []string{"science", "math", "art", "history"}

	// Add 10k vectors with metadata
	for i := 0; i < 10000; i++ {
		vector := make([]float32, 128)
		for j := range vector {
			vector[j] = rand.Float32()
		}
		meta := map[string]interface{}{
			"category": categories[i%len(categories)],
		}
		if err := idx.Add(uint64(i), vector, meta); err != nil {
			b.Fatal(err)
		}
	}
	idx.flushBatch()

	query := make([]float32, 128)
	for i := range query {
		query[i] = rand.Float32()
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := idx.SearchWithFilter(query, 10, "science"); err != nil {
			b.Fatal(err)
		}
	}
}

// Add parallel benchmarks
func BenchmarkAddParallel(b *testing.B) {
	idx, err := New(Config{
		Dimension:       128,
		StoragePath:     "",
		MaxElements:     100000,
		HNSWM:           32,
		HNSWEfConstruct: 200,
		HNSWEfSearch:    200,
		BatchSize:       1000,
		Distance:        Cosine,
	}, benchLogger)
	if err != nil {
		b.Fatal(err)
	}
	defer idx.Close()

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		vector := make([]float32, 128)
		for i := range vector {
			vector[i] = rand.Float32()
		}
		meta := map[string]interface{}{"category": "bench"}

		i := uint64(0)
		for pb.Next() {
			if err := idx.Add(atomic.AddUint64(&i, 1), vector, meta); err != nil {
				b.Fatal(err)
			}
		}
	})
}

// Add after existing tests
func TestAppendFromArrow(t *testing.T) {
	// Create test index
	idx, err := New(Config{
		Dimension:       3,
		StoragePath:     t.TempDir() + "/test.db",
		MaxElements:     1000,
		HNSWM:           32,
		HNSWEfConstruct: 200,
		HNSWEfSearch:    200,
		BatchSize:       100,
		Distance:        Cosine,
	}, testLogger)
	assert.NoError(t, err)
	defer idx.Close()

	// Create Arrow record
	pool := memory.NewGoAllocator()
	schema := NewVectorSchema(3)

	// Create builders
	b := array.NewRecordBuilder(pool, schema)
	defer b.Release()

	// Add data
	b.Field(0).(*array.Uint64Builder).AppendValues([]uint64{1, 2}, nil)

	// Create the vector list properly
	listBuilder := b.Field(1).(*array.FixedSizeListBuilder)
	valueBuilder := listBuilder.ValueBuilder().(*array.Float32Builder)

	// Add first vector [0.1, 0.2, 0.3]
	valueBuilder.AppendValues([]float32{0.1, 0.2, 0.3}, nil)
	listBuilder.Append(true)

	// Add second vector [0.4, 0.5, 0.6]
	valueBuilder.AppendValues([]float32{0.4, 0.5, 0.6}, nil)
	listBuilder.Append(true)

	// Add metadata
	b.Field(2).(*array.StringBuilder).AppendValues([]string{
		`{"category": "test1"}`,
		`{"category": "test2"}`,
	}, nil)

	// Create record
	rec := b.NewRecord()
	defer rec.Release()

	// Append to index
	err = idx.AppendFromArrow(rec)
	assert.NoError(t, err)

	// Force flush
	idx.flushBatch()

	// Verify data
	results, err := idx.Search([]float32{0.1, 0.2, 0.3}, 1)
	assert.NoError(t, err)
	assert.Equal(t, uint64(1), results[0].ID)
	assert.Equal(t, "test1", results[0].Metadata["category"])
}

func BenchmarkAppendFromArrow(b *testing.B) {
	// Create test index
	idx, err := New(Config{
		Dimension:       128,
		StoragePath:     "",
		MaxElements:     100000,
		HNSWM:           32,
		HNSWEfConstruct: 200,
		HNSWEfSearch:    200,
		BatchSize:       1000,
		Distance:        Cosine,
	}, benchLogger)
	if err != nil {
		b.Fatal(err)
	}
	defer idx.Close()

	// Create Arrow record with random data
	pool := memory.NewGoAllocator()
	schema := NewVectorSchema(128)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		b.StopTimer()
		builder := array.NewRecordBuilder(pool, schema)

		// Add 1000 rows of random data
		numRows := 1000
		ids := make([]uint64, numRows)
		for j := range ids {
			ids[j] = uint64(j)
		}
		builder.Field(0).(*array.Uint64Builder).AppendValues(ids, nil)

		// Create the vector list properly
		listBuilder := builder.Field(1).(*array.FixedSizeListBuilder)
		valueBuilder := listBuilder.ValueBuilder().(*array.Float32Builder)

		// Add vectors
		for j := 0; j < numRows; j++ {
			for k := 0; k < 128; k++ {
				valueBuilder.Append(rand.Float32())
			}
			listBuilder.Append(true)
		}

		// Add metadata
		metadata := make([]string, numRows)
		for j := range metadata {
			metadata[j] = fmt.Sprintf(`{"category":"bench%d"}`, j)
		}
		builder.Field(2).(*array.StringBuilder).AppendValues(metadata, nil)

		rec := builder.NewRecord()
		b.StartTimer()

		if err := idx.AppendFromArrow(rec); err != nil {
			b.Fatal(err)
		}

		b.StopTimer()
		builder.Release()
		rec.Release()
	}
}
