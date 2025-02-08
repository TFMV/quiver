package main

import (
	"fmt"
	"log"

	"github.com/TFMV/quiver"
	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"
)

func main() {
	// Initialize Arrow memory allocator
	pool := memory.NewCheckedAllocator(memory.DefaultAllocator)

	// Define Arrow schema with a vector field
	schema := arrow.NewSchema([]arrow.Field{
		{Name: "id", Type: arrow.PrimitiveTypes.Int64},
		{Name: "vector", Type: arrow.FixedSizeListOf(3, arrow.PrimitiveTypes.Float32)},
	}, nil)

	// Create Arrow builders
	recordBuilder := array.NewRecordBuilder(pool, schema)
	defer recordBuilder.Release()

	// Insert some vectors
	vectors := [][]float32{
		{0.1, 0.2, 0.3},
		{0.9, 0.8, 0.7},
		{0.4, 0.5, 0.6},
	}

	for i, vec := range vectors {
		idBuilder := recordBuilder.Field(0).(*array.Int64Builder)
		vecBuilder := recordBuilder.Field(1).(*array.FixedSizeListBuilder)
		subBuilder := vecBuilder.ValueBuilder().(*array.Float32Builder)

		idBuilder.Append(int64(i))
		vecBuilder.Append(true)
		for _, v := range vec {
			subBuilder.Append(v)
		}
	}

	record := recordBuilder.NewRecord()
	defer record.Release()

	// Initialize vector index
	index, err := quiver.NewVectorIndex(3, "test.db", "test_index.hnsw")
	if err != nil {
		log.Fatalf("Failed to create vector index: %v", err)
	}

	// Add vectors to index
	for i, vec := range vectors {
		index.AddVector(i, vec)
	}

	// Search for nearest neighbors
	query := []float32{0.5, 0.5, 0.5}
	neighbors := index.Search(query, 2)

	fmt.Println("Nearest neighbors:", neighbors)
}
