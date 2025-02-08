package quiver

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"time"

	hnsw "github.com/TFMV/hnswgo"
	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	_ "github.com/marcboeker/go-duckdb"
	"go.uber.org/zap"
)

// VectorIndex represents an HNSW-based vector search index
// with persistence and DuckDB integration.
type VectorIndex struct {
	index     *hnsw.HnswIndex
	db        *sql.DB
	dbPath    string
	indexPath string
	logger    *zap.Logger
}

// SpaceType represents the distance metric type
type SpaceType int

const (
	Cosine SpaceType = iota
	Euclidean
)

// initDB initializes the DuckDB database and creates required tables
func initDB(dbPath string) (*sql.DB, error) {
	db, err := sql.Open("duckdb", dbPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open DuckDB: %v", err)
	}

	// Create table for vector storage
	_, err = db.Exec(`CREATE TABLE IF NOT EXISTS vectors (
		id INTEGER PRIMARY KEY,
		vector TEXT
	)`)
	if err != nil {
		return nil, fmt.Errorf("failed to create DuckDB table: %v", err)
	}

	return db, nil
}

// NewVectorIndex initializes a vector index with DuckDB storage.
func NewVectorIndex(dim int, dbPath, indexPath string) (*VectorIndex, error) {
	logger, _ := zap.NewProduction()
	defer logger.Sync()

	logger.Info("Initializing HNSW Index", zap.Int("maxElements", 10000))

	index := hnsw.New(dim, 16, 10000, 10, uint64(hnsw.Cosine), hnsw.SpaceType(hnsw.Cosine), true)
	index.ResizeIndex(50000)

	// Load existing index if available
	if _, err := os.Stat(indexPath); err == nil {
		fmt.Println("Loading existing index...")
		if idx, err := hnsw.Load(indexPath, hnsw.Cosine, dim, 10000, true); err == nil {
			index = idx
		} else {
			logger.Info("Failed to load index, starting fresh.")
		}
	}

	db, err := initDB(dbPath) // Use the helper function
	if err != nil {
		return nil, err
	}

	return &VectorIndex{index: index, db: db, dbPath: dbPath, indexPath: indexPath, logger: logger}, nil
}

// NewVectorIndexWithMetric creates a vector index with specified distance metric
func NewVectorIndexWithMetric(dim int, dbPath, indexPath string, metric SpaceType) (*VectorIndex, error) {
	var spaceType uint64
	switch metric {
	case Euclidean:
		spaceType = 1 // L2Space in hnswgo
	default: // Cosine
		spaceType = 3 // CosineSpace in hnswgo
	}

	index := hnsw.New(dim, 16, 10000, 10, spaceType, hnsw.SpaceType(spaceType), true)
	index.ResizeIndex(50000)

	db, err := initDB(dbPath) // Use the helper function
	if err != nil {
		return nil, err
	}

	return &VectorIndex{index: index, db: db, dbPath: dbPath, indexPath: indexPath}, nil
}

// AddVector inserts a vector into HNSW and DuckDB
func (vi *VectorIndex) AddVector(id int, vector []float32) {
	err := vi.index.AddPoints([][]float32{vector}, []uint64{uint64(id)}, 2, true)
	if err != nil {
		vi.logger.Error("Failed to add vector", zap.Error(err))
	}

	vectorJSON, _ := json.Marshal(vector)
	_, err = vi.db.Exec("INSERT INTO vectors (id, vector) VALUES (?, ?) ON CONFLICT(id) DO NOTHING", id, string(vectorJSON))
	if err != nil {
		vi.logger.Error("Failed to insert vector into DuckDB", zap.Error(err))
	}
}

// Search finds k nearest neighbors
func (vi *VectorIndex) Search(query []float32, k int) []int {
	results, err := vi.index.SearchKNN([][]float32{query}, k, 2)
	if err != nil {
		vi.logger.Error("Search error", zap.Error(err))
		return nil
	}

	neighbors := make([]int, len(results[0]))
	for i, r := range results[0] {
		neighbors[i] = int(r.Label)
	}
	return neighbors
}

// Save persists the index to disk
func (vi *VectorIndex) Save() {
	fmt.Println("Saving HNSW index...")
	vi.index.Save(vi.indexPath)
}

// Benchmark measures search performance with 10,000 vectors
func (vi *VectorIndex) Benchmark(dim int) {
	fmt.Println("Running benchmark with 10,000 vectors...")
	numVectors := 10000

	// Generate random vectors
	vectors := make([][]float32, numVectors)
	for i := range vectors {
		vectors[i] = make([]float32, dim)
		for j := range vectors[i] {
			vectors[i][j] = float32(i) * 0.01
		}
		vi.AddVector(i, vectors[i])
	}

	// Benchmark search
	start := time.Now()
	query := vectors[5000]
	_ = vi.Search(query, 10)
	duration := time.Since(start)
	fmt.Printf("Search completed in: %v\n", duration)
}

// LoadVectorsFromArrow extracts vector data from Apache Arrow
func LoadVectorsFromArrow(record arrow.Record) ([][]float32, error) {
	vecColumn := record.Column(1)
	vecArray, ok := vecColumn.(*array.FixedSizeList)
	if !ok {
		return nil, fmt.Errorf("expected FixedSizeList, got %T", vecColumn)
	}

	vectors := make([][]float32, vecArray.Len())
	values := vecArray.ListValues().(*array.Float32)
	listSize := vecArray.DataType().(*arrow.FixedSizeListType).Len()

	for i := 0; i < vecArray.Len(); i++ {
		start := i * int(listSize)
		end := start + int(listSize)
		vectors[i] = values.Float32Values()[start:end]
	}

	return vectors, nil
}

// LoadVectorIndex loads a saved HNSW index from disk
func LoadVectorIndex(dim int, indexPath string) (*VectorIndex, error) {
	// Load HNSW index
	index, err := hnsw.Load(indexPath, hnsw.Cosine, dim, 10000, true)
	if err != nil {
		return nil, fmt.Errorf("failed to load index: %v", err)
	}

	// Create new VectorIndex with loaded data
	return &VectorIndex{
		index:     index,
		indexPath: indexPath,
	}, nil
}
