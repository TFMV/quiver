package quiver

import (
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"

	"github.com/TFMV/hnswgo"
	"github.com/apache/arrow/go/v18/arrow"
	"github.com/apache/arrow/go/v18/arrow/array"
	"github.com/apache/arrow/go/v18/arrow/memory"
	_ "github.com/marcboeker/go-duckdb"
)

// DistanceMetric defines the similarity metrics.
type DistanceMetric int

const (
	Cosine DistanceMetric = iota
	L2
)

// Config holds the index settings.
type Config struct {
	Dimension   int
	StoragePath string
	Distance    DistanceMetric
	MaxElements uint64
}

// Index represents the vector search index.
type Index struct {
	config   Config
	hnsw     *hnswgo.HnswIndex
	metadata map[uint64]map[string]interface{}
	db       *sql.DB
	lock     sync.RWMutex
}

// New initializes a vector index with HNSW & DuckDB.
func New(config Config) (*Index, error) {
	hnsw := hnswgo.New(config.Dimension, 16, 200, 42, config.MaxElements, hnswgo.Cosine, true)

	db, err := sql.Open("duckdb", config.StoragePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open DuckDB: %w", err)
	}

	_, err = db.Exec(`CREATE TABLE IF NOT EXISTS metadata (id INTEGER PRIMARY KEY, json TEXT)`)
	if err != nil {
		return nil, fmt.Errorf("failed to create metadata table: %w", err)
	}

	return &Index{
		config:   config,
		hnsw:     hnsw,
		metadata: make(map[uint64]map[string]interface{}),
		db:       db,
	}, nil
}

// Add inserts a vector with metadata.
func (idx *Index) Add(id uint64, vector []float32, meta map[string]interface{}) error {
	idx.lock.Lock()
	defer idx.lock.Unlock()

	if len(vector) != idx.config.Dimension {
		return errors.New("dimension mismatch")
	}
	if err := idx.hnsw.AddPoints([][]float32{vector}, []uint64{id}, 1, true); err != nil {
		return err
	}
	idx.metadata[id] = meta

	// Store metadata in DuckDB
	metaJSON, _ := json.Marshal(meta)
	_, err := idx.db.Exec(`INSERT INTO metadata (id, json) VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET json = excluded.json`, id, string(metaJSON))
	return err
}

// Search finds the nearest neighbors.
func (idx *Index) Search(query []float32, k int) ([]SearchResult, error) {
	idx.lock.RLock()
	defer idx.lock.RUnlock()

	results, err := idx.hnsw.SearchKNN([][]float32{query}, k, 1)
	if err != nil {
		return nil, err
	}

	var searchResults []SearchResult
	for _, r := range results[0] {
		searchResults = append(searchResults, SearchResult{
			ID:       r.Label,
			Distance: r.Distance,
			Metadata: idx.metadata[r.Label],
		})
	}

	return searchResults, nil
}

// Hybrid search with SQL-based metadata filtering.
func (idx *Index) SearchWithFilter(query []float32, k int, filter string) ([]SearchResult, error) {
	// Get initial results
	results, err := idx.Search(query, k*2)
	if err != nil {
		return nil, fmt.Errorf("vector search failed: %w", err)
	}

	// Build a query with IDs from the vector search
	var ids []string
	for _, r := range results {
		ids = append(ids, fmt.Sprint(r.ID))
	}

	if len(ids) == 0 {
		return nil, nil
	}

	// Use JSON_EXTRACT_STRING for DuckDB JSON handling
	sqlQuery := fmt.Sprintf(`
		SELECT m.id 
		FROM metadata m 
		WHERE m.id IN (%s) 
		AND JSON_EXTRACT_STRING(m.json, '$.category') = ?`,
		strings.Join(ids, ","),
	)

	rows, err := idx.db.Query(sqlQuery, filter)
	if err != nil {
		return nil, fmt.Errorf("metadata filter failed: %w", err)
	}
	defer rows.Close()

	validIDs := make(map[uint64]struct{})
	for rows.Next() {
		var id uint64
		if err := rows.Scan(&id); err == nil {
			validIDs[id] = struct{}{}
		}
	}

	var filteredResults []SearchResult
	for _, res := range results {
		if _, exists := validIDs[res.ID]; exists {
			filteredResults = append(filteredResults, res)
			if len(filteredResults) >= k {
				break
			}
		}
	}
	return filteredResults, nil
}

// Save persists the index and metadata.
func (idx *Index) Save(path string) error {
	idx.hnsw.Save(path + "/index.hnsw")

	// Save metadata
	metaFile, err := os.Create(path + "/metadata.json")
	if err != nil {
		return err
	}
	defer metaFile.Close()
	return json.NewEncoder(metaFile).Encode(idx.metadata)
}

// Load restores an index from disk.
func Load(path string) (*Index, error) {
	hnsw, err := hnswgo.Load(path+"/index.hnsw", hnswgo.Cosine, 128, 100000, true)
	if err != nil {
		return nil, err
	}

	metaFile, err := os.Open(path + "/metadata.json")
	if err != nil {
		return nil, err
	}
	defer metaFile.Close()

	var metadata map[uint64]map[string]interface{}
	if err := json.NewDecoder(metaFile).Decode(&metadata); err != nil {
		return nil, err
	}

	return &Index{
		hnsw:     hnsw,
		metadata: metadata,
	}, nil
}

// ToArrow exports vectors as an Apache Arrow table.
func (idx *Index) ToArrow() (arrow.Record, error) {
	idx.lock.RLock()
	defer idx.lock.RUnlock()

	pool := memory.NewGoAllocator()
	builder := array.NewFloat32Builder(pool)

	var ids []uint64
	for id := range idx.metadata {
		ids = append(ids, id)
	}

	// Construct Arrow table
	for _, id := range ids {
		vec := idx.hnsw.GetDataByLabel(id)
		builder.AppendValues(vec, nil)
	}

	idBuilder := array.NewUint64Builder(pool)
	idBuilder.AppendValues(ids, nil)

	schema := arrow.NewSchema([]arrow.Field{
		{Name: "id", Type: arrow.PrimitiveTypes.Uint64},
		{Name: "vector", Type: arrow.PrimitiveTypes.Float32},
	}, nil)

	record := array.NewRecord(schema, []arrow.Array{idBuilder.NewArray(), builder.NewArray()}, int64(len(ids)))
	return record, nil
}

// Close releases resources.
func (idx *Index) Close() error {
	idx.hnsw.Free()
	return idx.db.Close()
}

// SearchResult holds search output.
type SearchResult struct {
	ID       uint64
	Distance float32
	Metadata map[string]interface{}
}
