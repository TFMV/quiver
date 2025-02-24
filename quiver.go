// Package quiver provides a lightweight, high-performance vector search engine for structured datasets.
// It uses HNSW for efficient vector indexing and DuckDB for metadata storage.
//
// Quiver is designed to be used in conjunction with Arrow tables, which are a popular format for storing and
// processing tabular data in Go.
//
// Quiver supports two distance metrics:
// - Cosine distance: Measures the angle between two vectors.
// - L2 distance: Measures the Euclidean distance between two vectors.
package quiver

import (
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/TFMV/hnswgo"
	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	_ "github.com/marcboeker/go-duckdb"
)

// DistanceMetric defines the similarity metrics.
type DistanceMetric int

const (
	Cosine DistanceMetric = iota
	L2
)

// Config holds the index settings and tunable hyperparameters.
type Config struct {
	Dimension       int
	StoragePath     string
	Distance        DistanceMetric
	MaxElements     uint64
	HNSWM           int // HNSW hyperparameter M
	HNSWEfConstruct int // HNSW hyperparameter efConstruction
	HNSWEfSearch    int // HNSW hyperparameter ef used during queries
	BatchSize       int // Number of vectors to batch before insertion
}

// vectorMeta holds a vector and its associated metadata.
type vectorMeta struct {
	id     uint64
	vector []float32
	meta   map[string]interface{}
}

// Index represents the vector search index with caching and batch support.
type Index struct {
	config      Config
	hnsw        *hnswgo.HnswIndex
	metadata    map[uint64]map[string]interface{}
	db          *sql.DB
	lock        sync.RWMutex
	batchBuffer []vectorMeta
	batchLock   sync.Mutex
	cache       sync.Map // Caches metadata: key = id, value = metadata map
	batchTicker *time.Ticker
	batchDone   chan struct{}
	logger      *zap.Logger
}

// New initializes a vector index with HNSW and DuckDB using tunable hyperparameters.
// A zap.Logger must be provided for structured logging.
func New(config Config, logger *zap.Logger) (*Index, error) {
	// Validate configuration.
	if config.Dimension <= 0 {
		return nil, errors.New("dimension must be > 0")
	}
	if config.BatchSize <= 0 {
		config.BatchSize = 100 // default batch size
	}
	if config.HNSWM <= 16 {
		config.HNSWM = 32 // default M
	}
	if config.HNSWEfConstruct <= 0 {
		config.HNSWEfConstruct = 200 // default efConstruction
	}
	if config.HNSWEfSearch <= 100 {
		config.HNSWEfSearch = 200 // default ef for search
	}
	if config.MaxElements == 0 {
		config.MaxElements = 100000 // default max elements
	}

	// Set the HNSW space type.
	var spaceType hnswgo.SpaceType
	switch config.Distance {
	case Cosine:
		spaceType = hnswgo.Cosine
	case L2:
		spaceType = hnswgo.L2
	default:
		spaceType = hnswgo.Cosine
	}

	// Initialize HNSW index.
	hnsw := hnswgo.New(
		config.Dimension,
		config.HNSWM,
		config.HNSWEfConstruct,
		42,
		config.MaxElements,
		spaceType,
		true,
	)

	// Open DuckDB connection.
	db, err := sql.Open("duckdb", config.StoragePath)
	if err != nil {
		hnsw.Free()
		return nil, fmt.Errorf("failed to open DuckDB: %w", err)
	}

	// Create metadata table if it does not exist.
	_, err = db.Exec(`CREATE TABLE IF NOT EXISTS metadata (
		id INTEGER PRIMARY KEY,
		json TEXT NOT NULL
	)`)
	if err != nil {
		hnsw.Free()
		db.Close()
		return nil, fmt.Errorf("failed to create metadata table: %w", err)
	}

	idx := &Index{
		config:      config,
		hnsw:        hnsw,
		metadata:    make(map[uint64]map[string]interface{}),
		db:          db,
		batchBuffer: make([]vectorMeta, 0, config.BatchSize),
		batchTicker: time.NewTicker(100 * time.Millisecond), // configurable flush interval
		batchDone:   make(chan struct{}),
		logger:      logger,
	}

	// Start background batch processor.
	go idx.batchProcessor()

	logger.Info("Quiver index initialized",
		zap.Int("dimension", config.Dimension),
		zap.Int("batchSize", config.BatchSize))
	return idx, nil
}

// batchProcessor processes batch insertions in the background.
func (idx *Index) batchProcessor() {
	for {
		select {
		case <-idx.batchTicker.C:
			if err := idx.flushBatch(); err != nil {
				idx.logger.Error("failed to flush batch", zap.Error(err))
			}
		case <-idx.batchDone:
			idx.logger.Info("batch processor shutting down")
			return
		}
	}
}

// flushBatch flushes the accumulated batch to the HNSW index and DuckDB.
func (idx *Index) flushBatch() error {
	idx.batchLock.Lock()
	defer idx.batchLock.Unlock()

	if len(idx.batchBuffer) == 0 {
		return nil
	}

	n := len(idx.batchBuffer)
	vectors := make([][]float32, n)
	ids := make([]uint64, n)

	tx, err := idx.db.Begin()
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	stmt, err := tx.Prepare(`INSERT INTO metadata (id, json) VALUES (?, ?)
		ON CONFLICT(id) DO UPDATE SET json = excluded.json`)
	if err != nil {
		tx.Rollback()
		return fmt.Errorf("failed to prepare statement: %w", err)
	}
	defer stmt.Close()

	for i, item := range idx.batchBuffer {
		vectors[i] = item.vector
		ids[i] = item.id
		idx.metadata[item.id] = item.meta
		idx.cache.Store(item.id, item.meta)

		metaJSON, err := json.Marshal(item.meta)
		if err != nil {
			idx.logger.Error("failed to marshal metadata", zap.Uint64("id", item.id), zap.Error(err))
			tx.Rollback()
			return err
		}
		if _, err := stmt.Exec(item.id, string(metaJSON)); err != nil {
			idx.logger.Error("failed to execute statement", zap.Uint64("id", item.id), zap.Error(err))
			tx.Rollback()
			return err
		}
	}

	if err := idx.hnsw.AddPoints(vectors, ids, 1, true); err != nil {
		tx.Rollback()
		return fmt.Errorf("failed to add points to HNSW: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}

	idx.logger.Info("batch flushed", zap.Int("num_points", n))
	idx.batchBuffer = idx.batchBuffer[:0]
	return nil
}

// Add inserts a vector with its metadata into the index.
// Vectors are batched before being inserted to improve throughput.
func (idx *Index) Add(id uint64, vector []float32, meta map[string]interface{}) error {
	if len(vector) != idx.config.Dimension {
		return errors.New("dimension mismatch")
	}

	idx.batchLock.Lock()
	idx.batchBuffer = append(idx.batchBuffer, vectorMeta{id: id, vector: vector, meta: meta})
	currentBatchSize := len(idx.batchBuffer)
	idx.batchLock.Unlock()

	if currentBatchSize >= idx.config.BatchSize {
		go func() {
			idx.lock.Lock()
			defer idx.lock.Unlock()
			if err := idx.flushBatch(); err != nil {
				idx.logger.Error("async batch flush error", zap.Error(err))
			}
		}()
	}
	return nil
}

// Search performs an approximate nearest neighbor search using the HNSW index.
func (idx *Index) Search(query []float32, k int) ([]SearchResult, error) {
	idx.lock.RLock()
	defer idx.lock.RUnlock()

	results, err := idx.hnsw.SearchKNN([][]float32{query}, k, idx.config.HNSWEfSearch)
	if err != nil {
		return nil, fmt.Errorf("failed to perform search: %w", err)
	}

	var searchResults []SearchResult
	for _, r := range results[0] {
		meta := idx.getMetadata(r.Label)
		searchResults = append(searchResults, SearchResult{
			ID:       r.Label,
			Distance: r.Distance,
			Metadata: meta,
		})
	}
	return searchResults, nil
}

// SearchWithFilter performs a hybrid search: first using vector search then filtering via DuckDB metadata.
func (idx *Index) SearchWithFilter(query []float32, k int, filter string) ([]SearchResult, error) {
	sqlQuery := `
		SELECT id 
		FROM metadata 
		WHERE JSON_EXTRACT_STRING(json, '$.category') = ?
		LIMIT ?`
	rows, err := idx.db.Query(sqlQuery, filter, k*10)
	if err != nil {
		return nil, fmt.Errorf("metadata filter query failed: %w", err)
	}
	defer rows.Close()

	var filteredIDs []uint64
	for rows.Next() {
		var id uint64
		if err := rows.Scan(&id); err != nil {
			return nil, fmt.Errorf("failed to scan row: %w", err)
		}
		filteredIDs = append(filteredIDs, id)
	}

	// If the metadata filter is selective, perform a targeted vector search.
	if len(filteredIDs) < k*10 {
		results, err := idx.hnsw.SearchKNN([][]float32{query}, k, idx.config.HNSWEfSearch)
		if err != nil {
			return nil, fmt.Errorf("failed to search in HNSW: %w", err)
		}

		var filteredResults []SearchResult
		for _, r := range results[0] {
			for _, id := range filteredIDs {
				if r.Label == id {
					filteredResults = append(filteredResults, SearchResult{
						ID:       r.Label,
						Distance: r.Distance,
						Metadata: idx.getMetadata(r.Label),
					})
					break
				}
			}
			if len(filteredResults) >= k {
				break
			}
		}
		return filteredResults, nil
	}

	// Otherwise, perform a full vector search and filter the results by metadata.
	results, err := idx.Search(query, k*2)
	if err != nil {
		return nil, err
	}

	var filteredResults []SearchResult
	for _, res := range results {
		meta := idx.getMetadata(res.ID)
		if meta != nil {
			if category, ok := meta["category"].(string); ok && category == filter {
				filteredResults = append(filteredResults, res)
				if len(filteredResults) >= k {
					break
				}
			}
		}
	}
	return filteredResults, nil
}

// getMetadata retrieves metadata from the in-memory maps.
func (idx *Index) getMetadata(id uint64) map[string]interface{} {
	if meta, ok := idx.metadata[id]; ok {
		return meta
	}
	if cached, ok := idx.cache.Load(id); ok {
		return cached.(map[string]interface{})
	}
	return nil
}

// Save persists the HNSW index and metadata to disk.
func (idx *Index) Save(path string) error {
	idx.lock.Lock()
	defer idx.lock.Unlock()

	if err := idx.flushBatch(); err != nil {
		idx.logger.Error("failed to flush batch before save", zap.Error(err))
		return err
	}

	idx.hnsw.Save(path + "/index.hnsw")

	metaFile, err := os.Create(path + "/metadata.json")
	if err != nil {
		return fmt.Errorf("failed to create metadata file: %w", err)
	}
	defer metaFile.Close()

	if err := json.NewEncoder(metaFile).Encode(idx.metadata); err != nil {
		return fmt.Errorf("failed to encode metadata: %w", err)
	}
	idx.logger.Info("index saved successfully", zap.String("path", path))
	return nil
}

// Load restores an index from disk.
func Load(path string, logger *zap.Logger) (*Index, error) {
	hnsw, err := hnswgo.Load(path+"/index.hnsw", hnswgo.Cosine, 128, 100000, true)
	if err != nil {
		return nil, fmt.Errorf("failed to load HNSW index: %w", err)
	}

	metaFile, err := os.Open(path + "/metadata.json")
	if err != nil {
		return nil, fmt.Errorf("failed to open metadata file: %w", err)
	}
	defer metaFile.Close()

	var metadata map[uint64]map[string]interface{}
	if err := json.NewDecoder(metaFile).Decode(&metadata); err != nil {
		return nil, fmt.Errorf("failed to decode metadata: %w", err)
	}

	idx := &Index{
		hnsw:     hnsw,
		metadata: metadata,
		logger:   logger,
	}
	logger.Info("index loaded successfully", zap.String("path", path))
	return idx, nil
}

// Close releases resources associated with the index.
func (idx *Index) Close() error {
	idx.batchTicker.Stop()
	close(idx.batchDone)

	if err := idx.flushBatch(); err != nil {
		idx.logger.Error("failed to flush batch during close", zap.Error(err))
	}
	idx.hnsw.Free()
	if err := idx.db.Close(); err != nil {
		return fmt.Errorf("failed to close database: %w", err)
	}
	idx.logger.Info("index closed successfully")
	return nil
}

// SearchResult holds the output of a search.
type SearchResult struct {
	ID       uint64
	Distance float32
	Metadata map[string]interface{}
}

// NewVectorSchema creates an Arrow schema for vector data.
func NewVectorSchema(dimension int) *arrow.Schema {
	return arrow.NewSchema(
		[]arrow.Field{
			{Name: "id", Type: arrow.PrimitiveTypes.Uint64, Nullable: false},
			{Name: "vector", Type: arrow.FixedSizeListOf(int32(dimension), arrow.PrimitiveTypes.Float32), Nullable: false},
			{Name: "metadata", Type: arrow.BinaryTypes.String, Nullable: true},
		},
		nil,
	)
}

// AppendFromArrow appends vectors and metadata from an Arrow record to the index.
func (idx *Index) AppendFromArrow(rec arrow.Record) error {
	if rec.NumCols() < 3 {
		return errors.New("arrow record must have at least 3 columns: id, vector, metadata")
	}

	idCol, ok := rec.Column(0).(*array.Uint64)
	if !ok {
		return errors.New("expected column 0 (id) to be an Uint64 array")
	}
	vectorCol, ok := rec.Column(1).(*array.FixedSizeList)
	if !ok {
		return errors.New("expected column 1 (vector) to be a FixedSizeList array")
	}
	metadataCol, ok := rec.Column(2).(*array.String)
	if !ok {
		return errors.New("expected column 2 (metadata) to be a String array")
	}

	fsType, ok := vectorCol.DataType().(*arrow.FixedSizeListType)
	if !ok {
		return errors.New("failed to get FixedSizeList type from vector column")
	}
	dim := int(fsType.Len())

	valuesArr, ok := vectorCol.ListValues().(*array.Float32)
	if !ok {
		return errors.New("expected underlying vector array to be of type Float32")
	}

	numRows := int(rec.NumRows())
	for i := 0; i < numRows; i++ {
		if idCol.IsNull(i) {
			return fmt.Errorf("id column contains null value at row %d", i)
		}
		id := idCol.Value(i)

		vector := make([]float32, dim)
		if vectorCol.IsNull(i) {
			return fmt.Errorf("vector column contains null value at row %d", i)
		}
		start := i * dim
		for j := 0; j < dim; j++ {
			vector[j] = valuesArr.Value(start + j)
		}

		meta := make(map[string]interface{})
		if !metadataCol.IsNull(i) {
			metaJSON := metadataCol.Value(i)
			if err := json.Unmarshal([]byte(metaJSON), &meta); err != nil {
				idx.logger.Warn("failed to unmarshal metadata, storing raw JSON", zap.Uint64("id", id), zap.Error(err))
				meta = map[string]interface{}{"metadata": metaJSON}
			}
		}

		if err := idx.Add(id, vector, meta); err != nil {
			return fmt.Errorf("failed to add vector with id %d: %w", id, err)
		}
	}
	return nil
}
