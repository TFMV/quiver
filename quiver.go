package quiver

import (
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sync"
	"time"

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
}

// New initializes a vector index with HNSW & DuckDB using tunable hyperparameters.
func New(config Config) (*Index, error) {
	// Validate config
	if config.Dimension <= 0 {
		return nil, errors.New("dimension must be > 0")
	}
	if config.BatchSize <= 0 {
		config.BatchSize = 100 // Default batch size
	}
	if config.HNSWM <= 16 {
		config.HNSWM = 32 // Default M
	}
	if config.HNSWEfConstruct <= 0 {
		config.HNSWEfConstruct = 200 // Default efConstruction
	}
	if config.HNSWEfSearch <= 100 {
		config.HNSWEfSearch = 200 // Default ef for search
	}
	if config.MaxElements == 0 {
		config.MaxElements = 100000 // Default max elements
	}

	// Initialize HNSW index with proper parameters
	var spaceType hnswgo.SpaceType
	switch config.Distance {
	case Cosine:
		spaceType = hnswgo.Cosine
	case L2:
		spaceType = hnswgo.L2
	default:
		spaceType = hnswgo.Cosine
	}

	hnsw := hnswgo.New(
		config.Dimension,
		config.HNSWM,
		config.HNSWEfConstruct,
		42,
		config.MaxElements,
		spaceType,
		true,
	)

	// Initialize DuckDB with proper cleanup
	db, err := sql.Open("duckdb", config.StoragePath)
	if err != nil {
		hnsw.Free() // Clean up HNSW
		return nil, fmt.Errorf("failed to open DuckDB: %w", err)
	}

	// Create metadata table with proper schema
	_, err = db.Exec(`CREATE TABLE IF NOT EXISTS metadata (
		id INTEGER PRIMARY KEY,
		json TEXT NOT NULL
	)`)
	if err != nil {
		hnsw.Free() // Clean up HNSW
		db.Close()  // Clean up DB
		return nil, fmt.Errorf("failed to create metadata table: %w", err)
	}

	idx := &Index{
		config:      config,
		hnsw:        hnsw,
		metadata:    make(map[uint64]map[string]interface{}),
		db:          db,
		batchBuffer: make([]vectorMeta, 0, config.BatchSize),
		batchTicker: time.NewTicker(100 * time.Millisecond), // Configurable flush interval
		batchDone:   make(chan struct{}),
	}

	// Start background batch processor
	go idx.batchProcessor()

	return idx, nil
}

// batchProcessor processes batch insertions in the background.
func (idx *Index) batchProcessor() {
	for {
		select {
		case <-idx.batchTicker.C:
			idx.flushBatch()
		case <-idx.batchDone:
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

	// Begin a transaction for batch insertion.
	tx, err := idx.db.Begin()
	if err != nil {
		return err
	}
	stmt, err := tx.Prepare(`INSERT INTO metadata (id, json) VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET json = excluded.json`)
	if err != nil {
		tx.Rollback()
		return err
	}
	defer stmt.Close()

	// Process each buffered vector.
	for i, item := range idx.batchBuffer {
		vectors[i] = item.vector
		ids[i] = item.id
		idx.metadata[item.id] = item.meta
		idx.cache.Store(item.id, item.meta)

		metaJSON, _ := json.Marshal(item.meta)
		if _, err := stmt.Exec(item.id, string(metaJSON)); err != nil {
			tx.Rollback()
			return err
		}
	}

	// Insert vectors in a single batch.
	if err := idx.hnsw.AddPoints(vectors, ids, 1, true); err != nil {
		tx.Rollback()
		return err
	}

	if err := tx.Commit(); err != nil {
		return err
	}

	// Clear the batch buffer.
	idx.batchBuffer = idx.batchBuffer[:0]
	return nil
}

// Add inserts a vector with metadata. It batches insertions to improve throughput.
func (idx *Index) Add(id uint64, vector []float32, meta map[string]interface{}) error {
	if len(vector) != idx.config.Dimension {
		return errors.New("dimension mismatch")
	}

	idx.batchLock.Lock()
	idx.batchBuffer = append(idx.batchBuffer, vectorMeta{id: id, vector: vector, meta: meta})
	currentBatchSize := len(idx.batchBuffer)
	idx.batchLock.Unlock()

	// When the batch size threshold is reached, flush asynchronously.
	if currentBatchSize >= idx.config.BatchSize {
		go func() {
			idx.lock.Lock()
			defer idx.lock.Unlock()
			if err := idx.flushBatch(); err != nil {
				fmt.Printf("Batch flush error: %v\n", err)
			}
		}()
	}
	return nil
}

// Search finds the nearest neighbors using HNSW.
func (idx *Index) Search(query []float32, k int) ([]SearchResult, error) {
	idx.lock.RLock()
	defer idx.lock.RUnlock()

	results, err := idx.hnsw.SearchKNN([][]float32{query}, k, idx.config.HNSWEfSearch)
	if err != nil {
		return nil, err
	}

	var searchResults []SearchResult
	for _, r := range results[0] {
		// Retrieve metadata from in-memory map or cache.
		meta, ok := idx.metadata[r.Label]
		if !ok {
			if cached, found := idx.cache.Load(r.Label); found {
				meta = cached.(map[string]interface{})
			}
		}
		searchResults = append(searchResults, SearchResult{
			ID:       r.Label,
			Distance: r.Distance,
			Metadata: meta,
		})
	}

	return searchResults, nil
}

// SearchWithFilter performs a hybrid search: first vector search then metadata filtering via DuckDB.
func (idx *Index) SearchWithFilter(query []float32, k int, filter string) ([]SearchResult, error) {
	// First try metadata filtering
	sqlQuery := `
		SELECT id 
		FROM metadata 
		WHERE JSON_EXTRACT_STRING(json, '$.category') = ?
		LIMIT ?`

	rows, err := idx.db.Query(sqlQuery, filter, k*10) // Get more candidates for better results
	if err != nil {
		return nil, fmt.Errorf("metadata filter failed: %w", err)
	}
	defer rows.Close()

	// Collect filtered IDs
	var filteredIDs []uint64
	for rows.Next() {
		var id uint64
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		filteredIDs = append(filteredIDs, id)
	}

	// If metadata filter is selective enough, search only those vectors
	if len(filteredIDs) < k*10 {
		// Get vectors for filtered IDs
		vectors := make([][]float32, 1)
		vectors[0] = query // Put query vector first

		// Search among filtered vectors
		results, err := idx.hnsw.SearchKNN(vectors, k, idx.config.HNSWEfSearch)
		if err != nil {
			return nil, err
		}

		// Filter results by ID
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

	// If too many matches, do vector search first then filter
	results, err := idx.Search(query, k*2)
	if err != nil {
		return nil, err
	}

	// Filter results by metadata
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

// Helper method to get metadata
func (idx *Index) getMetadata(id uint64) map[string]interface{} {
	// Try in-memory cache first
	if meta, ok := idx.metadata[id]; ok {
		return meta
	}
	// Try sync.Map cache
	if cached, found := idx.cache.Load(id); found {
		return cached.(map[string]interface{})
	}
	return nil
}

// Save persists the HNSW index and metadata to disk.
func (idx *Index) Save(path string) error {
	// Flush any pending batch insertions.
	idx.lock.Lock()
	idx.flushBatch()
	idx.lock.Unlock()

	idx.hnsw.Save(path + "/index.hnsw")

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
		// Note: The DuckDB connection should be re-established by the caller if needed.
	}, nil
}

// Close releases resources associated with the index.
func (idx *Index) Close() error {
	// Stop batch processor
	idx.batchTicker.Stop()
	close(idx.batchDone)

	// Flush any remaining vectors
	idx.flushBatch()

	idx.hnsw.Free()
	return idx.db.Close()
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
	// Check that the record has at least 3 columns.
	if rec.NumCols() < 3 {
		return errors.New("arrow record must have at least 3 columns: id, vector, metadata")
	}

	// Extract the columns.
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

	// Get the fixed size (dimension) from the vector column’s type.
	fsType := vectorCol.DataType().(*arrow.FixedSizeListType)
	dim := int(fsType.Len())

	// The actual float32 values are stored in the underlying values array.
	valuesArr, ok := vectorCol.ListValues().(*array.Float32)
	if !ok {
		return errors.New("expected the underlying vector array to be of type Float32")
	}

	numRows := int(rec.NumRows())
	for i := 0; i < numRows; i++ {
		// Get the id value.
		if idCol.IsNull(i) {
			return fmt.Errorf("id column contains null value at row %d", i)
		}
		id := idCol.Value(i)

		// For each row, extract the vector values.
		vector := make([]float32, dim)
		if vectorCol.IsNull(i) {
			return fmt.Errorf("vector column contains null value at row %d", i)
		}
		start := i * dim
		for j := 0; j < dim; j++ {
			vector[j] = valuesArr.Value(start + j)
		}

		// Get metadata (as a JSON string) and unmarshal into a map.
		meta := make(map[string]interface{})
		if !metadataCol.IsNull(i) {
			metaJSON := metadataCol.Value(i)
			if err := json.Unmarshal([]byte(metaJSON), &meta); err != nil {
				// Fallback: if unmarshaling fails, store the raw JSON string.
				meta = map[string]interface{}{"metadata": metaJSON}
			}
		}

		// Append the row to the index using the existing Add method.
		if err := idx.Add(id, vector, meta); err != nil {
			return fmt.Errorf("failed to add vector with id %d: %w", id, err)
		}
	}
	return nil
}
