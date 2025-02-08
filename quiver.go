// Package quiver implements an HNSW-based vector search index with DuckDB integration.
// The VectorIndex is safe for concurrent use.
package quiver

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"

	hnsw "github.com/TFMV/hnswgo"
	duckdb "github.com/marcboeker/go-duckdb"
	"go.uber.org/zap"
)

// SpaceType represents the distance metric type.
type SpaceType int

const (
	// Cosine distance metric.
	Cosine SpaceType = iota
	// Euclidean distance metric.
	Euclidean
)

// VectorIndex encapsulates an HNSW index, a DuckDB connection and an appender,
// along with a structured logger and separate mutexes for index vs. DB operations.
type VectorIndex struct {
	// indexMu guards operations on the HNSW index.
	indexMu sync.RWMutex
	// dbMu guards operations on the DuckDB appender (if not inherently thread–safe).
	dbMu sync.Mutex

	index     *hnsw.HnswIndex
	db        *sql.DB
	dbPath    string
	indexPath string
	appender  *duckdb.Appender
	logger    *zap.Logger

	// bufferPool is used to reuse JSON encoding buffers to reduce allocations.
	bufferPool sync.Pool
}

// initDB initializes DuckDB and creates the required table.
func initDB(dbPath string) (*sql.DB, *duckdb.Appender, error) {
	db, err := sql.Open("duckdb", dbPath)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to open DuckDB: %w", err)
	}

	// Create table with a unique primary key constraint.
	_, err = db.Exec(`CREATE TABLE IF NOT EXISTS vectors (
		id INTEGER PRIMARY KEY,
		vector TEXT
	)`)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create DuckDB table: %w", err)
	}

	connector, err := duckdb.NewConnector(dbPath, nil)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create DuckDB connector: %w", err)
	}

	conn, err := connector.Connect(context.Background())
	if err != nil {
		return nil, nil, fmt.Errorf("failed to connect to DuckDB: %w", err)
	}

	appender, err := duckdb.NewAppenderFromConn(conn, "", "vectors")
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create DuckDB appender: %w", err)
	}

	return db, appender, nil
}

// NewVectorIndex initializes a new VectorIndex with the provided parameters.
// It creates (or loads) an HNSW index and initializes a DuckDB-backed storage.
func NewVectorIndex(dim int, dbPath, indexPath string, metric SpaceType) (*VectorIndex, error) {
	// Create a production-ready zap logger.
	logger, err := zap.NewProduction()
	if err != nil {
		return nil, fmt.Errorf("failed to create logger: %w", err)
	}

	const maxElements = 200000

	logger.Info("Initializing HNSW Index", zap.Int("maxElements", maxElements))

	// Set space type based on the metric.
	var spaceType uint64
	switch metric {
	case Euclidean:
		spaceType = 1 // L2Space in hnswgo.
	default: // Cosine
		spaceType = 3 // CosineSpace in hnswgo.
	}

	// Initialize the HNSW index
	index := hnsw.New(dim, 32, maxElements, 100, spaceType, hnsw.SpaceType(spaceType), true)
	index.ResizeIndex(maxElements)

	// Attempt to load an existing index from disk
	if _, err := os.Stat(indexPath); err == nil {
		logger.Info("Loading existing index from disk", zap.String("path", indexPath))
		if idx, err := hnsw.Load(indexPath, hnsw.SpaceType(spaceType), dim, 10000, true); err == nil {
			index = idx
		} else {
			logger.Warn("Failed to load index, starting fresh", zap.Error(err))
		}
	}

	db, appender, err := initDB(dbPath)
	if err != nil {
		return nil, err
	}

	return &VectorIndex{
		index:     index,
		db:        db,
		appender:  appender,
		dbPath:    dbPath,
		indexPath: indexPath,
		logger:    logger,
		bufferPool: sync.Pool{
			New: func() interface{} {
				return new(bytes.Buffer)
			},
		},
	}, nil
}

// AddVector inserts a vector into both the DuckDB storage and the HNSW index.
// The DB update and index update are done under separate locks to reduce contention.
func (vi *VectorIndex) AddVector(id int, vector []float32) error {
	// Reuse a buffer from the pool to encode JSON.
	buf := vi.bufferPool.Get().(*bytes.Buffer)
	buf.Reset()
	encoder := json.NewEncoder(buf)
	if err := encoder.Encode(vector); err != nil {
		vi.logger.Error("Failed to encode vector", zap.Int("id", id), zap.Error(err))
		vi.bufferPool.Put(buf)
		return fmt.Errorf("failed to encode vector: %w", err)
	}
	// Remove trailing newline that Encoder adds.
	vectorJSON := strings.TrimSpace(buf.String())
	vi.bufferPool.Put(buf)

	// First, update DuckDB (the DB driver is thread-safe, but we synchronize flush operations).
	_, err := vi.db.Exec("INSERT OR REPLACE INTO vectors (id, vector) VALUES (?, ?)", id, vectorJSON)
	if err != nil {
		vi.logger.Error("Failed to insert vector into DuckDB", zap.Int("id", id), zap.Error(err))
		return fmt.Errorf("failed to insert vector into DuckDB: %w", err)
	}
	vi.dbMu.Lock()
	if err := vi.appender.Flush(); err != nil {
		vi.logger.Error("Failed to flush DuckDB appender", zap.Int("id", id), zap.Error(err))
		vi.dbMu.Unlock()
		return fmt.Errorf("failed to flush appender: %w", err)
	}
	vi.dbMu.Unlock()

	// Next, update the HNSW index.
	vi.indexMu.Lock()
	defer vi.indexMu.Unlock()
	if err := vi.index.AddPoints([][]float32{vector}, []uint64{uint64(id)}, 2, true); err != nil {
		vi.logger.Error("Failed to add vector to HNSW index", zap.Int("id", id), zap.Error(err))
		return fmt.Errorf("failed to add vector to index: %w", err)
	}

	return nil
}

// Search performs a k-nearest neighbors search on the HNSW index using the provided query vector.
// It then retrieves and returns the corresponding IDs from DuckDB.
// Search performs a k-nearest neighbors search on the HNSW index using the provided query vector.
// It then retrieves and returns the corresponding IDs from DuckDB.
func (vi *VectorIndex) Search(queryVector []float32, k int) ([]int, error) {
	vi.indexMu.RLock()
	hnswResults, err := vi.index.SearchKNN([][]float32{queryVector}, k, 2)
	vi.indexMu.RUnlock()
	if err != nil {
		vi.logger.Error("Search error in HNSW", zap.Error(err))
		return nil, fmt.Errorf("HNSW search error: %w", err)
	}

	if len(hnswResults) == 0 || len(hnswResults[0]) == 0 {
		vi.logger.Warn("No results found from HNSW search")
		return []int{}, nil
	}

	// Collect IDs from HNSW search results.
	ids := make([]uint64, len(hnswResults[0]))
	for i, r := range hnswResults[0] {
		ids[i] = r.Label
	}

	// 🔥 Use prepared statement for batch ID lookup
	args := make([]interface{}, len(ids))
	for i, id := range ids {
		args[i] = id
	}

	placeholders := placeHolders(len(ids))
	sqlQuery := fmt.Sprintf("SELECT id FROM vectors WHERE id IN (%s)", placeholders)
	rows, err := vi.db.Query(sqlQuery, args...)
	if err != nil {
		return nil, fmt.Errorf("DuckDB batch query error: %w", err)
	}
	defer rows.Close()

	// Efficiently map IDs
	idMap := make(map[uint64]int)
	for rows.Next() {
		var duckDBID int
		if err := rows.Scan(&duckDBID); err != nil {
			vi.logger.Warn("Failed to scan DuckDB ID", zap.Error(err))
			continue
		}
		idMap[uint64(duckDBID)] = duckDBID
	}

	// Use DuckDB IDs when available; otherwise fallback to the HNSW result.
	results := make([]int, len(ids))
	for i, id := range ids {
		if mappedID, found := idMap[id]; found {
			results[i] = mappedID
		} else {
			vi.logger.Warn("ID not found in DuckDB, using HNSW ID", zap.Uint64("HNSW_ID", id))
			results[i] = int(id)
		}
	}

	return results, nil
}

// Save persists the current state of the HNSW index to disk and flushes the DuckDB appender.
func (vi *VectorIndex) Save() error {
	vi.indexMu.Lock()
	defer vi.indexMu.Unlock()
	vi.logger.Info("Saving HNSW index to disk", zap.String("path", vi.indexPath))
	vi.index.Save(vi.indexPath)

	vi.dbMu.Lock()
	defer vi.dbMu.Unlock()
	if err := vi.appender.Flush(); err != nil {
		vi.logger.Error("Failed to flush DuckDB appender during Save", zap.Error(err))
		return fmt.Errorf("failed to flush DuckDB appender: %w", err)
	}

	return nil
}

// Close releases resources held by the VectorIndex.
// It closes the DuckDB connection and flushes the logger.
func (vi *VectorIndex) Close() error {
	vi.indexMu.Lock()
	vi.dbMu.Lock()
	defer vi.indexMu.Unlock()
	defer vi.dbMu.Unlock()

	var errs []string

	if err := vi.db.Close(); err != nil {
		vi.logger.Error("Failed to close DuckDB", zap.Error(err))
		errs = append(errs, err.Error())
	}

	if err := vi.logger.Sync(); err != nil {
		errs = append(errs, err.Error())
	}

	if len(errs) > 0 {
		return fmt.Errorf("error(s) on close: %s", strings.Join(errs, "; "))
	}

	return nil
}

// uint64SliceToString converts a slice of uint64 IDs to a comma-separated string.
func uint64SliceToString(slice []uint64) string {
	strs := make([]string, len(slice))
	for i, id := range slice {
		strs[i] = fmt.Sprintf("%d", id)
	}
	return strings.Join(strs, ",")
}

// Flush explicitly flushes the DuckDB appender.
func (vi *VectorIndex) Flush() error {
	vi.dbMu.Lock()
	defer vi.dbMu.Unlock()
	return vi.appender.Flush()
}

func placeHolders(n int) string {
	return strings.Repeat("?,", n-1) + "?"
}
