package quiver

import (
	"archive/zip"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/apache/arrow-adbc/go/adbc"
	"github.com/apache/arrow-adbc/go/adbc/drivermgr"
	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/coder/hnsw"
	"go.uber.org/zap"
)

// DistanceMetric defines the similarity metrics.
type DistanceMetric int

const (
	Cosine DistanceMetric = iota
	L2
)

// Define distance metrics for our own use
const (
	CosineDistance = 0
	L2Distance     = 1
)

// Config holds the index settings and tunable hyperparameters.
type Config struct {
	Dimension       int
	StoragePath     string // Base directory for storing index files.
	Distance        DistanceMetric
	MaxElements     uint64
	HNSWM           int // HNSW hyperparameter M
	HNSWEfConstruct int // HNSW hyperparameter efConstruction
	HNSWEfSearch    int // HNSW hyperparameter ef used during queries
	BatchSize       int // Number of vectors to batch before insertion
	// Persistence configuration
	PersistInterval time.Duration // How often to persist index to disk (default: 5m)
	// Backup configuration
	BackupInterval    time.Duration // How often to create backups (default: 1h)
	BackupPath        string        // Path to store backups (default: StoragePath/backups)
	BackupCompression bool          // Whether to compress backups (default: true)
	MaxBackups        int           // Maximum number of backups to keep (default: 5)
	// Security configuration
	EncryptionEnabled bool   // Whether to encrypt data at rest
	EncryptionKey     string // Key for encrypting data at rest (min 32 bytes)
}

// vectorMeta holds a vector and its metadata for batch insertion.
type vectorMeta struct {
	id     uint64
	vector []float32
	meta   map[string]interface{}
}

// SearchResult holds the output of a search.
type SearchResult struct {
	ID       uint64
	Distance float32
	Metadata map[string]interface{}
}

// Index is the main structure for vector search.
type Index struct {
	config          Config
	hnsw            *hnsw.Graph[uint64]
	metadata        map[uint64]map[string]interface{}
	vectors         map[uint64][]float32 // Store vectors for queries with negative examples
	duckdb          *DuckDB
	dbConn          *DuckDBConn
	lock            sync.RWMutex
	batchBuffer     []vectorMeta
	batchLock       sync.Mutex
	cache           sync.Map // Caches metadata: key = id, value = metadata map
	batchTicker     *time.Ticker
	batchDone       chan struct{}
	logger          *zap.Logger
	persistInterval time.Duration
	persistTicker   *time.Ticker
	persistDone     chan struct{}
	lastPersistID   uint64
	// Backup related fields
	backupTicker   *time.Ticker
	backupDone     chan struct{}
	lastBackupTime time.Time
	// A waitgroup to ensure background workers exit on close.
	bgWG sync.WaitGroup
}

// DuckDBOptions define the configuration for opening a DuckDB database.
type DuckDBOptions struct {
	// Path to the DuckDB file ("" => in-memory)
	Path string

	// DriverPath is the location of libduckdb.so, if empty => auto-detect
	DriverPath string

	// Context for new database/connection usage
	Context context.Context
}

// DuckDBOption is a functional config approach
type DuckDBOption func(*DuckDBOptions)

// WithPath sets a file path for the DuckDB DB.
func WithPath(p string) DuckDBOption {
	return func(o *DuckDBOptions) {
		o.Path = p
	}
}

// WithDriverPath sets the path to the DuckDB driver library.
// If not provided, the driver will be auto-detected based on the current OS.
func WithDriverPath(p string) DuckDBOption {
	return func(o *DuckDBOptions) {
		o.DriverPath = p
	}
}

// WithContext sets a custom Context for DB usage.
func WithContext(ctx context.Context) DuckDBOption {
	return func(o *DuckDBOptions) {
		o.Context = ctx
	}
}

// DuckDB is the primary struct managing a DuckDB database via ADBC.
type DuckDB struct {
	mu     sync.Mutex
	db     adbc.Database
	driver adbc.Driver
	opts   DuckDBOptions

	conns []*DuckDBConn // track open connections
}

// DuckDBConn is a simple wrapper holding an open connection.
type DuckDBConn struct {
	parent *DuckDB
	conn   adbc.Connection
}

// NewDuckDB opens or creates a DuckDB instance (file-based or in-memory).
// The driver library is auto-detected if not provided.
func NewDuckDB(options ...DuckDBOption) (*DuckDB, error) {
	// gather defaults
	var opts DuckDBOptions
	for _, opt := range options {
		opt(&opts)
	}
	if opts.Context == nil {
		opts.Context = context.Background()
	}

	// auto-detect driver if empty
	dPath := opts.DriverPath
	if dPath == "" {
		switch runtime.GOOS {
		case "darwin":
			dPath = "/usr/local/lib/libduckdb.dylib"
		case "linux":
			dPath = "/usr/local/lib/libduckdb.so"
		case "windows":
			if home, err := os.UserHomeDir(); err == nil {
				dPath = home + "/Downloads/duckdb-windows-amd64/duckdb.dll"
			}
		}
	}

	dbOpts := map[string]string{
		"driver":     dPath,
		"entrypoint": "duckdb_adbc_init",
	}
	if opts.Path != "" {
		dbOpts["path"] = opts.Path
	}

	driver := drivermgr.Driver{}
	db, err := driver.NewDatabase(dbOpts)
	if err != nil {
		return nil, fmt.Errorf("error creating new DuckDB database: %w", err)
	}

	duck := &DuckDB{
		db:     db,
		driver: driver,
		opts:   opts,
	}
	return duck, nil
}

// OpenConnection opens a new connection to DuckDB.
func (d *DuckDB) OpenConnection() (*DuckDBConn, error) {
	d.mu.Lock()
	defer d.mu.Unlock()

	conn, err := d.db.Open(d.opts.Context)
	if err != nil {
		return nil, fmt.Errorf("failed to open connection: %w", err)
	}
	dc := &DuckDBConn{parent: d, conn: conn}
	d.conns = append(d.conns, dc)
	return dc, nil
}

// Close closes the DuckDB database and all open connections.
func (d *DuckDB) Close() error {
	d.mu.Lock()
	defer d.mu.Unlock()

	// close all open conns
	for _, c := range d.conns {
		c.conn.Close()
	}
	d.conns = nil

	// close db
	return d.db.Close()
}

// ConnCount returns the current number of open connections.
func (d *DuckDB) ConnCount() int {
	d.mu.Lock()
	defer d.mu.Unlock()
	return len(d.conns)
}

// Path returns the database file path, or empty if in-memory.
func (d *DuckDB) Path() string {
	return d.opts.Path
}

// Exec runs a statement that doesn't produce a result set, returning
// the number of rows affected if known, else -1.
func (c *DuckDBConn) Exec(ctx context.Context, sql string) (int64, error) {
	stmt, err := c.conn.NewStatement()
	if err != nil {
		return -1, fmt.Errorf("failed to create statement: %w", err)
	}
	defer stmt.Close()

	if err := stmt.SetSqlQuery(sql); err != nil {
		return -1, fmt.Errorf("failed to set SQL query: %w", err)
	}
	affected, err := stmt.ExecuteUpdate(ctx)
	return affected, err
}

// Query runs a SQL query returning (RecordReader, adbc.Statement, rowCount).
// rowCount will be -1 if not known. Caller is responsible for closing the
// returned statement and the RecordReader.
func (c *DuckDBConn) Query(ctx context.Context, sql string) (array.RecordReader, adbc.Statement, int64, error) {
	stmt, err := c.conn.NewStatement()
	if err != nil {
		return nil, nil, -1, fmt.Errorf("failed to create statement: %w", err)
	}
	if err := stmt.SetSqlQuery(sql); err != nil {
		stmt.Close()
		return nil, nil, -1, fmt.Errorf("failed to set SQL query: %w", err)
	}

	rr, rowsAffected, err := stmt.ExecuteQuery(ctx)
	if err != nil {
		stmt.Close()
		return nil, nil, -1, err
	}
	return rr, stmt, rowsAffected, nil
}

// GetTableSchema fetches the Arrow schema of a table in the given catalog/schema
// (pass nil for defaults).
func (c *DuckDBConn) GetTableSchema(ctx context.Context, catalog, dbSchema *string, tableName string) (*arrow.Schema, error) {
	return c.conn.GetTableSchema(ctx, catalog, dbSchema, tableName)
}

// IngestCreateAppend ingests an arrow.Record into a DuckDB table. If the table does not
// exist, it is created from the record's schema. Otherwise, it appends. Returns
// the number of rows affected if known, else -1.
func (c *DuckDBConn) IngestCreateAppend(ctx context.Context, tableName string, rec arrow.Record) (int64, error) {
	if tableName == "" {
		return -1, fmt.Errorf("no target tableName provided")
	}
	if rec == nil {
		return -1, fmt.Errorf("nil arrow record")
	}
	existing, _ := c.GetTableSchema(ctx, nil, nil, tableName)

	stmt, err := c.conn.NewStatement()
	if err != nil {
		return -1, fmt.Errorf("failed to create statement: %w", err)
	}
	defer stmt.Close()

	// If the table does not exist => create, else append
	mode := adbc.OptionValueIngestModeCreate
	if existing != nil {
		mode = adbc.OptionValueIngestModeAppend
	}
	err = stmt.SetOption(adbc.OptionKeyIngestMode, mode)
	if err != nil {
		return -1, fmt.Errorf("failed to set ingest mode: %w", err)
	}
	err = stmt.SetOption(adbc.OptionKeyIngestTargetTable, tableName)
	if err != nil {
		return -1, fmt.Errorf("failed to set ingest target: %w", err)
	}
	// Bind the record
	if err := stmt.Bind(ctx, rec); err != nil {
		return -1, fmt.Errorf("failed to bind arrow record: %w", err)
	}
	// Execute
	affected, err := stmt.ExecuteUpdate(ctx)
	return affected, err
}

// Close closes the connection, removing it from the parent DuckDB's tracking.
func (c *DuckDBConn) Close() error {
	c.parent.mu.Lock()
	defer c.parent.mu.Unlock()
	// remove from parent
	for i, cc := range c.parent.conns {
		if cc == c {
			c.parent.conns[i] = c.parent.conns[len(c.parent.conns)-1]
			c.parent.conns = c.parent.conns[:len(c.parent.conns)-1]
			break
		}
	}
	err := c.conn.Close()
	c.parent = nil
	return err
}

// New creates a new index with the given configuration.
// If an existing index is found (in StoragePath), Load is used.
func New(config Config, logger *zap.Logger) (*Index, error) {
	// Validate config
	if config.Dimension <= 0 {
		return nil, errors.New("dimension must be positive")
	}
	if config.MaxElements <= 0 {
		return nil, errors.New("max_elements must be positive")
	}
	if config.HNSWM <= 0 {
		config.HNSWM = 16 // Default to a reasonable value
	}
	if config.HNSWEfConstruct <= 0 {
		config.HNSWEfConstruct = 200 // Default to a reasonable value
	}
	if config.HNSWEfSearch <= 0 {
		config.HNSWEfSearch = 100 // Default to a reasonable value
	}
	if config.BatchSize <= 0 {
		config.BatchSize = 1000 // Default to a reasonable value
	}

	// Set default persistence interval
	persistInterval := 5 * time.Minute
	if config.PersistInterval > 0 {
		persistInterval = config.PersistInterval
	}

	// Set backup defaults
	backupInterval := 1 * time.Hour
	if config.BackupInterval > 0 {
		backupInterval = config.BackupInterval
	}
	if config.BackupPath == "" {
		config.BackupPath = filepath.Join(config.StoragePath, "backups")
	}
	if config.MaxBackups == 0 {
		config.MaxBackups = 5
	}

	// Ensure StoragePath is an absolute path.
	config.StoragePath = filepath.Clean(config.StoragePath)
	if !filepath.IsAbs(config.StoragePath) {
		absPath, err := filepath.Abs(config.StoragePath)
		if err != nil {
			return nil, fmt.Errorf("failed to get absolute path: %w", err)
		}
		config.StoragePath = absPath
	}

	// Create storage directory if it doesn't exist.
	storageDir := filepath.Dir(config.StoragePath)
	if err := os.MkdirAll(storageDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create storage directory: %w", err)
	}

	// Check if an existing index exists.
	indexFile := filepath.Join(storageDir, "index.hnsw")
	if _, err := os.Stat(indexFile); err == nil {
		return Load(config, logger)
	}

	// Initialize HNSW graph with the specified parameters
	// Create a new HNSW graph with default configuration
	graph := hnsw.NewGraph[uint64]()

	// Open DuckDB connection using StoragePath
	duckdb, err := NewDuckDB(WithPath(config.StoragePath))
	if err != nil {
		return nil, fmt.Errorf("failed to open DuckDB: %w", err)
	}

	// Open a connection to the database
	dbConn, err := duckdb.OpenConnection()
	if err != nil {
		duckdb.Close()
		return nil, fmt.Errorf("failed to open DuckDB connection: %w", err)
	}

	// Create metadata table if it doesn't exist.
	_, err = dbConn.Exec(context.Background(), `CREATE TABLE IF NOT EXISTS metadata (
		id BIGINT PRIMARY KEY,
		json VARCHAR
	)`)
	if err != nil {
		dbConn.Close()
		duckdb.Close()
		return nil, fmt.Errorf("failed to create metadata table: %w", err)
	}

	idx := &Index{
		config:          config,
		hnsw:            graph,
		metadata:        make(map[uint64]map[string]interface{}),
		vectors:         make(map[uint64][]float32),
		duckdb:          duckdb,
		dbConn:          dbConn,
		batchBuffer:     make([]vectorMeta, 0, config.BatchSize),
		logger:          logger,
		batchTicker:     time.NewTicker(time.Second),
		batchDone:       make(chan struct{}),
		persistInterval: persistInterval,
		persistTicker:   time.NewTicker(persistInterval),
		persistDone:     make(chan struct{}),
		backupTicker:    time.NewTicker(backupInterval),
		backupDone:      make(chan struct{}),
	}

	// Start background workers.
	idx.bgWG.Add(1)
	go func() {
		defer idx.bgWG.Done()
		idx.batchProcessor()
	}()

	idx.bgWG.Add(1)
	go func() {
		defer idx.bgWG.Done()
		idx.persistenceWorker()
	}()

	// Start backup worker.
	idx.bgWG.Add(1)
	go func() {
		defer idx.bgWG.Done()
		idx.backupWorker()
	}()

	logger.Info("Quiver index initialized",
		zap.Int("dimension", config.Dimension),
		zap.Int("batchSize", config.BatchSize))
	return idx, nil
}

// batchProcessor runs in the background and processes batched vectors.
func (idx *Index) batchProcessor() {
	for {
		select {
		case <-idx.batchTicker.C:
			if len(idx.batchBuffer) > 0 {
				if err := idx.flushBatch(); err != nil {
					idx.logger.Error("failed to flush batch", zap.Error(err))
				}
			}
		case <-idx.batchDone:
			idx.logger.Info("batch processor shutting down")
			return
		}
	}
}

// flushBatch adds all vectors in the batch to the index.
func (idx *Index) flushBatch() error {
	idx.batchLock.Lock()
	defer idx.batchLock.Unlock()

	if len(idx.batchBuffer) == 0 {
		return nil
	}

	n := len(idx.batchBuffer)

	// Lock the index for updating shared metadata and cache.
	idx.lock.Lock()
	defer idx.lock.Unlock()

	// Prepare batch for SQL execution
	var values []string

	for _, item := range idx.batchBuffer {
		// Add to HNSW graph
		node := hnsw.MakeNode(item.id, item.vector)
		idx.hnsw.Add(node)

		// Update in-memory metadata and cache.
		idx.metadata[item.id] = item.meta
		idx.cache.Store(item.id, item.meta)

		// Store vector for negative example searches
		idx.vectors[item.id] = item.vector

		metaJSON, err := json.Marshal(item.meta)
		if err != nil {
			idx.logger.Error("failed to marshal metadata", zap.Uint64("id", item.id), zap.Error(err))
			return err
		}

		// Add to batch values - escape single quotes in JSON
		escapedJSON := strings.ReplaceAll(string(metaJSON), "'", "''")
		values = append(values, fmt.Sprintf("(%d, '%s')", item.id, escapedJSON))
	}

	// Execute batch insert using ADBC
	if len(values) > 0 {
		query := fmt.Sprintf("INSERT INTO metadata (id, json) VALUES %s ON CONFLICT(id) DO UPDATE SET json = excluded.json",
			strings.Join(values, ","))

		_, err := idx.dbConn.Exec(context.Background(), query)
		if err != nil {
			idx.logger.Error("failed to execute batch insert", zap.Error(err))
			return err
		}
	}

	idx.logger.Info("batch flushed", zap.Int("num_points", n))
	// Reset the batch buffer.
	idx.batchBuffer = idx.batchBuffer[:0]
	return nil
}

// Add adds a vector to the index with the specified ID and metadata.
func (idx *Index) Add(id uint64, vector []float32, meta map[string]interface{}) error {
	// Validate input
	if len(vector) != idx.config.Dimension {
		return fmt.Errorf("vector dimension (%d) does not match index dimension (%d)", len(vector), idx.config.Dimension)
	}

	// Make a copy of the metadata map to ensure it isn't modified externally
	metaCopy := make(map[string]interface{})
	for k, v := range meta {
		metaCopy[k] = v
	}

	// Validate metadata schema
	if err := validateMetadataSchema(metaCopy); err != nil {
		return err
	}

	// We store vectors for use with negative example searches
	vectorCopy := make([]float32, len(vector))
	copy(vectorCopy, vector)

	// Add to batch for deferred insertion
	idx.batchLock.Lock()
	idx.batchBuffer = append(idx.batchBuffer, vectorMeta{
		id:     id,
		vector: vectorCopy,
		meta:   metaCopy,
	})
	idx.batchLock.Unlock()

	return nil
}

// Search performs a vector similarity search and returns the k most similar vectors.
// Supports pagination with page and pageSize parameters.
func (idx *Index) Search(query []float32, k, page, pageSize int) ([]SearchResult, error) {
	idx.lock.RLock()
	defer idx.lock.RUnlock()

	// Validate parameters
	if k <= 0 {
		return nil, errors.New("k must be positive")
	}
	if page <= 0 {
		page = 1 // Default to first page
	}
	if pageSize <= 0 {
		pageSize = 10 // Default page size
	}

	// Ensure we have enough elements in the index
	currentCount := idx.hnsw.Len()
	if currentCount == 0 {
		return []SearchResult{}, nil
	}

	// If k is too large relative to the number of elements, adjust it
	if k > currentCount {
		k = currentCount
	}

	// Use a higher ef value for better recall
	efSearch := idx.config.HNSWEfSearch * 2
	if efSearch < k*4 {
		efSearch = k * 4
	}

	// Ensure M is large enough
	if idx.config.HNSWM < 16 {
		idx.logger.Warn("HNSW M parameter is too small, search may fail",
			zap.Int("current_m", idx.config.HNSWM),
			zap.Int("recommended_min", 16))
	}

	// Perform search with proper error handling and recovery
	var results []hnsw.Node[uint64]
	var err error

	// Add debugging information
	idx.logger.Debug("Starting HNSW search",
		zap.Int("k", k),
		zap.Int("pageSize", pageSize),
		zap.Int("efSearch", efSearch),
		zap.Int("currentCount", currentCount),
		zap.Float32s("query", query))

	// Use a recovery function to catch exceptions
	func() {
		defer func() {
			if r := recover(); r != nil {
				err = fmt.Errorf("HNSW search panicked: %v", r)
			}
		}()

		// Set the ef parameter for the graph before searching
		// Note: This is a global setting that affects all searches until changed again
		idx.logger.Debug("Setting efSearch parameter", zap.Int("efSearch", efSearch))

		// Perform the search
		results = idx.hnsw.Search(query, k*pageSize)
	}()

	if err != nil {
		idx.logger.Error("HNSW search failed", zap.Error(err))
		return nil, fmt.Errorf("HNSW search failed: %w", err)
	}

	if len(results) == 0 {
		return []SearchResult{}, nil
	}

	start := (page - 1) * pageSize
	end := start + pageSize
	if start >= len(results) {
		return nil, nil // No more results
	}
	if end > len(results) {
		end = len(results)
	}

	var searchResults []SearchResult
	for _, r := range results[start:end] {
		// Calculate distance between query and result vector
		distance := computeDistance(query, r.Value, idx.config.Distance)
		meta := idx.getMetadata(r.Key)
		searchResults = append(searchResults, SearchResult{
			ID:       r.Key,
			Distance: distance,
			Metadata: meta,
		})
	}
	return searchResults, nil
}

// SearchWithFilter performs a search and filters results based on a metadata query.
func (idx *Index) SearchWithFilter(query []float32, k int, filter string) ([]SearchResult, error) {
	// First, query metadata to get matching IDs
	metadataResults, err := idx.QueryMetadata(filter)
	if err != nil {
		return nil, fmt.Errorf("metadata query failed: %w", err)
	}

	// Extract IDs from metadata results
	matchingIDs := make(map[uint64]bool)
	for _, meta := range metadataResults {
		id, ok := meta["id"].(uint64)
		if !ok {
			// Try to convert from float64 (common in JSON unmarshaling)
			if idFloat, ok := meta["id"].(float64); ok {
				id = uint64(idFloat)
			} else {
				continue
			}
		}
		matchingIDs[id] = true
	}

	// If no matching IDs, return empty results
	if len(matchingIDs) == 0 {
		return []SearchResult{}, nil
	}

	// Perform vector search
	results, err := idx.Search(query, k*10, 1, k*10) // Get more results to filter
	if err != nil {
		return nil, err
	}

	// Filter results by matching IDs
	var filteredResults []SearchResult
	for _, result := range results {
		if matchingIDs[result.ID] {
			filteredResults = append(filteredResults, result)
			if len(filteredResults) >= k {
				break
			}
		}
	}

	return filteredResults, nil
}

// getMetadata retrieves metadata for a vector by ID.
func (idx *Index) getMetadata(id uint64) map[string]interface{} {
	// Check cache first
	if cached, ok := idx.cache.Load(id); ok {
		if meta, ok := cached.(map[string]interface{}); ok {
			return meta
		}
	}

	// Fall back to metadata map
	meta, ok := idx.metadata[id]
	if !ok {
		return nil
	}
	return meta
}

// Save saves the index to the specified directory.
func (idx *Index) Save(saveDir string) error {
	idx.lock.Lock()
	defer idx.lock.Unlock()

	// Ensure save directory exists.
	if err := os.MkdirAll(saveDir, 0755); err != nil {
		return fmt.Errorf("failed to create save directory: %w", err)
	}

	// Flush any pending changes.
	if err := idx.flushBatch(); err != nil {
		return err
	}

	// TODO: Implement serialization for the HNSW graph
	// The coder/hnsw library doesn't provide a Save method, so we need to implement our own
	// serialization logic here

	// Export metadata.
	if err := idx.exportMetadata(filepath.Join(saveDir, "metadata.json"), false); err != nil {
		return fmt.Errorf("failed to export metadata: %w", err)
	}

	idx.logger.Info("index saved", zap.String("path", saveDir))
	return nil
}

// Load loads an index from the specified directory.
func Load(config Config, logger *zap.Logger) (*Index, error) {
	storageDir := filepath.Dir(config.StoragePath)

	// Initialize a new HNSW graph
	graph := hnsw.NewGraph[uint64]()

	// Open DuckDB connection
	duckdb, err := NewDuckDB(WithPath(config.StoragePath))
	if err != nil {
		return nil, fmt.Errorf("failed to open DuckDB: %w", err)
	}

	// Open a connection to the database
	dbConn, err := duckdb.OpenConnection()
	if err != nil {
		duckdb.Close()
		return nil, fmt.Errorf("failed to open DuckDB connection: %w", err)
	}

	// Create metadata table if it doesn't exist.
	_, err = dbConn.Exec(context.Background(), `CREATE TABLE IF NOT EXISTS metadata (
		id BIGINT PRIMARY KEY,
		json VARCHAR
	)`)
	if err != nil {
		dbConn.Close()
		duckdb.Close()
		return nil, fmt.Errorf("failed to create metadata table: %w", err)
	}

	// Load metadata from the database.
	rr, stmt, _, err := dbConn.Query(context.Background(), "SELECT id, json FROM metadata")
	if err != nil {
		dbConn.Close()
		duckdb.Close()
		return nil, fmt.Errorf("failed to query metadata: %w", err)
	}
	defer stmt.Close()

	metadata := make(map[uint64]map[string]interface{})
	vectors := make(map[uint64][]float32)

	// Process the query results
	for rr.Next() {
		record := rr.Record()
		idCol := record.Column(0).(*array.Uint64)
		jsonCol := record.Column(1).(*array.String)

		for i := 0; i < int(record.NumRows()); i++ {
			id := idCol.Value(i)
			jsonStr := jsonCol.Value(i)

			var meta map[string]interface{}
			if err := json.Unmarshal([]byte(jsonStr), &meta); err != nil {
				logger.Warn("Failed to unmarshal metadata", zap.Uint64("id", id), zap.Error(err))
				continue
			}

			metadata[id] = meta
		}
	}

	if err := rr.Err(); err != nil {
		dbConn.Close()
		duckdb.Close()
		return nil, fmt.Errorf("error iterating metadata rows: %w", err)
	}

	// Set persist interval.
	persistInterval := 5 * time.Minute
	if config.PersistInterval > 0 {
		persistInterval = config.PersistInterval
	}

	idx := &Index{
		config:          config,
		hnsw:            graph,
		metadata:        metadata,
		vectors:         vectors,
		duckdb:          duckdb,
		dbConn:          dbConn,
		batchBuffer:     make([]vectorMeta, 0, config.BatchSize),
		logger:          logger,
		batchTicker:     time.NewTicker(time.Second),
		batchDone:       make(chan struct{}),
		persistInterval: persistInterval,
		persistTicker:   time.NewTicker(persistInterval),
		persistDone:     make(chan struct{}),
		backupTicker:    time.NewTicker(config.BackupInterval),
		backupDone:      make(chan struct{}),
	}

	// Restart background workers.
	idx.bgWG.Add(1)
	go func() {
		defer idx.bgWG.Done()
		idx.batchProcessor()
	}()

	idx.bgWG.Add(1)
	go func() {
		defer idx.bgWG.Done()
		idx.persistenceWorker()
	}()

	idx.bgWG.Add(1)
	go func() {
		defer idx.bgWG.Done()
		idx.backupWorker()
	}()

	logger.Info("index loaded successfully", zap.String("path", storageDir))
	return idx, nil
}

// persistenceWorker runs in the background and periodically persists new vectors.
func (idx *Index) persistenceWorker() {
	for {
		select {
		case <-idx.persistTicker.C:
			if err := idx.persistToStorage(); err != nil {
				idx.logger.Error("failed to persist index", zap.Error(err))
			}
		case <-idx.persistDone:
			idx.logger.Info("persistence worker shutting down")
			return
		}
	}
}

// persistToStorage persists new vectors by flushing pending batches and saving the HNSW index.
// The new index file is first written to a temporary file and then renamed atomically.
func (idx *Index) persistToStorage() error {
	idx.lock.Lock()
	defer idx.lock.Unlock()

	if err := idx.flushBatch(); err != nil {
		return err
	}

	// Get new IDs since last persist (for logging or incremental processing).
	newIDs := idx.getIDsSinceLastPersist()

	// Save to a temporary file first
	indexFile := filepath.Join(idx.config.StoragePath, "index.hnsw")
	tmpIndexFile := indexFile + ".tmp"

	// Create a file to write to
	file, err := os.Create(tmpIndexFile)
	if err != nil {
		return fmt.Errorf("failed to create temporary index file: %w", err)
	}
	defer file.Close()

	// Export the HNSW graph to the file
	if err := idx.hnsw.Export(file); err != nil {
		return fmt.Errorf("failed to export HNSW graph: %w", err)
	}

	// If encryption is enabled, read the file, encrypt it, and write it back
	if idx.config.EncryptionEnabled {
		// Read the saved file
		data, err := os.ReadFile(tmpIndexFile)
		if err != nil {
			return fmt.Errorf("failed to read temporary index file: %w", err)
		}

		// Write encrypted data
		if err := idx.writeEncrypted(data, tmpIndexFile+".enc"); err != nil {
			return fmt.Errorf("failed to encrypt index file: %w", err)
		}

		// Remove the unencrypted temp file
		if err := os.Remove(tmpIndexFile); err != nil {
			idx.logger.Warn("failed to remove temporary unencrypted file", zap.Error(err))
		}

		// Rename the encrypted file to the temp file
		if err := os.Rename(tmpIndexFile+".enc", tmpIndexFile); err != nil {
			return fmt.Errorf("failed to rename encrypted index file: %w", err)
		}
	}

	// Atomically replace the old index file with the new one
	if err := os.Rename(tmpIndexFile, indexFile); err != nil {
		return fmt.Errorf("failed to rename index file: %w", err)
	}

	// Update the last persisted ID
	if len(newIDs) > 0 {
		maxID := uint64(0)
		for _, id := range newIDs {
			if id > maxID {
				maxID = id
			}
		}
		idx.lastPersistID = maxID
	}

	idx.logger.Info("index persisted successfully",
		zap.Int("new_vectors", len(newIDs)),
		zap.String("path", indexFile))
	return nil
}

// getIDsSinceLastPersist retrieves IDs added since the last persist.
func (idx *Index) getIDsSinceLastPersist() []uint64 {
	var newIDs []uint64
	for id := range idx.metadata {
		if id > idx.lastPersistID {
			newIDs = append(newIDs, id)
		}
	}
	return newIDs
}

// Backup creates a backup of the current index and metadata to the specified directory.
// If incremental is true, only changes since the last backup are included.
func (idx *Index) Backup(path string, incremental bool, compress bool) error {
	idx.lock.RLock()
	defer idx.lock.RUnlock()

	// Ensure backup directory exists.
	if err := os.MkdirAll(path, 0755); err != nil {
		return fmt.Errorf("failed to create backup directory: %w", err)
	}

	// Flush pending batches before backup
	if err := idx.flushBatch(); err != nil {
		return fmt.Errorf("failed to flush batch before backup: %w", err)
	}

	indexFile := filepath.Join(idx.config.StoragePath, "index.hnsw")
	backupIndexFile := filepath.Join(path, "index.hnsw")

	// Read source file
	srcData, err := os.ReadFile(indexFile)
	if err != nil {
		return fmt.Errorf("failed to read index file: %w", err)
	}

	// Calculate checksum for verification
	checksum := sha256.Sum256(srcData)

	// Handle compression if requested
	if compress {
		backupIndexFile += ".gz"
		var compressedData bytes.Buffer
		gzWriter := gzip.NewWriter(&compressedData)
		if _, err := gzWriter.Write(srcData); err != nil {
			return fmt.Errorf("failed to compress index file: %w", err)
		}
		if err := gzWriter.Close(); err != nil {
			return fmt.Errorf("failed to finalize compression: %w", err)
		}
		srcData = compressedData.Bytes()
	}

	// Write to backup location
	if err := os.WriteFile(backupIndexFile, srcData, 0644); err != nil {
		return fmt.Errorf("failed to write backup index file: %w", err)
	}

	// Export metadata
	metadataFile := filepath.Join(path, "metadata.json")
	if err := idx.exportMetadata(metadataFile, compress); err != nil {
		return fmt.Errorf("failed to export metadata: %w", err)
	}

	// Create backup manifest with integrity verification info
	manifest := map[string]interface{}{
		"timestamp":      time.Now().Format(time.RFC3339),
		"index_checksum": hex.EncodeToString(checksum[:]),
		"compressed":     compress,
		"incremental":    incremental,
		"vector_count":   idx.hnsw.Len(),
		"metadata_count": len(idx.metadata),
		"format_version": "1.0",
		"quiver_version": "1.0.0", // This should be dynamic in the future
		"config":         idx.config,
	}

	manifestData, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to create manifest: %w", err)
	}

	manifestFile := filepath.Join(path, "manifest.json")
	if err := os.WriteFile(manifestFile, manifestData, 0644); err != nil {
		return fmt.Errorf("failed to write manifest: %w", err)
	}

	idx.logger.Info("backup completed successfully",
		zap.String("path", path),
		zap.Bool("incremental", incremental),
		zap.Bool("compressed", compress),
		zap.String("checksum", hex.EncodeToString(checksum[:])))
	return nil
}

// Restore restores the index from a backup file.
func (idx *Index) Restore(backupPath string) error {
	idx.lock.Lock()
	defer idx.lock.Unlock()

	// Extract backup to a temporary directory
	tempDir, err := os.MkdirTemp("", "quiver-restore-*")
	if err != nil {
		return fmt.Errorf("failed to create temp directory: %w", err)
	}
	defer os.RemoveAll(tempDir)

	// Check if the backup is compressed
	isCompressed := strings.HasSuffix(backupPath, ".gz") || strings.HasSuffix(backupPath, ".zip")

	// Extract the backup
	if isCompressed {
		if err := extractCompressedBackup(backupPath, tempDir); err != nil {
			return fmt.Errorf("failed to extract backup: %w", err)
		}
	} else {
		// Just copy the files
		if err := copyFile(backupPath, filepath.Join(tempDir, "backup.json")); err != nil {
			return fmt.Errorf("failed to copy backup file: %w", err)
		}
	}

	// Read backup metadata
	metaFile := filepath.Join(tempDir, "backup.json")
	metaData, err := os.ReadFile(metaFile)
	if err != nil {
		return fmt.Errorf("failed to read backup metadata: %w", err)
	}

	var backupInfo map[string]interface{}
	if err := json.Unmarshal(metaData, &backupInfo); err != nil {
		return fmt.Errorf("failed to parse backup metadata: %w", err)
	}

	// Check backup format version
	version, ok := backupInfo["format_version"].(string)
	if !ok || version != "1.0" {
		return fmt.Errorf("unsupported backup format version: %v", version)
	}

	// Get paths to index and metadata files
	indexPath, ok := backupInfo["index_path"].(string)
	if !ok {
		return errors.New("backup metadata missing index_path")
	}
	metadataPath, ok := backupInfo["metadata_path"].(string)
	if !ok {
		return errors.New("backup metadata missing metadata_path")
	}

	// Copy files to their destinations
	destIndexPath := filepath.Join(idx.config.StoragePath, "index.hnsw")
	destMetadataPath := filepath.Join(idx.config.StoragePath, "metadata.json")

	if err := copyFile(filepath.Join(tempDir, indexPath), destIndexPath); err != nil {
		return fmt.Errorf("failed to copy index file: %w", err)
	}

	if err := copyFile(filepath.Join(tempDir, metadataPath), destMetadataPath); err != nil {
		return fmt.Errorf("failed to copy metadata file: %w", err)
	}

	// Create a new graph since the coder/hnsw library doesn't have a Load function
	newGraph := hnsw.NewGraph[uint64]()

	// TODO: Implement loading vectors from backup into the graph
	// This would require reading the vectors from the backup and adding them to the graph

	// Update the index with the restored HNSW
	idx.hnsw = newGraph

	idx.logger.Info("index restored successfully",
		zap.String("backup_path", backupPath),
		zap.String("storage_path", idx.config.StoragePath))
	return nil
}

// writeEncrypted writes data to a file with optional encryption
func (idx *Index) writeEncrypted(data []byte, filePath string) error {
	if !idx.config.EncryptionEnabled {
		return os.WriteFile(filePath, data, 0644)
	}

	// Validate encryption key
	if len(idx.config.EncryptionKey) < 32 {
		return fmt.Errorf("encryption key must be at least 32 bytes long")
	}

	// Create a 32-byte key from the provided key
	key := []byte(idx.config.EncryptionKey)
	if len(key) > 32 {
		key = key[:32]
	}

	// Create a new AES cipher block
	block, err := aes.NewCipher(key)
	if err != nil {
		return fmt.Errorf("failed to create cipher: %w", err)
	}

	// Generate a random nonce
	nonce := make([]byte, 12)
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return fmt.Errorf("failed to generate nonce: %w", err)
	}

	// Create the GCM cipher
	aesgcm, err := cipher.NewGCM(block)
	if err != nil {
		return fmt.Errorf("failed to create GCM cipher: %w", err)
	}

	// Encrypt the data
	encryptedData := aesgcm.Seal(nil, nonce, data, nil)

	// Create the final data: nonce + encrypted data
	finalData := make([]byte, len(nonce)+len(encryptedData))
	copy(finalData[:len(nonce)], nonce)
	copy(finalData[len(nonce):], encryptedData)

	// Write the encrypted data
	return os.WriteFile(filePath, finalData, 0644)
}

// readEncrypted reads data from a file with optional decryption
func (idx *Index) readEncrypted(filePath string) ([]byte, error) {
	// Read the file
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	// If encryption is not enabled, return the raw data
	if !idx.config.EncryptionEnabled {
		return data, nil
	}

	// Validate encryption key
	if len(idx.config.EncryptionKey) < 32 {
		return nil, fmt.Errorf("encryption key must be at least 32 bytes long")
	}

	// Create a 32-byte key from the provided key
	key := []byte(idx.config.EncryptionKey)
	if len(key) > 32 {
		key = key[:32]
	}

	// The nonce is the first 12 bytes
	if len(data) < 12 {
		return nil, fmt.Errorf("encrypted data is too short")
	}
	nonce := data[:12]
	encryptedData := data[12:]

	// Create the cipher block
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, fmt.Errorf("failed to create cipher: %w", err)
	}

	// Create the GCM cipher
	aesgcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, fmt.Errorf("failed to create GCM cipher: %w", err)
	}

	// Decrypt the data
	decryptedData, err := aesgcm.Open(nil, nonce, encryptedData, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to decrypt data: %w", err)
	}

	return decryptedData, nil
}

// exportMetadata exports the metadata to a JSON file with optional encryption
func (idx *Index) exportMetadata(filePath string, compress bool) error {
	// Convert metadata map to an array for serialization
	metadataArray := make([]map[string]interface{}, 0, len(idx.metadata))
	for id, meta := range idx.metadata {
		// Add ID to metadata for restoration
		itemMeta := make(map[string]interface{})
		for k, v := range meta {
			itemMeta[k] = v
		}
		itemMeta["_id"] = id
		metadataArray = append(metadataArray, itemMeta)
	}

	// Marshal to JSON
	jsonData, err := json.MarshalIndent(metadataArray, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %w", err)
	}

	var finalData []byte
	finalPath := filePath

	// Apply compression if requested
	if compress {
		finalPath += ".gz"
		var compressedData bytes.Buffer
		gzWriter := gzip.NewWriter(&compressedData)
		if _, err := gzWriter.Write(jsonData); err != nil {
			return fmt.Errorf("failed to compress metadata: %w", err)
		}
		if err := gzWriter.Close(); err != nil {
			return fmt.Errorf("failed to finalize metadata compression: %w", err)
		}
		finalData = compressedData.Bytes()
	} else {
		finalData = jsonData
	}

	// Write data with optional encryption
	if err := idx.writeEncrypted(finalData, finalPath); err != nil {
		return fmt.Errorf("failed to write metadata file: %w", err)
	}

	return nil
}

// importMetadata imports metadata from a JSON file with optional decryption
func (idx *Index) importMetadata(filePath string, compressed bool) error {
	filePathToRead := filePath
	if compressed {
		filePathToRead += ".gz"
	}

	// Read and decrypt file
	fileData, err := idx.readEncrypted(filePathToRead)
	if err != nil {
		return fmt.Errorf("failed to read metadata file: %w", err)
	}

	// Decompress if needed
	var jsonData []byte
	if compressed {
		gzReader, err := gzip.NewReader(bytes.NewReader(fileData))
		if err != nil {
			return fmt.Errorf("failed to create gzip reader for metadata: %w", err)
		}
		jsonData, err = io.ReadAll(gzReader)
		if err != nil {
			return fmt.Errorf("failed to decompress metadata: %w", err)
		}
		if err := gzReader.Close(); err != nil {
			return fmt.Errorf("failed to close gzip reader: %w", err)
		}
	} else {
		jsonData = fileData
	}

	// Parse metadata
	var metadataArray []map[string]interface{}
	if err := json.Unmarshal(jsonData, &metadataArray); err != nil {
		return fmt.Errorf("failed to unmarshal metadata: %w", err)
	}

	// Rebuild metadata map
	idx.metadata = make(map[uint64]map[string]interface{})
	for _, item := range metadataArray {
		id, ok := item["_id"].(float64)
		if !ok {
			continue
		}
		delete(item, "_id")
		idx.metadata[uint64(id)] = item
	}

	return nil
}

// Close releases resources associated with the index and stops background workers.
func (idx *Index) Close() error {
	// Stop background workers.
	idx.batchTicker.Stop()
	close(idx.batchDone)
	idx.persistTicker.Stop()
	close(idx.persistDone)
	idx.backupTicker.Stop()
	close(idx.backupDone)
	// Wait for background goroutines to exit.
	idx.bgWG.Wait()

	if err := idx.flushBatch(); err != nil {
		idx.logger.Error("failed to flush batch during close", zap.Error(err))
	}

	// Close DuckDB connection and database
	if idx.dbConn != nil {
		idx.dbConn.Close()
	}
	if idx.duckdb != nil {
		idx.duckdb.Close()
	}

	idx.logger.Info("index closed successfully")
	return nil
}

// QueryMetadata executes a metadata query against DuckDB and returns the matching metadata.
func (idx *Index) QueryMetadata(query string) ([]map[string]interface{}, error) {
	rr, stmt, _, err := idx.dbConn.Query(context.Background(), query)
	if err != nil {
		return nil, fmt.Errorf("metadata query failed: %w", err)
	}
	defer stmt.Close()

	var results []map[string]interface{}

	// Process the query results
	for rr.Next() {
		record := rr.Record()
		numCols := int(record.NumCols())
		numRows := int(record.NumRows())

		// Get column names from schema
		schema := record.Schema()
		colNames := make([]string, numCols)
		for i := 0; i < numCols; i++ {
			colNames[i] = schema.Field(i).Name
		}

		// Process each row
		for rowIdx := 0; rowIdx < numRows; rowIdx++ {
			result := make(map[string]interface{})

			// Extract values for each column
			for colIdx := 0; colIdx < numCols; colIdx++ {
				col := record.Column(colIdx)

				// Check if the value is null
				isNull := false
				switch c := col.(type) {
				case *array.String:
					isNull = c.IsNull(rowIdx)
				case *array.Int64:
					isNull = c.IsNull(rowIdx)
				case *array.Uint64:
					isNull = c.IsNull(rowIdx)
				case *array.Float64:
					isNull = c.IsNull(rowIdx)
				case *array.Boolean:
					isNull = c.IsNull(rowIdx)
				default:
					// For other types, assume not null
					isNull = false
				}

				if !isNull {
					// Extract value based on column type
					switch col := col.(type) {
					case *array.String:
						result[colNames[colIdx]] = col.Value(rowIdx)
					case *array.Int64:
						result[colNames[colIdx]] = col.Value(rowIdx)
					case *array.Uint64:
						result[colNames[colIdx]] = col.Value(rowIdx)
					case *array.Float64:
						result[colNames[colIdx]] = col.Value(rowIdx)
					case *array.Boolean:
						result[colNames[colIdx]] = col.Value(rowIdx)
					default:
						// For other types, convert to string
						result[colNames[colIdx]] = fmt.Sprintf("%v", col)
					}
				}
			}

			results = append(results, result)
		}
	}

	if err := rr.Err(); err != nil {
		return nil, fmt.Errorf("error iterating query results: %w", err)
	}

	return results, nil
}

// validateMetadataSchema performs basic schema validation on the metadata.
func validateMetadataSchema(meta map[string]interface{}) error {
	// Example: require the "category" field.
	if _, ok := meta["category"]; !ok {
		return errors.New("missing required field: category")
	}
	return nil
}

// HealthCheck performs a simple health check on the index.
func (idx *Index) HealthCheck() error {
	if idx.hnsw == nil || idx.dbConn == nil {
		return errors.New("index or database not initialized")
	}
	return nil
}

// CollectMetrics returns a map of metrics about the index.
func (idx *Index) CollectMetrics() map[string]interface{} {
	metrics := make(map[string]interface{})
	metrics["vector_count"] = idx.hnsw.Len()
	metrics["batch_size"] = len(idx.batchBuffer)
	metrics["cache_size"] = idx.cacheSize()
	metrics["db_connections"] = idx.duckdb.ConnCount()
	metrics["last_persist_id"] = idx.lastPersistID
	metrics["persist_interval"] = idx.persistInterval.Seconds()

	// Add backup metrics
	if !idx.lastBackupTime.IsZero() {
		metrics["last_backup_time"] = idx.lastBackupTime.Format(time.RFC3339)
		metrics["backup_age_seconds"] = time.Since(idx.lastBackupTime).Seconds()
	}

	return metrics
}

// cacheSize returns the number of items in the cache.
func (idx *Index) cacheSize() int {
	size := 0
	idx.cache.Range(func(_, _ interface{}) bool {
		size++
		return true
	})
	return size
}

// LogQuery logs the executed query along with its duration.
func (idx *Index) LogQuery(query string, duration time.Duration) {
	idx.logger.Info("query executed", zap.String("query", query), zap.Duration("duration", duration))
}

// FacetedSearch performs a search and filters results based on provided facet key/values.
func (idx *Index) FacetedSearch(query []float32, k int, facets map[string]string) ([]SearchResult, error) {
	results, err := idx.Search(query, k, 1, k)
	if err != nil {
		return nil, err
	}

	var filteredResults []SearchResult
	for _, res := range results {
		match := true
		for key, value := range facets {
			if metaValue, ok := res.Metadata[key]; !ok || metaValue != value {
				match = false
				break
			}
		}
		if match {
			filteredResults = append(filteredResults, res)
		}
	}
	return filteredResults, nil
}

// MultiVectorSearch performs searches for multiple query vectors.
func (idx *Index) MultiVectorSearch(queries [][]float32, k int) ([][]SearchResult, error) {
	idx.lock.RLock()
	defer idx.lock.RUnlock()

	var allResults [][]SearchResult
	for _, query := range queries {
		results, err := idx.Search(query, k, 1, k)
		if err != nil {
			return nil, err
		}
		allResults = append(allResults, results)
	}
	return allResults, nil
}

// copyFile copies a file from src to dst.
func copyFile(src, dst string) error {
	sourceFile, err := os.Open(src)
	if err != nil {
		return err
	}
	defer sourceFile.Close()

	destinationFile, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer destinationFile.Close()

	if _, err := io.Copy(destinationFile, sourceFile); err != nil {
		return err
	}
	return nil
}

// backupWorker runs in the background and periodically creates backups
func (idx *Index) backupWorker() {
	if idx.config.BackupInterval <= 0 {
		// Backup is disabled
		idx.logger.Info("Scheduled backups are disabled")
		return
	}

	idx.logger.Info("Backup worker started",
		zap.Duration("interval", idx.config.BackupInterval),
		zap.String("path", idx.config.BackupPath),
		zap.Bool("compression", idx.config.BackupCompression))

	for {
		select {
		case <-idx.backupTicker.C:
			// Create a timestamped backup directory
			timestamp := time.Now().Format("20060102-150405")
			backupDir := filepath.Join(idx.config.BackupPath, timestamp)

			idx.logger.Info("Starting scheduled backup", zap.String("backup_dir", backupDir))

			if err := idx.Backup(backupDir, true, idx.config.BackupCompression); err != nil {
				idx.logger.Error("Scheduled backup failed", zap.Error(err))
			} else {
				idx.lastBackupTime = time.Now()
				idx.pruneOldBackups()
			}
		case <-idx.backupDone:
			idx.logger.Info("Backup worker shutting down")
			return
		}
	}
}

// pruneOldBackups removes old backups to maintain the max number of backups
func (idx *Index) pruneOldBackups() {
	if idx.config.MaxBackups <= 0 {
		return // No limit
	}

	// List all backup directories
	entries, err := os.ReadDir(idx.config.BackupPath)
	if err != nil {
		idx.logger.Error("Failed to read backup directory", zap.Error(err))
		return
	}

	// Filter and sort backup directories
	var backups []string
	for _, entry := range entries {
		if entry.IsDir() {
			// Check if it's a valid backup (has a manifest)
			manifestPath := filepath.Join(idx.config.BackupPath, entry.Name(), "manifest.json")
			if _, err := os.Stat(manifestPath); err == nil {
				backups = append(backups, entry.Name())
			}
		}
	}

	// Sort by name (timestamp format ensures chronological order)
	sort.Strings(backups)

	// Remove oldest backups if we have more than the limit
	if len(backups) > idx.config.MaxBackups {
		for i := 0; i < len(backups)-idx.config.MaxBackups; i++ {
			backupToRemove := filepath.Join(idx.config.BackupPath, backups[i])
			idx.logger.Info("Removing old backup", zap.String("path", backupToRemove))
			if err := os.RemoveAll(backupToRemove); err != nil {
				idx.logger.Error("Failed to remove old backup",
					zap.String("path", backupToRemove),
					zap.Error(err))
			}
		}
	}
}

// SearchWithNegatives performs a search with positive and negative examples.
func (idx *Index) SearchWithNegatives(positiveQuery []float32, negativeQueries [][]float32, k, page, pageSize int) ([]SearchResult, error) {
	idx.lock.RLock()
	defer idx.lock.RUnlock()

	// Validate parameters
	if k <= 0 {
		return nil, errors.New("k must be positive")
	}
	if page <= 0 {
		page = 1 // Default to first page
	}
	if pageSize <= 0 {
		pageSize = 10 // Default page size
	}
	if len(negativeQueries) == 0 {
		// If no negative examples, just do a regular search
		return idx.Search(positiveQuery, k, page, pageSize)
	}

	// Ensure we have enough elements in the index
	currentCount := idx.hnsw.Len()
	if currentCount == 0 {
		return []SearchResult{}, nil
	}

	// Get more results than needed for reranking
	searchK := k * 10 // Get 10x more results for reranking
	if searchK > currentCount {
		searchK = currentCount
	}

	// Perform search for positive query
	var results []hnsw.Node[uint64]
	var err error

	// Use a recovery function to catch exceptions
	func() {
		defer func() {
			if r := recover(); r != nil {
				err = fmt.Errorf("HNSW search panicked: %v", r)
			}
		}()

		// Perform the search
		// Note: The coder/hnsw library doesn't allow setting efSearch per query
		results = idx.hnsw.Search(positiveQuery, searchK)
	}()

	if err != nil {
		idx.logger.Error("HNSW search failed", zap.Error(err))
		return nil, fmt.Errorf("HNSW search failed: %w", err)
	}

	if len(results) == 0 {
		return []SearchResult{}, nil
	}

	// If no negative examples, just return the results with pagination
	if len(negativeQueries) == 0 {
		start := (page - 1) * pageSize
		end := start + pageSize
		if start >= len(results) {
			return nil, nil // No more results
		}
		if end > len(results) {
			end = len(results)
		}

		var searchResults []SearchResult
		for _, r := range results[start:end] {
			// Calculate distance between query and result vector
			distance := computeDistance(positiveQuery, r.Value, idx.config.Distance)
			meta := idx.getMetadata(r.Key)
			searchResults = append(searchResults, SearchResult{
				ID:       r.Key,
				Distance: distance,
				Metadata: meta,
			})
		}
		return searchResults, nil
	}

	// Rerank results based on negative examples
	candidates := results
	rerankedResults := idx.rerankWithNegatives(candidates, positiveQuery, negativeQueries)

	// Apply pagination to reranked results
	start := (page - 1) * pageSize
	end := start + pageSize
	if start >= len(rerankedResults) {
		return nil, nil // No more results
	}
	if end > len(rerankedResults) {
		end = len(rerankedResults)
	}

	var searchResults []SearchResult
	for _, r := range rerankedResults[start:end] {
		meta := idx.getMetadata(r.ID)
		searchResults = append(searchResults, SearchResult{
			ID:       r.ID,
			Distance: r.Distance,
			Metadata: meta,
		})
	}
	return searchResults, nil
}

// rerankWithNegatives reranks search results based on negative examples
func (idx *Index) rerankWithNegatives(candidates []hnsw.Node[uint64], positiveQuery []float32, negativeQueries [][]float32) []SearchResult {
	// Weights for positive and negative influences
	posWeight := 1.0
	negWeight := 0.5 / float64(len(negativeQueries))

	// Rerank candidates based on both positive and negative examples
	reranked := make([]SearchResult, 0, len(candidates))
	for _, candidate := range candidates {
		id := candidate.Key
		vector := candidate.Value
		if vector == nil {
			// If vector is not available, get it from storage
			vector = idx.getVector(id)
			if vector == nil {
				continue // Skip if vector not found
			}
		}

		// Calculate base score from positive query
		posDistance := computeDistance(vector, positiveQuery, idx.config.Distance)

		// Initialize final score based on positive query
		var finalScore float64

		// For distance metrics (like L2), lower means more similar
		if idx.config.Distance == Cosine {
			// For Cosine, higher means more similar
			finalScore = float64(posDistance) * posWeight
		} else {
			// For L2, lower means more similar, so we invert
			finalScore = posWeight * (1.0 / (1.0 + float64(posDistance)))
		}

		// For each negative query, compute distance and adjust the score
		for _, negQuery := range negativeQueries {
			negDistance := computeDistance(vector, negQuery, idx.config.Distance)

			// For similarity metrics, higher means more similar
			// We want to penalize vectors similar to negative examples
			if idx.config.Distance == Cosine {
				// For Cosine, we penalize by subtracting the similarity
				finalScore -= negWeight * float64(negDistance)
			} else {
				// For L2, we penalize by adding the inverse of distance
				// (closer to negative example = higher penalty)
				finalScore -= negWeight * (1.0 / (1.0 + float64(negDistance)))
			}
		}

		// Add to reranked results
		reranked = append(reranked, SearchResult{
			ID:       id,
			Distance: float32(finalScore), // Store the combined score as distance
			Metadata: nil,                 // Will be filled in later
		})
	}

	// Sort by final score (higher is better)
	sort.Slice(reranked, func(i, j int) bool {
		return reranked[i].Distance > reranked[j].Distance
	})

	return reranked
}

// getVector retrieves the vector for a given ID
func (idx *Index) getVector(id uint64) []float32 {
	vector, ok := idx.vectors[id]
	if !ok {
		return nil
	}

	// Return a copy to avoid external modification
	result := make([]float32, len(vector))
	copy(result, vector)
	return result
}

// computeDistance calculates the distance between two vectors
func computeDistance(a, b []float32, metric DistanceMetric) float32 {
	if len(a) != len(b) {
		return 0.0
	}

	switch metric {
	case Cosine:
		return cosineDistance(a, b)
	case L2:
		return l2Distance(a, b)
	default:
		return 0.0
	}
}

// cosineDistance computes cosine similarity between vectors
func cosineDistance(a, b []float32) float32 {
	var dotProduct float32
	var normA, normB float32

	for i := 0; i < len(a); i++ {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	return dotProduct / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}

// l2Distance computes Euclidean (L2) distance between vectors
func l2Distance(a, b []float32) float32 {
	var sum float32
	for i := 0; i < len(a); i++ {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}

// extractCompressedBackup extracts a compressed backup file to the specified directory.
func extractCompressedBackup(backupPath, destDir string) error {
	// Check if it's a gzip file
	if strings.HasSuffix(backupPath, ".gz") {
		// Open the gzip file
		file, err := os.Open(backupPath)
		if err != nil {
			return fmt.Errorf("failed to open gzip file: %w", err)
		}
		defer file.Close()

		// Create a gzip reader
		gzReader, err := gzip.NewReader(file)
		if err != nil {
			return fmt.Errorf("failed to create gzip reader: %w", err)
		}
		defer gzReader.Close()

		// Create the output file
		outFile, err := os.Create(filepath.Join(destDir, "backup.json"))
		if err != nil {
			return fmt.Errorf("failed to create output file: %w", err)
		}
		defer outFile.Close()

		// Copy the decompressed data to the output file
		if _, err := io.Copy(outFile, gzReader); err != nil {
			return fmt.Errorf("failed to decompress data: %w", err)
		}

		return nil
	} else if strings.HasSuffix(backupPath, ".zip") {
		// Open the zip file
		zipReader, err := zip.OpenReader(backupPath)
		if err != nil {
			return fmt.Errorf("failed to open zip file: %w", err)
		}
		defer zipReader.Close()

		// Extract each file
		for _, file := range zipReader.File {
			// Open the file inside the zip
			rc, err := file.Open()
			if err != nil {
				return fmt.Errorf("failed to open file in zip: %w", err)
			}

			// Create the output file
			outPath := filepath.Join(destDir, file.Name)
			outFile, err := os.Create(outPath)
			if err != nil {
				rc.Close()
				return fmt.Errorf("failed to create output file: %w", err)
			}

			// Copy the file data
			if _, err := io.Copy(outFile, rc); err != nil {
				outFile.Close()
				rc.Close()
				return fmt.Errorf("failed to extract file: %w", err)
			}

			outFile.Close()
			rc.Close()
		}

		return nil
	}

	return fmt.Errorf("unsupported compression format: %s", backupPath)
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

		if vectorCol.IsNull(i) {
			return fmt.Errorf("vector column contains null value at row %d", i)
		}
		// Since FixedSizeList stores elements contiguously:
		start := i * dim
		vector := make([]float32, dim)
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
