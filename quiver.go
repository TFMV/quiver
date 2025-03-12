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
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/TFMV/hnsw"
	"github.com/apache/arrow-adbc/go/adbc"
	"github.com/apache/arrow-adbc/go/adbc/drivermgr"
	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/bytedance/sonic"
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
	// Parallel search configuration
	EnableParallelSearch bool // Whether to use parallel search for large queries
	NumSearchWorkers     int  // Number of workers for parallel search (0 = use all available cores)
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
	// Dimensionality reduction configuration
	EnableDimReduction      bool    // Whether to enable dimensionality reduction
	DimReductionMethod      string  // Method to use for dimensionality reduction (e.g., "PCA")
	DimReductionTarget      int     // Target dimension for reduction
	DimReductionAdaptive    bool    // Whether to use adaptive dimensionality reduction
	DimReductionMinVariance float64 // Minimum variance to explain (0.0-1.0) for adaptive reduction
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
	connPool        *ConnectionPool // Connection pool for concurrent database access
	lock            sync.RWMutex
	searchLock      sync.Mutex // Dedicated mutex for search operations
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
	// Dimensionality reduction fields
	dimReducer        interface{} // Will hold a dimreduce.DimReducer if enabled
	originalDimension int         // Original dimension before reduction
	allocator         memory.Allocator
	flushSemaphore    chan struct{}
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
	inUse  bool // Track whether the connection is currently in use
}

// ConnectionPool manages a pool of DuckDB connections.
type ConnectionPool struct {
	db                *DuckDB
	connections       []*DuckDBConn
	maxSize           int
	initialSize       int
	mu                sync.Mutex
	connAvailable     *sync.Cond
	preparedStmts     map[string]*PreparedStatement // Cache for prepared statements
	stmtMu            sync.RWMutex                  // Mutex for prepared statement cache
	lastUsedConn      *DuckDBConn                   // Track the last used connection for potential reuse
	lastUsedConnMu    sync.Mutex                    // Mutex for last used connection
	batchConn         *DuckDBConn                   // Dedicated connection for batch operations
	batchConnMu       sync.Mutex                    // Mutex for batch connection
	threadConnections sync.Map                      // Thread-local connections (goroutine ID -> connection)
	stats             poolStats                     // Connection pool statistics
	stopCleanup       chan struct{}                 // Channel to signal cleanup goroutine to stop
}

// poolStats tracks statistics about the connection pool usage
type poolStats struct {
	gets              int64
	releases          int64
	waits             int64
	hits              int64 // Connection reuse hits
	misses            int64 // Connection reuse misses
	creations         int64 // New connection creations
	threadReuseHits   int64 // Thread-local connection reuse hits
	threadReuseMisses int64 // Thread-local connection reuse misses
	mu                sync.Mutex
}

// NewConnectionPool creates a new connection pool for DuckDB
func NewConnectionPool(duckdb *DuckDB, initialSize, maxSize int) (*ConnectionPool, error) {
	if initialSize <= 0 {
		initialSize = 1
	}
	if maxSize <= 0 {
		maxSize = 10
	}
	if initialSize > maxSize {
		initialSize = maxSize
	}

	pool := &ConnectionPool{
		db:            duckdb,
		connections:   make([]*DuckDBConn, 0, initialSize),
		maxSize:       maxSize,
		initialSize:   initialSize,
		preparedStmts: make(map[string]*PreparedStatement),
		stopCleanup:   make(chan struct{}), // Initialize the stop channel
	}
	pool.connAvailable = sync.NewCond(&pool.mu)

	// Initialize with initial connections
	for i := 0; i < initialSize; i++ {
		conn, err := duckdb.OpenConnection()
		if err != nil {
			// Close any connections we've already opened
			for _, c := range pool.connections {
				c.Close()
			}
			return nil, fmt.Errorf("failed to initialize connection pool: %w", err)
		}
		pool.connections = append(pool.connections, conn)

		// Track connection creation
		pool.stats.mu.Lock()
		pool.stats.creations++
		pool.stats.mu.Unlock()
	}

	// Create a dedicated connection for batch operations
	batchConn, err := duckdb.OpenConnection()
	if err != nil {
		// Close any connections we've already opened
		for _, c := range pool.connections {
			c.Close()
		}
		return nil, fmt.Errorf("failed to create batch connection: %w", err)
	}
	pool.batchConn = batchConn

	// Track connection creation
	pool.stats.mu.Lock()
	pool.stats.creations++
	pool.stats.mu.Unlock()

	// Create metadata table if it doesn't exist using the batch connection
	_, err = pool.batchConn.Exec(context.Background(), `CREATE TABLE IF NOT EXISTS metadata (
		id BIGINT PRIMARY KEY,
		json VARCHAR
	)`)
	if err != nil {
		// Close all connections
		pool.batchConn.Close()
		for _, c := range pool.connections {
			c.Close()
		}
		return nil, fmt.Errorf("failed to create metadata table: %w", err)
	}

	// Start a background goroutine to clean up unused prepared statements
	go pool.cleanupPreparedStatements()

	return pool, nil
}

// getGoroutineID returns a unique identifier for the current goroutine
// This is used for thread-local connection affinity
func getGoroutineID() uint64 {
	b := make([]byte, 64)
	b = b[:runtime.Stack(b, false)]
	// Parse the goroutine ID from the first line of the stack trace
	// Format: "goroutine 123 [running]:"
	s := strings.TrimPrefix(string(b), "goroutine ")
	s = s[:strings.IndexByte(s, ' ')]
	id, _ := strconv.ParseUint(s, 10, 64)
	return id
}

// GetBatchConnection gets the dedicated connection for batch operations
func (p *ConnectionPool) GetBatchConnection() (*DuckDBConn, error) {
	p.batchConnMu.Lock()
	defer p.batchConnMu.Unlock()

	// If the batch connection is not in use, mark it as in use and return it
	if !p.batchConn.inUse {
		p.batchConn.inUse = true

		// Track statistics
		p.stats.mu.Lock()
		p.stats.gets++
		p.stats.hits++
		p.stats.mu.Unlock()

		return p.batchConn, nil
	}

	// If the batch connection is in use, fall back to getting a regular connection
	p.stats.mu.Lock()
	p.stats.misses++
	p.stats.mu.Unlock()

	return p.GetConnection()
}

// ReleaseBatchConnection releases the dedicated batch connection
func (p *ConnectionPool) ReleaseBatchConnection(conn *DuckDBConn) {
	p.batchConnMu.Lock()
	defer p.batchConnMu.Unlock()

	// If this is the batch connection, mark it as not in use
	if conn == p.batchConn {
		p.batchConn.inUse = false

		// Track statistics
		p.stats.mu.Lock()
		p.stats.releases++
		p.stats.mu.Unlock()

		return
	}

	// Otherwise, release it as a regular connection
	p.ReleaseConnection(conn)
}

// GetConnection gets a connection from the pool, creating a new one if necessary.
// This method blocks until a connection is available or can be created.
func (p *ConnectionPool) GetConnection() (*DuckDBConn, error) {
	// Track statistics
	p.stats.mu.Lock()
	p.stats.gets++
	p.stats.mu.Unlock()

	// First, try to get a thread-local connection if one exists
	goroutineID := getGoroutineID()
	if conn, ok := p.threadConnections.Load(goroutineID); ok {
		threadConn := conn.(*DuckDBConn)

		// Check if the connection is in use
		p.mu.Lock()
		if !threadConn.inUse {
			threadConn.inUse = true
			p.mu.Unlock()

			// Track statistics
			p.stats.mu.Lock()
			p.stats.threadReuseHits++
			p.stats.mu.Unlock()

			return threadConn, nil
		}
		p.mu.Unlock()

		// Track statistics
		p.stats.mu.Lock()
		p.stats.threadReuseMisses++
		p.stats.mu.Unlock()
	}

	// Next, try to reuse the last used connection if it's available
	p.lastUsedConnMu.Lock()
	if p.lastUsedConn != nil {
		p.mu.Lock()
		if !p.lastUsedConn.inUse {
			p.lastUsedConn.inUse = true
			p.mu.Unlock()
			p.lastUsedConnMu.Unlock()

			// Store this connection as thread-local for this goroutine
			p.threadConnections.Store(goroutineID, p.lastUsedConn)

			// Track statistics
			p.stats.mu.Lock()
			p.stats.hits++
			p.stats.mu.Unlock()

			return p.lastUsedConn, nil
		}
		p.mu.Unlock()
	}
	p.lastUsedConnMu.Unlock()

	// Track statistics
	p.stats.mu.Lock()
	p.stats.misses++
	p.stats.mu.Unlock()

	p.mu.Lock()
	defer p.mu.Unlock()

	// Try to find an available connection
	for {
		// First, check for an existing available connection
		for _, conn := range p.connections {
			if !conn.inUse {
				conn.inUse = true

				// Update the last used connection
				p.lastUsedConnMu.Lock()
				p.lastUsedConn = conn
				p.lastUsedConnMu.Unlock()

				// Store this connection as thread-local for this goroutine
				p.threadConnections.Store(goroutineID, conn)

				return conn, nil
			}
		}

		// If we have room to create a new connection, do so
		if len(p.connections) < p.maxSize {
			conn, err := p.db.OpenConnection()
			if err != nil {
				return nil, fmt.Errorf("failed to open new connection: %w", err)
			}
			conn.inUse = true
			p.connections = append(p.connections, conn)

			// Update the last used connection
			p.lastUsedConnMu.Lock()
			p.lastUsedConn = conn
			p.lastUsedConnMu.Unlock()

			// Store this connection as thread-local for this goroutine
			p.threadConnections.Store(goroutineID, conn)

			// Track statistics
			p.stats.mu.Lock()
			p.stats.creations++
			p.stats.mu.Unlock()

			return conn, nil
		}

		// Otherwise, wait for a connection to become available
		p.stats.mu.Lock()
		p.stats.waits++
		p.stats.mu.Unlock()

		p.connAvailable.Wait()
	}
}

// ReleaseConnection returns a connection to the pool.
func (p *ConnectionPool) ReleaseConnection(conn *DuckDBConn) {
	// Track statistics
	p.stats.mu.Lock()
	p.stats.releases++
	p.stats.mu.Unlock()

	p.mu.Lock()
	defer p.mu.Unlock()

	// Mark the connection as available
	for _, c := range p.connections {
		if c == conn {
			c.inUse = false
			break
		}
	}

	// Signal that a connection is available
	p.connAvailable.Signal()
}

// GetPreparedStatement gets a prepared statement from the cache or creates a new one
func (p *ConnectionPool) GetPreparedStatement(query string) (*PreparedStatement, error) {
	// Check if the statement is already in the cache
	p.stmtMu.RLock()
	stmt, exists := p.preparedStmts[query]
	if exists {
		// Update usage statistics
		stmt.lastUsed = time.Now()
		stmt.usageCount++
		p.stmtMu.RUnlock()
		return stmt, nil
	}
	p.stmtMu.RUnlock()

	// Get a connection for the new prepared statement
	conn, err := p.GetConnection()
	if err != nil {
		return nil, fmt.Errorf("failed to get connection for prepared statement: %w", err)
	}

	// Create a new prepared statement entry
	newStmt := &PreparedStatement{
		query:      query,
		conn:       conn,
		lastUsed:   time.Now(),
		usageCount: 1,
	}

	// Add to cache
	p.stmtMu.Lock()
	p.preparedStmts[query] = newStmt
	p.stmtMu.Unlock()

	return newStmt, nil
}

// ReleasePreparedStatement releases a prepared statement back to the pool
func (p *ConnectionPool) ReleasePreparedStatement(stmt *PreparedStatement) {
	if stmt != nil && stmt.conn != nil {
		p.ReleaseConnection(stmt.conn)
	}
}

// Close closes all connections in the pool and cleans up prepared statements.
func (p *ConnectionPool) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	// Signal the cleanup goroutine to stop
	close(p.stopCleanup)

	// Close all prepared statements
	p.stmtMu.Lock()
	// Just clear the map since we don't have a direct way to close statements
	p.preparedStmts = nil
	p.stmtMu.Unlock()

	// Close all connections
	var lastErr error
	for _, conn := range p.connections {
		if err := conn.Close(); err != nil {
			lastErr = err
		}
	}

	// Close the batch connection if it exists
	if p.batchConn != nil {
		if err := p.batchConn.Close(); err != nil && lastErr == nil {
			lastErr = err
		}
	}

	p.connections = nil
	return lastErr
}

// Size returns the current number of connections in the pool.
func (p *ConnectionPool) Size() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.connections)
}

// AvailableConnections returns the number of available connections in the pool.
func (p *ConnectionPool) AvailableConnections() int {
	p.mu.Lock()
	defer p.mu.Unlock()

	count := 0
	for _, conn := range p.connections {
		if !conn.inUse {
			count++
		}
	}
	return count
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

// New creates a new Quiver index with the given configuration.
func New(config Config, logger *zap.Logger) (*Index, error) {
	// Validate configuration
	issues := ValidateConfig(config)

	// Check for critical errors
	var errorCount int
	for _, issue := range issues {
		if issue.Severity == Error {
			errorCount++
			logger.Error("Configuration error",
				zap.String("field", issue.Field),
				zap.Any("value", issue.Value),
				zap.String("message", issue.Message),
				zap.String("suggestion", issue.Suggestion))
		}
	}

	// If there are errors, return a detailed error message
	if errorCount > 0 {
		return nil, fmt.Errorf("invalid configuration: %d critical errors found. See logs for details", errorCount)
	}

	// Log warnings
	for _, issue := range issues {
		if issue.Severity == Warning {
			logger.Warn("Configuration warning",
				zap.String("field", issue.Field),
				zap.Any("value", issue.Value),
				zap.String("message", issue.Message),
				zap.String("suggestion", issue.Suggestion))
		}
	}

	// Log informational issues
	for _, issue := range issues {
		if issue.Severity == Info {
			logger.Info("Configuration suggestion",
				zap.String("field", issue.Field),
				zap.Any("value", issue.Value),
				zap.String("message", issue.Message),
				zap.String("suggestion", issue.Suggestion))
		}
	}

	// Set default values for optional fields
	if config.BatchSize <= 0 {
		config.BatchSize = 1000
		logger.Info("Using default batch size", zap.Int("batchSize", config.BatchSize))
	}

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

	// If dimensionality reduction is enabled, validate the configuration
	originalDimension := config.Dimension
	if config.EnableDimReduction {
		if config.DimReductionTarget <= 0 || config.DimReductionTarget >= config.Dimension {
			return nil, fmt.Errorf("invalid target dimension for reduction: %d (must be > 0 and < %d)",
				config.DimReductionTarget, config.Dimension)
		}

		if config.DimReductionMethod == "" {
			config.DimReductionMethod = "PCA" // Default to PCA
		}

		if config.DimReductionAdaptive && (config.DimReductionMinVariance <= 0 || config.DimReductionMinVariance > 1.0) {
			return nil, fmt.Errorf("invalid minimum variance for adaptive reduction: %f (must be > 0 and <= 1.0)",
				config.DimReductionMinVariance)
		}

		// Update the dimension to the target dimension for the HNSW graph
		config.Dimension = config.DimReductionTarget
		logger.Info("Dimensionality reduction enabled",
			zap.Int("original_dimension", originalDimension),
			zap.Int("reduced_dimension", config.Dimension),
			zap.String("method", config.DimReductionMethod))
	}

	// Initialize HNSW graph with the specified parameters
	var graph *hnsw.Graph[uint64]
	var err error

	// Configure the distance function based on the selected metric
	var distanceFunc hnsw.DistanceFunc
	switch config.Distance {
	case Cosine:
		distanceFunc = hnsw.CosineDistance
	case L2:
		distanceFunc = hnsw.EuclideanDistance
	default:
		distanceFunc = hnsw.CosineDistance // Default to cosine distance
	}

	// Create a new HNSW graph with the specified configuration
	graph, err = hnsw.NewGraphWithConfig[uint64](
		config.HNSWM,
		0.5, // Default Ml value (layer size ratio)
		config.HNSWEfSearch,
		distanceFunc,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create HNSW graph: %w", err)
	}

	// Open DuckDB connection using StoragePath
	duckdb, err := NewDuckDB(WithPath(config.StoragePath))
	if err != nil {
		return nil, fmt.Errorf("failed to open DuckDB: %w", err)
	}

	// Create a connection pool with appropriate sizing
	initialPoolSize := 2                // Start with a smaller pool
	maxPoolSize := runtime.NumCPU() * 2 // Scale with available CPUs

	// Cap the maximum pool size to avoid excessive connections
	if maxPoolSize > 20 {
		maxPoolSize = 20
	}

	connPool, err := NewConnectionPool(duckdb, initialPoolSize, maxPoolSize)
	if err != nil {
		duckdb.Close()
		return nil, fmt.Errorf("failed to create connection pool: %w", err)
	}

	idx := &Index{
		config:            config,
		hnsw:              graph,
		metadata:          make(map[uint64]map[string]interface{}),
		vectors:           make(map[uint64][]float32),
		duckdb:            duckdb,
		connPool:          connPool,
		lock:              sync.RWMutex{},
		searchLock:        sync.Mutex{}, // Initialize the search lock
		batchBuffer:       make([]vectorMeta, 0, config.BatchSize),
		batchLock:         sync.Mutex{},
		logger:            logger,
		batchTicker:       time.NewTicker(time.Second),
		batchDone:         make(chan struct{}),
		persistInterval:   persistInterval,
		persistTicker:     time.NewTicker(persistInterval),
		persistDone:       make(chan struct{}),
		backupTicker:      time.NewTicker(backupInterval),
		backupDone:        make(chan struct{}),
		originalDimension: originalDimension,
		allocator:         memory.NewGoAllocator(),
		flushSemaphore:    make(chan struct{}, 1),
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

// flushBatch flushes the current batch buffer to the database and updates in-memory structures.
func (idx *Index) flushBatch() error {
	// Acquire batch lock and copy the current batch
	idx.batchLock.Lock()
	if len(idx.batchBuffer) == 0 {
		idx.batchLock.Unlock()
		return nil // Nothing to flush
	}

	// Make a copy of the batch buffer to process
	batchToProcess := make([]vectorMeta, len(idx.batchBuffer))
	copy(batchToProcess, idx.batchBuffer)

	// Reset the batch buffer and release the lock
	idx.batchBuffer = idx.batchBuffer[:0]
	idx.batchLock.Unlock()

	n := len(batchToProcess)

	// Process batch in parallel - prepare metadata outside of any locks
	metadataValues := make([]string, 0, n)
	vectorsToAdd := make([]vectorMeta, 0, n)

	// Use a worker pool to parallelize JSON marshaling
	type marshalResult struct {
		index int
		value string
		err   error
	}

	// Create a fixed-size worker pool based on CPU count
	numWorkers := runtime.NumCPU()
	if numWorkers > n {
		numWorkers = n
	}

	// Use buffered channels to avoid goroutine blocking
	resultChan := make(chan marshalResult, n)
	workChan := make(chan int, n)

	// Start worker goroutines
	var wg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for idx := range workChan {
				item := batchToProcess[idx]
				metaJSON, err := sonic.Marshal(item.meta)
				resultChan <- marshalResult{
					index: idx,
					value: string(metaJSON),
					err:   err,
				}
			}
		}()
	}

	// Send work to workers
	for i := range batchToProcess {
		workChan <- i
	}
	close(workChan)

	// Wait for all workers to finish
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// Process results
	results := make([]marshalResult, n)
	for res := range resultChan {
		if res.err != nil {
			idx.logger.Error("failed to marshal metadata",
				zap.Uint64("id", batchToProcess[res.index].id),
				zap.Error(res.err))
			continue
		}
		results[res.index] = res
	}

	// Build the values list
	for i, res := range results {
		if res.value == "" {
			continue // Skip items that failed to marshal
		}

		item := batchToProcess[i]
		// Escape single quotes in JSON
		escapedJSON := strings.ReplaceAll(res.value, "'", "''")
		metadataValues = append(metadataValues, fmt.Sprintf("(%d, '%s')", item.id, escapedJSON))
		vectorsToAdd = append(vectorsToAdd, item)
	}

	// Execute batch insert using the dedicated batch connection - this is done outside of any locks
	if len(metadataValues) > 0 && idx.connPool != nil {
		// Get the dedicated batch connection
		conn, err := idx.connPool.GetBatchConnection()
		if err != nil {
			idx.logger.Error("failed to get batch connection", zap.Error(err))
			return fmt.Errorf("failed to get batch connection: %w", err)
		}

		// Keep the connection for future operations in this batch
		defer idx.connPool.ReleaseBatchConnection(conn)

		// Create a context with timeout
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()

		// Split large batches into smaller chunks to avoid query size limits
		// Use a more efficient chunking strategy with prepared statements
		const maxBatchSize = 1000

		// For very small batches, use a single query
		if len(metadataValues) <= 50 {
			query := fmt.Sprintf("INSERT INTO metadata (id, json) VALUES %s ON CONFLICT(id) DO UPDATE SET json = excluded.json",
				strings.Join(metadataValues, ","))

			// Execute the query
			_, err = conn.Exec(ctx, query)
			if err != nil {
				idx.logger.Error("failed to execute batch insert", zap.Error(err))
				return fmt.Errorf("failed to execute batch insert: %w", err)
			}
		} else {
			// For larger batches, use multiple queries with transaction
			// Start a transaction
			_, err = conn.Exec(ctx, "BEGIN TRANSACTION")
			if err != nil {
				idx.logger.Error("failed to begin transaction", zap.Error(err))
				return fmt.Errorf("failed to begin transaction: %w", err)
			}

			// Process in chunks
			for i := 0; i < len(metadataValues); i += maxBatchSize {
				end := i + maxBatchSize
				if end > len(metadataValues) {
					end = len(metadataValues)
				}

				chunk := metadataValues[i:end]
				query := fmt.Sprintf("INSERT INTO metadata (id, json) VALUES %s ON CONFLICT(id) DO UPDATE SET json = excluded.json",
					strings.Join(chunk, ","))

				// Execute the query
				_, err = conn.Exec(ctx, query)
				if err != nil {
					// Rollback on error
					conn.Exec(ctx, "ROLLBACK")
					idx.logger.Error("failed to execute batch insert chunk",
						zap.Int("chunk", i/maxBatchSize),
						zap.Error(err))
					return fmt.Errorf("failed to execute batch insert chunk: %w", err)
				}
			}

			// Commit the transaction
			_, err = conn.Exec(ctx, "COMMIT")
			if err != nil {
				idx.logger.Error("failed to commit transaction", zap.Error(err))
				return fmt.Errorf("failed to commit transaction: %w", err)
			}
		}
	}

	// Update in-memory structures in parallel
	// Lock the index for updating shared in-memory structures only
	idx.lock.Lock()
	defer idx.lock.Unlock()

	// Process vectors sequentially to avoid concurrent map writes in HNSW
	for _, item := range vectorsToAdd {
		// Add to HNSW graph
		node := hnsw.MakeNode(item.id, item.vector)
		if err := idx.hnsw.Add(node); err != nil {
			return fmt.Errorf("failed to add vector %d to HNSW graph: %w", item.id, err)
		}

		// Store vector for negative queries
		idx.vectors[item.id] = item.vector

		// Store metadata
		idx.metadata[item.id] = item.meta
	}

	// Update last persist ID
	for _, item := range vectorsToAdd {
		if item.id > idx.lastPersistID {
			idx.lastPersistID = item.id
		}
	}

	idx.logger.Debug("flushed batch", zap.Int("count", n))
	return nil
}

// Add adds a vector to the index with the given ID and metadata.
func (idx *Index) Add(id uint64, vector []float32, metadata map[string]interface{}) error {
	// Validate vector dimension
	if !idx.config.EnableDimReduction || idx.dimReducer == nil {
		if len(vector) != idx.config.Dimension {
			return fmt.Errorf("vector dimension (%d) does not match index dimension (%d)", len(vector), idx.config.Dimension)
		}
	} else {
		// If dimensionality reduction is enabled, check against original dimension
		if len(vector) != idx.originalDimension {
			return fmt.Errorf("vector dimension (%d) does not match original dimension (%d)", len(vector), idx.originalDimension)
		}
	}

	// Make a copy of the metadata to avoid modifying the caller's map
	metaCopy := make(map[string]interface{}, len(metadata))
	for k, v := range metadata {
		metaCopy[k] = v
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

	// Check if batch buffer has reached the configured size
	needsFlush := len(idx.batchBuffer) >= idx.config.BatchSize
	idx.batchLock.Unlock()

	// If batch buffer is full, flush it immediately in a separate goroutine
	// to avoid blocking the caller
	if needsFlush {
		// Use a separate goroutine with a semaphore to limit concurrent flushes
		select {
		case idx.flushSemaphore <- struct{}{}:
			go func() {
				defer func() { <-idx.flushSemaphore }()
				if err := idx.flushBatch(); err != nil {
					idx.logger.Error("failed to flush batch", zap.Error(err))
				}
			}()
		default:
			// If we can't acquire the semaphore, just continue without flushing
			// The batch will be flushed by another goroutine or during the next Add
			idx.logger.Debug("skipping flush due to too many concurrent flushes")
		}
	}

	return nil
}

// Search performs a vector similarity search and returns the k most similar vectors.
// Supports pagination with page and pageSize parameters.
func (idx *Index) Search(query []float32, k, page, pageSize int) ([]SearchResult, error) {
	// Validate query dimension
	if !idx.config.EnableDimReduction || idx.dimReducer == nil {
		if len(query) != idx.config.Dimension {
			return nil, fmt.Errorf("query dimension (%d) does not match index dimension (%d)", len(query), idx.config.Dimension)
		}
	} else {
		// If dimensionality reduction is enabled, check against original dimension
		if len(query) != idx.originalDimension {
			return nil, fmt.Errorf("query dimension (%d) does not match original dimension (%d)", len(query), idx.originalDimension)
		}
	}

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
		// Store the original value so we can restore it after the search
		originalEf := idx.hnsw.EfSearch

		idx.hnsw.EfSearch = efSearch

		// Determine whether to use parallel search
		useParallelSearch := idx.config.EnableParallelSearch &&
			(len(query) >= 512 || currentCount >= 5000)

		// Perform the search
		if useParallelSearch {
			results, err = idx.hnsw.ParallelSearch(query, k*pageSize, idx.config.NumSearchWorkers)
		} else {
			results, err = idx.hnsw.Search(query, k*pageSize)
		}

		// Restore the original efSearch value
		idx.hnsw.EfSearch = originalEf
	}()

	if err != nil {
		idx.logger.Error("HNSW search failed", zap.Error(err))
		return nil, fmt.Errorf("HNSW search failed: %w", err)
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

	// If dimensionality reduction is enabled, validate the configuration
	originalDimension := config.Dimension
	if config.EnableDimReduction {
		if config.DimReductionTarget <= 0 || config.DimReductionTarget >= config.Dimension {
			return nil, fmt.Errorf("invalid target dimension for reduction: %d (must be > 0 and < %d)",
				config.DimReductionTarget, config.Dimension)
		}

		if config.DimReductionMethod == "" {
			config.DimReductionMethod = "PCA" // Default to PCA
		}

		if config.DimReductionAdaptive && (config.DimReductionMinVariance <= 0 || config.DimReductionMinVariance > 1.0) {
			return nil, fmt.Errorf("invalid minimum variance for adaptive reduction: %f (must be > 0 and <= 1.0)",
				config.DimReductionMinVariance)
		}

		// Update the dimension to the target dimension for the HNSW graph
		config.Dimension = config.DimReductionTarget
		logger.Info("Dimensionality reduction enabled",
			zap.Int("original_dimension", originalDimension),
			zap.Int("reduced_dimension", config.Dimension),
			zap.String("method", config.DimReductionMethod))
	}

	// Configure the distance function based on the selected metric
	var distanceFunc hnsw.DistanceFunc
	switch config.Distance {
	case Cosine:
		distanceFunc = hnsw.CosineDistance
	case L2:
		distanceFunc = hnsw.EuclideanDistance
	default:
		distanceFunc = hnsw.CosineDistance // Default to cosine distance
	}

	// Create a new HNSW graph with the specified configuration
	graph, err := hnsw.NewGraphWithConfig[uint64](
		config.HNSWM,
		0.5, // Default Ml value (layer size ratio)
		config.HNSWEfSearch,
		distanceFunc,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create HNSW graph: %w", err)
	}

	// Open DuckDB connection
	duckdb, err := NewDuckDB(WithPath(config.StoragePath))
	if err != nil {
		return nil, fmt.Errorf("failed to open DuckDB: %w", err)
	}

	// Create a connection pool with appropriate sizing
	initialPoolSize := 2                // Start with a smaller pool
	maxPoolSize := runtime.NumCPU() * 2 // Scale with available CPUs

	// Cap the maximum pool size to avoid excessive connections
	if maxPoolSize > 20 {
		maxPoolSize = 20
	}

	connPool, err := NewConnectionPool(duckdb, initialPoolSize, maxPoolSize)
	if err != nil {
		duckdb.Close()
		return nil, fmt.Errorf("failed to create connection pool: %w", err)
	}

	// Load metadata from the database
	metadata := make(map[uint64]map[string]interface{})
	vectors := make(map[uint64][]float32)

	// Get a connection from the pool
	conn, err := connPool.GetConnection()
	if err != nil {
		connPool.Close()
		duckdb.Close()
		return nil, fmt.Errorf("failed to get database connection: %w", err)
	}

	// Query metadata
	rr, stmt, _, err := conn.Query(context.Background(), "SELECT id, json FROM metadata")
	if err != nil {
		connPool.ReleaseConnection(conn)
		connPool.Close()
		duckdb.Close()
		return nil, fmt.Errorf("failed to query metadata: %w", err)
	}
	defer stmt.Close()

	// Process the query results
	for rr.Next() {
		record := rr.Record()
		idCol := record.Column(0).(*array.Uint64)
		jsonCol := record.Column(1).(*array.String)

		for i := 0; i < int(record.NumRows()); i++ {
			id := idCol.Value(i)
			jsonStr := jsonCol.Value(i)

			var meta map[string]interface{}
			if err := sonic.Unmarshal([]byte(jsonStr), &meta); err != nil {
				logger.Warn("failed to unmarshal metadata, storing raw JSON", zap.Uint64("id", id), zap.Error(err))
				meta = map[string]interface{}{"metadata": jsonStr}
			}

			metadata[id] = meta
		}
	}

	// Release the connection
	connPool.ReleaseConnection(conn)

	if err := rr.Err(); err != nil {
		connPool.Close()
		duckdb.Close()
		return nil, fmt.Errorf("error iterating metadata rows: %w", err)
	}

	// Set persist interval.
	persistInterval := 5 * time.Minute
	if config.PersistInterval > 0 {
		persistInterval = config.PersistInterval
	}

	// Set backup interval
	backupInterval := 1 * time.Hour
	if config.BackupInterval > 0 {
		backupInterval = config.BackupInterval
	}

	idx := &Index{
		config:            config,
		hnsw:              graph,
		metadata:          metadata,
		vectors:           vectors,
		duckdb:            duckdb,
		connPool:          connPool,
		lock:              sync.RWMutex{},
		searchLock:        sync.Mutex{}, // Initialize the search lock
		batchBuffer:       make([]vectorMeta, 0, config.BatchSize),
		batchLock:         sync.Mutex{},
		logger:            logger,
		batchTicker:       time.NewTicker(time.Second),
		batchDone:         make(chan struct{}),
		persistInterval:   persistInterval,
		persistTicker:     time.NewTicker(persistInterval),
		persistDone:       make(chan struct{}),
		backupTicker:      time.NewTicker(backupInterval),
		backupDone:        make(chan struct{}),
		originalDimension: originalDimension,
		allocator:         memory.NewGoAllocator(),
		flushSemaphore:    make(chan struct{}, 1),
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

	// Export HNSW graph directly using the Export method
	indexFile := filepath.Join(path, "index.hnsw")
	var indexBuffer bytes.Buffer

	// Use the Export method to write the graph to our buffer
	if err := idx.hnsw.Export(&indexBuffer); err != nil {
		return fmt.Errorf("failed to export HNSW graph: %w", err)
	}

	// Calculate checksum for verification
	indexData := indexBuffer.Bytes()
	checksum := sha256.Sum256(indexData)

	// Handle compression if requested
	if compress {
		indexFile += ".gz"
		var compressedData bytes.Buffer
		gzWriter := gzip.NewWriter(&compressedData)
		if _, err := gzWriter.Write(indexData); err != nil {
			return fmt.Errorf("failed to compress index file: %w", err)
		}
		if err := gzWriter.Close(); err != nil {
			return fmt.Errorf("failed to finalize compression: %w", err)
		}
		indexData = compressedData.Bytes()
	}

	// Write to backup location
	if err := os.WriteFile(indexFile, indexData, 0644); err != nil {
		return fmt.Errorf("failed to write backup index file: %w", err)
	}

	// Export vectors for complete backup
	vectorsFile := filepath.Join(path, "vectors.json")
	if err := idx.exportVectors(vectorsFile, compress); err != nil {
		return fmt.Errorf("failed to export vectors: %w", err)
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
		"index_path":     "index.hnsw",
		"metadata_path":  "metadata.json",
		"vectors_path":   "vectors.json",
	}

	// Create a manifest file with backup metadata
	manifestData, err := sonic.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal manifest: %w", err)
	}

	// Write manifest to both manifest.json (for backward compatibility) and backup.json (for restore)
	manifestFile := filepath.Join(path, "manifest.json")
	if err := os.WriteFile(manifestFile, manifestData, 0644); err != nil {
		return fmt.Errorf("failed to write manifest: %w", err)
	}

	backupFile := filepath.Join(path, "backup.json")
	if err := os.WriteFile(backupFile, manifestData, 0644); err != nil {
		return fmt.Errorf("failed to write backup.json: %w", err)
	}

	idx.logger.Info("backup completed successfully",
		zap.String("path", path),
		zap.Bool("incremental", incremental),
		zap.Bool("compressed", compress),
		zap.String("checksum", hex.EncodeToString(checksum[:])))
	return nil
}

// exportVectors exports the vectors to a JSON file
func (idx *Index) exportVectors(filePath string, compress bool) error {
	// Create a slice to hold all vectors
	type vectorEntry struct {
		ID     uint64    `json:"id"`
		Vector []float32 `json:"vector"`
	}

	vectorsArray := make([]vectorEntry, 0, len(idx.vectors))

	// Add all vectors to the array
	for id, vector := range idx.vectors {
		vectorsArray = append(vectorsArray, vectorEntry{
			ID:     id,
			Vector: vector,
		})
	}

	// Marshal to JSON
	jsonData, err := sonic.MarshalIndent(vectorsArray, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal vectors: %w", err)
	}

	// Handle compression if requested
	if compress {
		filePath += ".gz"
		var compressedData bytes.Buffer
		gzWriter := gzip.NewWriter(&compressedData)
		if _, err := gzWriter.Write(jsonData); err != nil {
			return fmt.Errorf("failed to compress vectors: %w", err)
		}
		if err := gzWriter.Close(); err != nil {
			return fmt.Errorf("failed to finalize compression: %w", err)
		}
		jsonData = compressedData.Bytes()
	}

	// Write to file
	if err := os.WriteFile(filePath, jsonData, 0644); err != nil {
		return fmt.Errorf("failed to write vectors file: %w", err)
	}

	return nil
}

// importVectors imports vectors from a JSON file
func (idx *Index) importVectors(filePath string, compressed bool) error {
	// Read the file
	var jsonData []byte
	var err error

	if compressed {
		// Read compressed file
		file, err := os.Open(filePath)
		if err != nil {
			return fmt.Errorf("failed to open compressed vectors file: %w", err)
		}
		defer file.Close()

		gzReader, err := gzip.NewReader(file)
		if err != nil {
			return fmt.Errorf("failed to create gzip reader: %w", err)
		}
		defer gzReader.Close()

		jsonData, err = io.ReadAll(gzReader)
		if err != nil {
			return fmt.Errorf("failed to read compressed vectors file: %w", err)
		}
	} else {
		// Read uncompressed file
		jsonData, err = os.ReadFile(filePath)
		if err != nil {
			return fmt.Errorf("failed to read vectors file: %w", err)
		}
	}

	// Parse vectors
	type vectorEntry struct {
		ID     uint64    `json:"id"`
		Vector []float32 `json:"vector"`
	}

	var vectorsArray []vectorEntry
	if err := sonic.Unmarshal(jsonData, &vectorsArray); err != nil {
		return fmt.Errorf("failed to unmarshal vectors: %w", err)
	}

	// Clear existing vectors
	idx.vectors = make(map[uint64][]float32, len(vectorsArray))

	// Add vectors to the map
	for _, entry := range vectorsArray {
		// Make a copy of the vector to ensure it's not shared
		vector := make([]float32, len(entry.Vector))
		copy(vector, entry.Vector)
		idx.vectors[entry.ID] = vector
	}

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
		// If backupPath is a directory, copy all files from it
		fileInfo, err := os.Stat(backupPath)
		if err != nil {
			return fmt.Errorf("failed to stat backup path: %w", err)
		}

		if fileInfo.IsDir() {
			// Copy all files from the backup directory to the temp directory
			entries, err := os.ReadDir(backupPath)
			if err != nil {
				return fmt.Errorf("failed to read backup directory: %w", err)
			}

			for _, entry := range entries {
				srcPath := filepath.Join(backupPath, entry.Name())
				dstPath := filepath.Join(tempDir, entry.Name())
				if err := copyFile(srcPath, dstPath); err != nil {
					return fmt.Errorf("failed to copy file %s: %w", entry.Name(), err)
				}
			}
		} else {
			// Just copy the backup file
			if err := copyFile(backupPath, filepath.Join(tempDir, "backup.json")); err != nil {
				return fmt.Errorf("failed to copy backup file: %w", err)
			}
		}
	}

	// Read backup metadata
	metaFile := filepath.Join(tempDir, "backup.json")
	metaData, err := os.ReadFile(metaFile)
	if err != nil {
		return fmt.Errorf("failed to read backup metadata: %w", err)
	}

	var backupInfo map[string]interface{}
	if err := sonic.Unmarshal(metaData, &backupInfo); err != nil {
		return fmt.Errorf("failed to unmarshal backup metadata: %w", err)
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
	vectorsPath, ok := backupInfo["vectors_path"].(string) // Optional for backward compatibility
	if !ok {
		vectorsPath = "vectors.json" // Default path if not specified
	}

	// Check if the index file is compressed
	indexFilePath := filepath.Join(tempDir, indexPath)
	compress, _ := backupInfo["compressed"].(bool)
	if compress {
		indexFilePath += ".gz"
	}

	// Verify the index file exists
	if _, err := os.Stat(indexFilePath); os.IsNotExist(err) {
		return fmt.Errorf("failed to read index file: %w", err)
	}

	// Read the index file
	var indexData []byte
	if strings.HasSuffix(indexFilePath, ".gz") {
		// Decompress the index file
		file, err := os.Open(indexFilePath)
		if err != nil {
			return fmt.Errorf("failed to open compressed index file: %w", err)
		}
		defer file.Close()

		gzReader, err := gzip.NewReader(file)
		if err != nil {
			return fmt.Errorf("failed to create gzip reader: %w", err)
		}
		defer gzReader.Close()

		indexData, err = io.ReadAll(gzReader)
		if err != nil {
			return fmt.Errorf("failed to read compressed index file: %w", err)
		}
	} else {
		// Read uncompressed index file
		indexData, err = os.ReadFile(indexFilePath)
		if err != nil {
			return fmt.Errorf("failed to read index file: %w", err)
		}
	}

	// Create a new graph
	newGraph := hnsw.NewGraph[uint64]()

	// Import the graph data using the Import method
	if err := newGraph.Import(bytes.NewReader(indexData)); err != nil {
		return fmt.Errorf("failed to import HNSW graph: %w", err)
	}

	// Update the index with the restored HNSW
	idx.hnsw = newGraph

	// Import metadata
	metadataFilePath := filepath.Join(tempDir, metadataPath)
	if compress && !strings.HasSuffix(metadataFilePath, ".gz") {
		metadataFilePath += ".gz"
	}

	// Verify the metadata file exists
	if _, err := os.Stat(metadataFilePath); os.IsNotExist(err) {
		return fmt.Errorf("failed to read metadata file: %w", err)
	}

	if err := idx.importMetadata(metadataFilePath, strings.HasSuffix(metadataFilePath, ".gz")); err != nil {
		return fmt.Errorf("failed to import metadata: %w", err)
	}

	// Import vectors if available
	vectorsFilePath := filepath.Join(tempDir, vectorsPath)
	if compress && !strings.HasSuffix(vectorsFilePath, ".gz") {
		vectorsFilePath += ".gz"
	}

	// Verify the vectors file exists
	if _, err := os.Stat(vectorsFilePath); os.IsNotExist(err) {
		idx.logger.Warn("vectors file not found, creating empty vectors", zap.String("path", vectorsFilePath))
		idx.vectors = make(map[uint64][]float32)
		for id := range idx.metadata {
			idx.vectors[id] = make([]float32, idx.config.Dimension)
		}
	} else {
		if err := idx.importVectors(vectorsFilePath, strings.HasSuffix(vectorsFilePath, ".gz")); err != nil {
			// If vectors file is not available, create empty vectors
			idx.logger.Warn("failed to import vectors, creating empty vectors", zap.Error(err))
			idx.vectors = make(map[uint64][]float32)
			for id := range idx.metadata {
				idx.vectors[id] = make([]float32, idx.config.Dimension)
			}
		}
	}

	idx.logger.Info("index restored successfully",
		zap.String("backup_path", backupPath),
		zap.String("storage_path", idx.config.StoragePath),
		zap.Int("vector_count", idx.hnsw.Len()),
		zap.Int("metadata_count", len(idx.metadata)),
		zap.Int("vectors_count", len(idx.vectors)))
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

	// Marshal metadata to JSON
	jsonData, err := sonic.MarshalIndent(metadataArray, "", "  ")
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
	// Read and decrypt file
	fileData, err := idx.readEncrypted(filePath)
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
	if err := sonic.Unmarshal(jsonData, &metadataArray); err != nil {
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

		// Convert float64 values back to int when appropriate
		for k, v := range item {
			if floatVal, ok := v.(float64); ok && floatVal == float64(int(floatVal)) {
				item[k] = int(floatVal)
			}
		}

		idx.metadata[uint64(id)] = item
	}

	return nil
}

// Close releases resources associated with the index and stops background workers.
func (idx *Index) Close() error {
	// Stop background workers
	if idx.batchTicker != nil {
		idx.batchTicker.Stop()
		close(idx.batchDone)
	}
	if idx.persistTicker != nil {
		idx.persistTicker.Stop()
		close(idx.persistDone)
	}
	if idx.backupTicker != nil {
		idx.backupTicker.Stop()
		close(idx.backupDone)
	}

	// Wait for background workers to exit
	idx.bgWG.Wait()

	// Flush any remaining batch items
	if len(idx.batchBuffer) > 0 {
		if err := idx.flushBatch(); err != nil {
			idx.logger.Error("failed to flush batch during close", zap.Error(err))
		}
	}

	// Close DuckDB connection pool and database
	if idx.connPool != nil {
		idx.connPool.Close()
	}
	if idx.duckdb != nil {
		idx.duckdb.Close()
	}

	return nil
}

// QueryMetadata executes a metadata query against DuckDB and returns the matching metadata.
func (idx *Index) QueryMetadata(query string) ([]map[string]interface{}, error) {
	// Only cache SELECT queries, not DELETE or other modifying queries
	shouldCache := strings.HasPrefix(strings.ToUpper(strings.TrimSpace(query)), "SELECT")

	// Create a cache key for the query if it's cacheable
	var cacheKey string
	if shouldCache {
		cacheKey = "query:" + query

		// Check if we have a cached result
		if cachedResult, ok := idx.cache.Load(cacheKey); ok {
			// Return a copy of the cached result to avoid modification
			originalResults := cachedResult.([]map[string]interface{})
			results := make([]map[string]interface{}, len(originalResults))

			// Deep copy each result map
			for i, original := range originalResults {
				results[i] = make(map[string]interface{}, len(original))
				for k, v := range original {
					results[i][k] = v
				}
			}

			return results, nil
		}
	}

	// Try to get a thread-local connection for better performance
	conn, err := idx.connPool.GetConnection()
	if err != nil {
		return nil, fmt.Errorf("failed to get database connection: %w", err)
	}
	defer idx.connPool.ReleaseConnection(conn)

	// Create a context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Execute the query
	rr, stmt, _, err := conn.Query(ctx, query)
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

		// Pre-allocate results array to avoid reallocations
		if results == nil {
			results = make([]map[string]interface{}, 0, numRows)
		}

		// Get column names from schema
		schema := record.Schema()
		colNames := make([]string, numCols)
		for i := 0; i < numCols; i++ {
			colNames[i] = schema.Field(i).Name
		}

		// Process each row
		for rowIdx := 0; rowIdx < numRows; rowIdx++ {
			result := make(map[string]interface{}, numCols) // Pre-allocate map with expected capacity

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

	// Cache the results for future queries if the result set is not too large and it's a SELECT query
	if shouldCache && len(results) > 0 && len(results) < 100 {
		// Create a deep copy for caching to avoid modification
		cachedResults := make([]map[string]interface{}, len(results))
		for i, result := range results {
			cachedResults[i] = make(map[string]interface{}, len(result))
			for k, v := range result {
				cachedResults[i][k] = v
			}
		}

		// Store in cache with a TTL (will be automatically cleaned up by GC)
		idx.cache.Store(cacheKey, cachedResults)

		// Schedule cache cleanup after 5 minutes
		go func(key string) {
			time.Sleep(5 * time.Minute)
			idx.cache.Delete(key)
		}(cacheKey)
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
	if idx.hnsw == nil || idx.connPool == nil {
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
	metrics["db_connections"] = idx.connPool.Size()
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
	if len(queries) == 0 {
		return nil, errors.New("no query vectors provided")
	}

	// Validate dimensions for all queries
	for i, query := range queries {
		if !idx.config.EnableDimReduction || idx.dimReducer == nil {
			if len(query) != idx.config.Dimension {
				return nil, fmt.Errorf("query %d dimension (%d) does not match index dimension (%d)",
					i, len(query), idx.config.Dimension)
			}
		} else {
			if len(query) != idx.originalDimension {
				return nil, fmt.Errorf("query %d dimension (%d) does not match original dimension (%d)",
					i, len(query), idx.originalDimension)
			}
		}
	}

	// Always use sequential processing to avoid concurrent map writes
	return idx.sequentialMultiVectorSearch(queries, k)
}

// sequentialMultiVectorSearch processes multiple query vectors sequentially.
func (idx *Index) sequentialMultiVectorSearch(queries [][]float32, k int) ([][]SearchResult, error) {
	results := make([][]SearchResult, len(queries))

	for i, query := range queries {
		searchResults, err := idx.Search(query, k, 1, k)
		if err != nil {
			return nil, fmt.Errorf("error searching for query %d: %w", i, err)
		}
		results[i] = searchResults
	}

	return results, nil
}

// copyFile copies a file from src to dst.
func copyFile(src, dst string) error {
	sourceFile, err := os.Open(src)
	if err != nil {
		return err
	}
	defer sourceFile.Close()

	destFile, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer destFile.Close()

	_, err = io.Copy(destFile, sourceFile)
	if err != nil {
		return err
	}

	return destFile.Sync()
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

		// Set efSearch parameter for better recall
		efSearch := idx.config.HNSWEfSearch * 2
		if efSearch < searchK*4 {
			efSearch = searchK * 4
		}

		// Store the original value so we can restore it after the search
		originalEf := idx.hnsw.EfSearch
		idx.hnsw.EfSearch = efSearch
		idx.logger.Debug("Setting efSearch parameter for negative search", zap.Int("efSearch", efSearch))

		// Perform the search
		results, err = idx.hnsw.Search(positiveQuery, searchK)
		if err != nil {
			idx.logger.Error("HNSW search failed", zap.Error(err))
			return
		}

		// Restore the original efSearch value
		idx.hnsw.EfSearch = originalEf
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
	// Just delegate to BatchAppendFromArrow for consistency
	return idx.BatchAppendFromArrow([]arrow.Record{rec})
}

// BatchAppendFromArrow efficiently appends vectors and metadata from multiple Arrow records.
func (idx *Index) BatchAppendFromArrow(records []arrow.Record) error {
	if len(records) == 0 {
		return nil
	}

	// Calculate total number of rows across all records
	totalRows := 0
	for _, rec := range records {
		totalRows += int(rec.NumRows())
	}

	// Prepare all vectors and metadata before acquiring the lock
	allBatchItems := make([]vectorMeta, 0, totalRows)

	// Process each record
	for _, rec := range records {
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

		// Process all vectors in this record
		for i := 0; i < numRows; i++ {
			if idCol.IsNull(i) {
				return fmt.Errorf("id column contains null value at row %d", i)
			}
			id := idCol.Value(i)

			if vectorCol.IsNull(i) {
				return fmt.Errorf("vector column contains null value at row %d", i)
			}

			// Extract vector
			start := i * dim
			vector := make([]float32, dim)
			for j := 0; j < dim; j++ {
				vector[j] = valuesArr.Value(start + j)
			}

			// Process metadata
			var meta map[string]interface{}
			if !metadataCol.IsNull(i) {
				metaJSON := metadataCol.Value(i)
				meta = make(map[string]interface{})

				err := sonic.Unmarshal([]byte(metaJSON), &meta)
				if err != nil {
					idx.logger.Warn("failed to unmarshal metadata, storing raw JSON", zap.Uint64("id", id), zap.Error(err))
					meta = map[string]interface{}{"metadata": metaJSON}
				}

				// Validate metadata schema
				if err := validateMetadataSchema(meta); err != nil {
					return err
				}
			} else {
				meta = make(map[string]interface{})
			}

			// Add to our batch items
			allBatchItems = append(allBatchItems, vectorMeta{
				id:     id,
				vector: vector,
				meta:   meta,
			})
		}
	}

	// If we have a large number of items, process them in smaller batches
	// to avoid holding the lock for too long
	batchSize := idx.config.BatchSize
	if batchSize <= 0 {
		batchSize = 100 // Default batch size
	}

	// For benchmarks and tests, we'll directly add items to the index
	// This avoids potential deadlocks with background workers
	if len(allBatchItems) > 0 {
		// Lock the index for updating shared metadata and cache
		idx.lock.Lock()

		// Check if HNSW is initialized
		if idx.hnsw == nil {
			idx.lock.Unlock()
			return errors.New("HNSW graph is not initialized")
		}

		// Process sequentially to avoid data races in HNSW
		for _, item := range allBatchItems {
			// Add to HNSW graph
			node := hnsw.MakeNode(item.id, item.vector)
			if err := idx.hnsw.Add(node); err != nil {
				// Log the error but continue with other operations
				idx.logger.Warn("Failed to add node to HNSW graph during batch append",
					zap.Uint64("id", item.id),
					zap.Error(err))
			}

			// Update in-memory metadata and cache
			idx.metadata[item.id] = item.meta
			idx.cache.Store(item.id, item.meta)

			// Store vector for negative example searches
			idx.vectors[item.id] = item.vector
		}

		// If we have a database connection, store metadata there too
		if idx.connPool != nil {
			// Use the index's allocator if available, otherwise create a new one
			var pool memory.Allocator
			if idx.allocator != nil {
				pool = idx.allocator
			} else {
				pool = memory.NewGoAllocator()
			}

			// Create schema for metadata table
			schema := arrow.NewSchema(
				[]arrow.Field{
					{Name: "id", Type: arrow.PrimitiveTypes.Uint64, Nullable: false},
					{Name: "json", Type: arrow.BinaryTypes.String, Nullable: true},
				},
				nil,
			)

			// Process in batches to avoid overwhelming the database
			for i := 0; i < len(allBatchItems); i += batchSize {
				end := i + batchSize
				if end > len(allBatchItems) {
					end = len(allBatchItems)
				}

				// Create record builder
				builder := array.NewRecordBuilder(pool, schema)

				// Get builders for each column
				idBuilder := builder.Field(0).(*array.Uint64Builder)
				jsonBuilder := builder.Field(1).(*array.StringBuilder)

				// Pre-allocate capacity for better performance
				idBuilder.Reserve(end - i)
				jsonBuilder.Reserve(end - i)

				// Add data to builders
				for _, item := range allBatchItems[i:end] {
					// Convert metadata to JSON
					metaJSON, err := sonic.Marshal(item.meta)
					if err != nil {
						idx.logger.Error("failed to marshal metadata", zap.Error(err), zap.Uint64("id", item.id))
						metaJSON = []byte("{}")
					}

					// Append values
					idBuilder.Append(item.id)
					jsonBuilder.Append(string(metaJSON))
				}

				// Build the record
				record := builder.NewRecord()

				// Get a connection from the pool
				conn, err := idx.connPool.GetConnection()
				if err != nil {
					return fmt.Errorf("failed to get database connection: %w", err)
				}
				defer idx.connPool.ReleaseConnection(conn)

				// Insert using ADBC
				_, err = conn.IngestCreateAppend(context.Background(), "metadata", record)

				// Clean up
				record.Release()
				builder.Release()

				if err != nil {
					idx.lock.Unlock()
					return fmt.Errorf("failed to insert metadata: %w", err)
				}
			}
		}

		idx.lock.Unlock()
	}

	return nil
}

// DeleteVector removes a vector from the index.
func (idx *Index) DeleteVector(id uint64) error {
	idx.lock.Lock()
	defer idx.lock.Unlock()

	// Check if the vector exists
	if _, exists := idx.vectors[id]; !exists {
		return fmt.Errorf("vector with id %d not found", id)
	}

	// Remove from HNSW graph
	idx.hnsw.Delete(id)

	// Remove from in-memory maps
	delete(idx.vectors, id)
	delete(idx.metadata, id)
	idx.cache.Delete(id)

	// Remove from database if connected
	if idx.connPool != nil {
		conn, err := idx.connPool.GetConnection()
		if err != nil {
			return fmt.Errorf("failed to get database connection: %w", err)
		}
		defer idx.connPool.ReleaseConnection(conn)

		_, err = conn.Exec(context.Background(), fmt.Sprintf("DELETE FROM metadata WHERE id = %d", id))
		if err != nil {
			return fmt.Errorf("failed to delete from database: %w", err)
		}

		// Invalidate query cache after modifying the database
		idx.invalidateQueryCache()
	}

	return nil
}

// DeleteVectors deletes multiple vectors from the index.
func (idx *Index) DeleteVectors(ids []uint64) error {
	if len(ids) == 0 {
		return nil
	}

	idx.lock.Lock()
	defer idx.lock.Unlock()

	// Check if all vectors exist
	for _, id := range ids {
		if _, exists := idx.vectors[id]; !exists {
			return fmt.Errorf("vector with id %d not found", id)
		}
	}

	// Remove from all relevant maps and HNSW graph
	for _, id := range ids {
		delete(idx.vectors, id)
		delete(idx.metadata, id)
		idx.cache.Delete(id)
		idx.hnsw.Delete(id)
	}

	// Remove from database if connected
	if idx.connPool != nil {
		conn, err := idx.connPool.GetConnection()
		if err != nil {
			return fmt.Errorf("failed to get database connection: %w", err)
		}
		defer idx.connPool.ReleaseConnection(conn)

		// Create a comma-separated list of IDs
		var idList strings.Builder
		for i, id := range ids {
			if i > 0 {
				idList.WriteString(", ")
			}
			idList.WriteString(fmt.Sprintf("%d", id))
		}

		// Execute the DELETE query
		_, err = conn.Exec(context.Background(), fmt.Sprintf("DELETE FROM metadata WHERE id IN (%s)", idList.String()))
		if err != nil {
			return fmt.Errorf("failed to delete from database: %w", err)
		}

		// Invalidate query cache after modifying the database
		idx.invalidateQueryCache()
	}

	return nil
}

// Config returns a copy of the index configuration
func (idx *Index) Config() Config {
	idx.lock.RLock()
	defer idx.lock.RUnlock()
	return idx.config
}

// AnalyzeGraph returns quality metrics for the HNSW graph.
// This is useful for understanding the health and performance characteristics of the index.
func (idx *Index) AnalyzeGraph() (map[string]interface{}, error) {
	idx.lock.RLock()
	defer idx.lock.RUnlock()

	// Create an analyzer for the graph
	analyzer := &hnsw.Analyzer[uint64]{Graph: idx.hnsw}

	// Get quality metrics
	metrics := analyzer.QualityMetrics()

	// Get connectivity information
	connectivity := analyzer.Connectivity()

	// Get topography information
	topography := analyzer.Topography()

	// Create a map of all metrics
	result := map[string]interface{}{
		"node_count":          metrics.NodeCount,
		"avg_connectivity":    metrics.AvgConnectivity,
		"connectivity_stddev": metrics.ConnectivityStdDev,
		"distortion_ratio":    metrics.DistortionRatio,
		"layer_balance":       metrics.LayerBalance,
		"graph_height":        metrics.GraphHeight,
		"layer_connectivity":  connectivity,
		"layer_topography":    topography,
	}

	return result, nil
}

// GetGraphAnalyzer returns the HNSW graph analyzer for advanced analysis.
// This is intended for advanced users who need direct access to the analyzer.
// Note: This method does not acquire a lock, so the caller must ensure thread safety.
func (idx *Index) GetGraphAnalyzer() *hnsw.Analyzer[uint64] {
	return &hnsw.Analyzer[uint64]{Graph: idx.hnsw}
}

// PreparedStatement represents a cached prepared statement
type PreparedStatement struct {
	query      string
	stmt       interface{} // Generic statement type to avoid dependency issues
	conn       *DuckDBConn
	lastUsed   time.Time
	usageCount int
}

// cleanupPreparedStatements periodically cleans up unused prepared statements
func (p *ConnectionPool) cleanupPreparedStatements() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			p.stmtMu.Lock()
			now := time.Now()
			for key, stmt := range p.preparedStmts {
				// Remove statements that haven't been used in the last 30 minutes and have low usage
				if now.Sub(stmt.lastUsed) > 30*time.Minute && stmt.usageCount < 10 {
					stmt.stmt = nil // Just clear the statement to avoid holding references
					delete(p.preparedStmts, key)
				}
			}
			p.stmtMu.Unlock()
		case <-p.stopCleanup:
			return // Exit the goroutine when signaled
		}
	}
}

// invalidateQueryCache invalidates all query cache entries
func (idx *Index) invalidateQueryCache() {
	// Iterate through the cache and remove all query cache entries
	idx.cache.Range(func(key, value interface{}) bool {
		keyStr, ok := key.(string)
		if ok && strings.HasPrefix(keyStr, "query:") {
			idx.cache.Delete(key)
		}
		return true
	})
}
