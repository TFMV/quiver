package db

import (
	"cmp"
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/TFMV/hnsw"
	"github.com/TFMV/hnsw/hnsw-extensions/facets"
	"github.com/TFMV/hnsw/hnsw-extensions/meta"
	"github.com/TFMV/hnsw/hnsw-extensions/parquet"
	"github.com/bytedance/sonic"
	_ "github.com/marcboeker/go-duckdb/v2"
)

// PersistentStorage provides persistent storage for the vector database
type PersistentStorage[K cmp.Ordered] struct {
	// Base directory for storage
	baseDir string

	// Parquet storage for vectors and graph structure
	parquetStorage *parquet.ParquetStorage[K]

	// Metadata storage
	metadataFile string

	// Facets storage
	facetsFile string

	// DuckDB storage
	duckDBStorage *DuckDBStorage[K]

	// Mutex for thread safety
	mu sync.RWMutex
}

// Storage type constants
const (
	StorageTypeParquet = "parquet"
	StorageTypeDuckDB  = "duckdb"
)

// Link represents a connection between nodes in the HNSW graph
type Link[K cmp.Ordered] struct {
	To       K       // Target node ID
	Distance float32 // Distance to target
}

// DuckDBStorage provides vector storage using DuckDB
type DuckDBStorage[K cmp.Ordered] struct {
	// DuckDB connection
	db *sql.DB

	// Configuration
	config DuckDBStorageConfig

	// Table names
	vectorTable    string
	metadataTable  string
	facetsTable    string
	graphInfoTable string
	linksTable     string

	// Mutex for thread safety
	mu sync.RWMutex

	// Context for query execution
	ctx context.Context
}

// DuckDBStorageConfig contains configuration for DuckDB storage
type DuckDBStorageConfig struct {
	// Connection string (file path or :memory:)
	ConnectionString string `json:"connection_string"`

	// Table prefix for DuckDB tables
	TablePrefix string `json:"table_prefix"`

	// Schema name
	Schema string `json:"schema"`

	// MotherDuck token for cloud access
	MotherDuckToken string `json:"motherduck_token"`

	// Connection options
	ReadOnly           bool          `json:"read_only"`
	AllowUnsafe        bool          `json:"allow_unsafe"`
	MemoryLimit        string        `json:"memory_limit"`
	AutoLoadExtensions bool          `json:"auto_load_extensions"`
	QueryTimeout       time.Duration `json:"query_timeout"`
}

// DefaultDuckDBStorageConfig returns default configuration for DuckDB storage
func DefaultDuckDBStorageConfig() DuckDBStorageConfig {
	return DuckDBStorageConfig{
		ConnectionString:   ":memory:",
		TablePrefix:        "quiver_",
		Schema:             "main",
		ReadOnly:           false,
		AllowUnsafe:        true,
		MemoryLimit:        "4GB",
		AutoLoadExtensions: true,
		QueryTimeout:       30 * time.Second,
	}
}

// NewPersistentStorage creates a new persistent storage instance
func NewPersistentStorage[K cmp.Ordered](baseDir string) (*PersistentStorage[K], error) {
	if baseDir == "" {
		baseDir = "vectordb_data"
	}

	// Create base directory if it doesn't exist
	if err := os.MkdirAll(baseDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create base directory: %w", err)
	}

	// Create Parquet storage
	parquetConfig := parquet.DefaultParquetStorageConfig()
	parquetConfig.Directory = filepath.Join(baseDir, "vectors")
	parquetStorage, err := parquet.NewParquetStorage[K](parquetConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create Parquet storage: %w", err)
	}

	storage := &PersistentStorage[K]{
		baseDir:        baseDir,
		parquetStorage: parquetStorage,
		metadataFile:   filepath.Join(baseDir, "metadata.json"),
		facetsFile:     filepath.Join(baseDir, "facets.json"),
	}

	return storage, nil
}

// NewPersistentStorageWithDuckDB creates a new persistent storage instance with DuckDB
func NewPersistentStorageWithDuckDB[K cmp.Ordered](baseDir string, duckDBConfig DuckDBStorageConfig) (*PersistentStorage[K], error) {
	if baseDir == "" {
		baseDir = "vectordb_data"
	}

	// Create base directory if it doesn't exist
	if err := os.MkdirAll(baseDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create base directory: %w", err)
	}

	// If connection string is relative path, make it absolute from baseDir
	if !strings.HasPrefix(duckDBConfig.ConnectionString, ":") &&
		!strings.HasPrefix(duckDBConfig.ConnectionString, "/") {
		duckDBConfig.ConnectionString = filepath.Join(baseDir, duckDBConfig.ConnectionString)
	}

	// Create DuckDB storage
	duckDBStorage, err := NewDuckDBStorage[K](duckDBConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create DuckDB storage: %w", err)
	}

	storage := &PersistentStorage[K]{
		baseDir:       baseDir,
		metadataFile:  filepath.Join(baseDir, "metadata.json"),
		facetsFile:    filepath.Join(baseDir, "facets.json"),
		duckDBStorage: duckDBStorage,
	}

	return storage, nil
}

// NewDuckDBStorage creates a new DuckDB storage instance
func NewDuckDBStorage[K cmp.Ordered](config DuckDBStorageConfig) (*DuckDBStorage[K], error) {
	// Construct DSN
	dsnParams := []string{}
	if config.ReadOnly {
		dsnParams = append(dsnParams, "access_mode=read_only")
	}
	if config.AllowUnsafe {
		dsnParams = append(dsnParams, "allow_unsigned_extensions=true")
	}
	if config.MemoryLimit != "" {
		dsnParams = append(dsnParams, fmt.Sprintf("memory_limit=%s", config.MemoryLimit))
	}
	if config.AutoLoadExtensions {
		dsnParams = append(dsnParams, "auto_load_extension=true")
	}

	// Add MotherDuck token if provided
	if config.MotherDuckToken != "" && strings.HasPrefix(config.ConnectionString, "md:") {
		dsnParams = append(dsnParams, fmt.Sprintf("motherduck_token=%s", config.MotherDuckToken))
	}

	dsn := config.ConnectionString
	if len(dsnParams) > 0 {
		dsn += "?" + strings.Join(dsnParams, "&")
	}

	// Open DuckDB connection
	db, err := sql.Open("duckdb", dsn)
	if err != nil {
		return nil, fmt.Errorf("failed to open DuckDB connection: %w", err)
	}

	// Set connection properties
	db.SetMaxOpenConns(8)
	db.SetMaxIdleConns(8)
	db.SetConnMaxLifetime(time.Hour)

	// Create context with cancellation for queries
	ctx := context.Background()

	// Create tables if they don't exist
	storage := &DuckDBStorage[K]{
		db:             db,
		config:         config,
		vectorTable:    config.TablePrefix + "vectors",
		metadataTable:  config.TablePrefix + "metadata",
		facetsTable:    config.TablePrefix + "facets",
		graphInfoTable: config.TablePrefix + "graph_info",
		linksTable:     config.TablePrefix + "links",
		ctx:            ctx,
	}

	// Initialize tables
	if err := storage.initTables(); err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to initialize tables: %w", err)
	}

	return storage, nil
}

// initTables creates the necessary tables if they don't exist
func (ds *DuckDBStorage[K]) initTables() error {
	ds.mu.Lock()
	defer ds.mu.Unlock()

	// Create a timeout context for initialization
	ctx, cancel := context.WithTimeout(ds.ctx, ds.config.QueryTimeout)
	defer cancel()

	// Create vectors table
	createVectorsSQL := fmt.Sprintf(`
		CREATE TABLE IF NOT EXISTS %s.%s (
			id VARCHAR PRIMARY KEY,
			vector JSON NOT NULL,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)
	`, ds.config.Schema, ds.vectorTable)

	if _, err := ds.db.ExecContext(ctx, createVectorsSQL); err != nil {
		return fmt.Errorf("failed to create vectors table: %w", err)
	}

	// Create metadata table
	createMetadataSQL := fmt.Sprintf(`
		CREATE TABLE IF NOT EXISTS %s.%s (
			id VARCHAR PRIMARY KEY,
			metadata JSON NOT NULL,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)
	`, ds.config.Schema, ds.metadataTable)

	if _, err := ds.db.ExecContext(ctx, createMetadataSQL); err != nil {
		return fmt.Errorf("failed to create metadata table: %w", err)
	}

	// Create facets table
	createFacetsSQL := fmt.Sprintf(`
		CREATE TABLE IF NOT EXISTS %s.%s (
			id VARCHAR PRIMARY KEY,
			facets JSON NOT NULL,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)
	`, ds.config.Schema, ds.facetsTable)

	if _, err := ds.db.ExecContext(ctx, createFacetsSQL); err != nil {
		return fmt.Errorf("failed to create facets table: %w", err)
	}

	// Create graph info table
	createGraphInfoSQL := fmt.Sprintf(`
		CREATE TABLE IF NOT EXISTS %s.%s (
			key VARCHAR PRIMARY KEY,
			value JSON NOT NULL,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)
	`, ds.config.Schema, ds.graphInfoTable)

	if _, err := ds.db.ExecContext(ctx, createGraphInfoSQL); err != nil {
		return fmt.Errorf("failed to create graph info table: %w", err)
	}

	// Create links table
	createLinksSQL := fmt.Sprintf(`
		CREATE TABLE IF NOT EXISTS %s.%s (
			id VARCHAR NOT NULL,
			level INTEGER NOT NULL,
			link_to VARCHAR NOT NULL,
			distance REAL NOT NULL,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			PRIMARY KEY (id, level, link_to)
		)
	`, ds.config.Schema, ds.linksTable)

	if _, err := ds.db.ExecContext(ctx, createLinksSQL); err != nil {
		return fmt.Errorf("failed to create links table: %w", err)
	}

	return nil
}

// SaveVector saves a vector to DuckDB
func (ds *DuckDBStorage[K]) SaveVector(key K, vector []float32) error {
	ds.mu.Lock()
	defer ds.mu.Unlock()

	// Convert key to string for storage
	keyStr := fmt.Sprintf("%v", key)

	// Serialize vector to JSON
	vectorJSON, err := json.Marshal(vector)
	if err != nil {
		return fmt.Errorf("failed to marshal vector: %w", err)
	}

	// Create a timeout context
	ctx, cancel := context.WithTimeout(ds.ctx, ds.config.QueryTimeout)
	defer cancel()

	// Upsert vector
	upsertSQL := fmt.Sprintf(`
		INSERT INTO %s.%s (id, vector, updated_at)
		VALUES (?, ?, CURRENT_TIMESTAMP)
		ON CONFLICT (id) 
		DO UPDATE SET 
			vector = EXCLUDED.vector,
			updated_at = CURRENT_TIMESTAMP
	`, ds.config.Schema, ds.vectorTable)

	if _, err := ds.db.ExecContext(ctx, upsertSQL, keyStr, string(vectorJSON)); err != nil {
		return fmt.Errorf("failed to save vector: %w", err)
	}

	return nil
}

// BatchSaveVectors saves multiple vectors to DuckDB
func (ds *DuckDBStorage[K]) BatchSaveVectors(keys []K, vectors [][]float32) error {
	if len(keys) != len(vectors) {
		return errors.New("keys and vectors lengths don't match")
	}

	ds.mu.Lock()
	defer ds.mu.Unlock()

	// Create a timeout context
	ctx, cancel := context.WithTimeout(ds.ctx, ds.config.QueryTimeout)
	defer cancel()

	// Start a transaction
	tx, err := ds.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to start transaction: %w", err)
	}

	// Prepare the insert statement
	upsertSQL := fmt.Sprintf(`
		INSERT INTO %s.%s (id, vector, updated_at)
		VALUES (?, ?, CURRENT_TIMESTAMP)
		ON CONFLICT (id) 
		DO UPDATE SET 
			vector = EXCLUDED.vector,
			updated_at = CURRENT_TIMESTAMP
	`, ds.config.Schema, ds.vectorTable)

	stmt, err := tx.PrepareContext(ctx, upsertSQL)
	if err != nil {
		tx.Rollback()
		return fmt.Errorf("failed to prepare statement: %w", err)
	}
	defer stmt.Close()

	// Insert vectors
	for i, key := range keys {
		keyStr := fmt.Sprintf("%v", key)
		vectorJSON, err := json.Marshal(vectors[i])
		if err != nil {
			tx.Rollback()
			return fmt.Errorf("failed to marshal vector: %w", err)
		}

		if _, err := stmt.ExecContext(ctx, keyStr, string(vectorJSON)); err != nil {
			tx.Rollback()
			return fmt.Errorf("failed to save vector: %w", err)
		}
	}

	// Commit the transaction
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}

	return nil
}

// GetVector retrieves a vector from DuckDB
func (ds *DuckDBStorage[K]) GetVector(key K) ([]float32, error) {
	ds.mu.RLock()
	defer ds.mu.RUnlock()

	// Convert key to string
	keyStr := fmt.Sprintf("%v", key)

	// Create a timeout context
	ctx, cancel := context.WithTimeout(ds.ctx, ds.config.QueryTimeout)
	defer cancel()

	// Query the vector
	querySQL := fmt.Sprintf(`
		SELECT vector FROM %s.%s WHERE id = ?
	`, ds.config.Schema, ds.vectorTable)

	var vectorJSON string
	err := ds.db.QueryRowContext(ctx, querySQL, keyStr).Scan(&vectorJSON)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, fmt.Errorf("vector not found for key %v", key)
		}
		return nil, fmt.Errorf("failed to query vector: %w", err)
	}

	// Deserialize the vector
	var vector []float32
	if err := json.Unmarshal([]byte(vectorJSON), &vector); err != nil {
		return nil, fmt.Errorf("failed to unmarshal vector: %w", err)
	}

	return vector, nil
}

// GetVectors retrieves multiple vectors from DuckDB
func (ds *DuckDBStorage[K]) GetVectors(keys []K) (map[K][]float32, error) {
	ds.mu.RLock()
	defer ds.mu.RUnlock()

	result := make(map[K][]float32)
	if len(keys) == 0 {
		return result, nil
	}

	// Convert keys to string array for IN clause
	keyStrs := make([]string, len(keys))
	keyMap := make(map[string]K)
	for i, key := range keys {
		keyStr := fmt.Sprintf("%v", key)
		keyStrs[i] = keyStr
		keyMap[keyStr] = key
	}

	// Build placeholder string for IN clause
	placeholders := strings.Repeat("?,", len(keyStrs))
	placeholders = placeholders[:len(placeholders)-1]

	// Create a timeout context
	ctx, cancel := context.WithTimeout(ds.ctx, ds.config.QueryTimeout)
	defer cancel()

	// Query the vectors
	querySQL := fmt.Sprintf(`
		SELECT id, vector FROM %s.%s WHERE id IN (%s)
	`, ds.config.Schema, ds.vectorTable, placeholders)

	// Create args for the query
	args := make([]interface{}, len(keyStrs))
	for i, keyStr := range keyStrs {
		args[i] = keyStr
	}

	// Execute the query
	rows, err := ds.db.QueryContext(ctx, querySQL, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to query vectors: %w", err)
	}
	defer rows.Close()

	// Process the results
	for rows.Next() {
		var keyStr string
		var vectorJSON string
		if err := rows.Scan(&keyStr, &vectorJSON); err != nil {
			return nil, fmt.Errorf("failed to scan row: %w", err)
		}

		var vector []float32
		if err := json.Unmarshal([]byte(vectorJSON), &vector); err != nil {
			return nil, fmt.Errorf("failed to unmarshal vector: %w", err)
		}

		// Look up the original key
		originalKey, ok := keyMap[keyStr]
		if !ok {
			continue
		}

		result[originalKey] = vector
	}

	// Check for errors from iterating
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error during iteration: %w", err)
	}

	return result, nil
}

// SaveGraphInfo saves graph information to DuckDB
func (ds *DuckDBStorage[K]) SaveGraphInfo(key string, value interface{}) error {
	ds.mu.Lock()
	defer ds.mu.Unlock()

	// Serialize value to JSON
	valueJSON, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("failed to marshal value: %w", err)
	}

	// Create a timeout context
	ctx, cancel := context.WithTimeout(ds.ctx, ds.config.QueryTimeout)
	defer cancel()

	// Upsert graph info
	upsertSQL := fmt.Sprintf(`
		INSERT INTO %s.%s (key, value, updated_at)
		VALUES (?, ?, CURRENT_TIMESTAMP)
		ON CONFLICT (key) 
		DO UPDATE SET 
			value = EXCLUDED.value,
			updated_at = CURRENT_TIMESTAMP
	`, ds.config.Schema, ds.graphInfoTable)

	if _, err := ds.db.ExecContext(ctx, upsertSQL, key, string(valueJSON)); err != nil {
		return fmt.Errorf("failed to save graph info: %w", err)
	}

	return nil
}

// GetGraphInfo retrieves graph information from DuckDB
func (ds *DuckDBStorage[K]) GetGraphInfo(key string) (json.RawMessage, error) {
	ds.mu.RLock()
	defer ds.mu.RUnlock()

	// Create a timeout context
	ctx, cancel := context.WithTimeout(ds.ctx, ds.config.QueryTimeout)
	defer cancel()

	// Query the graph info
	querySQL := fmt.Sprintf(`
		SELECT value FROM %s.%s WHERE key = ?
	`, ds.config.Schema, ds.graphInfoTable)

	var valueJSON string
	err := ds.db.QueryRowContext(ctx, querySQL, key).Scan(&valueJSON)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, fmt.Errorf("graph info not found for key %s", key)
		}
		return nil, fmt.Errorf("failed to query graph info: %w", err)
	}

	return json.RawMessage(valueJSON), nil
}

// SaveLink saves a link between nodes
func (ds *DuckDBStorage[K]) SaveLink(from K, level int, to K, distance float32) error {
	ds.mu.Lock()
	defer ds.mu.Unlock()

	// Convert keys to strings
	fromStr := fmt.Sprintf("%v", from)
	toStr := fmt.Sprintf("%v", to)

	// Create a timeout context
	ctx, cancel := context.WithTimeout(ds.ctx, ds.config.QueryTimeout)
	defer cancel()

	// Insert the link
	insertSQL := fmt.Sprintf(`
		INSERT INTO %s.%s (id, level, link_to, distance)
		VALUES (?, ?, ?, ?)
		ON CONFLICT (id, level, link_to) 
		DO UPDATE SET 
			distance = EXCLUDED.distance
	`, ds.config.Schema, ds.linksTable)

	if _, err := ds.db.ExecContext(ctx, insertSQL, fromStr, level, toStr, distance); err != nil {
		return fmt.Errorf("failed to save link: %w", err)
	}

	return nil
}

// GetLinks retrieves links for a node at a specific level
func (ds *DuckDBStorage[K]) GetLinks(id K, level int) ([]Link[K], error) {
	ds.mu.RLock()
	defer ds.mu.RUnlock()

	// Convert key to string
	idStr := fmt.Sprintf("%v", id)

	// Create a timeout context
	ctx, cancel := context.WithTimeout(ds.ctx, ds.config.QueryTimeout)
	defer cancel()

	// Query the links
	querySQL := fmt.Sprintf(`
		SELECT link_to, distance FROM %s.%s 
		WHERE id = ? AND level = ?
	`, ds.config.Schema, ds.linksTable)

	rows, err := ds.db.QueryContext(ctx, querySQL, idStr, level)
	if err != nil {
		return nil, fmt.Errorf("failed to query links: %w", err)
	}
	defer rows.Close()

	var links []Link[K]
	for rows.Next() {
		var toStr string
		var distance float32
		if err := rows.Scan(&toStr, &distance); err != nil {
			return nil, fmt.Errorf("failed to scan row: %w", err)
		}

		// Convert string back to key
		var to K
		if err := convertStringToKey(toStr, &to); err != nil {
			return nil, fmt.Errorf("failed to convert link target ID: %w", err)
		}

		links = append(links, Link[K]{To: to, Distance: distance})
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error during iteration: %w", err)
	}

	return links, nil
}

// convertStringToKey converts a string key to the appropriate type
func convertStringToKey[K any](keyStr string, key *K) error {
	var err error
	switch any(key).(type) {
	case *string:
		*key = any(keyStr).(K)
	case *int:
		var intKey int
		_, err = fmt.Sscanf(keyStr, "%d", &intKey)
		*key = any(intKey).(K)
	case *int64:
		var int64Key int64
		_, err = fmt.Sscanf(keyStr, "%d", &int64Key)
		*key = any(int64Key).(K)
	case *uint64:
		var uint64Key uint64
		_, err = fmt.Sscanf(keyStr, "%d", &uint64Key)
		*key = any(uint64Key).(K)
	default:
		err = fmt.Errorf("unsupported key type: %T", key)
	}
	return err
}

// Close closes the DuckDB connection
func (ds *DuckDBStorage[K]) Close() error {
	ds.mu.Lock()
	defer ds.mu.Unlock()

	if ds.db != nil {
		return ds.db.Close()
	}
	return nil
}

// GetDuckDBStorage returns the DuckDB storage instance
func (ps *PersistentStorage[K]) GetDuckDBStorage() *DuckDBStorage[K] {
	return ps.duckDBStorage
}

// UseDuckDB checks if DuckDB storage is being used
func (ps *PersistentStorage[K]) UseDuckDB() bool {
	return ps.duckDBStorage != nil
}

// Close closes the storage
func (ps *PersistentStorage[K]) Close() error {
	ps.mu.Lock()
	defer ps.mu.Unlock()

	var errs []error

	// Close Parquet storage if available
	if ps.parquetStorage != nil {
		if err := ps.parquetStorage.Close(); err != nil {
			errs = append(errs, fmt.Errorf("failed to close Parquet storage: %w", err))
		}
	}

	// Close DuckDB storage if available
	if ps.duckDBStorage != nil {
		if err := ps.duckDBStorage.Close(); err != nil {
			errs = append(errs, fmt.Errorf("failed to close DuckDB storage: %w", err))
		}
	}

	if len(errs) > 0 {
		return errors.Join(errs...)
	}
	return nil
}

// SaveMetadata saves metadata to disk
func (ps *PersistentStorage[K]) SaveMetadata(key K, metadata json.RawMessage) error {
	ps.mu.Lock()
	defer ps.mu.Unlock()

	// Create metadata map
	metadataMap := make(map[string]json.RawMessage)

	// Check if metadata file exists
	if _, err := os.Stat(ps.metadataFile); !os.IsNotExist(err) {
		// Read existing metadata
		data, err := os.ReadFile(ps.metadataFile)
		if err != nil {
			return fmt.Errorf("failed to read metadata file: %w", err)
		}

		// Parse existing metadata
		if err := sonic.Unmarshal(data, &metadataMap); err != nil {
			return fmt.Errorf("failed to parse metadata: %w", err)
		}
	}

	// Convert key to string
	keyBytes, err := sonic.Marshal(key)
	if err != nil {
		return fmt.Errorf("failed to marshal key: %w", err)
	}
	keyStr := string(keyBytes)

	// Add new metadata
	metadataMap[keyStr] = metadata

	// Serialize to JSON
	data, err := sonic.Marshal(metadataMap)
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %w", err)
	}

	// Write to file
	if err := os.WriteFile(ps.metadataFile, data, 0644); err != nil {
		return fmt.Errorf("failed to write metadata file: %w", err)
	}

	return nil
}

// BatchSaveMetadata saves multiple metadata entries to disk
func (ps *PersistentStorage[K]) BatchSaveMetadata(keys []K, metadataList []json.RawMessage) error {
	if len(keys) != len(metadataList) {
		return errors.New("keys and metadataList must have the same length")
	}

	ps.mu.Lock()
	defer ps.mu.Unlock()

	// Create metadata map
	metadataMap := make(map[string]json.RawMessage)

	// Check if metadata file exists
	if _, err := os.Stat(ps.metadataFile); !os.IsNotExist(err) {
		// Read existing metadata
		data, err := os.ReadFile(ps.metadataFile)
		if err != nil {
			return fmt.Errorf("failed to read metadata file: %w", err)
		}

		// Parse existing metadata
		if err := sonic.Unmarshal(data, &metadataMap); err != nil {
			return fmt.Errorf("failed to parse metadata: %w", err)
		}
	}

	// Add new metadata
	for i, key := range keys {
		if i >= len(metadataList) || metadataList[i] == nil {
			continue
		}

		// Convert key to string
		keyBytes, err := sonic.Marshal(key)
		if err != nil {
			return fmt.Errorf("failed to marshal key: %w", err)
		}
		keyStr := string(keyBytes)

		// Add metadata
		metadataMap[keyStr] = metadataList[i]
	}

	// Serialize to JSON
	data, err := sonic.Marshal(metadataMap)
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %w", err)
	}

	// Write to file
	if err := os.WriteFile(ps.metadataFile, data, 0644); err != nil {
		return fmt.Errorf("failed to write metadata file: %w", err)
	}

	return nil
}

// LoadMetadata loads metadata from disk
func (ps *PersistentStorage[K]) LoadMetadata() (*meta.MemoryMetadataStore[K], error) {
	ps.mu.Lock()
	defer ps.mu.Unlock()

	// Check if metadata file exists
	if _, err := os.Stat(ps.metadataFile); os.IsNotExist(err) {
		// Return empty store if file doesn't exist
		return meta.NewMemoryMetadataStore[K](), nil
	}

	// Read metadata file
	data, err := os.ReadFile(ps.metadataFile)
	if err != nil {
		return nil, fmt.Errorf("failed to read metadata file: %w", err)
	}

	// Parse metadata
	var metadataMap map[string]json.RawMessage
	if err := sonic.Unmarshal(data, &metadataMap); err != nil {
		return nil, fmt.Errorf("failed to parse metadata: %w", err)
	}

	// Create metadata store
	metaStore := meta.NewMemoryMetadataStore[K]()

	// Add metadata to store
	for keyStr, metadata := range metadataMap {
		// Convert string key to K
		var key K
		if err := sonic.Unmarshal([]byte(keyStr), &key); err != nil {
			return nil, fmt.Errorf("failed to parse key: %w", err)
		}

		// Add metadata to store
		if err := metaStore.Add(key, metadata); err != nil {
			return nil, fmt.Errorf("failed to add metadata: %w", err)
		}
	}

	return metaStore, nil
}

// SaveFacets saves facets to disk
func (ps *PersistentStorage[K]) SaveFacets(facetedNode facets.FacetedNode[K]) error {
	ps.mu.Lock()
	defer ps.mu.Unlock()

	// Create a serializable representation
	type SerializableFacet struct {
		Name  string      `json:"name"`
		Value interface{} `json:"value"`
	}

	type SerializableNode struct {
		Key    string              `json:"key"`
		Vector []float32           `json:"vector"`
		Facets []SerializableFacet `json:"facets"`
	}

	// Create facets list
	var facetsList []SerializableNode

	// Check if facets file exists
	if _, err := os.Stat(ps.facetsFile); !os.IsNotExist(err) {
		// Read existing facets
		data, err := os.ReadFile(ps.facetsFile)
		if err != nil {
			return fmt.Errorf("failed to read facets file: %w", err)
		}

		// Parse existing facets
		if err := json.Unmarshal(data, &facetsList); err != nil {
			return fmt.Errorf("failed to parse facets: %w", err)
		}
	}

	// Convert key to string
	keyBytes, err := sonic.Marshal(facetedNode.Node.Key)
	if err != nil {
		return fmt.Errorf("failed to marshal key: %w", err)
	}
	keyStr := string(keyBytes)

	// Convert facets to serializable format
	serializableFacets := make([]SerializableFacet, len(facetedNode.Facets))
	for i, facet := range facetedNode.Facets {
		serializableFacets[i] = SerializableFacet{
			Name:  facet.Name(),
			Value: facet.Value(),
		}
	}

	// Add new facets
	facetsList = append(facetsList, SerializableNode{
		Key:    keyStr,
		Vector: facetedNode.Node.Value,
		Facets: serializableFacets,
	})

	// Serialize to JSON
	data, err := sonic.Marshal(facetsList)
	if err != nil {
		return fmt.Errorf("failed to marshal facets: %w", err)
	}

	// Write to file
	if err := os.WriteFile(ps.facetsFile, data, 0644); err != nil {
		return fmt.Errorf("failed to write facets file: %w", err)
	}

	return nil
}

// BatchSaveFacets saves multiple faceted nodes to disk
func (ps *PersistentStorage[K]) BatchSaveFacets(facetedNodes []facets.FacetedNode[K]) error {
	ps.mu.Lock()
	defer ps.mu.Unlock()

	// Create a serializable representation
	type SerializableFacet struct {
		Name  string      `json:"name"`
		Value interface{} `json:"value"`
	}

	type SerializableNode struct {
		Key    string              `json:"key"`
		Vector []float32           `json:"vector"`
		Facets []SerializableFacet `json:"facets"`
	}

	// Create facets list
	var facetsList []SerializableNode

	// Check if facets file exists
	if _, err := os.Stat(ps.facetsFile); !os.IsNotExist(err) {
		// Read existing facets
		data, err := os.ReadFile(ps.facetsFile)
		if err != nil {
			return fmt.Errorf("failed to read facets file: %w", err)
		}

		// Parse existing facets
		if err := json.Unmarshal(data, &facetsList); err != nil {
			return fmt.Errorf("failed to parse facets: %w", err)
		}
	}

	// Add new facets
	for _, facetedNode := range facetedNodes {
		// Convert key to string
		keyBytes, err := sonic.Marshal(facetedNode.Node.Key)
		if err != nil {
			return fmt.Errorf("failed to marshal key: %w", err)
		}
		keyStr := string(keyBytes)

		// Convert facets to serializable format
		serializableFacets := make([]SerializableFacet, len(facetedNode.Facets))
		for i, facet := range facetedNode.Facets {
			serializableFacets[i] = SerializableFacet{
				Name:  facet.Name(),
				Value: facet.Value(),
			}
		}

		// Add facets
		facetsList = append(facetsList, SerializableNode{
			Key:    keyStr,
			Vector: facetedNode.Node.Value,
			Facets: serializableFacets,
		})
	}

	// Serialize to JSON
	data, err := sonic.Marshal(facetsList)
	if err != nil {
		return fmt.Errorf("failed to marshal facets: %w", err)
	}

	// Write to file
	if err := os.WriteFile(ps.facetsFile, data, 0644); err != nil {
		return fmt.Errorf("failed to write facets file: %w", err)
	}

	return nil
}

// LoadFacets loads facets from disk
func (ps *PersistentStorage[K]) LoadFacets() (*facets.MemoryFacetStore[K], error) {
	ps.mu.Lock()
	defer ps.mu.Unlock()

	// Create a serializable representation
	type SerializableFacet struct {
		Name  string      `json:"name"`
		Value interface{} `json:"value"`
	}

	type SerializableNode struct {
		Key    string              `json:"key"`
		Vector []float32           `json:"vector"`
		Facets []SerializableFacet `json:"facets"`
	}

	// Check if facets file exists
	if _, err := os.Stat(ps.facetsFile); os.IsNotExist(err) {
		// Return empty store if file doesn't exist
		return facets.NewMemoryFacetStore[K](), nil
	}

	// Read facets file
	data, err := os.ReadFile(ps.facetsFile)
	if err != nil {
		return nil, fmt.Errorf("failed to read facets file: %w", err)
	}

	// Parse facets
	var facetsList []SerializableNode
	if err := sonic.Unmarshal(data, &facetsList); err != nil {
		return nil, fmt.Errorf("failed to parse facets: %w", err)
	}

	// Create facet store
	facetStore := facets.NewMemoryFacetStore[K]()

	// Add facets to store
	for _, item := range facetsList {
		// Convert string key to K
		var key K
		if err := sonic.Unmarshal([]byte(item.Key), &key); err != nil {
			return nil, fmt.Errorf("failed to parse key: %w", err)
		}

		// Create node
		node := hnsw.MakeNode(key, item.Vector)

		// Convert serializable facets to facets
		facetList := make([]facets.Facet, len(item.Facets))
		for i, serializableFacet := range item.Facets {
			facetList[i] = facets.NewBasicFacet(serializableFacet.Name, serializableFacet.Value)
		}

		// Create faceted node
		facetedNode := facets.NewFacetedNode(node, facetList)

		// Add faceted node to store
		if err := facetStore.Add(facetedNode); err != nil {
			return nil, fmt.Errorf("failed to add facets: %w", err)
		}
	}

	return facetStore, nil
}

// OptimizeStorage performs optimization operations on the storage
// This can include compacting files, removing deleted entries, etc.
func (ps *PersistentStorage[K]) OptimizeStorage() error {
	ps.mu.Lock()
	defer ps.mu.Unlock()

	// Perform optimization operations
	// For example, compact Parquet files, remove deleted entries, etc.
	// This is a placeholder for actual optimization logic

	return nil
}

// Backup creates a backup of the database to the specified directory
func (ps *PersistentStorage[K]) Backup(backupDir string) error {
	ps.mu.RLock()
	defer ps.mu.RUnlock()

	// Create backup directory if it doesn't exist
	if err := os.MkdirAll(backupDir, 0755); err != nil {
		return fmt.Errorf("failed to create backup directory: %w", err)
	}

	// Backup metadata
	if _, err := os.Stat(ps.metadataFile); !os.IsNotExist(err) {
		metadataBackupFile := filepath.Join(backupDir, "metadata.json")
		if err := copyFile(ps.metadataFile, metadataBackupFile); err != nil {
			return fmt.Errorf("failed to backup metadata: %w", err)
		}
	}

	// Backup facets
	if _, err := os.Stat(ps.facetsFile); !os.IsNotExist(err) {
		facetsBackupFile := filepath.Join(backupDir, "facets.json")
		if err := copyFile(ps.facetsFile, facetsBackupFile); err != nil {
			return fmt.Errorf("failed to backup facets: %w", err)
		}
	}

	// Backup vectors using Parquet storage
	vectorsBackupDir := filepath.Join(backupDir, "vectors")
	if err := os.MkdirAll(vectorsBackupDir, 0755); err != nil {
		return fmt.Errorf("failed to create vectors backup directory: %w", err)
	}

	// Get the source directory from Parquet storage config
	sourceDir := filepath.Join(ps.baseDir, "vectors")

	// Copy all files from the source directory to the backup directory
	if err := copyDir(sourceDir, vectorsBackupDir); err != nil {
		return fmt.Errorf("failed to backup vectors: %w", err)
	}

	return nil
}

// Restore restores the database from the specified backup directory
func (ps *PersistentStorage[K]) Restore(backupDir string) error {
	ps.mu.Lock()
	defer ps.mu.Unlock()

	// Check if backup directory exists
	if _, err := os.Stat(backupDir); os.IsNotExist(err) {
		return fmt.Errorf("backup directory does not exist: %s", backupDir)
	}

	// Restore metadata
	metadataBackupFile := filepath.Join(backupDir, "metadata.json")
	if _, err := os.Stat(metadataBackupFile); !os.IsNotExist(err) {
		if err := copyFile(metadataBackupFile, ps.metadataFile); err != nil {
			return fmt.Errorf("failed to restore metadata: %w", err)
		}
	}

	// Restore facets
	facetsBackupFile := filepath.Join(backupDir, "facets.json")
	if _, err := os.Stat(facetsBackupFile); !os.IsNotExist(err) {
		if err := copyFile(facetsBackupFile, ps.facetsFile); err != nil {
			return fmt.Errorf("failed to restore facets: %w", err)
		}
	}

	// Restore vectors using Parquet storage
	vectorsBackupDir := filepath.Join(backupDir, "vectors")
	if _, err := os.Stat(vectorsBackupDir); !os.IsNotExist(err) {
		// Get the target directory from Parquet storage config
		targetDir := filepath.Join(ps.baseDir, "vectors")

		// Remove existing files in the target directory
		if err := os.RemoveAll(targetDir); err != nil {
			return fmt.Errorf("failed to clean target directory: %w", err)
		}

		// Create the target directory
		if err := os.MkdirAll(targetDir, 0755); err != nil {
			return fmt.Errorf("failed to create target directory: %w", err)
		}

		// Copy all files from the backup directory to the target directory
		if err := copyDir(vectorsBackupDir, targetDir); err != nil {
			return fmt.Errorf("failed to restore vectors: %w", err)
		}
	}

	return nil
}

// copyFile copies a file from src to dst
func copyFile(src, dst string) error {
	// Read the source file
	data, err := os.ReadFile(src)
	if err != nil {
		return fmt.Errorf("failed to read source file: %w", err)
	}

	// Write to the destination file
	if err := os.WriteFile(dst, data, 0644); err != nil {
		return fmt.Errorf("failed to write destination file: %w", err)
	}

	return nil
}

// copyDir copies all files from src directory to dst directory
func copyDir(src, dst string) error {
	// Get all files in the source directory
	entries, err := os.ReadDir(src)
	if err != nil {
		return fmt.Errorf("failed to read source directory: %w", err)
	}

	// Copy each file to the destination directory
	for _, entry := range entries {
		srcPath := filepath.Join(src, entry.Name())
		dstPath := filepath.Join(dst, entry.Name())

		if entry.IsDir() {
			// Create the destination directory
			if err := os.MkdirAll(dstPath, 0755); err != nil {
				return fmt.Errorf("failed to create destination directory: %w", err)
			}

			// Recursively copy the directory
			if err := copyDir(srcPath, dstPath); err != nil {
				return err
			}
		} else {
			// Copy the file
			if err := copyFile(srcPath, dstPath); err != nil {
				return err
			}
		}
	}

	return nil
}

// GetParquetStorage returns the underlying Parquet storage
func (ps *PersistentStorage[K]) GetParquetStorage() *parquet.ParquetStorage[K] {
	return ps.parquetStorage
}

// GetDB returns the underlying database connection
func (ds *DuckDBStorage[K]) GetDB() *sql.DB {
	return ds.db
}

// GetVectorTableName returns the name of the vectors table
func (ds *DuckDBStorage[K]) GetVectorTableName() string {
	return ds.vectorTable
}

// ExportToMotherDuck exports data to MotherDuck cloud
func (ds *DuckDBStorage[K]) ExportToMotherDuck(targetDB string, tables []string) error {
	// Check if MotherDuck token is provided
	if ds.config.MotherDuckToken == "" {
		return errors.New("MotherDuck token not provided")
	}

	ds.mu.Lock()
	defer ds.mu.Unlock()

	// Create a timeout context
	ctx, cancel := context.WithTimeout(ds.ctx, ds.config.QueryTimeout*2)
	defer cancel()

	// Create a MotherDuck connection string
	motherDuckConn := fmt.Sprintf("md:%s?motherduck_token=%s", targetDB, ds.config.MotherDuckToken)

	// Begin a transaction
	tx, err := ds.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer tx.Rollback()

	// Attach MotherDuck database
	_, err = tx.ExecContext(ctx, fmt.Sprintf("ATTACH '%s' AS motherduck", motherDuckConn))
	if err != nil {
		return fmt.Errorf("failed to attach MotherDuck database: %w", err)
	}

	// Export tables
	for _, table := range tables {
		sourceTable := fmt.Sprintf("%s.%s", ds.config.Schema, table)
		targetTable := fmt.Sprintf("motherduck.%s", table)

		// Create table if not exists
		_, err = tx.ExecContext(ctx, fmt.Sprintf("CREATE TABLE IF NOT EXISTS %s AS SELECT * FROM %s WHERE 1=0", targetTable, sourceTable))
		if err != nil {
			return fmt.Errorf("failed to create table %s: %w", targetTable, err)
		}

		// Insert data
		_, err = tx.ExecContext(ctx, fmt.Sprintf("INSERT INTO %s SELECT * FROM %s", targetTable, sourceTable))
		if err != nil {
			return fmt.Errorf("failed to export data to %s: %w", targetTable, err)
		}
	}

	// Commit transaction
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}

	return nil
}
