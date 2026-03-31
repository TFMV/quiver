package persistence

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// CollectionConfig holds configuration information for a collection
type CollectionConfig struct {
	// Name of the collection
	Name string `json:"name"`
	// Dimension of vectors in the collection
	Dimension int `json:"dimension"`
	// Distance function used by the collection
	DistanceFunc string `json:"distance_func"`
	// Creation time of the collection
	CreatedAt time.Time `json:"created_at"`
	// Fields to be indexed as facets
	FacetFields []string `json:"facet_fields,omitempty"`
	// Version for schema evolution
	Version int `json:"version"`
}

// VectorRecord represents a vector record that can be saved to storage
type VectorRecord struct {
	ID       string            `json:"id"`
	Vector   []float32         `json:"vector"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

// GetCollectionCallback is a function type that retrieves a collection by name
type GetCollectionCallback func(name string) (Persistable, error)

// WAL types for Write-Ahead Log
type WalType string

const (
	WalTypeAdd    WalType = "add"
	WalTypeDelete WalType = "delete"
)

// WalEntry represents a single mutation in the Write-Ahead Log
type WalEntry struct {
	Timestamp time.Time         `json:"timestamp"`
	Type      WalType           `json:"type"`
	VectorID  string            `json:"vector_id,omitempty"`
	Vector    []float32         `json:"vector,omitempty"`
	Metadata  map[string]string `json:"metadata,omitempty"`
}

// WalFile represents a WAL segment file
type WalFile struct {
	Entries []WalEntry `json:"entries"`
}

// Persistable is an interface for objects that can be persisted
type Persistable interface {
	// GetName returns the name of the collection
	GetName() string
	// GetDimension returns the dimension of vectors in the collection
	GetDimension() int
	// GetVectors returns all vectors in the collection
	GetVectors() []VectorRecord
	// AddVector adds a vector to the collection (used during loading)
	AddVector(id string, vector []float32, metadata map[string]string) error
	// GetFacetFields returns the fields that are indexed as facets
	GetFacetFields() []string
	// SetFacetFields sets the fields to be indexed as facets
	SetFacetFields(fields []string)
}

// Manager handles persistence operations for the database
type Manager struct {
	// Root directory for storage
	rootDir string
	// Flush interval for background saving
	flushInterval time.Duration
	// Collections that need to be flushed
	dirtyCollections map[string]bool
	// Lock for thread safety
	mu sync.RWMutex
	// Stop channel for the background flush goroutine
	stopCh chan struct{}
	// Whether the manager is running
	running bool
	// Callback to get a collection by name
	getCollectionCallback GetCollectionCallback

	// WAL-related fields
	walDir        string
	walFile       *os.File
	walMu         sync.Mutex
	currentWalSeq uint64
	enableWAL     bool
}

// NewManager creates a new persistence manager
func NewManager(rootDir string, flushInterval time.Duration) (*Manager, error) {
	// Ensure root directory exists
	if err := os.MkdirAll(rootDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create root directory: %w", err)
	}

	manager := &Manager{
		rootDir:          rootDir,
		flushInterval:    flushInterval,
		dirtyCollections: make(map[string]bool),
		mu:               sync.RWMutex{},
		stopCh:           make(chan struct{}),
		running:          true,
		enableWAL:        true,
	}

	// Initialize WAL directory
	manager.walDir = filepath.Join(rootDir, ".wal")
	if err := os.MkdirAll(manager.walDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create WAL directory: %w", err)
	}

	// Start background flush goroutine only if the interval is greater than zero
	if flushInterval > 0 {
		go manager.backgroundFlush()
	} else {
		manager.running = false // Mark as not running if no background flush
	}

	return manager, nil
}

// backgroundFlush periodically flushes dirty collections to disk
func (m *Manager) backgroundFlush() {
	ticker := time.NewTicker(m.flushInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			m.FlushDirtyCollections()
		case <-m.stopCh:
			return
		}
	}
}

// Stop stops the background flush goroutine and closes WAL
func (m *Manager) Stop() {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.running {
		close(m.stopCh)
		m.running = false
	}

	// Close WAL
	if m.walFile != nil {
		m.walFile.Close()
	}
}

// SetGetCollectionCallback sets the callback function for retrieving collections
func (m *Manager) SetGetCollectionCallback(callback GetCollectionCallback) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.getCollectionCallback = callback
}

// FlushDirtyCollections flushes all dirty collections to disk
func (m *Manager) FlushDirtyCollections() {
	m.mu.Lock()
	collectionNames := make([]string, 0, len(m.dirtyCollections))
	getCallback := m.getCollectionCallback
	for name := range m.dirtyCollections {
		collectionNames = append(collectionNames, name)
	}
	m.mu.Unlock()

	// If no callback is set, we can't retrieve collections
	if getCallback == nil {
		return
	}

	for _, name := range collectionNames {
		// Check if still dirty (another flush might have cleared it)
		m.mu.Lock()
		if !m.dirtyCollections[name] {
			m.mu.Unlock()
			continue
		}
		m.mu.Unlock()

		collection, err := getCallback(name)
		if err != nil {
			// Log the error and continue
			fmt.Printf("Error retrieving collection %s: %v\n", name, err)
			continue
		}

		// Flush the collection
		collectionPath := filepath.Join(m.rootDir, name)
		err = m.FlushCollection(collection, collectionPath)
		if err != nil {
			// Log the error and continue
			fmt.Printf("Error flushing collection %s: %v\n", name, err)
			continue
		}

		// Mark as not dirty
		m.mu.Lock()
		delete(m.dirtyCollections, name)
		m.mu.Unlock()
	}

	// Truncate WAL after successful flush
	if m.enableWAL {
		m.truncateWal()
	}
}

// MarkCollectionDirty marks a collection as dirty, indicating it needs to be flushed to disk
func (m *Manager) MarkCollectionDirty(collectionName string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.dirtyCollections[collectionName] = true
}

// SaveCollectionConfig saves a collection's configuration to disk using safe write
func SaveCollectionConfig(config CollectionConfig, configPath string) error {
	// Create parent directory if it doesn't exist
	if err := os.MkdirAll(filepath.Dir(configPath), 0755); err != nil {
		return fmt.Errorf("failed to create config directory: %w", err)
	}

	// Marshal config to JSON
	data, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal collection config: %w", err)
	}

	// Use safe write pattern
	return safeWriteFile(configPath, data, 0644)
}

// LoadCollectionConfig loads a collection's configuration from disk
func LoadCollectionConfig(configPath string) (CollectionConfig, error) {
	// Read file
	data, err := os.ReadFile(configPath)
	if err != nil {
		return CollectionConfig{}, fmt.Errorf("failed to read collection config: %w", err)
	}

	// Unmarshal JSON
	var config CollectionConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return CollectionConfig{}, fmt.Errorf("failed to unmarshal collection config: %w", err)
	}

	return config, nil
}

// FlushCollection saves a collection to disk using safe write pattern
func (m *Manager) FlushCollection(collection interface{}, collectionPath string) error {
	// Create collection directory if it doesn't exist
	if err := os.MkdirAll(collectionPath, 0755); err != nil {
		return fmt.Errorf("failed to create collection directory: %w", err)
	}

	// Try to access collection methods via type assertion or reflection
	var (
		vectors      []VectorRecord
		name         string
		dimension    int
		distanceFunc string
		facetFields  []string
	)

	// Get collection name
	if nameGetter, ok := collection.(interface{ GetName() string }); ok {
		name = nameGetter.GetName()
	} else {
		return fmt.Errorf("collection does not implement GetName method")
	}

	// Get dimension
	if dimGetter, ok := collection.(interface{ GetDimension() int }); ok {
		dimension = dimGetter.GetDimension()
	} else {
		return fmt.Errorf("collection does not implement GetDimension method")
	}

	// Get vectors - either as VectorRecord slice or via extraction
	if vectorGetter, ok := collection.(interface{ GetVectors() []VectorRecord }); ok {
		vectors = vectorGetter.GetVectors()
	} else {
		return fmt.Errorf("collection does not implement GetVectors method")
	}

	// Try to get distance function type if supported
	if dfGetter, ok := collection.(interface{ GetDistanceFunction() string }); ok {
		distanceFunc = dfGetter.GetDistanceFunction()
	} else {
		// Default to cosine distance
		distanceFunc = "cosine"
	}

	// Get facet fields if supported
	if facetGetter, ok := collection.(interface{ GetFacetFields() []string }); ok {
		facetFields = facetGetter.GetFacetFields()
	}

	// Save vectors to parquet file using safe write
	vectorsPath := filepath.Join(collectionPath, "vectors.parquet")

	// Try using Parquet first
	parquetErr := WriteVectorsToParquetFile(vectors, vectorsPath)
	if parquetErr != nil {
		// If Parquet fails, fall back to JSON
		fallbackPath := filepath.Join(collectionPath, "vectors.json")
		fmt.Printf("Warning: Parquet save failed, falling back to JSON: %v\n", parquetErr)
		if err := WriteVectorsToFile(vectors, fallbackPath); err != nil {
			return fmt.Errorf("failed to write vectors: %w", err)
		}
	}

	// Update collection config
	config := CollectionConfig{
		Name:         name,
		Dimension:    dimension,
		DistanceFunc: distanceFunc,
		CreatedAt:    time.Now(),
		FacetFields:  facetFields,
		Version:      1,
	}

	configPath := filepath.Join(collectionPath, "config.json")
	if err := SaveCollectionConfig(config, configPath); err != nil {
		return fmt.Errorf("failed to save config: %w", err)
	}

	// Remove collection from dirty list
	m.mu.Lock()
	delete(m.dirtyCollections, name)
	m.mu.Unlock()

	return nil
}

// LoadCollection loads a collection from disk with recovery
func (m *Manager) LoadCollection(collection interface{}, collectionPath string) error {
	// Check if the collection supports vector addition
	vectorAdder, ok := collection.(interface {
		AddVector(id string, vector []float32, metadata map[string]string) error
	})
	if !ok {
		return fmt.Errorf("collection does not implement AddVector method")
	}

	// First, recover from WAL if available
	if m.enableWAL {
		walPath := filepath.Join(m.walDir, filepath.Base(collectionPath)+".wal")
		if _, err := os.Stat(walPath); err == nil {
			// WAL exists, recover
			fmt.Printf("Found WAL for %s, recovering...\n", collectionPath)
			if err := m.recoverFromWal(walPath, vectorAdder); err != nil {
				fmt.Printf("WAL recovery failed: %v\n", err)
			}
		}
	}

	// Load collection configuration
	configPath := filepath.Join(collectionPath, "config.json")
	config, err := LoadCollectionConfig(configPath)
	if err == nil {
		// If config was loaded successfully, check if collection supports facet fields
		if facetSetter, ok := collection.(interface{ SetFacetFields(fields []string) }); ok && len(config.FacetFields) > 0 {
			// Set facet fields from config
			facetSetter.SetFacetFields(config.FacetFields)
		}
	}

	// Then try to load from Parquet format
	vectorsPath := filepath.Join(collectionPath, "vectors.parquet")
	if _, err := os.Stat(vectorsPath); err == nil {
		// Parquet file exists, try to load it
		vectors, err := ReadVectorsFromParquetFile(vectorsPath)
		if err == nil {
			// Successfully loaded from Parquet
			for _, vector := range vectors {
				if err := vectorAdder.AddVector(vector.ID, vector.Vector, vector.Metadata); err != nil {
					return fmt.Errorf("failed to add vector: %w", err)
				}
			}
			return nil
		}

		// If Parquet loading fails, log and fall through to try JSON
		fmt.Printf("Warning: Failed to load from Parquet, trying JSON: %v\n", err)
	}

	// Fall back to JSON format
	jsonPath := filepath.Join(collectionPath, "vectors.json")
	if _, err := os.Stat(jsonPath); os.IsNotExist(err) {
		// No vectors to load
		return nil
	}

	vectors, err := ReadVectorsFromFile(jsonPath)
	if err != nil {
		return fmt.Errorf("failed to read vectors: %w", err)
	}

	// Add vectors to collection
	for _, vector := range vectors {
		if err := vectorAdder.AddVector(vector.ID, vector.Vector, vector.Metadata); err != nil {
			return fmt.Errorf("failed to add vector: %w", err)
		}
	}

	return nil
}

// recoverFromWal replays WAL entries to recover uncommitted changes
func (m *Manager) recoverFromWal(walPath string, vectorAdder interface {
	AddVector(id string, vector []float32, metadata map[string]string) error
}) error {
	data, err := os.ReadFile(walPath)
	if err != nil {
		return fmt.Errorf("failed to read WAL: %w", err)
	}

	var walFile WalFile
	if err := json.Unmarshal(data, &walFile); err != nil {
		return fmt.Errorf("failed to parse WAL: %w", err)
	}

	// Replay entries
	for _, entry := range walFile.Entries {
		switch entry.Type {
		case WalTypeAdd:
			if err := vectorAdder.AddVector(entry.VectorID, entry.Vector, entry.Metadata); err != nil {
				fmt.Printf("Warning: failed to replay add for %s: %v\n", entry.VectorID, err)
			}
		case WalTypeDelete:
			// Note: Delete not replayed since we're loading from snapshot
			fmt.Printf("Note: Delete operation in WAL not replayed (exists in snapshot): %s\n", entry.VectorID)
		}
	}

	return nil
}

// logMutation writes a mutation to the WAL
func (m *Manager) logMutation(entry WalEntry) {
	if !m.enableWAL {
		return
	}

	m.walMu.Lock()
	defer m.walMu.Unlock()

	// Lazy init WAL file
	if m.walFile == nil {
		walPath := filepath.Join(m.walDir, fmt.Sprintf("wal.%d", m.currentWalSeq))
		f, err := os.OpenFile(walPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
		if err != nil {
			fmt.Printf("Warning: failed to open WAL: %v\n", err)
			return
		}
		m.walFile = f
	}

	// Write entry
	data, err := json.Marshal(entry)
	if err != nil {
		fmt.Printf("Warning: failed to marshal WAL entry: %v\n", err)
		return
	}
	m.walFile.Write(data)
	m.walFile.Write([]byte("\n"))
}

// truncateWal truncates the WAL after successful flush
func (m *Manager) truncateWal() {
	m.walMu.Lock()
	defer m.walMu.Unlock()

	if m.walFile != nil {
		m.walFile.Close()
		m.walFile = nil

		// Remove old WAL files
		filepath.Walk(m.walDir, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return err
			}
			if !info.IsDir() && filepath.Ext(info.Name()) == ".wal" {
				os.Remove(path)
			}
			return nil
		})
	}
}

// CreateBackup creates a backup of the database
func (m *Manager) CreateBackup(sourcePath, destPath string) error {
	// Create destination directory if it doesn't exist
	if err := os.MkdirAll(destPath, 0755); err != nil {
		return fmt.Errorf("failed to create backup directory: %w", err)
	}

	// Walk through source directory and copy all files
	return filepath.Walk(sourcePath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Skip WAL directory
		if filepath.Base(path) == ".wal" {
			return nil
		}

		// Skip the root directory itself
		if path == sourcePath {
			return nil
		}

		// Get relative path
		relPath, err := filepath.Rel(sourcePath, path)
		if err != nil {
			return fmt.Errorf("failed to get relative path: %w", err)
		}

		// Create destination path
		destFile := filepath.Join(destPath, relPath)

		if info.IsDir() {
			// Create directory
			return os.MkdirAll(destFile, info.Mode())
		} else {
			// Copy file
			return copyFile(path, destFile)
		}
	})
}

// RestoreBackup restores the database from a backup
func (m *Manager) RestoreBackup(sourcePath, destPath string) error {
	// Create destination directory if it doesn't exist
	if err := os.MkdirAll(destPath, 0755); err != nil {
		return fmt.Errorf("failed to create destination directory: %w", err)
	}

	// Walk through source directory and copy all files
	return filepath.Walk(sourcePath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Skip the root directory itself
		if path == sourcePath {
			return nil
		}

		// Get relative path
		relPath, err := filepath.Rel(sourcePath, path)
		if err != nil {
			return fmt.Errorf("failed to get relative path: %w", err)
		}

		// Create destination path
		destFile := filepath.Join(destPath, relPath)

		if info.IsDir() {
			// Create directory
			return os.MkdirAll(destFile, info.Mode())
		} else {
			// Copy file
			return copyFile(path, destFile)
		}
	})
}

// copyFile copies a file from src to dst
func copyFile(src, dst string) error {
	// Open source file
	srcFile, err := os.Open(src)
	if err != nil {
		return fmt.Errorf("failed to open source file: %w", err)
	}
	defer srcFile.Close()

	// Create destination file
	dstFile, err := os.Create(dst)
	if err != nil {
		return fmt.Errorf("failed to create destination file: %w", err)
	}
	defer dstFile.Close()

	// Copy content
	_, err = io.Copy(dstFile, srcFile)
	if err != nil {
		return fmt.Errorf("failed to copy file content: %w", err)
	}

	// Sync to ensure data is written to disk
	err = dstFile.Sync()
	if err != nil {
		return fmt.Errorf("failed to sync destination file: %w", err)
	}

	return nil
}

// VectorStorage represents a collection's vector storage
type VectorStorage struct {
	Vectors []VectorRecord `json:"vectors"`
}

// safeWriteFile writes data to a temp file and atomically renames it
func safeWriteFile(path string, data []byte, perm os.FileMode) error {
	dir := filepath.Dir(path)
	tempPath := filepath.Join(dir, "."+filepath.Base(path)+".tmp")

	// Write to temp file
	if err := os.WriteFile(tempPath, data, perm); err != nil {
		return fmt.Errorf("failed to write temp file: %w", err)
	}

	// Sync the temp file
	if err := syncFile(tempPath); err != nil {
		os.Remove(tempPath)
		return err
	}

	// Atomic rename
	if err := os.Rename(tempPath, path); err != nil {
		os.Remove(tempPath)
		return fmt.Errorf("failed to rename temp file: %w", err)
	}

	return nil
}

// syncFile syncs file data to disk
func syncFile(path string) error {
	f, err := os.OpenFile(path, os.O_RDWR, 0644)
	if err != nil {
		return err
	}
	defer f.Close()

	if err := f.Sync(); err != nil {
		return err
	}

	return nil
}

// WriteVectorsToFile writes vector records to a JSON file using safe write
func WriteVectorsToFile(records []VectorRecord, filePath string) error {
	// Create storage object
	storage := VectorStorage{
		Vectors: records,
	}

	// Marshal to JSON
	data, err := json.MarshalIndent(storage, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal vectors: %w", err)
	}

	// Create parent directory if it doesn't exist
	if err := os.MkdirAll(filepath.Dir(filePath), 0755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	// Use safe write pattern
	return safeWriteFile(filePath, data, 0644)
}

// ReadVectorsFromFile reads vector records from a JSON file
func ReadVectorsFromFile(filePath string) ([]VectorRecord, error) {
	// Read file
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read vectors file: %w", err)
	}

	// Unmarshal JSON
	var storage VectorStorage
	if err := json.Unmarshal(data, &storage); err != nil {
		return nil, fmt.Errorf("failed to unmarshal vectors: %w", err)
	}

	return storage.Vectors, nil
}
