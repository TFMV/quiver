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
}

// VectorRecord represents a vector record that can be saved to storage
type VectorRecord struct {
	ID       string            `json:"id"`
	Vector   []float32         `json:"vector"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

// GetCollectionCallback is a function type that retrieves a collection by name
type GetCollectionCallback func(name string) (Persistable, error)

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

// Stop stops the background flush goroutine
func (m *Manager) Stop() {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.running {
		close(m.stopCh)
		m.running = false
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
}

// MarkCollectionDirty marks a collection as dirty, indicating it needs to be flushed to disk
func (m *Manager) MarkCollectionDirty(collectionName string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.dirtyCollections[collectionName] = true
}

// SaveCollectionConfig saves a collection's configuration to disk
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

	// Write to file
	if err := os.WriteFile(configPath, data, 0644); err != nil {
		return fmt.Errorf("failed to write collection config: %w", err)
	}

	return nil
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

// FlushCollection saves a collection to disk
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

	// Save vectors to parquet file if possible
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

// LoadCollection loads a collection from disk
func (m *Manager) LoadCollection(collection interface{}, collectionPath string) error {
	// Check if the collection supports vector addition
	vectorAdder, ok := collection.(interface {
		AddVector(id string, vector []float32, metadata map[string]string) error
	})
	if !ok {
		return fmt.Errorf("collection does not implement AddVector method")
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

	// First try to load from Parquet format
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

// WriteVectorsToFile writes vector records to a JSON file
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

	// Write to file
	if err := os.WriteFile(filePath, data, 0644); err != nil {
		return fmt.Errorf("failed to write vectors file: %w", err)
	}

	return nil
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
