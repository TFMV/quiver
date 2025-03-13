package db

import (
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/TFMV/hnsw"
	"github.com/TFMV/hnsw/hnsw-extensions/facets"
	"github.com/TFMV/hnsw/hnsw-extensions/meta"
	"github.com/TFMV/hnsw/hnsw-extensions/parquet"
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

	// Mutex for thread safety
	mu sync.RWMutex
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
		if err := json.Unmarshal(data, &metadataMap); err != nil {
			return fmt.Errorf("failed to parse metadata: %w", err)
		}
	}

	// Convert key to string
	keyBytes, err := json.Marshal(key)
	if err != nil {
		return fmt.Errorf("failed to marshal key: %w", err)
	}
	keyStr := string(keyBytes)

	// Add new metadata
	metadataMap[keyStr] = metadata

	// Serialize to JSON
	data, err := json.Marshal(metadataMap)
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
		if err := json.Unmarshal(data, &metadataMap); err != nil {
			return fmt.Errorf("failed to parse metadata: %w", err)
		}
	}

	// Add new metadata
	for i, key := range keys {
		if i >= len(metadataList) || metadataList[i] == nil {
			continue
		}

		// Convert key to string
		keyBytes, err := json.Marshal(key)
		if err != nil {
			return fmt.Errorf("failed to marshal key: %w", err)
		}
		keyStr := string(keyBytes)

		// Add metadata
		metadataMap[keyStr] = metadataList[i]
	}

	// Serialize to JSON
	data, err := json.Marshal(metadataMap)
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
	if err := json.Unmarshal(data, &metadataMap); err != nil {
		return nil, fmt.Errorf("failed to parse metadata: %w", err)
	}

	// Create metadata store
	metaStore := meta.NewMemoryMetadataStore[K]()

	// Add metadata to store
	for keyStr, metadata := range metadataMap {
		// Convert string key to K
		var key K
		if err := json.Unmarshal([]byte(keyStr), &key); err != nil {
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
	keyBytes, err := json.Marshal(facetedNode.Node.Key)
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
	data, err := json.Marshal(facetsList)
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
		keyBytes, err := json.Marshal(facetedNode.Node.Key)
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
	data, err := json.Marshal(facetsList)
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
	if err := json.Unmarshal(data, &facetsList); err != nil {
		return nil, fmt.Errorf("failed to parse facets: %w", err)
	}

	// Create facet store
	facetStore := facets.NewMemoryFacetStore[K]()

	// Add facets to store
	for _, item := range facetsList {
		// Convert string key to K
		var key K
		if err := json.Unmarshal([]byte(item.Key), &key); err != nil {
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

// Close releases resources used by the storage
func (ps *PersistentStorage[K]) Close() error {
	ps.mu.Lock()
	defer ps.mu.Unlock()

	// Close Parquet storage
	if err := ps.parquetStorage.Close(); err != nil {
		return fmt.Errorf("failed to close Parquet storage: %w", err)
	}

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
