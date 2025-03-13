// Package db provides a high-performance vector database extension that integrates
// all the capabilities of the HNSW library, including metadata filtering, hybrid search,
// faceted search, and advanced analytics with durability.
package db

import (
	"cmp"
	"context"
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
	"github.com/TFMV/hnsw/hnsw-extensions/hybrid"
	"github.com/TFMV/hnsw/hnsw-extensions/meta"
	"github.com/TFMV/hnsw/hnsw-extensions/parquet"
)

// VectorDB is a high-performance vector database that integrates all HNSW capabilities
type VectorDB[K cmp.Ordered] struct {
	// Core search index adapter
	index *IndexAdapter[K]

	// Metadata store
	metaStore meta.MetadataStore[K]

	// Facet store
	facetStore facets.FacetStore[K]

	// Persistent storage
	storage *PersistentStorage[K]

	// Configuration
	config DBConfig

	// Statistics and metrics
	stats DBStats

	// Mutex for statistics
	statsMu sync.RWMutex

	// Mutex for thread safety
	mu sync.RWMutex
}

// DBConfig defines configuration options for the vector database
type DBConfig struct {
	// Base directory for storage
	BaseDir string

	// Hybrid index configuration
	Hybrid hybrid.IndexConfig

	// Parquet storage configuration (if using parquet)
	Parquet parquet.ParquetStorageConfig

	// Cache configuration
	CacheSize int // Maximum number of vectors to cache in memory

	// Query execution configuration
	MaxConcurrentQueries int // Maximum number of concurrent queries
	QueryTimeout         time.Duration

	// Analytics configuration
	EnableAnalytics bool // Whether to collect and store analytics data
}

// SerializableDBConfig is a serializable version of DBConfig for backup/restore
type SerializableDBConfig struct {
	// Base directory for storage
	BaseDir string `json:"base_dir"`

	// Hybrid index configuration (without DistanceFunc)
	Hybrid struct {
		Type     hybrid.IndexType `json:"type"`
		M        int              `json:"m"`
		Ml       float64          `json:"ml"`
		EfSearch int              `json:"ef_search"`
		// Distance function name (for serialization)
		DistanceName string `json:"distance_name"`
	} `json:"hybrid"`

	// Parquet storage configuration
	Parquet parquet.ParquetStorageConfig `json:"parquet"`

	// Cache configuration
	CacheSize int `json:"cache_size"`

	// Query execution configuration
	MaxConcurrentQueries int           `json:"max_concurrent_queries"`
	QueryTimeout         time.Duration `json:"query_timeout"`

	// Analytics configuration
	EnableAnalytics bool `json:"enable_analytics"`
}

// toSerializable converts DBConfig to SerializableDBConfig
func (c DBConfig) toSerializable() SerializableDBConfig {
	var sc SerializableDBConfig
	sc.BaseDir = c.BaseDir
	sc.Hybrid.Type = c.Hybrid.Type
	sc.Hybrid.M = c.Hybrid.M
	sc.Hybrid.Ml = c.Hybrid.Ml
	sc.Hybrid.EfSearch = c.Hybrid.EfSearch

	// Convert distance function to name based on function pointer
	// Since we can't directly compare function pointers, use a different approach
	if c.Hybrid.Distance != nil {
		// Get the function name through reflection
		distFuncName := fmt.Sprintf("%T", c.Hybrid.Distance)

		// Extract just the function name from the full path
		if distFuncName == "func([]float32, []float32) float32" {
			// Default to Cosine if we can't determine the specific function
			sc.Hybrid.DistanceName = "Cosine"
		} else if strings.Contains(distFuncName, "L1") {
			sc.Hybrid.DistanceName = "L1"
		} else if strings.Contains(distFuncName, "L2") {
			sc.Hybrid.DistanceName = "L2"
		} else if strings.Contains(distFuncName, "Cosine") {
			sc.Hybrid.DistanceName = "Cosine"
		} else if strings.Contains(distFuncName, "DotProduct") {
			sc.Hybrid.DistanceName = "DotProduct"
		} else {
			sc.Hybrid.DistanceName = "Cosine" // Default
		}
	} else {
		sc.Hybrid.DistanceName = "Cosine" // Default
	}

	sc.Parquet = c.Parquet
	sc.CacheSize = c.CacheSize
	sc.MaxConcurrentQueries = c.MaxConcurrentQueries
	sc.QueryTimeout = c.QueryTimeout
	sc.EnableAnalytics = c.EnableAnalytics

	return sc
}

// fromSerializable converts SerializableDBConfig to DBConfig
func (sc SerializableDBConfig) toDBConfig() DBConfig {
	var c DBConfig
	c.BaseDir = sc.BaseDir
	c.Hybrid.Type = sc.Hybrid.Type
	c.Hybrid.M = sc.Hybrid.M
	c.Hybrid.Ml = sc.Hybrid.Ml
	c.Hybrid.EfSearch = sc.Hybrid.EfSearch

	// Convert distance name to function
	switch sc.Hybrid.DistanceName {
	case "L1":
		c.Hybrid.Distance = func(a, b []float32) float32 {
			// Simple L1 distance implementation
			var sum float32
			for i := range a {
				diff := a[i] - b[i]
				if diff < 0 {
					sum -= diff
				} else {
					sum += diff
				}
			}
			return sum
		}
	case "L2":
		c.Hybrid.Distance = hnsw.EuclideanDistance
	case "Cosine":
		c.Hybrid.Distance = hnsw.CosineDistance
	case "DotProduct":
		c.Hybrid.Distance = func(a, b []float32) float32 {
			// Simple dot product distance implementation
			var sum float32
			for i := range a {
				sum += a[i] * b[i]
			}
			return -sum // Negative because smaller distances are better
		}
	default:
		c.Hybrid.Distance = hnsw.CosineDistance // Default
	}

	c.Parquet = sc.Parquet
	c.CacheSize = sc.CacheSize
	c.MaxConcurrentQueries = sc.MaxConcurrentQueries
	c.QueryTimeout = sc.QueryTimeout
	c.EnableAnalytics = sc.EnableAnalytics

	return c
}

// DefaultDBConfig returns the default configuration for the vector database
func DefaultDBConfig() DBConfig {
	return DBConfig{
		BaseDir:              "vectordb_data",
		Hybrid:               hybrid.DefaultIndexConfig(),
		Parquet:              parquet.DefaultParquetStorageConfig(),
		CacheSize:            10000,
		MaxConcurrentQueries: 10,
		QueryTimeout:         time.Second * 30,
		EnableAnalytics:      true,
	}
}

// DBStats contains statistics about the vector database
type DBStats struct {
	// Total number of vectors
	VectorCount int

	// Index statistics
	IndexStats hybrid.IndexStats

	// Query statistics
	TotalQueries      int64
	SuccessfulQueries int64
	FailedQueries     int64
	AverageQueryTime  time.Duration

	// Storage statistics
	StorageSize int64 // in bytes
}

// QueryOptions defines options for vector search queries
type QueryOptions struct {
	// Number of results to return
	K int

	// Metadata filter (JSON query)
	MetadataFilter json.RawMessage

	// Facet filters
	FacetFilters []facets.FacetFilter

	// Negative examples for search
	NegativeExamples [][]float32

	// Weight for negative examples (0.0 to 1.0)
	NegativeWeight float32

	// Context for cancellation
	Ctx context.Context
}

// WithK sets the number of results to return
func (o QueryOptions) WithK(k int) QueryOptions {
	o.K = k
	return o
}

// WithMetadataFilter sets the metadata filter
func (o QueryOptions) WithMetadataFilter(filter json.RawMessage) QueryOptions {
	o.MetadataFilter = filter
	return o
}

// WithFacetFilters sets the facet filters
func (o QueryOptions) WithFacetFilters(filters ...facets.FacetFilter) QueryOptions {
	o.FacetFilters = filters
	return o
}

// WithNegativeExample adds a negative example
func (o QueryOptions) WithNegativeExample(negative []float32) QueryOptions {
	o.NegativeExamples = [][]float32{negative}
	return o
}

// WithNegativeExamples sets multiple negative examples
func (o QueryOptions) WithNegativeExamples(negatives [][]float32) QueryOptions {
	o.NegativeExamples = negatives
	return o
}

// WithNegativeWeight sets the weight for negative examples
func (o QueryOptions) WithNegativeWeight(weight float32) QueryOptions {
	o.NegativeWeight = weight
	return o
}

// WithContext sets the context for cancellation
func (o QueryOptions) WithContext(ctx context.Context) QueryOptions {
	o.Ctx = ctx
	return o
}

// DefaultQueryOptions returns the default query options
func DefaultQueryOptions() QueryOptions {
	return QueryOptions{
		K:              10,
		NegativeWeight: 0.5,
		Ctx:            context.Background(),
	}
}

// SearchResult represents a search result with vector, metadata, and facets
type SearchResult[K cmp.Ordered] struct {
	// Key of the vector
	Key K

	// Distance to the query vector
	Distance float32

	// Vector data
	Vector []float32

	// Metadata (if available)
	Metadata json.RawMessage

	// Facets (if available)
	Facets []facets.Facet
}

// NewVectorDB creates a new vector database with the given configuration
func NewVectorDB[K cmp.Ordered](config DBConfig) (*VectorDB[K], error) {
	if config.BaseDir == "" {
		config = DefaultDBConfig()
	}

	// Create persistent storage if needed
	var storage *PersistentStorage[K]
	if config.BaseDir != "" {
		var err error
		storage, err = NewPersistentStorage[K](config.BaseDir)
		if err != nil {
			return nil, fmt.Errorf("failed to create persistent storage: %w", err)
		}
	}

	// Create the appropriate index based on configuration
	var adapter *IndexAdapter[K]

	switch config.Hybrid.Type {
	case hybrid.HybridIndexType:
		// Create a hybrid index
		hybridIndex, err := hybrid.NewHybridIndex[K](config.Hybrid)
		if err != nil {
			return nil, fmt.Errorf("failed to create hybrid index: %w", err)
		}
		adapter = NewHybridAdapter(hybridIndex)

	case hybrid.HNSWIndexType:
		// If using HNSW with Parquet, create a ParquetGraph
		if config.Parquet.Directory != "" {
			pgConfig := parquet.ParquetGraphConfig{
				M:        config.Hybrid.M,
				Ml:       config.Hybrid.Ml,
				EfSearch: config.Hybrid.EfSearch,
				Distance: config.Hybrid.Distance,
				Storage:  config.Parquet,
			}
			pg, err := parquet.NewParquetGraph[K](pgConfig)
			if err != nil {
				return nil, fmt.Errorf("failed to create parquet graph: %w", err)
			}
			adapter = NewParquetAdapter(pg)
		} else {
			// Otherwise, create a standard HNSW graph
			graph := hnsw.NewGraph[K]()
			graph.M = config.Hybrid.M
			graph.Ml = config.Hybrid.Ml
			graph.EfSearch = config.Hybrid.EfSearch
			graph.Distance = config.Hybrid.Distance
			adapter = NewHNSWAdapter(graph)
		}

	default:
		// Default to hybrid index
		hybridIndex, err := hybrid.NewHybridIndex[K](config.Hybrid)
		if err != nil {
			return nil, fmt.Errorf("failed to create hybrid index: %w", err)
		}
		adapter = NewHybridAdapter(hybridIndex)
	}

	// Create metadata store
	metaStore := meta.NewMemoryMetadataStore[K]()

	// Create facet store
	facetStore := facets.NewMemoryFacetStore[K]()

	// Load metadata and facets from storage if available
	if storage != nil {
		// Load metadata
		loadedMetaStore, err := storage.LoadMetadata()
		if err != nil {
			return nil, fmt.Errorf("failed to load metadata: %w", err)
		}
		if loadedMetaStore != nil {
			metaStore = loadedMetaStore
		}

		// Load facets
		loadedFacetStore, err := storage.LoadFacets()
		if err != nil {
			return nil, fmt.Errorf("failed to load facets: %w", err)
		}
		if loadedFacetStore != nil {
			facetStore = loadedFacetStore
		}
	}

	db := &VectorDB[K]{
		index:      adapter,
		metaStore:  metaStore,
		facetStore: facetStore,
		storage:    storage,
		config:     config,
	}

	return db, nil
}

// Add adds a vector to the database
func (db *VectorDB[K]) Add(key K, vector []float32, metadata interface{}, facetList []facets.Facet) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	// Add to index
	if err := db.index.Add(key, vector); err != nil {
		return fmt.Errorf("failed to add vector to index: %w", err)
	}

	// Add metadata if provided
	if metadata != nil {
		var rawMetadata json.RawMessage
		var err error

		switch m := metadata.(type) {
		case json.RawMessage:
			rawMetadata = m
		case []byte:
			if !json.Valid(m) {
				return errors.New("invalid JSON metadata")
			}
			rawMetadata = m
		case string:
			if !json.Valid([]byte(m)) {
				return errors.New("invalid JSON metadata string")
			}
			rawMetadata = json.RawMessage(m)
		default:
			rawMetadata, err = json.Marshal(metadata)
			if err != nil {
				return fmt.Errorf("failed to marshal metadata: %w", err)
			}
		}

		// Add to metadata store
		if err := db.metaStore.Add(key, rawMetadata); err != nil {
			return fmt.Errorf("failed to add metadata: %w", err)
		}

		// Save to storage if available
		if db.storage != nil {
			if err := db.storage.SaveMetadata(key, rawMetadata); err != nil {
				return fmt.Errorf("failed to save metadata: %w", err)
			}
		}
	}

	// Add facets if provided
	if len(facetList) > 0 {
		node := hnsw.MakeNode(key, vector)
		facetedNode := facets.NewFacetedNode(node, facetList)

		// Add to facet store
		if err := db.facetStore.Add(facetedNode); err != nil {
			return fmt.Errorf("failed to add facets: %w", err)
		}

		// Save to storage if available
		if db.storage != nil {
			if err := db.storage.SaveFacets(facetedNode); err != nil {
				return fmt.Errorf("failed to save facets: %w", err)
			}
		}
	}

	// Update statistics
	db.statsMu.Lock()
	db.stats.VectorCount++
	db.statsMu.Unlock()

	return nil
}

// BatchAdd adds multiple vectors to the database with optional metadata and facets
func (db *VectorDB[K]) BatchAdd(keys []K, vectors [][]float32, metadataList []interface{}, facetsList [][]facets.Facet) error {
	if len(keys) != len(vectors) {
		return errors.New("keys and vectors must have the same length")
	}

	// Check for empty inputs
	if len(keys) == 0 || len(vectors) == 0 {
		return errors.New("keys and vectors cannot be empty")
	}

	db.mu.Lock()
	defer db.mu.Unlock()

	// Add vectors to the index
	if err := db.index.BatchAdd(keys, vectors); err != nil {
		return fmt.Errorf("failed to add vectors to index: %w", err)
	}

	// Process metadata in batches for better performance
	if metadataList != nil && len(metadataList) == len(keys) {
		rawMetadataList := make([]json.RawMessage, len(metadataList))
		for i, metadata := range metadataList {
			if metadata == nil {
				continue
			}

			var rawMetadata json.RawMessage
			var err error

			switch m := metadata.(type) {
			case json.RawMessage:
				rawMetadata = m
			case []byte:
				if !json.Valid(m) {
					return errors.New("invalid JSON metadata")
				}
				rawMetadata = m
			case string:
				if !json.Valid([]byte(m)) {
					return errors.New("invalid JSON metadata string")
				}
				rawMetadata = json.RawMessage(m)
			default:
				rawMetadata, err = json.Marshal(metadata)
				if err != nil {
					return fmt.Errorf("failed to marshal metadata: %w", err)
				}
			}

			rawMetadataList[i] = rawMetadata
		}

		// Add metadata in batch
		if err := db.metaStore.BatchAdd(keys, rawMetadataList); err != nil {
			return fmt.Errorf("failed to add metadata: %w", err)
		}

		// Save metadata to storage if available
		if db.storage != nil {
			if err := db.storage.BatchSaveMetadata(keys, rawMetadataList); err != nil {
				return fmt.Errorf("failed to save metadata: %w", err)
			}
		}
	}

	// Process facets in batches for better performance
	if facetsList != nil && len(facetsList) == len(keys) {
		// Create a batch of faceted nodes
		facetedNodes := make([]facets.FacetedNode[K], 0, len(keys))

		for i, key := range keys {
			if i >= len(facetsList) || len(facetsList[i]) == 0 {
				continue
			}

			node := hnsw.MakeNode(key, vectors[i])
			facetedNode := facets.NewFacetedNode(node, facetsList[i])
			facetedNodes = append(facetedNodes, facetedNode)

			// Add to facet store
			if err := db.facetStore.Add(facetedNode); err != nil {
				return fmt.Errorf("failed to add facets: %w", err)
			}
		}

		// Save facets to storage if available
		if db.storage != nil && len(facetedNodes) > 0 {
			if err := db.storage.BatchSaveFacets(facetedNodes); err != nil {
				return fmt.Errorf("failed to save facets: %w", err)
			}
		}
	}

	// Update statistics
	db.statsMu.Lock()
	db.stats.VectorCount += len(keys)
	db.statsMu.Unlock()

	return nil
}

// Delete removes a vector from the database
func (db *VectorDB[K]) Delete(key K) bool {
	db.mu.Lock()
	defer db.mu.Unlock()

	// Delete from index
	deleted := db.index.Delete(key)

	// Delete metadata if it exists
	db.metaStore.Delete(key)

	// Delete facets if they exist
	db.facetStore.Delete(key)

	// Update statistics if deleted
	if deleted {
		db.statsMu.Lock()
		db.stats.VectorCount--
		db.statsMu.Unlock()
	}

	return deleted
}

// BatchDelete removes multiple vectors from the database
func (db *VectorDB[K]) BatchDelete(keys []K) []bool {
	db.mu.Lock()
	defer db.mu.Unlock()

	// Delete from index
	results := db.index.BatchDelete(keys)

	// Delete from metadata store
	for i, key := range keys {
		if results[i] {
			db.metaStore.Delete(key)
		}
	}

	// Delete from facet store
	for i, key := range keys {
		if results[i] {
			db.facetStore.Delete(key)
		}
	}

	// Update statistics
	deletedCount := 0
	for _, deleted := range results {
		if deleted {
			deletedCount++
		}
	}

	if deletedCount > 0 {
		db.statsMu.Lock()
		db.stats.VectorCount -= deletedCount
		db.statsMu.Unlock()
	}

	return results
}

// Search performs a vector search with the given query options
func (db *VectorDB[K]) Search(query []float32, options QueryOptions) ([]SearchResult[K], error) {
	if options.Ctx == nil {
		options.Ctx = context.Background()
	}

	// Create a context with timeout if not provided
	_, cancel := context.WithTimeout(options.Ctx, db.config.QueryTimeout)
	defer cancel()

	// Start query timer
	startTime := time.Now()

	db.mu.RLock()
	defer db.mu.RUnlock()

	// Track query statistics
	db.statsMu.Lock()
	db.stats.TotalQueries++
	db.statsMu.Unlock()

	var results []hnsw.Node[K]
	var err error

	// Perform search based on options
	if len(options.NegativeExamples) > 0 {
		// Use negative examples if provided
		if len(options.NegativeExamples) == 1 {
			// Single negative example
			results, err = db.index.SearchWithNegative(query, options.NegativeExamples[0], options.K, options.NegativeWeight)
		} else {
			// Multiple negative examples
			results, err = db.index.SearchWithNegatives(query, options.NegativeExamples, options.K, options.NegativeWeight)
		}
	} else {
		// Regular search
		results, err = db.index.Search(query, options.K)
	}

	if err != nil {
		db.statsMu.Lock()
		db.stats.FailedQueries++
		db.statsMu.Unlock()
		return nil, fmt.Errorf("search failed: %w", err)
	}

	// Apply facet filters if provided
	if len(options.FacetFilters) > 0 {
		filteredResults := make([]hnsw.Node[K], 0, len(results))
		for _, node := range results {
			facetedNode, found := db.facetStore.Get(node.Key)
			if !found {
				continue
			}

			if facetedNode.MatchesAllFilters(options.FacetFilters) {
				filteredResults = append(filteredResults, node)
			}
		}
		results = filteredResults
	}

	// Construct search results with metadata and facets
	searchResults := make([]SearchResult[K], len(results))
	for i, node := range results {
		searchResult := SearchResult[K]{
			Key:      node.Key,
			Vector:   node.Value,
			Distance: 0, // Will be calculated if available
		}

		// Add metadata if available
		metadata, found := db.metaStore.Get(node.Key)
		if found {
			searchResult.Metadata = metadata
		}

		// Add facets if available
		facetedNode, found := db.facetStore.Get(node.Key)
		if found {
			searchResult.Facets = facetedNode.Facets
		}

		searchResults[i] = searchResult
	}

	// Update query statistics
	db.statsMu.Lock()
	db.stats.SuccessfulQueries++
	queryTime := time.Since(startTime)
	db.stats.AverageQueryTime = time.Duration((float64(db.stats.AverageQueryTime)*float64(db.stats.SuccessfulQueries-1) + float64(queryTime)) / float64(db.stats.SuccessfulQueries))
	db.statsMu.Unlock()

	return searchResults, nil
}

// GetStats returns statistics about the database
func (db *VectorDB[K]) GetStats() DBStats {
	db.mu.RLock()
	defer db.mu.RUnlock()

	db.statsMu.RLock()
	defer db.statsMu.RUnlock()

	// Get index stats if available
	if db.index.GetIndexType() == "hybrid" {
		hybridIndex, ok := db.index.GetUnderlyingIndex().(*hybrid.HybridIndex[K])
		if ok {
			db.stats.IndexStats = hybridIndex.GetStats()
		}
	}

	// Return a copy of the stats
	return db.stats
}

// OptimizeStorage performs optimization operations on the storage
// This can include compacting files, removing deleted entries, etc.
func (db *VectorDB[K]) OptimizeStorage() error {
	db.mu.Lock()
	defer db.mu.Unlock()

	if db.storage == nil {
		return nil
	}

	// Optimize storage
	if err := db.storage.OptimizeStorage(); err != nil {
		return fmt.Errorf("failed to optimize storage: %w", err)
	}

	return nil
}

// Backup creates a backup of the database to the specified directory
func (db *VectorDB[K]) Backup(backupDir string) error {
	db.mu.RLock()
	defer db.mu.RUnlock()

	// Check if storage is available
	if db.storage == nil {
		return errors.New("storage is not available, cannot create backup")
	}

	// Create backup directory if it doesn't exist
	if err := os.MkdirAll(backupDir, 0755); err != nil {
		return fmt.Errorf("failed to create backup directory: %w", err)
	}

	// Create a backup configuration file
	configFile := filepath.Join(backupDir, "config.json")

	// Convert config to serializable format
	serializableConfig := db.config.toSerializable()

	configData, err := json.MarshalIndent(serializableConfig, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal configuration: %w", err)
	}
	if err := os.WriteFile(configFile, configData, 0644); err != nil {
		return fmt.Errorf("failed to write configuration file: %w", err)
	}

	// Create a backup statistics file
	statsFile := filepath.Join(backupDir, "stats.json")
	statsData, err := json.MarshalIndent(db.stats, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal statistics: %w", err)
	}
	if err := os.WriteFile(statsFile, statsData, 0644); err != nil {
		return fmt.Errorf("failed to write statistics file: %w", err)
	}

	// Use the storage to backup the data
	if err := db.storage.Backup(backupDir); err != nil {
		return fmt.Errorf("failed to backup data: %w", err)
	}

	return nil
}

// Restore restores the database from the specified backup directory
func (db *VectorDB[K]) Restore(backupDir string) error {
	// Check if backup directory exists
	if _, err := os.Stat(backupDir); os.IsNotExist(err) {
		return fmt.Errorf("backup directory does not exist: %s", backupDir)
	}

	// Read the backup configuration file
	configFile := filepath.Join(backupDir, "config.json")
	if _, err := os.Stat(configFile); os.IsNotExist(err) {
		return fmt.Errorf("backup configuration file does not exist: %s", configFile)
	}
	configData, err := os.ReadFile(configFile)
	if err != nil {
		return fmt.Errorf("failed to read configuration file: %w", err)
	}

	// Parse serializable config
	var serializableConfig SerializableDBConfig
	if err := json.Unmarshal(configData, &serializableConfig); err != nil {
		return fmt.Errorf("failed to unmarshal configuration: %w", err)
	}

	// Convert to DBConfig
	config := serializableConfig.toDBConfig()

	// Read the backup statistics file
	statsFile := filepath.Join(backupDir, "stats.json")
	if _, err := os.Stat(statsFile); !os.IsNotExist(err) {
		statsData, err := os.ReadFile(statsFile)
		if err != nil {
			return fmt.Errorf("failed to read statistics file: %w", err)
		}
		var stats DBStats
		if err := json.Unmarshal(statsData, &stats); err != nil {
			return fmt.Errorf("failed to unmarshal statistics: %w", err)
		}
		db.stats = stats
	}

	// Lock the database for writing
	db.mu.Lock()
	defer db.mu.Unlock()

	// Check if storage is available
	if db.storage == nil {
		return errors.New("storage is not available, cannot restore backup")
	}

	// Use the storage to restore the data
	if err := db.storage.Restore(backupDir); err != nil {
		return fmt.Errorf("failed to restore data: %w", err)
	}

	// Reload metadata and facets
	metaStore, err := db.storage.LoadMetadata()
	if err != nil {
		return fmt.Errorf("failed to load metadata: %w", err)
	}
	db.metaStore = metaStore

	facetStore, err := db.storage.LoadFacets()
	if err != nil {
		return fmt.Errorf("failed to load facets: %w", err)
	}
	db.facetStore = facetStore

	// Update configuration
	db.config = config

	// Recreate the index with the restored data
	// This is necessary because the index is in-memory and needs to be rebuilt
	var adapter *IndexAdapter[K]

	switch config.Hybrid.Type {
	case hybrid.HybridIndexType:
		// Create a hybrid index
		hybridIndex, err := hybrid.NewHybridIndex[K](config.Hybrid)
		if err != nil {
			return fmt.Errorf("failed to create hybrid index: %w", err)
		}
		adapter = NewHybridAdapter(hybridIndex)

	case hybrid.HNSWIndexType:
		// If using HNSW with Parquet, create a ParquetGraph
		if config.Parquet.Directory != "" {
			pgConfig := parquet.ParquetGraphConfig{
				M:        config.Hybrid.M,
				Ml:       config.Hybrid.Ml,
				EfSearch: config.Hybrid.EfSearch,
				Distance: config.Hybrid.Distance,
				Storage:  config.Parquet,
			}
			pg, err := parquet.NewParquetGraph[K](pgConfig)
			if err != nil {
				return fmt.Errorf("failed to create parquet graph: %w", err)
			}
			adapter = NewParquetAdapter(pg)
		} else {
			// Otherwise, create a standard HNSW graph
			graph := hnsw.NewGraph[K]()
			graph.M = config.Hybrid.M
			graph.Ml = config.Hybrid.Ml
			graph.EfSearch = config.Hybrid.EfSearch
			graph.Distance = config.Hybrid.Distance
			adapter = NewHNSWAdapter(graph)
		}

	default:
		// Default to hybrid index
		hybridIndex, err := hybrid.NewHybridIndex[K](config.Hybrid)
		if err != nil {
			return fmt.Errorf("failed to create hybrid index: %w", err)
		}
		adapter = NewHybridAdapter(hybridIndex)
	}

	// Replace the old index with the new one
	db.index = adapter

	// Rebuild the index from facets
	// We can access the facet store directly to get all the vectors
	if facetStore, ok := db.facetStore.(*facets.MemoryFacetStore[K]); ok {
		// Use Filter with no filters to get all nodes
		for _, facetedNode := range facetStore.Filter(nil) {
			// Add the vector to the index
			db.index.Add(facetedNode.Node.Key, facetedNode.Node.Value)
		}
	}

	return nil
}

// Close releases resources used by the database
func (db *VectorDB[K]) Close() error {
	db.mu.Lock()
	defer db.mu.Unlock()

	// Close the index
	if err := db.index.Close(); err != nil {
		return fmt.Errorf("failed to close index: %w", err)
	}

	// Close storage if available
	if db.storage != nil {
		if err := db.storage.Close(); err != nil {
			return fmt.Errorf("failed to close storage: %w", err)
		}
	}

	return nil
}

// Analyze returns analytics data about the database
func (db *VectorDB[K]) Analyze() (*hnsw.GraphQualityMetrics, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	// Check if the index is a standard HNSW graph
	if db.index.GetIndexType() != "hnsw" {
		return nil, errors.New("analytics are only available for standard HNSW graphs")
	}

	// Get the underlying graph
	graph, ok := db.index.GetUnderlyingIndex().(*hnsw.Graph[K])
	if !ok {
		return nil, errors.New("failed to get underlying HNSW graph")
	}

	// Create an analyzer
	analyzer := &hnsw.Analyzer[K]{Graph: graph}

	// Get quality metrics
	metrics := analyzer.QualityMetrics()

	return &metrics, nil
}

// DefaultQueryOptions returns the default query options for this database
func (db *VectorDB[K]) DefaultQueryOptions() QueryOptions {
	return DefaultQueryOptions()
}
