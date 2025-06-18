package core

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/TFMV/quiver/pkg/hnsw"
	"github.com/TFMV/quiver/pkg/hybrid"
	"github.com/TFMV/quiver/pkg/metrics"
	"github.com/TFMV/quiver/pkg/persistence"
	"github.com/TFMV/quiver/pkg/types"
	"github.com/TFMV/quiver/pkg/vectortypes"
)

var (
	ErrCollectionExists     = errors.New("collection already exists")
	ErrCollectionNotFound   = errors.New("collection not found")
	ErrInvalidConfiguration = errors.New("invalid configuration")
	ErrBackupFailed         = errors.New("backup failed")
	ErrRestoreFailed        = errors.New("restore failed")
)

// DBOptions contains configuration parameters for the Quiver database
type DBOptions struct {
	// Base path for database storage
	StoragePath string
	// Whether to enable metrics collection
	EnableMetrics bool
	// Whether to enable persistence
	EnablePersistence bool
	// Persistence flush interval
	FlushInterval time.Duration
	// Default HNSW configuration for new collections
	DefaultHNSWConfig hnsw.Config
	// Hybrid search configuration
	EnableHybridSearch bool
	// Hybrid index configuration
	HybridConfig hybrid.IndexConfig
}

// DefaultDBOptions returns the default configuration for Quiver
func DefaultDBOptions() DBOptions {
	return DBOptions{
		StoragePath:       "./data",
		EnableMetrics:     true,
		EnablePersistence: true,
		FlushInterval:     5 * time.Minute,
		DefaultHNSWConfig: hnsw.Config{
			M:              16,
			MaxM0:          32,
			EfConstruction: 200,
			EfSearch:       100,
			MaxLevel:       16,
			DistanceFunc:   hnsw.CosineDistanceFunc,
		},
		EnableHybridSearch: true,
		HybridConfig:       hybrid.DefaultIndexConfig(),
	}
}

// DB represents the Quiver vector database
type DB struct {
	// Map of collection name to collection instance
	collections map[string]*Collection
	// Database options
	options DBOptions
	// Metrics collector
	metrics *metrics.Collector
	// Persistence manager
	persistenceManager *persistence.Manager
	// Lock for thread safety
	mu sync.RWMutex
}

// NewDB creates a new Quiver database with the given options
func NewDB(options DBOptions) (*DB, error) {
	// Validate options
	if options.EnablePersistence && options.StoragePath == "" {
		return nil, ErrInvalidConfiguration
	}

	// Create storage directory if it doesn't exist and persistence is enabled
	if options.EnablePersistence && options.StoragePath != "" {
		if err := os.MkdirAll(options.StoragePath, 0755); err != nil {
			return nil, fmt.Errorf("failed to create storage directory: %w", err)
		}
	}

	// Initialize database
	db := &DB{
		collections: make(map[string]*Collection),
		options:     options,
		mu:          sync.RWMutex{},
	}

	// Initialize metrics collector if enabled
	if options.EnableMetrics {
		db.metrics = metrics.NewCollector(true)
	}

	// Initialize persistence manager if enabled
	if options.EnablePersistence {
		var err error
		db.persistenceManager, err = persistence.NewManager(options.StoragePath, options.FlushInterval)
		if err != nil {
			return nil, fmt.Errorf("failed to initialize persistence: %w", err)
		}

		// Set up persistence callback
		db.persistenceManager.SetGetCollectionCallback(func(name string) (persistence.Persistable, error) {
			col, err := db.GetCollection(name)
			if err != nil {
				return nil, err
			}
			return col, nil
		})
	}

	// Load existing collections if persistence is enabled
	if options.EnablePersistence {
		if err := db.loadCollections(); err != nil {
			return nil, fmt.Errorf("failed to load collections: %w", err)
		}
	}

	return db, nil
}

// loadCollections loads existing collections from storage
func (db *DB) loadCollections() error {
	// List collection directories
	entries, err := os.ReadDir(db.options.StoragePath)
	if err != nil {
		return err
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		collectionName := entry.Name()
		collectionPath := filepath.Join(db.options.StoragePath, collectionName)

		// Check if this is a valid collection directory
		configPath := filepath.Join(collectionPath, "config.json")
		if _, err := os.Stat(configPath); err != nil {
			continue // Not a valid collection directory
		}

		// Load collection metadata
		config, err := persistence.LoadCollectionConfig(configPath)
		if err != nil {
			return fmt.Errorf("failed to load collection config for %s: %w", collectionName, err)
		}

		// Create HNSW index with the appropriate distance function
		hnswConfig := db.options.DefaultHNSWConfig

		// Update distance function based on the stored config
		if config.DistanceFunc == "euclidean" {
			hnswConfig.DistanceFunc = hnsw.EuclideanDistanceFunc
		} else if config.DistanceFunc == "dot_product" {
			hnswConfig.DistanceFunc = hnsw.DotProductDistanceFunc
		} else {
			// Default to cosine
			hnswConfig.DistanceFunc = hnsw.CosineDistanceFunc
		}

		// Create adapter
		adapter := hnsw.NewAdapter(hnswConfig)

		// Create collection
		collection := NewCollection(collectionName, config.Dimension, adapter)

		// Load vectors and metadata
		if err := db.persistenceManager.LoadCollection(collection, collectionPath); err != nil {
			return fmt.Errorf("failed to load collection %s: %w", collectionName, err)
		}

		// Add collection to DB
		db.collections[collectionName] = collection
	}

	return nil
}

// GetName returns the name of the collection (for persistence)
func (c *Collection) GetName() string {
	return c.Name
}

// GetDimension returns the dimension of vectors in the collection (for persistence)
func (c *Collection) GetDimension() int {
	return c.Dimension
}

// GetVectors returns all vectors in the collection for persistence
func (c *Collection) GetVectors() []persistence.VectorRecord {
	c.RLock()
	defer c.RUnlock()

	records := make([]persistence.VectorRecord, 0, len(c.Vectors))
	for id, vector := range c.Vectors {
		var metadata map[string]string
		if metaJSON, exists := c.Metadata[id]; exists && len(metaJSON) > 0 {
			// Convert JSON metadata to map[string]string
			var metaMap map[string]interface{}
			if err := json.Unmarshal(metaJSON, &metaMap); err == nil {
				metadata = make(map[string]string)
				for k, v := range metaMap {
					metadata[k] = fmt.Sprintf("%v", v)
				}
			}
		}

		records = append(records, persistence.VectorRecord{
			ID:       id,
			Vector:   vector,
			Metadata: metadata,
		})
	}

	return records
}

// AddVector adds a vector to the collection (for persistence)
func (c *Collection) AddVector(id string, vector []float32, metadata map[string]string) error {
	// Convert metadata map to JSON
	var metadataJSON json.RawMessage
	if len(metadata) > 0 {
		metaMap := make(map[string]interface{})
		for k, v := range metadata {
			metaMap[k] = v
		}
		if data, err := json.Marshal(metaMap); err == nil {
			metadataJSON = data
		}
	}

	// Add to collection
	return c.Add(id, vector, metadataJSON)
}

// GetDistanceFunction returns the distance function name for serialization
func (c *Collection) GetDistanceFunction() string {
	// This is a simple way to determine the distance function type
	// In a real implementation, you might want to store this info in the collection
	return "cosine" // Default to cosine
}

// NewPersistableAdapter creates a new adapter that implements the Persistable interface
func NewPersistableAdapter(collection *Collection) persistence.Persistable {
	return collection
}

// Close closes the database and performs cleanup
func (db *DB) Close() error {
	// Flush data to disk if persistence is enabled
	if db.options.EnablePersistence && db.persistenceManager != nil {
		for name, collection := range db.collections {
			if err := db.persistenceManager.FlushCollection(collection,
				filepath.Join(db.options.StoragePath, name)); err != nil {
				return fmt.Errorf("failed to flush collection %s: %w", name, err)
			}
		}
	}

	return nil
}

// CreateCollection creates a new vector collection with the given name and parameters
func (db *DB) CreateCollection(name string, dimension int, distanceFunc vectortypes.Surface[vectortypes.F32]) (*Collection, error) {
	db.mu.Lock()
	defer db.mu.Unlock()

	// Check if collection already exists
	if _, exists := db.collections[name]; exists {
		return nil, fmt.Errorf("%w: %s", ErrCollectionExists, name)
	}

	// Validate dimension
	if dimension <= 0 {
		return nil, ErrInvalidDimension
	}

	var index Index
	// Default distance function name for persistence
	var distFuncName string = "cosine"

	// Determine if we should use hybrid search or standard HNSW
	if db.options.EnableHybridSearch {
		// Configure the hybrid index
		hybridConfig := db.options.HybridConfig

		// Update the distance function based on the provided one
		if basicSurface, ok := distanceFunc.(vectortypes.BasicSurface); ok {
			hybridConfig.DistanceFunc = basicSurface.DistFunc

			// Determine the distance function name for persistence
			distFuncStr := fmt.Sprintf("%p", basicSurface.DistFunc)
			euclideanStr := fmt.Sprintf("%p", vectortypes.EuclideanDistance)
			dotProductStr := fmt.Sprintf("%p", vectortypes.DotProductDistance)

			if distFuncStr == euclideanStr {
				distFuncName = "euclidean"
			} else if distFuncStr == dotProductStr {
				distFuncName = "dot_product"
			}
		} else {
			// Fall back to cosine distance if we can't extract the function
			hybridConfig.DistanceFunc = vectortypes.CosineDistance
		}

		// Create hybrid index
		hybridIndex := hybrid.NewHybridIndex(hybridConfig)

		// Create a wrapper that implements the Index interface
		index = &HybridIndexWrapper{
			hybridIndex: hybridIndex,
		}
	} else {
		// Create HNSW index
		hnswConfig := db.options.DefaultHNSWConfig

		if basicSurface, ok := distanceFunc.(vectortypes.BasicSurface); ok {
			// Update the distance function by wrapping the BasicSurface.Distance method
			distVectorFunc := basicSurface.DistFunc
			hnswConfig.DistanceFunc = func(a, b []float32) (float32, error) {
				return distVectorFunc(a, b), nil
			}

			// Determine the distance function name based on string comparison
			distFuncStr := fmt.Sprintf("%p", distVectorFunc)
			euclideanStr := fmt.Sprintf("%p", vectortypes.EuclideanDistance)
			dotProductStr := fmt.Sprintf("%p", vectortypes.DotProductDistance)

			if distFuncStr == euclideanStr {
				distFuncName = "euclidean"
			} else if distFuncStr == dotProductStr {
				distFuncName = "dot_product"
			}
		} else {
			// If we can't determine the type, fall back to cosine distance
			hnswConfig.DistanceFunc = hnsw.CosineDistanceFunc
		}

		index = hnsw.NewAdapter(hnswConfig)
	}

	// Create collection
	collection := NewCollection(name, dimension, index)

	// Initialize collection directory if persistence is enabled
	if db.options.EnablePersistence && db.persistenceManager != nil {
		collectionPath := filepath.Join(db.options.StoragePath, name)
		if err := os.MkdirAll(collectionPath, 0755); err != nil {
			return nil, fmt.Errorf("failed to create collection directory: %w", err)
		}

		// Save collection config
		config := persistence.CollectionConfig{
			Name:         name,
			Dimension:    dimension,
			DistanceFunc: distFuncName,
			CreatedAt:    time.Now(),
		}

		if err := persistence.SaveCollectionConfig(config, filepath.Join(collectionPath, "config.json")); err != nil {
			return nil, fmt.Errorf("failed to save collection config: %w", err)
		}
	}

	// Add collection to DB
	db.collections[name] = collection

	return collection, nil
}

// GetCollection retrieves a collection by name
func (db *DB) GetCollection(name string) (*Collection, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	collection, exists := db.collections[name]
	if !exists {
		return nil, fmt.Errorf("%w: %s", ErrCollectionNotFound, name)
	}

	return collection, nil
}

// ListCollections returns a list of all collections
func (db *DB) ListCollections() []string {
	db.mu.RLock()
	defer db.mu.RUnlock()

	collections := make([]string, 0, len(db.collections))
	for name := range db.collections {
		collections = append(collections, name)
	}

	return collections
}

// DeleteCollection deletes a collection by name
func (db *DB) DeleteCollection(name string) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	_, exists := db.collections[name]
	if !exists {
		return fmt.Errorf("%w: %s", ErrCollectionNotFound, name)
	}

	// Delete collection data from disk if persistence is enabled
	if db.options.EnablePersistence && db.persistenceManager != nil {
		collectionPath := filepath.Join(db.options.StoragePath, name)
		if err := os.RemoveAll(collectionPath); err != nil {
			return fmt.Errorf("failed to delete collection directory: %w", err)
		}
	}

	// Remove collection from memory
	delete(db.collections, name)

	// Log the collection deletion if metrics enabled
	if db.options.EnableMetrics && db.metrics != nil {
		// Record the event
		db.metrics.RecordSystemMetrics(50, 100) // Placeholder metrics
	}

	return nil
}

// BackupDatabase creates a snapshot of the entire database
func (db *DB) BackupDatabase(backupPath string) error {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if !db.options.EnablePersistence || db.persistenceManager == nil {
		return fmt.Errorf("%w: persistence must be enabled to create backups", ErrBackupFailed)
	}

	// Create backup directory
	if err := os.MkdirAll(backupPath, 0755); err != nil {
		return fmt.Errorf("%w: failed to create backup directory: %v", ErrBackupFailed, err)
	}

	// Flush all collections to disk before backing up
	for name, collection := range db.collections {
		collectionPath := filepath.Join(db.options.StoragePath, name)
		if err := db.persistenceManager.FlushCollection(collection, collectionPath); err != nil {
			return fmt.Errorf("%w: failed to flush collection %s: %v", ErrBackupFailed, name, err)
		}
	}

	// Create the backup (use persistence manager to copy the data)
	if err := db.persistenceManager.CreateBackup(db.options.StoragePath, backupPath); err != nil {
		return fmt.Errorf("%w: %v", ErrBackupFailed, err)
	}

	return nil
}

// RestoreDatabase restores the database from a snapshot
func (db *DB) RestoreDatabase(backupPath string) error {
	// Lock for write as we're updating all collections
	db.mu.Lock()
	defer db.mu.Unlock()

	if !db.options.EnablePersistence || db.persistenceManager == nil {
		return fmt.Errorf("%w: persistence must be enabled to restore backups", ErrRestoreFailed)
	}

	// Verify the backup directory exists
	if _, err := os.Stat(backupPath); err != nil {
		return fmt.Errorf("%w: backup not found at %s", ErrRestoreFailed, backupPath)
	}

	// Clear existing collections
	db.collections = make(map[string]*Collection)

	// Restore from backup
	if err := db.persistenceManager.RestoreBackup(backupPath, db.options.StoragePath); err != nil {
		return fmt.Errorf("%w: %v", ErrRestoreFailed, err)
	}

	// Load collections from the restored data
	if err := db.loadCollections(); err != nil {
		return fmt.Errorf("%w: failed to load collections after restore: %v", ErrRestoreFailed, err)
	}

	return nil
}

// GetMetrics returns the current performance metrics
func (db *DB) GetMetrics() (*metrics.PerformanceMetrics, error) {
	if !db.options.EnableMetrics || db.metrics == nil {
		return nil, errors.New("metrics collection is not enabled")
	}

	metrics := db.metrics.GetRecentMetrics()
	return &metrics, nil
}

// Search performs a search across collections with detailed results
func (db *DB) Search(collectionName string, request types.SearchRequest) (types.SearchResponse, error) {
	// Start timing the search operation
	startTime := time.Now()

	// Get the specified collection
	collection, err := db.GetCollection(collectionName)
	if err != nil {
		return types.SearchResponse{}, err
	}

	// If metrics are enabled, record this search operation
	if db.options.EnableMetrics && db.metrics != nil {
		// Record the search operation (dimension and number of results requested)
		defer func() {
			searchTime := time.Since(startTime).Milliseconds()
			db.metrics.RecordLatency(collectionName, "vector_search", float64(searchTime))
		}()
	}

	// Delegate to the collection's search
	return collection.Search(request)
}

// HybridIndexWrapper wraps a hybrid index to implement the core.Index interface
type HybridIndexWrapper struct {
	hybridIndex *hybrid.HybridIndex
}

// Insert adds a vector to the index
func (w *HybridIndexWrapper) Insert(id string, vector vectortypes.F32) error {
	return w.hybridIndex.Insert(id, vector)
}

// Delete removes a vector from the index
func (w *HybridIndexWrapper) Delete(id string) error {
	return w.hybridIndex.Delete(id)
}

// Search performs a similarity search
func (w *HybridIndexWrapper) Search(query vectortypes.F32, k int) ([]types.BasicSearchResult, error) {
	return w.hybridIndex.Search(query, k)
}

// Size returns the number of vectors in the index
func (w *HybridIndexWrapper) Size() int {
	return w.hybridIndex.Size()
}

// BatchInsert adds multiple vectors in a batch operation
func (w *HybridIndexWrapper) BatchInsert(vectors map[string]vectortypes.F32) error {
	return w.hybridIndex.InsertBatch(vectors)
}

// BatchDelete removes multiple vectors in a batch operation
func (w *HybridIndexWrapper) BatchDelete(ids []string) error {
	return w.hybridIndex.DeleteBatch(ids)
}

// BatchSearch performs multiple searches in parallel
func (w *HybridIndexWrapper) BatchSearch(req hybrid.BatchSearchRequest) (hybrid.BatchSearchResponse, error) {
	return w.hybridIndex.BatchSearch(req)
}

// GetHybridConfig returns the configuration for hybrid search
func (db *DB) GetHybridConfig() hybrid.IndexConfig {
	return db.options.HybridConfig
}

// SetHybridConfig updates the configuration for hybrid search
func (db *DB) SetHybridConfig(config hybrid.IndexConfig) {
	db.mu.Lock()
	defer db.mu.Unlock()
	db.options.HybridConfig = config
}

// BatchInsertRequest contains data for a batch insertion operation
type BatchInsertRequest struct {
	// Collection name
	Collection string
	// Vectors to insert, mapped by ID
	Vectors map[string]vectortypes.F32
	// Metadata for each vector, mapped by ID (optional)
	Metadata map[string]json.RawMessage
}

// BatchInsert inserts multiple vectors into a collection in a single operation
func (db *DB) BatchInsert(request BatchInsertRequest) error {
	collection, err := db.GetCollection(request.Collection)
	if err != nil {
		return err
	}

	// If collection uses the hybrid index, use its batch insert for better performance
	if hybridWrapper, ok := collection.Index.(*HybridIndexWrapper); ok {
		return hybridWrapper.hybridIndex.InsertBatch(request.Vectors)
	}

	// Otherwise, use collection's batch add
	vectors := make([]vectortypes.Vector, 0, len(request.Vectors))
	for id, vector := range request.Vectors {
		var metadata json.RawMessage
		if request.Metadata != nil {
			if meta, exists := request.Metadata[id]; exists {
				metadata = meta
			}
		}

		vectors = append(vectors, vectortypes.Vector{
			ID:       id,
			Values:   vector,
			Metadata: metadata,
		})
	}

	return collection.AddBatch(vectors)
}

// BatchDeleteRequest contains data for a batch deletion operation
type BatchDeleteRequest struct {
	// Collection name
	Collection string
	// Vector IDs to delete
	IDs []string
}

// BatchDelete removes multiple vectors from a collection in a single operation
func (db *DB) BatchDelete(request BatchDeleteRequest) error {
	collection, err := db.GetCollection(request.Collection)
	if err != nil {
		return err
	}

	// If collection uses the hybrid index, use its batch delete for better performance
	if hybridWrapper, ok := collection.Index.(*HybridIndexWrapper); ok {
		return hybridWrapper.hybridIndex.DeleteBatch(request.IDs)
	}

	// Otherwise, use collection's batch delete
	return collection.DeleteBatch(request.IDs)
}

// BatchSearchRequest contains data for a batch search operation
type BatchSearchRequest struct {
	// Collection name
	Collection string
	// Multiple search requests to process
	Requests []types.SearchRequest
}

// BatchSearchResponse contains results from multiple searches
type BatchSearchResponse struct {
	// Results for each search request
	Results []types.SearchResponse
	// Total time spent processing all requests
	TotalTimeMs float64
	// Average time per search
	AvgTimeMs float64
}

// BatchSearch performs multiple searches in parallel
func (db *DB) BatchSearch(request BatchSearchRequest) (BatchSearchResponse, error) {
	// Start timing the batch search operation
	startTime := time.Now()

	collection, err := db.GetCollection(request.Collection)
	if err != nil {
		return BatchSearchResponse{}, err
	}

	// If hybrid index supports batch search, use it for better performance
	if hybridWrapper, ok := collection.Index.(*HybridIndexWrapper); ok {
		// Check if we can use the optimized batch search
		if len(request.Requests) > 0 && allSameKAndOptions(request.Requests) {
			// Extract queries for the batch search
			queries := make([]vectortypes.F32, len(request.Requests))
			for i, req := range request.Requests {
				queries[i] = req.Vector
			}

			// Use the K value from the first request
			k := request.Requests[0].TopK

			// Perform batch search
			batchRequest := hybrid.BatchSearchRequest{
				Queries:      queries,
				K:            k,
				IncludeStats: true,
			}

			batchResponse, err := hybridWrapper.hybridIndex.BatchSearch(batchRequest)
			if err != nil {
				return BatchSearchResponse{}, err
			}

			// Convert results to SearchResponse format
			results := make([]types.SearchResponse, len(batchResponse.Results))
			for i, result := range batchResponse.Results {
				searchItems := make([]types.SearchResultItem, len(result))
				for j, item := range result {
					searchItems[j] = types.SearchResultItem{
						ID:       item.ID,
						Distance: item.Distance,
						Score:    1.0 - item.Distance,
					}

					// Add vector and metadata if requested
					if request.Requests[i].Options.IncludeVectors {
						if vec, exists := collection.Vectors[item.ID]; exists {
							searchItems[j].Vector = vec
						}
					}

					if request.Requests[i].Options.IncludeMetadata {
						if meta, exists := collection.Metadata[item.ID]; exists {
							searchItems[j].Metadata = meta
						}
					}
				}

				// Create response
				results[i] = types.SearchResponse{
					Results: searchItems,
					Metadata: types.SearchResultMetadata{
						TotalCount: len(result),
						SearchTime: float64(batchResponse.SearchTimes[i].Microseconds()) / 1000.0,
						IndexSize:  collection.Count(),
						IndexName:  collection.Name,
						Timestamp:  time.Now(),
					},
				}

				// Include query vector if requested
				if request.Requests[i].Options.IncludeVectors {
					results[i].Query = request.Requests[i].Vector
				}
			}

			// Calculate total and average time
			totalTime := time.Since(startTime)
			totalTimeMs := float64(totalTime.Microseconds()) / 1000.0

			return BatchSearchResponse{
				Results:     results,
				TotalTimeMs: totalTimeMs,
				AvgTimeMs:   totalTimeMs / float64(len(request.Requests)),
			}, nil
		}
	}

	// Fallback to parallel individual searches
	results := make([]types.SearchResponse, len(request.Requests))
	var wg sync.WaitGroup
	var errMu sync.Mutex
	var firstErr error

	for i, req := range request.Requests {
		wg.Add(1)
		go func(index int, searchReq types.SearchRequest) {
			defer wg.Done()

			res, err := collection.Search(searchReq)
			if err != nil {
				errMu.Lock()
				if firstErr == nil {
					firstErr = fmt.Errorf("search %d failed: %w", index, err)
				}
				errMu.Unlock()
				return
			}

			results[index] = res
		}(i, req)
	}

	wg.Wait()

	if firstErr != nil {
		return BatchSearchResponse{}, firstErr
	}

	// Calculate total and average time
	totalTime := time.Since(startTime)
	totalTimeMs := float64(totalTime.Microseconds()) / 1000.0

	return BatchSearchResponse{
		Results:     results,
		TotalTimeMs: totalTimeMs,
		AvgTimeMs:   totalTimeMs / float64(len(request.Requests)),
	}, nil
}

// allSameKAndOptions checks if all search requests have the same K value and options
func allSameKAndOptions(requests []types.SearchRequest) bool {
	if len(requests) <= 1 {
		return true
	}

	k := requests[0].TopK
	includeVectors := requests[0].Options.IncludeVectors
	includeMetadata := requests[0].Options.IncludeMetadata
	exactSearch := requests[0].Options.ExactSearch

	for i := 1; i < len(requests); i++ {
		if requests[i].TopK != k ||
			requests[i].Options.IncludeVectors != includeVectors ||
			requests[i].Options.IncludeMetadata != includeMetadata ||
			requests[i].Options.ExactSearch != exactSearch {
			return false
		}
	}

	return true
}
