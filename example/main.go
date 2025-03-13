// Package main provides a comprehensive example of using Quiver
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/TFMV/hnsw"
	"github.com/TFMV/hnsw/hnsw-extensions/facets"
	"github.com/TFMV/hnsw/hnsw-extensions/hybrid"
	db "github.com/TFMV/quiver"
	"github.com/TFMV/quiver/adaptive"
)

func main() {
	// Initialize logging
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	log.Println("Starting Quiver comprehensive example")

	// Create a context that can be cancelled
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle signals for graceful shutdown
	signalCh := make(chan os.Signal, 1)
	signal.Notify(signalCh, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-signalCh
		log.Println("Received shutdown signal")
		cancel()
	}()

	// Create a temporary directory for examples
	tempDir, err := os.MkdirTemp("", "quiver-example")
	if err != nil {
		log.Fatalf("Failed to create temporary directory: %v", err)
	}
	defer os.RemoveAll(tempDir)
	log.Printf("Using temporary directory: %s", tempDir)

	// Initialize Adaptive Parameter Tuning (APT)
	log.Println("\n=== Initializing Adaptive Parameter Tuning (APT) ===")
	log.Println("APT is enabled by default in Quiver and requires no configuration.")
	log.Println("It automatically optimizes HNSW parameters based on your workload.")

	// Initialize APT system
	aptDir := filepath.Join(tempDir, "apt")
	if err := os.MkdirAll(aptDir, 0755); err != nil {
		log.Printf("Warning: Failed to create APT directory: %v", err)
	}
	if err := adaptive.Initialize(aptDir, true); err != nil {
		log.Printf("Warning: Failed to initialize APT: %v", err)
	}

	// Demonstrate different index types
	demoHNSWIndex(ctx, tempDir)
	demoParquetIndex(tempDir)
	demoHybridIndex(ctx, tempDir)

	// Demonstrate APT features
	demoAPT()

	// Shutdown APT system
	adaptive.Shutdown()

	log.Println("\nQuiver comprehensive example completed successfully")
}

// demoHNSWIndex demonstrates the HNSW index type
func demoHNSWIndex(ctx context.Context, baseDir string) {
	log.Println("\n=== HNSW Index Demonstration ===")

	// Create a database configuration for HNSW
	config := db.DefaultDBConfig()
	config.BaseDir = baseDir + "/hnsw"
	config.Hybrid.Type = hybrid.HNSWIndexType
	config.Hybrid.Distance = hnsw.CosineDistance
	config.Hybrid.M = 16
	config.Hybrid.EfSearch = 100

	// Create a Quiver database with HNSW index
	database, err := db.NewVectorDB[uint64](config)
	if err != nil {
		log.Fatalf("Failed to create HNSW database: %v", err)
	}
	defer database.Close()
	log.Println("HNSW database created successfully")

	// Add vectors with metadata and facets
	addVectorsWithMetadata(database)

	// Perform different types of searches
	basicSearch(database)
	searchWithNegatives(database)
	searchWithFacets(database)
	searchWithMetadataFilters(database)

	// Demonstrate batch operations
	demoBatchOperations(database)

	log.Println("HNSW index demonstration completed")
}

// demoParquetIndex demonstrates the Parquet index type
func demoParquetIndex(baseDir string) {
	log.Println("\n=== Parquet Index Demonstration ===")

	// Create a database configuration for Parquet
	config := db.DefaultDBConfig()
	config.BaseDir = baseDir + "/parquet"
	config.Hybrid.Type = hybrid.HNSWIndexType // We'll use HNSW with Parquet storage
	config.Hybrid.Distance = hnsw.CosineDistance
	config.Hybrid.M = 16
	config.Hybrid.EfSearch = 100

	// Configure Parquet storage
	config.Parquet.Directory = baseDir + "/parquet/data"
	// Parquet configuration is handled internally

	// Create a Quiver database with Parquet storage
	database, err := db.NewVectorDB[uint64](config)
	if err != nil {
		log.Fatalf("Failed to create Parquet database: %v", err)
	}
	defer database.Close()
	log.Println("Parquet database created successfully")

	// Add vectors with metadata and facets
	addVectorsWithMetadata(database)

	// Perform basic search
	basicSearch(database)

	// Demonstrate backup and restore
	demoBackupRestore(database, baseDir+"/parquet/backup")

	log.Println("Parquet index demonstration completed")
}

// demoHybridIndex demonstrates the Hybrid index type
func demoHybridIndex(ctx context.Context, baseDir string) {
	log.Println("\n=== Hybrid Index Demonstration ===")

	// Create a database configuration for Hybrid index
	config := db.DefaultDBConfig()
	config.BaseDir = baseDir + "/hybrid"
	config.Hybrid.Type = hybrid.HybridIndexType
	config.Hybrid.Distance = hnsw.CosineDistance
	config.Hybrid.M = 16
	config.Hybrid.EfSearch = 100

	// Hybrid-specific settings
	config.Hybrid.ExactThreshold = 1000 // Use exact search for datasets smaller than 1000 vectors
	// Note: LSH settings are configured internally by the hybrid index

	// Create a Quiver database with Hybrid index
	database, err := db.NewVectorDB[uint64](config)
	if err != nil {
		log.Fatalf("Failed to create Hybrid database: %v", err)
	}
	defer database.Close()
	log.Println("Hybrid database created successfully")

	// Add vectors with metadata and facets
	addVectorsWithMetadata(database)

	// Perform basic search
	basicSearch(database)

	// Perform search with complex query options
	searchWithComplexOptions(database)

	log.Println("Hybrid index demonstration completed")
}

// addVectorsWithMetadata adds sample vectors with metadata and facets
func addVectorsWithMetadata(database *db.VectorDB[uint64]) {
	log.Println("Adding vectors with metadata and facets...")

	// Create 1000 random vectors
	numVectors := 1000
	dimension := 5 // Using small dimension for example purposes

	for i := 0; i < numVectors; i++ {
		// Create a random vector
		vector := make([]float32, dimension)
		for j := 0; j < dimension; j++ {
			vector[j] = rand.Float32()
		}

		// Create metadata
		metadata := map[string]interface{}{
			"id":          i,
			"timestamp":   time.Now().Unix(),
			"description": fmt.Sprintf("Vector %d", i),
			"score":       rand.Float64(),
			"tags":        []string{"example", fmt.Sprintf("tag-%d", i%5)},
		}

		// Create facets
		vectorFacets := []facets.Facet{
			facets.NewBasicFacet("category", fmt.Sprintf("category-%d", i%5)),
			facets.NewBasicFacet("region", fmt.Sprintf("region-%d", i%3)),
			facets.NewBasicFacet("is_active", i%2 == 0),
		}

		// Add the vector with metadata and facets
		key := uint64(i + 1) // Keys start from 1
		metadataBytes, _ := json.Marshal(metadata)
		err := database.Add(key, vector, metadataBytes, vectorFacets)
		if err != nil {
			log.Printf("Failed to add vector %d: %v", i, err)
		}
	}

	log.Printf("Added %d vectors with metadata and facets", numVectors)
}

// basicSearch demonstrates a basic vector search
func basicSearch(database *db.VectorDB[uint64]) {
	log.Println("\nPerforming basic search...")

	// Create a query vector
	queryVector := []float32{0.2, 0.3, 0.4, 0.5, 0.6}

	// Create search options
	options := database.DefaultQueryOptions().WithK(3) // Return top 3 results

	// Perform the search
	results, err := database.Search(queryVector, options)
	if err != nil {
		log.Printf("Search failed: %v", err)
		return
	}

	// Display results
	log.Printf("Found %d results:", len(results))
	for i, result := range results {
		log.Printf("  Result %d: Key=%d, Distance=%f", i+1, result.Key, result.Distance)

		// Display metadata if available
		if len(result.Metadata) > 0 {
			var metadataMap map[string]interface{}
			if err := json.Unmarshal(result.Metadata, &metadataMap); err == nil {
				log.Printf("    Metadata: %v", metadataMap)
			}
		}

		// Display facets if available
		if len(result.Facets) > 0 {
			log.Printf("    Facets: %v", result.Facets)
		}
	}
}

// searchWithNegatives demonstrates search with negative examples
func searchWithNegatives(database *db.VectorDB[uint64]) {
	log.Println("\nPerforming search with negative examples...")

	// Create a query vector
	queryVector := []float32{0.2, 0.3, 0.4, 0.5, 0.6}

	// Create a negative example vector
	negativeVector := []float32{0.9, 0.8, 0.7, 0.6, 0.5}

	// Create search options with negative example
	options := database.DefaultQueryOptions().
		WithK(3).
		WithNegativeExample(negativeVector).
		WithNegativeWeight(0.7) // Higher weight gives more importance to avoiding negative examples

	// Perform the search
	results, err := database.Search(queryVector, options)
	if err != nil {
		log.Printf("Search with negatives failed: %v", err)
		return
	}

	// Display results
	log.Printf("Found %d results (with negative examples):", len(results))
	for i, result := range results {
		log.Printf("  Result %d: Key=%d, Distance=%f", i+1, result.Key, result.Distance)
	}
}

// searchWithFacets demonstrates search with facet filters
func searchWithFacets(database *db.VectorDB[uint64]) {
	log.Println("\nPerforming search with facet filters...")

	// Create a query vector
	queryVector := []float32{0.2, 0.3, 0.4, 0.5, 0.6}

	// Create a facet filter for category-1
	categoryFilter := facets.NewEqualityFilter("category", "category-1")

	// Create search options with facet filter
	options := database.DefaultQueryOptions().
		WithK(3).
		WithFacetFilters(categoryFilter)

	// Perform the search
	results, err := database.Search(queryVector, options)
	if err != nil {
		log.Printf("Search with facets failed: %v", err)
		return
	}

	// Display results
	log.Printf("Found %d results (with facet filters):", len(results))
	for i, result := range results {
		log.Printf("  Result %d: Key=%d, Distance=%f", i+1, result.Key, result.Distance)

		// Display facets if available
		if len(result.Facets) > 0 {
			log.Printf("    Facets: %v", result.Facets)
		}
	}
}

// searchWithMetadataFilters demonstrates search with metadata filters
func searchWithMetadataFilters(database *db.VectorDB[uint64]) {
	log.Println("\nPerforming search with metadata filters...")

	// Create a query vector
	queryVector := []float32{0.2, 0.3, 0.4, 0.5, 0.6}

	// Create a metadata filter for vectors with id < 500
	metadataFilter := []byte(`{"id": {"$lt": 500}}`)

	// Create search options with metadata filter
	options := database.DefaultQueryOptions().
		WithK(3).
		WithMetadataFilter(metadataFilter)

	// Perform the search
	results, err := database.Search(queryVector, options)
	if err != nil {
		log.Printf("Search with metadata filters failed: %v", err)
		return
	}

	// Display results
	log.Printf("Found %d results (with metadata filters):", len(results))
	for i, result := range results {
		log.Printf("  Result %d: Key=%d, Distance=%f", i+1, result.Key, result.Distance)

		// Display metadata if available
		if len(result.Metadata) > 0 {
			var metadataMap map[string]interface{}
			if err := json.Unmarshal(result.Metadata, &metadataMap); err == nil {
				log.Printf("    Metadata: %v", metadataMap)
			}
		}
	}
}

// searchWithComplexOptions demonstrates search with complex query options
func searchWithComplexOptions(database *db.VectorDB[uint64]) {
	log.Println("\nPerforming search with complex options...")

	// Create a query vector
	queryVector := []float32{0.2, 0.3, 0.4, 0.5, 0.6}

	// Create a negative example vector
	negativeVector := []float32{0.9, 0.8, 0.7, 0.6, 0.5}

	// Create a facet filter for active items
	activeFilter := facets.NewEqualityFilter("is_active", true)

	// Create a metadata filter for vectors with score > 0.5
	metadataFilter := []byte(`{"score": {"$gt": 0.5}}`)

	// Create search options with all filters
	options := database.DefaultQueryOptions().
		WithK(5).
		WithNegativeExample(negativeVector).
		WithNegativeWeight(0.5).
		WithFacetFilters(activeFilter).
		WithMetadataFilter(metadataFilter)

	// Perform the search
	results, err := database.Search(queryVector, options)
	if err != nil {
		log.Printf("Search with complex options failed: %v", err)
		return
	}

	// Display results
	log.Printf("Found %d results (with complex options):", len(results))
	for i, result := range results {
		log.Printf("  Result %d: Key=%d, Distance=%f", i+1, result.Key, result.Distance)
	}
}

// demoBatchOperations demonstrates batch operations
func demoBatchOperations(database *db.VectorDB[uint64]) {
	log.Println("\nDemonstrating batch operations...")

	// Create batch data
	batchSize := 10
	keys := make([]uint64, batchSize)
	vectors := make([][]float32, batchSize)
	metadataList := make([]interface{}, batchSize)
	facetsList := make([][]facets.Facet, batchSize)

	// Prepare batch data
	for i := 0; i < batchSize; i++ {
		// Create key (starting from 2000 to avoid conflicts)
		keys[i] = uint64(2000 + i)

		// Create vector
		vector := make([]float32, 5)
		for j := 0; j < 5; j++ {
			vector[j] = rand.Float32()
		}
		vectors[i] = vector

		// Create metadata
		metadata := map[string]interface{}{
			"batch_id":    i,
			"timestamp":   time.Now().Unix(),
			"description": fmt.Sprintf("Batch vector %d", i),
		}
		metadataBytes, _ := json.Marshal(metadata)
		metadataList[i] = metadataBytes

		// Create facets
		facetsList[i] = []facets.Facet{
			facets.NewBasicFacet("batch", true),
			facets.NewBasicFacet("index", i),
		}
	}

	// Perform batch add
	log.Println("Adding vectors in batch...")
	err := database.BatchAdd(keys, vectors, metadataList, facetsList)
	if err != nil {
		log.Printf("Batch add failed: %v", err)
		return
	}
	log.Printf("Added %d vectors in batch", batchSize)

	// Perform batch delete
	log.Println("Deleting some vectors in batch...")
	keysToDelete := []uint64{2000, 2002, 2004, 2006, 2008}
	deleteResults := database.BatchDelete(keysToDelete)

	// Check delete results
	successCount := 0
	for i, success := range deleteResults {
		if success {
			successCount++
			log.Printf("Successfully deleted key %d", keysToDelete[i])
		} else {
			log.Printf("Failed to delete key %d", keysToDelete[i])
		}
	}
	log.Printf("Deleted %d/%d vectors in batch", successCount, len(keysToDelete))
}

// demoBackupRestore demonstrates backup and restore functionality
func demoBackupRestore(database *db.VectorDB[uint64], backupDir string) {
	log.Println("\nDemonstrating backup and restore...")

	// Create backup directory
	if err := os.MkdirAll(filepath.Dir(backupDir), 0755); err != nil {
		log.Printf("Failed to create backup directory: %v", err)
		return
	}

	// Perform backup
	log.Printf("Backing up database to %s...", backupDir)
	err := database.Backup(backupDir)
	if err != nil {
		log.Printf("Backup failed: %v", err)
		return
	}
	log.Println("Backup completed successfully")

	// Create a new database for restore
	log.Println("Creating a new database for restore...")
	config := db.DefaultDBConfig()
	config.BaseDir = backupDir + "_restored"

	restoredDB, err := db.NewVectorDB[uint64](config)
	if err != nil {
		log.Printf("Failed to create database for restore: %v", err)
		return
	}
	defer restoredDB.Close()

	// Perform restore
	log.Printf("Restoring database from %s...", backupDir)
	err = restoredDB.Restore(backupDir)
	if err != nil {
		log.Printf("Restore failed: %v", err)
		return
	}
	log.Println("Restore completed successfully")

	// Verify restore by performing a search
	log.Println("Verifying restore by performing a search...")
	queryVector := []float32{0.2, 0.3, 0.4, 0.5, 0.6}
	options := restoredDB.DefaultQueryOptions().WithK(3)

	results, err := restoredDB.Search(queryVector, options)
	if err != nil {
		log.Printf("Search on restored database failed: %v", err)
		return
	}

	log.Printf("Found %d results in restored database", len(results))
}

// demoAPT demonstrates Adaptive Parameter Tuning features
func demoAPT() {
	log.Println("\n=== Adaptive Parameter Tuning (APT) Demonstration ===")

	// Check if APT is enabled
	log.Println("Checking if APT is enabled...")
	enabled := adaptive.IsEnabled()
	log.Printf("APT enabled: %v", enabled)

	// Get current parameters
	log.Println("\nGetting current APT parameters...")
	if adaptive.DefaultInstance == nil {
		log.Println("APT DefaultInstance is nil, cannot get parameters")
	} else {
		params := adaptive.DefaultInstance.GetCurrentParameters()
		log.Printf("Current parameters: %+v", params)
	}

	// Get workload analysis
	log.Println("\nGetting workload analysis...")
	if adaptive.DefaultInstance == nil {
		log.Println("APT DefaultInstance is nil, cannot get workload analysis")
	} else {
		analysis := adaptive.DefaultInstance.GetWorkloadAnalysis()
		log.Printf("Workload analysis: %+v", analysis)
	}

	// Get performance report
	log.Println("\nGetting performance report...")
	if adaptive.DefaultInstance == nil {
		log.Println("APT DefaultInstance is nil, cannot get performance report")
	} else {
		report := adaptive.DefaultInstance.GetPerformanceReport()
		log.Printf("Performance report: %+v", report)
	}

	// Demonstrate enabling/disabling APT
	log.Println("\nDemonstrating enabling/disabling APT...")
	log.Println("Disabling APT...")
	adaptive.SetEnabled(false)
	log.Printf("APT enabled: %v", adaptive.IsEnabled())

	log.Println("Re-enabling APT...")
	adaptive.SetEnabled(true)
	log.Printf("APT enabled: %v", adaptive.IsEnabled())

	log.Println("APT demonstration completed")
}
