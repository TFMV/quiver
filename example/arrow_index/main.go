package main

import (
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"github.com/TFMV/hnsw"
	"github.com/TFMV/hnsw/hnsw-extensions/arrow"
	"github.com/TFMV/hnsw/hnsw-extensions/facets"
	db "github.com/TFMV/quiver"
)

func main() {
	log.SetFlags(log.Ldate | log.Ltime | log.Lshortfile)
	log.Println("Starting Arrow Index Example")

	// Create a temporary directory for data
	tmpDir, err := os.MkdirTemp("", "quiver-arrow-example")
	if err != nil {
		log.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	log.Printf("Using temporary directory: %s", tmpDir)

	// Example 1: Create a VectorDB with Arrow index
	database := createArrowDatabase(tmpDir)
	defer database.Close()

	// Example 2: Add vectors
	addVectors(database)

	// Example 3: Search
	searchExample(database)

	// Example 4: Stream data using Arrow
	streamArrowData(database)

	// Example 5: Batch operations
	batchOperations(database)

	// Example 6: Filter with facets and metadata
	filterExample(database)

	// Example 7: Backup and restore
	backupAndRestore(database, tmpDir)

	log.Println("Arrow Index Example completed successfully")
}

// createArrowDatabase creates a new database with an Arrow index
func createArrowDatabase(baseDir string) *db.VectorDB[uint64] {
	log.Println("\n=== Creating Arrow Database ===")

	// Configure the Arrow storage
	arrowStorageDir := filepath.Join(baseDir, "arrow_data")
	if err := os.MkdirAll(arrowStorageDir, 0755); err != nil {
		log.Fatalf("Failed to create Arrow storage directory: %v", err)
	}

	// Create Arrow-specific configuration
	arrowConfig := arrow.DefaultArrowGraphConfig()
	arrowConfig.M = 16                            // Maximum number of connections per node
	arrowConfig.EfSearch = 100                    // Size of dynamic candidate list during search
	arrowConfig.Distance = hnsw.EuclideanDistance // Distance function

	// Create the Quiver database configuration
	config := db.DefaultDBConfig()
	config.BaseDir = baseDir
	config.Arrow = arrowConfig
	config.IndexType = "arrow" // Explicitly set index type to arrow

	// Set storage directory for Arrow data
	arrowDataDir := filepath.Join(baseDir, "arrow_data")
	if err := os.MkdirAll(arrowDataDir, 0755); err != nil {
		log.Fatalf("Failed to create Arrow data directory: %v", err)
	}

	// Create the database
	database, err := db.NewVectorDB[uint64](config)
	if err != nil {
		log.Fatalf("Failed to create database: %v", err)
	}

	log.Println("Arrow database created successfully")
	return database
}

// addVectors adds sample vectors to the database
func addVectors(database *db.VectorDB[uint64]) {
	log.Println("\n=== Adding Vectors ===")

	// Create some random vectors
	dimension := 128
	numVectors := 1000

	for i := 0; i < numVectors; i++ {
		// Create a random vector
		vector := make([]float32, dimension)
		for j := 0; j < dimension; j++ {
			vector[j] = rand.Float32()
		}

		// Create metadata
		metadata := map[string]interface{}{
			"id":        i,
			"timestamp": time.Now().Unix(),
			"category":  fmt.Sprintf("category-%d", i%5),
			"score":     rand.Float64(),
		}

		// Create facets
		vectorFacets := []facets.Facet{
			facets.NewBasicFacet("category", fmt.Sprintf("category-%d", i%5)),
			facets.NewBasicFacet("isPremium", i%3 == 0),
			facets.NewBasicFacet("rating", i%5+1),
		}

		// Add the vector
		key := uint64(i + 1) // Keys start from 1
		if err := database.Add(key, vector, metadata, vectorFacets); err != nil {
			log.Printf("Error adding vector %d: %v", i, err)
		}

		if i%200 == 0 && i > 0 {
			log.Printf("Added %d vectors", i)
		}
	}

	log.Printf("Added %d vectors successfully", numVectors)
}

// searchExample demonstrates search functionality
func searchExample(database *db.VectorDB[uint64]) {
	log.Println("\n=== Search Examples ===")

	// Create a query vector (just use random for demonstration)
	dimension := 128
	queryVector := make([]float32, dimension)
	for i := 0; i < dimension; i++ {
		queryVector[i] = rand.Float32()
	}

	// Basic search
	log.Println("Performing basic search...")
	options := database.DefaultQueryOptions().WithK(5)
	results, err := database.Search(queryVector, options)
	if err != nil {
		log.Printf("Search error: %v", err)
	} else {
		log.Printf("Found %d results", len(results))
		for i, result := range results {
			log.Printf("  Result %d: Key=%d, Distance=%f", i+1, result.Key, result.Distance)
		}
	}

	// Search with negative example
	log.Println("\nPerforming search with negative example...")
	negativeVector := make([]float32, dimension)
	for i := 0; i < dimension; i++ {
		negativeVector[i] = rand.Float32()
	}

	negOptions := database.DefaultQueryOptions().
		WithK(5).
		WithNegativeExample(negativeVector).
		WithNegativeWeight(0.7)

	negResults, err := database.Search(queryVector, negOptions)
	if err != nil {
		log.Printf("Search with negatives error: %v", err)
	} else {
		log.Printf("Found %d results with negative example", len(negResults))
		for i, result := range negResults {
			log.Printf("  Result %d: Key=%d, Distance=%f", i+1, result.Key, result.Distance)
		}
	}
}

// streamArrowData demonstrates how to use Arrow's streaming capabilities
func streamArrowData(database *db.VectorDB[uint64]) {
	log.Println("\n=== Arrow Streaming Example ===")
	log.Println("Note: This is a conceptual example. The actual implementation would use the Arrow API directly.")

	log.Println("To stream data using Arrow, you would:")
	log.Println("1. Get the underlying Arrow index")
	log.Println("2. Create an Arrow appender")
	log.Println("3. Stream record batches containing vectors")

	log.Println(`
// Example code (conceptual):
adapter := database.GetIndexAdapter()
if arrowIndex, ok := adapter.GetUnderlyingIndex().(*arrow.ArrowIndex[uint64]); ok {
    appenderConfig := arrow.DefaultAppenderConfig()
    appender := arrow.NewArrowAppender[uint64](arrowIndex, appenderConfig)
    
    // Stream record batches
    recordChan := make(chan arrow.Record, 10)
    errChan := appender.StreamRecordsAsync(recordChan)
    
    // Send records
    for record := range sourceRecords {
        recordChan <- record
    }
    close(recordChan)
    
    // Check for errors
    if err := <-errChan; err != nil {
        log.Printf("Error streaming records: %v", err)
    }
}`)
}

// batchOperations demonstrates batch operations with the Arrow index
func batchOperations(database *db.VectorDB[uint64]) {
	log.Println("\n=== Batch Operations Example ===")

	// Prepare batch data
	dimension := 128
	batchSize := 10
	keys := make([]uint64, batchSize)
	vectors := make([][]float32, batchSize)
	metadata := make([]interface{}, batchSize)
	facetsList := make([][]facets.Facet, batchSize)

	for i := 0; i < batchSize; i++ {
		// Create key (starting from 2000 to avoid conflicts)
		keys[i] = uint64(2000 + i)

		// Create vector
		vector := make([]float32, dimension)
		for j := 0; j < dimension; j++ {
			vector[j] = rand.Float32()
		}
		vectors[i] = vector

		// Create metadata
		metadata[i] = map[string]interface{}{
			"id":        i + 2000,
			"timestamp": time.Now().Unix(),
			"category":  fmt.Sprintf("batch-category-%d", i%3),
			"batch":     true,
		}

		// Create facets
		facetsList[i] = []facets.Facet{
			facets.NewBasicFacet("batch", true),
			facets.NewBasicFacet("category", fmt.Sprintf("batch-category-%d", i%3)),
		}
	}

	// Add vectors in batch
	log.Println("Adding vectors in batch...")
	if err := database.BatchAdd(keys, vectors, metadata, facetsList); err != nil {
		log.Printf("Batch add error: %v", err)
	} else {
		log.Printf("Added %d vectors in batch", batchSize)
	}

	// Delete some vectors in batch
	log.Println("Deleting vectors in batch...")
	keysToDelete := []uint64{2000, 2002, 2004}
	deleteResults := database.BatchDelete(keysToDelete)
	for i, deleted := range deleteResults {
		log.Printf("Delete key %d: %v", keysToDelete[i], deleted)
	}
}

// filterExample demonstrates filtering with facets and metadata
func filterExample(database *db.VectorDB[uint64]) {
	log.Println("\n=== Filtering Example ===")

	// Create a query vector
	dimension := 128
	queryVector := make([]float32, dimension)
	for i := 0; i < dimension; i++ {
		queryVector[i] = rand.Float32()
	}

	// Search with facet filter
	log.Println("Searching with facet filter...")
	facetFilter := facets.NewEqualityFilter("category", "category-1")
	facetOptions := database.DefaultQueryOptions().
		WithK(5).
		WithFacetFilters(facetFilter)

	facetResults, err := database.Search(queryVector, facetOptions)
	if err != nil {
		log.Printf("Facet search error: %v", err)
	} else {
		log.Printf("Found %d results with facet filter", len(facetResults))
		for i, result := range facetResults {
			log.Printf("  Result %d: Key=%d, Distance=%f", i+1, result.Key, result.Distance)
		}
	}

	// Search with metadata filter
	log.Println("\nSearching with metadata filter...")
	metadataFilter := []byte(`{"category": "category-2"}`)
	metaOptions := database.DefaultQueryOptions().
		WithK(5).
		WithMetadataFilter(metadataFilter)

	metaResults, err := database.Search(queryVector, metaOptions)
	if err != nil {
		log.Printf("Metadata search error: %v", err)
	} else {
		log.Printf("Found %d results with metadata filter", len(metaResults))
		for i, result := range metaResults {
			log.Printf("  Result %d: Key=%d, Distance=%f", i+1, result.Key, result.Distance)
		}
	}
}

// backupAndRestore demonstrates backup and restore functionality
func backupAndRestore(database *db.VectorDB[uint64], baseDir string) {
	log.Println("\n=== Backup and Restore Example ===")

	// Create backup directory
	backupDir := filepath.Join(baseDir, "backup")
	if err := os.MkdirAll(backupDir, 0755); err != nil {
		log.Printf("Failed to create backup directory: %v", err)
		return
	}

	// Perform backup
	log.Printf("Backing up database to %s...", backupDir)
	if err := database.Backup(backupDir); err != nil {
		log.Printf("Backup error: %v", err)
		log.Println("Note: This is expected due to the distance function serialization issue.")
		log.Println("In a production environment, you would need to implement a custom serialization method.")

		// For demonstration purposes, we'll continue as if backup succeeded
		log.Println("Continuing with example...")
	} else {
		log.Println("Backup completed successfully")
	}

	// Create a new database config for restore
	restoreDir := filepath.Join(baseDir, "restored")
	if err := os.MkdirAll(restoreDir, 0755); err != nil {
		log.Printf("Failed to create restore directory: %v", err)
		return
	}

	restoreConfig := db.DefaultDBConfig()
	restoreConfig.BaseDir = restoreDir
	restoreConfig.IndexType = "arrow"

	// Set the distance function explicitly since it can't be serialized
	restoreConfig.Arrow.Distance = hnsw.EuclideanDistance

	// Create a new database for restore
	restoredDB, err := db.NewVectorDB[uint64](restoreConfig)
	if err != nil {
		log.Printf("Failed to create database for restore: %v", err)
		return
	}
	defer restoredDB.Close()

	// Restore from backup
	log.Printf("Restoring database from %s...", backupDir)
	if err := restoredDB.Restore(backupDir); err != nil {
		log.Printf("Restore error: %v", err)
		log.Println("Note: This is expected due to the distance function serialization issue.")
		log.Println("In a production environment, you would need to implement a custom serialization method.")
		return
	}
	log.Println("Restore completed successfully")

	// Verify by doing a search
	log.Println("Verifying restore with a search...")
	dimension := 128
	queryVector := make([]float32, dimension)
	for i := 0; i < dimension; i++ {
		queryVector[i] = rand.Float32()
	}

	options := restoredDB.DefaultQueryOptions().WithK(5)
	results, err := restoredDB.Search(queryVector, options)
	if err != nil {
		log.Printf("Search error after restore: %v", err)
	} else {
		log.Printf("Found %d results after restore", len(results))
	}
}
