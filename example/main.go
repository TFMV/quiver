package main

import (
	"fmt"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/TFMV/hnsw"
	"github.com/TFMV/hnsw/hnsw-extensions/facets"
	"github.com/TFMV/hnsw/hnsw-extensions/hybrid"
	"github.com/bytedance/sonic"
	"github.com/TFMV/quiver"
)

const (
	NUM_VECTORS = 10000
	DIMS        = 128
)

// Document represents a document with metadata
type Document struct {
	ID       string   `json:"id"`
	Title    string   `json:"title"`
	Content  string   `json:"content"`
	Category string   `json:"category"`
	Tags     []string `json:"tags"`
	Score    float64  `json:"score"`
}

// indexTypeToString converts an IndexType to a string
func indexTypeToString(indexType hybrid.IndexType) string {
	switch indexType {
	case hybrid.ExactIndexType:
		return "Exact"
	case hybrid.HNSWIndexType:
		return "HNSW"
	case hybrid.LSHIndexType:
		return "LSH"
	case hybrid.HybridIndexType:
		return "Hybrid"
	default:
		return "Unknown"
	}
}

func main() {
	// Set random seed
	rand.Seed(time.Now().UnixNano())

	// Create a vector database
	fmt.Println("Creating vector database...")
	db, err := createVectorDB()
	if err != nil {
		log.Fatalf("Failed to create vector database: %v", err)
	}
	defer db.Close()

	// Add vectors
	fmt.Println("Adding vectors...")
	addVectors(db)

	// Search
	fmt.Println("\nPerforming simple search...")
	simpleSearch(db)

	fmt.Println("\nPerforming search with facet filters...")
	searchWithFacetFilters(db)

	fmt.Println("\nPerforming search with negative examples...")
	searchWithNegativeExamples(db)

	// Create a backup
	fmt.Println("\nCreating backup...")
	backupDir := "./backup"
	if err := db.Backup(backupDir); err != nil {
		log.Fatalf("Failed to create backup: %v", err)
	}
	fmt.Printf("Backup created at %s\n", backupDir)

	// Delete some vectors
	fmt.Println("\nDeleting vectors...")
	deleteVectors(db)

	// Display statistics
	fmt.Println("\nDatabase statistics after deletion:")
	displayStats(db)

	// Create a new database instance for restoration
	fmt.Println("\nCreating new database instance for restoration...")
	restoredDB, err := createEmptyVectorDB()
	if err != nil {
		log.Fatalf("Failed to create empty vector database: %v", err)
	}
	defer restoredDB.Close()

	// Restore from backup
	fmt.Println("Restoring from backup...")
	if err := restoredDB.Restore(backupDir); err != nil {
		log.Fatalf("Failed to restore from backup: %v", err)
	}
	fmt.Println("Database restored successfully!")

	// Display statistics of restored database
	fmt.Println("\nRestored database statistics:")
	displayStats(restoredDB)

	// Verify restoration by performing a search
	fmt.Println("\nVerifying restoration with a search...")
	simpleSearchOnDB(restoredDB)
}

// createVectorDB creates a new vector database
func createVectorDB() (*db.VectorDB[string], error) {
	// Create a temporary directory for the database
	tempDir, err := os.MkdirTemp("", "vectordb-example")
	if err != nil {
		return nil, fmt.Errorf("failed to create temporary directory: %w", err)
	}

	fmt.Printf("Using temporary directory: %s\n", tempDir)

	// Configure the vector database
	config := db.DBConfig{
		BaseDir: tempDir,
		Hybrid: hybrid.IndexConfig{
			Type:     hybrid.HybridIndexType,
			M:        16,
			Ml:       0.25,
			EfSearch: 50,
			Distance: hnsw.CosineDistance,
		},
	}

	// Create the vector database
	return db.NewVectorDB[string](config)
}

// createEmptyVectorDB creates an empty vector database for restoration
func createEmptyVectorDB() (*db.VectorDB[string], error) {
	// Create a temporary directory for the database
	dbDir := "./temp_db"
	if err := os.MkdirAll(dbDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create database directory: %w", err)
	}

	// Create database configuration
	config := db.DBConfig{
		BaseDir: dbDir,
		Hybrid: hybrid.IndexConfig{
			Type:     hybrid.HybridIndexType,
			M:        16,
			Ml:       0.25,
			EfSearch: 50,
			Distance: hnsw.CosineDistance,
		},
	}

	// Create the database
	return db.NewVectorDB[string](config)
}

// generateRandomVector generates a random vector of the specified dimension
func generateRandomVector(dim int) []float32 {
	vector := make([]float32, dim)
	for i := range vector {
		vector[i] = rand.Float32()
	}
	return vector
}

// generateRandomDocument generates a random document with the specified ID
func generateRandomDocument(id string) Document {
	categories := []string{"technology", "science", "business", "entertainment", "health"}
	tags := []string{"ai", "machine learning", "data science", "cloud", "security", "mobile", "web"}

	// Select random category and tags
	category := categories[rand.Intn(len(categories))]
	numTags := rand.Intn(3) + 1
	docTags := make([]string, numTags)
	for i := 0; i < numTags; i++ {
		docTags[i] = tags[rand.Intn(len(tags))]
	}

	return Document{
		ID:       id,
		Title:    fmt.Sprintf("Document %s", id),
		Content:  fmt.Sprintf("This is the content of document %s", id),
		Category: category,
		Tags:     docTags,
		Score:    rand.Float64() * 10.0,
	}
}

// addVectors adds random vectors to the database
func addVectors(db *db.VectorDB[string]) {
	// Generate random vectors and documents for batch insertion
	keys := make([]string, NUM_VECTORS)
	vectors := make([][]float32, NUM_VECTORS)
	metadataList := make([]interface{}, NUM_VECTORS)
	facetsList := make([][]facets.Facet, NUM_VECTORS)

	for i := 0; i < NUM_VECTORS; i++ {
		id := fmt.Sprintf("%d", i)
		keys[i] = id
		vectors[i] = generateRandomVector(DIMS)

		// Create document
		doc := generateRandomDocument(id)
		metadataList[i] = doc

		// Create facets
		docFacets := []facets.Facet{
			facets.NewBasicFacet("category", doc.Category),
			facets.NewBasicFacet("score", doc.Score),
		}

		// Add tags as facets
		for _, tag := range doc.Tags {
			docFacets = append(docFacets, facets.NewBasicFacet("tag", tag))
		}

		facetsList[i] = docFacets
	}

	// Measure time for batch insertion
	startTime := time.Now()

	// Add vectors in batches
	batchSize := 1000
	for i := 0; i < NUM_VECTORS; i += batchSize {
		end := i + batchSize
		if end > NUM_VECTORS {
			end = NUM_VECTORS
		}

		batchKeys := keys[i:end]
		batchVectors := vectors[i:end]
		batchMetadata := metadataList[i:end]
		batchFacets := facetsList[i:end]

		if err := db.BatchAdd(batchKeys, batchVectors, batchMetadata, batchFacets); err != nil {
			log.Fatalf("Failed to add batch of vectors: %v", err)
		}

		fmt.Printf("Added batch %d/%d (%d vectors)\n", i/batchSize+1, (NUM_VECTORS+batchSize-1)/batchSize, len(batchKeys))
	}

	elapsed := time.Since(startTime)
	fmt.Printf("Added %d vectors in %v (%.2f vectors/sec)\n", NUM_VECTORS, elapsed, float64(NUM_VECTORS)/elapsed.Seconds())

	// Optimize storage
	fmt.Println("Optimizing storage...")
	if err := db.OptimizeStorage(); err != nil {
		log.Fatalf("Failed to optimize storage: %v", err)
	}
}

// simpleSearch performs a simple search on the database
func simpleSearch(db *db.VectorDB[string]) {
	// Create a random vector for search
	queryVector := generateRandomVector(DIMS)

	// Use the fluent API for query options
	options := db.DefaultQueryOptions().WithK(5)

	// Perform search
	results, err := db.Search(queryVector, options)
	if err != nil {
		log.Fatalf("Failed to search: %v", err)
	}

	// Display results
	fmt.Printf("Found %d results\n", len(results))
	for i, result := range results {
		var doc Document
		if err := sonic.Unmarshal(result.Metadata, &doc); err != nil {
			log.Printf("Failed to unmarshal metadata: %v", err)
			continue
		}
		fmt.Printf("  %d. %s (distance: %.4f, category: %s)\n", i+1, doc.Title, result.Distance, doc.Category)
	}
}

// searchWithFacetFilters performs a search with facet filters
func searchWithFacetFilters(db *db.VectorDB[string]) {
	// Create a random vector for search
	queryVector := generateRandomVector(DIMS)

	// Use the fluent API for query options with facet filters
	options := db.DefaultQueryOptions().
		WithK(20).
		WithFacetFilters(facets.NewEqualityFilter("category", "technology"))

	// Perform search with filters
	filteredResults, err := db.Search(queryVector, options)
	if err != nil {
		log.Fatalf("Failed to search with filters: %v", err)
	}

	// Display results
	fmt.Printf("Found %d filtered results\n", len(filteredResults))
	for i, result := range filteredResults {
		var doc Document
		if err := sonic.Unmarshal(result.Metadata, &doc); err != nil {
			log.Printf("Failed to unmarshal metadata: %v", err)
			continue
		}
		fmt.Printf("  %d. %s (distance: %.4f, category: %s, score: %.2f)\n",
			i+1, doc.Title, result.Distance, doc.Category, doc.Score)
	}
}

// searchWithNegativeExamples performs a search with negative examples
func searchWithNegativeExamples(db *db.VectorDB[string]) {
	// Create random vectors for search
	queryVector := generateRandomVector(DIMS)
	negativeVector := generateRandomVector(DIMS)

	// Use the fluent API for query options with negative examples
	options := db.DefaultQueryOptions().
		WithK(5).
		WithNegativeExample(negativeVector).
		WithNegativeWeight(0.3)

	// Perform search with negative examples
	negativeResults, err := db.Search(queryVector, options)
	if err != nil {
		log.Fatalf("Failed to search with negative example: %v", err)
	}

	// Display results
	fmt.Printf("Found %d results with negative example\n", len(negativeResults))
	for i, result := range negativeResults {
		var doc Document
		if err := sonic.Unmarshal(result.Metadata, &doc); err != nil {
			log.Printf("Failed to unmarshal metadata: %v", err)
			continue
		}
		fmt.Printf("  %d. %s (distance: %.4f, category: %s)\n", i+1, doc.Title, result.Distance, doc.Category)
	}
}

// deleteVectors deletes a batch of vectors from the database
func deleteVectors(db *db.VectorDB[string]) {
	// Create a list of keys to delete
	deleteKeys := make([]string, 100)
	for i := 0; i < 100; i++ {
		deleteKeys[i] = fmt.Sprintf("%d", i)
	}

	// Delete the vectors
	deleteResults := db.BatchDelete(deleteKeys)

	// Count successful deletions
	successCount := 0
	for _, success := range deleteResults {
		if success {
			successCount++
		}
	}
	fmt.Printf("Successfully deleted %d/%d vectors\n", successCount, len(deleteKeys))
}

// displayStats displays the database statistics
func displayStats(db *db.VectorDB[string]) {
	// Get database statistics
	stats := db.GetStats()

	// Display statistics
	fmt.Printf("  Vector count: %d\n", stats.VectorCount)
	// Use the hybrid index type from the stats
	hybridType := hybrid.HybridIndexType // Default value if not available
	fmt.Printf("  Index type: %s\n", indexTypeToString(hybridType))
}

// simpleSearchOnDB performs a simple search on the specified database
func simpleSearchOnDB(db *db.VectorDB[string]) {
	// Create a random vector for search
	vector := make([]float32, DIMS)
	for i := range vector {
		vector[i] = rand.Float32()
	}

	// Use the fluent API for query options
	options := db.DefaultQueryOptions().WithK(5)

	// Perform search
	results, err := db.Search(vector, options)
	if err != nil {
		log.Fatalf("Failed to search: %v", err)
	}

	// Display results
	fmt.Printf("Found %d results:\n", len(results))
	for i, result := range results {
		var doc Document
		if err := sonic.Unmarshal(result.Metadata, &doc); err != nil {
			log.Printf("Failed to unmarshal metadata: %v", err)
			continue
		}

		fmt.Printf("  %d. Key: %s, Distance: %.4f\n", i+1, result.Key, result.Distance)
		fmt.Printf("     Category: %s\n", doc.Category)
		if len(result.Facets) > 0 {
			fmt.Printf("     Facets: %+v\n", result.Facets)
		}
	}
}
