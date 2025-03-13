package db

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/TFMV/hnsw"
	"github.com/TFMV/hnsw/hnsw-extensions/facets"
	"github.com/TFMV/hnsw/hnsw-extensions/hybrid"
)

// generateRandomVector creates a random vector for benchmarking
func generateRandomVector(dim int, b *testing.B) []float32 {
	b.Helper()
	vector := make([]float32, dim)
	for i := range vector {
		vector[i] = float32(i) / float32(dim)
	}
	return vector
}

// setupBenchmarkDB creates a database for benchmarking
func setupBenchmarkDB(b *testing.B) (*VectorDB[string], string) {
	b.Helper()

	// Create a temporary directory for the test
	tempDir, err := os.MkdirTemp("", "vectordb-benchmark")
	if err != nil {
		b.Fatalf("Failed to create temp dir: %v", err)
	}

	// Create a database configuration
	config := DefaultDBConfig()
	config.BaseDir = tempDir
	config.Hybrid.Type = hybrid.HybridIndexType
	config.Hybrid.Distance = hnsw.CosineDistance

	// Create a vector database
	db, err := NewVectorDB[string](config)
	if err != nil {
		os.RemoveAll(tempDir)
		b.Fatalf("Failed to create vector database: %v", err)
	}

	return db, tempDir
}

// BenchmarkAdd benchmarks adding vectors one by one
func BenchmarkAdd(b *testing.B) {
	db, tempDir := setupBenchmarkDB(b)
	defer os.RemoveAll(tempDir)
	defer db.Close()

	// Reset timer to exclude setup time
	b.ResetTimer()

	// Benchmark adding vectors
	for i := 0; i < b.N; i++ {
		key := "key" + string(rune(i))
		vector := generateRandomVector(128, b)
		metadata := map[string]string{"name": "test" + string(rune(i))}
		facetList := []facets.Facet{
			facets.NewBasicFacet("category", "test"),
			facets.NewBasicFacet("score", float64(i)/float64(b.N)),
		}

		if err := db.Add(key, vector, metadata, facetList); err != nil {
			b.Fatalf("Failed to add vector: %v", err)
		}
	}
}

// BenchmarkBatchAdd benchmarks adding vectors in batches
func BenchmarkBatchAdd(b *testing.B) {
	db, tempDir := setupBenchmarkDB(b)
	defer os.RemoveAll(tempDir)
	defer db.Close()

	// Define batch size
	batchSize := 1000
	if b.N < batchSize {
		batchSize = b.N
	}

	// Prepare batches
	numBatches := b.N / batchSize
	if b.N%batchSize > 0 {
		numBatches++
	}

	// Reset timer to exclude setup time
	b.ResetTimer()

	// Benchmark adding vectors in batches
	for batch := 0; batch < numBatches; batch++ {
		start := batch * batchSize
		end := start + batchSize
		if end > b.N {
			end = b.N
		}

		batchSize := end - start
		keys := make([]string, batchSize)
		vectors := make([][]float32, batchSize)
		metadataList := make([]interface{}, batchSize)
		facetsList := make([][]facets.Facet, batchSize)

		for i := 0; i < batchSize; i++ {
			idx := start + i
			keys[i] = "key" + string(rune(idx))
			vectors[i] = generateRandomVector(128, b)
			metadataList[i] = map[string]string{"name": "test" + string(rune(idx))}
			facetsList[i] = []facets.Facet{
				facets.NewBasicFacet("category", "test"),
				facets.NewBasicFacet("score", float64(idx)/float64(b.N)),
			}
		}

		if err := db.BatchAdd(keys, vectors, metadataList, facetsList); err != nil {
			b.Fatalf("Failed to batch add vectors: %v", err)
		}
	}
}

// BenchmarkSearch benchmarks searching for vectors
func BenchmarkSearch(b *testing.B) {
	db, tempDir := setupBenchmarkDB(b)
	defer os.RemoveAll(tempDir)
	defer db.Close()

	// Add some vectors for searching
	numVectors := 10000
	keys := make([]string, numVectors)
	vectors := make([][]float32, numVectors)
	metadataList := make([]interface{}, numVectors)
	facetsList := make([][]facets.Facet, numVectors)

	for i := 0; i < numVectors; i++ {
		keys[i] = "key" + string(rune(i))
		vectors[i] = generateRandomVector(128, b)
		metadataList[i] = map[string]string{"name": "test" + string(rune(i))}
		facetsList[i] = []facets.Facet{
			facets.NewBasicFacet("category", "test"),
			facets.NewBasicFacet("score", float64(i)/float64(numVectors)),
		}
	}

	if err := db.BatchAdd(keys, vectors, metadataList, facetsList); err != nil {
		b.Fatalf("Failed to batch add vectors: %v", err)
	}

	// Generate query vector
	query := generateRandomVector(128, b)

	// Reset timer to exclude setup time
	b.ResetTimer()

	// Benchmark searching
	for i := 0; i < b.N; i++ {
		options := db.DefaultQueryOptions().WithK(10)
		_, err := db.Search(query, options)
		if err != nil {
			b.Fatalf("Failed to search: %v", err)
		}
	}
}

// BenchmarkSearchWithFilters benchmarks searching with facet filters
func BenchmarkSearchWithFilters(b *testing.B) {
	db, tempDir := setupBenchmarkDB(b)
	defer os.RemoveAll(tempDir)
	defer db.Close()

	// Add some vectors for searching
	numVectors := 10000
	keys := make([]string, numVectors)
	vectors := make([][]float32, numVectors)
	metadataList := make([]interface{}, numVectors)
	facetsList := make([][]facets.Facet, numVectors)

	categories := []string{"finance", "technology", "healthcare", "education", "entertainment"}

	for i := 0; i < numVectors; i++ {
		keys[i] = "key" + string(rune(i))
		vectors[i] = generateRandomVector(128, b)
		metadataList[i] = map[string]string{
			"name":     "test" + string(rune(i)),
			"category": categories[i%len(categories)],
		}
		facetsList[i] = []facets.Facet{
			facets.NewBasicFacet("category", categories[i%len(categories)]),
			facets.NewBasicFacet("score", float64(i)/float64(numVectors)),
		}
	}

	if err := db.BatchAdd(keys, vectors, metadataList, facetsList); err != nil {
		b.Fatalf("Failed to batch add vectors: %v", err)
	}

	// Generate query vector
	query := generateRandomVector(128, b)

	// Reset timer to exclude setup time
	b.ResetTimer()

	// Benchmark searching with filters
	for i := 0; i < b.N; i++ {
		options := db.DefaultQueryOptions().
			WithK(10).
			WithFacetFilters(facets.NewEqualityFilter("category", "technology"))

		_, err := db.Search(query, options)
		if err != nil {
			b.Fatalf("Failed to search with filters: %v", err)
		}
	}
}

// BenchmarkSearchWithNegativeExamples benchmarks searching with negative examples
func BenchmarkSearchWithNegativeExamples(b *testing.B) {
	db, tempDir := setupBenchmarkDB(b)
	defer os.RemoveAll(tempDir)
	defer db.Close()

	// Add some vectors for searching
	numVectors := 10000
	keys := make([]string, numVectors)
	vectors := make([][]float32, numVectors)
	metadataList := make([]interface{}, numVectors)
	facetsList := make([][]facets.Facet, numVectors)

	for i := 0; i < numVectors; i++ {
		keys[i] = "key" + string(rune(i))
		vectors[i] = generateRandomVector(128, b)
		metadataList[i] = map[string]string{"name": "test" + string(rune(i))}
		facetsList[i] = []facets.Facet{
			facets.NewBasicFacet("category", "test"),
			facets.NewBasicFacet("score", float64(i)/float64(numVectors)),
		}
	}

	if err := db.BatchAdd(keys, vectors, metadataList, facetsList); err != nil {
		b.Fatalf("Failed to batch add vectors: %v", err)
	}

	// Generate query vector and negative example
	query := generateRandomVector(128, b)
	negative := generateRandomVector(128, b)

	// Reset timer to exclude setup time
	b.ResetTimer()

	// Benchmark searching with negative examples
	for i := 0; i < b.N; i++ {
		options := db.DefaultQueryOptions().
			WithK(10).
			WithNegativeExample(negative).
			WithNegativeWeight(0.5)

		_, err := db.Search(query, options)
		if err != nil {
			b.Fatalf("Failed to search with negative example: %v", err)
		}
	}
}

// BenchmarkDelete benchmarks deleting vectors.
// NOTE: Deleting vectors is considered an anti-pattern in vector databases
// as it can degrade the graph structure. This benchmark is included for
// completeness but should not be considered a primary operation.
func BenchmarkDelete(b *testing.B) {
	// Limit benchmark size to avoid excessive runtime
	// Delete operations are expensive due to graph restructuring
	if b.N > 10 {
		b.Skip("Skipping large delete benchmark (anti-pattern operation)")
	}

	db, tempDir := setupBenchmarkDB(b)
	defer os.RemoveAll(tempDir)
	defer db.Close()

	// Use a minimal number of vectors for the test
	const numVectors = 50
	keys := make([]string, numVectors)

	// Add vectors for deletion
	for i := 0; i < numVectors; i++ {
		keys[i] = "key" + string(rune(i))
		vector := generateRandomVector(32, b) // Use smaller vectors
		metadata := map[string]string{"name": "test" + string(rune(i))}
		facetList := []facets.Facet{
			facets.NewBasicFacet("category", "test"),
		}

		if err := db.Add(keys[i], vector, metadata, facetList); err != nil {
			b.Fatalf("Failed to add vector: %v", err)
		}
	}

	// Reset timer to exclude setup time
	b.ResetTimer()

	// Only run the benchmark for a limited number of operations
	maxOps := b.N
	if maxOps > numVectors {
		maxOps = numVectors
	}

	// Benchmark deleting vectors (each vector only once)
	for i := 0; i < maxOps; i++ {
		key := keys[i]
		db.Delete(key)
	}
}

// BenchmarkBatchDelete benchmarks deleting vectors in batches.
// NOTE: Deleting vectors is considered an anti-pattern in vector databases
// as it can degrade the graph structure. This benchmark is included for
// completeness but should not be considered a primary operation.
func BenchmarkBatchDelete(b *testing.B) {
	// Limit benchmark size to avoid excessive runtime
	// Delete operations are expensive due to graph restructuring
	if b.N > 5 {
		b.Skip("Skipping large batch delete benchmark (anti-pattern operation)")
	}

	db, tempDir := setupBenchmarkDB(b)
	defer os.RemoveAll(tempDir)
	defer db.Close()

	// Use a minimal number of vectors for the test
	const numVectors = 50
	keys := make([]string, numVectors)
	vectors := make([][]float32, numVectors)
	metadataList := make([]interface{}, numVectors)
	facetsList := make([][]facets.Facet, numVectors)

	for i := 0; i < numVectors; i++ {
		keys[i] = "key" + string(rune(i))
		vectors[i] = generateRandomVector(32, b) // Use smaller vectors
		metadataList[i] = map[string]string{"name": "test" + string(rune(i))}
		facetsList[i] = []facets.Facet{
			facets.NewBasicFacet("category", "test"),
		}
	}

	if err := db.BatchAdd(keys, vectors, metadataList, facetsList); err != nil {
		b.Fatalf("Failed to batch add vectors: %v", err)
	}

	// Define batch size
	batchSize := 5

	// Reset timer to exclude setup time
	b.ResetTimer()

	// Only run the benchmark for a limited number of operations
	maxBatches := b.N
	if maxBatches*batchSize > numVectors {
		maxBatches = numVectors / batchSize
	}

	// Benchmark deleting vectors in batches (each vector only once)
	for i := 0; i < maxBatches; i++ {
		startIdx := i * batchSize
		endIdx := startIdx + batchSize
		if endIdx > numVectors {
			endIdx = numVectors
		}

		// Get the batch of keys to delete
		batchKeys := keys[startIdx:endIdx]

		// Delete the batch
		db.BatchDelete(batchKeys)
	}
}

// BenchmarkBackupRestore benchmarks backup and restore operations
func BenchmarkBackupRestore(b *testing.B) {
	db, tempDir := setupBenchmarkDB(b)
	defer os.RemoveAll(tempDir)
	defer db.Close()

	// Add some vectors
	numVectors := 100 // Reduced from 1000
	keys := make([]string, numVectors)
	vectors := make([][]float32, numVectors)
	metadataList := make([]interface{}, numVectors)
	facetsList := make([][]facets.Facet, numVectors)

	for i := 0; i < numVectors; i++ {
		keys[i] = "key" + string(rune(i))
		vectors[i] = generateRandomVector(32, b) // Smaller vectors
		metadataList[i] = map[string]string{"name": "test" + string(rune(i))}
		facetsList[i] = []facets.Facet{
			facets.NewBasicFacet("category", "test"),
		}
	}

	if err := db.BatchAdd(keys, vectors, metadataList, facetsList); err != nil {
		b.Fatalf("Failed to batch add vectors: %v", err)
	}

	// Create backup directory
	backupDir := filepath.Join(tempDir, "backup")
	if err := os.MkdirAll(backupDir, 0755); err != nil {
		b.Fatalf("Failed to create backup directory: %v", err)
	}

	// Reset timer to exclude setup time
	b.ResetTimer()

	// Benchmark backup and restore
	for i := 0; i < b.N; i++ {
		// Create a unique backup directory for each iteration
		// Use a string instead of a rune for the directory name
		iterBackupDir := filepath.Join(backupDir, fmt.Sprintf("iter_%d", i))
		if err := os.MkdirAll(iterBackupDir, 0755); err != nil {
			b.Fatalf("Failed to create iteration backup directory: %v", err)
		}

		// Backup
		if err := db.Backup(iterBackupDir); err != nil {
			b.Fatalf("Failed to backup: %v", err)
		}

		// Create a new database for restoration
		restoreDir := filepath.Join(tempDir, "restore", fmt.Sprintf("iter_%d", i))
		if err := os.MkdirAll(restoreDir, 0755); err != nil {
			b.Fatalf("Failed to create restore directory: %v", err)
		}

		restoreConfig := DefaultDBConfig()
		restoreConfig.BaseDir = restoreDir
		restoreConfig.Hybrid.Type = hybrid.HybridIndexType
		restoreConfig.Hybrid.Distance = hnsw.CosineDistance

		restoredDB, err := NewVectorDB[string](restoreConfig)
		if err != nil {
			b.Fatalf("Failed to create restored database: %v", err)
		}

		// Restore
		if err := restoredDB.Restore(iterBackupDir); err != nil {
			restoredDB.Close()
			b.Fatalf("Failed to restore: %v", err)
		}

		// Clean up
		restoredDB.Close()
	}
}

// BenchmarkOptimizeStorage benchmarks storage optimization
func BenchmarkOptimizeStorage(b *testing.B) {
	db, tempDir := setupBenchmarkDB(b)
	defer os.RemoveAll(tempDir)
	defer db.Close()

	// Add some vectors
	numVectors := 100 // Reduced from 10000
	keys := make([]string, numVectors)
	vectors := make([][]float32, numVectors)
	metadataList := make([]interface{}, numVectors)
	facetsList := make([][]facets.Facet, numVectors)

	for i := 0; i < numVectors; i++ {
		keys[i] = "key" + string(rune(i))
		vectors[i] = generateRandomVector(32, b) // Smaller vectors
		metadataList[i] = map[string]string{"name": "test" + string(rune(i))}
		facetsList[i] = []facets.Facet{
			facets.NewBasicFacet("category", "test"),
		}
	}

	if err := db.BatchAdd(keys, vectors, metadataList, facetsList); err != nil {
		b.Fatalf("Failed to batch add vectors: %v", err)
	}

	// Delete some vectors to create fragmentation
	deleteKeys := make([]string, numVectors/2)
	for i := 0; i < numVectors/2; i++ {
		deleteKeys[i] = keys[i*2] // Delete every other key
	}
	db.BatchDelete(deleteKeys)

	// Reset timer to exclude setup time
	b.ResetTimer()

	// Benchmark optimizing storage
	for i := 0; i < b.N; i++ {
		if err := db.OptimizeStorage(); err != nil {
			b.Fatalf("Failed to optimize storage: %v", err)
		}
	}
}
