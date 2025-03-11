package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/TFMV/quiver"
	"github.com/TFMV/quiver/api"
	"github.com/gofiber/fiber/v2"
	"go.uber.org/zap"
)

func main() {
	// Initialize logger
	logger, err := zap.NewDevelopment()
	if err != nil {
		log.Fatalf("Failed to create logger: %v", err)
	}
	defer logger.Sync()

	// Create a Quiver configuration with validation
	config := quiver.Config{
		Dimension:       128,
		StoragePath:     "./data.db",
		MaxElements:     100000,
		HNSWM:           16,
		HNSWEfConstruct: 200,
		HNSWEfSearch:    100,
		BatchSize:       1000,
		Distance:        quiver.Cosine,
		PersistInterval: 5 * time.Minute, // Add persistence interval to avoid warning
		// Enable dimensionality reduction with our fixed implementation
		EnableDimReduction: true,
		DimReductionMethod: "PCA",
		DimReductionTarget: 64,
	}

	// Validate the configuration
	if !quiver.ValidateConfigAndPrint(config) {
		log.Fatalf("Invalid configuration")
	}

	// Create a new Quiver index
	idx, err := quiver.New(config, logger)
	if err != nil {
		log.Fatalf("Failed to create index: %v", err)
	}
	defer idx.Close()

	// Add some random vectors for demonstration
	fmt.Println("Adding random vectors...")
	addRandomVectors(idx, 1000)

	// Create a Fiber app
	app := fiber.New()

	// Register the dashboard
	dashboardConfig := api.DefaultDashboardConfig()
	dashboardConfig.CustomTitle = "Quiver Demo Dashboard"
	dashboardConfig.RefreshInterval = 3 // Refresh every 3 seconds
	api.RegisterDashboard(app, idx, dashboardConfig, logger)

	// Start a background goroutine to add more vectors over time
	// This will make the dashboard more interesting
	go func() {
		for {
			time.Sleep(5 * time.Second)
			addRandomVectors(idx, 100)
			fmt.Println("Added 100 more vectors")
		}
	}()

	// Start a background goroutine to perform searches
	go func() {
		for {
			time.Sleep(1 * time.Second)
			// Generate a random query vector
			query := make([]float32, config.Dimension)
			for i := range query {
				query[i] = rand.Float32()
			}

			// Perform a search
			_, err := idx.Search(query, 10, 1, 10)
			if err != nil {
				logger.Error("Search failed", zap.Error(err))
			}
		}
	}()

	fmt.Println("Server running at http://localhost:8080/dashboard")
	fmt.Println("Press Ctrl+C to stop")

	// Start the server
	if err := app.Listen(":8080"); err != nil {
		logger.Fatal("Failed to start server", zap.Error(err))
	}
}

// addRandomVectors adds random vectors to the index
func addRandomVectors(idx *quiver.Index, count int) {
	// Generate a batch of vectors with some structure
	// This helps PCA work better than completely random data
	vectors := make([][]float32, count)

	// Determine the dimension to use
	dimension := idx.Config().Dimension
	if idx.Config().EnableDimReduction {
		// If dimensionality reduction is enabled, we need to use the original dimension
		dimension = 128 // Hard-coded for now, but should be obtained from the index
	}

	// Create a few "concept" vectors to give structure to the data
	concepts := make([][]float32, 5)
	for i := range concepts {
		concepts[i] = make([]float32, dimension)
		for j := range concepts[i] {
			concepts[i][j] = rand.Float32()*2 - 1 // Values between -1 and 1
		}
	}

	// Generate vectors as combinations of concepts plus noise
	for i := range vectors {
		vectors[i] = make([]float32, dimension)

		// Mix concepts with different weights
		for j := range vectors[i] {
			// Start with some noise
			vectors[i][j] = (rand.Float32() * 0.1) - 0.05 // Small noise component

			// Add weighted concepts
			for _, concept := range concepts {
				weight := rand.Float32() * 0.5 // Random weight for each concept
				vectors[i][j] += concept[j] * weight
			}
		}
	}

	// Add vectors to the index
	for i, vector := range vectors {
		id := uint64(time.Now().UnixNano()) + uint64(i)

		// Generate random metadata
		categories := []string{"electronics", "clothing", "books", "food", "toys"}
		category := categories[rand.Intn(len(categories))]

		price := float64(rand.Intn(10000)) / 100.0

		metadata := map[string]interface{}{
			"category": category,
			"price":    price,
			"in_stock": rand.Intn(2) == 1,
			"rating":   float64(rand.Intn(50)) / 10.0,
		}

		// Add to index
		if err := idx.Add(id, vector, metadata); err != nil {
			log.Printf("Failed to add vector: %v", err)
		}
	}
}
