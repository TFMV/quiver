package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/TFMV/quiver"
	"github.com/TFMV/quiver/dimreduce"
	"github.com/TFMV/quiver/extensions"
	"github.com/TFMV/quiver/router"
	"go.uber.org/zap"
)

func main() {
	// Initialize logger
	logger, err := zap.NewDevelopment()
	if err != nil {
		log.Fatalf("Failed to create logger: %v", err)
	}
	defer logger.Sync()

	// Example 1: Dimensionality Reduction
	fmt.Println("=== Example 1: Dimensionality Reduction ===")
	dimReduceExample(logger)

	// Example 2: Semantic Routing
	fmt.Println("\n=== Example 2: Semantic Routing ===")
	semanticRoutingExample(logger)
}

func dimReduceExample(logger *zap.Logger) {
	// Create a Quiver index with dimensionality reduction
	quiverConfig := quiver.Config{
		Dimension:       128,
		Distance:        quiver.CosineDistance,
		HNSWM:           16,
		HNSWEfConstruct: 200,
		HNSWEfSearch:    100,
	}

	dimReduceConfig := dimreduce.DefaultDimReducerConfig()
	dimReduceConfig.TargetDimension = 64 // Reduce to 64 dimensions
	dimReduceConfig.Logger = logger

	// Create the index with dimensionality reduction
	idx, err := extensions.NewIndexWithDimReduce(quiverConfig, dimReduceConfig, logger)
	if err != nil {
		logger.Fatal("Failed to create index with dimensionality reduction", zap.Error(err))
	}

	// Create some example vectors (512 dimensions)
	vectors := make([][]float32, 5)
	for i := range vectors {
		vectors[i] = make([]float32, 512)
		for j := range vectors[i] {
			vectors[i][j] = float32(i*j) / 512.0
		}
	}

	// Add vectors with dimensionality reduction
	for i, vec := range vectors {
		err := idx.AddWithDimReduce(uint64(i+1), vec, map[string]interface{}{
			"name": fmt.Sprintf("Vector %d", i+1),
		})
		if err != nil {
			logger.Error("Failed to add vector", zap.Error(err))
			continue
		}
	}

	// Search with dimensionality reduction
	results, err := idx.SearchWithDimReduce(vectors[0], 3, 0, 0)
	if err != nil {
		logger.Error("Failed to search", zap.Error(err))
	} else {
		fmt.Println("Search results:")
		for i, result := range results {
			fmt.Printf("  %d. ID: %d, Distance: %.4f\n", i+1, result.ID, result.Distance)
		}
	}

	// Try adaptive dimensionality reduction
	reducedVectors, err := idx.AdaptiveReduce(vectors)
	if err != nil {
		logger.Error("Failed to perform adaptive reduction", zap.Error(err))
	} else {
		fmt.Printf("Adaptively reduced vectors from %d to %d dimensions\n",
			len(vectors[0]), len(reducedVectors[0]))
	}
}

func semanticRoutingExample(logger *zap.Logger) {
	// Create router configuration
	routerConfig := router.DefaultRouterConfig()
	routerConfig.EnableCache = true
	routerConfig.EnableLogging = true

	// Create multi-index manager
	manager, err := extensions.NewMultiIndexManager(routerConfig, logger)
	if err != nil {
		logger.Fatal("Failed to create multi-index manager", zap.Error(err))
	}
	defer manager.Close()

	// Create specialized indices
	indices := make(map[router.IndexType]*quiver.Index)
	indexTypes := []router.IndexType{
		router.GeneralIndex,
		router.TechnicalIndex,
		router.CreativeIndex,
	}

	// Create embeddings for each index type
	embeddings := map[router.IndexType][]float32{
		router.GeneralIndex:   {0.1, 0.2, 0.3, 0.4},
		router.TechnicalIndex: {0.8, 0.1, 0.0, 0.1},
		router.CreativeIndex:  {0.1, 0.1, 0.8, 0.0},
	}

	// Create and register indices
	for _, indexType := range indexTypes {
		config := quiver.Config{
			Dimension:       128,
			Distance:        quiver.CosineDistance,
			HNSWM:           16,
			HNSWEfConstruct: 200,
			HNSWEfSearch:    100,
		}

		idx, err := quiver.New(config, logger)
		if err != nil {
			logger.Error("Failed to create index", zap.String("type", string(indexType)), zap.Error(err))
			continue
		}

		indices[indexType] = idx

		// Register the index with the manager
		err = manager.RegisterIndex(indexType, idx, embeddings[indexType])
		if err != nil {
			logger.Error("Failed to register index", zap.String("type", string(indexType)), zap.Error(err))
		}
	}

	// Add some vectors to each index
	for indexType, idx := range indices {
		for i := 0; i < 5; i++ {
			vector := make([]float32, 128)
			for j := range vector {
				vector[j] = float32(i*j) / 128.0
			}

			err := idx.Add(uint64(i+1), vector, map[string]interface{}{
				"name":  fmt.Sprintf("Vector %d", i+1),
				"index": string(indexType),
			})
			if err != nil {
				logger.Error("Failed to add vector",
					zap.String("index", string(indexType)),
					zap.Error(err))
			}
		}
	}

	// Create example queries for different index types
	queries := map[string][]float32{
		"technical": {0.9, 0.1, 0.0, 0.0},
		"creative":  {0.1, 0.0, 0.9, 0.0},
		"general":   {0.3, 0.3, 0.3, 0.1},
	}

	// Route and search with each query
	for queryName, query := range queries {
		fmt.Printf("\nRouting query: %s\n", queryName)

		// Route the query
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		decision, err := manager.Route(ctx, query)
		cancel()

		if err != nil {
			logger.Error("Failed to route query", zap.String("query", queryName), zap.Error(err))
			continue
		}

		fmt.Printf("  Routed to: %s (confidence: %.2f)\n", decision.TargetIndex, decision.Confidence)

		// Search using the routed index
		results, err := manager.MultiIndexSearch(context.Background(), query, 3)
		if err != nil {
			logger.Error("Failed to search", zap.String("query", queryName), zap.Error(err))
			continue
		}

		fmt.Printf("  Search results:\n")
		for i, result := range results {
			indexInfo := "unknown"
			if routing, ok := result.Metadata["_routing"].(map[string]interface{}); ok {
				if idx, ok := routing["index"].(string); ok {
					indexInfo = idx
				}
			}

			fmt.Printf("    %d. ID: %d, Index: %s, Distance: %.4f\n",
				i+1, result.ID, indexInfo, result.Distance)
		}
	}
}
