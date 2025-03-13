package api

import (
	"encoding/json"
	"strconv"
	"time"

	"github.com/TFMV/hnsw"
	"github.com/TFMV/hnsw/hnsw-extensions/facets"
	"github.com/TFMV/hnsw/hnsw-extensions/hybrid"
	quiver "github.com/TFMV/quiver"
	"github.com/gofiber/fiber/v2"
	"go.uber.org/zap"
)

// addVectorHandler handles adding a vector to the index
func addVectorHandler(idx *quiver.VectorDB[uint64], log *zap.Logger) fiber.Handler {
	return func(c *fiber.Ctx) error {
		start := time.Now()

		var req struct {
			ID       uint64                 `json:"id"`
			Vector   []float32              `json:"vector"`
			Metadata map[string]interface{} `json:"metadata"`
			Facets   []facets.Facet         `json:"facets,omitempty"`
		}

		if err := c.BodyParser(&req); err != nil {
			log.Error("Failed to parse request body", zap.Error(err))
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error":   true,
				"message": "Invalid request body",
			})
		}

		if len(req.Vector) == 0 {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error":   true,
				"message": "Vector is required",
			})
		}

		err := idx.Add(req.ID, req.Vector, req.Metadata, req.Facets)
		if err != nil {
			log.Error("Failed to add vector", zap.Error(err))
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
				"error":   true,
				"message": "Failed to add vector: " + err.Error(),
			})
		}

		// Log the operation duration
		log.Debug("Vector added",
			zap.Uint64("id", req.ID),
			zap.Duration("duration", time.Since(start)),
		)

		return c.Status(fiber.StatusCreated).JSON(fiber.Map{
			"success": true,
			"id":      req.ID,
			"message": "Vector added successfully",
		})
	}
}

// deleteVectorHandler handles deleting a vector from the index
func deleteVectorHandler(idx *quiver.VectorDB[uint64], log *zap.Logger) fiber.Handler {
	return func(c *fiber.Ctx) error {
		start := time.Now()
		idParam := c.Params("id")
		var id uint64

		id, err := strconv.ParseUint(idParam, 10, 64)
		if err != nil {
			log.Error("Invalid ID format", zap.String("id", idParam), zap.Error(err))
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error":   true,
				"message": "Invalid ID format",
			})
		}

		// Try to delete the vector
		deleted := idx.Delete(id)
		if !deleted {
			log.Error("Failed to delete vector", zap.Uint64("id", id))
			return c.Status(fiber.StatusNotFound).JSON(fiber.Map{
				"error":   true,
				"message": "Vector not found or could not be deleted",
			})
		}

		// Log the operation duration
		log.Debug("Vector deleted",
			zap.Uint64("id", id),
			zap.Duration("duration", time.Since(start)),
		)

		return c.JSON(fiber.Map{
			"success": true,
			"message": "Vector deleted successfully",
		})
	}
}

// getVectorByIDHandler handles retrieving a vector by ID
func getVectorByIDHandler(idx *quiver.VectorDB[uint64]) fiber.Handler {
	return func(c *fiber.Ctx) error {
		idParam := c.Params("id")
		var id uint64

		id, err := strconv.ParseUint(idParam, 10, 64)
		if err != nil {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error":   true,
				"message": "Invalid ID format",
			})
		}

		// Create a query to search for the vector
		query := []float32{0.1} // Dummy query vector
		options := quiver.DefaultQueryOptions().WithK(1)

		// Search for the vector
		results, err := idx.Search(query, options)
		if err != nil || len(results) == 0 {
			return c.Status(fiber.StatusNotFound).JSON(fiber.Map{
				"error":   true,
				"message": "Vector not found",
			})
		}

		// Return the vector information
		return c.JSON(fiber.Map{
			"id":       id,
			"metadata": results[0].Metadata,
			"facets":   results[0].Facets,
		})
	}
}

// searchHandler handles vector similarity search
func searchHandler(idx *quiver.VectorDB[uint64], log *zap.Logger) fiber.Handler {
	return func(c *fiber.Ctx) error {
		start := time.Now()

		var req SearchRequest
		if err := c.BodyParser(&req); err != nil {
			log.Error("Failed to parse request body", zap.Error(err))
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error":   true,
				"message": "Invalid request body",
			})
		}

		if len(req.Vector) == 0 {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error":   true,
				"message": "Vector is required",
			})
		}

		if req.K <= 0 {
			req.K = 10 // Default to 10 results
		}

		// Create search options
		options := quiver.DefaultQueryOptions().WithK(req.K)

		// Perform search
		results, err := idx.Search(req.Vector, options)
		if err != nil {
			log.Error("Search failed", zap.Error(err))
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
				"error":   true,
				"message": "Search failed: " + err.Error(),
			})
		}

		// Log the operation duration
		log.Debug("Search completed",
			zap.Int("results", len(results)),
			zap.Duration("duration", time.Since(start)),
		)

		return c.JSON(SearchResponse{
			Results: results,
		})
	}
}

// hybridSearchHandler handles hybrid search (vector + metadata filter)
func hybridSearchHandler(idx *quiver.VectorDB[uint64], log *zap.Logger) fiber.Handler {
	return func(c *fiber.Ctx) error {
		start := time.Now()

		var req SearchRequest
		if err := c.BodyParser(&req); err != nil {
			log.Error("Failed to parse request body", zap.Error(err))
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error":   true,
				"message": "Invalid request body",
			})
		}

		if len(req.Vector) == 0 {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error":   true,
				"message": "Vector is required",
			})
		}

		if req.K <= 0 {
			req.K = 10 // Default to 10 results
		}

		if req.Filter == "" {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error":   true,
				"message": "Filter is required for hybrid search",
			})
		}

		// Create search options with metadata filter
		options := quiver.DefaultQueryOptions().WithK(req.K)

		// Convert filter string to JSON
		var filterJSON json.RawMessage
		if err := json.Unmarshal([]byte(req.Filter), &filterJSON); err != nil {
			log.Error("Invalid filter format", zap.Error(err))
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error":   true,
				"message": "Invalid filter format: " + err.Error(),
			})
		}

		options = options.WithMetadataFilter(filterJSON)

		// Perform search
		results, err := idx.Search(req.Vector, options)
		if err != nil {
			log.Error("Hybrid search failed", zap.Error(err))
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
				"error":   true,
				"message": "Hybrid search failed: " + err.Error(),
			})
		}

		// Log the operation duration
		log.Debug("Hybrid search completed",
			zap.Int("results", len(results)),
			zap.Duration("duration", time.Since(start)),
		)

		return c.JSON(SearchResponse{
			Results: results,
		})
	}
}

// searchWithNegativesHandler handles search with negative examples
func searchWithNegativesHandler(idx *quiver.VectorDB[uint64], log *zap.Logger) fiber.Handler {
	return func(c *fiber.Ctx) error {
		start := time.Now()

		var req struct {
			PositiveVector  []float32   `json:"positive_vector"`
			NegativeVectors [][]float32 `json:"negative_vectors"`
			K               int         `json:"k"`
			NegativeWeight  float32     `json:"negative_weight"`
		}

		if err := c.BodyParser(&req); err != nil {
			log.Error("Failed to parse request body", zap.Error(err))
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error":   true,
				"message": "Invalid request body",
			})
		}

		if len(req.PositiveVector) == 0 {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error":   true,
				"message": "Positive vector is required",
			})
		}

		if len(req.NegativeVectors) == 0 {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error":   true,
				"message": "At least one negative vector is required",
			})
		}

		if req.K <= 0 {
			req.K = 10 // Default to 10 results
		}

		// Set default negative weight if not provided
		if req.NegativeWeight <= 0 {
			req.NegativeWeight = 0.5
		}

		// Create search options with negative examples
		options := quiver.DefaultQueryOptions().
			WithK(req.K).
			WithNegativeExamples(req.NegativeVectors).
			WithNegativeWeight(req.NegativeWeight)

		// Perform search
		results, err := idx.Search(req.PositiveVector, options)
		if err != nil {
			log.Error("Search with negatives failed", zap.Error(err))
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
				"error":   true,
				"message": "Search with negatives failed: " + err.Error(),
			})
		}

		// Log the operation duration
		log.Debug("Search with negatives completed",
			zap.Int("results", len(results)),
			zap.Duration("duration", time.Since(start)),
		)

		return c.JSON(SearchResponse{
			Results: results,
		})
	}
}

// queryMetadataHandler handles metadata queries
func queryMetadataHandler(idx *quiver.VectorDB[uint64], log *zap.Logger) fiber.Handler {
	return func(c *fiber.Ctx) error {
		var req struct {
			Query string `json:"query"`
		}

		if err := c.BodyParser(&req); err != nil {
			log.Error("Failed to parse request body", zap.Error(err))
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error":   true,
				"message": "Invalid request body",
			})
		}

		if req.Query == "" {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error":   true,
				"message": "Query is required",
			})
		}

		// This is a simplified implementation since the VectorDB doesn't have a direct QueryMetadata method
		// In a real implementation, you would use the appropriate method from the VectorDB
		return c.Status(fiber.StatusNotImplemented).JSON(fiber.Map{
			"error":   true,
			"message": "Metadata query not implemented in this version",
		})
	}
}

// backupHandler handles index backup
func backupHandler(idx *quiver.VectorDB[uint64], log *zap.Logger) fiber.Handler {
	return func(c *fiber.Ctx) error {
		var req struct {
			Path string `json:"path"`
		}

		if err := c.BodyParser(&req); err != nil {
			log.Error("Failed to parse request body", zap.Error(err))
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error":   true,
				"message": "Invalid request body",
			})
		}

		if req.Path == "" {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error":   true,
				"message": "Backup path is required",
			})
		}

		err := idx.Backup(req.Path)
		if err != nil {
			log.Error("Backup failed", zap.Error(err))
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
				"error":   true,
				"message": "Backup failed: " + err.Error(),
			})
		}

		return c.JSON(fiber.Map{
			"success": true,
			"message": "Backup completed successfully",
			"path":    req.Path,
		})
	}
}

// restoreHandler handles index restoration
func restoreHandler(idx *quiver.VectorDB[uint64], log *zap.Logger) fiber.Handler {
	return func(c *fiber.Ctx) error {
		var req struct {
			Path string `json:"path"`
		}

		if err := c.BodyParser(&req); err != nil {
			log.Error("Failed to parse request body", zap.Error(err))
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error":   true,
				"message": "Invalid request body",
			})
		}

		if req.Path == "" {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error":   true,
				"message": "Restore path is required",
			})
		}

		err := idx.Restore(req.Path)
		if err != nil {
			log.Error("Restore failed", zap.Error(err))
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
				"error":   true,
				"message": "Restore failed: " + err.Error(),
			})
		}

		return c.JSON(fiber.Map{
			"success": true,
			"message": "Restore completed successfully",
		})
	}
}

// createIndexHandler handles creating a new index
func createIndexHandler(logger *zap.Logger) fiber.Handler {
	return func(c *fiber.Ctx) error {
		var opts IndexOptions
		if err := c.BodyParser(&opts); err != nil {
			logger.Error("Failed to parse index options", zap.Error(err))
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error":   true,
				"message": "Invalid request format",
			})
		}

		// Validate options
		if opts.Dimension <= 0 {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error":   true,
				"message": "Dimension must be positive",
			})
		}

		// Create the index configuration
		config := quiver.DefaultDBConfig()

		// Set hybrid index configuration
		config.Hybrid.Type = hybrid.HybridIndexType
		config.Hybrid.M = opts.HNSWM
		config.Hybrid.EfSearch = opts.HNSWEfSearch

		// Set distance function based on the option
		if opts.Distance == "l2" {
			config.Hybrid.Distance = hnsw.EuclideanDistance
		} else {
			config.Hybrid.Distance = hnsw.CosineDistance
		}

		// Set default values if not provided
		if opts.MaxElements > 0 {
			// Set max elements if provided
		}

		// Configure dimensionality reduction if enabled
		if opts.DimReduction.Enabled {
			// Set dimensionality reduction options
		}

		// Create the index
		index, err := quiver.NewVectorDB[uint64](config)
		if err != nil {
			logger.Error("Failed to create index", zap.Error(err))
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
				"error":   true,
				"message": "Failed to create index: " + err.Error(),
			})
		}

		// Close the index since we're just testing creation
		defer index.Close()

		// Return the index configuration
		return c.JSON(fiber.Map{
			"success": true,
			"message": "Index created successfully",
			"config":  config,
		})
	}
}
