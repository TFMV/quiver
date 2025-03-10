package api

import (
	"github.com/TFMV/quiver/dimreduce"
	"github.com/TFMV/quiver/extensions"
	"github.com/gofiber/fiber/v2"
	"go.uber.org/zap"
)

// DimReduceRequest represents a request for dimensionality reduction
type DimReduceRequest struct {
	Vectors [][]float32 `json:"vectors"`
	Method  string      `json:"method,omitempty"`
	Target  int         `json:"target_dimension,omitempty"`
}

// DimReduceResponse represents a response from dimensionality reduction
type DimReduceResponse struct {
	ReducedVectors [][]float32 `json:"reduced_vectors"`
}

// RegisterExtensionRoutes registers the extension routes with the Fiber app
func RegisterExtensionRoutes(app *fiber.App, manager *extensions.MultiIndexManager, logger *zap.Logger) {
	// Create a router group for extensions
	ext := app.Group("/ext")

	// Register dimensionality reduction routes
	ext.Post("/reduce", dimReduceHandler(manager, logger))
	ext.Post("/adaptive-reduce", adaptiveDimReduceHandler(manager, logger))

	// Register semantic routing routes
	ext.Post("/route", routeHandler(manager, logger))
	ext.Post("/multi-search", multiSearchHandler(manager, logger))
}

// dimReduceHandler handles dimensionality reduction requests
func dimReduceHandler(manager *extensions.MultiIndexManager, logger *zap.Logger) fiber.Handler {
	return func(c *fiber.Ctx) error {
		var req DimReduceRequest
		if err := c.BodyParser(&req); err != nil {
			logger.Error("Failed to parse request", zap.Error(err))
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error":   true,
				"message": "Invalid request format",
			})
		}

		if len(req.Vectors) == 0 {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error":   true,
				"message": "No vectors provided",
			})
		}

		// Create a dimensionality reducer
		config := dimreduce.DefaultDimReducerConfig()
		if req.Target > 0 {
			config.TargetDimension = req.Target
		}
		if req.Method != "" {
			config.Method = dimreduce.ReductionMethod(req.Method)
		}
		config.Logger = logger

		reducer, err := dimreduce.NewDimReducer(config)
		if err != nil {
			logger.Error("Failed to create dimensionality reducer", zap.Error(err))
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
				"error":   true,
				"message": "Failed to create dimensionality reducer: " + err.Error(),
			})
		}

		// Reduce the vectors
		reducedVectors, err := reducer.Reduce(req.Vectors)
		if err != nil {
			logger.Error("Failed to reduce vectors", zap.Error(err))
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
				"error":   true,
				"message": "Failed to reduce vectors: " + err.Error(),
			})
		}

		return c.JSON(DimReduceResponse{
			ReducedVectors: reducedVectors,
		})
	}
}

// adaptiveDimReduceHandler handles adaptive dimensionality reduction requests
func adaptiveDimReduceHandler(manager *extensions.MultiIndexManager, logger *zap.Logger) fiber.Handler {
	return func(c *fiber.Ctx) error {
		var req DimReduceRequest
		if err := c.BodyParser(&req); err != nil {
			logger.Error("Failed to parse request", zap.Error(err))
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error":   true,
				"message": "Invalid request format",
			})
		}

		if len(req.Vectors) == 0 {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error":   true,
				"message": "No vectors provided",
			})
		}

		// Create a dimensionality reducer with adaptive settings
		config := dimreduce.DefaultDimReducerConfig()
		config.Adaptive = true
		if req.Method != "" {
			config.Method = dimreduce.ReductionMethod(req.Method)
		}
		config.Logger = logger

		reducer, err := dimreduce.NewDimReducer(config)
		if err != nil {
			logger.Error("Failed to create dimensionality reducer", zap.Error(err))
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
				"error":   true,
				"message": "Failed to create dimensionality reducer: " + err.Error(),
			})
		}

		// Reduce the vectors adaptively
		reducedVectors, err := reducer.AdaptiveReduce(req.Vectors)
		if err != nil {
			logger.Error("Failed to reduce vectors adaptively", zap.Error(err))
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
				"error":   true,
				"message": "Failed to reduce vectors adaptively: " + err.Error(),
			})
		}

		return c.JSON(DimReduceResponse{
			ReducedVectors: reducedVectors,
		})
	}
}

// RouteRequest represents a request for semantic routing
type RouteRequest struct {
	Vector []float32 `json:"vector"`
}

// RouteResponse represents a response from semantic routing
type RouteResponse struct {
	TargetIndex        string             `json:"target_index"`
	Confidence         float32            `json:"confidence"`
	AlternativeIndices map[string]float32 `json:"alternative_indices,omitempty"`
	DecisionID         string             `json:"decision_id"`
}

// routeHandler handles semantic routing requests
func routeHandler(manager *extensions.MultiIndexManager, logger *zap.Logger) fiber.Handler {
	return func(c *fiber.Ctx) error {
		var req RouteRequest
		if err := c.BodyParser(&req); err != nil {
			logger.Error("Failed to parse request", zap.Error(err))
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error":   true,
				"message": "Invalid request format",
			})
		}

		if len(req.Vector) == 0 {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error":   true,
				"message": "Vector is required",
			})
		}

		// Route the query
		decision, err := manager.Route(c.Context(), req.Vector)
		if err != nil {
			logger.Error("Failed to route query", zap.Error(err))
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
				"error":   true,
				"message": "Failed to route query: " + err.Error(),
			})
		}

		// Convert alternative indices to string map
		alternatives := make(map[string]float32)
		for k, v := range decision.AlternativeIndices {
			alternatives[string(k)] = v
		}

		return c.JSON(RouteResponse{
			TargetIndex:        string(decision.TargetIndex),
			Confidence:         decision.Confidence,
			AlternativeIndices: alternatives,
			DecisionID:         decision.DecisionID,
		})
	}
}

// MultiSearchRequest represents a request for multi-index search
type MultiSearchRequest struct {
	Vector []float32 `json:"vector"`
	K      int       `json:"k"`
}

// multiSearchHandler handles multi-index search requests
func multiSearchHandler(manager *extensions.MultiIndexManager, logger *zap.Logger) fiber.Handler {
	return func(c *fiber.Ctx) error {
		var req MultiSearchRequest
		if err := c.BodyParser(&req); err != nil {
			logger.Error("Failed to parse request", zap.Error(err))
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error":   true,
				"message": "Invalid request format",
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

		// Search across multiple indices
		results, err := manager.MultiIndexSearch(c.Context(), req.Vector, req.K)
		if err != nil {
			logger.Error("Failed to search across indices", zap.Error(err))
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
				"error":   true,
				"message": "Failed to search across indices: " + err.Error(),
			})
		}

		return c.JSON(fiber.Map{
			"results": results,
		})
	}
}
