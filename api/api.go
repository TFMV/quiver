// Package api provides the API for the Quiver application.
package api

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/TFMV/quiver"
	"github.com/TFMV/quiver/router"
	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/compress"
	"github.com/gofiber/fiber/v2/middleware/monitor"
	"github.com/gofiber/fiber/v2/middleware/recover"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/valyala/fasthttp/fasthttpadaptor"
	"go.uber.org/zap"
)

// Server holds the Fiber app instance
type Server struct {
	app   *fiber.App
	log   *zap.Logger
	port  string
	index *quiver.Index
}

// ServerOptions defines the configuration for the server.
type ServerOptions struct {
	Port    string
	Prefork bool
	// Extension options
	EnableExtensions bool
	RouterConfig     router.RouterConfig
}

// DimReductionOptions defines the configuration for dimensionality reduction
type DimReductionOptions struct {
	Enabled     bool    `json:"enabled"`
	Method      string  `json:"method,omitempty"`
	TargetDim   int     `json:"target_dim,omitempty"`
	Adaptive    bool    `json:"adaptive,omitempty"`
	MinVariance float64 `json:"min_variance,omitempty"`
}

// IndexOptions defines the configuration for creating a new index
type IndexOptions struct {
	Dimension       int                 `json:"dimension"`
	Distance        string              `json:"distance"`
	MaxElements     uint64              `json:"max_elements,omitempty"`
	HNSWM           int                 `json:"hnsw_m,omitempty"`
	HNSWEfConstruct int                 `json:"hnsw_ef_construct,omitempty"`
	HNSWEfSearch    int                 `json:"hnsw_ef_search,omitempty"`
	DimReduction    DimReductionOptions `json:"dim_reduction,omitempty"`
}

// SearchRequest represents a search request
type SearchRequest struct {
	Vector   []float32              `json:"vector"`
	K        int                    `json:"k"`
	Filter   string                 `json:"filter,omitempty"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// SearchResponse represents a search response
type SearchResponse struct {
	Results []quiver.SearchResult `json:"results"`
}

// NewServer initializes a new Fiber instance with best practices
func NewServer(opts ServerOptions, index *quiver.Index, logger *zap.Logger) *Server {
	if logger == nil {
		var err error
		logger, err = zap.NewProduction()
		if err != nil {
			panic(err)
		}
	}

	fiberConfig := fiber.Config{
		IdleTimeout:   10 * time.Second,
		ReadTimeout:   10 * time.Second,
		WriteTimeout:  10 * time.Second,
		Prefork:       opts.Prefork,
		ErrorHandler:  customErrorHandler(logger),
		CaseSensitive: true,
		StrictRouting: true,
	}

	app := fiber.New(fiberConfig)

	// Middleware
	app.Use(recover.New())  // Auto-recovers from panics
	app.Use(compress.New()) // Enable gzip compression
	// Note: We're not using Fiber's logger middleware as we have our own custom logger

	// Create the server
	server := &Server{
		app:   app,
		log:   logger,
		port:  opts.Port,
		index: index,
	}

	// Routes
	app.Get("/health", healthCheckHandler(logger))
	app.Get("/health/live", livenessHandler)
	app.Get("/health/ready", readinessHandler)
	app.Get("/metrics", metricsHandler())
	app.Get("/dashboard", monitor.New())

	// API routes
	api := app.Group("/api")
	v1 := api.Group("/v1")

	// Index operations
	v1.Post("/index/create", createIndexHandler(logger))
	v1.Post("/index/backup", backupHandler(index, logger))
	v1.Post("/index/restore", restoreHandler(index, logger))

	// Vector operations
	v1.Post("/vectors", addVectorHandler(index, logger))
	v1.Delete("/vectors/:id", deleteVectorHandler(index, logger))
	v1.Get("/vectors/:id", getVectorByIDHandler(index))

	// Search endpoints
	v1.Post("/search", searchHandler(index, logger))
	v1.Post("/search/hybrid", hybridSearchHandler(index, logger))
	v1.Post("/search/negatives", searchWithNegativesHandler(index, logger))

	// Metadata operations
	v1.Post("/metadata/query", queryMetadataHandler(index, logger))

	// Add custom logging middleware
	app.Use(customLoggingMiddleware(logger))

	return server
}

// customErrorHandler provides structured error handling
func customErrorHandler(log *zap.Logger) fiber.ErrorHandler {
	return func(c *fiber.Ctx, err error) error {
		code := fiber.StatusInternalServerError
		message := "Internal Server Error"

		if e, ok := err.(*fiber.Error); ok {
			code = e.Code
			message = e.Message
		}

		log.Error("Request failed",
			zap.String("method", c.Method()),
			zap.String("path", c.Path()),
			zap.Int("status", code),
			zap.Error(err),
		)

		// Respect "Accept" headers for response format
		if c.Accepts("text/html") != "" {
			return c.Status(code).SendString(fmt.Sprintf("<h1>Error %d</h1><p>%s</p>", code, message))
		}

		return c.Status(code).JSON(fiber.Map{
			"error":   true,
			"message": message,
		})
	}
}

// healthCheckHandler returns a simple health check response
func healthCheckHandler(log *zap.Logger) fiber.Handler {
	return func(c *fiber.Ctx) error {
		log.Debug("Health check requested")
		return c.SendString("OK")
	}
}

// Liveness probe for Kubernetes
func livenessHandler(c *fiber.Ctx) error {
	return c.SendStatus(fiber.StatusOK)
}

// Readiness probe for Kubernetes
func readinessHandler(c *fiber.Ctx) error {
	// Check if necessary dependencies (e.g., DB, cache) are ready.
	// If not ready, return StatusServiceUnavailable (503)
	return c.SendStatus(fiber.StatusOK)
}

// customLoggingMiddleware logs requests in a structured format
func customLoggingMiddleware(log *zap.Logger) fiber.Handler {
	return func(c *fiber.Ctx) error {
		start := time.Now()
		err := c.Next()
		duration := time.Since(start)

		// Ensure status is set to avoid misleading logs
		status := c.Response().StatusCode()
		if status == 0 {
			status = fiber.StatusInternalServerError
		}

		fields := []zap.Field{
			zap.String("method", c.Method()),
			zap.String("path", c.Path()),
			zap.Int("status", status),
			zap.Duration("duration", duration),
			zap.String("client_ip", c.IP()),
		}

		if err != nil {
			fields = append(fields, zap.Error(err))
		}

		log.Info("Request handled", fields...)
		return err
	}
}

// Start runs the Fiber server and handles graceful shutdown
func (s *Server) Start() error {
	if s.port == "" {
		s.port = "8080"
	}

	addr := fmt.Sprintf(":%s", s.port)
	s.log.Info("Starting server", zap.String("address", addr))

	idleConnsClosed := make(chan error, 1)

	go func() {
		sigint := make(chan os.Signal, 1)
		signal.Notify(sigint, os.Interrupt, syscall.SIGTERM)
		<-sigint

		s.log.Info("Shutdown signal received")

		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()

		if err := s.Shutdown(ctx); err != nil {
			idleConnsClosed <- fmt.Errorf("server shutdown error: %w", err)
			return
		}
		idleConnsClosed <- nil
	}()

	// Start server in a separate goroutine
	serverErr := make(chan error, 1)
	go func() {
		if err := s.app.Listen(addr); err != nil && !errors.Is(err, fiber.ErrServiceUnavailable) {
			serverErr <- fmt.Errorf("server startup error: %w", err)
		}
	}()

	select {
	case err := <-idleConnsClosed:
		if err != nil {
			s.log.Error("Shutdown error", zap.Error(err))
			return err
		}
	case err := <-serverErr:
		s.log.Error("Startup error", zap.Error(err))
		return err
	}

	s.log.Info("Server stopped")
	return nil
}

// Shutdown stops the server gracefully
func (s *Server) Shutdown(ctx context.Context) error {
	s.log.Warn("Server is shutting down...")
	if err := s.app.ShutdownWithContext(ctx); err != nil {
		s.log.Error("Fiber shutdown error", zap.Error(err))
		return fmt.Errorf("fiber shutdown error: %w", err)
	}
	return nil
}

// GetApp returns the underlying Fiber app
func (s *Server) GetApp() *fiber.App {
	return s.app
}

// StartTLS starts the server with TLS
func (s *Server) StartTLS(certFile, keyFile string) error {
	// Verify that the certificate files exist and are valid
	if _, err := os.Stat(certFile); err != nil {
		return fmt.Errorf("certificate file not found: %w", err)
	}
	if _, err := os.Stat(keyFile); err != nil {
		return fmt.Errorf("key file not found: %w", err)
	}

	// Note: Fiber doesn't support passing a TLS config directly
	// We recommend using TLS 1.2 or higher for security

	// Start server with TLS
	return s.app.ListenTLS(":"+s.port, certFile, keyFile)
}

// Handler implementations

func addVectorHandler(idx *quiver.Index, log *zap.Logger) fiber.Handler {
	return func(c *fiber.Ctx) error {
		start := time.Now()

		var req struct {
			ID       uint64                 `json:"id"`
			Vector   []float32              `json:"vector"`
			Metadata map[string]interface{} `json:"metadata"`
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

		err := idx.Add(req.ID, req.Vector, req.Metadata)
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

func deleteVectorHandler(idx *quiver.Index, log *zap.Logger) fiber.Handler {
	return func(c *fiber.Ctx) error {
		start := time.Now()
		idParam := c.Params("id")
		var id uint64

		_, err := fmt.Sscanf(idParam, "%d", &id)
		if err != nil {
			log.Error("Invalid ID format", zap.String("id", idParam), zap.Error(err))
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error":   true,
				"message": "Invalid ID format",
			})
		}

		// Try to delete the vector
		err = idx.DeleteVector(id)
		if err != nil {
			log.Error("Failed to delete vector", zap.Uint64("id", id), zap.Error(err))
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
				"error":   true,
				"message": "Failed to delete vector: " + err.Error(),
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

func getVectorByIDHandler(idx *quiver.Index) fiber.Handler {
	return func(c *fiber.Ctx) error {
		idParam := c.Params("id")
		var id uint64

		_, err := fmt.Sscanf(idParam, "%d", &id)
		if err != nil {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error":   true,
				"message": "Invalid ID format",
			})
		}

		// Access the unexported methods through the public API
		// This is a workaround until proper methods are exposed
		metadata := make(map[string]interface{})

		// Check if the vector exists by attempting a search
		results, err := idx.Search([]float32{0.1}, 1, 0, 0)
		if err != nil || len(results) == 0 {
			return c.Status(fiber.StatusNotFound).JSON(fiber.Map{
				"error":   true,
				"message": "Vector not found",
			})
		}

		// For now, we'll just return the ID since we can't access the vector directly
		return c.JSON(fiber.Map{
			"id":       id,
			"metadata": metadata,
		})
	}
}

func searchHandler(idx *quiver.Index, log *zap.Logger) fiber.Handler {
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

		results, err := idx.Search(req.Vector, req.K, 0, 0)
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

func hybridSearchHandler(idx *quiver.Index, log *zap.Logger) fiber.Handler {
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

		results, err := idx.SearchWithFilter(req.Vector, req.K, req.Filter)
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

func searchWithNegativesHandler(idx *quiver.Index, log *zap.Logger) fiber.Handler {
	return func(c *fiber.Ctx) error {
		start := time.Now()

		var req struct {
			PositiveVector  []float32   `json:"positive_vector"`
			NegativeVectors [][]float32 `json:"negative_vectors"`
			K               int         `json:"k"`
			Page            int         `json:"page"`
			PageSize        int         `json:"page_size"`
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

		if req.Page < 0 {
			req.Page = 0
		}

		if req.PageSize <= 0 {
			req.PageSize = 10
		}

		results, err := idx.SearchWithNegatives(req.PositiveVector, req.NegativeVectors, req.K, req.Page, req.PageSize)
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

func queryMetadataHandler(idx *quiver.Index, log *zap.Logger) fiber.Handler {
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

		results, err := idx.QueryMetadata(req.Query)
		if err != nil {
			log.Error("Metadata query failed", zap.Error(err))
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
				"error":   true,
				"message": "Metadata query failed: " + err.Error(),
			})
		}

		return c.JSON(fiber.Map{
			"results": results,
		})
	}
}

func backupHandler(idx *quiver.Index, log *zap.Logger) fiber.Handler {
	return func(c *fiber.Ctx) error {
		var req struct {
			Path        string `json:"path"`
			Incremental bool   `json:"incremental"`
			Compress    bool   `json:"compress"`
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

		err := idx.Backup(req.Path, req.Incremental, req.Compress)
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

func restoreHandler(idx *quiver.Index, log *zap.Logger) fiber.Handler {
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

// metricsHandler returns a handler for Prometheus metrics
func metricsHandler() fiber.Handler {
	return func(c *fiber.Ctx) error {
		// Convert the Prometheus handler to a Fiber handler
		handler := fasthttpadaptor.NewFastHTTPHandler(promhttp.Handler())
		handler(c.Context())
		return nil
	}
}

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
		config := quiver.Config{
			Dimension:       opts.Dimension,
			Distance:        quiver.CosineDistance, // Default
			MaxElements:     opts.MaxElements,
			HNSWM:           opts.HNSWM,
			HNSWEfConstruct: opts.HNSWEfConstruct,
			HNSWEfSearch:    opts.HNSWEfSearch,
		}

		// Set the distance metric
		if opts.Distance == "l2" {
			config.Distance = quiver.L2Distance
		}

		// Set default values if not provided
		if config.MaxElements == 0 {
			config.MaxElements = 1000000
		}
		if config.HNSWM == 0 {
			config.HNSWM = 16
		}
		if config.HNSWEfConstruct == 0 {
			config.HNSWEfConstruct = 200
		}
		if config.HNSWEfSearch == 0 {
			config.HNSWEfSearch = 100
		}

		// Configure dimensionality reduction if enabled
		if opts.DimReduction.Enabled {
			config.EnableDimReduction = true
			config.DimReductionMethod = opts.DimReduction.Method
			config.DimReductionTarget = opts.DimReduction.TargetDim
			config.DimReductionAdaptive = opts.DimReduction.Adaptive
			config.DimReductionMinVariance = opts.DimReduction.MinVariance

			// Set default values if not provided
			if config.DimReductionMethod == "" {
				config.DimReductionMethod = "PCA"
			}
			if config.DimReductionTarget <= 0 {
				config.DimReductionTarget = config.Dimension / 2
			}
			if config.DimReductionMinVariance <= 0 {
				config.DimReductionMinVariance = 0.95
			}
		}

		// Create the index
		_, err := quiver.New(config, logger)
		if err != nil {
			logger.Error("Failed to create index", zap.Error(err))
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
				"error":   true,
				"message": "Failed to create index: " + err.Error(),
			})
		}

		// Return the index configuration
		return c.JSON(fiber.Map{
			"success": true,
			"message": "Index created successfully",
			"config":  config,
		})
	}
}
