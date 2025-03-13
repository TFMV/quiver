// Package api provides the API for the Quiver application.
package api

import (
	"time"

	db "github.com/TFMV/quiver"
	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/cors"
	"github.com/gofiber/fiber/v2/middleware/logger"
	"github.com/gofiber/fiber/v2/middleware/recover"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/valyala/fasthttp/fasthttpadaptor"
	"go.uber.org/zap"
)

// Server represents the API server
type Server struct {
	app   *fiber.App
	index *db.VectorDB[uint64]
	log   *zap.Logger
}

// ServerOptions defines the configuration for the server.
type ServerOptions struct {
	Port    string
	Prefork bool
	// Logging options
	LogToConsole bool
	LogLevel     string
}

// DimReductionOptions defines the configuration for dimensionality reduction
type DimReductionOptions struct {
	Enabled     bool    `json:"enabled"`
	Method      string  `json:"method,omitempty"`
	TargetDim   int     `json:"target_dim,omitempty"`
	Adaptive    bool    `json:"adaptive,omitempty"`
	MinVariance float64 `json:"min_variance,omitempty"`
}

// IndexOptions represents the options for creating a new index
type IndexOptions struct {
	Dimension    int    `json:"dimension"`
	Distance     string `json:"distance"`
	MaxElements  int    `json:"max_elements"`
	HNSWM        int    `json:"hnsw_m"`
	HNSWEfSearch int    `json:"hnsw_ef_search"`
	DimReduction struct {
		Enabled     bool    `json:"enabled"`
		Method      string  `json:"method"`
		TargetDim   int     `json:"target_dim"`
		Adaptive    bool    `json:"adaptive"`
		MinVariance float64 `json:"min_variance"`
	} `json:"dim_reduction"`
}

// SearchRequest represents a search request
type SearchRequest struct {
	Vector []float32 `json:"vector"`
	K      int       `json:"k"`
	Filter string    `json:"filter,omitempty"`
}

// SearchResponse represents a search response
type SearchResponse struct {
	Results []db.SearchResult[uint64] `json:"results"`
}

// NewServer creates a new API server
func NewServer(index *db.VectorDB[uint64], log *zap.Logger) *Server {
	app := fiber.New(fiber.Config{
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 10 * time.Second,
		IdleTimeout:  30 * time.Second,
	})

	// Add middleware
	app.Use(recover.New())
	app.Use(cors.New())
	app.Use(logger.New())

	server := &Server{
		app:   app,
		index: index,
		log:   log,
	}

	// Register routes
	server.registerRoutes()

	return server
}

// registerRoutes registers the API routes
func (s *Server) registerRoutes() {
	// Health check
	s.app.Get("/health", func(c *fiber.Ctx) error {
		return c.JSON(fiber.Map{
			"status": "ok",
			"time":   time.Now().Format(time.RFC3339),
		})
	})

	// Vector operations
	s.app.Post("/vectors", addVectorHandler(s.index, s.log))
	s.app.Delete("/vectors/:id", deleteVectorHandler(s.index, s.log))
	s.app.Get("/vectors/:id", getVectorByIDHandler(s.index))

	// Search operations
	s.app.Post("/search", searchHandler(s.index, s.log))
	s.app.Post("/search/hybrid", hybridSearchHandler(s.index, s.log))
	s.app.Post("/search/negatives", searchWithNegativesHandler(s.index, s.log))
	s.app.Post("/query", queryMetadataHandler(s.index, s.log))

	// Index operations
	s.app.Post("/backup", backupHandler(s.index, s.log))
	s.app.Post("/restore", restoreHandler(s.index, s.log))
	s.app.Post("/index", createIndexHandler(s.log))
}

// Start starts the API server
func (s *Server) Start(port string) error {
	return s.app.Listen(":" + port)
}

// Shutdown gracefully shuts down the API server
func (s *Server) Shutdown() error {
	return s.app.Shutdown()
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
