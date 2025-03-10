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
	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/compress"
	"github.com/gofiber/fiber/v2/middleware/monitor"
	"github.com/gofiber/fiber/v2/middleware/recover"
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
}

// Add these types for request/response handling
type SearchRequest struct {
	Vector   []float32              `json:"vector"`
	K        int                    `json:"k"`
	Filter   string                 `json:"filter,omitempty"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

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

	app := fiber.New(fiber.Config{
		ErrorHandler: customErrorHandler(logger),
	})

	// Middleware
	app.Use(recover.New())  // Auto-recovers from panics
	app.Use(compress.New()) // Enable gzip compression

	// Routes
	app.Get("/health", healthCheckHandler(logger))
	app.Get("/health/live", livenessHandler)
	app.Get("/health/ready", readinessHandler)

	// Metrics endpoint
	app.Get("/metrics", metricsHandler(logger))

	// Monitoring
	app.Get("/dashboard", monitor.New())

	// Add custom logging middleware
	app.Use(customLoggingMiddleware(logger))

	// API routes
	api := app.Group("/api")
	v1 := api.Group("/v1")

	// Search endpoints
	v1.Post("/search", searchHandler(index, logger))
	v1.Post("/search/hybrid", hybridSearchHandler(index, logger))

	// Create server
	server := &Server{
		app:   app,
		log:   logger,
		port:  opts.Port,
		index: index,
	}

	// Start metrics collection in a goroutine
	go server.recordMetrics()

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
		s.port = "3000"
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

// Add these handlers
func searchHandler(idx *quiver.Index, log *zap.Logger) fiber.Handler {
	return func(c *fiber.Ctx) error {
		var req SearchRequest
		if err := c.BodyParser(&req); err != nil {
			return fiber.NewError(fiber.StatusBadRequest, "Invalid request body")
		}

		if len(req.Vector) == 0 {
			return fiber.NewError(fiber.StatusBadRequest, "Vector is required")
		}

		if req.K <= 0 {
			req.K = 10 // Default to 10 results
		}

		// Default to first page with 10 results per page
		page := 1
		pageSize := req.K

		results, err := idx.Search(req.Vector, req.K, page, pageSize)
		if err != nil {
			log.Error("Search failed", zap.Error(err))
			return fiber.NewError(fiber.StatusInternalServerError, "Search failed")
		}

		return c.JSON(SearchResponse{Results: results})
	}
}

func hybridSearchHandler(idx *quiver.Index, log *zap.Logger) fiber.Handler {
	return func(c *fiber.Ctx) error {
		var req SearchRequest
		if err := c.BodyParser(&req); err != nil {
			return fiber.NewError(fiber.StatusBadRequest, "Invalid request body")
		}

		if len(req.Vector) == 0 {
			return fiber.NewError(fiber.StatusBadRequest, "Vector is required")
		}

		if req.K <= 0 {
			req.K = 10 // Default to 10 results
		}

		if req.Filter == "" {
			return fiber.NewError(fiber.StatusBadRequest, "Filter is required for hybrid search")
		}

		results, err := idx.SearchWithFilter(req.Vector, req.K, req.Filter)
		if err != nil {
			log.Error("Hybrid search failed", zap.Error(err))
			return fiber.NewError(fiber.StatusInternalServerError, "Search failed")
		}

		return c.JSON(SearchResponse{Results: results})
	}
}

// StartTLS starts the server with TLS
func (s *Server) StartTLS(certFile, keyFile string) error {
	return s.app.ListenTLS(":"+s.port, certFile, keyFile)
}
