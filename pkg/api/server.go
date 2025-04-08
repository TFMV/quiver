package api

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/TFMV/quiver/pkg/core"
	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// ServerConfig holds configuration options for the API server
type ServerConfig struct {
	// Host is the server host (default: localhost)
	Host string
	// Port is the server port (default: 8080)
	Port int
	// AllowedOrigins is a list of CORS allowed origins
	AllowedOrigins []string
	// EnableMetrics enables Prometheus metrics endpoint
	EnableMetrics bool
	// MetricsPort is the port for the metrics server (default: 9090)
	MetricsPort int
	// ReadTimeout is the maximum duration for reading the entire request (default: 5s)
	ReadTimeout time.Duration
	// WriteTimeout is the maximum duration before timing out writes of the response (default: 10s)
	WriteTimeout time.Duration
	// ShutdownTimeout is the maximum duration to wait for server shutdown (default: 30s)
	ShutdownTimeout time.Duration
	// RateLimit is the number of requests per minute allowed per client (default: 60)
	RateLimit int
	// JWTSecret is used for API token authentication (if empty, JWT auth is disabled)
	JWTSecret string
	// LogLevel controls the verbosity of logging (debug, info, warn, error)
	LogLevel string
}

// DefaultServerConfig returns default configuration options
func DefaultServerConfig() ServerConfig {
	return ServerConfig{
		Host:            "localhost",
		Port:            8080,
		AllowedOrigins:  []string{"*"},
		EnableMetrics:   true,
		MetricsPort:     9090,
		ReadTimeout:     5 * time.Second,
		WriteTimeout:    10 * time.Second,
		ShutdownTimeout: 30 * time.Second,
		RateLimit:       60,
		LogLevel:        "info",
	}
}

// Server represents the API server for Quiver
type Server struct {
	config      ServerConfig
	db          *core.DB
	router      *gin.Engine
	httpServer  *http.Server
	metricsHTTP *http.Server
	handlers    *Handlers
}

// NewServer creates a new API server
func NewServer(db *core.DB, config ServerConfig) *Server {
	// Set sensible defaults for zero values
	if config.Host == "" {
		config.Host = "localhost"
	}
	if config.Port == 0 {
		config.Port = 8080
	}
	if config.MetricsPort == 0 {
		config.MetricsPort = 9090
	}
	if config.ReadTimeout == 0 {
		config.ReadTimeout = 5 * time.Second
	}
	if config.WriteTimeout == 0 {
		config.WriteTimeout = 10 * time.Second
	}
	if config.ShutdownTimeout == 0 {
		config.ShutdownTimeout = 30 * time.Second
	}
	if config.RateLimit == 0 {
		config.RateLimit = 60
	}
	if len(config.AllowedOrigins) == 0 {
		config.AllowedOrigins = []string{"*"}
	}
	if config.LogLevel == "" {
		config.LogLevel = "info"
	}

	// Set Gin mode based on log level
	if config.LogLevel == "debug" {
		gin.SetMode(gin.DebugMode)
	} else {
		gin.SetMode(gin.ReleaseMode)
	}

	// Create router
	router := gin.Default()

	// Setup CORS
	corsConfig := cors.DefaultConfig()
	corsConfig.AllowOrigins = config.AllowedOrigins
	corsConfig.AllowMethods = []string{"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"}
	corsConfig.AllowHeaders = []string{"Origin", "Content-Type", "Authorization"}
	router.Use(cors.New(corsConfig))

	// Create server
	server := &Server{
		config: config,
		db:     db,
		router: router,
		httpServer: &http.Server{
			Addr:         fmt.Sprintf("%s:%d", config.Host, config.Port),
			Handler:      router,
			ReadTimeout:  config.ReadTimeout,
			WriteTimeout: config.WriteTimeout,
		},
	}

	// Create API handlers
	server.handlers = NewHandlers(db)

	// Set up metrics server if enabled
	if config.EnableMetrics {
		mux := http.NewServeMux()
		mux.Handle("/metrics", promhttp.Handler())
		server.metricsHTTP = &http.Server{
			Addr:    fmt.Sprintf("%s:%d", config.Host, config.MetricsPort),
			Handler: mux,
		}
	}

	// Setup routes
	server.setupRoutes()

	return server
}

// setupRoutes configures all API routes
func (s *Server) setupRoutes() {
	// API version group
	v1 := s.router.Group("/api/v1")

	// Health check
	v1.GET("/health", s.handlers.HealthCheck)

	// Database-wide endpoints
	v1.GET("/collections", s.handlers.ListCollections)
	v1.POST("/collections", s.handlers.CreateCollection)
	v1.GET("/metrics", s.handlers.GetMetrics)
	v1.POST("/backup", s.handlers.CreateBackup)
	v1.POST("/restore", s.handlers.RestoreBackup)

	// Collection-specific endpoints
	collection := v1.Group("/collections/:collection")
	{
		collection.GET("", s.handlers.GetCollection)
		collection.DELETE("", s.handlers.DeleteCollection)
		collection.GET("/stats", s.handlers.GetCollectionStats)

		// Vector operations
		collection.POST("/vectors", s.handlers.AddVector)
		collection.POST("/vectors/batch", s.handlers.AddVectorBatch)
		collection.GET("/vectors/:id", s.handlers.GetVector)
		collection.PUT("/vectors/:id", s.handlers.UpdateVector)
		collection.DELETE("/vectors/:id", s.handlers.DeleteVector)
		collection.POST("/vectors/delete/batch", s.handlers.DeleteVectorBatch)

		// Search
		collection.POST("/search", s.handlers.Search)
	}
}

// Start starts the HTTP server
func (s *Server) Start() {
	// Start metrics server if enabled
	if s.config.EnableMetrics && s.metricsHTTP != nil {
		go func() {
			log.Printf("Starting metrics server on %s", s.metricsHTTP.Addr)
			if err := s.metricsHTTP.ListenAndServe(); err != nil && err != http.ErrServerClosed {
				log.Fatalf("Failed to start metrics server: %v", err)
			}
		}()
	}

	// Start main API server
	go func() {
		log.Printf("Starting API server on %s", s.httpServer.Addr)
		if err := s.httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Failed to start server: %v", err)
		}
	}()

	// Wait for interrupt signal to gracefully shut down the server
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Println("Shutting down server...")

	// Create shutdown context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), s.config.ShutdownTimeout)
	defer cancel()

	// Shutdown metrics server if it was started
	if s.config.EnableMetrics && s.metricsHTTP != nil {
		if err := s.metricsHTTP.Shutdown(ctx); err != nil {
			log.Printf("Metrics server forced to shutdown: %v", err)
		}
	}

	// Attempt to gracefully shutdown the server
	if err := s.httpServer.Shutdown(ctx); err != nil {
		log.Printf("Server forced to shutdown: %v", err)
	}

	log.Println("Server exiting")
}

// GetAddr returns the server's address as a string
func (s *Server) GetAddr() string {
	return fmt.Sprintf("%s:%d", s.config.Host, s.config.Port)
}

// GetMetricsAddr returns the metrics server's address as a string
func (s *Server) GetMetricsAddr() string {
	return fmt.Sprintf("%s:%d", s.config.Host, s.config.MetricsPort)
}

// SetJWTSecret sets the JWT secret for authentication
func (s *Server) SetJWTSecret(secret string) {
	s.config.JWTSecret = secret
	// Re-setup routes with authentication middleware if a secret is provided
	if secret != "" {
		s.setupRoutes()
	}
}

// GetRouter returns the Gin router for testing
func (s *Server) GetRouter() *gin.Engine {
	return s.router
}

// GetConfig returns the server configuration
func (s *Server) GetConfig() ServerConfig {
	return s.config
}
