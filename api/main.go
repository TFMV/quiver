// Package api provides a REST API server for the Quiver vector database
package api

import (
	"flag"
	"os"

	"github.com/TFMV/hnsw"
	"github.com/TFMV/hnsw/hnsw-extensions/hybrid"
	quiver "github.com/TFMV/quiver"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

// RunServer initializes and starts the API server
func RunServer() {
	// Parse command line flags
	port := flag.String("port", "8080", "Port to listen on")
	debug := flag.Bool("debug", false, "Enable debug logging")
	storagePath := flag.String("storage-path", "./data", "Path to store index data")
	flag.Parse()

	// Setup logger
	logger := setupLogger(*debug)
	defer logger.Sync()

	// Ensure storage directory exists
	if err := os.MkdirAll(*storagePath, 0755); err != nil {
		logger.Fatal("Failed to create storage directory", zap.Error(err))
	}

	// Create index configuration
	config := quiver.DefaultDBConfig()
	config.BaseDir = *storagePath
	config.Hybrid.Type = hybrid.HybridIndexType
	config.Hybrid.Distance = hnsw.CosineDistance
	config.Hybrid.M = 16
	config.Hybrid.EfSearch = 100

	// Create or load the index
	logger.Info("Initializing Quiver index",
		zap.String("storage_path", config.BaseDir),
	)

	index, err := quiver.NewVectorDB[uint64](config)
	if err != nil {
		logger.Fatal("Failed to create index", zap.Error(err))
	}
	defer index.Close()

	// Create and start the server
	server := NewServer(index, logger)

	logger.Info("Starting API server", zap.String("port", *port))
	if err := server.Start(*port); err != nil {
		logger.Fatal("Server error", zap.Error(err))
	}
}

// setupLogger configures and returns a zap logger
func setupLogger(debug bool) *zap.Logger {
	config := zap.NewProductionConfig()

	// Set the log level based on the debug flag
	if debug {
		config.Level = zap.NewAtomicLevelAt(zapcore.DebugLevel)
	} else {
		config.Level = zap.NewAtomicLevelAt(zapcore.InfoLevel)
	}

	// Configure the encoder
	config.EncoderConfig.TimeKey = "time"
	config.EncoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder
	config.EncoderConfig.EncodeLevel = zapcore.CapitalLevelEncoder

	// Create the logger
	logger, err := config.Build()
	if err != nil {
		panic(err)
	}

	return logger
}

// This is just an example of how to use the API server
// In a real application, you would call RunServer() from your main package
func Example_main() {
	RunServer()
}
