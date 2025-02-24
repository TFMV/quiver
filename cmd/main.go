package main

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/TFMV/quiver"
	"github.com/TFMV/quiver/api"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"go.uber.org/zap"
)

var (
	logger  *zap.Logger
	cfgFile string
)

func init() {
	cobra.OnInitialize(initConfig)
	rootCmd.PersistentFlags().StringVar(&cfgFile, "config", "", "config file (default is ./config.yaml)")
}

// rootCmd represents the base command
var rootCmd = &cobra.Command{
	Use:   "quiver",
	Short: "Quiver is a high-performance vector database",
	Long: `Quiver is a lightweight, high-performance vector search engine 
designed for structured datasets. It uses HNSW for efficient vector 
indexing and DuckDB for metadata storage.`,
}

// serveCmd represents the serve command
var serveCmd = &cobra.Command{
	Use:   "serve",
	Short: "Start the Quiver server",
	RunE:  runServer,
}

// statusCmd represents the status command
var statusCmd = &cobra.Command{
	Use:   "status",
	Short: "Check Quiver server status",
	RunE:  checkStatus,
}

// backupCmd represents the backup command
var backupCmd = &cobra.Command{
	Use:   "backup [path]",
	Short: "Backup the index and metadata",
	Args:  cobra.ExactArgs(1),
	RunE:  runBackup,
}

// restoreCmd represents the restore command
var restoreCmd = &cobra.Command{
	Use:   "restore [path]",
	Short: "Restore the index from backup",
	Args:  cobra.ExactArgs(1),
	RunE:  runRestore,
}

func initConfig() {
	if cfgFile != "" {
		viper.SetConfigFile(cfgFile)
	} else {
		viper.SetConfigName("config")
		viper.AddConfigPath(".")
	}

	viper.SetEnvPrefix("QUIVER")
	viper.AutomaticEnv()

	if err := viper.ReadInConfig(); err == nil {
		logger.Info("Using config file", zap.String("file", viper.ConfigFileUsed()))
	}
}

func runServer(cmd *cobra.Command, args []string) error {
	port := viper.GetString("server.port")
	if port == "" {
		port = "8080"
	}

	// Set default index configuration
	viper.SetDefault("index.dimension", 128)
	viper.SetDefault("index.storage_path", "quiver.db")
	viper.SetDefault("index.max_elements", 100000)
	viper.SetDefault("index.hnsw_m", 32)
	viper.SetDefault("index.ef_construction", 200)
	viper.SetDefault("index.ef_search", 200)
	viper.SetDefault("index.batch_size", 1000)
	viper.SetDefault("index.distance", int(quiver.Cosine))

	// Create the vector index
	index, err := quiver.New(quiver.Config{
		Dimension:       viper.GetInt("index.dimension"),
		StoragePath:     viper.GetString("index.storage_path"),
		MaxElements:     uint64(viper.GetInt("index.max_elements")),
		HNSWM:           viper.GetInt("index.hnsw_m"),
		HNSWEfConstruct: viper.GetInt("index.ef_construction"),
		HNSWEfSearch:    viper.GetInt("index.ef_search"),
		BatchSize:       viper.GetInt("index.batch_size"),
		Distance:        quiver.DistanceMetric(viper.GetInt("index.distance")),
	}, logger)
	if err != nil {
		logger.Error("Failed to create index", zap.Error(err))
		return fmt.Errorf("failed to create index: %w", err)
	}
	defer func() {
		if err := index.Close(); err != nil {
			logger.Error("Failed to close index", zap.Error(err))
		}
	}()

	server := api.NewServer(api.ServerOptions{Port: port}, index, logger)

	// Handle graceful shutdown
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	// Channel for startup errors
	startupErr := make(chan error, 1)
	// Channel for shutdown completion
	shutdownComplete := make(chan struct{})

	// Start server
	go func() {
		logger.Info("Starting server", zap.String("port", port))
		if err := server.Start(); err != nil {
			if !errors.Is(err, http.ErrServerClosed) {
				logger.Error("Server startup failed", zap.Error(err))
				startupErr <- err
				return
			}
		}
	}()

	// Handle shutdown
	go func() {
		<-ctx.Done()
		logger.Info("Shutdown signal received")

		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		if err := server.Shutdown(shutdownCtx); err != nil {
			logger.Error("Server shutdown error", zap.Error(err))
		} else {
			logger.Info("Server shutdown completed successfully")
		}
		close(shutdownComplete)
	}()

	// Wait for either startup error or shutdown completion
	select {
	case err := <-startupErr:
		return fmt.Errorf("server startup failed: %w", err)
	case <-shutdownComplete:
		return nil
	}
}

func checkStatus(cmd *cobra.Command, args []string) error {
	port := viper.GetString("server.port")
	if port == "" {
		port = "8080"
	}

	url := fmt.Sprintf("http://localhost:%s/health", port)
	resp, err := http.Get(url)
	if err != nil {
		// Check if it's a connection error
		if errors.Is(err, syscall.ECONNREFUSED) {
			logger.Info("Server is not running")
			return fmt.Errorf("server is not running on port %s", port)
		}
		// For other errors
		logger.Warn("Failed to check server status", zap.Error(err))
		return fmt.Errorf("failed to check server status: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		logger.Warn("Server returned unhealthy status", zap.Int("status_code", resp.StatusCode))
		return fmt.Errorf("server is running but returned unhealthy status: %d", resp.StatusCode)
	}

	logger.Info("Server is running and healthy", zap.String("port", port))
	return nil
}

func runBackup(cmd *cobra.Command, args []string) error {
	logger.Info("Not implemented")
	return nil
}

func runRestore(cmd *cobra.Command, args []string) error {
	logger.Info("Not implemented")
	return nil
}

func main() {
	var err error
	logger, err = zap.NewProduction()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to initialize logger: %v\n", err)
		os.Exit(1)
	}
	defer logger.Sync()

	// Add commands
	rootCmd.AddCommand(serveCmd)
	rootCmd.AddCommand(statusCmd)
	rootCmd.AddCommand(backupCmd)
	rootCmd.AddCommand(restoreCmd)

	if err := rootCmd.Execute(); err != nil {
		// Don't log stack trace for expected errors
		fmt.Fprintln(os.Stderr, "Error:", err)
		os.Exit(1)
	}
}
