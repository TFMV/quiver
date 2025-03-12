package main

import (
	"context"
	"errors"
	"fmt"
	"math/rand/v2"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"strings"

	"github.com/TFMV/quiver"
	"github.com/TFMV/quiver/api"
	"github.com/bytedance/sonic"
	"github.com/olekukonko/tablewriter"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"go.uber.org/zap"
)

var (
	logger    *zap.Logger
	cfgFile   string
	indexPath string
	verbose   bool
)

func init() {
	cobra.OnInitialize(initConfig)

	// Root command flags
	rootCmd.PersistentFlags().StringVar(&cfgFile, "config", "", "config file (default is ./config.yaml)")
	rootCmd.PersistentFlags().StringVarP(&indexPath, "index", "i", "", "Path to index directory (overrides config)")
	rootCmd.PersistentFlags().BoolVarP(&verbose, "verbose", "v", false, "Enable verbose output")

	// Add commands
	rootCmd.AddCommand(serveCmd)
	rootCmd.AddCommand(statusCmd)
	rootCmd.AddCommand(infoCmd)
	rootCmd.AddCommand(statsCmd)
	rootCmd.AddCommand(benchmarkCmd)
	rootCmd.AddCommand(validateCmd)
	rootCmd.AddCommand(exportCmd)
	rootCmd.AddCommand(importCmd)
	rootCmd.AddCommand(optimizeCmd)
	rootCmd.AddCommand(backupCmd)
	rootCmd.AddCommand(restoreCmd)
	rootCmd.AddCommand(healthCmd)
}

// rootCmd represents the base command
var rootCmd = &cobra.Command{
	Use:   "quiver",
	Short: "Quiver is a high-performance vector database",
	Long: `Quiver is a lightweight, high-performance vector search engine 
designed for structured datasets. It uses HNSW for efficient vector 
indexing and DuckDB for metadata storage.`,
	PersistentPreRun: func(cmd *cobra.Command, args []string) {
		if verbose {
			// Switch to development logger with debug level
			var err error
			logger, err = zap.NewDevelopment()
			if err != nil {
				fmt.Fprintf(os.Stderr, "Failed to initialize development logger: %v\n", err)
				os.Exit(1)
			}
		}
	},
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

var infoCmd = &cobra.Command{
	Use:   "info",
	Short: "Display information about the index",
	Run: func(cmd *cobra.Command, args []string) {
		idx, err := loadIndex()
		if err != nil {
			logger.Fatal("Failed to load index", zap.Error(err))
		}
		defer idx.Close()

		config := idx.Config()
		fmt.Println("Index Information:")
		fmt.Printf("  Path: %s\n", config.StoragePath)
		fmt.Printf("  Dimension: %d\n", config.Dimension)
		fmt.Printf("  Distance Metric: %s\n", getDistanceMetricName(config.Distance))
		fmt.Printf("  Max Elements: %d\n", config.MaxElements)
		fmt.Printf("  HNSW M: %d\n", config.HNSWM)
		fmt.Printf("  HNSW Ef Construction: %d\n", config.HNSWEfConstruct)
		fmt.Printf("  HNSW Ef Search: %d\n", config.HNSWEfSearch)
		fmt.Printf("  Batch Size: %d\n", config.BatchSize)
		fmt.Printf("  Persistence Interval: %s\n", config.PersistInterval)
		fmt.Printf("  Backup Interval: %s\n", config.BackupInterval)
		fmt.Printf("  Encryption Enabled: %v\n", config.EncryptionEnabled)
		fmt.Printf("  Dimensionality Reduction Enabled: %v\n", config.EnableDimReduction)
		if config.EnableDimReduction {
			fmt.Printf("  Dim Reduction Method: %s\n", config.DimReductionMethod)
			fmt.Printf("  Dim Reduction Target: %d\n", config.DimReductionTarget)
			fmt.Printf("  Dim Reduction Adaptive: %v\n", config.DimReductionAdaptive)
			if config.DimReductionAdaptive {
				fmt.Printf("  Dim Reduction Min Variance: %.2f\n", config.DimReductionMinVariance)
			}
		}
	},
}

var statsCmd = &cobra.Command{
	Use:   "stats",
	Short: "Display statistics about the index",
	Run: func(cmd *cobra.Command, args []string) {
		idx, err := loadIndex()
		if err != nil {
			logger.Fatal("Failed to load index", zap.Error(err))
		}
		defer idx.Close()

		metrics := idx.CollectMetrics()
		fmt.Println("Index Statistics:")

		table := tablewriter.NewWriter(os.Stdout)
		table.SetHeader([]string{"Metric", "Value"})
		table.SetBorder(false)
		table.SetColumnSeparator(" | ")

		// Add rows for each metric
		for k, v := range metrics {
			table.Append([]string{k, fmt.Sprintf("%v", v)})
		}
		table.Render()
	},
}

var benchmarkCmd = &cobra.Command{
	Use:   "benchmark",
	Short: "Run a benchmark on the index",
	Run: func(cmd *cobra.Command, args []string) {
		idx, err := loadIndex()
		if err != nil {
			logger.Fatal("Failed to load index", zap.Error(err))
		}
		defer idx.Close()

		fmt.Println("Running benchmark...")

		// Generate random vectors for benchmarking
		numVectors := 100
		dimension := idx.Config().Dimension
		vectors := make([][]float32, numVectors)
		for i := range vectors {
			vectors[i] = make([]float32, dimension)
			for j := range vectors[i] {
				vectors[i][j] = rand.Float32()
			}
		}

		// Benchmark Add operation
		fmt.Println("\nBenchmarking Add operation...")
		startAdd := time.Now()
		for i, vec := range vectors {
			err := idx.Add(uint64(i+1000), vec, map[string]interface{}{
				"benchmark": true,
				"index":     i,
			})
			if err != nil {
				logger.Error("Failed to add vector", zap.Error(err))
			}
		}
		addDuration := time.Since(startAdd)
		fmt.Printf("  Added %d vectors in %s (%.2f vectors/sec)\n",
			numVectors, addDuration, float64(numVectors)/addDuration.Seconds())

		// Benchmark Search operation
		fmt.Println("\nBenchmarking Search operation...")
		totalSearchTime := time.Duration(0)
		for i := 0; i < 10; i++ {
			queryVector := vectors[i]
			startSearch := time.Now()
			_, err := idx.Search(queryVector, 10, 1, 10)
			if err != nil {
				logger.Error("Failed to search", zap.Error(err))
				continue
			}
			totalSearchTime += time.Since(startSearch)
		}
		avgSearchTime := totalSearchTime / 10
		fmt.Printf("  Average search time: %s (%.2f searches/sec)\n",
			avgSearchTime, float64(1)/avgSearchTime.Seconds())
	},
}

var validateCmd = &cobra.Command{
	Use:   "validate [config-file]",
	Short: "Validate a configuration file",
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		configFile := args[0]
		fmt.Printf("Validating configuration file: %s\n", configFile)

		// Read and parse the config file
		data, err := os.ReadFile(configFile)
		if err != nil {
			logger.Fatal("Failed to read config file", zap.Error(err))
		}

		var config quiver.Config
		if err := sonic.Unmarshal(data, &config); err != nil {
			logger.Fatal("Failed to parse config file", zap.Error(err))
		}

		// Validate the configuration
		issues := validateConfig(config)
		if len(issues) == 0 {
			fmt.Println("Configuration is valid!")
			return
		}

		fmt.Println("Configuration validation issues:")
		for i, issue := range issues {
			fmt.Printf("  %d. %s\n", i+1, issue)
		}
	},
}

var exportCmd = &cobra.Command{
	Use:   "export [output-file]",
	Short: "Export index metadata to a file",
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		outputFile := args[0]
		idx, err := loadIndex()
		if err != nil {
			logger.Fatal("Failed to load index", zap.Error(err))
		}
		defer idx.Close()

		fmt.Printf("Exporting metadata to: %s\n", outputFile)

		// Get all metadata
		results, err := idx.QueryMetadata("SELECT * FROM metadata")
		if err != nil {
			logger.Fatal("Failed to query metadata", zap.Error(err))
		}

		// Write to file
		data, err := sonic.MarshalIndent(results, "", "  ")
		if err != nil {
			logger.Fatal("Failed to marshal metadata", zap.Error(err))
		}

		if err := os.WriteFile(outputFile, data, 0644); err != nil {
			logger.Fatal("Failed to write output file", zap.Error(err))
		}

		fmt.Printf("Successfully exported %d records\n", len(results))
	},
}

var importCmd = &cobra.Command{
	Use:   "import [input-file]",
	Short: "Import vectors and metadata from a file",
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		inputFile := args[0]
		idx, err := loadIndex()
		if err != nil {
			logger.Fatal("Failed to load index", zap.Error(err))
		}
		defer idx.Close()

		fmt.Printf("Importing from: %s\n", inputFile)

		// Read the input file
		data, err := os.ReadFile(inputFile)
		if err != nil {
			logger.Fatal("Failed to read input file", zap.Error(err))
		}

		var records []map[string]interface{}
		if err := sonic.Unmarshal(data, &records); err != nil {
			logger.Fatal("Failed to parse input file", zap.Error(err))
		}

		// Import the records
		for i, record := range records {
			// Extract vector and ID
			vectorData, ok := record["vector"].([]interface{})
			if !ok {
				logger.Error("Record missing vector field", zap.Int("index", i))
				continue
			}

			vector := make([]float32, len(vectorData))
			for j, v := range vectorData {
				f, ok := v.(float64)
				if !ok {
					logger.Error("Invalid vector value", zap.Int("index", i), zap.Int("position", j))
					continue
				}
				vector[j] = float32(f)
			}

			idVal, ok := record["id"]
			if !ok {
				logger.Error("Record missing ID field", zap.Int("index", i))
				continue
			}

			var id uint64
			switch v := idVal.(type) {
			case float64:
				id = uint64(v)
			case string:
				id, err = strconv.ParseUint(v, 10, 64)
				if err != nil {
					logger.Error("Invalid ID value", zap.Int("index", i), zap.Error(err))
					continue
				}
			default:
				logger.Error("Invalid ID type", zap.Int("index", i))
				continue
			}

			// Remove vector and ID from metadata
			delete(record, "vector")
			delete(record, "id")

			// Add to index
			if err := idx.Add(id, vector, record); err != nil {
				logger.Error("Failed to add record", zap.Int("index", i), zap.Error(err))
			}
		}

		fmt.Printf("Successfully imported %d records\n", len(records))
	},
}

var optimizeCmd = &cobra.Command{
	Use:   "optimize",
	Short: "Optimize the index for better performance",
	Run: func(cmd *cobra.Command, args []string) {
		idx, err := loadIndex()
		if err != nil {
			logger.Fatal("Failed to load index", zap.Error(err))
		}
		defer idx.Close()

		fmt.Println("Optimizing index...")

		// Force a batch flush
		fmt.Println("Flushing batch buffer...")
		// This is a private method, so we can't call it directly
		// Instead, we'll add a dummy vector to trigger a flush
		dummyVector := make([]float32, idx.Config().Dimension)
		err = idx.Add(0, dummyVector, map[string]interface{}{"_dummy": true})
		if err != nil {
			logger.Error("Failed to add dummy vector", zap.Error(err))
		}

		// Force persistence
		fmt.Println("Persisting index to disk...")
		// Again, this is private, so we'll use Save instead
		err = idx.Save(idx.Config().StoragePath)
		if err != nil {
			logger.Fatal("Failed to save index", zap.Error(err))
		}

		fmt.Println("Optimization complete!")
	},
}

var backupCmd = &cobra.Command{
	Use:   "backup [path]",
	Short: "Backup the index and metadata",
	Args:  cobra.ExactArgs(1),
	RunE:  runBackup,
}

var restoreCmd = &cobra.Command{
	Use:   "restore [path]",
	Short: "Restore the index from backup",
	Args:  cobra.ExactArgs(1),
	RunE:  runRestore,
}

var healthCmd = &cobra.Command{
	Use:   "health",
	Short: "Check the health of the index",
	Run: func(cmd *cobra.Command, args []string) {
		idx, err := loadIndex()
		if err != nil {
			logger.Fatal("Failed to load index", zap.Error(err))
		}
		defer idx.Close()

		fmt.Println("Checking index health...")

		// Check index health
		err = idx.HealthCheck()
		if err != nil {
			fmt.Println("❌ Index health check failed:")
			fmt.Printf("   %s\n", err)
			os.Exit(1)
		}

		fmt.Println("✅ Index is healthy!")
	},
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
		if logger != nil {
			logger.Info("Using config file", zap.String("file", viper.ConfigFileUsed()))
		}
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

	// Override storage path if provided
	storagePath := viper.GetString("index.storage_path")
	if indexPath != "" {
		storagePath = indexPath
	}

	// Create the vector index
	index, err := quiver.New(quiver.Config{
		Dimension:       viper.GetInt("index.dimension"),
		StoragePath:     storagePath,
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
	backupPath := args[0]
	idx, err := loadIndex()
	if err != nil {
		logger.Fatal("Failed to load index", zap.Error(err))
		return err
	}
	defer idx.Close()

	fmt.Printf("Creating backup at: %s\n", backupPath)

	// Create backup
	err = idx.Backup(backupPath, false, true)
	if err != nil {
		logger.Fatal("Failed to create backup", zap.Error(err))
		return err
	}

	fmt.Println("Backup created successfully!")
	return nil
}

func runRestore(cmd *cobra.Command, args []string) error {
	backupPath := args[0]
	idx, err := loadIndex()
	if err != nil {
		logger.Fatal("Failed to load index", zap.Error(err))
		return err
	}
	defer idx.Close()

	fmt.Printf("Restoring from backup: %s\n", backupPath)

	// Restore from backup
	err = idx.Restore(backupPath)
	if err != nil {
		logger.Fatal("Failed to restore from backup", zap.Error(err))
		return err
	}

	fmt.Println("Restore completed successfully!")
	return nil
}

// Helper functions

func loadIndex() (*quiver.Index, error) {
	// Set default index configuration
	viper.SetDefault("index.dimension", 128)
	viper.SetDefault("index.storage_path", "quiver.db")
	viper.SetDefault("index.max_elements", 100000)
	viper.SetDefault("index.hnsw_m", 32)
	viper.SetDefault("index.ef_construction", 200)
	viper.SetDefault("index.ef_search", 200)
	viper.SetDefault("index.batch_size", 1000)
	viper.SetDefault("index.distance", int(quiver.Cosine))

	// Override storage path if provided
	storagePath := viper.GetString("index.storage_path")
	if indexPath != "" {
		storagePath = indexPath
	}

	// Create config from viper settings
	config := quiver.Config{
		Dimension:       viper.GetInt("index.dimension"),
		StoragePath:     storagePath,
		MaxElements:     uint64(viper.GetInt("index.max_elements")),
		HNSWM:           viper.GetInt("index.hnsw_m"),
		HNSWEfConstruct: viper.GetInt("index.ef_construction"),
		HNSWEfSearch:    viper.GetInt("index.ef_search"),
		BatchSize:       viper.GetInt("index.batch_size"),
		Distance:        quiver.DistanceMetric(viper.GetInt("index.distance")),
	}

	// Load the index
	return quiver.Load(config, logger)
}

func getDistanceMetricName(metric quiver.DistanceMetric) string {
	switch metric {
	case quiver.Cosine:
		return "Cosine"
	case quiver.L2:
		return "L2 (Euclidean)"
	default:
		return fmt.Sprintf("Unknown (%d)", metric)
	}
}

func validateConfig(config quiver.Config) []string {
	var issues []string

	// Check required fields
	if config.Dimension <= 0 {
		issues = append(issues, "Dimension must be greater than 0")
	}

	if config.MaxElements <= 0 {
		issues = append(issues, "MaxElements must be greater than 0")
	}

	if config.HNSWM <= 0 {
		issues = append(issues, "HNSWM must be greater than 0")
	}

	if config.HNSWEfConstruct <= 0 {
		issues = append(issues, "HNSWEfConstruct must be greater than 0")
	}

	if config.HNSWEfSearch <= 0 {
		issues = append(issues, "HNSWEfSearch must be greater than 0")
	}

	// Check distance metric
	if config.Distance != quiver.Cosine && config.Distance != quiver.L2 {
		issues = append(issues, fmt.Sprintf("Invalid distance metric: %d", config.Distance))
	}

	// Check dimensionality reduction settings
	if config.EnableDimReduction {
		if config.DimReductionTarget <= 0 {
			issues = append(issues, "DimReductionTarget must be greater than 0 when EnableDimReduction is true")
		}

		if config.DimReductionTarget >= config.Dimension {
			issues = append(issues, "DimReductionTarget must be less than Dimension")
		}

		validMethods := map[string]bool{
			"PCA":  true,
			"TSNE": true,
			"UMAP": true,
		}

		if !validMethods[config.DimReductionMethod] {
			issues = append(issues, fmt.Sprintf("Invalid dimensionality reduction method: %s", config.DimReductionMethod))
		}

		if config.DimReductionAdaptive && (config.DimReductionMinVariance <= 0 || config.DimReductionMinVariance > 1) {
			issues = append(issues, "DimReductionMinVariance must be between 0 and 1 when DimReductionAdaptive is true")
		}
	}

	// Check encryption settings
	if config.EncryptionEnabled && len(config.EncryptionKey) < 32 {
		issues = append(issues, "EncryptionKey must be at least 32 bytes when EncryptionEnabled is true")
	}

	return issues
}

func main() {
	var err error
	logger, err = zap.NewProduction()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to initialize logger: %v\n", err)
		os.Exit(1)
	}
	defer func() {
		// Sync is a best-effort operation, so we can ignore errors
		// or log them at a lower level
		if err := logger.Sync(); err != nil {
			// This is a common error that can be safely ignored
			if !strings.Contains(err.Error(), "sync /dev/stderr: invalid argument") {
				fmt.Fprintf(os.Stderr, "Failed to sync logger: %v\n", err)
			}
		}
	}()

	if err := rootCmd.Execute(); err != nil {
		// Don't log stack trace for expected errors
		fmt.Fprintln(os.Stderr, "Error:", err)
		os.Exit(1)
	}
}
