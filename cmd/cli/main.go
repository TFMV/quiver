package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/TFMV/hnsw"
	"github.com/TFMV/hnsw/hnsw-extensions/hybrid"
	quiver "github.com/TFMV/quiver"
	"github.com/TFMV/quiver/api"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

var (
	cfgFile     string
	logger      *zap.Logger
	rootCmd     *cobra.Command
	serverCmd   *cobra.Command
	searchCmd   *cobra.Command
	vectorCmd   *cobra.Command
	indexCmd    *cobra.Command
	backupCmd   *cobra.Command
	restoreCmd  *cobra.Command
	versionCmd  *cobra.Command
	verboseFlag bool
)

const (
	// Version information
	Version   = "0.1.0"
	BuildDate = "2023-06-01"
)

func init() {
	cobra.OnInitialize(initConfig)

	// Root command
	rootCmd = &cobra.Command{
		Use:   "quiver",
		Short: "Quiver is a high-performance vector database",
		Long: `Quiver is a high-performance vector database built on HNSW algorithm.
It provides fast vector similarity search with metadata filtering capabilities.`,
		PersistentPreRun: func(cmd *cobra.Command, args []string) {
			setupLogger()
		},
	}

	rootCmd.PersistentFlags().StringVar(&cfgFile, "config", "", "config file (default is $HOME/.quiver.yaml)")
	rootCmd.PersistentFlags().BoolVarP(&verboseFlag, "verbose", "v", false, "verbose output")

	// Server command
	serverCmd = &cobra.Command{
		Use:   "server",
		Short: "Start the Quiver API server",
		Long:  `Start the Quiver API server that provides REST endpoints for vector operations.`,
		Run:   runServer,
	}

	serverCmd.Flags().String("port", "8080", "Port to listen on")
	serverCmd.Flags().String("host", "0.0.0.0", "Host to bind to")
	serverCmd.Flags().String("storage", "./data", "Path to store index data")
	serverCmd.Flags().Int("dimension", 128, "Vector dimension")
	serverCmd.Flags().Int("max-elements", 1000000, "Maximum number of elements")
	serverCmd.Flags().String("distance", "cosine", "Distance function (cosine, l2)")
	serverCmd.Flags().Int("hnsw-m", 16, "HNSW M parameter")
	serverCmd.Flags().Int("hnsw-ef-search", 100, "HNSW EF search parameter")

	viper.BindPFlag("server.port", serverCmd.Flags().Lookup("port"))
	viper.BindPFlag("server.host", serverCmd.Flags().Lookup("host"))
	viper.BindPFlag("server.storage", serverCmd.Flags().Lookup("storage"))
	viper.BindPFlag("index.dimension", serverCmd.Flags().Lookup("dimension"))
	viper.BindPFlag("index.max_elements", serverCmd.Flags().Lookup("max-elements"))
	viper.BindPFlag("index.distance", serverCmd.Flags().Lookup("distance"))
	viper.BindPFlag("index.hnsw_m", serverCmd.Flags().Lookup("hnsw-m"))
	viper.BindPFlag("index.hnsw_ef_search", serverCmd.Flags().Lookup("hnsw-ef-search"))

	// Vector commands
	vectorCmd = &cobra.Command{
		Use:   "vector",
		Short: "Vector operations",
		Long:  `Commands for managing vectors in the database.`,
	}

	addVectorCmd := &cobra.Command{
		Use:   "add",
		Short: "Add a vector",
		Long:  `Add a vector to the database.`,
		Run:   addVector,
	}

	addVectorCmd.Flags().String("id", "", "Vector ID")
	addVectorCmd.Flags().String("vector", "", "Vector data (comma-separated floats)")
	addVectorCmd.Flags().String("metadata", "{}", "Vector metadata (JSON)")
	addVectorCmd.Flags().String("server", "http://localhost:8080", "API server URL")

	deleteVectorCmd := &cobra.Command{
		Use:   "delete",
		Short: "Delete a vector",
		Long:  `Delete a vector from the database.`,
		Run:   deleteVector,
	}

	deleteVectorCmd.Flags().String("id", "", "Vector ID")
	deleteVectorCmd.Flags().String("server", "http://localhost:8080", "API server URL")

	getVectorCmd := &cobra.Command{
		Use:   "get",
		Short: "Get a vector",
		Long:  `Get a vector from the database.`,
		Run:   getVector,
	}

	getVectorCmd.Flags().String("id", "", "Vector ID")
	getVectorCmd.Flags().String("server", "http://localhost:8080", "API server URL")

	vectorCmd.AddCommand(addVectorCmd, deleteVectorCmd, getVectorCmd)

	// Search commands
	searchCmd = &cobra.Command{
		Use:   "search",
		Short: "Search operations",
		Long:  `Commands for searching vectors in the database.`,
	}

	similaritySearchCmd := &cobra.Command{
		Use:   "similarity",
		Short: "Similarity search",
		Long:  `Search for similar vectors.`,
		Run:   similaritySearch,
	}

	similaritySearchCmd.Flags().String("vector", "", "Query vector (comma-separated floats)")
	similaritySearchCmd.Flags().Int("k", 10, "Number of results")
	similaritySearchCmd.Flags().String("server", "http://localhost:8080", "API server URL")

	hybridSearchCmd := &cobra.Command{
		Use:   "hybrid",
		Short: "Hybrid search",
		Long:  `Search for similar vectors with metadata filtering.`,
		Run:   hybridSearch,
	}

	hybridSearchCmd.Flags().String("vector", "", "Query vector (comma-separated floats)")
	hybridSearchCmd.Flags().Int("k", 10, "Number of results")
	hybridSearchCmd.Flags().String("filter", "{}", "Metadata filter (JSON)")
	hybridSearchCmd.Flags().String("server", "http://localhost:8080", "API server URL")

	negativeSearchCmd := &cobra.Command{
		Use:   "negative",
		Short: "Search with negative examples",
		Long:  `Search for similar vectors with negative examples.`,
		Run:   negativeSearch,
	}

	negativeSearchCmd.Flags().String("positive", "", "Positive query vector (comma-separated floats)")
	negativeSearchCmd.Flags().String("negative", "", "Negative query vectors (comma-separated floats, multiple vectors separated by semicolons)")
	negativeSearchCmd.Flags().Float32("weight", 0.5, "Negative weight")
	negativeSearchCmd.Flags().Int("k", 10, "Number of results")
	negativeSearchCmd.Flags().String("server", "http://localhost:8080", "API server URL")

	searchCmd.AddCommand(similaritySearchCmd, hybridSearchCmd, negativeSearchCmd)

	// Index commands
	indexCmd = &cobra.Command{
		Use:   "index",
		Short: "Index operations",
		Long:  `Commands for managing the index.`,
	}

	createIndexCmd := &cobra.Command{
		Use:   "create",
		Short: "Create a new index",
		Long:  `Create a new vector index with specified parameters.`,
		Run:   createIndex,
	}

	createIndexCmd.Flags().Int("dimension", 128, "Vector dimension")
	createIndexCmd.Flags().Int("max-elements", 1000000, "Maximum number of elements")
	createIndexCmd.Flags().String("distance", "cosine", "Distance function (cosine, l2)")
	createIndexCmd.Flags().Int("hnsw-m", 16, "HNSW M parameter")
	createIndexCmd.Flags().Int("hnsw-ef-search", 100, "HNSW EF search parameter")
	createIndexCmd.Flags().String("server", "http://localhost:8080", "API server URL")

	indexCmd.AddCommand(createIndexCmd)

	// Backup command
	backupCmd = &cobra.Command{
		Use:   "backup",
		Short: "Backup the index",
		Long:  `Backup the index to a specified path.`,
		Run:   backupIndex,
	}

	backupCmd.Flags().String("path", "./backup", "Backup path")
	backupCmd.Flags().String("server", "http://localhost:8080", "API server URL")

	// Restore command
	restoreCmd = &cobra.Command{
		Use:   "restore",
		Short: "Restore the index",
		Long:  `Restore the index from a specified path.`,
		Run:   restoreIndex,
	}

	restoreCmd.Flags().String("path", "./backup", "Restore path")
	restoreCmd.Flags().String("server", "http://localhost:8080", "API server URL")

	// Version command
	versionCmd = &cobra.Command{
		Use:   "version",
		Short: "Print the version number",
		Long:  `Print the version number of Quiver CLI.`,
		Run: func(cmd *cobra.Command, args []string) {
			fmt.Printf("Quiver CLI v%s (built: %s)\n", Version, BuildDate)
		},
	}

	// Add commands to root
	rootCmd.AddCommand(serverCmd, vectorCmd, searchCmd, indexCmd, backupCmd, restoreCmd, versionCmd)
}

func initConfig() {
	if cfgFile != "" {
		// Use config file from the flag
		viper.SetConfigFile(cfgFile)
	} else {
		// Find home directory
		home, err := os.UserHomeDir()
		cobra.CheckErr(err)

		// Search config in home directory with name ".quiver" (without extension)
		viper.AddConfigPath(home)
		viper.SetConfigType("yaml")
		viper.SetConfigName(".quiver")
	}

	viper.AutomaticEnv()
	viper.SetEnvPrefix("QUIVER")
	viper.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))

	// If a config file is found, read it in
	if err := viper.ReadInConfig(); err == nil {
		fmt.Fprintln(os.Stderr, "Using config file:", viper.ConfigFileUsed())
	}
}

func setupLogger() {
	config := zap.NewProductionConfig()

	// Set the log level based on the verbose flag
	if verboseFlag {
		config.Level = zap.NewAtomicLevelAt(zapcore.DebugLevel)
	} else {
		config.Level = zap.NewAtomicLevelAt(zapcore.InfoLevel)
	}

	// Configure the encoder
	config.EncoderConfig.TimeKey = "time"
	config.EncoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder
	config.EncoderConfig.EncodeLevel = zapcore.CapitalLevelEncoder

	// Create the logger
	var err error
	logger, err = config.Build()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error setting up logger: %v\n", err)
		os.Exit(1)
	}
}

func runServer(cmd *cobra.Command, args []string) {
	port := viper.GetString("server.port")
	storagePath := viper.GetString("server.storage")
	dimension := viper.GetInt("index.dimension")
	maxElements := viper.GetInt("index.max_elements")
	distance := viper.GetString("index.distance")
	hnswM := viper.GetInt("index.hnsw_m")
	hnswEfSearch := viper.GetInt("index.hnsw_ef_search")

	logger.Info("Starting Quiver server",
		zap.String("port", port),
		zap.String("storage", storagePath),
		zap.Int("dimension", dimension),
		zap.Int("max_elements", maxElements),
		zap.String("distance", distance),
		zap.Int("hnsw_m", hnswM),
		zap.Int("hnsw_ef_search", hnswEfSearch),
	)

	// Ensure storage directory exists
	if err := os.MkdirAll(storagePath, 0755); err != nil {
		logger.Fatal("Failed to create storage directory", zap.Error(err))
	}

	// Create index configuration
	config := quiver.DefaultDBConfig()
	config.BaseDir = storagePath
	config.Hybrid.Type = hybrid.HybridIndexType
	config.Hybrid.M = hnswM
	config.Hybrid.EfSearch = hnswEfSearch

	// Set distance function
	if distance == "l2" {
		config.Hybrid.Distance = hnsw.EuclideanDistance
	} else {
		config.Hybrid.Distance = hnsw.CosineDistance
	}

	// Create or load the index
	logger.Info("Initializing Quiver index", zap.String("storage_path", config.BaseDir))

	index, err := quiver.NewVectorDB[uint64](config)
	if err != nil {
		logger.Fatal("Failed to create index", zap.Error(err))
	}
	defer index.Close()

	// Create and start the server
	server := api.NewServer(index, logger)

	logger.Info("Starting API server", zap.String("port", port))
	if err := server.Start(port); err != nil {
		logger.Fatal("Server error", zap.Error(err))
	}
}

func addVector(cmd *cobra.Command, args []string) {
	// Get flags
	idStr, _ := cmd.Flags().GetString("id")
	vectorStr, _ := cmd.Flags().GetString("vector")
	metadataStr, _ := cmd.Flags().GetString("metadata")
	serverURL, _ := cmd.Flags().GetString("server")

	// Parse ID
	id, err := strconv.ParseUint(idStr, 10, 64)
	if err != nil {
		logger.Error("Invalid ID", zap.Error(err))
		fmt.Fprintf(os.Stderr, "Error: Invalid ID: %v\n", err)
		os.Exit(1)
	}

	// Parse vector
	vector, err := ParseVector(vectorStr)
	if err != nil {
		logger.Error("Invalid vector", zap.Error(err))
		fmt.Fprintf(os.Stderr, "Error: Invalid vector: %v\n", err)
		os.Exit(1)
	}

	// Parse metadata
	var metadata map[string]interface{}
	if metadataStr != "" {
		metadata, err = ParseMetadata(metadataStr)
		if err != nil {
			logger.Error("Invalid metadata", zap.Error(err))
			fmt.Fprintf(os.Stderr, "Error: Invalid metadata: %v\n", err)
			os.Exit(1)
		}
	}

	// Create API client
	client := NewAPIClient(serverURL)

	// Add vector
	resp, err := client.AddVector(id, vector, metadata, nil)
	if err != nil {
		logger.Error("Failed to add vector", zap.Error(err))
		fmt.Fprintf(os.Stderr, "Error: Failed to add vector: %v\n", err)
		os.Exit(1)
	}

	// Print response
	prettyJSON, _ := json.MarshalIndent(resp, "", "  ")
	fmt.Println(string(prettyJSON))
}

func deleteVector(cmd *cobra.Command, args []string) {
	// Get flags
	idStr, _ := cmd.Flags().GetString("id")
	serverURL, _ := cmd.Flags().GetString("server")

	// Parse ID
	id, err := strconv.ParseUint(idStr, 10, 64)
	if err != nil {
		logger.Error("Invalid ID", zap.Error(err))
		fmt.Fprintf(os.Stderr, "Error: Invalid ID: %v\n", err)
		os.Exit(1)
	}

	// Create API client
	client := NewAPIClient(serverURL)

	// Delete vector
	resp, err := client.DeleteVector(id)
	if err != nil {
		logger.Error("Failed to delete vector", zap.Error(err))
		fmt.Fprintf(os.Stderr, "Error: Failed to delete vector: %v\n", err)
		os.Exit(1)
	}

	// Print response
	prettyJSON, _ := json.MarshalIndent(resp, "", "  ")
	fmt.Println(string(prettyJSON))
}

func getVector(cmd *cobra.Command, args []string) {
	// Get flags
	idStr, _ := cmd.Flags().GetString("id")
	serverURL, _ := cmd.Flags().GetString("server")

	// Parse ID
	id, err := strconv.ParseUint(idStr, 10, 64)
	if err != nil {
		logger.Error("Invalid ID", zap.Error(err))
		fmt.Fprintf(os.Stderr, "Error: Invalid ID: %v\n", err)
		os.Exit(1)
	}

	// Create API client
	client := NewAPIClient(serverURL)

	// Get vector
	resp, err := client.GetVector(id)
	if err != nil {
		logger.Error("Failed to get vector", zap.Error(err))
		fmt.Fprintf(os.Stderr, "Error: Failed to get vector: %v\n", err)
		os.Exit(1)
	}

	// Print response
	prettyJSON, _ := json.MarshalIndent(resp, "", "  ")
	fmt.Println(string(prettyJSON))
}

func similaritySearch(cmd *cobra.Command, args []string) {
	// Get flags
	vectorStr, _ := cmd.Flags().GetString("vector")
	k, _ := cmd.Flags().GetInt("k")
	serverURL, _ := cmd.Flags().GetString("server")

	// Parse vector
	vector, err := ParseVector(vectorStr)
	if err != nil {
		logger.Error("Invalid vector", zap.Error(err))
		fmt.Fprintf(os.Stderr, "Error: Invalid vector: %v\n", err)
		os.Exit(1)
	}

	// Create API client
	client := NewAPIClient(serverURL)

	// Perform search
	resp, err := client.Search(vector, k)
	if err != nil {
		logger.Error("Failed to perform search", zap.Error(err))
		fmt.Fprintf(os.Stderr, "Error: Failed to perform search: %v\n", err)
		os.Exit(1)
	}

	// Print response
	prettyJSON, _ := json.MarshalIndent(resp, "", "  ")
	fmt.Println(string(prettyJSON))
}

func hybridSearch(cmd *cobra.Command, args []string) {
	// Get flags
	vectorStr, _ := cmd.Flags().GetString("vector")
	k, _ := cmd.Flags().GetInt("k")
	filter, _ := cmd.Flags().GetString("filter")
	serverURL, _ := cmd.Flags().GetString("server")

	// Parse vector
	vector, err := ParseVector(vectorStr)
	if err != nil {
		logger.Error("Invalid vector", zap.Error(err))
		fmt.Fprintf(os.Stderr, "Error: Invalid vector: %v\n", err)
		os.Exit(1)
	}

	// Create API client
	client := NewAPIClient(serverURL)

	// Perform hybrid search
	resp, err := client.HybridSearch(vector, k, filter)
	if err != nil {
		logger.Error("Failed to perform hybrid search", zap.Error(err))
		fmt.Fprintf(os.Stderr, "Error: Failed to perform hybrid search: %v\n", err)
		os.Exit(1)
	}

	// Print response
	prettyJSON, _ := json.MarshalIndent(resp, "", "  ")
	fmt.Println(string(prettyJSON))
}

func negativeSearch(cmd *cobra.Command, args []string) {
	// Get flags
	positiveStr, _ := cmd.Flags().GetString("positive")
	negativeStr, _ := cmd.Flags().GetString("negative")
	k, _ := cmd.Flags().GetInt("k")
	weight, _ := cmd.Flags().GetFloat32("weight")
	serverURL, _ := cmd.Flags().GetString("server")

	// Parse positive vector
	positiveVector, err := ParseVector(positiveStr)
	if err != nil {
		logger.Error("Invalid positive vector", zap.Error(err))
		fmt.Fprintf(os.Stderr, "Error: Invalid positive vector: %v\n", err)
		os.Exit(1)
	}

	// Parse negative vectors
	negativeVectors, err := ParseNegativeVectors(negativeStr)
	if err != nil {
		logger.Error("Invalid negative vectors", zap.Error(err))
		fmt.Fprintf(os.Stderr, "Error: Invalid negative vectors: %v\n", err)
		os.Exit(1)
	}

	// Create API client
	client := NewAPIClient(serverURL)

	// Perform search with negatives
	resp, err := client.SearchWithNegatives(positiveVector, negativeVectors, k, weight)
	if err != nil {
		logger.Error("Failed to perform search with negatives", zap.Error(err))
		fmt.Fprintf(os.Stderr, "Error: Failed to perform search with negatives: %v\n", err)
		os.Exit(1)
	}

	// Print response
	prettyJSON, _ := json.MarshalIndent(resp, "", "  ")
	fmt.Println(string(prettyJSON))
}

func createIndex(cmd *cobra.Command, args []string) {
	// Get flags
	dimension, _ := cmd.Flags().GetInt("dimension")
	maxElements, _ := cmd.Flags().GetInt("max-elements")
	distance, _ := cmd.Flags().GetString("distance")
	hnswM, _ := cmd.Flags().GetInt("hnsw-m")
	hnswEfSearch, _ := cmd.Flags().GetInt("hnsw-ef-search")
	serverURL, _ := cmd.Flags().GetString("server")

	// Create options
	opts := map[string]interface{}{
		"dimension":      dimension,
		"max_elements":   maxElements,
		"distance":       distance,
		"hnsw_m":         hnswM,
		"hnsw_ef_search": hnswEfSearch,
		"dim_reduction": map[string]interface{}{
			"enabled": false,
		},
	}

	// Create API client
	client := NewAPIClient(serverURL)

	// Create index
	resp, err := client.CreateIndex(opts)
	if err != nil {
		logger.Error("Failed to create index", zap.Error(err))
		fmt.Fprintf(os.Stderr, "Error: Failed to create index: %v\n", err)
		os.Exit(1)
	}

	// Print response
	prettyJSON, _ := json.MarshalIndent(resp, "", "  ")
	fmt.Println(string(prettyJSON))
}

func backupIndex(cmd *cobra.Command, args []string) {
	// Get flags
	path, _ := cmd.Flags().GetString("path")
	serverURL, _ := cmd.Flags().GetString("server")

	// Create API client
	client := NewAPIClient(serverURL)

	// Backup index
	resp, err := client.BackupIndex(path)
	if err != nil {
		logger.Error("Failed to backup index", zap.Error(err))
		fmt.Fprintf(os.Stderr, "Error: Failed to backup index: %v\n", err)
		os.Exit(1)
	}

	// Print response
	prettyJSON, _ := json.MarshalIndent(resp, "", "  ")
	fmt.Println(string(prettyJSON))
}

func restoreIndex(cmd *cobra.Command, args []string) {
	// Get flags
	path, _ := cmd.Flags().GetString("path")
	serverURL, _ := cmd.Flags().GetString("server")

	// Create API client
	client := NewAPIClient(serverURL)

	// Restore index
	resp, err := client.RestoreIndex(path)
	if err != nil {
		logger.Error("Failed to restore index", zap.Error(err))
		fmt.Fprintf(os.Stderr, "Error: Failed to restore index: %v\n", err)
		os.Exit(1)
	}

	// Print response
	prettyJSON, _ := json.MarshalIndent(resp, "", "  ")
	fmt.Println(string(prettyJSON))
}

func main() {
	defer func() {
		if logger != nil {
			logger.Sync()
		}
	}()

	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
