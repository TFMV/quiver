package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/docopt/docopt-go"
	"go.uber.org/zap"

	quiver "github.com/TFMV/quiver"
	"github.com/TFMV/quiver/loader"
)

// main is the CLI entry point. It parses command-line options using docopt,
// initializes the VectorIndex, loads all supported files from the specified directory,
// and then saves the updated index.
func main() {
	usage := `Usage:
  loader --dir=<directory> --db=<dbPath> --index=<indexPath> --dim=<dimension> [--metric=<metric>]

Options:
  --dir=<directory>       Directory containing files to load.
  --db=<dbPath>           DuckDB database file path.
  --index=<indexPath>     HNSW index file path.
  --dim=<dimension>       Vector dimension.
  --metric=<metric>       Distance metric (cosine|euclidean) [default: cosine].
`
	arguments, err := docopt.ParseDoc(usage)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing arguments: %v\n", err)
		os.Exit(1)
	}

	dir, _ := arguments.String("--dir")
	dbPath, _ := arguments.String("--db")
	indexPath, _ := arguments.String("--index")
	dimStr, _ := arguments.String("--dim")
	metricStr, _ := arguments.String("--metric")

	dim, err := strconv.Atoi(dimStr)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Invalid dimension value: %v\n", err)
		os.Exit(1)
	}

	var metric quiver.SpaceType
	if strings.ToLower(metricStr) == "euclidean" {
		metric = quiver.Euclidean
	} else {
		metric = quiver.Cosine
	}

	// Create a production zap logger.
	logger, err := zap.NewProduction()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create logger: %v\n", err)
		os.Exit(1)
	}
	defer logger.Sync()

	logger.Info("Starting loader",
		zap.String("directory", dir),
		zap.String("dbPath", dbPath),
		zap.String("indexPath", indexPath),
		zap.Int("dimension", dim),
		zap.String("metric", metricStr))

	// Initialize the VectorIndex from the quiver package.
	vi, err := quiver.NewVectorIndex(dim, dbPath, indexPath, metric)
	if err != nil {
		logger.Error("Failed to initialize VectorIndex", zap.Error(err))
		os.Exit(1)
	}
	defer func() {
		if err := vi.Close(); err != nil {
			logger.Error("Error closing VectorIndex", zap.Error(err))
		}
	}()

	// Load files from the specified directory.
	if err := loader.LoadFilesFromDirectory(dir, vi, logger); err != nil {
		logger.Error("Failed to load files", zap.Error(err))
		os.Exit(1)
	}

	// Save the updated index.
	if err := vi.Save(); err != nil {
		logger.Error("Failed to save VectorIndex", zap.Error(err))
		os.Exit(1)
	}

	logger.Info("Successfully loaded files and saved VectorIndex")
}
