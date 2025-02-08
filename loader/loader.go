package loader

import (
	"bufio"
	"bytes"
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"go.uber.org/zap"
	"golang.org/x/sync/errgroup"

	quiver "github.com/TFMV/quiver"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/apache/arrow-go/v18/parquet/file"
	"github.com/apache/arrow-go/v18/parquet/pqarrow"
)

// VectorRecord defines the expected schema for vector data files.
type VectorRecord struct {
	ID     int       `json:"id"`
	Vector []float32 `json:"vector"`
}

// LoadJSONFile reads the entire file and attempts to decode it as either a JSON array or as
// newline–delimited JSON objects.
func LoadJSONFile(path string, logger *zap.Logger) ([]VectorRecord, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read JSON file %s: %w", path, err)
	}

	var records []VectorRecord
	if err := json.Unmarshal(data, &records); err == nil {
		return records, nil
	}

	// Fallback to newline–delimited JSON
	var recs []VectorRecord
	scanner := bufio.NewScanner(bytes.NewReader(data))
	for scanner.Scan() {
		var r VectorRecord
		if err := json.Unmarshal(scanner.Bytes(), &r); err != nil {
			logger.Warn("failed to parse JSON line", zap.String("line", scanner.Text()), zap.Error(err))
			continue
		}
		recs = append(recs, r)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("failed scanning JSON file %s: %w", path, err)
	}
	return recs, nil
}

// LoadCSVFile reads a CSV file where the first column is the record ID and the remaining columns
// represent vector values. It expects a header row.
// LoadCSVFile reads a CSV file where the first column is the record ID
// and the second column contains a JSON-encoded vector.
func LoadCSVFile(path string, logger *zap.Logger) ([]VectorRecord, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open CSV file %s: %w", path, err)
	}
	defer f.Close()

	reader := csv.NewReader(f)
	reader.TrimLeadingSpace = true

	// Read header
	header, err := reader.Read()
	if err != nil {
		return nil, fmt.Errorf("failed to read CSV header from %s: %w", path, err)
	}
	if len(header) != 2 || header[0] != "id" || header[1] != "vector" {
		return nil, fmt.Errorf("CSV file %s must have exactly two columns: id, vector", path)
	}

	var records []VectorRecord
	for {
		row, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("error reading CSV file %s: %w", path, err)
		}

		id, err := strconv.Atoi(row[0])
		if err != nil {
			logger.Warn("failed to parse CSV id", zap.String("value", row[0]), zap.Error(err))
			continue
		}

		var vector []float32
		err = json.Unmarshal([]byte(row[1]), &vector)
		if err != nil {
			logger.Warn("failed to parse vector JSON", zap.String("value", row[1]), zap.Error(err))
			continue
		}

		records = append(records, VectorRecord{ID: id, Vector: vector})
	}

	return records, nil
}

// LoadParquetFile uses Apache Arrow's parquet reader to load vector records.
// NOTE: This implementation assumes the parquet file contains at least two columns:
// an "id" column (int32/int64) and a "vector" column (list of floats).
// In production you may wish to perform schema introspection and support alternate layouts.
func LoadParquetFile(path string, logger *zap.Logger) ([]VectorRecord, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open parquet file %s: %w", path, err)
	}
	defer f.Close()

	// Create Parquet file reader first
	reader, err := file.NewParquetReader(f)
	if err != nil {
		return nil, fmt.Errorf("failed to create parquet file reader: %w", err)
	}
	defer reader.Close()

	// Then create Arrow reader
	pr, err := pqarrow.NewFileReader(reader, pqarrow.ArrowReadProperties{
		Parallel:  true,
		BatchSize: 1000,
	}, memory.NewGoAllocator())
	if err != nil {
		return nil, fmt.Errorf("failed to create parquet reader: %w", err)
	}

	table, err := pr.ReadTable(context.Background())
	if err != nil {
		return nil, fmt.Errorf("failed to read parquet as table: %w", err)
	}
	defer table.Release()

	// Process the table rows
	var records []VectorRecord
	for i := range table.NumRows() {
		idCol := table.Column(0).Data().Chunk(0).(*array.Int64)
		vectorCol := table.Column(1).Data().Chunk(0).(*array.Float32)

		records = append(records, VectorRecord{
			ID:     int(idCol.Value(int(i))),
			Vector: vectorCol.Float32Values()[i : i+1],
		})
	}

	return records, nil
}

// LoadFilesFromDirectory scans the provided directory for files ending in .parquet, .csv, or .json,
// and concurrently processes them. For each loaded record, the vector is added to the provided VectorIndex.
func LoadFilesFromDirectory(dir string, vi *quiver.VectorIndex, logger *zap.Logger) error {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return fmt.Errorf("failed to read directory %s: %w", dir, err)
	}

	var eg errgroup.Group
	// Limit concurrency to 10 simultaneous file loaders.
	sem := make(chan struct{}, 10)

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		ext := strings.ToLower(filepath.Ext(entry.Name()))
		if ext != ".parquet" && ext != ".csv" && ext != ".json" {
			continue
		}
		// Capture local variables for the goroutine.
		filePath := filepath.Join(dir, entry.Name())
		fileExt := ext

		eg.Go(func() error {
			sem <- struct{}{}
			defer func() { <-sem }()
			logger.Info("Loading file", zap.String("file", filePath))

			var records []VectorRecord
			var err error
			switch fileExt {
			case ".parquet":
				records, err = LoadParquetFile(filePath, logger)
			case ".csv":
				records, err = LoadCSVFile(filePath, logger)
			case ".json":
				records, err = LoadJSONFile(filePath, logger)
			}
			if err != nil {
				logger.Error("Failed to load file", zap.String("file", filePath), zap.Error(err))
				return err
			}

			// Insert each record into the vector index.
			for _, record := range records {
				if err := vi.AddVector(record.ID, record.Vector); err != nil {
					logger.Error("Failed to add vector record", zap.Int("id", record.ID), zap.Error(err))
					// Continue processing other records.
				}
			}
			logger.Info("Finished loading file", zap.String("file", filePath), zap.Int("records", len(records)))
			return nil
		})
	}

	return eg.Wait()
}
