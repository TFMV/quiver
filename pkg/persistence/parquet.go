package persistence

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/xitongsys/parquet-go-source/local"
	"github.com/xitongsys/parquet-go/parquet"
	"github.com/xitongsys/parquet-go/reader"
	"github.com/xitongsys/parquet-go/writer"
)

// ParquetVectorRecord represents a vector record in Parquet format
type ParquetVectorRecord struct {
	ID       string    `parquet:"name=id, type=BYTE_ARRAY, convertedtype=UTF8, encoding=PLAIN_DICTIONARY"`
	Vector   []float32 `parquet:"name=vector, type=LIST, convertedtype=LIST, valuetype=FLOAT"`
	Metadata string    `parquet:"name=metadata, type=BYTE_ARRAY, convertedtype=UTF8, encoding=PLAIN"`
}

// writeVectorsToParquet writes vector records to a Parquet file using safe write pattern
func writeVectorsToParquet(records []VectorRecord, filePath string) error {
	// Create parent directory if it doesn't exist
	if err := os.MkdirAll(filepath.Dir(filePath), 0755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	// Use temp file for safe write
	tempPath := filePath + ".tmp"
	cleanupOnError := true
	defer func() {
		if cleanupOnError {
			os.Remove(tempPath)
		}
	}()

	// Create parquet file writer for temp file
	fw, err := local.NewLocalFileWriter(tempPath)
	if err != nil {
		return fmt.Errorf("failed to create parquet file writer: %w", err)
	}
	defer fw.Close()

	// Create Parquet writer
	pw, err := writer.NewParquetWriter(fw, new(ParquetVectorRecord), 4)
	if err != nil {
		return fmt.Errorf("failed to create parquet writer: %w", err)
	}

	// Set compression
	pw.CompressionType = parquet.CompressionCodec_SNAPPY

	// Convert and write records
	for _, record := range records {
		// Convert metadata map to JSON string
		metadataJSON, err := json.Marshal(record.Metadata)
		if err != nil {
			return fmt.Errorf("failed to marshal metadata: %w", err)
		}

		// Create parquet record
		parquetRecord := ParquetVectorRecord{
			ID:       record.ID,
			Vector:   record.Vector,
			Metadata: string(metadataJSON),
		}

		// Write record
		if err := pw.Write(parquetRecord); err != nil {
			return fmt.Errorf("failed to write parquet record: %w", err)
		}
	}

	// Close and flush data
	if err := pw.WriteStop(); err != nil {
		return fmt.Errorf("failed to finalize parquet file: %w", err)
	}

	// Sync the temp file before rename
	if err := syncFile(tempPath); err != nil {
		return fmt.Errorf("failed to sync temp file: %w", err)
	}

	// Atomic rename
	if err := os.Rename(tempPath, filePath); err != nil {
		return fmt.Errorf("failed to rename temp file: %w", err)
	}

	// Success - don't cleanup temp file
	cleanupOnError = false
	return nil
}

// readVectorsFromParquet reads vector records from a Parquet file
func readVectorsFromParquet(filePath string) ([]VectorRecord, error) {
	// Check file exists and is readable first
	if _, err := os.Stat(filePath); err != nil {
		return nil, fmt.Errorf("failed to stat parquet file: %w", err)
	}

	// Open parquet file reader
	fr, err := local.NewLocalFileReader(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open parquet file: %w", err)
	}
	defer fr.Close()

	// Create Parquet reader
	pr, err := reader.NewParquetReader(fr, new(ParquetVectorRecord), 4)
	if err != nil {
		return nil, fmt.Errorf("failed to create parquet reader: %w", err)
	}
	defer pr.ReadStop()

	// Get number of records
	numRecords := int(pr.GetNumRows())
	if numRecords == 0 {
		return []VectorRecord{}, nil
	}

	records := make([]VectorRecord, 0, numRecords)

	// Read records in batches for better performance
	batchSize := 1000
	for i := 0; i < numRecords; i += batchSize {
		// Adjust batch size for last batch
		currentBatchSize := batchSize
		if i+batchSize > numRecords {
			currentBatchSize = numRecords - i
		}

		// Read batch
		parquetRecords := make([]ParquetVectorRecord, currentBatchSize)
		if err := pr.Read(&parquetRecords); err != nil {
			return nil, fmt.Errorf("failed to read parquet records: %w", err)
		}

		// Convert to VectorRecord
		for _, parquetRecord := range parquetRecords {
			// Skip empty IDs
			if parquetRecord.ID == "" {
				continue
			}

			// Parse metadata JSON
			var metadata map[string]string
			if parquetRecord.Metadata != "" {
				if err := json.Unmarshal([]byte(parquetRecord.Metadata), &metadata); err != nil {
					// Skip corrupted metadata, use empty
					metadata = make(map[string]string)
				}
			} else {
				metadata = make(map[string]string)
			}

			// Skip empty vectors
			if len(parquetRecord.Vector) == 0 {
				continue
			}

			// Create vector record
			record := VectorRecord{
				ID:       parquetRecord.ID,
				Vector:   parquetRecord.Vector,
				Metadata: metadata,
			}

			records = append(records, record)
		}
	}

	return records, nil
}

// WriteVectorsToParquetFile writes vector records to a Parquet file (public wrapper)
func WriteVectorsToParquetFile(records []VectorRecord, filePath string) error {
	return writeVectorsToParquet(records, filePath)
}

// ReadVectorsFromParquetFile reads vector records from a Parquet file (public wrapper)
func ReadVectorsFromParquetFile(filePath string) ([]VectorRecord, error) {
	return readVectorsFromParquet(filePath)
}
