package persistence

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// TestParquetVectorRecordConversion tests converting between VectorRecord and ParquetVectorRecord
func TestParquetVectorRecordConversion(t *testing.T) {
	// Create a temporary directory for testing
	tempDir, err := os.MkdirTemp("", "quiver-parquet-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Create test vector records
	originalRecords := []VectorRecord{
		{
			ID:       "vec1",
			Vector:   []float32{0.1, 0.2, 0.3, 0.4},
			Metadata: map[string]string{"key1": "value1", "key2": "value2"},
		},
		{
			ID:       "vec2",
			Vector:   []float32{0.5, 0.6, 0.7, 0.8},
			Metadata: map[string]string{"key3": "value3"},
		},
		{
			ID:       "vec3",
			Vector:   []float32{0.9, 1.0, 1.1, 1.2},
			Metadata: nil,
		},
	}

	// Path for the parquet file
	parquetPath := filepath.Join(tempDir, "test_vectors.parquet")

	// Write records to parquet file
	err = WriteVectorsToParquetFile(originalRecords, parquetPath)
	if err != nil {
		t.Fatalf("Failed to write vectors to parquet: %v", err)
	}

	// Read records from parquet file
	loadedRecords, err := ReadVectorsFromParquetFile(parquetPath)
	if err != nil {
		t.Fatalf("Failed to read vectors from parquet: %v", err)
	}

	// Verify number of records matches
	if len(loadedRecords) != len(originalRecords) {
		t.Fatalf("Expected %d records, got %d", len(originalRecords), len(loadedRecords))
	}

	// Create maps for easier comparison
	originalMap := make(map[string]VectorRecord)
	for _, record := range originalRecords {
		originalMap[record.ID] = record
	}

	loadedMap := make(map[string]VectorRecord)
	for _, record := range loadedRecords {
		loadedMap[record.ID] = record
	}

	// Compare original and loaded records
	for id, originalRecord := range originalMap {
		loadedRecord, exists := loadedMap[id]
		if !exists {
			t.Errorf("Record with ID %s not found in loaded records", id)
			continue
		}

		// Compare vector values
		if !reflect.DeepEqual(loadedRecord.Vector, originalRecord.Vector) {
			t.Errorf("Vectors don't match for ID %s: expected %v, got %v",
				id, originalRecord.Vector, loadedRecord.Vector)
		}

		// Compare metadata
		if originalRecord.Metadata == nil {
			// If original metadata was nil, loaded should be empty map
			if len(loadedRecord.Metadata) > 0 {
				t.Errorf("Expected empty metadata for ID %s, got %v", id, loadedRecord.Metadata)
			}
		} else {
			for k, v := range originalRecord.Metadata {
				if loadedRecord.Metadata[k] != v {
					t.Errorf("Metadata mismatch for ID %s, key %s: expected %s, got %s",
						id, k, v, loadedRecord.Metadata[k])
				}
			}

			for k := range loadedRecord.Metadata {
				if _, exists := originalRecord.Metadata[k]; !exists {
					t.Errorf("Unexpected metadata key %s for ID %s", k, id)
				}
			}
		}
	}
}

// TestParquetLargeVectorCount tests saving and loading a large number of vectors
func TestParquetLargeVectorCount(t *testing.T) {
	// Create a temporary directory for testing
	tempDir, err := os.MkdirTemp("", "quiver-parquet-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Number of vectors to test
	numVectors := 2500 // Should test batch loading since batch size is 1000

	// Create test vector records
	originalRecords := make([]VectorRecord, numVectors)
	for i := 0; i < numVectors; i++ {
		originalRecords[i] = VectorRecord{
			ID:     generateTestID(i),
			Vector: generateTestVector(i, 4),
			Metadata: map[string]string{
				"index": generateTestID(i),
			},
		}
	}

	// Path for the parquet file
	parquetPath := filepath.Join(tempDir, "large_vectors.parquet")

	// Write records to parquet file
	err = WriteVectorsToParquetFile(originalRecords, parquetPath)
	if err != nil {
		t.Fatalf("Failed to write vectors to parquet: %v", err)
	}

	// Read records from parquet file
	loadedRecords, err := ReadVectorsFromParquetFile(parquetPath)
	if err != nil {
		t.Fatalf("Failed to read vectors from parquet: %v", err)
	}

	// Verify number of records matches
	if len(loadedRecords) != numVectors {
		t.Fatalf("Expected %d records, got %d", numVectors, len(loadedRecords))
	}

	// Check a few random records for correctness
	indicesToCheck := []int{0, 1, numVectors / 2, numVectors - 2, numVectors - 1}
	for _, idx := range indicesToCheck {
		if idx >= len(originalRecords) || idx >= len(loadedRecords) {
			continue // Skip if index is out of bounds
		}

		original := originalRecords[idx]
		loaded := loadedRecords[idx]

		if original.ID != loaded.ID {
			t.Errorf("ID mismatch at index %d: expected %s, got %s", idx, original.ID, loaded.ID)
		}

		if !reflect.DeepEqual(original.Vector, loaded.Vector) {
			t.Errorf("Vector mismatch at index %d", idx)
		}

		if original.Metadata["index"] != loaded.Metadata["index"] {
			t.Errorf("Metadata mismatch at index %d: expected %s, got %s",
				idx, original.Metadata["index"], loaded.Metadata["index"])
		}
	}
}

// TestParquetFileCreation tests that the parquet file is created correctly
func TestParquetFileCreation(t *testing.T) {
	// Create a temporary directory for testing
	tempDir, err := os.MkdirTemp("", "quiver-parquet-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Create nested directory path to test directory creation
	nestedDir := filepath.Join(tempDir, "nested", "dir")
	parquetPath := filepath.Join(nestedDir, "test_vectors.parquet")

	// Create test vector records
	records := []VectorRecord{
		{
			ID:       "test",
			Vector:   []float32{0.1, 0.2, 0.3},
			Metadata: map[string]string{"key": "value"},
		},
	}

	// Write records to parquet file
	err = WriteVectorsToParquetFile(records, parquetPath)
	if err != nil {
		t.Fatalf("Failed to write vectors to parquet: %v", err)
	}

	// Verify file exists
	if _, err := os.Stat(parquetPath); os.IsNotExist(err) {
		t.Errorf("Parquet file was not created at %s", parquetPath)
	}

	// Verify file contains data
	fileInfo, err := os.Stat(parquetPath)
	if err != nil {
		t.Fatalf("Failed to get file info: %v", err)
	}

	if fileInfo.Size() == 0 {
		t.Error("Parquet file is empty")
	}
}

// TestParquetEmptyRecords tests saving and loading empty records
func TestParquetEmptyRecords(t *testing.T) {
	// Create a temporary directory for testing
	tempDir, err := os.MkdirTemp("", "quiver-parquet-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Path for the parquet file
	parquetPath := filepath.Join(tempDir, "empty_vectors.parquet")

	// Create empty records slice
	var emptyRecords []VectorRecord

	// Write empty records to parquet file
	err = WriteVectorsToParquetFile(emptyRecords, parquetPath)
	if err != nil {
		t.Fatalf("Failed to write empty vectors to parquet: %v", err)
	}

	// Read records from parquet file
	loadedRecords, err := ReadVectorsFromParquetFile(parquetPath)
	if err != nil {
		t.Fatalf("Failed to read vectors from parquet: %v", err)
	}

	// Verify number of records matches
	if len(loadedRecords) != 0 {
		t.Fatalf("Expected 0 records, got %d", len(loadedRecords))
	}
}

// Helper functions for testing

// generateTestID generates a test ID for a given index
func generateTestID(index int) string {
	return "vec_" + string(rune('a'+index%26)) + "_" + string(rune('0'+index%10))
}

// generateTestVector generates a test vector for a given index and dimension
func generateTestVector(index, dim int) []float32 {
	vector := make([]float32, dim)
	for i := 0; i < dim; i++ {
		vector[i] = float32(index*10+i) / 100.0
	}
	return vector
}
