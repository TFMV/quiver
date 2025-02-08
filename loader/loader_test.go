package loader

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"

	quiver "github.com/TFMV/quiver"
)

// Helper function to create temporary test files
func createTempFile(t *testing.T, content string, ext string) string {
	t.Helper()
	tmpFile, err := os.CreateTemp("", "testfile-*"+ext)
	require.NoError(t, err)

	_, err = tmpFile.WriteString(content)
	require.NoError(t, err)
	tmpFile.Close()

	return tmpFile.Name()
}

// Test loading JSON files
func TestLoadJSONFile(t *testing.T) {
	logger := zap.NewNop()

	jsonData := `[{"id":1,"vector":[0.1,0.2,0.3]},{"id":2,"vector":[0.4,0.5,0.6]}]`
	jsonFile := createTempFile(t, jsonData, ".json")
	defer os.Remove(jsonFile)

	records, err := LoadJSONFile(jsonFile, logger)
	require.NoError(t, err)
	assert.Len(t, records, 2)
	assert.Equal(t, 1, records[0].ID)
	assert.Equal(t, []float32{0.1, 0.2, 0.3}, records[0].Vector)
}

// Test loading CSV files
func TestLoadCSVFile(t *testing.T) {
	// Create a temporary CSV file
	tmpfile, err := os.CreateTemp("", "testfile-*.csv")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpfile.Name())

	// Write test data
	csvData := `id,vector
1,"[0.1,0.2,0.3]"
2,"[0.4,0.5,0.6]"
`
	if _, err := tmpfile.WriteString(csvData); err != nil {
		t.Fatal(err)
	}
	if err := tmpfile.Close(); err != nil {
		t.Fatal(err)
	}

	// Load and verify
	logger, _ := zap.NewDevelopment()
	records, err := LoadCSVFile(tmpfile.Name(), logger)
	if err != nil {
		t.Fatal(err)
	}

	assert.Len(t, records, 2)
	assert.Equal(t, 1, records[0].ID)
	assert.Equal(t, []float32{0.1, 0.2, 0.3}, records[0].Vector)
}

// Test loading Parquet files (Requires valid Parquet fixture)
func TestLoadParquetFile(t *testing.T) {
	logger := zap.NewNop()
	parquetPath := "../data/data.parquet"

	records, err := LoadParquetFile(parquetPath, logger)
	require.NoError(t, err)

	// Optional: Validate that some records were loaded
	require.NotEmpty(t, records, "Expected at least one record to be loaded")
}

// Test directory scanning and file loading
func TestLoadFilesFromDirectory(t *testing.T) {
	logger := zap.NewNop()
	dir := t.TempDir()

	// Create test files
	jsonFile := filepath.Join(dir, "data.json")
	os.WriteFile(jsonFile, []byte(`[{"id":1,"vector":[0.1,0.2,0.3]}]`), 0644)

	csvFile := filepath.Join(dir, "data.csv")
	csvData := `id,vector
1,"[0.4,0.5,0.6]"
`
	os.WriteFile(csvFile, []byte(csvData), 0644)

	// Create vector index
	vi, err := quiver.NewVectorIndex(3, "test.db", "test.hnsw", quiver.Cosine)
	require.NoError(t, err)
	defer os.Remove("test.db")
	defer os.Remove("test.hnsw")

	err = LoadFilesFromDirectory(dir, vi, logger)
	require.NoError(t, err)

	time.Sleep(100 * time.Millisecond)

	query := []float32{0.3, 0.3, 0.3}
	results, err := vi.Search(query, 1)
	require.NoError(t, err)
	assert.NotEmpty(t, results)
}
