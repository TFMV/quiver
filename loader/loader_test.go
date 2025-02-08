package loader

import (
	"os"
	"path/filepath"
	"testing"

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
	logger := zap.NewNop()

	csvData := "id,v1,v2,v3\n1,0.1,0.2,0.3\n2,0.4,0.5,0.6\n"
	csvFile := createTempFile(t, csvData, ".csv")
	defer os.Remove(csvFile)

	records, err := LoadCSVFile(csvFile, logger)
	require.NoError(t, err)
	assert.Len(t, records, 2)
	assert.Equal(t, 1, records[0].ID)
	assert.Equal(t, []float32{0.1, 0.2, 0.3}, records[0].Vector)
}

// Test loading Parquet files (Requires valid Parquet fixture)
func TestLoadParquetFile(t *testing.T) {
	logger := zap.NewNop()

	// TODO: Generate a test Parquet file with the expected schema.
	// For now, this test is a placeholder.
	parquetFile := createTempFile(t, "", ".parquet")
	defer os.Remove(parquetFile)

	_, err := LoadParquetFile(parquetFile, logger)
	assert.Error(t, err) // Expect an error since file is empty
}

// Test directory scanning and file loading
func TestLoadFilesFromDirectory(t *testing.T) {
	logger := zap.NewNop()

	// Create temporary test directory
	dir := t.TempDir()

	// Create test JSON and CSV files
	jsonFile := filepath.Join(dir, "data.json")
	os.WriteFile(jsonFile, []byte(`[{"id":1,"vector":[0.1,0.2,0.3]}]`), 0644)

	csvFile := filepath.Join(dir, "data.csv")
	os.WriteFile(csvFile, []byte("id,v1,v2,v3\n2,0.4,0.5,0.6\n"), 0644)

	// Create a test vector index
	vi, err := quiver.NewVectorIndex(3, "test.db", "test.hnsw", quiver.Cosine)
	require.NoError(t, err)
	defer os.Remove("test.db")
	defer os.Remove("test.hnsw")

	// Run the loader
	err = LoadFilesFromDirectory(dir, vi, logger)
	require.NoError(t, err)

	// Check if vectors were added
	query := []float32{0.5, 0.5, 0.5}
	results, err := vi.Search(query, 2)
	require.NoError(t, err)
	assert.NotEmpty(t, results)
}
