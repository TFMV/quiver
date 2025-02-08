package appender

import (
	"testing"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/stretchr/testify/assert"
	"go.uber.org/zap"
)

func TestArrowAppender(t *testing.T) {
	logger, _ := zap.NewDevelopment()
	schema := arrow.NewSchema([]arrow.Field{
		{Name: "id", Type: arrow.PrimitiveTypes.Int64, Nullable: false},
		{Name: "score", Type: arrow.PrimitiveTypes.Float32, Nullable: false},
		{Name: "name", Type: arrow.BinaryTypes.String, Nullable: false},
	}, nil)

	appender, err := NewArrowAppender(schema, logger)
	assert.NoError(t, err, "Failed to create ArrowAppender")
	defer appender.Close()

	// Test appending valid rows
	err = appender.AppendRow([]interface{}{int64(1), float32(4.5), "Alice"})
	assert.NoError(t, err, "Failed to append row")

	err = appender.AppendRow([]interface{}{int64(2), float32(9.3), "Bob"})
	assert.NoError(t, err, "Failed to append row")

	// Test schema mismatch
	err = appender.AppendRow([]interface{}{1, "invalid_type", 10.5}) // Wrong types
	assert.Error(t, err, "Expected error on type mismatch")

	// Test flushing
	rowsFlushed, err := appender.FlushBatch()
	assert.NoError(t, err, "Failed to flush batch")
	assert.Equal(t, 2, rowsFlushed, "Expected 2 rows to be flushed")

	// Test empty flush
	rowsFlushed, err = appender.FlushBatch()
	assert.NoError(t, err, "Flush should not error")
	assert.Equal(t, 0, rowsFlushed, "Expected 0 rows to be flushed")

	// Test retrieving serialized Arrow IPC bytes
	data, err := appender.Flush()
	assert.NoError(t, err, "Failed to flush final batch")
	assert.Greater(t, len(data), 0, "Expected non-empty Arrow IPC buffer")
}
