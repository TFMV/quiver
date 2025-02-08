package appender

import (
	"testing"

	"github.com/apache/arrow-go/v18/arrow"
	"go.uber.org/zap"
)

func BenchmarkArrowAppender_AppendRow(b *testing.B) {
	logger, _ := zap.NewDevelopment()
	schema := arrow.NewSchema([]arrow.Field{
		{Name: "id", Type: arrow.PrimitiveTypes.Int64, Nullable: false},
		{Name: "score", Type: arrow.PrimitiveTypes.Float32, Nullable: false},
		{Name: "name", Type: arrow.BinaryTypes.String, Nullable: false},
	}, nil)

	appender, _ := NewArrowAppender(schema, logger)
	defer appender.Close()

	row := []interface{}{int64(1), float32(9.5), "Test"}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = appender.AppendRow(row)
	}
}

func BenchmarkArrowAppender_FlushBatch(b *testing.B) {
	logger, _ := zap.NewDevelopment()
	schema := arrow.NewSchema([]arrow.Field{
		{Name: "id", Type: arrow.PrimitiveTypes.Int64, Nullable: false},
		{Name: "score", Type: arrow.PrimitiveTypes.Float32, Nullable: false},
		{Name: "name", Type: arrow.BinaryTypes.String, Nullable: false},
	}, nil)

	appender, _ := NewArrowAppender(schema, logger)
	defer appender.Close()

	// Preload 10,000 rows
	for i := 0; i < 10000; i++ {
		_ = appender.AppendRow([]interface{}{int64(i), float32(i) * 1.5, "Bench"})
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = appender.FlushBatch()
	}
}

func BenchmarkArrowAppender_Flush(b *testing.B) {
	logger, _ := zap.NewDevelopment()
	schema := arrow.NewSchema([]arrow.Field{
		{Name: "id", Type: arrow.PrimitiveTypes.Int64, Nullable: false},
		{Name: "score", Type: arrow.PrimitiveTypes.Float32, Nullable: false},
		{Name: "name", Type: arrow.BinaryTypes.String, Nullable: false},
	}, nil)

	appender, _ := NewArrowAppender(schema, logger)
	defer appender.Close()

	// Preload 10,000 rows
	for i := 0; i < 10000; i++ {
		_ = appender.AppendRow([]interface{}{int64(i), float32(i) * 1.5, "Bench"})
	}

	_, _ = appender.FlushBatch()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = appender.Flush()
	}
}
