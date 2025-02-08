// Package appender provides a high-performance Arrow appender implementation for Quiver.
package appender

import (
	"bytes"
	"fmt"
	"sync"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/ipc"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"go.uber.org/zap"
)

// ArrowAppender buffers rows using persistent column builders and writes them
// as Arrow records using the IPC writer. It is safe for concurrent use.
type ArrowAppender struct {
	mem      memory.Allocator
	buf      *bytes.Buffer
	schema   *arrow.Schema
	writer   *ipc.Writer
	builders []array.Builder
	logger   *zap.Logger

	mu    sync.Mutex
	nRows int
}

// NewArrowAppender creates a new ArrowAppender with persistent builders.
func NewArrowAppender(schema *arrow.Schema, logger *zap.Logger) (*ArrowAppender, error) {
	mem := memory.NewGoAllocator()
	buf := new(bytes.Buffer)

	writer := ipc.NewWriter(buf, ipc.WithSchema(schema))
	builders, err := createBuilders(schema, mem)
	if err != nil {
		return nil, err
	}

	return &ArrowAppender{
		mem:      mem,
		buf:      buf,
		schema:   schema,
		writer:   writer,
		builders: builders,
		logger:   logger,
		nRows:    0,
	}, nil
}

// AppendRow appends a single row (as a slice of any) to the internal builders.
func (a *ArrowAppender) AppendRow(values []any) error {
	if len(values) != len(a.builders) {
		return fmt.Errorf("value count (%d) does not match schema field count (%d)", len(values), len(a.builders))
	}

	a.mu.Lock()
	defer a.mu.Unlock()

	for i, val := range values {
		if err := appendToBuilder(a.builders[i], val, i); err != nil {
			return err
		}
	}
	a.nRows++
	return nil
}

// FlushBatch writes the buffered rows as an Arrow record.
func (a *ArrowAppender) FlushBatch() (int, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.nRows == 0 {
		return 0, nil
	}

	// Store the current row count before resetting
	flushedRows := a.nRows

	arrays := make([]arrow.Array, len(a.builders))
	defer func() {
		for _, arr := range arrays {
			arr.Release()
		}
	}()

	for i, b := range a.builders {
		arrays[i] = b.NewArray()
	}

	rec := array.NewRecord(a.schema, arrays, int64(a.nRows))
	defer rec.Release()

	if err := a.writer.Write(rec); err != nil {
		return 0, fmt.Errorf("failed to write record: %w", err)
	}

	// Reset after successful write
	a.resetBuilders()
	a.nRows = 0

	return flushedRows, nil
}

// Flush finalizes the stream and returns the serialized Arrow IPC buffer.
func (a *ArrowAppender) Flush() ([]byte, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.writer == nil {
		return nil, fmt.Errorf("writer already closed")
	}

	if a.nRows > 0 {
		if _, err := a.FlushBatch(); err != nil {
			return nil, err
		}
	}

	if err := a.writer.Close(); err != nil {
		a.logger.Error("Failed to close IPC writer", zap.Error(err))
		return nil, err
	}
	a.writer = nil
	return a.buf.Bytes(), nil
}

// Bytes returns the current contents of the buffer.
func (a *ArrowAppender) Bytes() []byte {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.buf.Bytes()
}

// Close safely shuts down the appender and releases resources.
func (a *ArrowAppender) Close() error {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.writer != nil {
		if err := a.writer.Close(); err != nil {
			a.logger.Error("Failed to close IPC writer", zap.Error(err))
			return err
		}
		a.writer = nil
	}

	a.resetBuilders()
	return nil
}

// Reset clears builders and prepares for a new batch.
func (a *ArrowAppender) Reset() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.resetBuilders()
}

// resetBuilders releases builders and recreates them.
func (a *ArrowAppender) resetBuilders() {
	for _, b := range a.builders {
		b.Release()
	}
	a.builders, _ = createBuilders(a.schema, a.mem)
	a.nRows = 0
}

// createBuilders initializes Arrow builders based on schema.
func createBuilders(schema *arrow.Schema, mem memory.Allocator) ([]array.Builder, error) {
	builders := make([]array.Builder, len(schema.Fields()))
	for i, field := range schema.Fields() {
		switch field.Type.ID() {
		case arrow.INT64:
			builders[i] = array.NewInt64Builder(mem)
		case arrow.FLOAT32:
			builders[i] = array.NewFloat32Builder(mem)
		case arrow.STRING:
			builders[i] = array.NewStringBuilder(mem)
		default:
			return nil, fmt.Errorf("unsupported field type: %s", field.Type)
		}
	}
	return builders, nil
}

// appendToBuilder appends a value to a specific builder.
func appendToBuilder(b array.Builder, val any, index int) error {
	switch b := b.(type) {
	case *array.Int64Builder:
		v, ok := val.(int64)
		if !ok {
			return fmt.Errorf("expected int64 for field %d", index)
		}
		b.Append(v)
	case *array.Float32Builder:
		v, ok := val.(float32)
		if !ok {
			return fmt.Errorf("expected float32 for field %d", index)
		}
		b.Append(v)
	case *array.StringBuilder:
		v, ok := val.(string)
		if !ok {
			return fmt.Errorf("expected string for field %d", index)
		}
		b.Append(v)
	default:
		return fmt.Errorf("unsupported builder type at field %d", index)
	}
	return nil
}
