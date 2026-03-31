package observability

import (
	"context"
	"fmt"
	"io"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/exp/slog"
)

type Level = slog.Level

var (
	LevelDebug = slog.LevelDebug
	LevelInfo  = slog.LevelInfo
	LevelWarn  = slog.LevelWarn
	LevelError = slog.LevelError
)

type Logger struct {
	logger *slog.Logger
	output io.Writer
	level  atomic.Int64
	mu     sync.RWMutex
}

var defaultLogger *Logger

func init() {
	defaultLogger = NewLogger(os.Stdout, LevelInfo)
}

func Default() *Logger {
	return defaultLogger
}

func SetDefault(l *Logger) {
	defaultLogger = l
}

func NewLogger(output io.Writer, level Level) *Logger {
	l := &Logger{
		output: output,
	}
	l.level.Store(int64(level))

	handler := slog.NewJSONHandler(output, &slog.HandlerOptions{
		AddSource: true,
	})
	l.logger = slog.New(handler)

	return l
}

func (l *Logger) SetLevel(level Level) {
	l.level.Store(int64(level))
}

func (l *Logger) Debug(msg string, attrs ...any) {
	l.log(LevelDebug, msg, attrs...)
}

func (l *Logger) Info(msg string, attrs ...any) {
	l.log(LevelInfo, msg, attrs...)
}

func (l *Logger) Warn(msg string, attrs ...any) {
	l.log(LevelWarn, msg, attrs...)
}

func (l *Logger) Error(msg string, attrs ...any) {
	l.log(LevelError, msg, attrs...)
}

func (l *Logger) log(level Level, msg string, attrs ...any) {
	if int64(level) < l.level.Load() {
		return
	}
	l.logger.Log(context.Background(), level, msg, attrs...)
}

func (l *Logger) With(attrs ...any) *Logger {
	newLogger := &Logger{
		logger: l.logger.With(attrs...),
		output: l.output,
	}
	newLogger.level.Store(l.level.Load())
	return newLogger
}

func Debug(msg string, attrs ...any) {
	defaultLogger.Debug(msg, attrs...)
}

func Info(msg string, attrs ...any) {
	defaultLogger.Info(msg, attrs...)
}

func Warn(msg string, attrs ...any) {
	defaultLogger.Warn(msg, attrs...)
}

func Error(msg string, attrs ...any) {
	defaultLogger.Error(msg, attrs...)
}

type TraceID string

func NewTraceID() TraceID {
	return TraceID(time.Now().Format("20060102.150405.000000"))
}

type Span struct {
	TraceID   TraceID
	SpanID    string
	ParentID  string
	Name      string
	StartTime time.Time
	EndTime   time.Time
	Attrs     map[string]any
	mu        sync.Mutex
	completed bool
}

type Tracer struct {
	logger  *Logger
	enabled atomic.Bool
}

var defaultTracer *Tracer

func init() {
	defaultTracer = &Tracer{
		logger:  defaultLogger,
		enabled: atomic.Bool{},
	}
}

func DefaultTracer() *Tracer {
	return defaultTracer
}

func (t *Tracer) SetEnabled(enabled bool) {
	t.enabled.Store(enabled)
}

func (t *Tracer) Enabled() bool {
	return t.enabled.Load()
}

func (t *Tracer) StartSpan(ctx context.Context, name string, attrs ...any) *Span {
	if !t.enabled.Load() {
		return nil
	}

	span := &Span{
		TraceID:   NewTraceID(),
		SpanID:    string(NewTraceID())[:8],
		Name:      name,
		StartTime: time.Now(),
		Attrs:     make(map[string]any),
	}

	for i := 0; i < len(attrs); i += 2 {
		if i+1 < len(attrs) {
			span.Attrs[fmt.Sprintf("%v", attrs[i])] = attrs[i+1]
		}
	}

	t.logger.Debug("span started", "trace_id", span.TraceID, "span_id", span.SpanID, "name", name)
	return span
}

func (t *Tracer) EndSpan(span *Span) {
	if span == nil || !t.enabled.Load() {
		return
	}

	span.EndTime = time.Now()
	span.mu.Lock()
	if span.completed {
		span.mu.Unlock()
		return
	}
	span.completed = true
	span.mu.Unlock()

	duration := span.EndTime.Sub(span.StartTime)
	t.logger.Debug("span ended",
		"trace_id", span.TraceID,
		"span_id", span.SpanID,
		"name", span.Name,
		"duration_ms", duration.Milliseconds(),
	)
}

func (t *Tracer) RecordSpan(span *Span) {
	if span == nil || !t.enabled.Load() {
		return
	}

	span.EndTime = time.Now()
	duration := span.EndTime.Sub(span.StartTime)

	attrs := make([]any, 0, len(span.Attrs)*2+4)
	attrs = append(attrs, "trace_id", span.TraceID, "span_id", span.SpanID, "name", span.Name, "duration_ms", duration.Milliseconds())
	for k, v := range span.Attrs {
		attrs = append(attrs, k, v)
	}

	t.logger.Info("span", attrs...)
}

func (s *Span) SetAttr(key string, value any) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Attrs[key] = value
}

func (s *Span) End() {
	if s == nil {
		return
	}
	s.EndTime = time.Now()
	s.mu.Lock()
	s.completed = true
	s.mu.Unlock()
}

func (s *Span) Duration() time.Duration {
	if s.EndTime.IsZero() {
		return time.Since(s.StartTime)
	}
	return s.EndTime.Sub(s.StartTime)
}

func StartSpan(ctx context.Context, name string, attrs ...any) *Span {
	return defaultTracer.StartSpan(ctx, name, attrs...)
}

func EndSpan(span *Span) {
	defaultTracer.EndSpan(span)
}

type Observer struct {
	logger  *Logger
	enabled atomic.Bool
}

func (o *Observer) SetEnabled(enabled bool) {
	o.enabled.Store(enabled)
}

func (o *Observer) Observe(name string, duration time.Duration, attrs ...any) {
	if !o.enabled.Load() {
		return
	}

	attrs = append(attrs, "duration_ms", duration.Milliseconds())
	o.logger.Debug(name, attrs...)
}
