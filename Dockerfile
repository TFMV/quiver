# Build stage
FROM golang:1.24-alpine AS builder

# Install build dependencies
RUN apk add --no-cache git build-base

# Set working directory
WORKDIR /app

# Copy go.mod and go.sum files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the application with optimization flags
RUN mkdir -p /app/bin && \
    CGO_ENABLED=1 GOOS=linux GOARCH=amd64 \
    go build -a -ldflags="-w -s" -installsuffix cgo -o /app/bin/quiver ./cmd/quiver

# Final stage
FROM alpine:3.18

# Install runtime dependencies
RUN apk add --no-cache ca-certificates tzdata wget

# Create non-root user
RUN adduser -D -h /app quiver

# Set working directory
WORKDIR /app

# Copy binary from builder stage
COPY --from=builder /app/bin/quiver /app/quiver

# Copy default configuration
COPY --from=builder /app/cmd/quiver/config.yaml /app/config.yaml

# Create data directory and set permissions
RUN mkdir -p /app/data && chown -R quiver:quiver /app

# Switch to non-root user
USER quiver

# Expose API port
EXPOSE 8080

# Set volume for persistent data
VOLUME ["/app/data"]

# Set environment variables
ENV QUIVER_SERVER_PORT=8080
ENV QUIVER_SERVER_HOST=0.0.0.0
ENV QUIVER_SERVER_STORAGE=/app/data
ENV QUIVER_INDEX_DIMENSION=768
ENV QUIVER_INDEX_DISTANCE=cosine
ENV QUIVER_ENABLE_HYBRID=true
ENV QUIVER_BATCH_SIZE=100
ENV QUIVER_ENABLE_METRICS=true
ENV QUIVER_ENABLE_PERSISTENCE=true
ENV QUIVER_PERSISTENCE_FORMAT=parquet
ENV QUIVER_FLUSH_INTERVAL=300
ENV QUIVER_MAX_CONNECTIONS=16
ENV QUIVER_EF_CONSTRUCTION=200
ENV QUIVER_EF_SEARCH=100
ENV QUIVER_LOG_LEVEL=info

# Set entrypoint
ENTRYPOINT ["/app/quiver"]

# Set default command - server with improved defaults for production use
CMD ["server", "--port", "8080", "--storage", "/app/data", "--enable-hybrid", "--enable-metrics", "--enable-persistence"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget -qO- http://localhost:8080/health || exit 1 