# Build stage
FROM golang:1.21-alpine AS builder

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

# Build the application
RUN CGO_ENABLED=1 GOOS=linux go build -a -installsuffix cgo -o quiver ./quiver/cmd/cli

# Final stage
FROM alpine:3.18

# Install runtime dependencies
RUN apk add --no-cache ca-certificates tzdata

# Create non-root user
RUN adduser -D -h /app quiver

# Set working directory
WORKDIR /app

# Copy binary from builder stage
COPY --from=builder /app/quiver /app/quiver

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
ENV QUIVER_INDEX_DIMENSION=128
ENV QUIVER_INDEX_MAX_ELEMENTS=1000000
ENV QUIVER_INDEX_DISTANCE=cosine

# Set entrypoint
ENTRYPOINT ["/app/quiver"]

# Set default command
CMD ["server", "--port", "8080", "--storage", "/app/data"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget -qO- http://localhost:8080/health || exit 1 