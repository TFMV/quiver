FROM golang:1.24-alpine AS builder

WORKDIR /app

# Install build dependencies
RUN apk add --no-cache git build-base

# Copy go.mod and go.sum files
COPY go.mod go.sum ./
RUN go mod download

# Copy the source code
COPY . .

# Build the application
RUN CGO_ENABLED=1 GOOS=linux go build -a -o quiver-server ./cmd/server

# Use a smaller image for the final container
FROM alpine:3.18

WORKDIR /app

# Install runtime dependencies
RUN apk add --no-cache ca-certificates tzdata libc6-compat

# Copy the binary from the builder stage
COPY --from=builder /app/quiver-server /app/quiver-server

# Create directories for data persistence
RUN mkdir -p /data/quiver /data/backups

# Set environment variables
ENV QUIVER_STORAGE_PATH=/data/quiver
ENV QUIVER_BACKUP_PATH=/data/backups
ENV QUIVER_PORT=8080

# Expose the port
EXPOSE 8080

# Set the entrypoint
ENTRYPOINT ["/app/quiver-server"] 