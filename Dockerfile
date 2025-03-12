FROM --platform=linux/arm64 golang:1.24-bookworm

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    git \
    unzip \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set up DuckDB library for ARM64
RUN wget https://github.com/duckdb/duckdb/releases/download/v0.10.0/libduckdb-linux-aarch64.zip \
    && unzip libduckdb-linux-aarch64.zip -d /usr/lib/ \
    && ln -s /usr/lib/libduckdb.so /usr/lib/libduckdb.so.0 \
    && mkdir -p /usr/local/lib \
    && ln -s /usr/lib/libduckdb.so /usr/local/lib/libduckdb.so \
    && ln -s /usr/lib/libduckdb.so /usr/local/lib/libduckdb.so.0

# Copy and build the application
COPY go.mod go.sum ./
RUN go mod download

COPY . .

# Build for ARM64
RUN echo "Building for ARM64..." && \
    CGO_ENABLED=1 GOOS=linux GOARCH=arm64 go build -v -o /app/quiver-server cmd/main.go && \
    echo "Build complete. Checking binary..." && \
    ls -la /app/quiver-server && \
    echo "Binary details complete."

RUN chmod +x /app/quiver-server

# Set up environment
ENV QUIVER_STORAGE_PATH=/data
ENV QUIVER_BACKUP_PATH=/data/backups
ENV QUIVER_PORT=8080
ENV LD_LIBRARY_PATH=/usr/lib:/usr/local/lib

# Create data directories
RUN mkdir -p /data /backups

EXPOSE 8080

VOLUME ["/data", "/backups"]
CMD ["/app/quiver-server", "serve"]