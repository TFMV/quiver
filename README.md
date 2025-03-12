# Quiver

Quiver is an experimental hybrid vector search database that combines the speed of in-memory vector search with the durability and query capabilities of a relational database.

[![Build](https://github.com/TFMV/quiver/actions/workflows/go.yml/badge.svg)](https://github.com/TFMV/quiver/actions/workflows/go.yml)
[![Go Report Card](https://goreportcard.com/badge/github.com/TFMV/quiver)](https://goreportcard.com/report/github.com/TFMV/quiver)
[![GoDoc](https://pkg.go.dev/badge/github.com/TFMV/quiver)](https://pkg.go.dev/github.com/TFMV/quiver)
[![Release](https://img.shields.io/github/v/release/TFMV/quiver)](https://github.com/TFMV/quiver/releases)
[![Go 1.24](https://img.shields.io/badge/Go-1.24-blue)](https://golang.org/doc/go1.24)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## About Quiver

Quiver is designed as an experimental vector database that combines:

- **Fast Vector Search**: Using an optimized HNSW (Hierarchical Navigable Small World) graph implementation for efficient approximate nearest neighbor search
- **SQL Capabilities**: Powered by DuckDB for robust metadata filtering and complex queries
- **Hybrid Search**: Combining vector similarity search with metadata filtering in a single operation
- **Durability**: Persistent storage with automatic backups and recovery
- **Connection Pooling**: Optimized database connections for concurrent operations

> **Note**: Quiver is currently experimental and under active development. While it's designed with performance and stability in mind, it is not suitable for production workloads.

## Key Features

- **Efficient Vector Operations**: Optimized for both cosine similarity and L2 distance metrics
- **Metadata Filtering**: SQL-based filtering of vector search results
- **Negative Examples**: Support for search with negative examples to refine results
- **Batch Processing**: Efficient batch operations for vector additions
- **Automatic Persistence**: Configurable persistence intervals for durability
- **Backup & Restore**: Built-in backup and restore functionality
- **Connection Pooling**: Optimized database connection management
- **Thread Safety**: Designed for concurrent access with appropriate synchronization

## Current Status

Quiver is in active development with a focus on:

1. **Performance Optimization**: Continuous benchmarking and optimization of vector operations
2. **Stability Improvements**: Addressing resource management and concurrency issues
3. **API Refinement**: Evolving the API based on real-world usage patterns
4. **Feature Expansion**: Adding new capabilities while maintaining performance

Recent optimizations have focused on connection pooling, resource management, and eliminating data races in concurrent operations.

## Target State

The target state of the database is described in our [Documentation Site](https://tfmv.github.io/quiver/).

## License

[MIT License](LICENSE)
