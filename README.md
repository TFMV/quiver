# Quiver

Blazing-Fast, Embeddable, Structured Vector Search in Go

[![Go Report Card](https://goreportcard.com/badge/github.com/TFMV/quiver)](https://goreportcard.com/report/github.com/TFMV/quiver)
[![GoDoc](https://pkg.go.dev/badge/github.com/TFMV/quiver)](https://pkg.go.dev/github.com/TFMV/quiver)

## 🚀 Overview

Quiver is a lightweight, high-performance vector search engine designed for structured datasets.

## 🏗 Architecture

Quiver uses HNSW for efficient vector indexing and DuckDB for metadata storage.

```mermaid
flowchart TD
  %% Initialization Sequence
  A["Start: Initialize Quiver Index"]
  B["Validate Config"]
  C["Initialize HNSW Index\n(set dimension, M, efConstruction, etc.)"]
  D["Open DuckDB Connection\n(using StoragePath)"]
  E["Create Metadata Table\n(if not exists)"]
  F["Index Object\n(config, hnsw, db, metadata map,\nbatchBuffer, cache, locks)"]
  G["Start Background Batch Processor\n(ticker triggers flushBatch)"]

  A --> B
  B --> C
  C --> D
  D --> E
  E --> F
  F --> G

  %% Vector Insertion Flow
  H["Receive Add Vector Request\n(id, vector, metadata)"]
  I["Check Dimension & Append to BatchBuffer"]
  J["Current Batch Size >= Threshold?"]
  K["Trigger Async Flush Batch"]
  L["BatchProcessor: flushBatch()\n(acquire batchLock)"]
  M["Begin DB Transaction & Prepare SQL Statement"]
  N["Loop Through BatchBuffer:\n- Store vector in memory\n- Update metadata map & cache\n- Execute SQL insert/update"]
  O["Call HNSW.AddPoints() to insert vectors"]
  P["Commit Transaction"]
  Q["Clear BatchBuffer"]

  F --> H
  H --> I
  I --> J
  J -- Yes --> K
  J -- No --> end_insertion["Wait for ticker"]
  K --> L
  L --> M
  M --> N
  N --> O
  O --> P
  P --> Q

  %% Vector Search Flow
  R["Receive Search Query\n(vector, k)"]
  S["Acquire Read Lock"]
  T["Call HNSW.SearchKNN()"]
  U["For each result:\nRetrieve metadata from in-memory map or cache"]
  V["Return Search Results (id, distance, metadata)"]

  F --> R
  R --> S
  S --> T
  T --> U
  U --> V

  %% Search with Metadata Filter Flow
  W["Receive SearchWithFilter Query\n(vector, k, filter)"]
  X["Run DuckDB Query to filter metadata"]
  Y["Get Filtered IDs"]
  Z{"Is Filtered ID set small?"}
  AA["Perform HNSW search on query vector only"]
  AB["Loop through results to match filtered IDs"]
  AC["Else: Perform full vector search then filter results by metadata"]
  AD["Return filtered Search Results"]

  F --> W
  W --> X
  X --> Y
  Y --> Z
  Z -- Yes --> AA
  AA --> AB
  AB --> AD
  Z -- No --> AC
  AC --> AD

  %% Arrow Integration Flow
  AE["Receive Arrow Record"]
  AF["Extract Columns: id, vector (FixedSizeList), metadata (JSON)"]
  AG["For each row:\n- Unmarshal metadata\n- Build vector slice"]
  AH["Call Add() for each vector row"]

  F --> AE
  AE --> AF
  AF --> AG
  AG --> AH

  %% Persistence and Cleanup
  AI["Save Operation:\n- Flush pending batch\n- Save HNSW index to disk\n- Write metadata.json"]
  AJ["Load Operation:\n- Load HNSW index from disk\n- Read metadata.json"]
  AK["Close Operation:\n- Stop ticker\n- Flush remaining batch\n- Free HNSW index\n- Close DuckDB connection"]

  F --> AI
  F --> AJ
  F --> AK
```

## 🌟 Features

- **High-Performance**: Utilizes HNSW for efficient vector indexing
- **Structured Data**: Supports metadata storage for structured datasets
- **Lightweight**: Minimal dependencies, easy to embed in your Go applications
- **Flexible**: Supports various distance metrics (e.g., Euclidean, Cosine)
- **Configurable**: Customize HNSW parameters for optimal performance

## :small_airplane: Performance

The following benchmarks were performed on a 2024 MacBook Pro with an M2 Pro CPU.

| Operation | Operations/sec | Latency | Memory (B/op) | Allocations (allocs/op) |
|-----------|---------------|--------------|---------------|------------------------|
| Add | 4,872 | 3.178ms | 1,389 | 21 |
| Search | 30,246 | 44.338µs | 1,520 | 18 |
| Hybrid Search | 2,395 | 431.622µs | 7,533 | 278 |
| Add Parallel | 2,850 | 6.088ms | 1,282 | 19 |
| Arrow Append | 100 | 2.770s | 2,336,470 | 32377 |

Benchmark details:

- Vector dimension: 128
- Dataset size: 10,000 vectors
- HNSW M: 32
- HNSW efConstruction: 200
- HNSW efSearch: 200
- Batch size: 1,000

Benchmark notes:

- Arrow Append is a one-time operation that pre-allocates memory for the entire dataset.
- The other benchmarks are for typical insert/search operations.
