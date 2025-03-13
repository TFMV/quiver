# Adaptive Parameter Tuning (APT) System

The Adaptive Parameter Tuning (APT) system automatically optimizes HNSW parameters based on workload patterns and performance metrics. This helps Quiver achieve optimal performance without manual tuning.

## Architecture

The APT system consists of the following components:

1. **Workload Analyzer**: Collects and analyzes query patterns to understand the workload characteristics.
2. **Performance Monitor**: Tracks performance metrics such as search latency, memory usage, and CPU usage.
3. **Parameter Optimizer**: Suggests parameter changes based on workload analysis and performance metrics.
4. **Configuration Manager**: Persists parameter configurations and manages their application.

## Features

- **Automatic Parameter Tuning**: Automatically adjusts HNSW parameters based on workload and performance.
- **Workload Analysis**: Analyzes query patterns to understand the workload characteristics.
- **Performance Monitoring**: Tracks performance metrics to identify bottlenecks.
- **Multiple Optimization Strategies**: Includes strategies for latency, recall, resource efficiency, and more.
- **API Endpoints**: Provides API endpoints for monitoring and controlling the APT system.

## Configuration

The APT system can be configured through the `apt.json` file in the Quiver data directory. The configuration includes:

```json
{
  "enabled": true,
  "analyzer": {
    "history_size": 10000,
    "shift_thresholds": {
      "median_k_change": 0.3,
      "filtered_ratio_change": 0.2,
      "query_volume_change": 0.5
    }
  },
  "monitor": {
    "buffer_size": 1000,
    "thresholds": {
      "max_search_latency_ms": 100.0,
      "max_memory_bytes": 8589934592,
      "max_cpu_percent": 80.0
    }
  },
  "optimizer": {
    "min_change_interval_sec": 3600,
    "max_m": 64,
    "max_ef_construction": 500,
    "max_ef_search": 400,
    "min_m": 8,
    "min_ef_construction": 50,
    "min_ef_search": 20,
    "enabled_strategies": [
      "LatencyFocused",
      "RecallFocused",
      "WorkloadAdaptive",
      "ResourceEfficient",
      "Balanced"
    ]
  }
}
```

## API Endpoints

The APT system provides the following API endpoints:

- `GET /api/apt/status`: Get the current status of the APT system.
- `POST /api/apt/enable`: Enable the APT system.
- `POST /api/apt/disable`: Disable the APT system.
- `GET /api/apt/parameters`: Get the current HNSW parameters.
- `GET /api/apt/workload`: Get the current workload analysis.
- `GET /api/apt/performance`: Get the current performance report.
- `GET /api/apt/history`: Get the history of parameter changes.

## Optimization Strategies

The APT system includes the following optimization strategies:

1. **LatencyFocused**: Optimizes for search latency by reducing EfSearch and M.
2. **RecallFocused**: Optimizes for search recall by increasing EfSearch and M.
3. **WorkloadAdaptive**: Adjusts parameters based on query patterns, such as filtered queries and temporal patterns.
4. **ResourceEfficient**: Optimizes for resource usage by reducing M and EfSearch when memory or CPU usage is high.
5. **Balanced**: Balances recall, latency, and resource usage.

## Integration with Quiver

The APT system is integrated with Quiver through the `QuiverIntegration` class, which provides methods for recording queries, getting parameters, and controlling the APT system.

To enable or disable the APT system, use the following command:

```bash
quiver server config set adaptive_tuning true
```

## Monitoring

The APT system provides detailed monitoring information through the API endpoints. You can use these endpoints to monitor the workload, performance, and parameter changes.

For example, to get the current status of the APT system:

```bash
curl http://localhost:8080/api/apt/status
```

## Troubleshooting

If you encounter issues with the APT system, you can:

1. Check the logs for error messages.
2. Disable the APT system and use manual parameters.
3. Reset the APT system by deleting the `apt` directory in the Quiver data directory.

## License

The APT system is part of Quiver and is licensed under the same license as Quiver.
