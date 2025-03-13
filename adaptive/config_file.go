// Package adaptive implements the Adaptive Parameter Tuning (APT) system for Quiver.
package adaptive

import (
	"encoding/json"
	"io/ioutil"
	"os"
	"path/filepath"
	"time"
)

// ConfigFile represents the configuration file for the APT system
type ConfigFile struct {
	// Whether adaptive tuning is enabled
	Enabled bool `json:"enabled"`

	// Analyzer configuration
	Analyzer struct {
		// Size of the query history buffer
		HistorySize int `json:"history_size"`

		// Thresholds for detecting workload shifts
		ShiftThresholds struct {
			// Minimum change in median K to consider a shift
			MedianKChange float64 `json:"median_k_change"`

			// Minimum change in filtered query ratio to consider a shift
			FilteredRatioChange float64 `json:"filtered_ratio_change"`

			// Minimum change in query volume to consider a shift
			QueryVolumeChange float64 `json:"query_volume_change"`
		} `json:"shift_thresholds"`
	} `json:"analyzer"`

	// Monitor configuration
	Monitor struct {
		// Size of metric buffers
		BufferSize int `json:"buffer_size"`

		// Performance thresholds
		Thresholds struct {
			// Maximum acceptable search latency (p95) in milliseconds
			MaxSearchLatencyMs float64 `json:"max_search_latency_ms"`

			// Maximum acceptable memory usage in bytes
			MaxMemoryBytes uint64 `json:"max_memory_bytes"`

			// Maximum acceptable CPU usage percentage
			MaxCPUPercent float64 `json:"max_cpu_percent"`
		} `json:"thresholds"`
	} `json:"monitor"`

	// Optimizer configuration
	Optimizer struct {
		// Minimum time between parameter changes (in seconds)
		MinChangeIntervalSec int `json:"min_change_interval_sec"`

		// Maximum parameter values
		MaxM              int `json:"max_m"`
		MaxEfConstruction int `json:"max_ef_construction"`
		MaxEfSearch       int `json:"max_ef_search"`

		// Minimum parameter values
		MinM              int `json:"min_m"`
		MinEfConstruction int `json:"min_ef_construction"`
		MinEfSearch       int `json:"min_ef_search"`

		// Enabled strategies
		EnabledStrategies []string `json:"enabled_strategies"`
	} `json:"optimizer"`
}

// LoadConfigFile loads the APT configuration from a file
func LoadConfigFile(configDir string) (*ConfigFile, error) {
	// Create default config
	config := &ConfigFile{
		Enabled: true,
	}

	// Set default analyzer config
	config.Analyzer.HistorySize = 10000
	config.Analyzer.ShiftThresholds.MedianKChange = 0.3
	config.Analyzer.ShiftThresholds.FilteredRatioChange = 0.2
	config.Analyzer.ShiftThresholds.QueryVolumeChange = 0.5

	// Set default monitor config
	config.Monitor.BufferSize = 1000
	config.Monitor.Thresholds.MaxSearchLatencyMs = 100.0
	config.Monitor.Thresholds.MaxMemoryBytes = 8 * 1024 * 1024 * 1024 // 8GB
	config.Monitor.Thresholds.MaxCPUPercent = 80.0

	// Set default optimizer config
	config.Optimizer.MinChangeIntervalSec = 3600 // 1 hour
	config.Optimizer.MaxM = 64
	config.Optimizer.MaxEfConstruction = 500
	config.Optimizer.MaxEfSearch = 400
	config.Optimizer.MinM = 8
	config.Optimizer.MinEfConstruction = 50
	config.Optimizer.MinEfSearch = 20
	config.Optimizer.EnabledStrategies = []string{
		"LatencyFocused",
		"RecallFocused",
		"WorkloadAdaptive",
		"ResourceEfficient",
		"Balanced",
	}

	// Check if config file exists
	configFile := filepath.Join(configDir, "apt.json")
	if _, err := os.Stat(configFile); os.IsNotExist(err) {
		// Create config directory if it doesn't exist
		if err := os.MkdirAll(configDir, 0755); err != nil {
			return config, err
		}

		// Save default config
		if err := SaveConfigFile(config, configDir); err != nil {
			return config, err
		}

		return config, nil
	}

	// Read config file
	data, err := ioutil.ReadFile(configFile)
	if err != nil {
		return config, err
	}

	// Parse config file
	if err := json.Unmarshal(data, config); err != nil {
		return config, err
	}

	return config, nil
}

// SaveConfigFile saves the APT configuration to a file
func SaveConfigFile(config *ConfigFile, configDir string) error {
	// Create config directory if it doesn't exist
	if err := os.MkdirAll(configDir, 0755); err != nil {
		return err
	}

	// Marshal config
	data, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return err
	}

	// Write config file
	configFile := filepath.Join(configDir, "apt.json")
	return ioutil.WriteFile(configFile, data, 0644)
}

// ToIntegrationConfig converts a ConfigFile to an IntegrationConfig
func (cf *ConfigFile) ToIntegrationConfig(configDir string) IntegrationConfig {
	ic := IntegrationConfig{
		Enabled:   cf.Enabled,
		ConfigDir: configDir,
	}

	// Set analyzer config
	ic.AnalyzerConfig = AnalyzerConfig{
		HistorySize: cf.Analyzer.HistorySize,
		ShiftThresholds: WorkloadShiftThresholds{
			MedianKChange:       cf.Analyzer.ShiftThresholds.MedianKChange,
			FilteredRatioChange: cf.Analyzer.ShiftThresholds.FilteredRatioChange,
			QueryVolumeChange:   cf.Analyzer.ShiftThresholds.QueryVolumeChange,
		},
	}

	// Set monitor config
	ic.MonitorConfig = MonitorConfig{
		BufferSize: cf.Monitor.BufferSize,
		Thresholds: PerformanceThresholds{
			MaxSearchLatencyMs: cf.Monitor.Thresholds.MaxSearchLatencyMs,
			MaxMemoryBytes:     cf.Monitor.Thresholds.MaxMemoryBytes,
			MaxCPUPercent:      cf.Monitor.Thresholds.MaxCPUPercent,
		},
	}

	// Set optimizer config
	ic.OptimizerConfig = OptimizerConfig{
		MinChangeInterval: secondsToTime(cf.Optimizer.MinChangeIntervalSec),
		MaxM:              cf.Optimizer.MaxM,
		MaxEfConstruction: cf.Optimizer.MaxEfConstruction,
		MaxEfSearch:       cf.Optimizer.MaxEfSearch,
		MinM:              cf.Optimizer.MinM,
		MinEfConstruction: cf.Optimizer.MinEfConstruction,
		MinEfSearch:       cf.Optimizer.MinEfSearch,
		EnabledStrategies: cf.Optimizer.EnabledStrategies,
		Thresholds: PerformanceThresholds{
			MaxSearchLatencyMs: cf.Monitor.Thresholds.MaxSearchLatencyMs,
			MaxMemoryBytes:     cf.Monitor.Thresholds.MaxMemoryBytes,
			MaxCPUPercent:      cf.Monitor.Thresholds.MaxCPUPercent,
		},
	}

	return ic
}

// secondsToTime converts seconds to time.Duration
func secondsToTime(seconds int) time.Duration {
	return time.Duration(seconds) * time.Second
}
