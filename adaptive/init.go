// Package adaptive implements the Adaptive Parameter Tuning (APT) system for Quiver.
package adaptive

import (
	"log"
	"os"
	"path/filepath"
)

var (
	// DefaultInstance is the default APT integration instance
	DefaultInstance *QuiverIntegration
)

// Initialize initializes the APT system
func Initialize(dataDir string, enabled bool) error {
	// Create config directory
	configDir := filepath.Join(dataDir, "apt")
	if err := os.MkdirAll(configDir, 0755); err != nil {
		return err
	}

	// Load configuration
	config, err := LoadConfigFile(configDir)
	if err != nil {
		log.Printf("Failed to load APT configuration: %v", err)
		// Use default configuration
		config = &ConfigFile{
			Enabled: enabled,
		}
	}

	// Override enabled flag if specified
	if enabled != config.Enabled {
		config.Enabled = enabled
		// Save configuration
		if err := SaveConfigFile(config, configDir); err != nil {
			log.Printf("Failed to save APT configuration: %v", err)
		}
	}

	// Convert to integration config
	integrationConfig := config.ToIntegrationConfig(configDir)

	// Create integration
	DefaultInstance = NewQuiverIntegration(integrationConfig)

	return nil
}

// Shutdown shuts down the APT system
func Shutdown() {
	if DefaultInstance != nil {
		DefaultInstance.Shutdown()
		DefaultInstance = nil
	}
}

// IsEnabled returns whether adaptive tuning is enabled
func IsEnabled() bool {
	if DefaultInstance == nil {
		return false
	}
	return DefaultInstance.IsEnabled()
}

// SetEnabled enables or disables adaptive tuning
func SetEnabled(enabled bool) error {
	if DefaultInstance == nil {
		return nil
	}
	return DefaultInstance.SetEnabled(enabled)
}

// RecordQuery records a query for workload analysis
func RecordQuery(ef, k int, hasFilter bool, results int, duration int64) {
	if DefaultInstance == nil {
		return
	}
	DefaultInstance.RecordQuery(ef, k, hasFilter, results, secondsToTime(int(duration/1000000)))
}

// RecordIndexOperation records an index operation for performance monitoring
func RecordIndexOperation(op string, duration int64) {
	if DefaultInstance == nil {
		return
	}
	DefaultInstance.RecordIndexOperation(op, secondsToTime(int(duration/1000000)))
}

// GetCurrentParameters returns the current HNSW parameters
func GetCurrentParameters() *HNSWParameters {
	if DefaultInstance == nil {
		return &HNSWParameters{
			M:              16,
			EfConstruction: 200,
			EfSearch:       100,
			DelaunayType:   "simple",
		}
	}
	return DefaultInstance.GetCurrentParameters()
}

// GetWorkloadAnalysis returns the current workload analysis
func GetWorkloadAnalysis() *WorkloadAnalysis {
	if DefaultInstance == nil {
		return &WorkloadAnalysis{}
	}
	return DefaultInstance.GetWorkloadAnalysis()
}

// GetPerformanceReport returns the current performance report
func GetPerformanceReport() *PerformanceReport {
	if DefaultInstance == nil {
		return &PerformanceReport{}
	}
	return DefaultInstance.GetPerformanceReport()
}

// GetParameterHistory returns the history of parameter changes
func GetParameterHistory() []ParameterChangeRecord {
	if DefaultInstance == nil {
		return nil
	}
	return DefaultInstance.GetParameterHistory()
}
