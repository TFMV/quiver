// Package adaptive implements the Adaptive Parameter Tuning (APT) system for Quiver.
package adaptive

import (
	"log"
	"sync"
	"time"
)

// QuiverIntegration manages the integration between the APT system and Quiver
type QuiverIntegration struct {
	// APT system
	apt *APTSystem

	// Configuration
	config IntegrationConfig

	// Mutex for thread safety
	mu sync.RWMutex
}

// IntegrationConfig contains configuration for the integration
type IntegrationConfig struct {
	// Whether adaptive tuning is enabled
	Enabled bool

	// Path to the configuration directory
	ConfigDir string

	// Analyzer configuration
	AnalyzerConfig AnalyzerConfig

	// Monitor configuration
	MonitorConfig MonitorConfig

	// Optimizer configuration
	OptimizerConfig OptimizerConfig
}

// DefaultIntegrationConfig returns the default integration configuration
func DefaultIntegrationConfig() IntegrationConfig {
	return IntegrationConfig{
		Enabled: true,
		AnalyzerConfig: AnalyzerConfig{
			HistorySize: 10000,
			ShiftThresholds: WorkloadShiftThresholds{
				MedianKChange:       0.3,
				FilteredRatioChange: 0.2,
				QueryVolumeChange:   0.5,
			},
		},
		MonitorConfig: MonitorConfig{
			BufferSize: 1000,
			Thresholds: PerformanceThresholds{
				MaxSearchLatencyMs: 100.0,
				MaxMemoryBytes:     8 * 1024 * 1024 * 1024, // 8GB
				MaxCPUPercent:      80.0,
			},
		},
		OptimizerConfig: OptimizerConfig{
			MinChangeInterval: 1 * time.Hour,
			MaxM:              64,
			MaxEfConstruction: 500,
			MaxEfSearch:       400,
			MinM:              8,
			MinEfConstruction: 50,
			MinEfSearch:       20,
			EnabledStrategies: []string{
				"LatencyFocused",
				"RecallFocused",
				"WorkloadAdaptive",
				"ResourceEfficient",
				"Balanced",
			},
			Thresholds: PerformanceThresholds{
				MaxSearchLatencyMs: 100.0,
				MaxMemoryBytes:     8 * 1024 * 1024 * 1024, // 8GB
				MaxCPUPercent:      80.0,
			},
		},
	}
}

// NewQuiverIntegration creates a new integration between APT and Quiver
func NewQuiverIntegration(config IntegrationConfig) *QuiverIntegration {
	// Create the integration
	integration := &QuiverIntegration{
		config: config,
	}

	// Create the APT system if enabled
	if config.Enabled {
		integration.initAPT()
	}

	return integration
}

// initAPT initializes the APT system
func (qi *QuiverIntegration) initAPT() {
	// Create APT configuration
	aptConfig := &Config{
		Enabled:   qi.config.Enabled,
		Analyzer:  qi.config.AnalyzerConfig,
		Monitor:   qi.config.MonitorConfig,
		Optimizer: qi.config.OptimizerConfig,
		ConfigManager: ManagerConfig{
			ConfigDir:       qi.config.ConfigDir,
			PersistInterval: 10 * time.Minute,
			MaxBackups:      5,
			AutoApply:       true,
			Thresholds: ConfigThresholds{
				MaxSearchLatencyMs: qi.config.MonitorConfig.Thresholds.MaxSearchLatencyMs,
				MaxMemoryBytes:     qi.config.MonitorConfig.Thresholds.MaxMemoryBytes,
				MaxCPUPercent:      qi.config.MonitorConfig.Thresholds.MaxCPUPercent,
			},
		},
	}

	// Create the APT system
	qi.apt = NewAPTSystem(aptConfig)

	// Start the APT system
	if err := qi.apt.Start(); err != nil {
		log.Printf("Failed to start APT system: %v", err)
	}
}

// IsEnabled returns whether adaptive tuning is enabled
func (qi *QuiverIntegration) IsEnabled() bool {
	qi.mu.RLock()
	defer qi.mu.RUnlock()

	return qi.config.Enabled && qi.apt != nil
}

// SetEnabled enables or disables adaptive tuning
func (qi *QuiverIntegration) SetEnabled(enabled bool) error {
	qi.mu.Lock()
	defer qi.mu.Unlock()

	// Update configuration
	qi.config.Enabled = enabled

	// If APT is already in the desired state, do nothing
	if (qi.apt != nil) == enabled {
		return nil
	}

	// If enabling, initialize APT
	if enabled {
		qi.initAPT()
		return nil
	}

	// If disabling, stop APT
	if qi.apt != nil {
		qi.apt.Stop()
		qi.apt = nil
	}

	return nil
}

// RecordQuery records a query for workload analysis
func (qi *QuiverIntegration) RecordQuery(ef, k int, hasFilter bool, results int, duration time.Duration) {
	if !qi.IsEnabled() {
		return
	}

	// Create search params
	params := &SearchParams{
		Ef:     ef,
		K:      k,
		Filter: nil,
	}

	// Set filter if needed
	if hasFilter {
		params.Filter = struct{}{}
	}

	// Record the query
	qi.apt.RecordQuery(params, results, duration)
}

// RecordIndexOperation records an index operation for performance monitoring
func (qi *QuiverIntegration) RecordIndexOperation(op string, duration time.Duration) {
	if !qi.IsEnabled() {
		return
	}

	qi.apt.RecordIndexOperation(op, duration)
}

// GetCurrentParameters returns the current HNSW parameters
func (qi *QuiverIntegration) GetCurrentParameters() *HNSWParameters {
	if !qi.IsEnabled() {
		// Return default parameters if APT is disabled
		return &HNSWParameters{
			M:              16,
			EfConstruction: 200,
			EfSearch:       100,
			DelaunayType:   "simple",
		}
	}

	return qi.apt.GetCurrentParameters()
}

// GetWorkloadAnalysis returns the current workload analysis
func (qi *QuiverIntegration) GetWorkloadAnalysis() *WorkloadAnalysis {
	if !qi.IsEnabled() {
		return &WorkloadAnalysis{}
	}

	return qi.apt.GetWorkloadAnalysis()
}

// GetPerformanceReport returns the current performance report
func (qi *QuiverIntegration) GetPerformanceReport() *PerformanceReport {
	if !qi.IsEnabled() {
		return &PerformanceReport{}
	}

	return qi.apt.GetPerformanceReport()
}

// GetParameterHistory returns the history of parameter changes
func (qi *QuiverIntegration) GetParameterHistory() []ParameterChangeRecord {
	if !qi.IsEnabled() {
		return nil
	}

	return qi.apt.GetParameterHistory()
}

// Shutdown stops the APT system
func (qi *QuiverIntegration) Shutdown() {
	qi.mu.Lock()
	defer qi.mu.Unlock()

	if qi.apt != nil {
		qi.apt.Stop()
		qi.apt = nil
	}
}
