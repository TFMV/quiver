// File: quiver/adaptive/apt.go
package adaptive

import (
	"context"
	"log"
	"runtime"
	"sync"
	"time"
)

// SearchParams represents search parameters for HNSW
type SearchParams struct {
	Ef     int         // Size of the dynamic list for the nearest neighbors
	K      int         // Number of nearest neighbors to return
	Filter interface{} // Optional filter to apply during search
}

// APTSystem manages the adaptive parameter tuning for Quiver
type APTSystem struct {
	// Core components
	analyzer  *WorkloadAnalyzer
	monitor   *PerformanceMonitor
	optimizer *ParameterOptimizer
	configMgr *ConfigurationManager

	// Control
	enabled bool
	ctx     context.Context
	cancel  context.CancelFunc
	wg      sync.WaitGroup

	// Telemetry
	logger *log.Logger

	// Mutex for thread safety
	mu sync.RWMutex
}

// Config holds the configuration for the APT system
type Config struct {
	// Whether adaptive tuning is enabled
	Enabled bool

	// Analyzer configuration
	Analyzer AnalyzerConfig

	// Monitor configuration
	Monitor MonitorConfig

	// Optimizer configuration
	Optimizer OptimizerConfig

	// Configuration manager configuration
	ConfigManager ManagerConfig
}

// NewAPTSystem creates a new adaptive parameter tuning system
func NewAPTSystem(cfg *Config) *APTSystem {
	ctx, cancel := context.WithCancel(context.Background())

	apt := &APTSystem{
		enabled: cfg.Enabled,
		ctx:     ctx,
		cancel:  cancel,
		logger:  log.New(log.Writer(), "[APT] ", log.LstdFlags),
	}

	// Initialize components
	apt.analyzer = NewWorkloadAnalyzer(cfg.Analyzer)
	apt.monitor = NewPerformanceMonitor(cfg.Monitor)
	apt.optimizer = NewParameterOptimizer(cfg.Optimizer)
	apt.configMgr = NewConfigurationManager(cfg.ConfigManager)

	return apt
}

// Start begins the adaptive parameter tuning process
func (apt *APTSystem) Start() error {
	apt.mu.Lock()
	defer apt.mu.Unlock()

	if !apt.enabled {
		apt.logger.Println("APT system is disabled, not starting")
		return nil
	}

	// Set initial parameters for the default index
	defaultParams := apt.configMgr.GetParameters("default")
	apt.optimizer.SetCurrentParameters(defaultParams)

	// Start background monitoring
	apt.wg.Add(1)
	go apt.runBackgroundMonitoring()

	apt.logger.Println("APT system started")
	return nil
}

// Stop halts the adaptive parameter tuning process
func (apt *APTSystem) Stop() {
	apt.mu.Lock()
	defer apt.mu.Unlock()

	if !apt.enabled {
		return
	}

	apt.cancel()
	apt.wg.Wait()
	apt.logger.Println("APT system stopped")
}

// RecordQuery records a query for workload analysis
func (apt *APTSystem) RecordQuery(query *SearchParams, results int, duration time.Duration) {
	if !apt.enabled {
		return
	}

	apt.analyzer.RecordQuery(&QueryRecord{
		EfSearch:    query.Ef,
		K:           query.K,
		Filter:      query.Filter != nil,
		Timestamp:   time.Now(),
		Duration:    duration,
		ResultCount: results,
	})

	apt.monitor.RecordSearchMetrics(duration, query.Ef, results)
}

// RecordIndexOperation records an index operation for performance monitoring
func (apt *APTSystem) RecordIndexOperation(op string, duration time.Duration) {
	if !apt.enabled {
		return
	}

	apt.monitor.RecordIndexMetrics(op, duration)
}

// GetCurrentParameters returns the current HNSW parameters
func (apt *APTSystem) GetCurrentParameters() *HNSWParameters {
	apt.mu.RLock()
	defer apt.mu.RUnlock()

	return apt.optimizer.GetCurrentParameters()
}

// GetParameterHistory returns the history of parameter changes
func (apt *APTSystem) GetParameterHistory() []ParameterChangeRecord {
	apt.mu.RLock()
	defer apt.mu.RUnlock()

	return apt.optimizer.GetParameterHistory()
}

// GetWorkloadAnalysis returns the current workload analysis
func (apt *APTSystem) GetWorkloadAnalysis() *WorkloadAnalysis {
	apt.mu.RLock()
	defer apt.mu.RUnlock()

	return apt.analyzer.GetAnalysis()
}

// GetPerformanceReport returns the current performance report
func (apt *APTSystem) GetPerformanceReport() *PerformanceReport {
	apt.mu.RLock()
	defer apt.mu.RUnlock()

	return apt.monitor.GetReport()
}

// SetEnabled enables or disables adaptive parameter tuning
func (apt *APTSystem) SetEnabled(enabled bool) error {
	apt.mu.Lock()
	defer apt.mu.Unlock()

	if apt.enabled == enabled {
		return nil
	}

	apt.enabled = enabled

	if enabled {
		// Start the system if it's being enabled
		ctx, cancel := context.WithCancel(context.Background())
		apt.ctx = ctx
		apt.cancel = cancel

		apt.wg.Add(1)
		go apt.runBackgroundMonitoring()

		apt.logger.Println("APT system enabled")
	} else {
		// Stop the system if it's being disabled
		apt.cancel()
		apt.wg.Wait()
		apt.logger.Println("APT system disabled")
	}

	// Update configuration
	apt.configMgr.SetParameters("default", apt.optimizer.GetCurrentParameters())
	return apt.persistConfigurations()
}

// persistConfigurations persists the current configurations to disk
func (apt *APTSystem) persistConfigurations() error {
	return apt.configMgr.persistConfigurations()
}

// runBackgroundMonitoring is the main background monitoring routine
func (apt *APTSystem) runBackgroundMonitoring() {
	defer apt.wg.Done()

	metricsTicker := time.NewTicker(10 * time.Second)
	analysisTicker := time.NewTicker(5 * time.Minute)

	defer metricsTicker.Stop()
	defer analysisTicker.Stop()

	for {
		select {
		case <-apt.ctx.Done():
			return

		case <-metricsTicker.C:
			apt.updateSystemMetrics()

		case <-analysisTicker.C:
			apt.runFullAnalysis()
		}
	}
}

// updateSystemMetrics collects and records system metrics
func (apt *APTSystem) updateSystemMetrics() {
	// Get memory stats
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)

	// Record memory usage
	apt.monitor.RecordMemoryUsage(memStats.Alloc)

	// Get and record CPU usage
	cpuUsage := getCPUUsage()
	apt.monitor.RecordCPUUsage(cpuUsage)
}

// runFullAnalysis performs a complete workload analysis and parameter optimization
func (apt *APTSystem) runFullAnalysis() {
	apt.mu.Lock()
	defer apt.mu.Unlock()

	// Analyze workload
	workloadAnalysis := apt.analyzer.AnalyzeWorkload()

	// Get performance report
	performanceReport := apt.monitor.GenerateReport()

	// Check for workload shifts
	shifts := apt.analyzer.DetectWorkloadShifts(workloadAnalysis)

	// If we detected shifts or performance issues, optimize parameters
	if len(shifts) > 0 || performanceReport.HasIssues() {
		apt.logger.Printf("Detected %d workload shifts and %d performance issues, optimizing parameters",
			len(shifts), len(performanceReport.Issues))

		// Optimize parameters
		newParams, strategy := apt.optimizer.OptimizeParameters(workloadAnalysis, performanceReport)

		// If we have new parameters, apply them
		if newParams != nil {
			apt.applyNewParameters(newParams, strategy, workloadAnalysis, performanceReport)
		}
	}
}

// applyNewParameters applies new parameters and records the change
func (apt *APTSystem) applyNewParameters(params *HNSWParameters, strategy string,
	workload *WorkloadAnalysis, performance *PerformanceReport) {

	// Record the change
	apt.optimizer.RecordParameterChange(params, strategy, workload, performance)

	// Update configuration
	apt.configMgr.SetParameters("default", params)
	if err := apt.persistConfigurations(); err != nil {
		apt.logger.Printf("Failed to save configuration: %v", err)
	}

	// Log the change
	apt.logger.Printf("Updated parameters using %s strategy: M=%d, EfConstruction=%d, EfSearch=%d",
		strategy, params.M, params.EfConstruction, params.EfSearch)

	// Schedule evaluation of the change
	apt.wg.Add(1)
	go apt.evaluateParameterChange(params, strategy)
}

// evaluateParameterChange evaluates the impact of a parameter change
func (apt *APTSystem) evaluateParameterChange(params *HNSWParameters, strategy string) {
	defer apt.wg.Done()

	// Wait for enough data to evaluate
	select {
	case <-apt.ctx.Done():
		return
	case <-time.After(10 * time.Minute):
		// Continue with evaluation
	}

	apt.mu.Lock()
	defer apt.mu.Unlock()

	// Get performance after the change
	afterPerf := apt.monitor.GenerateReport()

	// Calculate improvement
	beforePerf := apt.optimizer.GetLastChangePerformance()
	improvement := calculateImprovement(beforePerf, afterPerf)

	// Record the evaluation
	apt.optimizer.RecordParameterEvaluation(improvement)

	// Log the evaluation
	apt.logger.Printf("Parameter change evaluation: strategy=%s, improvement=%.2f%%",
		strategy, improvement*100)

	// If change was negative, consider reverting
	if improvement < -0.1 {
		apt.logger.Printf("Parameter change had negative impact (%.2f%%), considering reverting",
			improvement*100)

		// Get previous parameters
		prevParams := apt.optimizer.GetPreviousParameters()
		if prevParams != nil {
			// Revert to previous parameters
			apt.applyNewParameters(prevParams, "revert",
				apt.analyzer.GetAnalysis(), afterPerf)
		}
	}
}

// getCPUUsage returns the current CPU usage percentage
func getCPUUsage() float64 {
	// This is a simplified implementation
	// In a real implementation, you would use OS-specific APIs to get CPU usage
	return 0.0
}

// calculateImprovement calculates the improvement between two performance reports
func calculateImprovement(before, after *PerformanceReport) float64 {
	if before == nil || after == nil {
		return 0.0
	}

	// Calculate improvement based on search latency
	if before.SearchLatency.P95 > 0 {
		latencyImprovement := (before.SearchLatency.P95 - after.SearchLatency.P95) / before.SearchLatency.P95
		return latencyImprovement
	}

	return 0.0
}
