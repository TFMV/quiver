// File: quiver/adaptive/optimizer.go
package adaptive

import (
	"math"
	"sync"
	"time"
)

// ParameterOptimizer adjusts HNSW parameters based on workload and performance
type ParameterOptimizer struct {
	// Configuration
	config OptimizerConfig

	// Current parameters
	currentParams *HNSWParameters

	// History of parameter changes
	paramHistory []ParameterChangeRecord

	// Optimization strategies
	strategies []OptimizationStrategy

	// Mutex for thread safety
	mu sync.RWMutex
}

// OptimizerConfig contains configuration for the parameter optimizer
type OptimizerConfig struct {
	// Minimum time between parameter changes
	MinChangeInterval time.Duration

	// Maximum parameter values
	MaxM              int
	MaxEfConstruction int
	MaxEfSearch       int

	// Minimum parameter values
	MinM              int
	MinEfConstruction int
	MinEfSearch       int

	// Enabled strategies
	EnabledStrategies []string

	// Performance thresholds
	Thresholds PerformanceThresholds
}

// HNSWParameters contains the tunable parameters for HNSW
type HNSWParameters struct {
	M              int    // Number of connections per element
	EfConstruction int    // Size of dynamic candidate list during construction
	EfSearch       int    // Size of dynamic candidate list during search
	DelaunayType   string // Type of graph construction algorithm
}

// ParameterChangeRecord tracks a change to the parameters
type ParameterChangeRecord struct {
	// When the change was made
	Timestamp time.Time

	// Parameters before and after
	OldParameters *HNSWParameters
	NewParameters *HNSWParameters

	// Strategy that suggested the change
	Strategy string

	// State at the time of change
	WorkloadState    *WorkloadAnalysis
	PerformanceState *PerformanceReport

	// Evaluation of the change
	Improvement float64
	EvaluatedAt time.Time
}

// OptimizationStrategy defines an approach to parameter optimization
type OptimizationStrategy interface {
	// Name of the strategy
	Name() string

	// Whether the strategy is applicable to the current state
	IsApplicable(workload *WorkloadAnalysis, performance *PerformanceReport) bool

	// Suggest parameter changes
	SuggestParameters(current *HNSWParameters, workload *WorkloadAnalysis,
		performance *PerformanceReport, config *OptimizerConfig) *HNSWParameters
}

// NewParameterOptimizer creates a new parameter optimizer
func NewParameterOptimizer(config OptimizerConfig) *ParameterOptimizer {
	// Use default config if needed
	if config.MinChangeInterval == 0 {
		config.MinChangeInterval = 1 * time.Hour
	}

	if config.MaxM == 0 {
		config.MaxM = 64
	}

	if config.MaxEfConstruction == 0 {
		config.MaxEfConstruction = 500
	}

	if config.MaxEfSearch == 0 {
		config.MaxEfSearch = 400
	}

	if config.MinM == 0 {
		config.MinM = 8
	}

	if config.MinEfConstruction == 0 {
		config.MinEfConstruction = 50
	}

	if config.MinEfSearch == 0 {
		config.MinEfSearch = 20
	}

	// Set default thresholds if not provided
	if config.Thresholds.MaxSearchLatencyMs == 0 {
		config.Thresholds.MaxSearchLatencyMs = 100.0 // 100ms
	}

	if config.Thresholds.MaxMemoryBytes == 0 {
		config.Thresholds.MaxMemoryBytes = 8 * 1024 * 1024 * 1024 // 8GB
	}

	if config.Thresholds.MaxCPUPercent == 0 {
		config.Thresholds.MaxCPUPercent = 80.0 // 80%
	}

	// Create optimizer
	po := &ParameterOptimizer{
		config: config,
		currentParams: &HNSWParameters{
			M:              16,
			EfConstruction: 200,
			EfSearch:       100,
			DelaunayType:   "simple",
		},
	}

	// Initialize strategies
	po.initStrategies()

	return po
}

// initStrategies initializes the optimization strategies
func (po *ParameterOptimizer) initStrategies() {
	// Create all available strategies
	allStrategies := []OptimizationStrategy{
		&LatencyFocusedStrategy{},
		&RecallFocusedStrategy{},
		&WorkloadAdaptiveStrategy{},
		&ResourceEfficientStrategy{},
		&BalancedStrategy{},
	}

	// Filter strategies based on configuration
	if len(po.config.EnabledStrategies) == 0 {
		// If none specified, use all
		po.strategies = allStrategies
	} else {
		// Use only enabled strategies
		for _, s := range allStrategies {
			for _, enabled := range po.config.EnabledStrategies {
				if s.Name() == enabled {
					po.strategies = append(po.strategies, s)
					break
				}
			}
		}
	}
}

// SetCurrentParameters sets the current HNSW parameters
func (po *ParameterOptimizer) SetCurrentParameters(params *HNSWParameters) {
	po.mu.Lock()
	defer po.mu.Unlock()

	po.currentParams = params
}

// GetCurrentParameters returns the current HNSW parameters
func (po *ParameterOptimizer) GetCurrentParameters() *HNSWParameters {
	po.mu.RLock()
	defer po.mu.RUnlock()

	return po.currentParams
}

// GetParameterHistory returns the history of parameter changes
func (po *ParameterOptimizer) GetParameterHistory() []ParameterChangeRecord {
	po.mu.RLock()
	defer po.mu.RUnlock()

	return po.paramHistory
}

// GetPreviousParameters returns the parameters before the most recent change
func (po *ParameterOptimizer) GetPreviousParameters() *HNSWParameters {
	po.mu.RLock()
	defer po.mu.RUnlock()

	if len(po.paramHistory) == 0 {
		return nil
	}

	return po.paramHistory[len(po.paramHistory)-1].OldParameters
}

// GetLastChangePerformance returns the performance state at the time of the last change
func (po *ParameterOptimizer) GetLastChangePerformance() *PerformanceReport {
	po.mu.RLock()
	defer po.mu.RUnlock()

	if len(po.paramHistory) == 0 {
		return nil
	}

	return po.paramHistory[len(po.paramHistory)-1].PerformanceState
}

// OptimizeParameters suggests parameter changes based on workload and performance
func (po *ParameterOptimizer) OptimizeParameters(workload *WorkloadAnalysis,
	performance *PerformanceReport) (*HNSWParameters, string) {

	po.mu.Lock()
	defer po.mu.Unlock()

	// Check if we've made a change recently
	if len(po.paramHistory) > 0 {
		lastChange := po.paramHistory[len(po.paramHistory)-1].Timestamp
		if time.Since(lastChange) < po.config.MinChangeInterval {
			// Too soon for another change
			return nil, ""
		}
	}

	// Make sure strategies are initialized
	if len(po.strategies) == 0 {
		po.initStrategies()
	}

	// Find applicable strategies
	var applicableStrategies []OptimizationStrategy
	for _, strategy := range po.strategies {
		if strategy.IsApplicable(workload, performance) {
			applicableStrategies = append(applicableStrategies, strategy)
		}
	}

	// If no applicable strategies, use balanced strategy as fallback
	if len(applicableStrategies) == 0 {
		applicableStrategies = append(applicableStrategies, &BalancedStrategy{})
	}

	// Try each strategy and pick the best suggestion
	var bestParams *HNSWParameters
	var bestStrategy string
	var bestScore float64

	for _, strategy := range applicableStrategies {
		suggestedParams := strategy.SuggestParameters(po.currentParams, workload, performance, &po.config)
		if suggestedParams == nil {
			continue // Skip if strategy didn't suggest parameters
		}

		score := po.evaluateParameters(suggestedParams, workload, performance)

		if score > bestScore {
			bestScore = score
			bestParams = suggestedParams
			bestStrategy = strategy.Name()
		}
	}

	// If we have a better configuration, return it
	if bestParams != nil && bestScore > po.evaluateParameters(po.currentParams, workload, performance) {
		return bestParams, bestStrategy
	}

	// No better configuration found
	return nil, ""
}

// RecordParameterChange records a change to the parameters
func (po *ParameterOptimizer) RecordParameterChange(params *HNSWParameters, strategy string,
	workload *WorkloadAnalysis, performance *PerformanceReport) {

	po.mu.Lock()
	defer po.mu.Unlock()

	// Record the change
	changeRecord := ParameterChangeRecord{
		Timestamp:        time.Now(),
		OldParameters:    po.currentParams,
		NewParameters:    params,
		Strategy:         strategy,
		WorkloadState:    workload,
		PerformanceState: performance,
	}

	// Update current parameters
	po.currentParams = params

	// Add to history
	po.paramHistory = append(po.paramHistory, changeRecord)

	// Limit history size
	if len(po.paramHistory) > 100 {
		po.paramHistory = po.paramHistory[len(po.paramHistory)-100:]
	}
}

// RecordParameterEvaluation records the evaluation of a parameter change
func (po *ParameterOptimizer) RecordParameterEvaluation(improvement float64) {
	po.mu.Lock()
	defer po.mu.Unlock()

	if len(po.paramHistory) == 0 {
		return
	}

	// Update the most recent change record
	po.paramHistory[len(po.paramHistory)-1].Improvement = improvement
	po.paramHistory[len(po.paramHistory)-1].EvaluatedAt = time.Now()
}

// evaluateParameters evaluates a set of parameters against the current workload and performance
func (po *ParameterOptimizer) evaluateParameters(params *HNSWParameters, workload *WorkloadAnalysis,
	performance *PerformanceReport) float64 {

	if params == nil || workload == nil || performance == nil {
		return 0.0
	}

	// Start with a base score
	score := 1.0

	// ---- Parameter evaluation factors ----

	// 1. M parameter evaluation
	// Higher M improves recall but increases memory usage and construction time
	mRatio := float64(params.M) / float64(po.config.MaxM)

	// 2. EfConstruction parameter evaluation
	// Higher EfConstruction improves index quality but increases construction time
	efConstructionRatio := float64(params.EfConstruction) / float64(po.config.MaxEfConstruction)

	// 3. EfSearch parameter evaluation
	// Higher EfSearch improves recall but increases search latency
	efSearchRatio := float64(params.EfSearch) / float64(po.config.MaxEfSearch)

	// ---- Performance issue detection ----

	// Check for specific performance issues
	hasHighLatency := false
	hasHighMemory := false
	hasHighCPU := false

	// Check issues array
	for _, issue := range performance.Issues {
		if issue == "high_search_latency" {
			hasHighLatency = true
		} else if issue == "high_memory_usage" {
			hasHighMemory = true
		} else if issue == "high_cpu_usage" {
			hasHighCPU = true
		}
	}

	// Also check raw metrics against thresholds
	if performance.SearchLatency.P95 > po.config.Thresholds.MaxSearchLatencyMs {
		hasHighLatency = true
	}
	if performance.MemoryUsage > po.config.Thresholds.MaxMemoryBytes {
		hasHighMemory = true
	}
	if performance.CPUUsage > po.config.Thresholds.MaxCPUPercent {
		hasHighCPU = true
	}

	// ---- Workload characteristics ----

	// Calculate workload factors
	highKQueries := workload.MedianK > 20
	veryHighKQueries := workload.MedianK > 100
	highFilterUsage := workload.FilteredQueryRatio > 0.5
	highQueryVolume := workload.QueriesPerSec > 50

	// Normalize K factor (0.0-1.0)
	kFactor := float64(workload.MedianK) / 200.0 // Normalize to 1.0 at K=200
	if kFactor > 1.0 {
		kFactor = 1.0
	}

	// ---- Score adjustments based on performance issues ----

	// Base parameter scores
	mScore := 0.5 + 0.5*mRatio // Balance between low and high M
	efConstructionScore := 0.5 + 0.5*efConstructionRatio
	efSearchScore := 0.5 + 0.5*efSearchRatio

	// Adjust scores based on performance issues
	if hasHighLatency {
		// For high latency, strongly prefer lower EfSearch
		inverseEfSearchRatio := 1.0 - efSearchRatio
		efSearchScore = 0.3 + 0.7*inverseEfSearchRatio

		// If latency is very high, also slightly prefer lower M
		if performance.SearchLatency.P95 > 1.5*po.config.Thresholds.MaxSearchLatencyMs {
			inverseMRatio := 1.0 - mRatio
			mScore = 0.4 + 0.6*inverseMRatio
		}
	}

	if hasHighMemory {
		// For high memory usage, strongly prefer lower M
		inverseMRatio := 1.0 - mRatio
		mScore = 0.2 + 0.8*inverseMRatio
	}

	if hasHighCPU {
		// For high CPU usage, prefer lower EfConstruction and slightly lower EfSearch
		inverseEfConstructionRatio := 1.0 - efConstructionRatio
		efConstructionScore = 0.3 + 0.7*inverseEfConstructionRatio

		inverseEfSearchRatio := 1.0 - efSearchRatio
		efSearchScore = 0.4 + 0.6*inverseEfSearchRatio
	}

	// ---- Score adjustments based on workload characteristics ----

	// Adjust for high K queries
	if highKQueries && !hasHighLatency {
		// For high K queries without latency issues, prefer higher EfSearch
		efSearchScore = 0.3 + 0.7*efSearchRatio
	} else if veryHighKQueries && hasHighLatency {
		// For very high K with latency issues, find a balance
		// Still reduce EfSearch but not as aggressively
		efSearchScore = 0.5 // Neutral score
	}

	// Adjust for filtered queries
	if highFilterUsage {
		// Filtered queries benefit from higher M
		mScore = 0.3 + 0.7*mRatio
	}

	// Adjust for high query volume
	if highQueryVolume && !hasHighLatency {
		// For high query volume without latency issues, slightly prefer lower EfSearch
		// to improve throughput
		inverseEfSearchRatio := 1.0 - efSearchRatio
		efSearchScore = 0.4 + 0.6*inverseEfSearchRatio
	}

	// ---- Combine scores ----

	// Apply weights to different parameters based on their importance
	// These weights could be adjusted based on specific use cases
	const (
		mWeight              = 0.3
		efConstructionWeight = 0.2
		efSearchWeight       = 0.5
	)

	// Combine weighted scores
	weightedScore := (mWeight * mScore) +
		(efConstructionWeight * efConstructionScore) +
		(efSearchWeight * efSearchScore)

	// Scale back to original range
	score *= weightedScore

	// ---- Special case handling ----

	// Penalize parameters outside of reasonable ranges
	if params.EfSearch < workload.MedianK {
		// EfSearch should generally be at least as large as K
		score *= 0.8
	}

	// Bonus for balanced parameters
	if !hasHighLatency && !hasHighMemory && !hasHighCPU {
		// If no performance issues, slightly prefer balanced parameters
		balancedMRatio := 1.0 - math.Abs(mRatio-0.5)*2
		balancedEfSearchRatio := 1.0 - math.Abs(efSearchRatio-0.5)*2
		balancedBonus := 0.1 * (balancedMRatio + balancedEfSearchRatio) / 2
		score *= (1.0 + balancedBonus)
	}

	return score
}

// Strategy implementations

// LatencyFocusedStrategy optimizes for search latency
type LatencyFocusedStrategy struct{}

func (s *LatencyFocusedStrategy) Name() string {
	return "LatencyFocused"
}

func (s *LatencyFocusedStrategy) IsApplicable(workload *WorkloadAnalysis, performance *PerformanceReport) bool {
	// Applicable if we have high search latency
	if performance == nil {
		return false
	}

	// Check if the performance report has high search latency issues
	for _, issue := range performance.Issues {
		if issue == "high_search_latency" {
			return true
		}
	}

	// Or if the P95 latency is high
	return performance.SearchLatency.P95 > 50.0 // 50ms
}

func (s *LatencyFocusedStrategy) SuggestParameters(current *HNSWParameters, workload *WorkloadAnalysis,
	performance *PerformanceReport, config *OptimizerConfig) *HNSWParameters {

	if current == nil {
		return nil
	}

	// Clone current parameters
	params := *current

	// For high latency, we need to make more aggressive changes
	// Reduce EfSearch significantly to improve latency
	if performance.SearchLatency.P95 > 100.0 {
		// More aggressive reduction for very high latency
		params.EfSearch = max(params.EfSearch/2, config.MinEfSearch)
	} else if performance.SearchLatency.P95 > 50.0 {
		// Moderate reduction for moderately high latency
		params.EfSearch = max(params.EfSearch-20, config.MinEfSearch)
	}

	// If we have very high latency, also reduce M
	if performance.SearchLatency.P95 > 150.0 && params.M > config.MinM {
		params.M = max(params.M-4, config.MinM)
	}

	// Make sure we're not returning the same parameters
	if params.EfSearch == current.EfSearch && params.M == current.M {
		// Force a change to EfSearch
		params.EfSearch = max(params.EfSearch-10, config.MinEfSearch)
	}

	return &params
}

// RecallFocusedStrategy optimizes for search recall
type RecallFocusedStrategy struct{}

func (s *RecallFocusedStrategy) Name() string {
	return "RecallFocused"
}

func (s *RecallFocusedStrategy) IsApplicable(workload *WorkloadAnalysis, performance *PerformanceReport) bool {
	// Always applicable, but especially if we have low latency
	return performance.SearchLatency.P95 < 20.0 // 20ms
}

func (s *RecallFocusedStrategy) SuggestParameters(current *HNSWParameters, workload *WorkloadAnalysis,
	performance *PerformanceReport, config *OptimizerConfig) *HNSWParameters {

	// Clone current parameters
	params := *current

	// If we have good latency, increase EfSearch to improve recall
	if performance.SearchLatency.P95 < 20.0 {
		// Increase EfSearch to improve recall
		params.EfSearch = min(params.EfSearch+20, config.MaxEfSearch)
	}

	// If we're seeing many high-K queries, increase EfSearch proportionally
	if workload.MedianK > 50 {
		// EfSearch should be at least 2x the median K value
		params.EfSearch = max(params.EfSearch, min(workload.MedianK*2, config.MaxEfSearch))
	}

	// If we have memory available, consider increasing M
	if performance.MemoryUsage < config.Thresholds.MaxMemoryBytes*80/100 {
		params.M = min(params.M+2, config.MaxM)
	}

	return &params
}

// WorkloadAdaptiveStrategy adjusts based on query patterns
type WorkloadAdaptiveStrategy struct{}

func (s *WorkloadAdaptiveStrategy) Name() string {
	return "WorkloadAdaptive"
}

func (s *WorkloadAdaptiveStrategy) IsApplicable(workload *WorkloadAnalysis, performance *PerformanceReport) bool {
	// Applicable if we have a significant number of queries
	return workload.QueryCount > 1000
}

func (s *WorkloadAdaptiveStrategy) SuggestParameters(current *HNSWParameters, workload *WorkloadAnalysis,
	performance *PerformanceReport, config *OptimizerConfig) *HNSWParameters {

	// Clone current parameters
	params := *current

	// If we're seeing many filtered queries, adjust for that workload
	if workload.FilteredQueryRatio > 0.7 {
		// For filtered queries, higher M often helps
		params.M = min(params.M+4, config.MaxM)
	}

	// If query volume is high, optimize for throughput
	if workload.QueriesPerSec > 100 {
		// Reduce EfSearch slightly to handle higher throughput
		params.EfSearch = max(params.EfSearch-5, config.MinEfSearch)
	}

	// If we're seeing temporal patterns (e.g., batch workloads)
	if workload.HasTemporalPatterns {
		// Adjust based on time of day or workload pattern
		hour := time.Now().Hour()
		if hour >= 9 && hour <= 17 { // Business hours
			// Optimize for latency during business hours
			params.EfSearch = max(params.EfSearch-10, config.MinEfSearch)
		} else {
			// Optimize for recall during off-hours
			params.EfSearch = min(params.EfSearch+20, config.MaxEfSearch)
		}
	}

	return &params
}

// ResourceEfficientStrategy optimizes for resource usage
type ResourceEfficientStrategy struct{}

func (s *ResourceEfficientStrategy) Name() string {
	return "ResourceEfficient"
}

func (s *ResourceEfficientStrategy) IsApplicable(workload *WorkloadAnalysis, performance *PerformanceReport) bool {
	// Applicable if we have high memory or CPU usage
	// 0.8GB in bytes
	memoryThreshold := uint64(858993459) // ~0.8GB
	return performance.MemoryUsage > memoryThreshold || performance.CPUUsage > 70.0
}

func (s *ResourceEfficientStrategy) SuggestParameters(current *HNSWParameters, workload *WorkloadAnalysis,
	performance *PerformanceReport, config *OptimizerConfig) *HNSWParameters {

	// Clone current parameters
	params := *current

	// If memory usage is high, reduce M
	// 0.8GB in bytes
	memoryThreshold := uint64(858993459) // ~0.8GB
	if performance.MemoryUsage > memoryThreshold && params.M > config.MinM {
		params.M = max(params.M-2, config.MinM)
	}

	// If CPU usage is high, reduce EfSearch
	if performance.CPUUsage > 70.0 {
		params.EfSearch = max(params.EfSearch-10, config.MinEfSearch)
	}

	return &params
}

// BalancedStrategy tries to balance recall, latency, and resource usage
type BalancedStrategy struct{}

func (s *BalancedStrategy) Name() string {
	return "Balanced"
}

func (s *BalancedStrategy) IsApplicable(workload *WorkloadAnalysis, performance *PerformanceReport) bool {
	// Always applicable
	return true
}

func (s *BalancedStrategy) SuggestParameters(current *HNSWParameters, workload *WorkloadAnalysis,
	performance *PerformanceReport, config *OptimizerConfig) *HNSWParameters {

	// Clone current parameters
	params := *current

	// Calculate target EfSearch based on median K
	targetEfSearch := workload.MedianK * 2
	if targetEfSearch < config.MinEfSearch {
		targetEfSearch = config.MinEfSearch
	}
	if targetEfSearch > config.MaxEfSearch {
		targetEfSearch = config.MaxEfSearch
	}

	// Move EfSearch towards target
	if params.EfSearch < targetEfSearch {
		params.EfSearch = min(params.EfSearch+10, targetEfSearch)
	} else if params.EfSearch > targetEfSearch {
		params.EfSearch = max(params.EfSearch-10, targetEfSearch)
	}

	// Adjust M based on memory usage
	memoryRatio := float64(performance.MemoryUsage) / float64(config.Thresholds.MaxMemoryBytes)
	if memoryRatio < 0.5 && params.M < config.MaxM {
		// We have memory to spare, increase M
		params.M = min(params.M+2, config.MaxM)
	} else if memoryRatio > 0.8 && params.M > config.MinM {
		// Memory usage is high, decrease M
		params.M = max(params.M-2, config.MinM)
	}

	return &params
}

// Helper functions
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
