// Package adaptive implements the Adaptive Parameter Tuning (APT) system for Quiver.
package adaptive

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// ConfigurationManager handles parameter persistence and application
type ConfigurationManager struct {
	// Configuration
	config ManagerConfig

	// Current parameters
	currentParams map[string]*HNSWParameters

	// History of parameter changes
	changeHistory []ConfigChangeRecord

	// Mutex for thread safety
	mu sync.RWMutex
}

// ManagerConfig contains configuration for the configuration manager
type ManagerConfig struct {
	// Directory to store parameter configurations
	ConfigDir string

	// How often to persist parameters to disk
	PersistInterval time.Duration

	// Maximum number of configuration backups to keep
	MaxBackups int

	// Whether to automatically apply parameter changes
	AutoApply bool

	// Thresholds for performance monitoring
	Thresholds ConfigThresholds
}

// ConfigThresholds defines limits for acceptable performance
type ConfigThresholds struct {
	// Maximum acceptable search latency (ms)
	MaxSearchLatencyMs float64

	// Maximum acceptable memory usage (bytes)
	MaxMemoryBytes uint64

	// Maximum acceptable CPU usage (percent)
	MaxCPUPercent float64
}

// ConfigChangeRecord tracks a change to the configuration
type ConfigChangeRecord struct {
	// When the change was made
	Timestamp time.Time

	// Index affected
	IndexName string

	// Parameters before and after
	OldParameters *HNSWParameters
	NewParameters *HNSWParameters

	// Whether the change was applied
	Applied bool

	// When the change was applied
	AppliedAt time.Time
}

// NewConfigurationManager creates a new configuration manager
func NewConfigurationManager(config ManagerConfig) *ConfigurationManager {
	// Use default config if needed
	if config.ConfigDir == "" {
		homeDir, err := os.UserHomeDir()
		if err == nil {
			config.ConfigDir = filepath.Join(homeDir, ".quiver", "config")
		} else {
			config.ConfigDir = "config"
		}
	}

	if config.PersistInterval == 0 {
		config.PersistInterval = 10 * time.Minute
	}

	if config.MaxBackups == 0 {
		config.MaxBackups = 5
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

	// Create config directory if it doesn't exist
	os.MkdirAll(config.ConfigDir, 0755)

	// Create manager
	cm := &ConfigurationManager{
		config:        config,
		currentParams: make(map[string]*HNSWParameters),
	}

	// Load existing configurations
	cm.loadConfigurations()

	// Start background persistence
	go cm.persistenceLoop()

	return cm
}

// GetParameters returns the current parameters for an index
func (cm *ConfigurationManager) GetParameters(indexName string) *HNSWParameters {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	params, exists := cm.currentParams[indexName]
	if !exists {
		// Return default parameters
		return &HNSWParameters{
			M:              16,
			EfConstruction: 200,
			EfSearch:       100,
			DelaunayType:   "simple",
		}
	}

	return params
}

// SetParameters updates the parameters for an index
func (cm *ConfigurationManager) SetParameters(indexName string, params *HNSWParameters) {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	// Record the change
	oldParams := cm.currentParams[indexName]
	if oldParams == nil {
		oldParams = &HNSWParameters{
			M:              16,
			EfConstruction: 200,
			EfSearch:       100,
			DelaunayType:   "simple",
		}
	}

	// Create change record
	change := ConfigChangeRecord{
		Timestamp:     time.Now(),
		IndexName:     indexName,
		OldParameters: oldParams,
		NewParameters: params,
		Applied:       cm.config.AutoApply,
	}

	// If auto-apply is enabled, mark as applied
	if cm.config.AutoApply {
		change.AppliedAt = time.Now()
	}

	// Update current parameters
	cm.currentParams[indexName] = params

	// Add to history
	cm.changeHistory = append(cm.changeHistory, change)

	// Limit history size
	if len(cm.changeHistory) > 100 {
		cm.changeHistory = cm.changeHistory[len(cm.changeHistory)-100:]
	}
}

// ApplyParameters marks a parameter change as applied
func (cm *ConfigurationManager) ApplyParameters(indexName string) {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	// Find the most recent change for this index
	for i := len(cm.changeHistory) - 1; i >= 0; i-- {
		change := &cm.changeHistory[i]
		if change.IndexName == indexName && !change.Applied {
			change.Applied = true
			change.AppliedAt = time.Now()
			break
		}
	}
}

// GetChangeHistory returns the history of parameter changes
func (cm *ConfigurationManager) GetChangeHistory() []ConfigChangeRecord {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	return cm.changeHistory
}

// GetPendingChanges returns parameter changes that haven't been applied
func (cm *ConfigurationManager) GetPendingChanges() []ConfigChangeRecord {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	var pending []ConfigChangeRecord
	for _, change := range cm.changeHistory {
		if !change.Applied {
			pending = append(pending, change)
		}
	}

	return pending
}

// GetThresholds returns the performance thresholds
func (cm *ConfigurationManager) GetThresholds() ConfigThresholds {
	return cm.config.Thresholds
}

// persistenceLoop periodically persists configurations to disk
func (cm *ConfigurationManager) persistenceLoop() {
	ticker := time.NewTicker(cm.config.PersistInterval)
	defer ticker.Stop()

	for range ticker.C {
		cm.persistConfigurations()
	}
}

// persistConfigurations saves the current configurations to disk
func (cm *ConfigurationManager) persistConfigurations() error {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	// Create a backup of the current config file
	configFile := filepath.Join(cm.config.ConfigDir, "parameters.json")
	if _, err := os.Stat(configFile); err == nil {
		// File exists, create backup
		backupFile := filepath.Join(cm.config.ConfigDir,
			fmt.Sprintf("parameters.%s.json", time.Now().Format("20060102-150405")))

		// Copy file
		data, err := ioutil.ReadFile(configFile)
		if err == nil {
			ioutil.WriteFile(backupFile, data, 0644)
		}

		// Clean up old backups
		cm.cleanupBackups()
	}

	// Marshal current parameters
	data, err := json.MarshalIndent(cm.currentParams, "", "  ")
	if err != nil {
		return err
	}

	// Write to file
	return ioutil.WriteFile(configFile, data, 0644)
}

// loadConfigurations loads configurations from disk
func (cm *ConfigurationManager) loadConfigurations() error {
	configFile := filepath.Join(cm.config.ConfigDir, "parameters.json")

	// Check if file exists
	if _, err := os.Stat(configFile); os.IsNotExist(err) {
		return nil // No file to load
	}

	// Read file
	data, err := ioutil.ReadFile(configFile)
	if err != nil {
		return err
	}

	// Unmarshal data
	var params map[string]*HNSWParameters
	err = json.Unmarshal(data, &params)
	if err != nil {
		return err
	}

	// Update current parameters
	cm.mu.Lock()
	cm.currentParams = params
	cm.mu.Unlock()

	return nil
}

// cleanupBackups removes old backup files
func (cm *ConfigurationManager) cleanupBackups() {
	// Get all backup files
	pattern := filepath.Join(cm.config.ConfigDir, "parameters.*.json")
	files, err := filepath.Glob(pattern)
	if err != nil {
		return
	}

	// If we have fewer files than the max, do nothing
	if len(files) <= cm.config.MaxBackups {
		return
	}

	// Sort files by modification time (oldest first)
	type fileInfo struct {
		path    string
		modTime time.Time
	}

	fileInfos := make([]fileInfo, 0, len(files))
	for _, file := range files {
		info, err := os.Stat(file)
		if err != nil {
			continue
		}
		fileInfos = append(fileInfos, fileInfo{file, info.ModTime()})
	}

	// Sort by modification time (oldest first)
	for i := 0; i < len(fileInfos); i++ {
		for j := i + 1; j < len(fileInfos); j++ {
			if fileInfos[i].modTime.After(fileInfos[j].modTime) {
				fileInfos[i], fileInfos[j] = fileInfos[j], fileInfos[i]
			}
		}
	}

	// Delete oldest files
	for i := 0; i < len(fileInfos)-cm.config.MaxBackups; i++ {
		os.Remove(fileInfos[i].path)
	}
}
