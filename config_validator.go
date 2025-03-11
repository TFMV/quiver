package quiver

import (
	"fmt"
	"strings"
	"time"
)

// ValidationIssue represents a configuration validation issue
type ValidationIssue struct {
	Field      string             // The field with the issue
	Value      interface{}        // The current value
	Message    string             // Description of the issue
	Severity   ValidationSeverity // How severe the issue is
	Suggestion string             // Suggested fix
}

// ValidationSeverity indicates how severe a validation issue is
type ValidationSeverity int

const (
	// Error indicates a configuration that will not work
	Error ValidationSeverity = iota
	// Warning indicates a configuration that may cause problems
	Warning
	// Info indicates a configuration that could be improved
	Info
)

// String returns a string representation of the severity
func (s ValidationSeverity) String() string {
	switch s {
	case Error:
		return "ERROR"
	case Warning:
		return "WARNING"
	case Info:
		return "INFO"
	default:
		return "UNKNOWN"
	}
}

// ValidateConfig performs comprehensive validation of a Quiver configuration
// and returns a list of validation issues
func ValidateConfig(config Config) []ValidationIssue {
	var issues []ValidationIssue

	// Validate dimension
	if config.Dimension <= 0 {
		issues = append(issues, ValidationIssue{
			Field:      "Dimension",
			Value:      config.Dimension,
			Message:    "Dimension must be greater than 0",
			Severity:   Error,
			Suggestion: "Set Dimension to a positive value based on your embedding model's output size",
		})
	} else if config.Dimension > 10000 {
		issues = append(issues, ValidationIssue{
			Field:      "Dimension",
			Value:      config.Dimension,
			Message:    "Dimension is unusually high",
			Severity:   Warning,
			Suggestion: "Consider using dimensionality reduction for better performance",
		})
	}

	// Validate storage path
	if config.StoragePath == "" {
		issues = append(issues, ValidationIssue{
			Field:      "StoragePath",
			Value:      config.StoragePath,
			Message:    "StoragePath is empty",
			Severity:   Warning,
			Suggestion: "Set StoragePath to a valid directory for persistence",
		})
	}

	// Validate distance metric
	if config.Distance != Cosine && config.Distance != L2 {
		issues = append(issues, ValidationIssue{
			Field:      "Distance",
			Value:      config.Distance,
			Message:    "Invalid distance metric",
			Severity:   Error,
			Suggestion: "Use quiver.Cosine or quiver.L2",
		})
	}

	// Validate HNSW parameters
	if config.HNSWM <= 0 {
		issues = append(issues, ValidationIssue{
			Field:      "HNSWM",
			Value:      config.HNSWM,
			Message:    "HNSWM must be greater than 0",
			Severity:   Error,
			Suggestion: "Set HNSWM to a value between 5 and 100 (16 is a good default)",
		})
	} else if config.HNSWM < 5 {
		issues = append(issues, ValidationIssue{
			Field:      "HNSWM",
			Value:      config.HNSWM,
			Message:    "HNSWM is too low for good recall",
			Severity:   Warning,
			Suggestion: "Consider increasing HNSWM to at least 5 for better recall",
		})
	} else if config.HNSWM > 100 {
		issues = append(issues, ValidationIssue{
			Field:      "HNSWM",
			Value:      config.HNSWM,
			Message:    "HNSWM is unusually high",
			Severity:   Warning,
			Suggestion: "High HNSWM values increase memory usage and construction time",
		})
	}

	if config.HNSWEfConstruct <= 0 {
		issues = append(issues, ValidationIssue{
			Field:      "HNSWEfConstruct",
			Value:      config.HNSWEfConstruct,
			Message:    "HNSWEfConstruct must be greater than 0",
			Severity:   Error,
			Suggestion: "Set HNSWEfConstruct to a value between 50 and 500 (200 is a good default)",
		})
	} else if config.HNSWEfConstruct < 50 {
		issues = append(issues, ValidationIssue{
			Field:      "HNSWEfConstruct",
			Value:      config.HNSWEfConstruct,
			Message:    "HNSWEfConstruct is too low for good recall",
			Severity:   Warning,
			Suggestion: "Consider increasing HNSWEfConstruct to at least 50 for better recall",
		})
	}

	if config.HNSWEfSearch <= 0 {
		issues = append(issues, ValidationIssue{
			Field:      "HNSWEfSearch",
			Value:      config.HNSWEfSearch,
			Message:    "HNSWEfSearch must be greater than 0",
			Severity:   Error,
			Suggestion: "Set HNSWEfSearch to a value between 20 and 500 (100 is a good default)",
		})
	} else if config.HNSWEfSearch < 20 {
		issues = append(issues, ValidationIssue{
			Field:      "HNSWEfSearch",
			Value:      config.HNSWEfSearch,
			Message:    "HNSWEfSearch is too low for good recall",
			Severity:   Warning,
			Suggestion: "Consider increasing HNSWEfSearch to at least 20 for better recall",
		})
	}

	// Validate batch size
	if config.BatchSize <= 0 {
		issues = append(issues, ValidationIssue{
			Field:      "BatchSize",
			Value:      config.BatchSize,
			Message:    "BatchSize must be greater than 0",
			Severity:   Error,
			Suggestion: "Set BatchSize to a positive value (1000 is a good default)",
		})
	} else if config.BatchSize < 100 {
		issues = append(issues, ValidationIssue{
			Field:      "BatchSize",
			Value:      config.BatchSize,
			Message:    "BatchSize is unusually low",
			Severity:   Info,
			Suggestion: "Small batch sizes may reduce throughput for bulk operations",
		})
	} else if config.BatchSize > 10000 {
		issues = append(issues, ValidationIssue{
			Field:      "BatchSize",
			Value:      config.BatchSize,
			Message:    "BatchSize is unusually high",
			Severity:   Warning,
			Suggestion: "Very large batch sizes may cause memory pressure",
		})
	}

	// Validate persistence interval
	if config.PersistInterval < time.Second {
		issues = append(issues, ValidationIssue{
			Field:      "PersistInterval",
			Value:      config.PersistInterval,
			Message:    "PersistInterval is too short",
			Severity:   Warning,
			Suggestion: "Set PersistInterval to at least 1 second (5 minutes is recommended)",
		})
	}

	// Validate backup settings
	if config.BackupInterval > 0 && config.BackupPath == "" {
		issues = append(issues, ValidationIssue{
			Field:      "BackupPath",
			Value:      config.BackupPath,
			Message:    "BackupPath is empty but BackupInterval is set",
			Severity:   Error,
			Suggestion: "Specify a BackupPath for scheduled backups",
		})
	}

	if config.MaxBackups < 1 && config.BackupInterval > 0 {
		issues = append(issues, ValidationIssue{
			Field:      "MaxBackups",
			Value:      config.MaxBackups,
			Message:    "MaxBackups must be at least 1 when backups are enabled",
			Severity:   Error,
			Suggestion: "Set MaxBackups to at least 1 (5 is recommended)",
		})
	}

	// Validate encryption settings
	if config.EncryptionEnabled && len(config.EncryptionKey) < 32 {
		issues = append(issues, ValidationIssue{
			Field:      "EncryptionKey",
			Value:      "***",
			Message:    "EncryptionKey is too short",
			Severity:   Error,
			Suggestion: "EncryptionKey must be at least 32 bytes long",
		})
	}

	// Validate dimensionality reduction settings
	if config.EnableDimReduction {
		if config.DimReductionTarget <= 0 {
			issues = append(issues, ValidationIssue{
				Field:      "DimReductionTarget",
				Value:      config.DimReductionTarget,
				Message:    "DimReductionTarget must be greater than 0",
				Severity:   Error,
				Suggestion: "Set DimReductionTarget to a positive value",
			})
		} else if config.DimReductionTarget >= config.Dimension {
			issues = append(issues, ValidationIssue{
				Field:      "DimReductionTarget",
				Value:      config.DimReductionTarget,
				Message:    "DimReductionTarget must be less than Dimension",
				Severity:   Error,
				Suggestion: fmt.Sprintf("Set DimReductionTarget to a value less than %d", config.Dimension),
			})
		}

		validMethods := map[string]bool{
			"PCA":  true,
			"TSNE": false,
			"UMAP": false,
		}

		if !validMethods[config.DimReductionMethod] {
			methods := []string{"PCA", "TSNE", "UMAP"}
			issues = append(issues, ValidationIssue{
				Field:      "DimReductionMethod",
				Value:      config.DimReductionMethod,
				Message:    "Invalid dimensionality reduction method",
				Severity:   Error,
				Suggestion: fmt.Sprintf("Use one of: %s", strings.Join(methods, ", ")),
			})
		}

		if config.DimReductionAdaptive && (config.DimReductionMinVariance <= 0 || config.DimReductionMinVariance > 1) {
			issues = append(issues, ValidationIssue{
				Field:      "DimReductionMinVariance",
				Value:      config.DimReductionMinVariance,
				Message:    "DimReductionMinVariance must be between 0 and 1",
				Severity:   Error,
				Suggestion: "Set DimReductionMinVariance to a value between 0 and 1 (0.95 is recommended)",
			})
		}
	}

	// Validate max elements
	if config.MaxElements <= 0 {
		issues = append(issues, ValidationIssue{
			Field:      "MaxElements",
			Value:      config.MaxElements,
			Message:    "MaxElements must be greater than 0",
			Severity:   Error,
			Suggestion: "Set MaxElements to the maximum number of vectors you expect to store",
		})
	}

	return issues
}

// FormatValidationIssues returns a formatted string representation of validation issues
func FormatValidationIssues(issues []ValidationIssue) string {
	if len(issues) == 0 {
		return "Configuration is valid."
	}

	var errorCount, warningCount, infoCount int
	var sb strings.Builder

	sb.WriteString(fmt.Sprintf("Found %d configuration issues:\n\n", len(issues)))

	for i, issue := range issues {
		switch issue.Severity {
		case Error:
			errorCount++
		case Warning:
			warningCount++
		case Info:
			infoCount++
		}

		sb.WriteString(fmt.Sprintf("%d. [%s] %s: %v\n", i+1, issue.Severity, issue.Field, issue.Message))
		sb.WriteString(fmt.Sprintf("   Current value: %v\n", issue.Value))
		sb.WriteString(fmt.Sprintf("   Suggestion: %s\n\n", issue.Suggestion))
	}

	sb.WriteString(fmt.Sprintf("Summary: %d errors, %d warnings, %d informational\n",
		errorCount, warningCount, infoCount))

	return sb.String()
}

// ValidateConfigAndPrint validates a configuration and prints the results
func ValidateConfigAndPrint(config Config) bool {
	issues := ValidateConfig(config)
	fmt.Println(FormatValidationIssues(issues))

	// Return true if there are no errors (warnings and info are ok)
	for _, issue := range issues {
		if issue.Severity == Error {
			return false
		}
	}
	return true
}
