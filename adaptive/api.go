// Package adaptive implements the Adaptive Parameter Tuning (APT) system for Quiver.
package adaptive

import (
	"encoding/json"
	"net/http"
	"time"
)

// APIHandler handles API requests for the APT system
type APIHandler struct {
	// Integration instance
	integration *QuiverIntegration
}

// NewAPIHandler creates a new API handler
func NewAPIHandler(integration *QuiverIntegration) *APIHandler {
	return &APIHandler{
		integration: integration,
	}
}

// RegisterHandlers registers the API handlers with the HTTP server
func (h *APIHandler) RegisterHandlers(mux *http.ServeMux) {
	mux.HandleFunc("/api/apt/status", h.handleStatus)
	mux.HandleFunc("/api/apt/enable", h.handleEnable)
	mux.HandleFunc("/api/apt/disable", h.handleDisable)
	mux.HandleFunc("/api/apt/parameters", h.handleParameters)
	mux.HandleFunc("/api/apt/workload", h.handleWorkload)
	mux.HandleFunc("/api/apt/performance", h.handlePerformance)
	mux.HandleFunc("/api/apt/history", h.handleHistory)
}

// handleStatus handles the status API request
func (h *APIHandler) handleStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Get status
	status := map[string]interface{}{
		"enabled": h.integration.IsEnabled(),
	}

	// Add parameters if enabled
	if h.integration.IsEnabled() {
		params := h.integration.GetCurrentParameters()
		status["parameters"] = params
	}

	// Write response
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(status)
}

// handleEnable handles the enable API request
func (h *APIHandler) handleEnable(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Enable APT
	err := h.integration.SetEnabled(true)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Write response
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]bool{"enabled": true})
}

// handleDisable handles the disable API request
func (h *APIHandler) handleDisable(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Disable APT
	err := h.integration.SetEnabled(false)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Write response
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]bool{"enabled": false})
}

// handleParameters handles the parameters API request
func (h *APIHandler) handleParameters(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Get parameters
	params := h.integration.GetCurrentParameters()

	// Write response
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(params)
}

// handleWorkload handles the workload API request
func (h *APIHandler) handleWorkload(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Get workload analysis
	workload := h.integration.GetWorkloadAnalysis()

	// Write response
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(workload)
}

// handlePerformance handles the performance API request
func (h *APIHandler) handlePerformance(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Get performance report
	performance := h.integration.GetPerformanceReport()

	// Write response
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(performance)
}

// handleHistory handles the history API request
func (h *APIHandler) handleHistory(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Get parameter history
	history := h.integration.GetParameterHistory()

	// Format history for JSON
	formattedHistory := make([]map[string]interface{}, 0, len(history))
	for _, record := range history {
		formattedRecord := map[string]interface{}{
			"timestamp":    record.Timestamp.Format(time.RFC3339),
			"strategy":     record.Strategy,
			"old_params":   record.OldParameters,
			"new_params":   record.NewParameters,
			"improvement":  record.Improvement,
			"evaluated_at": record.EvaluatedAt.Format(time.RFC3339),
		}
		formattedHistory = append(formattedHistory, formattedRecord)
	}

	// Write response
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(formattedHistory)
}
