package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// APIClient represents a client for the Quiver API
type APIClient struct {
	BaseURL    string
	HTTPClient *http.Client
}

// NewAPIClient creates a new API client
func NewAPIClient(baseURL string) *APIClient {
	return &APIClient{
		BaseURL: baseURL,
		HTTPClient: &http.Client{
			Timeout: time.Second * 30,
		},
	}
}

// AddVector adds a vector to the database
func (c *APIClient) AddVector(id uint64, vector []float32, metadata map[string]interface{}, facets []map[string]interface{}) (map[string]interface{}, error) {
	payload := map[string]interface{}{
		"id":       id,
		"vector":   vector,
		"metadata": metadata,
		"facets":   facets,
	}

	resp, err := c.sendRequest("POST", "/vectors", payload)
	if err != nil {
		return nil, err
	}

	return resp, nil
}

// DeleteVector deletes a vector from the database
func (c *APIClient) DeleteVector(id uint64) (map[string]interface{}, error) {
	resp, err := c.sendRequest("DELETE", fmt.Sprintf("/vectors/%d", id), nil)
	if err != nil {
		return nil, err
	}

	return resp, nil
}

// GetVector gets a vector from the database
func (c *APIClient) GetVector(id uint64) (map[string]interface{}, error) {
	resp, err := c.sendRequest("GET", fmt.Sprintf("/vectors/%d", id), nil)
	if err != nil {
		return nil, err
	}

	return resp, nil
}

// Search performs a vector similarity search
func (c *APIClient) Search(vector []float32, k int) (map[string]interface{}, error) {
	payload := map[string]interface{}{
		"vector": vector,
		"k":      k,
	}

	resp, err := c.sendRequest("POST", "/search", payload)
	if err != nil {
		return nil, err
	}

	return resp, nil
}

// HybridSearch performs a hybrid search (vector + metadata filter)
func (c *APIClient) HybridSearch(vector []float32, k int, filter string) (map[string]interface{}, error) {
	payload := map[string]interface{}{
		"vector": vector,
		"k":      k,
		"filter": filter,
	}

	resp, err := c.sendRequest("POST", "/search/hybrid", payload)
	if err != nil {
		return nil, err
	}

	return resp, nil
}

// SearchWithNegatives performs a search with negative examples
func (c *APIClient) SearchWithNegatives(positiveVector []float32, negativeVectors [][]float32, k int, weight float32) (map[string]interface{}, error) {
	payload := map[string]interface{}{
		"positive_vector":  positiveVector,
		"negative_vectors": negativeVectors,
		"k":                k,
		"negative_weight":  weight,
	}

	resp, err := c.sendRequest("POST", "/search/negatives", payload)
	if err != nil {
		return nil, err
	}

	return resp, nil
}

// CreateIndex creates a new index
func (c *APIClient) CreateIndex(opts map[string]interface{}) (map[string]interface{}, error) {
	resp, err := c.sendRequest("POST", "/index", opts)
	if err != nil {
		return nil, err
	}

	return resp, nil
}

// BackupIndex backs up the index to a specified path
func (c *APIClient) BackupIndex(path string) (map[string]interface{}, error) {
	payload := map[string]interface{}{
		"path": path,
	}

	resp, err := c.sendRequest("POST", "/backup", payload)
	if err != nil {
		return nil, err
	}

	return resp, nil
}

// RestoreIndex restores the index from a specified path
func (c *APIClient) RestoreIndex(path string) (map[string]interface{}, error) {
	payload := map[string]interface{}{
		"path": path,
	}

	resp, err := c.sendRequest("POST", "/restore", payload)
	if err != nil {
		return nil, err
	}

	return resp, nil
}

// sendRequest sends an HTTP request to the API
func (c *APIClient) sendRequest(method, endpoint string, payload interface{}) (map[string]interface{}, error) {
	var body io.Reader
	if payload != nil {
		jsonData, err := json.Marshal(payload)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal payload: %w", err)
		}
		body = bytes.NewBuffer(jsonData)
	}

	url := c.BaseURL + endpoint
	req, err := http.NewRequest(method, url, body)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(bodyBytes))
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return result, nil
}

// ParseVector parses a comma-separated string of floats into a []float32
func ParseVector(vectorStr string) ([]float32, error) {
	if vectorStr == "" {
		return nil, fmt.Errorf("vector string is empty")
	}

	parts := strings.Split(vectorStr, ",")
	vector := make([]float32, len(parts))

	for i, part := range parts {
		val, err := strconv.ParseFloat(strings.TrimSpace(part), 32)
		if err != nil {
			return nil, fmt.Errorf("failed to parse vector component %d: %w", i, err)
		}
		vector[i] = float32(val)
	}

	return vector, nil
}

// ParseNegativeVectors parses a semicolon-separated list of comma-separated vectors
func ParseNegativeVectors(vectorsStr string) ([][]float32, error) {
	if vectorsStr == "" {
		return nil, fmt.Errorf("vectors string is empty")
	}

	vectorStrings := strings.Split(vectorsStr, ";")
	vectors := make([][]float32, len(vectorStrings))

	for i, vectorStr := range vectorStrings {
		vector, err := ParseVector(vectorStr)
		if err != nil {
			return nil, fmt.Errorf("failed to parse vector %d: %w", i, err)
		}
		vectors[i] = vector
	}

	return vectors, nil
}

// ParseMetadata parses a JSON string into a map
func ParseMetadata(metadataStr string) (map[string]interface{}, error) {
	if metadataStr == "" {
		return nil, fmt.Errorf("metadata string is empty")
	}

	var metadata map[string]interface{}
	if err := json.Unmarshal([]byte(metadataStr), &metadata); err != nil {
		return nil, fmt.Errorf("failed to parse metadata: %w", err)
	}

	return metadata, nil
}
