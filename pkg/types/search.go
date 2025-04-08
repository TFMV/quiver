package types

import (
	"encoding/json"
	"time"
)

// BasicSearchResult represents a minimal search result with just ID and distance
type BasicSearchResult struct {
	// ID of the vector
	ID string
	// Distance from the query vector (lower is better)
	Distance float32
}

// SearchResultMetadata contains additional information about the search results
type SearchResultMetadata struct {
	// TotalCount is the total number of vectors that matched the search criteria
	TotalCount int `json:"total_count"`
	// SearchTime is the time taken to execute the search in milliseconds
	SearchTime float64 `json:"search_time_ms"`
	// IndexSize is the total number of vectors in the index
	IndexSize int `json:"index_size"`
	// IndexName is the name of the index that was searched
	IndexName string `json:"index_name,omitempty"`
	// Timestamp is when the search was performed
	Timestamp time.Time `json:"timestamp"`
}

// SearchResultItem represents a single search result with detailed information
type SearchResultItem struct {
	// ID of the vector
	ID string `json:"id"`
	// Distance from the query vector (lower is better)
	Distance float32 `json:"distance"`
	// Score is the similarity score (1.0 - distance); higher is more similar
	Score float32 `json:"score"`
	// Vector holds the actual vector values (if requested)
	Vector []float32 `json:"vector,omitempty"`
	// Metadata is the user-defined metadata associated with this vector
	Metadata json.RawMessage `json:"metadata,omitempty"`
}

// SearchOptions defines options for search operations
type SearchOptions struct {
	// IncludeVectors determines whether vector values should be included in results
	IncludeVectors bool `json:"include_vectors"`
	// IncludeMetadata determines whether metadata should be included in results
	IncludeMetadata bool `json:"include_metadata"`
	// ExactSearch determines whether to use exact search (slower but more accurate)
	ExactSearch bool `json:"exact_search"`
}

// SearchResponse is the complete response returned by a search operation
type SearchResponse struct {
	// Results is the list of matching vectors
	Results []SearchResultItem `json:"results"`
	// Metadata contains information about the search operation
	Metadata SearchResultMetadata `json:"metadata"`
	// Query is the original query vector (if echoing is enabled)
	Query []float32 `json:"query,omitempty"`
}

// Filter represents a condition for filtering vectors by metadata
type Filter struct {
	// Field is the metadata field name to filter on
	Field string `json:"field"`
	// Operator is the comparison operator (=, !=, >, <, etc.)
	Operator string `json:"operator"`
	// Value is the value to compare against
	Value interface{} `json:"value"`
}

// SearchRequest represents a complete search query
type SearchRequest struct {
	// Vector is the query vector to search for
	Vector []float32 `json:"vector"`
	// TopK is the number of results to return
	TopK int `json:"top_k"`
	// Filters are metadata constraints to apply
	Filters []Filter `json:"filters,omitempty"`
	// Options contains additional search options
	Options SearchOptions `json:"options,omitempty"`
	// NamespaceID is an optional namespace to restrict the search to
	NamespaceID string `json:"namespace_id,omitempty"`
}

// ToSearchResultItem converts a BasicSearchResult to a SearchResultItem
func (basic BasicSearchResult) ToSearchResultItem() SearchResultItem {
	return SearchResultItem{
		ID:       basic.ID,
		Distance: basic.Distance,
		Score:    1.0 - basic.Distance, // Normalize to a similarity score (higher is better)
	}
}

// NewSearchResponse creates a new search response from basic results
func NewSearchResponse(results []BasicSearchResult, indexName string, searchTime float64, totalSize int) SearchResponse {
	items := make([]SearchResultItem, len(results))
	for i, res := range results {
		items[i] = res.ToSearchResultItem()
	}

	return SearchResponse{
		Results: items,
		Metadata: SearchResultMetadata{
			TotalCount: len(results),
			SearchTime: searchTime,
			IndexSize:  totalSize,
			IndexName:  indexName,
			Timestamp:  time.Now(),
		},
	}
}
