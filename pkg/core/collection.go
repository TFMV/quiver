package core

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/TFMV/quiver/pkg/facets"
	"github.com/TFMV/quiver/pkg/types"
	"github.com/TFMV/quiver/pkg/vectortypes"
)

var (
	ErrVectorNotFound     = errors.New("vector not found")
	ErrInvalidDimension   = errors.New("invalid vector dimension")
	ErrVectorAlreadyExist = errors.New("vector with the same ID already exists")
	ErrInvalidMetadata    = errors.New("invalid metadata format")
)

// FilterOperator defines the type of comparison for metadata filtering
type FilterOperator string

const (
	// Equals checks if a field exactly matches a value
	Equals FilterOperator = "="
	// NotEquals checks if a field does not match a value
	NotEquals FilterOperator = "!="
	// GreaterThan checks if a field is greater than a value
	GreaterThan FilterOperator = ">"
	// GreaterThanOrEqual checks if a field is greater than or equal to a value
	GreaterThanOrEqual FilterOperator = ">="
	// LessThan checks if a field is less than a value
	LessThan FilterOperator = "<"
	// LessThanOrEqual checks if a field is less than or equal to a value
	LessThanOrEqual FilterOperator = "<="
	// In checks if a field exists in a list of values
	In FilterOperator = "in"
	// NotIn checks if a field does not exist in a list of values
	NotIn FilterOperator = "not_in"
)

// Filter represents a condition for filtering vectors by metadata
type Filter struct {
	Field    string         `json:"field"`
	Operator FilterOperator `json:"operator"`
	Value    interface{}    `json:"value"`
}

// SearchResult contains a single search result
type SearchResult struct {
	// ID of the vector
	ID string `json:"id"`
	// Distance from the query vector (lower is better)
	Distance float32 `json:"distance"`
	// Vector values
	Vector []float32 `json:"vector,omitempty"`
	// Metadata
	Metadata json.RawMessage `json:"metadata,omitempty"`
}

// CollectionStats contains statistics about a collection
type CollectionStats struct {
	// Name of the collection
	Name string `json:"name"`
	// Number of vectors in the collection
	VectorCount int `json:"vector_count"`
	// Dimension of vectors in the collection
	Dimension int `json:"dimension"`
	// Creation time of the collection
	CreatedAt time.Time `json:"created_at"`
}

// Index defines the interface for a vector index implementation
type Index interface {
	// Insert adds a vector to the index
	Insert(id string, vector vectortypes.F32) error
	// Delete removes a vector from the index
	Delete(id string) error
	// Search finds the k nearest vectors to the query vector
	Search(vector vectortypes.F32, k int) ([]types.BasicSearchResult, error)
	// Size returns the number of vectors in the index
	Size() int
}

// BatchIndex defines optional batch operations that an index can implement
// for more efficient batch processing
type BatchIndex interface {
	// InsertBatch adds multiple vectors to the index in one operation
	InsertBatch(vectors map[string]vectortypes.F32) error
	// DeleteBatch removes multiple vectors from the index in one operation
	DeleteBatch(ids []string) error
}

// Collection represents a named collection of vectors with their metadata
type Collection struct {
	// Name of the collection
	Name string
	// Dimension of vectors in this collection
	Dimension int
	// Vector index implementation
	Index Index
	// Map of vector IDs to their metadata
	Metadata map[string]json.RawMessage
	// Map of vector IDs to their vector values for direct access
	Vectors map[string]vectortypes.F32
	// Creation time of the collection
	CreatedAt time.Time
	// Lock for concurrent access
	sync.RWMutex
	// Facet fields for future vectors
	FacetFields  []string                       `json:"facet_fields,omitempty"`
	vectorFacets map[string][]facets.FacetValue `json:"-"`
}

// NewCollection creates a new vector collection
func NewCollection(name string, dimension int, index Index) *Collection {
	return &Collection{
		Name:         name,
		Dimension:    dimension,
		Index:        index,
		Metadata:     make(map[string]json.RawMessage),
		Vectors:      make(map[string]vectortypes.F32),
		CreatedAt:    time.Now(),
		vectorFacets: make(map[string][]facets.FacetValue),
	}
}

// Add inserts a new vector with optional metadata
func (c *Collection) Add(id string, vector vectortypes.F32, metadata json.RawMessage) error {
	c.Lock()
	defer c.Unlock()

	// Validate vector dimension
	if len(vector) != c.Dimension {
		return fmt.Errorf("%w: expected %d, got %d", ErrInvalidDimension, c.Dimension, len(vector))
	}

	// Validate metadata JSON format if provided
	if len(metadata) > 0 {
		var metadataMap map[string]interface{}
		if err := json.Unmarshal(metadata, &metadataMap); err != nil {
			return fmt.Errorf("%w: %v", ErrInvalidMetadata, err)
		}
	}

	// Add to index
	if err := c.Index.Insert(id, vector); err != nil {
		return err
	}

	// Store vector and metadata
	c.Vectors[id] = vector
	c.Metadata[id] = metadata

	// Extract and store facets if facet fields are defined
	if len(c.FacetFields) > 0 && len(metadata) > 0 {
		var metadataMap map[string]interface{}
		if err := json.Unmarshal(metadata, &metadataMap); err == nil {
			c.vectorFacets[id] = facets.ExtractFacets(metadataMap, c.FacetFields)
		}
	}

	return nil
}

// AddBatch inserts multiple vectors in batch
func (c *Collection) AddBatch(vectors []vectortypes.Vector) error {
	c.Lock()
	defer c.Unlock()

	// Pre-validate all vectors
	for _, v := range vectors {
		if len(v.Values) != c.Dimension {
			return fmt.Errorf("%w for vector %s: expected %d, got %d",
				ErrInvalidDimension, v.ID, c.Dimension, len(v.Values))
		}

		if len(v.Metadata) > 0 {
			var metadataMap map[string]interface{}
			if err := json.Unmarshal(v.Metadata, &metadataMap); err != nil {
				return fmt.Errorf("%w for vector %s: %v", ErrInvalidMetadata, v.ID, err)
			}
		}
	}

	// Check if the index supports batch operations
	if batchIndex, ok := c.Index.(BatchIndex); ok {
		// Create a map for batch insertion
		vectorsMap := make(map[string]vectortypes.F32, len(vectors))
		for _, v := range vectors {
			vectorsMap[v.ID] = v.Values
		}

		// Perform batch insert
		if err := batchIndex.InsertBatch(vectorsMap); err != nil {
			return err
		}

		// Store vectors and metadata
		for _, v := range vectors {
			c.Vectors[v.ID] = v.Values
			c.Metadata[v.ID] = v.Metadata

			// Extract and store facets if facet fields are defined
			if len(c.FacetFields) > 0 && len(v.Metadata) > 0 {
				var metadataMap map[string]interface{}
				if err := json.Unmarshal(v.Metadata, &metadataMap); err == nil {
					c.vectorFacets[v.ID] = facets.ExtractFacets(metadataMap, c.FacetFields)
				}
			}
		}

		return nil
	}

	// Fallback to individual insertions if the index doesn't support batch operations
	for _, v := range vectors {
		if err := c.Index.Insert(v.ID, v.Values); err != nil {
			return err
		}
		c.Vectors[v.ID] = v.Values
		c.Metadata[v.ID] = v.Metadata

		// Extract and store facets if facet fields are defined
		if len(c.FacetFields) > 0 && len(v.Metadata) > 0 {
			var metadataMap map[string]interface{}
			if err := json.Unmarshal(v.Metadata, &metadataMap); err == nil {
				c.vectorFacets[v.ID] = facets.ExtractFacets(metadataMap, c.FacetFields)
			}
		}
	}

	return nil
}

// Get retrieves a vector by ID
func (c *Collection) Get(id string) (*vectortypes.Vector, error) {
	c.RLock()
	defer c.RUnlock()

	vector, exists := c.Vectors[id]
	if !exists {
		return nil, ErrVectorNotFound
	}

	metadata, exists := c.Metadata[id]
	if !exists {
		metadata = json.RawMessage("{}")
	}

	return &vectortypes.Vector{
		ID:       id,
		Values:   vector,
		Metadata: metadata,
	}, nil
}

// Delete removes a vector by ID
func (c *Collection) Delete(id string) error {
	c.Lock()
	defer c.Unlock()

	if _, exists := c.Vectors[id]; !exists {
		return ErrVectorNotFound
	}

	if err := c.Index.Delete(id); err != nil {
		return err
	}

	delete(c.Vectors, id)
	delete(c.Metadata, id)
	delete(c.vectorFacets, id)
	return nil
}

// DeleteBatch removes multiple vectors by their IDs
func (c *Collection) DeleteBatch(ids []string) error {
	c.Lock()
	defer c.Unlock()

	// Verify all vectors exist first
	for _, id := range ids {
		if _, exists := c.Vectors[id]; !exists {
			return fmt.Errorf("%w: %s", ErrVectorNotFound, id)
		}
	}

	// Check if the index supports batch operations
	if batchIndex, ok := c.Index.(BatchIndex); ok {
		// Perform batch delete
		if err := batchIndex.DeleteBatch(ids); err != nil {
			return err
		}

		// Remove vectors and metadata from local maps
		for _, id := range ids {
			delete(c.Vectors, id)
			delete(c.Metadata, id)
			delete(c.vectorFacets, id)
		}

		return nil
	}

	// Fallback to individual deletions if the index doesn't support batch operations
	for _, id := range ids {
		if err := c.Index.Delete(id); err != nil {
			return err
		}
		delete(c.Vectors, id)
		delete(c.Metadata, id)
		delete(c.vectorFacets, id)
	}

	return nil
}

// Update updates a vector and/or its metadata
func (c *Collection) Update(id string, vector []float32, metadata json.RawMessage) error {
	c.Lock()
	defer c.Unlock()

	if _, exists := c.Vectors[id]; !exists {
		return ErrVectorNotFound
	}

	// Check dimension if vector is provided
	if vector != nil && len(vector) != c.Dimension {
		return fmt.Errorf("%w: expected %d, got %d", ErrInvalidDimension, c.Dimension, len(vector))
	}

	// Validate metadata if provided
	if len(metadata) > 0 {
		var metadataMap map[string]interface{}
		if err := json.Unmarshal(metadata, &metadataMap); err != nil {
			return fmt.Errorf("%w: %v", ErrInvalidMetadata, err)
		}
	}

	// Update vector if provided
	if vector != nil {
		// Delete and re-insert with the same ID to update vector
		if err := c.Index.Delete(id); err != nil {
			return err
		}
		if err := c.Index.Insert(id, vectortypes.F32(vector)); err != nil {
			return err
		}
		c.Vectors[id] = vectortypes.F32(vector)
	}

	// Update metadata if provided
	if len(metadata) > 0 {
		c.Metadata[id] = metadata

		// Update facets if facet fields are defined
		if len(c.FacetFields) > 0 && len(metadata) > 0 {
			var metadataMap map[string]interface{}
			if err := json.Unmarshal(metadata, &metadataMap); err == nil {
				c.vectorFacets[id] = facets.ExtractFacets(metadataMap, c.FacetFields)
			}
		} else {
			delete(c.vectorFacets, id)
		}
	}

	return nil
}

// UpdateBatch updates multiple vectors in batch
func (c *Collection) UpdateBatch(vectors []vectortypes.Vector) error {
	c.Lock()
	defer c.Unlock()

	// Validate all vectors first
	for _, v := range vectors {
		if _, exists := c.Vectors[v.ID]; !exists {
			return fmt.Errorf("%w: %s", ErrVectorNotFound, v.ID)
		}

		if len(v.Values) != c.Dimension {
			return fmt.Errorf("%w for vector %s: expected %d, got %d",
				ErrInvalidDimension, v.ID, c.Dimension, len(v.Values))
		}

		if len(v.Metadata) > 0 {
			var metadataMap map[string]interface{}
			if err := json.Unmarshal(v.Metadata, &metadataMap); err != nil {
				return fmt.Errorf("%w for vector %s: %v", ErrInvalidMetadata, v.ID, err)
			}
		}
	}

	// Update all vectors
	for _, v := range vectors {
		// Update vector
		if err := c.Index.Delete(v.ID); err != nil {
			return err
		}
		if err := c.Index.Insert(v.ID, v.Values); err != nil {
			return err
		}
		c.Vectors[v.ID] = v.Values

		// Update metadata
		if len(v.Metadata) > 0 {
			c.Metadata[v.ID] = v.Metadata

			// Update facets if facet fields are defined
			if len(c.FacetFields) > 0 && len(v.Metadata) > 0 {
				var metadataMap map[string]interface{}
				if err := json.Unmarshal(v.Metadata, &metadataMap); err == nil {
					c.vectorFacets[v.ID] = facets.ExtractFacets(metadataMap, c.FacetFields)
				}
			} else {
				delete(c.vectorFacets, v.ID)
			}
		}
	}

	return nil
}

// matchesFilter checks if a metadata map matches a filter
func matchesFilter(metadata map[string]interface{}, filter Filter) bool {
	value, exists := metadata[filter.Field]
	if !exists {
		return false
	}

	switch filter.Operator {
	case Equals:
		return valuesEqual(value, filter.Value)
	case NotEquals:
		return !valuesEqual(value, filter.Value)
	case GreaterThan:
		return compareValues(value, filter.Value) > 0
	case GreaterThanOrEqual:
		return compareValues(value, filter.Value) >= 0
	case LessThan:
		return compareValues(value, filter.Value) < 0
	case LessThanOrEqual:
		return compareValues(value, filter.Value) <= 0
	case In:
		// Value should be a slice
		if values, ok := filter.Value.([]interface{}); ok {
			for _, v := range values {
				if valuesEqual(value, v) {
					return true
				}
			}
		}
		return false
	case NotIn:
		// Value should be a slice
		if values, ok := filter.Value.([]interface{}); ok {
			for _, v := range values {
				if valuesEqual(value, v) {
					return false
				}
			}
			return true
		}
		return true
	default:
		return false
	}
}

func asFloat64(value interface{}) (float64, bool) {
	switch v := value.(type) {
	case int:
		return float64(v), true
	case int8:
		return float64(v), true
	case int16:
		return float64(v), true
	case int32:
		return float64(v), true
	case int64:
		return float64(v), true
	case float32:
		return float64(v), true
	case float64:
		return v, true
	case json.Number:
		f, err := v.Float64()
		return f, err == nil
	default:
		return 0, false
	}
}

func valuesEqual(a, b interface{}) bool {
	if af, ok := asFloat64(a); ok {
		if bf, ok := asFloat64(b); ok {
			return math.Abs(af-bf) <= 1e-9
		}
	}
	return fmt.Sprintf("%v", a) == fmt.Sprintf("%v", b)
}

func compareValues(a, b interface{}) int {
	if af, ok := asFloat64(a); ok {
		if bf, ok := asFloat64(b); ok {
			switch {
			case af < bf:
				return -1
			case af > bf:
				return 1
			default:
				return 0
			}
		}
	}

	as := fmt.Sprintf("%v", a)
	bs := fmt.Sprintf("%v", b)
	switch {
	case as < bs:
		return -1
	case as > bs:
		return 1
	default:
		return 0
	}
}

// Search performs a vector similarity search with enhanced results
func (c *Collection) Search(request types.SearchRequest) (types.SearchResponse, error) {
	startTime := time.Now()

	c.RLock()
	defer c.RUnlock()

	// Validate query vector dimension
	if len(request.Vector) != c.Dimension {
		return types.SearchResponse{}, fmt.Errorf("%w: expected %d, got %d", ErrInvalidDimension, c.Dimension, len(request.Vector))
	}

	// Validate TopK
	if request.TopK <= 0 {
		return types.SearchResponse{}, errors.New("top_k must be greater than 0")
	}
	if c.Index.Size() == 0 {
		return types.SearchResponse{
			Results: []types.SearchResultItem{},
			Metadata: types.SearchResultMetadata{
				TotalCount: 0,
				SearchTime: float64(time.Since(startTime).Microseconds()) / 1000.0,
				IndexSize:  0,
				IndexName:  c.Name,
				Timestamp:  time.Now(),
			},
		}, nil
	}

	// Perform vector search
	searchK := request.TopK
	if len(request.Filters) > 0 {
		searchK = c.Index.Size()
	}

	basicResults, err := c.Index.Search(vectortypes.F32(request.Vector), searchK)
	if err != nil {
		return types.SearchResponse{}, err
	}

	// Apply metadata filters if specified
	if len(request.Filters) > 0 {
		// Convert filters to our internal format
		filters := make([]Filter, len(request.Filters))
		for i, f := range request.Filters {
			filters[i] = Filter{
				Field:    f.Field,
				Operator: FilterOperator(f.Operator),
				Value:    f.Value,
			}
		}

		filteredResults := make([]types.BasicSearchResult, 0, min(request.TopK, len(basicResults)))
		for _, result := range basicResults {
			metadataJSON, exists := c.Metadata[result.ID]
			if !exists || len(metadataJSON) == 0 {
				continue
			}

			// Parse metadata
			var metadataMap map[string]interface{}
			if err := json.Unmarshal(metadataJSON, &metadataMap); err != nil {
				continue // Skip this result if metadata can't be parsed
			}

			// Check if metadata matches all filters
			match := true
			for _, filter := range filters {
				if !matchesFilter(metadataMap, filter) {
					match = false
					break
				}
			}

			if match {
				filteredResults = append(filteredResults, result)
				if len(filteredResults) >= request.TopK {
					break
				}
			}
		}

		basicResults = filteredResults
	} else if len(basicResults) > request.TopK {
		basicResults = basicResults[:request.TopK]
	}

	// Convert to detailed search results
	results := make([]types.SearchResultItem, len(basicResults))
	for i, res := range basicResults {
		results[i] = types.SearchResultItem{
			ID:       res.ID,
			Distance: res.Distance,
			Score:    1.0 - res.Distance, // Convert distance to similarity score
		}

		// Add vector values if requested
		if request.Options.IncludeVectors {
			if vector, exists := c.Vectors[res.ID]; exists {
				results[i].Vector = vector
			}
		}

		// Add metadata if requested
		if request.Options.IncludeMetadata {
			if metadata, exists := c.Metadata[res.ID]; exists {
				results[i].Metadata = metadata
			}
		}
	}

	// Prepare the response
	searchTime := float64(time.Since(startTime).Microseconds()) / 1000.0 // Convert to milliseconds

	response := types.SearchResponse{
		Results: results,
		Metadata: types.SearchResultMetadata{
			TotalCount: len(results),
			SearchTime: searchTime,
			IndexSize:  c.Index.Size(),
			IndexName:  c.Name,
			Timestamp:  time.Now(),
		},
	}

	// Include query vector if requested
	if request.Options.IncludeVectors {
		response.Query = request.Vector
	}

	return response, nil
}

// LegacySearch performs a search and returns the results in the old format for backward compatibility
func (c *Collection) LegacySearch(vector vectortypes.F32, k int, filters []Filter) ([]SearchResult, error) {
	// Create a types.SearchRequest from the legacy parameters
	request := types.SearchRequest{
		Vector: vector,
		TopK:   k,
		Options: types.SearchOptions{
			IncludeVectors:  true,
			IncludeMetadata: true,
		},
	}

	// Convert filters if provided
	if len(filters) > 0 {
		typesFilters := make([]types.Filter, len(filters))
		for i, f := range filters {
			typesFilters[i] = types.Filter{
				Field:    f.Field,
				Operator: string(f.Operator),
				Value:    f.Value,
			}
		}
		request.Filters = typesFilters
	}

	// Execute the new search
	response, err := c.Search(request)
	if err != nil {
		return nil, err
	}

	// Convert the new response to the old format
	results := make([]SearchResult, len(response.Results))
	for i, item := range response.Results {
		results[i] = SearchResult{
			ID:       item.ID,
			Distance: item.Distance,
			Vector:   item.Vector,
			Metadata: item.Metadata,
		}
	}

	return results, nil
}

// Count returns the number of vectors in the collection
func (c *Collection) Count() int {
	c.RLock()
	defer c.RUnlock()
	return c.Index.Size()
}

// Stats returns statistics about the collection
func (c *Collection) Stats() CollectionStats {
	c.RLock()
	defer c.RUnlock()
	return CollectionStats{
		Name:        c.Name,
		VectorCount: c.Index.Size(),
		Dimension:   c.Dimension,
		CreatedAt:   c.CreatedAt,
	}
}

// FluentSearch provides a fluent API for building search queries
type FluentSearch struct {
	collection  *Collection
	vector      vectortypes.F32
	k           int
	filters     []types.Filter
	options     types.SearchOptions
	namespaceID string
}

// FluentSearch creates a new fluent search builder
func (c *Collection) FluentSearch(vector vectortypes.F32) *FluentSearch {
	return &FluentSearch{
		collection: c,
		vector:     vector,
		k:          10, // Default to 10 results
		options: types.SearchOptions{
			IncludeMetadata: true,
		},
	}
}

// WithK sets the number of results to return
func (fs *FluentSearch) WithK(k int) *FluentSearch {
	fs.k = k
	return fs
}

// WithNamespace sets the namespace to search in
func (fs *FluentSearch) WithNamespace(namespace string) *FluentSearch {
	fs.namespaceID = namespace
	return fs
}

// IncludeVectors specifies whether to include vector values in results
func (fs *FluentSearch) IncludeVectors(include bool) *FluentSearch {
	fs.options.IncludeVectors = include
	return fs
}

// IncludeMetadata specifies whether to include metadata in results
func (fs *FluentSearch) IncludeMetadata(include bool) *FluentSearch {
	fs.options.IncludeMetadata = include
	return fs
}

// UseExactSearch enables exact search for higher accuracy
func (fs *FluentSearch) UseExactSearch() *FluentSearch {
	fs.options.ExactSearch = true
	return fs
}

// Filter adds a metadata filter with equals operator
func (fs *FluentSearch) Filter(field string, value interface{}) *FluentSearch {
	fs.filters = append(fs.filters, types.Filter{
		Field:    field,
		Operator: string(Equals),
		Value:    value,
	})
	return fs
}

// FilterNotEquals adds a not-equals filter
func (fs *FluentSearch) FilterNotEquals(field string, value interface{}) *FluentSearch {
	fs.filters = append(fs.filters, types.Filter{
		Field:    field,
		Operator: string(NotEquals),
		Value:    value,
	})
	return fs
}

// FilterGreaterThan adds a greater-than filter
func (fs *FluentSearch) FilterGreaterThan(field string, value interface{}) *FluentSearch {
	fs.filters = append(fs.filters, types.Filter{
		Field:    field,
		Operator: string(GreaterThan),
		Value:    value,
	})
	return fs
}

// FilterLessThan adds a less-than filter
func (fs *FluentSearch) FilterLessThan(field string, value interface{}) *FluentSearch {
	fs.filters = append(fs.filters, types.Filter{
		Field:    field,
		Operator: string(LessThan),
		Value:    value,
	})
	return fs
}

// FilterIn adds an in-list filter
func (fs *FluentSearch) FilterIn(field string, values []interface{}) *FluentSearch {
	fs.filters = append(fs.filters, types.Filter{
		Field:    field,
		Operator: string(In),
		Value:    values,
	})
	return fs
}

// Execute runs the search with the configured parameters
func (fs *FluentSearch) Execute() (types.SearchResponse, error) {
	// Create a search request from the fluent parameters
	request := types.SearchRequest{
		Vector:      fs.vector,
		TopK:        fs.k,
		Filters:     fs.filters,
		Options:     fs.options,
		NamespaceID: fs.namespaceID,
	}

	// Execute the search
	return fs.collection.Search(request)
}

// SetFacetFields sets the fields to be indexed as facets for future vectors
func (c *Collection) SetFacetFields(fields []string) {
	c.Lock()
	defer c.Unlock()

	c.FacetFields = fields

	// Reindex existing vectors' facets
	c.vectorFacets = make(map[string][]facets.FacetValue)

	for id, metadata := range c.Metadata {
		if len(metadata) == 0 {
			continue
		}

		var metadataMap map[string]interface{}
		if err := json.Unmarshal(metadata, &metadataMap); err == nil {
			c.vectorFacets[id] = facets.ExtractFacets(metadataMap, fields)
		}
	}
}

// GetFacetFields returns the fields that are indexed as facets
func (c *Collection) GetFacetFields() []string {
	c.RLock()
	defer c.RUnlock()

	return c.FacetFields
}

// SearchWithFacets searches for vectors similar to the query vector, with optional facet filters.
func (c *Collection) SearchWithFacets(
	query []float32,
	k int,
	filters []facets.Filter,
) ([]SearchResult, error) {
	if k <= 0 {
		return nil, errors.New("k must be positive")
	}
	if len(query) != c.Dimension {
		return nil, fmt.Errorf("query vector dimension mismatch, expected %d, got %d", c.Dimension, len(query))
	}

	c.RLock()
	defer c.RUnlock()

	// If no filters, use the normal index search
	if len(filters) == 0 {
		results, err := c.Index.Search(vectortypes.F32(query), k)
		if err != nil {
			return nil, fmt.Errorf("index search failed: %w", err)
		}

		// Convert types.BasicSearchResult to SearchResult
		searchResults := make([]SearchResult, len(results))
		for i, result := range results {
			searchResults[i] = SearchResult{
				ID:       result.ID,
				Distance: result.Distance,
				Vector:   c.Vectors[result.ID],
				Metadata: c.Metadata[result.ID],
			}
		}

		return searchResults, nil
	}

	// With filters, request the full candidate set so facet filtering preserves
	// the true top-k instead of depending on a heuristic over-fetch multiplier.
	searchK := c.Index.Size()
	if searchK == 0 {
		return []SearchResult{}, nil
	}

	results, err := c.Index.Search(vectortypes.F32(query), searchK)
	if err != nil {
		return nil, fmt.Errorf("index search failed: %w", err)
	}

	var filteredResults []SearchResult
	for _, result := range results {
		if facetValues, exists := c.vectorFacets[result.ID]; exists {
			if facets.MatchesAllFilters(facetValues, filters) {
				filteredResults = append(filteredResults, SearchResult{
					ID:       result.ID,
					Distance: result.Distance,
					Vector:   c.Vectors[result.ID],
					Metadata: c.Metadata[result.ID],
				})
				if len(filteredResults) >= k {
					break
				}
			}
		}
	}

	return filteredResults, nil
}
