package persistence

import (
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/TFMV/quiver/pkg/facets"
	"github.com/TFMV/quiver/pkg/vectortypes"
)

// Collection implements a basic vector collection that can be persisted.
// This serves as a reference implementation of the Persistable interface.
type Collection struct {
	// Basic information
	name      string
	dimension int

	// Vectors and metadata
	vectors  map[string][]float32
	metadata map[string]map[string]string

	// Distance function
	distanceFunc vectortypes.DistanceFunc

	// Mutex for thread safety
	mu sync.RWMutex

	// Whether the collection has been modified since last save
	dirty bool

	// When the collection was created
	createdAt time.Time

	// Facet fields and values
	facetFields  []string
	vectorFacets map[string][]facets.FacetValue
}

// NewCollection creates a new persistable collection
func NewCollection(name string, dimension int, distanceFunc vectortypes.DistanceFunc) *Collection {
	return &Collection{
		name:         name,
		dimension:    dimension,
		vectors:      make(map[string][]float32),
		metadata:     make(map[string]map[string]string),
		distanceFunc: distanceFunc,
		mu:           sync.RWMutex{},
		dirty:        false,
		createdAt:    time.Now(),
		facetFields:  []string{},
		vectorFacets: make(map[string][]facets.FacetValue),
	}
}

// GetName implements the Persistable interface
func (c *Collection) GetName() string {
	return c.name
}

// GetDimension implements the Persistable interface
func (c *Collection) GetDimension() int {
	return c.dimension
}

// GetVectors implements the Persistable interface
func (c *Collection) GetVectors() []VectorRecord {
	c.mu.RLock()
	defer c.mu.RUnlock()

	records := make([]VectorRecord, 0, len(c.vectors))
	for id, vector := range c.vectors {
		vecCopy := make([]float32, len(vector))
		copy(vecCopy, vector)
		record := VectorRecord{
			ID:     id,
			Vector: vecCopy,
		}

		if meta, ok := c.metadata[id]; ok {
			metaCopy := make(map[string]string, len(meta))
			for k, v := range meta {
				metaCopy[k] = v
			}
			record.Metadata = metaCopy
		}

		records = append(records, record)
	}

	return records
}

// AddVector implements the Persistable interface
func (c *Collection) AddVector(id string, vector []float32, metadata map[string]string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Validate vector dimension
	if len(vector) != c.dimension {
		return fmt.Errorf("vector dimension mismatch: got %d, expected %d", len(vector), c.dimension)
	}

	// Store a copy of the vector to prevent external modifications
	vecCopy := make([]float32, len(vector))
	copy(vecCopy, vector)
	c.vectors[id] = vecCopy

	// Store metadata if provided
	if metadata != nil {
		metaCopy := make(map[string]string, len(metadata))
		for k, v := range metadata {
			metaCopy[k] = v
		}
		c.metadata[id] = metaCopy

		// Extract and store facets if facet fields are defined
		if len(c.facetFields) > 0 {
			// Convert string map to interface map for facet extraction
			metadataMap := make(map[string]interface{}, len(metadata))
			for k, v := range metadata {
				metadataMap[k] = v
			}
			c.vectorFacets[id] = facets.ExtractFacets(metadataMap, c.facetFields)
		}
	}

	// Mark as dirty
	c.dirty = true

	return nil
}

// DeleteVector removes a vector from the collection
func (c *Collection) DeleteVector(id string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Check if vector exists
	if _, exists := c.vectors[id]; !exists {
		return fmt.Errorf("vector with ID %s not found", id)
	}

	// Delete vector
	delete(c.vectors, id)

	// Delete metadata if exists
	delete(c.metadata, id)

	// Delete facets if exists
	delete(c.vectorFacets, id)

	// Mark as dirty
	c.dirty = true

	return nil
}

// GetVector retrieves a vector by ID
func (c *Collection) GetVector(id string) ([]float32, map[string]string, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	// Check if vector exists
	vector, exists := c.vectors[id]
	if !exists {
		return nil, nil, fmt.Errorf("vector with ID %s not found", id)
	}

	vecCopy := make([]float32, len(vector))
	copy(vecCopy, vector)

	meta, ok := c.metadata[id]
	var metaCopy map[string]string
	if ok {
		metaCopy = make(map[string]string, len(meta))
		for k, v := range meta {
			metaCopy[k] = v
		}
	}

	return vecCopy, metaCopy, nil
}

// IsDirty returns whether the collection has been modified since last save
func (c *Collection) IsDirty() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.dirty
}

// MarkClean marks the collection as clean (not dirty)
func (c *Collection) MarkClean() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.dirty = false
}

// Search finds the most similar vectors to a query vector
func (c *Collection) Search(query []float32, limit int) ([]SearchResult, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.distanceFunc == nil {
		return nil, fmt.Errorf("distance function is not set")
	}
	if query == nil {
		return nil, fmt.Errorf("query vector is nil")
	}

	// Validate query vector dimension
	if len(query) != c.dimension {
		return nil, fmt.Errorf("query vector dimension mismatch: got %d, expected %d", len(query), c.dimension)
	}

	// Calculate distances
	results := make([]SearchResult, 0, len(c.vectors))
	for id, vector := range c.vectors {
		distance := c.distanceFunc(query, vector)
		results = append(results, SearchResult{
			ID:       id,
			Distance: distance,
		})
	}

	// Sort by distance (nearest first)
	SortSearchResults(results)

	// Limit results
	if limit > 0 && limit < len(results) {
		results = results[:limit]
	}

	return results, nil
}

// SearchResult represents a search result with ID and distance
type SearchResult struct {
	ID       string
	Distance float32
}

// SortSearchResults sorts search results by distance (ascending)
func SortSearchResults(results []SearchResult) {
	sort.Slice(results, func(i, j int) bool {
		return results[i].Distance < results[j].Distance
	})
}

// Count returns the number of vectors in the collection
func (c *Collection) Count() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.vectors)
}

// SetFacetFields sets the fields to be indexed as facets for future vectors
func (c *Collection) SetFacetFields(fields []string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.facetFields = fields

	// Reindex existing vectors' facets
	c.vectorFacets = make(map[string][]facets.FacetValue)

	for id, metadata := range c.metadata {
		// Convert string map to interface map for facet extraction
		metadataMap := make(map[string]interface{})
		for k, v := range metadata {
			metadataMap[k] = v
		}
		c.vectorFacets[id] = facets.ExtractFacets(metadataMap, fields)
	}

	// Mark as dirty
	c.dirty = true
}

// GetFacetFields returns the fields that are indexed as facets
func (c *Collection) GetFacetFields() []string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.facetFields
}

// GetVectorFacets returns the facet values for a specific vector
func (c *Collection) GetVectorFacets(id string) ([]facets.FacetValue, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	facetValues, exists := c.vectorFacets[id]
	return facetValues, exists
}

// SearchWithFacets searches for vectors similar to the query vector, with optional facet filters
func (c *Collection) SearchWithFacets(query []float32, limit int, filters []facets.Filter) ([]SearchResult, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.distanceFunc == nil {
		return nil, fmt.Errorf("distance function is not set")
	}
	if query == nil {
		return nil, fmt.Errorf("query vector is nil")
	}

	// Validate query vector dimension
	if len(query) != c.dimension {
		return nil, fmt.Errorf("query vector dimension mismatch: got %d, expected %d", len(query), c.dimension)
	}

	// If no filters, use the normal search
	if len(filters) == 0 {
		return c.Search(query, limit)
	}

	// Calculate distances
	results := make([]SearchResult, 0, len(c.vectors))
	for id, vector := range c.vectors {
		// Check if vector passes facet filters
		if facetValues, exists := c.vectorFacets[id]; exists {
			if !facets.MatchesAllFilters(facetValues, filters) {
				continue
			}
		} else {
			// Skip vectors without facet values
			continue
		}

		// Calculate distance for vectors that passed filters
		distance := c.distanceFunc(query, vector)
		results = append(results, SearchResult{
			ID:       id,
			Distance: distance,
		})
	}

	// Sort by distance (nearest first)
	SortSearchResults(results)

	// Limit results
	if limit > 0 && limit < len(results) {
		results = results[:limit]
	}

	return results, nil
}
