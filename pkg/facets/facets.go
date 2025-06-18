// Package facets provides functionality for efficient categorical filtering in vector search
package facets

import (
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"strings"
)

// FilterType represents the type of facet filter
type FilterType string

const (
	// TypeEquality represents an equality filter
	TypeEquality FilterType = "equality"
	// TypeRange represents a range filter
	TypeRange FilterType = "range"
	// TypeSet represents a set membership filter
	TypeSet FilterType = "set"
	// TypeExists represents an existence filter
	TypeExists FilterType = "exists"
)

// Filter is the interface that all facet filters must implement
type Filter interface {
	// Type returns the type of the filter
	Type() FilterType
	// Field returns the field name the filter applies to
	Field() string
	// Match checks if the given facet value matches the filter
	Match(value interface{}) bool
	// String returns a string representation of the filter
	String() string
}

// EqualityFilter is a filter that checks if a facet equals a specific value
type EqualityFilter struct {
	FieldName string
	Value     interface{}
}

// NewEqualityFilter creates a new equality filter
func NewEqualityFilter(field string, value interface{}) *EqualityFilter {
	return &EqualityFilter{
		FieldName: field,
		Value:     value,
	}
}

// Type implements Filter.Type
func (f *EqualityFilter) Type() FilterType {
	return TypeEquality
}

// Field implements Filter.Field
func (f *EqualityFilter) Field() string {
	return f.FieldName
}

// Match implements Filter.Match
func (f *EqualityFilter) Match(value interface{}) bool {
	// Special case for nil values
	if f.Value == nil && value == nil {
		return true
	}
	if f.Value == nil || value == nil {
		return false
	}

	// Handle strings case-insensitively
	if strValue, ok := value.(string); ok {
		if strFilter, ok := f.Value.(string); ok {
			return strings.EqualFold(strValue, strFilter)
		}
	}

	// Use reflect for basic equality comparison
	return reflect.DeepEqual(f.Value, value)
}

// String implements Filter.String
func (f *EqualityFilter) String() string {
	return fmt.Sprintf("%s = %v", f.FieldName, f.Value)
}

// RangeFilter checks if a facet value is within a range
type RangeFilter struct {
	FieldName string
	Min       interface{}
	Max       interface{}
	// IncludeMin specifies whether to include the minimum value in the range
	IncludeMin bool
	// IncludeMax specifies whether to include the maximum value in the range
	IncludeMax bool
}

// NewRangeFilter creates a new range filter
func NewRangeFilter(field string, min, max interface{}, includeMin, includeMax bool) *RangeFilter {
	return &RangeFilter{
		FieldName:  field,
		Min:        min,
		Max:        max,
		IncludeMin: includeMin,
		IncludeMax: includeMax,
	}
}

// Type implements Filter.Type
func (f *RangeFilter) Type() FilterType {
	return TypeRange
}

// Field implements Filter.Field
func (f *RangeFilter) Field() string {
	return f.FieldName
}

// Match implements Filter.Match
func (f *RangeFilter) Match(value interface{}) bool {
	if value == nil {
		return false
	}

	// Only numeric values can be compared with ranges
	switch v := value.(type) {
	case int:
		return f.compareInt(v)
	case int32:
		return f.compareInt(int(v))
	case int64:
		if v > int64(math.MaxInt) || v < int64(math.MinInt) {
			return f.compareFloat(float64(v))
		}
		return f.compareInt(int(v))
	case float32:
		return f.compareFloat(float64(v))
	case float64:
		return f.compareFloat(v)
	default:
		return false
	}
}

// compareInt checks if an int value is within the range
func (f *RangeFilter) compareInt(value int) bool {
	var minOK, maxOK bool

	if f.Min == nil {
		minOK = true
	} else {
		switch min := f.Min.(type) {
		case int:
			minOK = (f.IncludeMin && value >= min) || (!f.IncludeMin && value > min)
		case int32:
			minOK = (f.IncludeMin && value >= int(min)) || (!f.IncludeMin && value > int(min))
		case int64:
			minOK = (f.IncludeMin && value >= int(min)) || (!f.IncludeMin && value > int(min))
		case float32:
			minOK = (f.IncludeMin && float64(value) >= float64(min)) || (!f.IncludeMin && float64(value) > float64(min))
		case float64:
			minOK = (f.IncludeMin && float64(value) >= min) || (!f.IncludeMin && float64(value) > min)
		default:
			minOK = false
		}
	}

	if f.Max == nil {
		maxOK = true
	} else {
		switch max := f.Max.(type) {
		case int:
			maxOK = (f.IncludeMax && value <= max) || (!f.IncludeMax && value < max)
		case int32:
			maxOK = (f.IncludeMax && value <= int(max)) || (!f.IncludeMax && value < int(max))
		case int64:
			maxOK = (f.IncludeMax && value <= int(max)) || (!f.IncludeMax && value < int(max))
		case float32:
			maxOK = (f.IncludeMax && float64(value) <= float64(max)) || (!f.IncludeMax && float64(value) < float64(max))
		case float64:
			maxOK = (f.IncludeMax && float64(value) <= max) || (!f.IncludeMax && float64(value) < max)
		default:
			maxOK = false
		}
	}

	return minOK && maxOK
}

// compareFloat checks if a float value is within the range
func (f *RangeFilter) compareFloat(value float64) bool {
	var minOK, maxOK bool

	if f.Min == nil {
		minOK = true
	} else {
		switch min := f.Min.(type) {
		case int:
			minOK = (f.IncludeMin && value >= float64(min)) || (!f.IncludeMin && value > float64(min))
		case int32:
			minOK = (f.IncludeMin && value >= float64(min)) || (!f.IncludeMin && value > float64(min))
		case int64:
			minOK = (f.IncludeMin && value >= float64(min)) || (!f.IncludeMin && value > float64(min))
		case float32:
			minOK = (f.IncludeMin && value >= float64(min)) || (!f.IncludeMin && value > float64(min))
		case float64:
			minOK = (f.IncludeMin && value >= min) || (!f.IncludeMin && value > min)
		default:
			minOK = false
		}
	}

	if f.Max == nil {
		maxOK = true
	} else {
		switch max := f.Max.(type) {
		case int:
			maxOK = (f.IncludeMax && value <= float64(max)) || (!f.IncludeMax && value < float64(max))
		case int32:
			maxOK = (f.IncludeMax && value <= float64(max)) || (!f.IncludeMax && value < float64(max))
		case int64:
			maxOK = (f.IncludeMax && value <= float64(max)) || (!f.IncludeMax && value < float64(max))
		case float32:
			maxOK = (f.IncludeMax && value <= float64(max)) || (!f.IncludeMax && value < float64(max))
		case float64:
			maxOK = (f.IncludeMax && value <= max) || (!f.IncludeMax && value < max)
		default:
			maxOK = false
		}
	}

	return minOK && maxOK
}

// String implements Filter.String
func (f *RangeFilter) String() string {
	minStr := "∞"
	if f.Min != nil {
		minStr = fmt.Sprintf("%v", f.Min)
	}
	maxStr := "∞"
	if f.Max != nil {
		maxStr = fmt.Sprintf("%v", f.Max)
	}

	leftBracket := "("
	if f.IncludeMin {
		leftBracket = "["
	}
	rightBracket := ")"
	if f.IncludeMax {
		rightBracket = "]"
	}

	return fmt.Sprintf("%s %s%s, %s%s", f.FieldName, leftBracket, minStr, maxStr, rightBracket)
}

// SetFilter checks if a facet value is in a set of allowed values
type SetFilter struct {
	FieldName string
	Values    []interface{}
}

// NewSetFilter creates a new set filter
func NewSetFilter(field string, values []interface{}) *SetFilter {
	return &SetFilter{
		FieldName: field,
		Values:    values,
	}
}

// Type implements Filter.Type
func (f *SetFilter) Type() FilterType {
	return TypeSet
}

// Field implements Filter.Field
func (f *SetFilter) Field() string {
	return f.FieldName
}

// Match implements Filter.Match
func (f *SetFilter) Match(value interface{}) bool {
	if value == nil {
		return false
	}

	// Handle string case-insensitive comparison
	if strValue, ok := value.(string); ok {
		for _, v := range f.Values {
			if strFilter, ok := v.(string); ok {
				if strings.EqualFold(strValue, strFilter) {
					return true
				}
			} else if reflect.DeepEqual(value, v) {
				return true
			}
		}
		return false
	}

	// Handle array/slice case - check if any element matches
	if reflect.TypeOf(value).Kind() == reflect.Slice || reflect.TypeOf(value).Kind() == reflect.Array {
		valueSlice := reflect.ValueOf(value)
		for i := 0; i < valueSlice.Len(); i++ {
			item := valueSlice.Index(i).Interface()
			for _, v := range f.Values {
				if reflect.DeepEqual(item, v) {
					return true
				}
			}
		}
		return false
	}

	// Regular equality check
	for _, v := range f.Values {
		if reflect.DeepEqual(value, v) {
			return true
		}
	}
	return false
}

// String implements Filter.String
func (f *SetFilter) String() string {
	values := make([]string, len(f.Values))
	for i, v := range f.Values {
		values[i] = fmt.Sprintf("%v", v)
	}
	return fmt.Sprintf("%s IN [%s]", f.FieldName, strings.Join(values, ", "))
}

// ExistsFilter checks if a facet field exists and is not null/empty
type ExistsFilter struct {
	FieldName   string
	ShouldExist bool
}

// NewExistsFilter creates a new exists filter
func NewExistsFilter(field string, shouldExist bool) *ExistsFilter {
	return &ExistsFilter{
		FieldName:   field,
		ShouldExist: shouldExist,
	}
}

// Type implements Filter.Type
func (f *ExistsFilter) Type() FilterType {
	return TypeExists
}

// Field implements Filter.Field
func (f *ExistsFilter) Field() string {
	return f.FieldName
}

// Match implements Filter.Match
func (f *ExistsFilter) Match(value interface{}) bool {
	exists := value != nil

	// For empty strings, arrays, etc.
	if exists {
		v := reflect.ValueOf(value)
		switch v.Kind() {
		case reflect.String:
			exists = v.Len() > 0
		case reflect.Slice, reflect.Array, reflect.Map:
			exists = v.Len() > 0
		}
	}

	return exists == f.ShouldExist
}

// String implements Filter.String
func (f *ExistsFilter) String() string {
	if f.ShouldExist {
		return fmt.Sprintf("%s EXISTS", f.FieldName)
	}
	return fmt.Sprintf("%s NOT EXISTS", f.FieldName)
}

// FacetValue represents a single facet value stored with a vector
type FacetValue struct {
	Field string      `json:"field"`
	Value interface{} `json:"value"`
}

// ExtractFacets extracts facet values from metadata
func ExtractFacets(metadata map[string]interface{}, facetFields []string) []FacetValue {
	if len(facetFields) == 0 || metadata == nil {
		return nil
	}

	result := make([]FacetValue, 0, len(facetFields))

	for _, field := range facetFields {
		// Handle nested fields with dot notation
		parts := strings.Split(field, ".")
		var value interface{} = metadata

		for _, part := range parts {
			if m, ok := value.(map[string]interface{}); ok {
				if v, exists := m[part]; exists {
					value = v
				} else {
					value = nil
					break
				}
			} else {
				value = nil
				break
			}
		}

		if value != nil {
			result = append(result, FacetValue{Field: field, Value: value})
		}
	}

	return result
}

// MatchesAllFilters checks if a set of facets matches all provided filters
func MatchesAllFilters(facets []FacetValue, filters []Filter) bool {
	if len(filters) == 0 {
		return true
	}

	if len(facets) == 0 {
		return false
	}

	// Create a map for O(1) field lookups
	facetMap := make(map[string]interface{}, len(facets))
	for _, facet := range facets {
		facetMap[facet.Field] = facet.Value
	}

	// Check each filter
	for _, filter := range filters {
		value, exists := facetMap[filter.Field()]
		if !exists && filter.Type() != TypeExists {
			return false
		}
		if !filter.Match(value) {
			return false
		}
	}

	return true
}

// FacetsFromJSON extracts facet values from JSON metadata
func FacetsFromJSON(metadataJSON json.RawMessage, facetFields []string) ([]FacetValue, error) {
	if len(metadataJSON) == 0 || len(facetFields) == 0 {
		return nil, nil
	}

	var metadata map[string]interface{}
	if err := json.Unmarshal(metadataJSON, &metadata); err != nil {
		return nil, fmt.Errorf("failed to parse metadata JSON: %w", err)
	}

	return ExtractFacets(metadata, facetFields), nil
}
