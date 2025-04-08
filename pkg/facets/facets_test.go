package facets

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestEqualityFilter(t *testing.T) {
	tests := []struct {
		name     string
		field    string
		value    interface{}
		testVal  interface{}
		expected bool
	}{
		{"String equality", "category", "electronics", "electronics", true},
		{"String case insensitive", "category", "Electronics", "electroNICS", true},
		{"Int equality", "count", 42, 42, true},
		{"Float equality", "price", 99.99, 99.99, true},
		{"Bool equality", "active", true, true, true},
		{"String inequality", "category", "electronics", "clothing", false},
		{"Int inequality", "count", 42, 43, false},
		{"Float inequality", "price", 99.99, 100.0, false},
		{"Bool inequality", "active", true, false, false},
		{"Nil values", "field", nil, nil, true},
		{"Left nil", "field", nil, "value", false},
		{"Right nil", "field", "value", nil, false},
		{"Different types", "field", 42, "42", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			filter := NewEqualityFilter(tt.field, tt.value)
			if filter.Type() != TypeEquality {
				t.Errorf("Expected filter type %s, got %s", TypeEquality, filter.Type())
			}
			if filter.Field() != tt.field {
				t.Errorf("Expected field name %s, got %s", tt.field, filter.Field())
			}
			if result := filter.Match(tt.testVal); result != tt.expected {
				t.Errorf("Match() = %v, expected %v", result, tt.expected)
			}
			if filter.String() == "" {
				t.Error("String() returned empty string")
			}
		})
	}
}

func TestRangeFilter(t *testing.T) {
	tests := []struct {
		name       string
		field      string
		min        interface{}
		max        interface{}
		includeMin bool
		includeMax bool
		testVal    interface{}
		expected   bool
	}{
		{"Int in range", "price", 10, 100, true, true, 50, true},
		{"Int at min inclusive", "price", 10, 100, true, true, 10, true},
		{"Int at min exclusive", "price", 10, 100, false, true, 10, false},
		{"Int at max inclusive", "price", 10, 100, true, true, 100, true},
		{"Int at max exclusive", "price", 10, 100, true, false, 100, false},
		{"Int below range", "price", 10, 100, true, true, 5, false},
		{"Int above range", "price", 10, 100, true, true, 150, false},
		{"Float in range", "price", 10.5, 100.5, true, true, 50.5, true},
		{"Mixed types", "price", 10, 100.5, true, true, 50, true},
		{"No min", "price", nil, 100, true, true, 50, true},
		{"No min edge", "price", nil, 100, true, true, 100, true},
		{"No max", "price", 10, nil, true, true, 5000, true},
		{"No max edge", "price", 10, nil, true, true, 10, true},
		{"Non-numeric value", "price", 10, 100, true, true, "price", false},
		{"Nil value", "price", 10, 100, true, true, nil, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			filter := NewRangeFilter(tt.field, tt.min, tt.max, tt.includeMin, tt.includeMax)
			if filter.Type() != TypeRange {
				t.Errorf("Expected filter type %s, got %s", TypeRange, filter.Type())
			}
			if filter.Field() != tt.field {
				t.Errorf("Expected field name %s, got %s", tt.field, filter.Field())
			}
			if result := filter.Match(tt.testVal); result != tt.expected {
				t.Errorf("Match() = %v, expected %v", result, tt.expected)
			}
			if filter.String() == "" {
				t.Error("String() returned empty string")
			}
		})
	}

	// Test various numeric type combinations
	intVal := 50
	int32Val := int32(50)
	int64Val := int64(50)
	float32Val := float32(50.0)
	float64Val := 50.0

	numericFilter := NewRangeFilter("num", 10, 100, true, true)
	if !numericFilter.Match(intVal) {
		t.Error("Failed to match int value")
	}
	if !numericFilter.Match(int32Val) {
		t.Error("Failed to match int32 value")
	}
	if !numericFilter.Match(int64Val) {
		t.Error("Failed to match int64 value")
	}
	if !numericFilter.Match(float32Val) {
		t.Error("Failed to match float32 value")
	}
	if !numericFilter.Match(float64Val) {
		t.Error("Failed to match float64 value")
	}
}

func TestSetFilter(t *testing.T) {
	tests := []struct {
		name     string
		field    string
		values   []interface{}
		testVal  interface{}
		expected bool
	}{
		{"String in set", "category", []interface{}{"electronics", "clothing"}, "electronics", true},
		{"String in set case insensitive", "category", []interface{}{"Electronics", "Clothing"}, "electronics", true},
		{"String not in set", "category", []interface{}{"electronics", "clothing"}, "furniture", false},
		{"Int in set", "count", []interface{}{1, 2, 3}, 2, true},
		{"Int not in set", "count", []interface{}{1, 2, 3}, 4, false},
		{"Mixed types in set", "value", []interface{}{"one", 2, true}, 2, true},
		{"Mixed types not in set", "value", []interface{}{"one", 2, true}, false, false},
		{"Array with matching element", "tags", []interface{}{"tag1", "tag2"}, []string{"tag0", "tag1", "tag3"}, true},
		{"Array without matching element", "tags", []interface{}{"tag1", "tag2"}, []string{"tag3", "tag4"}, false},
		{"Nil value", "field", []interface{}{"one", "two"}, nil, false},
		{"Empty set", "field", []interface{}{}, "value", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			filter := NewSetFilter(tt.field, tt.values)
			if filter.Type() != TypeSet {
				t.Errorf("Expected filter type %s, got %s", TypeSet, filter.Type())
			}
			if filter.Field() != tt.field {
				t.Errorf("Expected field name %s, got %s", tt.field, filter.Field())
			}
			if result := filter.Match(tt.testVal); result != tt.expected {
				t.Errorf("Match() = %v, expected %v", result, tt.expected)
			}
			if filter.String() == "" {
				t.Error("String() returned empty string")
			}
		})
	}
}

func TestExistsFilter(t *testing.T) {
	tests := []struct {
		name        string
		field       string
		shouldExist bool
		testVal     interface{}
		expected    bool
	}{
		{"Field exists", "category", true, "electronics", true},
		{"Field doesn't exist", "category", false, nil, true},
		{"Field should exist but doesn't", "category", true, nil, false},
		{"Field shouldn't exist but does", "category", false, "electronics", false},
		{"Empty string exists", "category", true, "", false},
		{"Empty array exists", "tags", true, []string{}, false},
		{"Empty map exists", "metadata", true, map[string]string{}, false},
		{"Non-empty array exists", "tags", true, []string{"tag1"}, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			filter := NewExistsFilter(tt.field, tt.shouldExist)
			if filter.Type() != TypeExists {
				t.Errorf("Expected filter type %s, got %s", TypeExists, filter.Type())
			}
			if filter.Field() != tt.field {
				t.Errorf("Expected field name %s, got %s", tt.field, filter.Field())
			}
			if result := filter.Match(tt.testVal); result != tt.expected {
				t.Errorf("Match() = %v, expected %v", result, tt.expected)
			}
			if filter.String() == "" {
				t.Error("String() returned empty string")
			}
		})
	}
}

func TestExtractFacets(t *testing.T) {
	metadata := map[string]interface{}{
		"category": "electronics",
		"price":    299.99,
		"active":   true,
		"tags":     []string{"smartphone", "android", "5G"},
		"specs": map[string]interface{}{
			"ram":   "8GB",
			"color": "black",
			"dimensions": map[string]interface{}{
				"height": 150,
				"width":  75,
			},
		},
	}

	tests := []struct {
		name        string
		facetFields []string
		expected    []FacetValue
	}{
		{
			"Single field",
			[]string{"category"},
			[]FacetValue{
				{Field: "category", Value: "electronics"},
			},
		},
		{
			"Multiple fields",
			[]string{"category", "price", "active"},
			[]FacetValue{
				{Field: "category", Value: "electronics"},
				{Field: "price", Value: 299.99},
				{Field: "active", Value: true},
			},
		},
		{
			"Array field",
			[]string{"tags"},
			[]FacetValue{
				{Field: "tags", Value: []string{"smartphone", "android", "5G"}},
			},
		},
		{
			"Nested field",
			[]string{"specs.ram"},
			[]FacetValue{
				{Field: "specs.ram", Value: "8GB"},
			},
		},
		{
			"Deeply nested field",
			[]string{"specs.dimensions.height"},
			[]FacetValue{
				{Field: "specs.dimensions.height", Value: 150},
			},
		},
		{
			"Mix of fields",
			[]string{"category", "specs.ram", "specs.dimensions.width"},
			[]FacetValue{
				{Field: "category", Value: "electronics"},
				{Field: "specs.ram", Value: "8GB"},
				{Field: "specs.dimensions.width", Value: 75},
			},
		},
		{
			"Non-existent field",
			[]string{"invalid_field"},
			[]FacetValue{},
		},
		{
			"Mix with non-existent field",
			[]string{"category", "invalid_field"},
			[]FacetValue{
				{Field: "category", Value: "electronics"},
			},
		},
		{
			"Empty fields list",
			[]string{},
			nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ExtractFacets(metadata, tt.facetFields)
			if len(result) != len(tt.expected) {
				t.Errorf("ExtractFacets() returned %d facets, expected %d", len(result), len(tt.expected))
				return
			}

			for _, expected := range tt.expected {
				found := false
				for _, actual := range result {
					if actual.Field == expected.Field && reflect.DeepEqual(actual.Value, expected.Value) {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("Expected facet %+v not found in result", expected)
				}
			}
		})
	}

	// Test with nil metadata
	if facets := ExtractFacets(nil, []string{"field"}); facets != nil {
		t.Error("ExtractFacets() with nil metadata should return nil")
	}
}

func TestMatchesAllFilters(t *testing.T) {
	facets := []FacetValue{
		{Field: "category", Value: "electronics"},
		{Field: "price", Value: 299.99},
		{Field: "active", Value: true},
		{Field: "tags", Value: []string{"smartphone", "android", "5G"}},
	}

	tests := []struct {
		name    string
		filters []Filter
		result  bool
	}{
		{
			"No filters",
			[]Filter{},
			true,
		},
		{
			"Single matching filter",
			[]Filter{NewEqualityFilter("category", "electronics")},
			true,
		},
		{
			"Single non-matching filter",
			[]Filter{NewEqualityFilter("category", "clothing")},
			false,
		},
		{
			"Multiple matching filters",
			[]Filter{
				NewEqualityFilter("category", "electronics"),
				NewRangeFilter("price", 200.0, 300.0, true, true),
				NewEqualityFilter("active", true),
			},
			true,
		},
		{
			"Mix of matching and non-matching filters",
			[]Filter{
				NewEqualityFilter("category", "electronics"),
				NewRangeFilter("price", 100.0, 200.0, true, true),
			},
			false,
		},
		{
			"Set filter matching",
			[]Filter{NewSetFilter("category", []interface{}{"clothing", "electronics", "furniture"})},
			true,
		},
		{
			"Set filter non-matching",
			[]Filter{NewSetFilter("category", []interface{}{"clothing", "furniture"})},
			false,
		},
		{
			"Exists filter matching",
			[]Filter{NewExistsFilter("tags", true)},
			true,
		},
		{
			"Exists filter non-matching",
			[]Filter{NewExistsFilter("nonexistent", true)},
			false,
		},
		{
			"Not exists filter matching",
			[]Filter{NewExistsFilter("nonexistent", false)},
			true,
		},
		{
			"Not exists filter non-matching",
			[]Filter{NewExistsFilter("category", false)},
			false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := MatchesAllFilters(facets, tt.filters)
			if result != tt.result {
				t.Errorf("MatchesAllFilters() = %v, expected %v", result, tt.result)
			}
		})
	}

	// Test with empty facets
	if MatchesAllFilters([]FacetValue{}, []Filter{NewEqualityFilter("field", "value")}) {
		t.Error("MatchesAllFilters() with empty facets should return false when filters are present")
	}
}

func TestFacetsFromJSON(t *testing.T) {
	jsonMetadata := []byte(`{
		"category": "electronics",
		"price": 299.99,
		"active": true,
		"tags": ["smartphone", "android", "5G"],
		"specs": {
			"ram": "8GB",
			"color": "black",
			"dimensions": {
				"height": 150,
				"width": 75
			}
		}
	}`)

	tests := []struct {
		name        string
		facetFields []string
		expected    int
	}{
		{"Single field", []string{"category"}, 1},
		{"Multiple fields", []string{"category", "price", "active"}, 3},
		{"Nested field", []string{"specs.ram"}, 1},
		{"Deep nested field", []string{"specs.dimensions.height"}, 1},
		{"Mix of fields", []string{"category", "specs.ram", "specs.dimensions.width"}, 3},
		{"Non-existent field", []string{"invalid"}, 0},
		{"Empty fields", []string{}, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			facets, err := FacetsFromJSON(jsonMetadata, tt.facetFields)
			if err != nil {
				t.Errorf("FacetsFromJSON() returned error: %v", err)
				return
			}
			if len(facets) != tt.expected {
				t.Errorf("FacetsFromJSON() returned %d facets, expected %d", len(facets), tt.expected)
			}
		})
	}

	// Test with invalid JSON
	_, err := FacetsFromJSON([]byte(`invalid json`), []string{"field"})
	if err == nil {
		t.Error("FacetsFromJSON() with invalid JSON should return error")
	}

	// Test with empty JSON
	facets, err := FacetsFromJSON([]byte{}, []string{"field"})
	if err != nil || facets != nil {
		t.Error("FacetsFromJSON() with empty JSON should return nil facets and no error")
	}
}

func TestFacetValue_JSONRoundTrip(t *testing.T) {
	// Test serialization and deserialization of FacetValue
	original := []FacetValue{
		{Field: "category", Value: "electronics"},
		{Field: "price", Value: 299.99},
		{Field: "active", Value: true},
		{Field: "count", Value: 42},
	}

	data, err := json.Marshal(original)
	if err != nil {
		t.Fatalf("Failed to marshal facet values: %v", err)
	}

	var decoded []FacetValue
	err = json.Unmarshal(data, &decoded)
	if err != nil {
		t.Fatalf("Failed to unmarshal facet values: %v", err)
	}

	if len(decoded) != len(original) {
		t.Errorf("Decoded facet count mismatch: got %d, want %d", len(decoded), len(original))
	}

	for i, original := range original {
		if decoded[i].Field != original.Field {
			t.Errorf("Decoded facet field mismatch at index %d: got %s, want %s",
				i, decoded[i].Field, original.Field)
		}

		// Note: JSON unmarshaling converts numbers to float64
		switch original.Value.(type) {
		case int, int32, int64:
			if decoded[i].Value.(float64) != float64(original.Value.(int)) {
				t.Errorf("Decoded facet value mismatch at index %d", i)
			}
		default:
			if !reflect.DeepEqual(decoded[i].Value, original.Value) {
				t.Errorf("Decoded facet value mismatch at index %d: got %v, want %v",
					i, decoded[i].Value, original.Value)
			}
		}
	}
}
