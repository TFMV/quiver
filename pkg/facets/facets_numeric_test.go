package facets

import (
	"testing"
)

func TestMatch_NumericJSON(t *testing.T) {
	// JSON unmarshals numbers to float64, while a user might create an EqualityFilter with int
	f := NewEqualityFilter("age", 42)

	// Simulate JSON unmarshaled float64
	var jsonValue interface{} = float64(42)

	if !f.Match(jsonValue) {
		t.Errorf("Expected EqualityFilter.Match to handle numeric type differences (int vs float64)")
	}

	// Test SetFilter
	sf := NewSetFilter("age", []interface{}{42, 43})
	if !sf.Match(jsonValue) {
		t.Errorf("Expected SetFilter.Match to handle numeric type differences")
	}
}
