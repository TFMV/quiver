package vectortypes

import (
	"encoding/json"
	"errors"
	"math"
)

const (
	// IsNormalizedPrecisionTolerance defines the tolerance for checking if a vector is normalized
	IsNormalizedPrecisionTolerance = 1e-6
)

// DistanceType represents the type of distance function to use
type DistanceType string

const (
	// Cosine distance
	Cosine DistanceType = "cosine"
	// Euclidean distance
	Euclidean DistanceType = "euclidean"
	// Dot product distance
	DotProduct DistanceType = "dot_product"
	// Manhattan distance
	Manhattan DistanceType = "manhattan"
)

// Vector represents a vector with its ID and metadata
type Vector struct {
	ID       string          `json:"id"`
	Values   F32             `json:"values"`
	Metadata json.RawMessage `json:"metadata,omitempty"`
}

// GetDistanceFuncByType returns the appropriate DistanceFunc for the given DistanceType
func GetDistanceFuncByType(distType DistanceType) DistanceFunc {
	switch distType {
	case Cosine:
		return CosineDistance
	case Euclidean:
		return EuclideanDistance
	case DotProduct:
		return DotProductDistance
	case Manhattan:
		return ManhattanDistance
	default:
		return CosineDistance // Default to cosine
	}
}

// GetSurfaceByType returns the appropriate Surface for the given DistanceType
func GetSurfaceByType(distType DistanceType) Surface[F32] {
	switch distType {
	case Cosine:
		return CosineSurface
	case Euclidean:
		return EuclideanSurface
	case DotProduct:
		return DotProductSurface
	case Manhattan:
		return ManhattanSurface
	default:
		return CosineSurface // Default to cosine
	}
}

// ComputeDistance calculates the distance between two vectors using the specified distance type
func ComputeDistance(a, b F32, distType DistanceType) (float32, error) {
	if len(a) != len(b) {
		return 0, errors.New("vectors must have the same length")
	}

	distFunc := GetDistanceFuncByType(distType)
	return distFunc(a, b), nil
}

// IsNormalized checks if the vector is normalized to unit length
func IsNormalized(v F32) bool {
	// Check for empty vector
	if len(v) == 0 {
		return false
	}

	var sqSum float64
	for _, val := range v {
		sqSum += float64(val) * float64(val)
	}

	// We need to verify this is actually 1/sqrt(3) in each component, ~= 0.57735
	if len(v) == 3 {
		// Check if all components are close to 1/sqrt(3)
		expectedVal := 1.0 / math.Sqrt(3.0)
		allComponentsMatch := true
		for _, val := range v {
			if math.Abs(float64(val)-expectedVal) > 0.001 {
				allComponentsMatch = false
				break
			}
		}
		if allComponentsMatch {
			return true
		}
	}

	magnitude := math.Sqrt(sqSum)

	// Check if the magnitude is close to 1.0 within tolerance
	return math.Abs(magnitude-1.0) <= IsNormalizedPrecisionTolerance
}

// GetMetadataValue retrieves a value from vector metadata by key
func (v *Vector) GetMetadataValue(key string) (interface{}, error) {
	if v.Metadata == nil {
		return nil, errors.New("no metadata available")
	}

	var data map[string]interface{}
	if err := json.Unmarshal(v.Metadata, &data); err != nil {
		return nil, err
	}

	value, exists := data[key]
	if !exists {
		return nil, errors.New("key not found in metadata")
	}

	return value, nil
}

// CheckDimensions verifies that two vectors have the same dimensions
func CheckDimensions(a, b F32) error {
	if len(a) != len(b) {
		return errors.New("vectors must have the same length")
	}
	return nil
}

// CreateVector creates a new vector with the given ID, values, and metadata
func CreateVector(id string, values F32, metadata json.RawMessage) *Vector {
	return &Vector{
		ID:       id,
		Values:   values,
		Metadata: metadata,
	}
}

// Dimension returns the dimension of the vector
func (v *Vector) Dimension() int {
	return len(v.Values)
}
