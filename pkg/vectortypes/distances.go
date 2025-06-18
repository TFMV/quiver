package vectortypes

import (
	"fmt"
	"math"
)

// Standard distance functions for vector similarity

// CosineDistance calculates the cosine distance between vectors
// Lower value means more similar vectors (0 being identical)
func CosineDistance(a, b F32) float32 {
	if len(a) != len(b) {
		panic("vectors must have the same length")
	}

	var dotProduct, magnitudeA, magnitudeB float64
	for i := 0; i < len(a); i++ {
		dotProduct += float64(a[i]) * float64(b[i])
		magnitudeA += float64(a[i]) * float64(a[i])
		magnitudeB += float64(b[i]) * float64(b[i])
	}

	// Guard against divide-by-zero
	if magnitudeA == 0 || magnitudeB == 0 {
		return 0
	}

	// Compute cosine similarity and convert to cosine distance
	similarity := dotProduct / (math.Sqrt(magnitudeA) * math.Sqrt(magnitudeB))
	// Clamp similarity to [-1, 1] to account for floating point errors
	if similarity > 1.0 {
		similarity = 1.0
	} else if similarity < -1.0 {
		similarity = -1.0
	}

	// Distance = 1 - similarity
	return float32(1.0 - similarity)
}

// EuclideanDistance calculates the Euclidean distance between vectors
func EuclideanDistance(a, b F32) float32 {
	if len(a) != len(b) {
		panic("vectors must have the same length")
	}

	var sum float64
	for i := 0; i < len(a); i++ {
		diff := float64(a[i] - b[i])
		sum += diff * diff
	}

	return float32(math.Sqrt(sum))
}

// SquaredEuclideanDistance calculates the squared Euclidean distance between vectors
// This avoids the final square root which can be useful in comparisons where only
// relative ordering matters.
func SquaredEuclideanDistance(a, b F32) float32 {
	if len(a) != len(b) {
		panic("vectors must have the same length")
	}

	var sum float32
	for i := 0; i < len(a); i++ {
		diff := a[i] - b[i]
		sum += diff * diff
	}

	return sum
}

// DotProductDistance calculates negative dot product as a distance
// For normalized vectors, higher dot product indicates higher similarity
// We negate this to make it a distance (where lower values = more similar)
func DotProductDistance(a, b F32) float32 {
	if len(a) != len(b) {
		panic("vectors must have the same length")
	}

	var dotProduct float64
	for i := 0; i < len(a); i++ {
		dotProduct += float64(a[i]) * float64(b[i])
	}

	// Convert to distance: 1 - dot(a, b)
	// For normalized vectors, this will range from 0 (identical) to 2 (opposite)
	return float32(1.0 - dotProduct)
}

// ManhattanDistance calculates the L1 norm (Manhattan distance) between vectors
func ManhattanDistance(a, b F32) float32 {
	if len(a) != len(b) {
		panic("vectors must have the same length")
	}

	var sum float64
	for i := 0; i < len(a); i++ {
		sum += math.Abs(float64(a[i] - b[i]))
	}

	return float32(sum)
}

// Create surfaces for the standard distance functions
var (
	CosineSurface           = CreateSurface(CosineDistance)
	EuclideanSurface        = CreateSurface(EuclideanDistance)
	SquaredEuclideanSurface = CreateSurface(SquaredEuclideanDistance)
	DotProductSurface       = CreateSurface(DotProductDistance)
	ManhattanSurface        = CreateSurface(ManhattanDistance)
)

// NormalizeVector normalizes a vector to unit length
func NormalizeVector(v F32) F32 {
	var magnitude float64
	for _, val := range v {
		magnitude += float64(val) * float64(val)
	}
	magnitude = math.Sqrt(magnitude)

	// Avoid division by zero
	if magnitude == 0 {
		return v
	}

	normalized := make(F32, len(v))
	for i, val := range v {
		normalized[i] = float32(float64(val) / magnitude)
	}

	return normalized
}

// VectorAdd adds two vectors
func VectorAdd(a, b F32) (F32, error) {
	if len(a) != len(b) {
		return nil, fmt.Errorf("vectors must have the same length: %d != %d", len(a), len(b))
	}

	result := make(F32, len(a))
	for i := 0; i < len(a); i++ {
		result[i] = a[i] + b[i]
	}
	return result, nil
}

// VectorSubtract subtracts vector b from vector a
func VectorSubtract(a, b F32) (F32, error) {
	if len(a) != len(b) {
		return nil, fmt.Errorf("vectors must have the same length: %d != %d", len(a), len(b))
	}

	result := make(F32, len(a))
	for i := 0; i < len(a); i++ {
		result[i] = a[i] - b[i]
	}
	return result, nil
}

// VectorMultiplyScalar multiplies a vector by a scalar
func VectorMultiplyScalar(v F32, scalar float32) F32 {
	result := make(F32, len(v))
	for i := 0; i < len(v); i++ {
		result[i] = v[i] * scalar
	}
	return result
}

// VectorMagnitude calculates the magnitude (Euclidean norm) of a vector.
func VectorMagnitude(v F32) float32 {
	var sumSquares float64
	for _, val := range v {
		sumSquares += float64(val) * float64(val)
	}
	return float32(math.Sqrt(sumSquares))
}

// CreateZeroVector creates a new vector filled with zeros of the specified dimension.
func CreateZeroVector(dimension int) F32 {
	return make(F32, dimension)
}

// CreateRandomVector creates a new vector with random values.
func CreateRandomVector(dimension int) F32 {
	v := make(F32, dimension)
	for i := 0; i < dimension; i++ {
		v[i] = float32(math.Sin(float64(i))) // Simple deterministic approach for demo
	}
	return v
}

// CloneVector creates a deep copy of a vector.
func CloneVector(v F32) F32 {
	clone := make(F32, len(v))
	copy(clone, v)
	return clone
}
