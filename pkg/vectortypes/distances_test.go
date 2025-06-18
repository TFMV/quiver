package vectortypes

import (
	"math"
	"reflect"
	"testing"
)

func TestCosineDistance(t *testing.T) {
	tests := []struct {
		name     string
		vecA     F32
		vecB     F32
		expected float32
	}{
		{
			name:     "Identical Vectors",
			vecA:     F32{1, 0, 0},
			vecB:     F32{1, 0, 0},
			expected: 0, // Identical vectors have cosine distance of 0
		},
		{
			name:     "Perpendicular Vectors",
			vecA:     F32{1, 0, 0},
			vecB:     F32{0, 1, 0},
			expected: 1, // Perpendicular vectors have cosine distance of 1
		},
		{
			name:     "Opposite Vectors",
			vecA:     F32{1, 0, 0},
			vecB:     F32{-1, 0, 0},
			expected: 2, // Opposite vectors have cosine distance of 2
		},
		{
			name:     "Zero Vector",
			vecA:     F32{0, 0, 0},
			vecB:     F32{1, 0, 0},
			expected: 0, // Zero vector case is handled specially
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := CosineDistance(tt.vecA, tt.vecB)
			if !floatEquals(result, tt.expected, 1e-6) {
				t.Errorf("CosineDistance(%v, %v) = %v, want %v", tt.vecA, tt.vecB, result, tt.expected)
			}
		})
	}
}

func TestEuclideanDistance(t *testing.T) {
	tests := []struct {
		name     string
		vecA     F32
		vecB     F32
		expected float32
	}{
		{
			name:     "Identical Vectors",
			vecA:     F32{1, 0, 0},
			vecB:     F32{1, 0, 0},
			expected: 0, // Identical vectors have Euclidean distance of 0
		},
		{
			name:     "Unit Vectors",
			vecA:     F32{1, 0, 0},
			vecB:     F32{0, 1, 0},
			expected: float32(math.Sqrt(2)), // sqrt(1² + 1²)
		},
		{
			name:     "3D Vectors",
			vecA:     F32{1, 2, 3},
			vecB:     F32{4, 5, 6},
			expected: float32(math.Sqrt(27)), // sqrt(3² + 3² + 3²)
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := EuclideanDistance(tt.vecA, tt.vecB)
			if !floatEquals(result, tt.expected, 1e-6) {
				t.Errorf("EuclideanDistance(%v, %v) = %v, want %v", tt.vecA, tt.vecB, result, tt.expected)
			}
		})
	}
}

func TestSquaredEuclideanDistance(t *testing.T) {
	tests := []struct {
		name     string
		vecA     F32
		vecB     F32
		expected float32
	}{
		{
			name:     "Identical Vectors",
			vecA:     F32{1, 0, 0},
			vecB:     F32{1, 0, 0},
			expected: 0,
		},
		{
			name:     "Unit Vectors",
			vecA:     F32{1, 0, 0},
			vecB:     F32{0, 1, 0},
			expected: 2,
		},
		{
			name:     "3D Vectors",
			vecA:     F32{1, 2, 3},
			vecB:     F32{4, 5, 6},
			expected: 27,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := SquaredEuclideanDistance(tt.vecA, tt.vecB)
			if !floatEquals(result, tt.expected, 1e-6) {
				t.Errorf("SquaredEuclideanDistance(%v, %v) = %v, want %v", tt.vecA, tt.vecB, result, tt.expected)
			}
		})
	}
}

func TestDotProductDistance(t *testing.T) {
	tests := []struct {
		name     string
		vecA     F32
		vecB     F32
		expected float32
	}{
		{
			name:     "Identical Unit Vectors",
			vecA:     F32{1, 0, 0},
			vecB:     F32{1, 0, 0},
			expected: 0, // 1 - dot product (1) = 0
		},
		{
			name:     "Perpendicular Vectors",
			vecA:     F32{1, 0, 0},
			vecB:     F32{0, 1, 0},
			expected: 1, // 1 - dot product (0) = 1
		},
		{
			name:     "Opposite Vectors",
			vecA:     F32{1, 0, 0},
			vecB:     F32{-1, 0, 0},
			expected: 2, // 1 - dot product (-1) = 2
		},
		{
			name:     "Scaled Vectors",
			vecA:     F32{2, 0, 0},
			vecB:     F32{3, 0, 0},
			expected: -5, // 1 - dot product (6) = -5
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := DotProductDistance(tt.vecA, tt.vecB)
			if !floatEquals(result, tt.expected, 1e-6) {
				t.Errorf("DotProductDistance(%v, %v) = %v, want %v", tt.vecA, tt.vecB, result, tt.expected)
			}
		})
	}
}

func TestManhattanDistance(t *testing.T) {
	tests := []struct {
		name     string
		vecA     F32
		vecB     F32
		expected float32
	}{
		{
			name:     "Identical Vectors",
			vecA:     F32{1, 0, 0},
			vecB:     F32{1, 0, 0},
			expected: 0, // Identical vectors have Manhattan distance of 0
		},
		{
			name:     "Unit Vectors",
			vecA:     F32{1, 0, 0},
			vecB:     F32{0, 1, 0},
			expected: 2, // |1-0| + |0-1| + |0-0| = 2
		},
		{
			name:     "3D Vectors",
			vecA:     F32{1, 2, 3},
			vecB:     F32{4, 5, 6},
			expected: 9, // |1-4| + |2-5| + |3-6| = 3 + 3 + 3 = 9
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ManhattanDistance(tt.vecA, tt.vecB)
			if !floatEquals(result, tt.expected, 1e-6) {
				t.Errorf("ManhattanDistance(%v, %v) = %v, want %v", tt.vecA, tt.vecB, result, tt.expected)
			}
		})
	}
}

func TestDistanceFunctionPanics(t *testing.T) {
	// Test that distance functions panic with different length vectors
	tests := []struct {
		name     string
		distFunc DistanceFunc
	}{
		{"CosineDistance", CosineDistance},
		{"EuclideanDistance", EuclideanDistance},
		{"SquaredEuclideanDistance", SquaredEuclideanDistance},
		{"DotProductDistance", DotProductDistance},
		{"ManhattanDistance", ManhattanDistance},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("%s did not panic with different length vectors", tt.name)
				}
			}()
			// Should panic
			tt.distFunc(F32{1, 2, 3}, F32{1, 2})
		})
	}
}

func TestNormalizeVector(t *testing.T) {
	tests := []struct {
		name     string
		vec      F32
		expected F32
	}{
		{
			name:     "Unit Vector",
			vec:      F32{1, 0, 0},
			expected: F32{1, 0, 0}, // Already normalized
		},
		{
			name:     "Non-unit Vector",
			vec:      F32{3, 0, 0},
			expected: F32{1, 0, 0}, // Normalized to unit vector
		},
		{
			name:     "3D Vector",
			vec:      F32{1, 1, 1},
			expected: F32{1 / float32(math.Sqrt(3)), 1 / float32(math.Sqrt(3)), 1 / float32(math.Sqrt(3))},
		},
		{
			name:     "Zero Vector",
			vec:      F32{0, 0, 0},
			expected: F32{0, 0, 0}, // Zero vector can't be normalized
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := NormalizeVector(tt.vec)

			// For zero vector case, just check it's unchanged
			if tt.name == "Zero Vector" {
				if !reflect.DeepEqual(result, tt.expected) {
					t.Errorf("NormalizeVector(%v) = %v, want %v", tt.vec, result, tt.expected)
				}
				return
			}

			// For other cases, check magnitude is 1 (or very close)
			magnitude := 0.0
			for _, v := range result {
				magnitude += float64(v * v)
			}
			magnitude = math.Sqrt(magnitude)

			if !float64Equals(magnitude, 1.0, 1e-6) {
				t.Errorf("NormalizeVector(%v) produced vector with magnitude %v, want 1.0",
					tt.vec, magnitude)
			}

			// Also check if result matches expected
			for i := range result {
				if !floatEquals(result[i], tt.expected[i], 1e-6) {
					t.Errorf("NormalizeVector(%v) = %v, want %v", tt.vec, result, tt.expected)
					break
				}
			}
		})
	}
}

func TestVectorAdd(t *testing.T) {
	tests := []struct {
		name     string
		vecA     F32
		vecB     F32
		expected F32
		wantErr  bool
	}{
		{
			name:     "Same Dimension",
			vecA:     F32{1, 2, 3},
			vecB:     F32{4, 5, 6},
			expected: F32{5, 7, 9},
			wantErr:  false,
		},
		{
			name:     "Different Dimensions",
			vecA:     F32{1, 2, 3},
			vecB:     F32{4, 5},
			expected: nil,
			wantErr:  true,
		},
		{
			name:     "Zero Vectors",
			vecA:     F32{0, 0, 0},
			vecB:     F32{0, 0, 0},
			expected: F32{0, 0, 0},
			wantErr:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := VectorAdd(tt.vecA, tt.vecB)

			// Check error
			if (err != nil) != tt.wantErr {
				t.Errorf("VectorAdd() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			// Check result if no error
			if !tt.wantErr {
				if !reflect.DeepEqual(result, tt.expected) {
					t.Errorf("VectorAdd(%v, %v) = %v, want %v", tt.vecA, tt.vecB, result, tt.expected)
				}
			}
		})
	}
}

func TestVectorSubtract(t *testing.T) {
	tests := []struct {
		name     string
		vecA     F32
		vecB     F32
		expected F32
		wantErr  bool
	}{
		{
			name:     "Same Dimension",
			vecA:     F32{5, 7, 9},
			vecB:     F32{4, 5, 6},
			expected: F32{1, 2, 3},
			wantErr:  false,
		},
		{
			name:     "Different Dimensions",
			vecA:     F32{1, 2, 3},
			vecB:     F32{4, 5},
			expected: nil,
			wantErr:  true,
		},
		{
			name:     "Zero Vectors",
			vecA:     F32{0, 0, 0},
			vecB:     F32{0, 0, 0},
			expected: F32{0, 0, 0},
			wantErr:  false,
		},
		{
			name:     "Same Vectors",
			vecA:     F32{1, 2, 3},
			vecB:     F32{1, 2, 3},
			expected: F32{0, 0, 0},
			wantErr:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := VectorSubtract(tt.vecA, tt.vecB)

			// Check error
			if (err != nil) != tt.wantErr {
				t.Errorf("VectorSubtract() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			// Check result if no error
			if !tt.wantErr {
				if !reflect.DeepEqual(result, tt.expected) {
					t.Errorf("VectorSubtract(%v, %v) = %v, want %v", tt.vecA, tt.vecB, result, tt.expected)
				}
			}
		})
	}
}

func TestVectorMultiplyScalar(t *testing.T) {
	tests := []struct {
		name     string
		vec      F32
		scalar   float32
		expected F32
	}{
		{
			name:     "Multiply by 2",
			vec:      F32{1, 2, 3},
			scalar:   2,
			expected: F32{2, 4, 6},
		},
		{
			name:     "Multiply by 0",
			vec:      F32{1, 2, 3},
			scalar:   0,
			expected: F32{0, 0, 0},
		},
		{
			name:     "Multiply by -1",
			vec:      F32{1, 2, 3},
			scalar:   -1,
			expected: F32{-1, -2, -3},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := VectorMultiplyScalar(tt.vec, tt.scalar)

			if !reflect.DeepEqual(result, tt.expected) {
				t.Errorf("VectorMultiplyScalar(%v, %v) = %v, want %v",
					tt.vec, tt.scalar, result, tt.expected)
			}
		})
	}
}

func TestVectorMagnitude(t *testing.T) {
	tests := []struct {
		name     string
		vec      F32
		expected float32
	}{
		{
			name:     "Unit Vector",
			vec:      F32{1, 0, 0},
			expected: 1,
		},
		{
			name:     "2D Vector",
			vec:      F32{3, 4},
			expected: 5, // 3-4-5 triangle
		},
		{
			name:     "Zero Vector",
			vec:      F32{0, 0, 0},
			expected: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := VectorMagnitude(tt.vec)

			if !floatEquals(result, tt.expected, 1e-6) {
				t.Errorf("VectorMagnitude(%v) = %v, want %v", tt.vec, result, tt.expected)
			}
		})
	}
}

func TestCreateZeroVector(t *testing.T) {
	tests := []struct {
		name      string
		dimension int
		expected  F32
	}{
		{
			name:      "3D Vector",
			dimension: 3,
			expected:  F32{0, 0, 0},
		},
		{
			name:      "Empty Vector",
			dimension: 0,
			expected:  F32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := CreateZeroVector(tt.dimension)

			if !reflect.DeepEqual(result, tt.expected) {
				t.Errorf("CreateZeroVector(%v) = %v, want %v", tt.dimension, result, tt.expected)
			}

			if len(result) != tt.dimension {
				t.Errorf("CreateZeroVector(%v) created vector of length %v, want %v",
					tt.dimension, len(result), tt.dimension)
			}
		})
	}
}

func TestCreateRandomVector(t *testing.T) {
	tests := []struct {
		name      string
		dimension int
	}{
		{
			name:      "3D Vector",
			dimension: 3,
		},
		{
			name:      "Empty Vector",
			dimension: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := CreateRandomVector(tt.dimension)

			if len(result) != tt.dimension {
				t.Errorf("CreateRandomVector(%v) created vector of length %v, want %v",
					tt.dimension, len(result), tt.dimension)
			}
		})
	}
}

func TestCloneVector(t *testing.T) {
	tests := []struct {
		name     string
		vec      F32
		expected F32
	}{
		{
			name:     "3D Vector",
			vec:      F32{1, 2, 3},
			expected: F32{1, 2, 3},
		},
		{
			name:     "Empty Vector",
			vec:      F32{},
			expected: F32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := CloneVector(tt.vec)

			// Check that the result is equal to the expected
			if !reflect.DeepEqual(result, tt.expected) {
				t.Errorf("CloneVector(%v) = %v, want %v", tt.vec, result, tt.expected)
			}

			// Verify it's a deep copy by modifying the original
			if len(tt.vec) > 0 {
				original := tt.vec[0]
				tt.vec[0] = original + 1

				if result[0] != original {
					t.Errorf("CloneVector did not make a deep copy")
				}
			}
		})
	}
}

// Helper function to compare floating point values with tolerance
func floatEquals(a, b float32, epsilon float32) bool {
	return (a-b) < epsilon && (b-a) < epsilon
}

// Helper function to compare double precision floating point values with tolerance
func float64Equals(a, b float64, epsilon float64) bool {
	return (a-b) < epsilon && (b-a) < epsilon
}
