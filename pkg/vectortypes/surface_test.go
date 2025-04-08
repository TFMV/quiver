package vectortypes

import (
	"testing"
)

func TestBasicSurface_Distance(t *testing.T) {
	tests := []struct {
		name     string
		distFunc DistanceFunc
		vecA     F32
		vecB     F32
		expected float32
	}{
		{
			name:     "Cosine Distance",
			distFunc: CosineDistance,
			vecA:     F32{1, 0, 0},
			vecB:     F32{0, 1, 0},
			expected: 1.0, // Perpendicular vectors have cosine distance of 1
		},
		{
			name:     "Euclidean Distance",
			distFunc: EuclideanDistance,
			vecA:     F32{1, 0, 0},
			vecB:     F32{0, 1, 0},
			expected: 1.4142135, // sqrt(2)
		},
		{
			name: "Custom Distance (Sum)",
			distFunc: func(a, b F32) float32 {
				var sum float32
				for i := range a {
					sum += a[i] + b[i]
				}
				return sum
			},
			vecA:     F32{1, 2, 3},
			vecB:     F32{4, 5, 6},
			expected: 21.0, // 1+2+3+4+5+6 = 21
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			surface := BasicSurface{DistFunc: tt.distFunc}
			result := surface.Distance(tt.vecA, tt.vecB)

			if !floatEquals(result, tt.expected, 1e-6) {
				t.Errorf("BasicSurface.Distance(%v, %v) = %v, want %v",
					tt.vecA, tt.vecB, result, tt.expected)
			}
		})
	}
}

func TestCreateSurface(t *testing.T) {
	tests := []struct {
		name     string
		distFunc DistanceFunc
		vecA     F32
		vecB     F32
		expected float32
	}{
		{
			name:     "Cosine Surface",
			distFunc: CosineDistance,
			vecA:     F32{1, 0, 0},
			vecB:     F32{0, 1, 0},
			expected: 1.0,
		},
		{
			name:     "Euclidean Surface",
			distFunc: EuclideanDistance,
			vecA:     F32{0, 0, 0},
			vecB:     F32{3, 4, 0},
			expected: 5.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			surface := CreateSurface(tt.distFunc)
			result := surface.Distance(tt.vecA, tt.vecB)

			if !floatEquals(result, tt.expected, 1e-6) {
				t.Errorf("CreateSurface(%v).Distance(%v, %v) = %v, want %v",
					tt.name, tt.vecA, tt.vecB, result, tt.expected)
			}
		})
	}
}

func TestStandardSurfaces(t *testing.T) {
	// Test the pre-defined standard surfaces
	tests := []struct {
		name     string
		surface  Surface[F32]
		vecA     F32
		vecB     F32
		expected float32
	}{
		{
			name:     "Cosine Surface",
			surface:  CosineSurface,
			vecA:     F32{1, 0, 0},
			vecB:     F32{0, 1, 0},
			expected: 1.0,
		},
		{
			name:     "Euclidean Surface",
			surface:  EuclideanSurface,
			vecA:     F32{1, 0, 0},
			vecB:     F32{0, 1, 0},
			expected: 1.4142135, // sqrt(2)
		},
		{
			name:     "Dot Product Surface",
			surface:  DotProductSurface,
			vecA:     F32{2, 0, 0},
			vecB:     F32{2, 0, 0},
			expected: -3.0, // 1 - (2*2) = -3
		},
		{
			name:     "Manhattan Surface",
			surface:  ManhattanSurface,
			vecA:     F32{1, 2, 3},
			vecB:     F32{4, 5, 6},
			expected: 9.0, // |1-4| + |2-5| + |3-6| = 9
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.surface.Distance(tt.vecA, tt.vecB)

			if !floatEquals(result, tt.expected, 1e-6) {
				t.Errorf("%s.Distance(%v, %v) = %v, want %v",
					tt.name, tt.vecA, tt.vecB, result, tt.expected)
			}
		})
	}
}

func TestContraMap(t *testing.T) {
	// Define a ContraMap that converts strings to vectors and then applies cosine distance
	type StringVector string

	stringToVector := func(s StringVector) F32 {
		// Create a vector from the ASCII values of up to 3 characters
		result := make(F32, 3)
		if len(s) > 0 {
			result[0] = float32(s[0])
		}
		if len(s) > 1 {
			result[1] = float32(s[1])
		}
		if len(s) > 2 {
			result[2] = float32(s[2])
		}
		return result
	}

	contraMap := ContraMap[F32, StringVector]{
		Surface:   CosineSurface,
		ContraMap: stringToVector,
	}

	tests := []struct {
		name     string
		strA     StringVector
		strB     StringVector
		expected float32
	}{
		{
			name:     "Same String",
			strA:     "abc",
			strB:     "abc",
			expected: 0.0, // Same vectors have cosine distance 0
		},
		{
			name:     "Different Strings",
			strA:     "abc",
			strB:     "xyz",
			expected: 0.0, // The ASCII vectors we're using produce a value very close to 0
		},
		{
			name:     "Empty Strings",
			strA:     "",
			strB:     "",
			expected: 0.0, // Both convert to zero vectors
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := contraMap.Distance(tt.strA, tt.strB)

			// Use a larger tolerance for the "Different Strings" test
			tolerance := float32(0.1)

			if !floatEquals(result, tt.expected, tolerance) {
				t.Errorf("ContraMap.Distance(%v, %v) = %v, want %v (±%v)",
					tt.strA, tt.strB, result, tt.expected, tolerance)
			}
		})
	}
}
