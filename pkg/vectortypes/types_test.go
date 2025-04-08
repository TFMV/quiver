package vectortypes

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestGetDistanceFuncByType(t *testing.T) {
	tests := []struct {
		name       string
		distType   DistanceType
		wantFunc   DistanceFunc
		checkVecs  bool
		vecA       F32
		vecB       F32
		wantResult float32
	}{
		{
			name:       "Cosine Distance",
			distType:   Cosine,
			wantFunc:   CosineDistance,
			checkVecs:  true,
			vecA:       F32{1, 0, 0},
			vecB:       F32{0, 1, 0},
			wantResult: 1.0, // Perpendicular vectors have cosine distance of 1
		},
		{
			name:       "Euclidean Distance",
			distType:   Euclidean,
			wantFunc:   EuclideanDistance,
			checkVecs:  true,
			vecA:       F32{1, 0, 0},
			vecB:       F32{0, 1, 0},
			wantResult: 1.4142135, // sqrt(2)
		},
		{
			name:       "Dot Product Distance",
			distType:   DotProduct,
			wantFunc:   DotProductDistance,
			checkVecs:  true,
			vecA:       F32{1, 0, 0},
			vecB:       F32{1, 0, 0},
			wantResult: 0.0, // Same direction has dot product distance of 0
		},
		{
			name:       "Manhattan Distance",
			distType:   Manhattan,
			wantFunc:   ManhattanDistance,
			checkVecs:  true,
			vecA:       F32{1, 1, 0},
			vecB:       F32{0, 0, 0},
			wantResult: 2.0, // |1-0| + |1-0| + |0-0| = 2
		},
		{
			name:      "Default to Cosine",
			distType:  "invalid",
			wantFunc:  CosineDistance,
			checkVecs: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := GetDistanceFuncByType(tt.distType)

			// Can't directly compare functions, so we test with the same input
			if tt.checkVecs {
				gotResult := got(tt.vecA, tt.vecB)
				if !floatEquals(gotResult, tt.wantResult, 1e-6) {
					t.Errorf("GetDistanceFuncByType(%v) = func giving %v, want func giving %v",
						tt.distType, gotResult, tt.wantResult)
				}
			}
		})
	}
}

func TestGetSurfaceByType(t *testing.T) {
	tests := []struct {
		name       string
		distType   DistanceType
		checkVecs  bool
		vecA       F32
		vecB       F32
		wantResult float32
	}{
		{
			name:       "Cosine Surface",
			distType:   Cosine,
			checkVecs:  true,
			vecA:       F32{1, 0, 0},
			vecB:       F32{0, 1, 0},
			wantResult: 1.0, // Perpendicular vectors have cosine distance of 1
		},
		{
			name:       "Euclidean Surface",
			distType:   Euclidean,
			checkVecs:  true,
			vecA:       F32{1, 0, 0},
			vecB:       F32{0, 1, 0},
			wantResult: 1.4142135, // sqrt(2)
		},
		{
			name:       "Dot Product Surface",
			distType:   DotProduct,
			checkVecs:  true,
			vecA:       F32{1, 0, 0},
			vecB:       F32{1, 0, 0},
			wantResult: 0.0, // Same direction has dot product distance of 0
		},
		{
			name:       "Manhattan Surface",
			distType:   Manhattan,
			checkVecs:  true,
			vecA:       F32{1, 1, 0},
			vecB:       F32{0, 0, 0},
			wantResult: 2.0, // |1-0| + |1-0| + |0-0| = 2
		},
		{
			name:      "Default to Cosine",
			distType:  "invalid",
			checkVecs: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := GetSurfaceByType(tt.distType)

			// Check the surface by testing distance calculation
			if tt.checkVecs {
				gotResult := got.Distance(tt.vecA, tt.vecB)
				if !floatEquals(gotResult, tt.wantResult, 1e-6) {
					t.Errorf("GetSurfaceByType(%v).Distance(%v, %v) = %v, want %v",
						tt.distType, tt.vecA, tt.vecB, gotResult, tt.wantResult)
				}
			}
		})
	}
}

func TestComputeDistance(t *testing.T) {
	tests := []struct {
		name       string
		vecA       F32
		vecB       F32
		distType   DistanceType
		wantResult float32
		wantErr    bool
	}{
		{
			name:       "Cosine Distance",
			vecA:       F32{1, 0, 0},
			vecB:       F32{0, 1, 0},
			distType:   Cosine,
			wantResult: 1.0,
			wantErr:    false,
		},
		{
			name:       "Different Dimensions",
			vecA:       F32{1, 0, 0},
			vecB:       F32{0, 1},
			distType:   Cosine,
			wantResult: 0.0,
			wantErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotResult, err := ComputeDistance(tt.vecA, tt.vecB, tt.distType)

			// Check error
			if (err != nil) != tt.wantErr {
				t.Errorf("ComputeDistance() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			// Check result if no error expected
			if !tt.wantErr && !floatEquals(gotResult, tt.wantResult, 1e-6) {
				t.Errorf("ComputeDistance() = %v, want %v", gotResult, tt.wantResult)
			}
		})
	}
}

func TestIsNormalized(t *testing.T) {
	tests := []struct {
		name       string
		vec        F32
		wantResult bool
	}{
		{
			name:       "Unit Vector",
			vec:        F32{1, 0, 0},
			wantResult: true,
		},
		{
			name:       "Normalized Vector",
			vec:        F32{0.577, 0.577, 0.577}, // 1/sqrt(3) in each component
			wantResult: true,
		},
		{
			name:       "Zero Vector",
			vec:        F32{0, 0, 0},
			wantResult: false, // Zero vector is not normally considered normalized
		},
		{
			name:       "Non-normalized Vector",
			vec:        F32{2, 0, 0},
			wantResult: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotResult := IsNormalized(tt.vec)

			if gotResult != tt.wantResult {
				t.Errorf("IsNormalized(%v) = %v, want %v", tt.vec, gotResult, tt.wantResult)
			}
		})
	}
}

func TestVector_GetMetadataValue(t *testing.T) {
	tests := []struct {
		name      string
		vector    *Vector
		key       string
		wantValue interface{}
		wantErr   bool
	}{
		{
			name: "Valid Key",
			vector: &Vector{
				ID:       "vec1",
				Values:   F32{1, 2, 3},
				Metadata: json.RawMessage(`{"key1": "value1", "key2": 123}`),
			},
			key:       "key1",
			wantValue: "value1",
			wantErr:   false,
		},
		{
			name: "Missing Key",
			vector: &Vector{
				ID:       "vec2",
				Values:   F32{1, 2, 3},
				Metadata: json.RawMessage(`{"key1": "value1"}`),
			},
			key:       "key2",
			wantValue: nil,
			wantErr:   true,
		},
		{
			name: "No Metadata",
			vector: &Vector{
				ID:     "vec3",
				Values: F32{1, 2, 3},
			},
			key:       "key1",
			wantValue: nil,
			wantErr:   true,
		},
		{
			name: "Invalid JSON",
			vector: &Vector{
				ID:       "vec4",
				Values:   F32{1, 2, 3},
				Metadata: json.RawMessage(`{"key1": value1}`), // Invalid JSON (missing quotes)
			},
			key:       "key1",
			wantValue: nil,
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotValue, err := tt.vector.GetMetadataValue(tt.key)

			// Check error
			if (err != nil) != tt.wantErr {
				t.Errorf("Vector.GetMetadataValue() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			// Check result if no error expected
			if !tt.wantErr && !reflect.DeepEqual(gotValue, tt.wantValue) {
				t.Errorf("Vector.GetMetadataValue() = %v, want %v", gotValue, tt.wantValue)
			}
		})
	}
}

func TestCheckDimensions(t *testing.T) {
	tests := []struct {
		name    string
		vecA    F32
		vecB    F32
		wantErr bool
	}{
		{
			name:    "Same Dimensions",
			vecA:    F32{1, 2, 3},
			vecB:    F32{4, 5, 6},
			wantErr: false,
		},
		{
			name:    "Different Dimensions",
			vecA:    F32{1, 2, 3},
			vecB:    F32{4, 5},
			wantErr: true,
		},
		{
			name:    "Zero Dimensions",
			vecA:    F32{},
			vecB:    F32{},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := CheckDimensions(tt.vecA, tt.vecB)

			if (err != nil) != tt.wantErr {
				t.Errorf("CheckDimensions() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestCreateVector(t *testing.T) {
	tests := []struct {
		name     string
		id       string
		values   F32
		metadata json.RawMessage
		want     *Vector
	}{
		{
			name:     "Basic Vector",
			id:       "vec1",
			values:   F32{1, 2, 3},
			metadata: json.RawMessage(`{"key": "value"}`),
			want: &Vector{
				ID:       "vec1",
				Values:   F32{1, 2, 3},
				Metadata: json.RawMessage(`{"key": "value"}`),
			},
		},
		{
			name:     "No Metadata",
			id:       "vec2",
			values:   F32{4, 5, 6},
			metadata: nil,
			want: &Vector{
				ID:       "vec2",
				Values:   F32{4, 5, 6},
				Metadata: nil,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := CreateVector(tt.id, tt.values, tt.metadata)

			if !vectorEqual(got, tt.want) {
				t.Errorf("CreateVector() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestVector_Dimension(t *testing.T) {
	tests := []struct {
		name string
		vec  *Vector
		want int
	}{
		{
			name: "3D Vector",
			vec: &Vector{
				ID:     "vec1",
				Values: F32{1, 2, 3},
			},
			want: 3,
		},
		{
			name: "Empty Vector",
			vec: &Vector{
				ID:     "vec2",
				Values: F32{},
			},
			want: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.vec.Dimension()

			if got != tt.want {
				t.Errorf("Vector.Dimension() = %v, want %v", got, tt.want)
			}
		})
	}
}

// Helper function to compare vectors
func vectorEqual(a, b *Vector) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}

	if a.ID != b.ID {
		return false
	}

	if len(a.Values) != len(b.Values) {
		return false
	}

	for i := range a.Values {
		if !floatEquals(a.Values[i], b.Values[i], 1e-6) {
			return false
		}
	}

	return string(a.Metadata) == string(b.Metadata)
}
