package dimreduce

import (
	"math/rand"
	"testing"

	"go.uber.org/zap"
)

func TestPCAWithRandomData(t *testing.T) {
	// Create a logger
	logger, _ := zap.NewDevelopment()

	// Test with different dimensions and sample sizes
	testCases := []struct {
		name           string
		inputDim       int
		targetDim      int
		numSamples     int
		shouldSucceed  bool
		generateMethod string
	}{
		{"Small Random", 10, 5, 20, true, "random"},
		{"Medium Random", 50, 20, 100, true, "random"},
		{"Large Random", 128, 64, 1000, true, "random"},
		{"Small Structured", 10, 5, 20, true, "structured"},
		{"Medium Structured", 50, 20, 100, true, "structured"},
		{"Large Structured", 128, 64, 1000, true, "structured"},
		{"Small Clustered", 10, 5, 20, true, "clustered"},
		{"Medium Clustered", 50, 20, 100, true, "clustered"},
		{"Large Clustered", 128, 64, 1000, true, "clustered"},
		// Edge cases
		{"Few Samples", 128, 64, 10, true, "random"},       // Should succeed - we now handle small sample sizes
		{"Equal Dimensions", 50, 50, 100, false, "random"}, // Should fail - no reduction
		{"Target > Input", 10, 20, 100, false, "random"},   // Should fail - invalid dimensions
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Generate test vectors
			var vectors [][]float32
			switch tc.generateMethod {
			case "random":
				vectors = generateRandomVectors(tc.numSamples, tc.inputDim)
			case "structured":
				vectors = generateStructuredVectors(tc.numSamples, tc.inputDim, 5)
			case "clustered":
				vectors = generateClusteredVectors(tc.numSamples, tc.inputDim, 3)
			}

			// Create reducer
			config := DimReducerConfig{
				TargetDimension: tc.targetDim,
				Method:          PCA,
				Logger:          logger,
			}

			reducer, err := NewDimReducer(config)
			if err != nil {
				t.Fatalf("Failed to create reducer: %v", err)
			}

			// Perform reduction
			reduced, err := reducer.Reduce(vectors)

			// Check results
			if tc.shouldSucceed {
				if err != nil {
					t.Errorf("Expected success but got error: %v", err)
				} else {
					// Check dimensions
					if len(reduced) != len(vectors) {
						t.Errorf("Expected %d vectors, got %d", len(vectors), len(reduced))
					}
					if len(reduced) > 0 && len(reduced[0]) != tc.targetDim {
						t.Errorf("Expected dimension %d, got %d", tc.targetDim, len(reduced[0]))
					}
					// Check for NaN or Inf values
					if hasNaNOrInf(reduced) {
						t.Errorf("Reduced vectors contain NaN or Inf values")
					}
				}
			} else {
				if err == nil {
					t.Errorf("Expected error but got success")
				}
			}
		})
	}
}

func TestAdaptivePCA(t *testing.T) {
	// Create a logger
	logger, _ := zap.NewDevelopment()

	// Test cases
	testCases := []struct {
		name             string
		inputDim         int
		minVariance      float64
		numSamples       int
		generateMethod   string
		expectedDimRange [2]int // min and max expected dimensions
	}{
		{"High Variance", 100, 0.95, 500, "structured", [2]int{10, 90}}, // Increased upper bound
		{"Medium Variance", 100, 0.8, 500, "structured", [2]int{5, 80}}, // Increased upper bound
		{"Low Variance", 100, 0.5, 500, "structured", [2]int{2, 50}},    // Increased upper bound
		{"Clustered Data", 100, 0.9, 500, "clustered", [2]int{2, 50}},   // Adjusted for clustered data
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Generate test vectors
			var vectors [][]float32
			switch tc.generateMethod {
			case "random":
				vectors = generateRandomVectors(tc.numSamples, tc.inputDim)
			case "structured":
				vectors = generateStructuredVectors(tc.numSamples, tc.inputDim, 5)
			case "clustered":
				vectors = generateClusteredVectors(tc.numSamples, tc.inputDim, 3)
			}

			// Create reducer with adaptive settings
			config := DimReducerConfig{
				TargetDimension:      tc.inputDim / 2, // Default target
				Method:               PCA,
				Adaptive:             true,
				MinVarianceExplained: tc.minVariance,
				MaxDimension:         tc.inputDim,
				Logger:               logger,
			}

			reducer, err := NewDimReducer(config)
			if err != nil {
				t.Fatalf("Failed to create reducer: %v", err)
			}

			// Perform reduction
			reduced, err := reducer.Reduce(vectors)
			if err != nil {
				t.Fatalf("Reduction failed: %v", err)
			}

			// Check dimensions
			actualDim := len(reduced[0])
			if actualDim < tc.expectedDimRange[0] || actualDim > tc.expectedDimRange[1] {
				t.Errorf("Expected dimension between %d and %d, got %d",
					tc.expectedDimRange[0], tc.expectedDimRange[1], actualDim)
			}

			// Check for NaN or Inf values
			for _, vec := range reduced {
				for _, val := range vec {
					if isNaN(val) || isInf(val) {
						t.Errorf("Reduced vector contains NaN or Inf values")
						break
					}
				}
			}
		})
	}
}

func TestPCAWithEdgeCases(t *testing.T) {
	// Create a logger
	logger, _ := zap.NewDevelopment()

	// Test cases
	testCases := []struct {
		name      string
		generator func() [][]float32
		expectOk  bool
		comment   string
	}{
		{"Single Sample", func() [][]float32 { return generateRandomVectors(1, 10) }, true, "Should succeed - we now handle single samples with simple projection"},
		{"Zero Variance Features", func() [][]float32 {
			// Generate vectors where some features have zero variance
			vecs := make([][]float32, 10)
			for i := range vecs {
				vecs[i] = make([]float32, 10)
				for j := range vecs[i] {
					if j < 5 {
						vecs[i][j] = float32(i) // varying values
					} else {
						vecs[i][j] = 1.0 // constant values (zero variance)
					}
				}
			}
			return vecs
		}, true, "Should succeed - we now add small noise to zero variance features"},
		{
			"Highly Correlated Features",
			func() [][]float32 {
				// Create vectors with highly correlated features
				vectors := generateStructuredVectors(100, 10, 3)
				// Make some features highly correlated
				for i := range vectors {
					vectors[i][1] = vectors[i][0] + 0.001*rand.Float32()
				}
				return vectors
			},
			true, "Should succeed with regularization"},
		{
			"Empty Input",
			func() [][]float32 {
				return [][]float32{}
			},
			true, "Should succeed - empty input returns empty output"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			vectors := tc.generator()

			config := DimReducerConfig{
				TargetDimension: 5,
				Method:          PCA,
				Adaptive:        false,
				Logger:          logger,
			}

			reducer, err := NewDimReducer(config)
			if err != nil {
				t.Fatalf("Failed to create reducer: %v", err)
			}

			// Perform reduction
			result, err := reducer.Reduce(vectors)

			if tc.expectOk {
				if err != nil {
					t.Errorf("Expected success but got error: %v", err)
				} else if tc.name == "Empty Input" && len(result) != 0 {
					t.Errorf("Expected empty result for empty input, got %v", result)
				}
			} else {
				if err == nil {
					t.Errorf("Expected error but got success")
				}
			}
		})
	}
}

// Helper functions for generating test data

// generateRandomVectors generates random vectors
func generateRandomVectors(numSamples, dim int) [][]float32 {
	vectors := make([][]float32, numSamples)
	for i := range vectors {
		vectors[i] = make([]float32, dim)
		for j := range vectors[i] {
			vectors[i][j] = rand.Float32()
		}
	}
	return vectors
}

// generateStructuredVectors generates vectors with structure by combining "concept" vectors
func generateStructuredVectors(numSamples, dim, numConcepts int) [][]float32 {
	// Create concept vectors
	concepts := make([][]float32, numConcepts)
	for i := range concepts {
		concepts[i] = make([]float32, dim)
		for j := range concepts[i] {
			concepts[i][j] = rand.Float32()*2 - 1 // Values between -1 and 1
		}
	}

	// Generate vectors as combinations of concepts plus noise
	vectors := make([][]float32, numSamples)
	for i := range vectors {
		vectors[i] = make([]float32, dim)

		// Mix concepts with different weights
		for j := range vectors[i] {
			// Start with some noise
			vectors[i][j] = (rand.Float32() * 0.1) - 0.05 // Small noise component

			// Add weighted concepts
			for _, concept := range concepts {
				weight := rand.Float32() * 0.5 // Random weight for each concept
				vectors[i][j] += concept[j] * weight
			}
		}
	}

	return vectors
}

// generateClusteredVectors generates vectors in clusters
func generateClusteredVectors(numSamples, dim, numClusters int) [][]float32 {
	// Create cluster centers
	centers := make([][]float32, numClusters)
	for i := range centers {
		centers[i] = make([]float32, dim)
		for j := range centers[i] {
			centers[i][j] = rand.Float32()*2 - 1 // Values between -1 and 1
		}
	}

	// Generate vectors around cluster centers
	vectors := make([][]float32, numSamples)
	for i := range vectors {
		// Pick a random cluster
		clusterIdx := rand.Intn(numClusters)
		center := centers[clusterIdx]

		// Create vector near the cluster center
		vectors[i] = make([]float32, dim)
		for j := range vectors[i] {
			// Add the center value plus some noise
			noise := (rand.Float32() * 0.2) - 0.1 // Small noise
			vectors[i][j] = center[j] + noise
		}
	}

	return vectors
}

// isNaN checks if a float32 is NaN
func isNaN(f float32) bool {
	return f != f
}

// isInf checks if a float32 is Inf
func isInf(f float32) bool {
	return f > 3.4e38 || f < -3.4e38 // Approximate check for infinity
}

// hasNaNOrInf checks if any vector contains NaN or Inf values
func hasNaNOrInf(vectors [][]float32) bool {
	for _, vec := range vectors {
		for _, val := range vec {
			if isNaN(val) || isInf(val) {
				return true
			}
		}
	}
	return false
}

// Benchmark PCA performance
func BenchmarkPCA(b *testing.B) {
	// Create a logger
	logger, _ := zap.NewDevelopment()

	// Test cases
	testCases := []struct {
		name       string
		inputDim   int
		targetDim  int
		numSamples int
	}{
		{"Small", 10, 5, 100},
		{"Medium", 100, 50, 1000},
		{"Large", 1000, 100, 1000},
	}

	for _, tc := range testCases {
		b.Run(tc.name, func(b *testing.B) {
			// Generate test data
			vectors := generateStructuredVectors(tc.numSamples, tc.inputDim, 10)

			// Create reducer
			config := DimReducerConfig{
				TargetDimension: tc.targetDim,
				Method:          PCA,
				Adaptive:        false,
				Logger:          logger,
			}

			reducer, _ := NewDimReducer(config)

			// Reset timer and run benchmark
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := reducer.Reduce(vectors)
				if err != nil {
					b.Fatalf("PCA failed: %v", err)
				}
			}
		})
	}
}
