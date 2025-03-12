package quiver

import (
	"math"
	"math/rand"
	"testing"

	"github.com/TFMV/hnsw"
)

// Benchmark results showed that our original implementations significantly outperform
// the HNSW library's implementations. See distance_benchmark_results.md for details.
// We're keeping both benchmark functions for future reference and comparison.

// Original implementation of cosine distance for benchmarking
func originalCosineDistance(a, b []float32) float32 {
	var dotProduct float32
	var normA, normB float32

	for i := 0; i < len(a); i++ {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	return dotProduct / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}

// Original implementation of L2 distance for benchmarking
func originalL2Distance(a, b []float32) float32 {
	var sum float32
	for i := 0; i < len(a); i++ {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}

// generateRandomVectorForBenchmark creates a random vector of the specified dimension
func generateRandomVectorForBenchmark(dim int) []float32 {
	vec := make([]float32, dim)
	for i := range vec {
		vec[i] = rand.Float32()
	}
	return vec
}

// BenchmarkOriginalCosineDistance benchmarks the original cosine distance implementation
func BenchmarkOriginalCosineDistance(b *testing.B) {
	dimensions := []int{32, 128, 256, 512, 1024}

	for _, dim := range dimensions {
		b.Run("dim="+string(rune(dim)), func(b *testing.B) {
			// Generate two random vectors
			vec1 := generateRandomVectorForBenchmark(dim)
			vec2 := generateRandomVectorForBenchmark(dim)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = originalCosineDistance(vec1, vec2)
			}
		})
	}
}

// BenchmarkOptimizedCosineDistance benchmarks the HNSW cosine distance implementation
// Note: Our benchmarks showed this is actually slower than our original implementation
func BenchmarkOptimizedCosineDistance(b *testing.B) {
	dimensions := []int{32, 128, 256, 512, 1024}

	for _, dim := range dimensions {
		b.Run("dim="+string(rune(dim)), func(b *testing.B) {
			// Generate two random vectors
			vec1 := generateRandomVectorForBenchmark(dim)
			vec2 := generateRandomVectorForBenchmark(dim)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = hnsw.CosineDistance(vec1, vec2)
			}
		})
	}
}

// BenchmarkOriginalL2Distance benchmarks the original L2 distance implementation
func BenchmarkOriginalL2Distance(b *testing.B) {
	dimensions := []int{32, 128, 256, 512, 1024}

	for _, dim := range dimensions {
		b.Run("dim="+string(rune(dim)), func(b *testing.B) {
			// Generate two random vectors
			vec1 := generateRandomVectorForBenchmark(dim)
			vec2 := generateRandomVectorForBenchmark(dim)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = originalL2Distance(vec1, vec2)
			}
		})
	}
}

// BenchmarkOptimizedL2Distance benchmarks the HNSW L2 distance implementation
// Note: Our benchmarks showed this is actually slower than our original implementation
func BenchmarkOptimizedL2Distance(b *testing.B) {
	dimensions := []int{32, 128, 256, 512, 1024}

	for _, dim := range dimensions {
		b.Run("dim="+string(rune(dim)), func(b *testing.B) {
			// Generate two random vectors
			vec1 := generateRandomVectorForBenchmark(dim)
			vec2 := generateRandomVectorForBenchmark(dim)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = hnsw.EuclideanDistance(vec1, vec2)
			}
		})
	}
}
