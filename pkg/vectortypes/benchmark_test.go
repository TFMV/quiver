package vectortypes

import "testing"

func benchmarkDistance(b *testing.B, fn DistanceFunc) {
	dim := 128
	a := make(F32, dim)
	c := make(F32, dim)
	for i := 0; i < dim; i++ {
		a[i] = float32(i) * 0.01
		c[i] = float32(i+1) * 0.02
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = fn(a, c)
	}
}

func BenchmarkCosineDistance(b *testing.B)           { benchmarkDistance(b, CosineDistance) }
func BenchmarkEuclideanDistance(b *testing.B)        { benchmarkDistance(b, EuclideanDistance) }
func BenchmarkSquaredEuclideanDistance(b *testing.B) { benchmarkDistance(b, SquaredEuclideanDistance) }
func BenchmarkDotProductDistance(b *testing.B)       { benchmarkDistance(b, DotProductDistance) }
func BenchmarkManhattanDistance(b *testing.B)        { benchmarkDistance(b, ManhattanDistance) }
