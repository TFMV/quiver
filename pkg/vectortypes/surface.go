// Package vectortypes provides common types for vector operations
package vectortypes

// F32 is a type alias for []float32 to make it more expressive
type F32 = []float32

// DistanceFunc is a function that computes the distance between two vectors.
type DistanceFunc func(a, b F32) float32

// Surface represents a distance function between two vectors
type Surface[T any] interface {
	// Distance calculates the distance between two vectors
	Distance(a, b T) float32
}

// ContraMap is a generic adapter that allows applying a distance function to a different type
// by first mapping that type to the vector type the distance function expects
type ContraMap[V, T any] struct {
	// The underlying surface (distance function)
	Surface Surface[V]

	// The mapping function from T to V
	ContraMap func(T) V
}

// Distance implements the Surface interface by first mapping the inputs and then applying the underlying distance function
func (c ContraMap[V, T]) Distance(a, b T) float32 {
	return c.Surface.Distance(c.ContraMap(a), c.ContraMap(b))
}

// BasicSurface wraps a standard distance function
type BasicSurface struct {
	DistFunc DistanceFunc
}

// Distance implements the Surface interface for F32 vectors
func (s BasicSurface) Distance(a, b F32) float32 {
	return s.DistFunc(a, b)
}

// CreateSurface creates a basic surface from a distance function
func CreateSurface(distFunc DistanceFunc) Surface[F32] {
	return BasicSurface{DistFunc: distFunc}
}
