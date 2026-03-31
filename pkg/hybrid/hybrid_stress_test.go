package hybrid

import (
	"fmt"
	"math/rand"
	"sync"
	"testing"
	"time"

	"github.com/TFMV/quiver/pkg/vectortypes"
)

// TestHybridIndex_ConcurrencyStress verifies parallel read-write safety spanning sub-indexes
func TestHybridIndex_ConcurrencyStress(t *testing.T) {
	config := IndexConfig{
		ExactThreshold: 1000,
		DistanceFunc:   vectortypes.EuclideanDistance,
	}
	idx := NewHybridIndex(config)
	dim := 128

	var wg sync.WaitGroup

	// Phase 1: Heavy concurrent insertions
	numWriters := 20
	insertsPerWriter := 100

	start := time.Now()
	for i := 0; i < numWriters; i++ {
		wg.Add(1)
		go func(writerID int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(writerID)))
			for j := 0; j < insertsPerWriter; j++ {
				vec := make([]float32, dim)
				for k := 0; k < dim; k++ {
					vec[k] = rng.Float32()
				}
				id := fmt.Sprintf("vec_%d_%d", writerID, j)
				err := idx.Insert(id, vec)
				if err != nil {
					t.Errorf("Insert failed: %v", err)
				}
			}
		}(i)
	}

	// Phase 2: Concurrent queries while inserting
	numReaders := 20
	queriesPerReader := 50

	for i := 0; i < numReaders; i++ {
		wg.Add(1)
		go func(readerID int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(readerID) + 1000))
			for j := 0; j < queriesPerReader; j++ {
				vec := make([]float32, dim)
				for k := 0; k < dim; k++ {
					vec[k] = rng.Float32()
				}
				_, err := idx.SearchWithRequest(HybridSearchRequest{
					Query: vec,
					K:     5,
				})
				// It's possible to fail organically if not enough vectors are in the system yet.
				// But we shouldn't race or panic.
				if err != nil {
					_ = err // Ignore empty graph errors safely
				}
				time.Sleep(time.Millisecond) // Yield slightly letting Inserts fight Lock
			}
		}(i)
	}

	wg.Wait()
	duration := time.Since(start)

	expectedCount := numWriters * insertsPerWriter
	if idx.Size() != expectedCount {
		t.Errorf("Expected index size %d, got %d", expectedCount, idx.Size())
	}
	t.Logf("Stress test processed %d mutations & %d hybrid queries across %d threads in %v",
		expectedCount, numReaders*queriesPerReader, numWriters+numReaders, duration)
}
