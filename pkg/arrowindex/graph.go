package arrowindex

import (
	"container/heap"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"

	"github.com/apache/arrow/go/arrow/array"
	"github.com/apache/arrow/go/arrow/memory"
)

// Node represents a point in the HNSW graph.
type Node struct {
	ID        int     // user-provided ID
	idx       int     // internal index in storage
	level     int     // maximum layer
	neighbors [][]int // neighbor indices per level
}

// NewNode creates a new node with properly initialized neighbor slices.
func NewNode(id, idx, level int) *Node {
	neighbors := make([][]int, level+1)
	// Pre-allocate neighbor slices to reduce allocations
	for i := range neighbors {
		neighbors[i] = make([]int, 0, 16) // Pre-allocate capacity
	}
	return &Node{
		ID:        id,
		idx:       idx,
		level:     level,
		neighbors: neighbors,
	}
}

// Graph is the main HNSW index structure.
type Graph struct {
	// HNSW parameters
	m              int
	efConstruction int
	efSearch       int

	// chunked arrow storage for vectors (flat float64 values)
	dim       int
	chunkSize int
	allocator memory.Allocator
	vectors   []*array.Float64      // stores flat float64 values: rows * dim
	builder   *array.Float64Builder // current chunk builder

	// graph structure
	maxLevel   int
	enterPoint *Node
	nodes      []*Node
	idToIdx    map[int]int
	levelFunc  func() int // function to determine node level

	// pools for zero-allocation operations
	pqPool   sync.Pool // *minHeap
	resPool  sync.Pool // *maxHeap
	vecPool  sync.Pool // []float64 for getVector
	candPool sync.Pool // []*candidate for reuse
	intPool  sync.Pool // []int for neighbor lists

	mu sync.RWMutex
}

// NewGraph initializes an HNSW index.
func NewGraph(dim, m, efConstruction, efSearch, chunkSize int, alloc memory.Allocator) *Graph {
	maxEF := max(efSearch, efConstruction)
	g := &Graph{
		m:              m,
		efConstruction: efConstruction,
		efSearch:       efSearch,
		dim:            dim,
		chunkSize:      chunkSize,
		allocator:      alloc,
		vectors:        make([]*array.Float64, 0),
		builder:        array.NewFloat64Builder(alloc),
		nodes:          make([]*Node, 0),
		idToIdx:        make(map[int]int),
		levelFunc:      randomLevel, // default to random level
	}

	// Initialize pools for better memory management
	g.pqPool = sync.Pool{New: func() any {
		hs := make(minHeap, 0, maxEF)
		heap.Init(&hs)
		return &hs
	}}
	g.resPool = sync.Pool{New: func() any {
		hs := make(maxHeap, 0, maxEF)
		heap.Init(&hs)
		return &hs
	}}
	g.vecPool = sync.Pool{New: func() any {
		return make([]float64, dim)
	}}
	g.candPool = sync.Pool{New: func() any {
		return make([]*candidate, 0, maxEF)
	}}
	g.intPool = sync.Pool{New: func() any {
		return make([]int, 0, m*2)
	}}

	return g
}

// AddBatch inserts multiple points into the index efficiently.
func (g *Graph) AddBatch(items []struct {
	ID  int
	Vec []float64
}) error {
	if len(items) == 0 {
		return nil
	}

	g.mu.Lock()
	defer g.mu.Unlock()

	// Validate all vectors first
	for _, item := range items {
		if len(item.Vec) != g.dim {
			return fmt.Errorf("vector dimension mismatch: got %d, want %d", len(item.Vec), g.dim)
		}
	}

	// Batch append vectors to builder
	for _, item := range items {
		g.builder.AppendValues(item.Vec, nil)
		if g.builder.Len()/g.dim >= g.chunkSize {
			chunk := g.builder.NewArray().(*array.Float64)
			g.vectors = append(g.vectors, chunk)
			g.builder = array.NewFloat64Builder(g.allocator)
		}
	}

	// Process nodes in batch
	nodes := make([]*Node, len(items))
	startPos := len(g.nodes)

	for i, item := range items {
		pos := startPos + i
		g.idToIdx[item.ID] = pos
		nodes[i] = NewNode(item.ID, pos, g.levelFunc())

		// Update max level if needed
		if nodes[i].level > g.maxLevel {
			g.maxLevel = nodes[i].level
			g.enterPoint = nodes[i]
		}
	}

	// Add all nodes to graph first
	g.nodes = append(g.nodes, nodes...)

	// Connect nodes to graph
	for i, item := range items {
		n := nodes[i]
		pos := startPos + i

		// Skip first node
		if pos == 0 {
			g.enterPoint = n
			g.maxLevel = n.level
			continue
		}

		// Find enter point if none exists
		if g.enterPoint == nil {
			g.enterPoint = g.nodes[0]
			g.maxLevel = g.enterPoint.level
		}

		// Connect to existing graph
		g.connectNodeToGraph(n, item.Vec)
	}

	return nil
}

// connectNodeToGraph connects a node to the existing graph structure.
func (g *Graph) connectNodeToGraph(n *Node, vec []float64) {
	cur := g.enterPoint

	// Navigate down through levels
	for lvl := g.maxLevel; lvl > n.level; lvl-- {
		next := g.greedySearchLayerFast(vec, cur, lvl)
		if next != nil {
			cur = next
		}
	}

	// Connect at each level from top to bottom
	for lvl := min(n.level, g.maxLevel); lvl >= 0; lvl-- {
		candidates := g.searchLayerFast(vec, cur, lvl, g.efConstruction)
		if len(candidates) == 0 {
			candidates = []*candidate{{idx: cur.idx, dist: euclideanSquared(vec, g.getVectorFast(cur.idx))}}
		}

		nbrs := selectNeighborsFast(candidates, g.m)

		// Connect bidirectionally with optimized neighbor management
		for _, c := range nbrs {
			ni := c.idx
			if ni >= len(g.nodes) {
				continue
			}
			peer := g.nodes[ni]

			// Connect n -> peer
			if lvl < len(n.neighbors) {
				n.neighbors[lvl] = append(n.neighbors[lvl], ni)
			}

			// Connect peer -> n with pruning
			if lvl < len(peer.neighbors) {
				peer.neighbors[lvl] = append(peer.neighbors[lvl], n.idx)

				// Prune peer's neighbors if exceeding limit
				if len(peer.neighbors[lvl]) > g.m {
					g.pruneNeighbors(peer, lvl)
				}
			}
		}

		// Update current for next level
		next := g.greedySearchLayerFast(vec, cur, lvl)
		if next != nil {
			cur = next
		}
	}
}

// pruneNeighbors efficiently prunes neighbors using pooled resources.
func (g *Graph) pruneNeighbors(node *Node, lvl int) {
	if lvl >= len(node.neighbors) || len(node.neighbors[lvl]) <= g.m {
		return
	}

	// Get pooled candidate slice
	candidates := g.candPool.Get().([]*candidate)[:0]
	defer g.candPool.Put(candidates)

	// Calculate distances to all neighbors
	nodeVec := g.getVectorFast(node.idx)
	for _, nbrIdx := range node.neighbors[lvl] {
		if nbrIdx >= len(g.nodes) {
			continue
		}
		nbrVec := g.getVectorFast(nbrIdx)
		d := euclideanSquared(nodeVec, nbrVec)
		candidates = append(candidates, &candidate{idx: nbrIdx, dist: d})
	}

	// Select top m neighbors
	selected := selectNeighborsFast(candidates, g.m)

	// Get pooled int slice for new neighbors
	newNbrs := g.intPool.Get().([]int)[:0]
	defer g.intPool.Put(newNbrs)

	for _, s := range selected {
		newNbrs = append(newNbrs, s.idx)
	}

	// Replace neighbors
	node.neighbors[lvl] = node.neighbors[lvl][:len(newNbrs)]
	copy(node.neighbors[lvl], newNbrs)
}

// Add inserts a single point into the index.
func (g *Graph) Add(id int, vec []float64) error {
	return g.AddBatch([]struct {
		ID  int
		Vec []float64
	}{{ID: id, Vec: vec}})
}

// Search finds the k nearest neighbors to query.
func (g *Graph) Search(query []float64, k int) ([]int, error) {
	if len(query) != g.dim {
		return nil, fmt.Errorf("query dimension mismatch: got %d, want %d", len(query), g.dim)
	}
	g.mu.RLock()
	defer g.mu.RUnlock()

	if len(g.nodes) == 0 {
		return nil, nil
	}

	if k > len(g.nodes) {
		k = len(g.nodes)
	}

	// Flush builder if needed
	if g.builder.Len() > 0 {
		chunk := g.builder.NewArray().(*array.Float64)
		g.vectors = append(g.vectors, chunk)
		g.builder = array.NewFloat64Builder(g.allocator)
	}

	// For small graphs, do optimized exhaustive search
	if len(g.nodes) <= g.m {
		return g.exhaustiveSearch(query, k), nil
	}

	// Fast HNSW search
	return g.hnswSearch(query, k), nil
}

// exhaustiveSearch performs optimized exhaustive search for small graphs.
func (g *Graph) exhaustiveSearch(query []float64, k int) []int {
	candidates := g.candPool.Get().([]*candidate)[:0]
	defer g.candPool.Put(candidates)

	for i := range g.nodes {
		vec := g.getVectorFast(i)
		d := euclideanSquared(query, vec)
		candidates = append(candidates, &candidate{idx: i, dist: d})
	}

	top := selectNeighborsFast(candidates, k)
	out := make([]int, len(top))
	for i, c := range top {
		out[i] = g.nodes[c.idx].ID
	}
	return out
}

// hnswSearch performs optimized HNSW search.
func (g *Graph) hnswSearch(query []float64, k int) []int {
	ep := g.enterPoint
	if ep == nil {
		ep = g.nodes[0]
	}

	// Descent through layers
	for lvl := g.maxLevel; lvl > 0; lvl-- {
		next := g.greedySearchLayerFast(query, ep, lvl)
		if next != nil {
			ep = next
		}
	}

	// Search at level 0
	candidates := g.searchLayerFast(query, ep, 0, max(g.efSearch, k))
	if len(candidates) == 0 {
		return g.exhaustiveSearch(query, k)
	}

	top := selectNeighborsFast(candidates, k)
	out := make([]int, len(top))
	for i, c := range top {
		out[i] = g.nodes[c.idx].ID
	}
	return out
}

// greedySearchLayerFast performs optimized greedy search.
func (g *Graph) greedySearchLayerFast(vec []float64, entry *Node, lvl int) *Node {
	if entry == nil || lvl >= len(entry.neighbors) {
		return entry
	}

	cur := entry
	curVec := g.getVectorFast(cur.idx)
	dMin := euclideanSquared(vec, curVec)

	improved := true
	for improved {
		improved = false

		for _, ni := range cur.neighbors[lvl] {
			if ni >= len(g.nodes) {
				continue
			}

			nbrVec := g.getVectorFast(ni)
			d := euclideanSquared(vec, nbrVec)
			if d < dMin {
				dMin = d
				cur = g.nodes[ni]
				improved = true
				break // Take first improvement for speed
			}
		}
	}
	return cur
}

// searchLayerFast performs optimized layer search.
func (g *Graph) searchLayerFast(query []float64, entry *Node, lvl, ef int) []*candidate {
	if entry == nil {
		return nil
	}

	visited := make(map[int]struct{}, ef*2)

	pqPtr := g.pqPool.Get().(*minHeap)
	*pqPtr = (*pqPtr)[:0]
	resPtr := g.resPool.Get().(*maxHeap)
	*resPtr = (*resPtr)[:0]

	defer func() {
		g.pqPool.Put(pqPtr)
		g.resPool.Put(resPtr)
	}()

	entryVec := g.getVectorFast(entry.idx)
	d0 := euclideanSquared(query, entryVec)
	heap.Push(pqPtr, &candidate{idx: entry.idx, dist: d0})
	heap.Push(resPtr, &candidate{idx: entry.idx, dist: d0})
	visited[entry.idx] = struct{}{}

	for pqPtr.Len() > 0 {
		c := heap.Pop(pqPtr).(*candidate)
		if resPtr.Len() > 0 && c.dist > (*resPtr)[0].dist {
			break
		}

		node := g.nodes[c.idx]
		if lvl >= len(node.neighbors) {
			continue
		}

		for _, ni := range node.neighbors[lvl] {
			if ni >= len(g.nodes) {
				continue
			}
			if _, ok := visited[ni]; ok {
				continue
			}
			visited[ni] = struct{}{}

			nbrVec := g.getVectorFast(ni)
			d := euclideanSquared(query, nbrVec)

			if resPtr.Len() < ef {
				heap.Push(resPtr, &candidate{idx: ni, dist: d})
				heap.Push(pqPtr, &candidate{idx: ni, dist: d})
			} else if d < (*resPtr)[0].dist {
				heap.Pop(resPtr)
				heap.Push(resPtr, &candidate{idx: ni, dist: d})
				heap.Push(pqPtr, &candidate{idx: ni, dist: d})
			}
		}
	}

	out := make([]*candidate, resPtr.Len())
	for i := len(out) - 1; i >= 0; i-- {
		out[i] = heap.Pop(resPtr).(*candidate)
	}
	return out
}

// getVectorFast retrieves vector using pooled slice to avoid allocations.
func (g *Graph) getVectorFast(idx int) []float64 {
	vec := g.vecPool.Get().([]float64)
	defer g.vecPool.Put(vec)

	off := idx * g.dim
	o := off

	for _, chunk := range g.vectors {
		n := chunk.Len()
		if o < n {
			d := min(n-o, g.dim)
			for i := 0; i < d; i++ {
				vec[i] = chunk.Value(o + i)
			}
			// Return a copy since we're returning the slice to the pool
			result := make([]float64, g.dim)
			copy(result, vec[:g.dim])
			return result
		}
		o -= n
	}
	return make([]float64, g.dim)
}

// selectNeighborsFast optimized neighbor selection.
func selectNeighborsFast(cands []*candidate, m int) []*candidate {
	if len(cands) <= m {
		return cands
	}

	// Use partial sort for better performance
	if m < len(cands)/4 {
		// For small m, use heap-based selection
		return selectNeighborsHeap(cands, m)
	}

	// For larger m, use full sort
	sort.Slice(cands, func(i, j int) bool { return cands[i].dist < cands[j].dist })
	return cands[:m]
}

// selectNeighborsHeap uses heap-based selection for small m.
func selectNeighborsHeap(cands []*candidate, m int) []*candidate {
	if len(cands) <= m {
		return cands
	}

	// Build max heap of size m
	maxHeap := make(maxHeap, 0, m)

	for _, c := range cands {
		if maxHeap.Len() < m {
			heap.Push(&maxHeap, c)
		} else if c.dist < maxHeap[0].dist {
			heap.Pop(&maxHeap)
			heap.Push(&maxHeap, c)
		}
	}

	// Extract results
	result := make([]*candidate, maxHeap.Len())
	for i := len(result) - 1; i >= 0; i-- {
		result[i] = heap.Pop(&maxHeap).(*candidate)
	}

	return result
}

// euclideanSquared computes squared euclidean distance (faster, same ordering).
func euclideanSquared(a, b []float64) float64 {
	var sum float64
	// Unroll loop for better performance on common dimensions
	i := 0
	for i < len(a)-3 {
		d0 := a[i] - b[i]
		d1 := a[i+1] - b[i+1]
		d2 := a[i+2] - b[i+2]
		d3 := a[i+3] - b[i+3]
		sum += d0*d0 + d1*d1 + d2*d2 + d3*d3
		i += 4
	}
	for i < len(a) {
		d := a[i] - b[i]
		sum += d * d
		i++
	}
	return sum
}

// candidate for search.
type candidate struct {
	idx  int
	dist float64
}

// minHeap for PQ.
type minHeap []*candidate

func (h minHeap) Len() int           { return len(h) }
func (h minHeap) Less(i, j int) bool { return h[i].dist < h[j].dist }
func (h minHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *minHeap) Push(x any)        { *h = append(*h, x.(*candidate)) }
func (h *minHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// maxHeap for results.
type maxHeap []*candidate

func (h maxHeap) Len() int           { return len(h) }
func (h maxHeap) Less(i, j int) bool { return h[i].dist > h[j].dist }
func (h maxHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *maxHeap) Push(x any)        { *h = append(*h, x.(*candidate)) }
func (h *maxHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// Len returns the number of indexed points.
func (g *Graph) Len() int {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return len(g.nodes)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// randomLevel samples a layer.
func randomLevel() int {
	lvl := 0
	for rand.Float64() < 1.0/math.E {
		lvl++
	}
	return lvl
}

// GetVector returns the vector at the given internal index.
func (g *Graph) GetVector(idx int) []float64 {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.getVectorFast(idx)
}

// GetVectorByID returns the vector for the given external ID.
func (g *Graph) GetVectorByID(id int) ([]float64, bool) {
	g.mu.RLock()
	defer g.mu.RUnlock()
	idx, ok := g.idToIdx[id]
	if !ok {
		return nil, false
	}
	return g.getVectorFast(idx), true
}
