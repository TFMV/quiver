package arrowindex

import (
	"container/heap"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"

	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"
)

// VisitedList represents a reusable visited tracking structure inspired by hnswlib
type VisitedList struct {
	curV uint16
	mass []uint16
}

// NewVisitedList creates a new visited list with the given capacity
func NewVisitedList(capacity int) *VisitedList {
	return &VisitedList{
		curV: 1,
		mass: make([]uint16, capacity),
	}
}

// Reset prepares the visited list for reuse
func (vl *VisitedList) Reset() {
	vl.curV++
	if vl.curV == 0 {
		for i := range vl.mass {
			vl.mass[i] = 0
		}
		vl.curV = 1
	}
}

// IsVisited checks if an index has been visited
func (vl *VisitedList) IsVisited(idx int) bool {
	if idx >= len(vl.mass) {
		oldLen := len(vl.mass)
		newMass := make([]uint16, idx+1)
		copy(newMass, vl.mass)
		vl.mass = newMass
		for i := oldLen; i < len(newMass); i++ {
			vl.mass[i] = 0
		}
	}
	return vl.mass[idx] == vl.curV
}

// Visit marks an index as visited
func (vl *VisitedList) Visit(idx int) {
	if idx >= len(vl.mass) {
		oldLen := len(vl.mass)
		newMass := make([]uint16, idx+1)
		copy(newMass, vl.mass)
		vl.mass = newMass
		for i := oldLen; i < len(newMass); i++ {
			vl.mass[i] = 0
		}
	}
	vl.mass[idx] = vl.curV
}

// VisitedListPool manages a pool of VisitedList instances
type VisitedListPool struct {
	pool    []*VisitedList
	mutex   sync.Mutex
	maxSize int
}

// NewVisitedListPool creates a new visited list pool
func NewVisitedListPool(maxSize int) *VisitedListPool {
	pool := &VisitedListPool{
		pool:    make([]*VisitedList, 0, 4),
		maxSize: maxSize,
	}
	for i := 0; i < 2; i++ {
		pool.pool = append(pool.pool, NewVisitedList(maxSize))
	}
	return pool
}

// Get retrieves a visited list from the pool
func (vlp *VisitedListPool) Get() *VisitedList {
	vlp.mutex.Lock()
	defer vlp.mutex.Unlock()

	if len(vlp.pool) > 0 {
		vl := vlp.pool[len(vlp.pool)-1]
		vlp.pool = vlp.pool[:len(vlp.pool)-1]
		vl.Reset()
		return vl
	}
	return NewVisitedList(vlp.maxSize)
}

// Return returns a visited list to the pool
func (vlp *VisitedListPool) Return(vl *VisitedList) {
	vlp.mutex.Lock()
	defer vlp.mutex.Unlock()

	if len(vlp.pool) < 8 {
		vlp.pool = append(vlp.pool, vl)
	}
}

// Node represents a point in the HNSW graph.
type Node struct {
	ID        int
	idx       int
	level     int
	neighbors [][]int
}

// NewNode creates a new node with properly initialized neighbor slices.
func NewNode(id, idx, level int) *Node {
	neighbors := make([][]int, level+1)
	for i := range neighbors {
		neighbors[i] = make([]int, 0, 32)
	}
	return &Node{ID: id, idx: idx, level: level, neighbors: neighbors}
}

// Graph is the main HNSW index structure.
type Graph struct {
	m              int
	efConstruction int
	efSearch       int
	ml             float64

	dim       int
	chunkSize int
	allocator memory.Allocator
	vectors   []*array.Float64
	builder   *array.Float64Builder

	maxLevel   int
	enterPoint *Node
	nodes      []*Node
	idToIdx    map[int]int
	levelFunc  func() int

	pqPool      sync.Pool
	resPool     sync.Pool
	vecPool     sync.Pool
	candPool    sync.Pool
	intPool     sync.Pool
	visitedPool *VisitedListPool

	mu sync.RWMutex
}

// NewGraph initializes an HNSW index.
func NewGraph(dim, m, efConstruction, efSearch, chunkSize int, alloc memory.Allocator) *Graph {
	if efConstruction < m {
		efConstruction = m
	}
	maxEF := max(efSearch, efConstruction)
	g := &Graph{
		m:              m,
		efConstruction: efConstruction,
		efSearch:       efSearch,
		ml:             1.0 / math.Log(2.0),
		dim:            dim,
		chunkSize:      chunkSize,
		allocator:      alloc,
		vectors:        make([]*array.Float64, 0),
		builder:        array.NewFloat64Builder(alloc),
		nodes:          make([]*Node, 0),
		idToIdx:        make(map[int]int),
		visitedPool:    NewVisitedListPool(50000),
	}

	g.levelFunc = func() int { return g.randomLevel() }

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
	g.vecPool = sync.Pool{New: func() any { return make([]float64, dim) }}
	g.candPool = sync.Pool{New: func() any { return make([]*candidate, 0, maxEF) }}
	g.intPool = sync.Pool{New: func() any { return make([]int, 0, m*4) }}

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

	for _, item := range items {
		if len(item.Vec) != g.dim {
			return fmt.Errorf("vector dimension mismatch: got %d, want %d", len(item.Vec), g.dim)
		}
	}

	for _, item := range items {
		g.builder.AppendValues(item.Vec, nil)
		if g.builder.Len()/g.dim >= g.chunkSize {
			chunk := g.builder.NewArray().(*array.Float64)
			g.vectors = append(g.vectors, chunk)
			g.builder = array.NewFloat64Builder(g.allocator)
		}
	}

	nodes := make([]*Node, len(items))
	startPos := len(g.nodes)

	for i, item := range items {
		pos := startPos + i
		g.idToIdx[item.ID] = pos
		nodes[i] = NewNode(item.ID, pos, g.levelFunc())
		if nodes[i].level > g.maxLevel {
			g.maxLevel = nodes[i].level
			g.enterPoint = nodes[i]
		}
	}

	g.nodes = append(g.nodes, nodes...)

	for i, item := range items {
		n := nodes[i]
		pos := startPos + i

		if pos == 0 {
			g.enterPoint = n
			g.maxLevel = n.level
			continue
		}

		if g.enterPoint == nil {
			g.enterPoint = g.nodes[0]
			g.maxLevel = g.enterPoint.level
		}

		g.connectNodeToGraph(n, item.Vec)
	}

	return nil
}

// connectNodeToGraph connects a node to the existing graph structure with enhanced hnswlib-style algorithm.
func (g *Graph) connectNodeToGraph(n *Node, vec []float64) {
	cur := g.enterPoint
	curDist := euclideanSquaredFast(vec, g.getVectorFast(cur.idx))

	for lvl := g.maxLevel; lvl > n.level; lvl-- {
		changed := true
		for changed {
			changed = false
			if lvl >= len(cur.neighbors) {
				break
			}
			for _, ni := range cur.neighbors[lvl] {
				if ni >= len(g.nodes) {
					continue
				}
				nbrVec := g.getVectorFast(ni)
				d := euclideanSquaredFast(vec, nbrVec)
				if d < curDist {
					curDist = d
					cur = g.nodes[ni]
					changed = true
					break
				}
			}
		}
	}

	for lvl := min(n.level, g.maxLevel); lvl >= 0; lvl-- {
		ef := g.efConstruction
		if lvl == 0 {
			ef = max(g.efConstruction, g.m*2)
		}

		candidates := g.searchLayerFast(vec, cur, lvl, ef)
		if len(candidates) == 0 {
			candidates = []*candidate{{idx: cur.idx, dist: curDist}}
		}

		mMax := g.m
		if lvl == 0 {
			mMax = g.m * 2
		}
		g.mutuallyConnectNewElement(n, candidates, lvl, mMax)

		if len(candidates) > 0 {
			cur = g.nodes[candidates[0].idx]
			curDist = candidates[0].dist
		}
	}
}

// mutuallyConnectNewElement performs bidirectional connections with enhanced pruning
func (g *Graph) mutuallyConnectNewElement(newNode *Node, candidates []*candidate, level, mMax int) []*candidate {
	selected := g.selectNeighborsHeuristic(candidates, g.m)

	if level < len(newNode.neighbors) {
		for _, c := range selected {
			newNode.neighbors[level] = append(newNode.neighbors[level], c.idx)
		}
	}

	for _, c := range selected {
		ni := c.idx
		if ni >= len(g.nodes) {
			continue
		}
		peer := g.nodes[ni]

		if level < len(peer.neighbors) {
			peer.neighbors[level] = append(peer.neighbors[level], newNode.idx)
			if len(peer.neighbors[level]) > mMax {
				g.pruneNeighborsHeuristic(peer, level)
			}
		}
	}

	return selected
}

// pruneNeighborsHeuristic efficiently prunes neighbors using improved hnswlib-style heuristic.
func (g *Graph) pruneNeighborsHeuristic(node *Node, lvl int) {
	mMax := g.m
	if lvl == 0 {
		mMax = g.m * 2
	}

	if lvl >= len(node.neighbors) || len(node.neighbors[lvl]) <= mMax {
		return
	}

	candidates := g.candPool.Get().([]*candidate)[:0]
	defer g.candPool.Put(candidates)

	nodeVec := g.getVectorFast(node.idx)
	for _, nbrIdx := range node.neighbors[lvl] {
		if nbrIdx >= len(g.nodes) {
			continue
		}
		nbrVec := g.getVectorFast(nbrIdx)
		d := euclideanSquaredFast(nodeVec, nbrVec)
		candidates = append(candidates, &candidate{idx: nbrIdx, dist: d})
	}

	var selected []*candidate
	if lvl == 0 && len(candidates) > mMax*3 {
		selected = g.selectNeighborsAdvancedHeuristic(candidates, mMax)
	} else {
		selected = g.selectNeighborsHeuristic(candidates, mMax)
	}

	newNbrs := g.intPool.Get().([]int)[:0]
	defer g.intPool.Put(newNbrs)

	for _, s := range selected {
		newNbrs = append(newNbrs, s.idx)
	}

	node.neighbors[lvl] = node.neighbors[lvl][:len(newNbrs)]
	copy(node.neighbors[lvl], newNbrs)
}

// selectNeighborsAdvancedHeuristic implements the full hnswlib neighbor selection heuristic
func (g *Graph) selectNeighborsAdvancedHeuristic(candidates []*candidate, m int) []*candidate {
	if len(candidates) <= m {
		return candidates
	}

	sort.Slice(candidates, func(i, j int) bool { return candidates[i].dist < candidates[j].dist })

	selected := make([]*candidate, 0, m)
	selected = append(selected, candidates[0])

	for i := 1; i < len(candidates) && len(selected) < m; i++ {
		cand := candidates[i]
		candVec := g.getVectorFast(cand.idx)

		shouldAdd := true
		for _, sel := range selected {
			selVec := g.getVectorFast(sel.idx)
			distToSel := euclideanSquaredFast(candVec, selVec)
			if distToSel < cand.dist {
				shouldAdd = false
				break
			}
		}

		if shouldAdd {
			selected = append(selected, cand)
		}
	}

	if len(selected) < m {
		for i := 1; i < len(candidates) && len(selected) < m; i++ {
			found := false
			for _, sel := range selected {
				if sel.idx == candidates[i].idx {
					found = true
					break
				}
			}
			if !found {
				selected = append(selected, candidates[i])
			}
		}
	}

	return selected
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

	if g.builder.Len() > 0 {
		chunk := g.builder.NewArray().(*array.Float64)
		g.vectors = append(g.vectors, chunk)
		g.builder = array.NewFloat64Builder(g.allocator)
	}

	if len(g.nodes) <= g.m {
		return g.exhaustiveSearch(query, k), nil
	}

	return g.hnswSearch(query, k), nil
}

// exhaustiveSearch performs optimized exhaustive search for small graphs.
func (g *Graph) exhaustiveSearch(query []float64, k int) []int {
	candidates := g.candPool.Get().([]*candidate)[:0]
	defer g.candPool.Put(candidates)

	for i := range g.nodes {
		vec := g.getVectorFast(i)
		d := euclideanSquaredFast(query, vec)
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

	for lvl := g.maxLevel; lvl > 0; lvl-- {
		next := g.greedySearchLayerFast(query, ep, lvl)
		if next != nil {
			ep = next
		}
	}

	ef := max(g.efSearch, k*2)
	candidates := g.searchLayerFast(query, ep, 0, ef)
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
			d := euclideanSquaredFast(vec, nbrVec)
			if d < dMin {
				dMin = d
				cur = g.nodes[ni]
				improved = true
				break
			}
		}
	}
	return cur
}

// searchLayerFast performs optimized layer search with better termination.
func (g *Graph) searchLayerFast(query []float64, entry *Node, lvl, ef int) []*candidate {
	if entry == nil {
		return nil
	}

	visited := g.visitedPool.Get()
	defer g.visitedPool.Return(visited)

	pqPtr := g.pqPool.Get().(*minHeap)
	*pqPtr = (*pqPtr)[:0]
	resPtr := g.resPool.Get().(*maxHeap)
	*resPtr = (*resPtr)[:0]

	defer func() {
		g.pqPool.Put(pqPtr)
		g.resPool.Put(resPtr)
	}()

	entryVec := g.getVectorFast(entry.idx)
	d0 := euclideanSquaredFast(query, entryVec)
	heap.Push(pqPtr, &candidate{idx: entry.idx, dist: d0})
	heap.Push(resPtr, &candidate{idx: entry.idx, dist: d0})
	visited.Visit(entry.idx)

	lowerBound := d0

	for pqPtr.Len() > 0 {
		c := heap.Pop(pqPtr).(*candidate)

		if resPtr.Len() >= ef && c.dist > lowerBound {
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
			if visited.IsVisited(ni) {
				continue
			}
			visited.Visit(ni)

			nbrVec := g.getVectorFast(ni)
			d := euclideanSquaredFast(query, nbrVec)

			if resPtr.Len() < ef {
				heap.Push(resPtr, &candidate{idx: ni, dist: d})
				heap.Push(pqPtr, &candidate{idx: ni, dist: d})
				if d < lowerBound {
					lowerBound = d
				}
			} else if d < (*resPtr)[0].dist {
				heap.Pop(resPtr)
				heap.Push(resPtr, &candidate{idx: ni, dist: d})
				heap.Push(pqPtr, &candidate{idx: ni, dist: d})
				lowerBound = (*resPtr)[0].dist
			}
		}
	}

	out := make([]*candidate, resPtr.Len())
	for i := len(out) - 1; i >= 0; i-- {
		out[i] = heap.Pop(resPtr).(*candidate)
	}
	return out
}

// getVectorFast retrieves vector with optimized memory access patterns.
func (g *Graph) getVectorFast(idx int) []float64 {
	result := make([]float64, g.dim)

	off := idx * g.dim
	o := off

	for chunkIdx, chunk := range g.vectors {
		n := chunk.Len()
		if o < n {
			d := min(n-o, g.dim)

			i := 0
			for i < d-4 {
				result[i] = chunk.Value(o + i)
				result[i+1] = chunk.Value(o + i + 1)
				result[i+2] = chunk.Value(o + i + 2)
				result[i+3] = chunk.Value(o + i + 3)
				i += 4
			}
			for i < d {
				result[i] = chunk.Value(o + i)
				i++
			}

			if d < g.dim {
				remaining := g.dim - d
				o = 0
				for j := chunkIdx + 1; j < len(g.vectors) && remaining > 0; j++ {
					nextChunk := g.vectors[j]
					nextN := nextChunk.Len()
					nextD := min(nextN, remaining)

					for k := 0; k < nextD; k++ {
						result[d+k] = nextChunk.Value(k)
					}
					remaining -= nextD
					d += nextD
					if remaining == 0 {
						break
					}
				}
			}
			return result
		}
		o -= n
	}
	return result
}

// selectNeighborsFast optimized neighbor selection.
func selectNeighborsFast(cands []*candidate, m int) []*candidate {
	if len(cands) <= m {
		return cands
	}

	if m < len(cands)/4 {
		return selectNeighborsHeap(cands, m)
	}

	sort.Slice(cands, func(i, j int) bool { return cands[i].dist < cands[j].dist })
	return cands[:m]
}

// selectNeighborsHeap uses heap-based selection for small m.
func selectNeighborsHeap(cands []*candidate, m int) []*candidate {
	if len(cands) <= m {
		return cands
	}

	maxHeap := make(maxHeap, 0, m)

	for _, c := range cands {
		if maxHeap.Len() < m {
			heap.Push(&maxHeap, c)
		} else if c.dist < maxHeap[0].dist {
			heap.Pop(&maxHeap)
			heap.Push(&maxHeap, c)
		}
	}

	result := make([]*candidate, maxHeap.Len())
	for i := len(result) - 1; i >= 0; i-- {
		result[i] = heap.Pop(&maxHeap).(*candidate)
	}

	return result
}

// euclideanSquared computes squared euclidean distance (faster, same ordering).
func euclideanSquared(a, b []float64) float64 {
	var sum float64
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

// euclideanSquaredFast computes squared euclidean distance with optimizations from hnswlib.
func euclideanSquaredFast(a, b []float64) float64 {
	var sum float64

	i := 0
	for i <= len(a)-8 {
		d0 := a[i] - b[i]
		d1 := a[i+1] - b[i+1]
		d2 := a[i+2] - b[i+2]
		d3 := a[i+3] - b[i+3]
		d4 := a[i+4] - b[i+4]
		d5 := a[i+5] - b[i+5]
		d6 := a[i+6] - b[i+6]
		d7 := a[i+7] - b[i+7]
		sum += d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6 + d7*d7
		i += 8
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

// randomLevel samples a layer with proper ml parameter.
func (g *Graph) randomLevel() int {
	lvl := 0
	for rand.Float64() < 1.0/math.E && lvl < 16 {
		lvl++
	}
	return lvl
}

// selectNeighborsHeuristic implements the improved neighbor selection heuristic.
func (g *Graph) selectNeighborsHeuristic(candidates []*candidate, m int) []*candidate {
	if len(candidates) <= m {
		return candidates
	}

	sort.Slice(candidates, func(i, j int) bool { return candidates[i].dist < candidates[j].dist })

	if len(candidates) <= m*2 {
		return candidates[:m]
	}

	selected := make([]*candidate, 0, m)
	selected = append(selected, candidates[0])

	for i := 1; i < len(candidates) && len(selected) < m; i++ {
		cand := candidates[i]
		shouldAdd := true

		if len(selected) > 0 {
			closest := selected[0]
			candVec := g.getVectorFast(cand.idx)
			closestVec := g.getVectorFast(closest.idx)
			distToClosest := euclideanSquaredFast(candVec, closestVec)
			if distToClosest < cand.dist*0.9 {
				shouldAdd = false
			}
		}

		if shouldAdd {
			selected = append(selected, cand)
		}
	}

	return selected
}
