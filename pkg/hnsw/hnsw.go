package hnsw

import (
	"container/heap"
	"errors"
	"fmt"
	"math/rand"
	"sort"
	"sync"
	"sync/atomic"
)

// DistanceFunction is a function type that computes the distance between two vectors
type DistanceFunction func(a, b []float32) (float32, error)

const (
	// DefaultM is the default number of bidirectional links created for each element
	DefaultM = 16
	// DefaultEfConstruction is the default size of the dynamic candidate list for construction
	DefaultEfConstruction = 200
	// DefaultEfSearch is the default size of the dynamic candidate list for search
	DefaultEfSearch = 100
	// DefaultMaxLevel is the default maximum level in the hierarchical graph
	DefaultMaxLevel = 16
)

// Config holds the configuration parameters for an HNSW index
type Config struct {
	// M defines the maximum number of connections per element in the graph
	M int
	// MaxM0 defines the maximum number of connections at layer 0
	MaxM0 int
	// EfConstruction controls the quality/time performance trade-off during index construction
	EfConstruction int
	// EfSearch controls the quality/time performance trade-off during search
	EfSearch int
	// MaxLevel is the maximum level in the hierarchical graph
	MaxLevel int
	// DistanceFunc specifies which distance function to use
	DistanceFunc DistanceFunction
}

// Node represents a single node in the graph
type Node struct {
	// VectorID is the unique identifier for the vector
	VectorID string
	// Vector holds the actual vector data
	Vector []float32
	// Connections holds the node connections at each level
	Connections [][]uint32
	// Level is the level of this node in the graph
	Level int
	// Lock for protecting concurrent modification of connections
	sync.RWMutex
}

// HNSW represents a hierarchical navigable small world graph
type HNSW struct {
	// Nodes stores all the nodes in the graph
	Nodes []*Node
	// NodesByID maps the vector ID to its index in Nodes slice
	NodesByID map[string]uint32
	// EntryPoint is the entry point to the graph
	EntryPoint uint32
	// M is the number of established connections
	M int
	// MaxM0 is the max number of connections for layer 0
	MaxM0 int
	// EfConstruction is the size of the dynamic candidate list for construction
	EfConstruction int
	// EfSearch is the size of the dynamic candidate list for search
	EfSearch int
	// MaxLevel is the maximum level in the graph
	MaxLevel int
	// CurrentLevel is the current maximum level in the graph
	CurrentLevel int
	// DistanceFunc is the distance function used
	DistanceFunc DistanceFunction
	// Lock for protecting concurrent modifications
	sync.RWMutex
	// size is the number of elements in the index
	size uint32
}

// Result represents a single search result
type Result struct {
	// VectorID is the ID of the vector
	VectorID string
	// Distance is the computed distance from the query vector
	Distance float32
	// VectorIndex is the internal index in the Nodes slice
	VectorIndex uint32
}

// ResultSet is a priority queue of search results
type ResultSet []Result

// Implement the heap.Interface
func (rs ResultSet) Len() int           { return len(rs) }
func (rs ResultSet) Less(i, j int) bool { return rs[i].Distance < rs[j].Distance }
func (rs ResultSet) Swap(i, j int)      { rs[i], rs[j] = rs[j], rs[i] }

func (rs *ResultSet) Push(x interface{}) {
	*rs = append(*rs, x.(Result))
}

func (rs *ResultSet) Pop() interface{} {
	old := *rs
	n := len(old)
	item := old[n-1]
	*rs = old[0 : n-1]
	return item
}

// NewHNSW creates a new HNSW index with the given config
func NewHNSW(config Config) *HNSW {
	if config.M <= 0 {
		config.M = DefaultM
	}
	if config.MaxM0 <= 0 {
		config.MaxM0 = config.M * 2
	}
	if config.EfConstruction <= 0 {
		config.EfConstruction = DefaultEfConstruction
	}
	if config.EfSearch <= 0 {
		config.EfSearch = DefaultEfSearch
	}
	if config.MaxLevel <= 0 {
		config.MaxLevel = DefaultMaxLevel
	}

	return &HNSW{
		Nodes:          make([]*Node, 0),
		NodesByID:      make(map[string]uint32),
		M:              config.M,
		MaxM0:          config.MaxM0,
		EfConstruction: config.EfConstruction,
		EfSearch:       config.EfSearch,
		MaxLevel:       config.MaxLevel,
		CurrentLevel:   -1,
		DistanceFunc:   config.DistanceFunc,
		EntryPoint:     0,
		size:           0,
	}
}

// Size returns the number of elements in the index
func (h *HNSW) Size() uint32 {
	return atomic.LoadUint32(&h.size)
}

// computeDistance calculates the distance between two vectors
func (h *HNSW) computeDistance(a, b []float32) (float32, error) {
	return h.DistanceFunc(a, b)
}

// Insert adds a new vector to the index
func (h *HNSW) Insert(id string, vector []float32) error {
	h.Lock()
	defer h.Unlock()

	// Check if the vector already exists
	if _, exists := h.NodesByID[id]; exists {
		return fmt.Errorf("vector with ID %s already exists", id)
	}

	// Generate random level for this node
	level := h.randomLevel()
	if level > h.CurrentLevel {
		h.CurrentLevel = level
	}

	// Create a new node
	newNodeIdx := uint32(len(h.Nodes))
	newNode := &Node{
		VectorID:    id,
		Vector:      vector,
		Connections: make([][]uint32, level+1),
		Level:       level,
	}

	// Initialize connections at each level
	for i := 0; i <= level; i++ {
		maxConnections := h.M
		if i == 0 {
			maxConnections = h.MaxM0
		}
		newNode.Connections[i] = make([]uint32, 0, maxConnections)
	}

	// Add the node to the graph
	h.Nodes = append(h.Nodes, newNode)
	h.NodesByID[id] = newNodeIdx
	atomic.AddUint32(&h.size, 1)

	// If this is the first node, make it the entry point and return
	if len(h.Nodes) == 1 {
		h.EntryPoint = 0
		return nil
	}

	// Connect the new node to the graph
	return h.connectNode(newNodeIdx, vector, level)
}

// connectNode connects a new node to the existing HNSW graph
func (h *HNSW) connectNode(nodeIdx uint32, vector []float32, level int) error {
	// Make sure level is within bounds
	if level >= h.MaxLevel {
		level = h.MaxLevel - 1
	}

	// If this is the first node, nothing to connect
	if len(h.Nodes) == 1 {
		h.EntryPoint = nodeIdx
		h.CurrentLevel = level
		return nil
	}

	entryPoint := h.EntryPoint

	// Safety check for entry point
	if int(entryPoint) >= len(h.Nodes) || h.Nodes[entryPoint] == nil {
		// If entry point is invalid, find another valid node
		for i, node := range h.Nodes {
			if i != int(nodeIdx) && node != nil {
				entryPoint = uint32(i)
				break
			}
		}
	}

	// Search for nearest neighbors on each level
	for lc := h.CurrentLevel; lc > level; lc-- {
		// Skip if current layer is beyond this node's level
		if lc >= len(h.Nodes[entryPoint].Connections) {
			continue
		}

		bestNode, err := h.searchLayer(vector, h.Nodes[entryPoint].Vector, entryPoint, 1, lc)
		if err != nil {
			return err
		}
		if len(bestNode) > 0 {
			entryPoint = bestNode[0].VectorIndex
		}
	}

	// For each level from top to bottom
	for lc := min(level, h.CurrentLevel); lc >= 0; lc-- {
		// Find the ef nearest neighbors in the current layer
		neighbors, err := h.searchLayer(vector, h.Nodes[entryPoint].Vector, entryPoint, h.EfConstruction, lc)
		if err != nil {
			return err
		}

		// Bail out if no neighbors found
		if len(neighbors) == 0 {
			continue
		}

		// Get the M nearest neighbors
		selectedNeighbors := h.selectNeighbors(neighbors, min(h.M, len(neighbors)))

		// Connect the new node to its neighbors
		newNode := h.Nodes[nodeIdx]
		newNode.Lock()

		// Ensure connections slice is large enough
		for len(newNode.Connections) <= lc {
			maxConns := h.M
			if len(newNode.Connections) == 0 {
				maxConns = h.MaxM0
			}
			newNode.Connections = append(newNode.Connections, make([]uint32, 0, maxConns))
		}

		for _, neighbor := range selectedNeighbors {
			newNode.Connections[lc] = append(newNode.Connections[lc], neighbor.VectorIndex)
		}
		newNode.Unlock()

		// Connect the neighbors back to the new node (bidirectional links)
		for _, neighbor := range selectedNeighbors {
			// Safety check
			if int(neighbor.VectorIndex) >= len(h.Nodes) || h.Nodes[neighbor.VectorIndex] == nil {
				continue
			}

			neighborNode := h.Nodes[neighbor.VectorIndex]
			neighborNode.Lock()

			// Ensure neighbor connections slice is large enough
			for len(neighborNode.Connections) <= lc {
				maxConns := h.M
				if len(neighborNode.Connections) == 0 {
					maxConns = h.MaxM0
				}
				neighborNode.Connections = append(neighborNode.Connections, make([]uint32, 0, maxConns))
			}

			// Check if we need to add the new node as a connection
			maxConnections := h.M
			if lc == 0 {
				maxConnections = h.MaxM0
			}

			// Add connection to the new node
			neighborNode.Connections[lc] = append(neighborNode.Connections[lc], nodeIdx)

			// If we exceed the maximum connections, need to prune
			if len(neighborNode.Connections[lc]) > maxConnections {
				// Get the distances from this neighbor to all its connections
				var neighborDists []Result
				for _, connIdx := range neighborNode.Connections[lc] {
					// Safety check
					if int(connIdx) >= len(h.Nodes) || h.Nodes[connIdx] == nil {
						continue
					}

					dist, err := h.computeDistance(neighborNode.Vector, h.Nodes[connIdx].Vector)
					if err != nil {
						neighborNode.Unlock()
						return err
					}
					neighborDists = append(neighborDists, Result{
						VectorIndex: connIdx,
						Distance:    dist,
					})
				}

				// Select the best M connections
				bestConns := h.selectNeighbors(neighborDists, maxConnections)

				// Update the connections
				neighborNode.Connections[lc] = make([]uint32, len(bestConns))
				for i, conn := range bestConns {
					neighborNode.Connections[lc][i] = conn.VectorIndex
				}
			}
			neighborNode.Unlock()
		}

		// Update entry point for the next layer
		if len(selectedNeighbors) > 0 {
			entryPoint = nodeIdx
		}
	}

	// Update the entry point if the new node has a higher level
	if level > h.CurrentLevel {
		h.EntryPoint = nodeIdx
		h.CurrentLevel = level
	}

	return nil
}

// searchLayer performs a greedy search in a single layer of the graph
func (h *HNSW) searchLayer(queryVector, entryVector []float32, entryPointID uint32, ef int, level int) ([]Result, error) {
	// Add safety check for empty index
	if len(h.Nodes) == 0 {
		return []Result{}, nil
	}

	// Add safety check for entry point
	if int(entryPointID) >= len(h.Nodes) || h.Nodes[entryPointID] == nil {
		return []Result{}, fmt.Errorf("invalid entry point ID: %d", entryPointID)
	}

	// Initialize visited set
	visited := make(map[uint32]bool)
	visited[entryPointID] = true

	// Initialize distance from query to entry point
	distance, err := h.computeDistance(queryVector, entryVector)
	if err != nil {
		return nil, err
	}

	// Initialize candidate set and result set
	candidates := &ResultSet{Result{VectorIndex: entryPointID, Distance: distance}}
	heap.Init(candidates)

	results := &ResultSet{Result{VectorIndex: entryPointID, Distance: distance}}
	heap.Init(results)

	// Continue search while candidates exist
	for candidates.Len() > 0 {
		// Get the closest candidate
		current := heap.Pop(candidates).(Result)

		// If the candidate is farther than the farthest in results, we can stop
		if results.Len() >= ef && current.Distance > (*results)[results.Len()-1].Distance {
			break
		}

		// Safety check for node access
		if int(current.VectorIndex) >= len(h.Nodes) || h.Nodes[current.VectorIndex] == nil {
			continue
		}

		// Get the node and explore its connections
		currentNode := h.Nodes[current.VectorIndex]
		currentNode.RLock()

		// Safety check for connections at this level
		if level >= len(currentNode.Connections) {
			currentNode.RUnlock()
			continue
		}

		connections := currentNode.Connections[level]
		currentNode.RUnlock()

		// Check each connection
		for _, connID := range connections {
			// Safety check - validate connection ID is in range
			if int(connID) >= len(h.Nodes) || h.Nodes[connID] == nil {
				continue
			}

			if !visited[connID] {
				visited[connID] = true

				// Compute distance to this connection
				connNode := h.Nodes[connID]
				connDist, err := h.computeDistance(queryVector, connNode.Vector)
				if err != nil {
					return nil, err
				}

				// If results not full or connection closer than furthest in results
				if results.Len() < ef || connDist < (*results)[results.Len()-1].Distance {
					heap.Push(candidates, Result{VectorIndex: connID, Distance: connDist})
					heap.Push(results, Result{VectorIndex: connID, Distance: connDist})

					// If results exceed ef, remove the furthest
					if results.Len() > ef {
						heap.Pop(results)
					}
				}
			}
		}
	}

	// Convert results to slice for returning
	resultSlice := make([]Result, results.Len())
	for i := results.Len() - 1; i >= 0; i-- {
		r := heap.Pop(results).(Result)

		// Safety check before accessing VectorID
		if int(r.VectorIndex) < len(h.Nodes) && h.Nodes[r.VectorIndex] != nil {
			r.VectorID = h.Nodes[r.VectorIndex].VectorID
		}

		resultSlice[i] = r
	}

	return resultSlice, nil
}

// selectNeighbors selects the k nearest neighbors from the candidates
func (h *HNSW) selectNeighbors(candidates []Result, k int) []Result {
	if len(candidates) <= k {
		return candidates
	}

	// Use simple heuristic: just return k closest
	results := make([]Result, k)
	pq := &ResultSet{}
	heap.Init(pq)

	// Add all candidates to a min heap
	for _, c := range candidates {
		heap.Push(pq, c)
		if pq.Len() > k {
			heap.Pop(pq)
		}
	}

	// Extract the k nearest neighbors
	for i := k - 1; i >= 0; i-- {
		if pq.Len() == 0 {
			break
		}
		results[i] = heap.Pop(pq).(Result)
	}

	return results
}

// Search finds the k nearest neighbors of the query vector
func (h *HNSW) Search(queryVector []float32, k int) ([]Result, error) {
	h.RLock()
	defer h.RUnlock()

	if len(h.Nodes) == 0 {
		return []Result{}, nil
	}

	if k <= 0 {
		return nil, errors.New("k must be positive")
	}

	// Limit k to the number of nodes
	if k > len(h.Nodes) {
		k = len(h.Nodes)
	}

	// Start from the entry point
	entryPointID := h.EntryPoint
	if entryPointID >= uint32(len(h.Nodes)) || h.Nodes[entryPointID] == nil {
		// Find a valid entry point
		for i, node := range h.Nodes {
			if node != nil {
				entryPointID = uint32(i)
				break
			}
		}
	}

	entryPoint := h.Nodes[entryPointID]
	if entryPoint == nil {
		return []Result{}, nil
	}

	// Calculate distance to entry point
	entryDistance, err := h.computeDistance(queryVector, entryPoint.Vector)
	if err != nil {
		return nil, err
	}

	// For each level, find the closest neighbors
	currentBest := Result{
		VectorID:    entryPoint.VectorID,
		Distance:    entryDistance,
		VectorIndex: entryPointID,
	}

	for level := h.CurrentLevel; level > 0; level-- {
		bestNodes, err := h.searchLayer(queryVector, entryPoint.Vector, entryPointID, 1, level)
		if err != nil || len(bestNodes) == 0 {
			continue
		}
		currentBest = bestNodes[0]
		entryPointID = currentBest.VectorIndex
		entryPoint = h.Nodes[entryPointID]
	}

	// Search the base layer with ef=k
	ef := h.EfSearch
	if ef < k {
		ef = k
	}
	results, err := h.searchLayer(queryVector, entryPoint.Vector, entryPointID, ef, 0)
	if err != nil {
		return nil, err
	}

	// Sort results by distance
	sort.Slice(results, func(i, j int) bool {
		return results[i].Distance < results[j].Distance
	})

	// Limit to k results
	if len(results) > k {
		results = results[:k]
	}

	return results, nil
}

// randomLevel generates a random level for a new node
func (h *HNSW) randomLevel() int {
	// Simplified level generation to avoid potential issues
	level := 0

	// Use a 1/4 probability for each higher level
	// This will produce a reasonable distribution without going too high
	maxAttempts := min(h.MaxLevel, 10) // Limit max attempts for sanity

	for i := 0; i < maxAttempts; i++ {
		if rand.Float64() < 0.25 {
			level++
		} else {
			break
		}
	}

	// Make sure we stay within bounds
	if level >= h.MaxLevel {
		level = h.MaxLevel - 1
	}

	return level
}

// Delete removes a vector from the index
func (h *HNSW) Delete(id string) error {
	h.Lock()
	defer h.Unlock()

	// Check if the vector exists
	idx, exists := h.NodesByID[id]
	if !exists {
		return fmt.Errorf("vector with ID %s not found", id)
	}

	// Safety check for node access
	if int(idx) >= len(h.Nodes) || h.Nodes[idx] == nil {
		return fmt.Errorf("vector index %d is invalid", idx)
	}

	// Get the node to be deleted
	nodeToDelete := h.Nodes[idx]

	// For each level and each connection, remove the reference to this node
	for level := 0; level <= nodeToDelete.Level; level++ {
		// Safety check for connections at this level
		if level >= len(nodeToDelete.Connections) {
			continue
		}

		for _, connIdx := range nodeToDelete.Connections[level] {
			// Safety check for connection index
			if int(connIdx) >= len(h.Nodes) || h.Nodes[connIdx] == nil {
				continue
			}

			connNode := h.Nodes[connIdx]
			connNode.Lock()

			// Safety check for connection's connections at this level
			if level < len(connNode.Connections) {
				// Remove the connection to the deleted node
				newConns := make([]uint32, 0, len(connNode.Connections[level]))
				for _, c := range connNode.Connections[level] {
					if c != idx {
						newConns = append(newConns, c)
					}
				}
				connNode.Connections[level] = newConns
			}

			connNode.Unlock()
		}
	}

	// If the node being deleted is the entry point, we need to update it
	if h.EntryPoint == idx {
		if len(h.Nodes) == 1 {
			// If this is the only node, reset the entry point
			h.EntryPoint = 0
			h.CurrentLevel = -1
		} else {
			// Otherwise, find a replacement entry point
			// Use the first connection at the highest level, or any node if none available
			replacementFound := false
			for level := nodeToDelete.Level; level >= 0 && !replacementFound; level-- {
				// Safety check for connections at this level
				if level < len(nodeToDelete.Connections) && len(nodeToDelete.Connections[level]) > 0 {
					// Safety check for the connection ID
					connID := nodeToDelete.Connections[level][0]
					if int(connID) < len(h.Nodes) && h.Nodes[connID] != nil {
						h.EntryPoint = connID
						h.CurrentLevel = level
						replacementFound = true
					}
				}
			}

			// If no replacement found at any level, use the first valid node
			if !replacementFound {
				for i, node := range h.Nodes {
					if uint32(i) != idx && node != nil {
						h.EntryPoint = uint32(i)
						h.CurrentLevel = node.Level
						break
					}
				}
			}
		}
	}

	// Mark the node as deleted by setting it to nil
	h.Nodes[idx] = nil

	// Remove it from the map
	delete(h.NodesByID, id)

	// Decrease size
	atomic.AddUint32(&h.size, ^uint32(0))

	return nil
}

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
