package router

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/TFMV/quiver"
	"github.com/google/uuid"
	"go.uber.org/zap"
)

// IndexType represents the type of index for specialized routing
type IndexType string

const (
	// Common index types
	GeneralIndex    IndexType = "general"
	TechnicalIndex  IndexType = "technical"
	CreativeIndex   IndexType = "creative"
	FinancialIndex  IndexType = "financial"
	ScientificIndex IndexType = "scientific"
	LegalIndex      IndexType = "legal"
	MedicalIndex    IndexType = "medical"
)

// RouterConfig contains configuration for the semantic router
type RouterConfig struct {
	// Default index to use when no suitable index is found
	DefaultIndex IndexType
	// Threshold for routing confidence (0.0-1.0)
	ConfidenceThreshold float32
	// Whether to log routing decisions
	EnableLogging bool
	// Whether to cache routing decisions
	EnableCache bool
	// Maximum size of the routing cache
	CacheSize int
	// TTL for cache entries
	CacheTTL time.Duration
	// Whether to use parallel routing
	ParallelRouting bool
	// Timeout for routing decisions
	RoutingTimeout time.Duration
	// Whether to fallback to default index on timeout
	FallbackOnTimeout bool
	// Whether to learn from routing decisions
	EnableLearning bool
	// Learning rate for updating routing models
	LearningRate float32
	// Dimension of the routing embeddings
	RoutingDimension int
}

// DefaultRouterConfig returns a default configuration for the router
func DefaultRouterConfig() RouterConfig {
	return RouterConfig{
		DefaultIndex:        GeneralIndex,
		ConfidenceThreshold: 0.75,
		EnableLogging:       true,
		EnableCache:         true,
		CacheSize:           10000,
		CacheTTL:            time.Hour,
		ParallelRouting:     true,
		RoutingTimeout:      time.Second * 2,
		FallbackOnTimeout:   true,
		EnableLearning:      true,
		LearningRate:        0.01,
		RoutingDimension:    128,
	}
}

// RoutingDecision represents the result of a routing decision
type RoutingDecision struct {
	TargetIndex        IndexType
	Confidence         float32
	AlternativeIndices map[IndexType]float32
	DecisionTime       time.Duration
	DecisionID         string
	Timestamp          time.Time
}

// IndexRegistry maintains a registry of available indices
type IndexRegistry struct {
	indices    map[IndexType]*quiver.Index
	embeddings map[IndexType][]float32
	lock       sync.RWMutex
	logger     *zap.Logger
}

// NewIndexRegistry creates a new index registry
func NewIndexRegistry(logger *zap.Logger) *IndexRegistry {
	return &IndexRegistry{
		indices:    make(map[IndexType]*quiver.Index),
		embeddings: make(map[IndexType][]float32),
		logger:     logger,
	}
}

// RegisterIndex registers an index with the registry
func (r *IndexRegistry) RegisterIndex(indexType IndexType, index *quiver.Index, embedding []float32) error {
	r.lock.Lock()
	defer r.lock.Unlock()

	if index == nil {
		return errors.New("cannot register nil index")
	}

	if _, exists := r.indices[indexType]; exists {
		return fmt.Errorf("index type %s already registered", indexType)
	}

	r.indices[indexType] = index

	// Store the embedding for this index type
	if embedding != nil {
		r.embeddings[indexType] = embedding
	}

	r.logger.Info("Registered index", zap.String("type", string(indexType)))
	return nil
}

// GetIndex retrieves an index from the registry
func (r *IndexRegistry) GetIndex(indexType IndexType) (*quiver.Index, error) {
	r.lock.RLock()
	defer r.lock.RUnlock()

	index, exists := r.indices[indexType]
	if !exists {
		return nil, fmt.Errorf("index type %s not found", indexType)
	}

	return index, nil
}

// GetAllIndices returns all registered indices
func (r *IndexRegistry) GetAllIndices() map[IndexType]*quiver.Index {
	r.lock.RLock()
	defer r.lock.RUnlock()

	// Create a copy to avoid race conditions
	result := make(map[IndexType]*quiver.Index, len(r.indices))
	for k, v := range r.indices {
		result[k] = v
	}

	return result
}

// GetIndexEmbedding returns the embedding for an index type
func (r *IndexRegistry) GetIndexEmbedding(indexType IndexType) ([]float32, error) {
	r.lock.RLock()
	defer r.lock.RUnlock()

	embedding, exists := r.embeddings[indexType]
	if !exists {
		return nil, fmt.Errorf("embedding for index type %s not found", indexType)
	}

	return embedding, nil
}

// GetAllEmbeddings returns all registered embeddings
func (r *IndexRegistry) GetAllEmbeddings() map[IndexType][]float32 {
	r.lock.RLock()
	defer r.lock.RUnlock()

	// Create a copy to avoid race conditions
	result := make(map[IndexType][]float32, len(r.embeddings))
	for k, v := range r.embeddings {
		result[k] = v
	}

	return result
}

// cacheEntry represents an entry in the routing cache
type cacheEntry struct {
	decision  RoutingDecision
	timestamp time.Time
}

// SemanticRouter routes queries to the appropriate index based on content
type SemanticRouter struct {
	config         RouterConfig
	registry       *IndexRegistry
	routingIndex   *quiver.Index
	routingCache   map[string]cacheEntry
	cacheLock      sync.RWMutex
	logger         *zap.Logger
	metrics        *RouterMetrics
	learningBuffer []learningExample
	learningLock   sync.Mutex
}

// learningExample represents an example for updating the routing model
type learningExample struct {
	query    []float32
	decision IndexType
	feedback float32 // 0.0-1.0, where 1.0 is positive feedback
}

// RouterMetrics tracks metrics for the router
type RouterMetrics struct {
	TotalRequests       int64
	CacheHits           int64
	CacheMisses         int64
	RoutingErrors       int64
	RoutingTimeouts     int64
	ConfidenceThreshold float32
	AvgRoutingTime      time.Duration
	RoutingDecisions    map[IndexType]int64
	lock                *sync.Mutex
}

// NewSemanticRouter creates a new semantic router
func NewSemanticRouter(config RouterConfig, registry *IndexRegistry, logger *zap.Logger) (*SemanticRouter, error) {
	if registry == nil {
		return nil, errors.New("index registry cannot be nil")
	}

	// Create a routing index for storing index type embeddings
	routingConfig := quiver.Config{
		Dimension:       config.RoutingDimension,
		Distance:        quiver.CosineDistance,
		HNSWM:           16,
		HNSWEfConstruct: 200,
		HNSWEfSearch:    100,
	}

	routingIndex, err := quiver.New(routingConfig, logger)
	if err != nil {
		return nil, fmt.Errorf("failed to create routing index: %w", err)
	}

	router := &SemanticRouter{
		config:       config,
		registry:     registry,
		routingIndex: routingIndex,
		routingCache: make(map[string]cacheEntry),
		logger:       logger,
		metrics: &RouterMetrics{
			RoutingDecisions:    make(map[IndexType]int64),
			ConfidenceThreshold: config.ConfidenceThreshold,
			lock:                &sync.Mutex{},
		},
		learningBuffer: make([]learningExample, 0, 100),
	}

	// Initialize the routing index with embeddings from the registry
	embeddings := registry.GetAllEmbeddings()
	for indexType, embedding := range embeddings {
		// Use the index type as the ID
		id := uint64(fnv1a(string(indexType)))
		meta := map[string]interface{}{
			"type": string(indexType),
		}

		if err := routingIndex.Add(id, embedding, meta); err != nil {
			return nil, fmt.Errorf("failed to add embedding for index type %s: %w", indexType, err)
		}
	}

	// Start background tasks if needed
	if config.EnableCache {
		go router.startCacheCleanup()
	}

	if config.EnableLearning {
		go router.startLearningUpdates()
	}

	return router, nil
}

// fnv1a computes a 64-bit FNV-1a hash of the given string
func fnv1a(s string) uint64 {
	const (
		offset64 = 14695981039346656037
		prime64  = 1099511628211
	)
	hash := uint64(offset64)
	for i := 0; i < len(s); i++ {
		hash ^= uint64(s[i])
		hash *= prime64
	}
	return hash
}

// startCacheCleanup periodically cleans up expired cache entries
func (r *SemanticRouter) startCacheCleanup() {
	ticker := time.NewTicker(r.config.CacheTTL / 2)
	defer ticker.Stop()

	for range ticker.C {
		r.cleanupCache()
	}
}

// cleanupCache removes expired entries from the cache
func (r *SemanticRouter) cleanupCache() {
	r.cacheLock.Lock()
	defer r.cacheLock.Unlock()

	now := time.Now()
	for key, entry := range r.routingCache {
		if now.Sub(entry.timestamp) > r.config.CacheTTL {
			delete(r.routingCache, key)
		}
	}
}

// startLearningUpdates periodically updates the routing model based on feedback
func (r *SemanticRouter) startLearningUpdates() {
	ticker := time.NewTicker(time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		r.updateRoutingModel()
	}
}

// updateRoutingModel updates the routing model based on collected examples
func (r *SemanticRouter) updateRoutingModel() {
	r.learningLock.Lock()
	examples := r.learningBuffer
	r.learningBuffer = make([]learningExample, 0, 100)
	r.learningLock.Unlock()

	if len(examples) == 0 {
		return
	}

	r.logger.Info("Updating routing model", zap.Int("examples", len(examples)))

	// Group examples by decision
	examplesByDecision := make(map[IndexType][]learningExample)
	for _, ex := range examples {
		examplesByDecision[ex.decision] = append(examplesByDecision[ex.decision], ex)
	}

	// Update embeddings for each index type
	for indexType, exs := range examplesByDecision {
		// Get current embedding
		currentEmbedding, err := r.registry.GetIndexEmbedding(indexType)
		if err != nil {
			r.logger.Error("Failed to get embedding for index type",
				zap.String("type", string(indexType)),
				zap.Error(err))
			continue
		}

		// Compute average update direction
		update := make([]float32, len(currentEmbedding))
		totalWeight := float32(0)

		for _, ex := range exs {
			weight := ex.feedback
			totalWeight += weight

			for i := 0; i < len(update); i++ {
				if i < len(ex.query) {
					update[i] += weight * (ex.query[i] - currentEmbedding[i])
				}
			}
		}

		// Apply update with learning rate
		if totalWeight > 0 {
			for i := 0; i < len(update); i++ {
				update[i] = update[i] / totalWeight
				currentEmbedding[i] += r.config.LearningRate * update[i]
			}

			// Normalize the embedding
			norm := float32(0)
			for _, v := range currentEmbedding {
				norm += v * v
			}
			norm = float32(math.Sqrt(float64(norm)))

			if norm > 0 {
				for i := range currentEmbedding {
					currentEmbedding[i] /= norm
				}
			}

			// Update the embedding in the registry and routing index
			id := uint64(fnv1a(string(indexType)))
			meta := map[string]interface{}{
				"type": string(indexType),
			}

			if err := r.routingIndex.DeleteVector(id); err != nil {
				r.logger.Error("Failed to delete old embedding",
					zap.String("type", string(indexType)),
					zap.Error(err))
			}

			if err := r.routingIndex.Add(id, currentEmbedding, meta); err != nil {
				r.logger.Error("Failed to update embedding",
					zap.String("type", string(indexType)),
					zap.Error(err))
			}
		}
	}
}

// Route routes a query to the appropriate index
func (r *SemanticRouter) Route(ctx context.Context, query []float32) (RoutingDecision, error) {
	start := time.Now()

	// Create a unique decision ID
	decisionID := uuid.New().String()

	// Update metrics
	r.metrics.lock.Lock()
	r.metrics.TotalRequests++
	r.metrics.lock.Unlock()

	// Check cache if enabled
	if r.config.EnableCache {
		cacheKey := computeCacheKey(query)
		cachedDecision, found := r.getCachedDecision(cacheKey)
		if found {
			r.metrics.lock.Lock()
			r.metrics.CacheHits++
			r.metrics.lock.Unlock()

			// Update the decision ID and timestamp
			cachedDecision.DecisionID = decisionID
			cachedDecision.Timestamp = time.Now()

			return cachedDecision, nil
		}

		r.metrics.lock.Lock()
		r.metrics.CacheMisses++
		r.metrics.lock.Unlock()
	}

	// Create a context with timeout if configured
	var routingCtx context.Context
	var cancel context.CancelFunc

	if r.config.RoutingTimeout > 0 {
		routingCtx, cancel = context.WithTimeout(ctx, r.config.RoutingTimeout)
		defer cancel()
	} else {
		routingCtx = ctx
	}

	// Perform the routing decision
	decision, err := r.makeRoutingDecision(routingCtx, query)
	if err != nil {
		r.metrics.lock.Lock()
		r.metrics.RoutingErrors++
		r.metrics.lock.Unlock()

		// Check if it's a timeout error
		if errors.Is(err, context.DeadlineExceeded) {
			r.metrics.lock.Lock()
			r.metrics.RoutingTimeouts++
			r.metrics.lock.Unlock()

			r.logger.Warn("Routing decision timed out", zap.Duration("timeout", r.config.RoutingTimeout))

			// Fallback to default index if configured
			if r.config.FallbackOnTimeout {
				return RoutingDecision{
					TargetIndex:  r.config.DefaultIndex,
					Confidence:   0.0,
					DecisionTime: time.Since(start),
					DecisionID:   decisionID,
					Timestamp:    time.Now(),
				}, nil
			}
		}

		return RoutingDecision{}, err
	}

	// Set decision metadata
	decision.DecisionTime = time.Since(start)
	decision.DecisionID = decisionID
	decision.Timestamp = time.Now()

	// Update metrics
	r.metrics.lock.Lock()
	r.metrics.RoutingDecisions[decision.TargetIndex]++
	r.metrics.AvgRoutingTime = (r.metrics.AvgRoutingTime*time.Duration(r.metrics.TotalRequests-1) + decision.DecisionTime) / time.Duration(r.metrics.TotalRequests)
	r.metrics.lock.Unlock()

	// Cache the decision if enabled
	if r.config.EnableCache {
		cacheKey := computeCacheKey(query)
		r.cacheDecision(cacheKey, decision)
	}

	// Log the decision if enabled
	if r.config.EnableLogging {
		r.logger.Info("Routing decision",
			zap.String("decisionID", decision.DecisionID),
			zap.String("targetIndex", string(decision.TargetIndex)),
			zap.Float32("confidence", decision.Confidence),
			zap.Duration("decisionTime", decision.DecisionTime))
	}

	return decision, nil
}

// makeRoutingDecision performs the actual routing decision
func (r *SemanticRouter) makeRoutingDecision(ctx context.Context, query []float32) (RoutingDecision, error) {
	// Search the routing index for the most similar index types
	results, err := r.routingIndex.Search(query, 5, 0, 0)
	if err != nil {
		return RoutingDecision{}, fmt.Errorf("failed to search routing index: %w", err)
	}

	if len(results) == 0 {
		return RoutingDecision{
			TargetIndex: r.config.DefaultIndex,
			Confidence:  0.0,
		}, nil
	}

	// Convert search results to routing decision
	decision := RoutingDecision{
		AlternativeIndices: make(map[IndexType]float32),
	}

	// The top result is the target index
	topResult := results[0]
	indexType, ok := topResult.Metadata["type"].(string)
	if !ok {
		return RoutingDecision{}, errors.New("invalid index type in routing result")
	}

	decision.TargetIndex = IndexType(indexType)
	decision.Confidence = 1.0 - topResult.Distance // Convert distance to confidence

	// Store alternative indices
	for i := 1; i < len(results); i++ {
		altType, ok := results[i].Metadata["type"].(string)
		if !ok {
			continue
		}

		decision.AlternativeIndices[IndexType(altType)] = 1.0 - results[i].Distance
	}

	// If confidence is below threshold, use default index
	if decision.Confidence < r.config.ConfidenceThreshold {
		decision.TargetIndex = r.config.DefaultIndex
		decision.Confidence = 0.0
	}

	return decision, nil
}

// computeCacheKey computes a cache key for a query
func computeCacheKey(query []float32) string {
	// Simple hash-based key for now
	// In production, you might want a more sophisticated approach
	hash := fnv1a(fmt.Sprintf("%v", query))
	return fmt.Sprintf("%d", hash)
}

// getCachedDecision retrieves a cached routing decision
func (r *SemanticRouter) getCachedDecision(key string) (RoutingDecision, bool) {
	r.cacheLock.RLock()
	defer r.cacheLock.RUnlock()

	entry, found := r.routingCache[key]
	if !found {
		return RoutingDecision{}, false
	}

	// Check if the entry has expired
	if time.Since(entry.timestamp) > r.config.CacheTTL {
		return RoutingDecision{}, false
	}

	return entry.decision, true
}

// cacheDecision caches a routing decision
func (r *SemanticRouter) cacheDecision(key string, decision RoutingDecision) {
	r.cacheLock.Lock()
	defer r.cacheLock.Unlock()

	// Check if cache is full
	if len(r.routingCache) >= r.config.CacheSize {
		// Simple eviction strategy: remove a random entry
		// In production, you might want LRU or another strategy
		for k := range r.routingCache {
			delete(r.routingCache, k)
			break
		}
	}

	r.routingCache[key] = cacheEntry{
		decision:  decision,
		timestamp: time.Now(),
	}
}

// ProvideRoutingFeedback provides feedback on a routing decision
func (r *SemanticRouter) ProvideRoutingFeedback(decisionID string, feedback float32) error {
	if !r.config.EnableLearning {
		return errors.New("learning is not enabled")
	}

	// Validate feedback
	if feedback < 0.0 || feedback > 1.0 {
		return errors.New("feedback must be between 0.0 and 1.0")
	}

	// Find the decision in the cache
	var decision RoutingDecision
	var query []float32
	found := false

	r.cacheLock.RLock()
	for _, entry := range r.routingCache {
		if entry.decision.DecisionID == decisionID {
			decision = entry.decision

			// Reconstruct the query from the cache key
			// This is a simplification; in production you might want to store the query
			found = true
			break
		}
	}
	r.cacheLock.RUnlock()

	if !found {
		return errors.New("decision not found")
	}

	// If we couldn't reconstruct the query, we can't provide feedback
	if query == nil {
		return errors.New("could not reconstruct query for feedback")
	}

	// Add to learning buffer
	r.learningLock.Lock()
	r.learningBuffer = append(r.learningBuffer, learningExample{
		query:    query,
		decision: decision.TargetIndex,
		feedback: feedback,
	})
	r.learningLock.Unlock()

	r.logger.Info("Received routing feedback",
		zap.String("decisionID", decisionID),
		zap.Float32("feedback", feedback))

	return nil
}

// GetMetrics returns a copy of the current router metrics
func (r *SemanticRouter) GetMetrics() RouterMetrics {
	r.metrics.lock.Lock()
	defer r.metrics.lock.Unlock()

	// Create a copy to avoid race conditions, but without the lock
	metricsCopy := RouterMetrics{
		TotalRequests:       r.metrics.TotalRequests,
		CacheHits:           r.metrics.CacheHits,
		CacheMisses:         r.metrics.CacheMisses,
		RoutingErrors:       r.metrics.RoutingErrors,
		RoutingTimeouts:     r.metrics.RoutingTimeouts,
		ConfidenceThreshold: r.metrics.ConfidenceThreshold,
		AvgRoutingTime:      r.metrics.AvgRoutingTime,
		RoutingDecisions:    make(map[IndexType]int64),
	}

	for k, v := range r.metrics.RoutingDecisions {
		metricsCopy.RoutingDecisions[k] = v
	}

	return metricsCopy
}

// Close closes the router and releases resources
func (r *SemanticRouter) Close() error {
	return r.routingIndex.Close()
}

// GetRegistry returns the underlying index registry
func (r *SemanticRouter) GetRegistry() *IndexRegistry {
	return r.registry
}

// BatchRoute routes multiple queries in parallel
func (r *SemanticRouter) BatchRoute(ctx context.Context, queries [][]float32) ([]RoutingDecision, error) {
	if !r.config.ParallelRouting {
		// Sequential routing
		decisions := make([]RoutingDecision, len(queries))
		for i, query := range queries {
			decision, err := r.Route(ctx, query)
			if err != nil {
				return nil, fmt.Errorf("failed to route query %d: %w", i, err)
			}
			decisions[i] = decision
		}
		return decisions, nil
	}

	// Parallel routing
	decisions := make([]RoutingDecision, len(queries))
	errors := make([]error, len(queries))
	var wg sync.WaitGroup

	for i, query := range queries {
		wg.Add(1)
		go func(idx int, q []float32) {
			defer wg.Done()
			decision, err := r.Route(ctx, q)
			if err != nil {
				errors[idx] = err
				return
			}
			decisions[idx] = decision
		}(i, query)
	}

	wg.Wait()

	// Check for errors
	for i, err := range errors {
		if err != nil {
			return nil, fmt.Errorf("failed to route query %d: %w", i, err)
		}
	}

	return decisions, nil
}

// GetTopIndices returns the top N indices for a query
func (r *SemanticRouter) GetTopIndices(ctx context.Context, query []float32, n int) (map[IndexType]float32, error) {
	// Search the routing index for the most similar index types
	results, err := r.routingIndex.Search(query, n, 0, 0)
	if err != nil {
		return nil, fmt.Errorf("failed to search routing index: %w", err)
	}

	indices := make(map[IndexType]float32)
	for _, result := range results {
		indexType, ok := result.Metadata["type"].(string)
		if !ok {
			continue
		}

		indices[IndexType(indexType)] = 1.0 - result.Distance
	}

	return indices, nil
}

// RoutingStats contains statistics about routing decisions
type RoutingStats struct {
	TotalDecisions      int64
	DecisionsByIndex    map[IndexType]int64
	AvgConfidence       float32
	ConfidenceHistogram map[string]int64 // Buckets: "0.0-0.1", "0.1-0.2", etc.
	AvgDecisionTime     time.Duration
}

// GetRoutingStats returns statistics about routing decisions
func (r *SemanticRouter) GetRoutingStats() RoutingStats {
	r.metrics.lock.Lock()
	defer r.metrics.lock.Unlock()

	stats := RoutingStats{
		TotalDecisions:      r.metrics.TotalRequests,
		DecisionsByIndex:    make(map[IndexType]int64),
		AvgDecisionTime:     r.metrics.AvgRoutingTime,
		ConfidenceHistogram: make(map[string]int64),
	}

	for k, v := range r.metrics.RoutingDecisions {
		stats.DecisionsByIndex[k] = v
	}

	// We don't have confidence histogram data in the current implementation
	// This would require additional tracking

	return stats
}

// RegisterEmbeddingProvider registers a function that provides embeddings for queries
type EmbeddingProvider func(query string) ([]float32, error)

// TextRouter is a wrapper around SemanticRouter that handles text queries
type TextRouter struct {
	router            *SemanticRouter
	embeddingProvider EmbeddingProvider
	logger            *zap.Logger
}

// NewTextRouter creates a new text router
func NewTextRouter(router *SemanticRouter, provider EmbeddingProvider, logger *zap.Logger) *TextRouter {
	return &TextRouter{
		router:            router,
		embeddingProvider: provider,
		logger:            logger,
	}
}

// RouteText routes a text query to the appropriate index
func (r *TextRouter) RouteText(ctx context.Context, query string) (RoutingDecision, error) {
	// Get embedding for the query
	embedding, err := r.embeddingProvider(query)
	if err != nil {
		return RoutingDecision{}, fmt.Errorf("failed to get embedding for query: %w", err)
	}

	// Route the embedding
	return r.router.Route(ctx, embedding)
}

// BatchRouteText routes multiple text queries in parallel
func (r *TextRouter) BatchRouteText(ctx context.Context, queries []string) ([]RoutingDecision, error) {
	// Get embeddings for all queries
	embeddings := make([][]float32, len(queries))
	errors := make([]error, len(queries))
	var wg sync.WaitGroup

	for i, query := range queries {
		wg.Add(1)
		go func(idx int, q string) {
			defer wg.Done()
			embedding, err := r.embeddingProvider(q)
			if err != nil {
				errors[idx] = err
				return
			}
			embeddings[idx] = embedding
		}(i, query)
	}

	wg.Wait()

	// Check for errors
	for i, err := range errors {
		if err != nil {
			return nil, fmt.Errorf("failed to get embedding for query %d: %w", i, err)
		}
	}

	// Route all embeddings
	return r.router.BatchRoute(ctx, embeddings)
}
