package extensions

import (
	"context"
	"fmt"

	"github.com/TFMV/quiver"
	"github.com/TFMV/quiver/dimreduce"
	"github.com/TFMV/quiver/router"
	"go.uber.org/zap"
)

// IndexWithDimReduce wraps a Quiver Index with dimensionality reduction capabilities
type IndexWithDimReduce struct {
	*quiver.Index
	dimReducer *dimreduce.DimReducer
}

// NewIndexWithDimReduce creates a new Index with dimensionality reduction
func NewIndexWithDimReduce(config quiver.Config, dimReduceConfig dimreduce.DimReducerConfig, logger *zap.Logger) (*IndexWithDimReduce, error) {
	idx, err := quiver.New(config, logger)
	if err != nil {
		return nil, err
	}

	// Set logger if not provided
	if dimReduceConfig.Logger == nil {
		dimReduceConfig.Logger = logger
	}

	dimReducer, err := dimreduce.NewDimReducer(dimReduceConfig)
	if err != nil {
		return nil, err
	}

	return &IndexWithDimReduce{
		Index:      idx,
		dimReducer: dimReducer,
	}, nil
}

// AddWithDimReduce adds a vector to the index with dimensionality reduction
func (idx *IndexWithDimReduce) AddWithDimReduce(id uint64, vector []float32, meta map[string]interface{}) error {
	// Apply dimensionality reduction to the vector
	reducedVectors, err := idx.dimReducer.Reduce([][]float32{vector})
	if err != nil {
		return fmt.Errorf("failed to reduce vector dimensions: %w", err)
	}

	// Add the reduced vector to the index
	return idx.Add(id, reducedVectors[0], meta)
}

// SearchWithDimReduce searches the index with dimensionality reduction
func (idx *IndexWithDimReduce) SearchWithDimReduce(query []float32, k, page, pageSize int) ([]quiver.SearchResult, error) {
	// Apply dimensionality reduction to the query
	reducedQueries, err := idx.dimReducer.Reduce([][]float32{query})
	if err != nil {
		return nil, fmt.Errorf("failed to reduce query dimensions: %w", err)
	}

	// Search with the reduced query
	return idx.Search(reducedQueries[0], k, page, pageSize)
}

// AdaptiveReduce applies adaptive dimensionality reduction to vectors
func (idx *IndexWithDimReduce) AdaptiveReduce(vectors [][]float32) ([][]float32, error) {
	return idx.dimReducer.AdaptiveReduce(vectors)
}

// GetDimReducer returns the underlying dimensionality reducer
func (idx *IndexWithDimReduce) GetDimReducer() *dimreduce.DimReducer {
	return idx.dimReducer
}

// MultiIndexManager manages multiple specialized indices
type MultiIndexManager struct {
	registry *router.IndexRegistry
	router   *router.SemanticRouter
	logger   *zap.Logger
}

// NewMultiIndexManager creates a new multi-index manager
func NewMultiIndexManager(routerConfig router.RouterConfig, logger *zap.Logger) (*MultiIndexManager, error) {
	registry := router.NewIndexRegistry(logger)

	semanticRouter, err := router.NewSemanticRouter(routerConfig, registry, logger)
	if err != nil {
		return nil, err
	}

	return &MultiIndexManager{
		registry: registry,
		router:   semanticRouter,
		logger:   logger,
	}, nil
}

// RegisterIndex registers an index with the manager
func (m *MultiIndexManager) RegisterIndex(indexType router.IndexType, index *quiver.Index, embedding []float32) error {
	return m.registry.RegisterIndex(indexType, index, embedding)
}

// GetIndex retrieves an index from the manager
func (m *MultiIndexManager) GetIndex(indexType router.IndexType) (*quiver.Index, error) {
	return m.registry.GetIndex(indexType)
}

// GetAllIndices returns all registered indices
func (m *MultiIndexManager) GetAllIndices() map[router.IndexType]*quiver.Index {
	indices := m.registry.GetAllIndices()
	result := make(map[router.IndexType]*quiver.Index, len(indices))
	for k, v := range indices {
		result[k] = v
	}
	return result
}

// Route routes a query to the appropriate index
func (m *MultiIndexManager) Route(ctx context.Context, query []float32) (router.RoutingDecision, error) {
	return m.router.Route(ctx, query)
}

// MultiIndexSearch searches across multiple indices using the semantic router
func (m *MultiIndexManager) MultiIndexSearch(ctx context.Context, query []float32, k int) ([]quiver.SearchResult, error) {
	// Route the query to the appropriate index
	decision, err := m.router.Route(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to route query: %w", err)
	}

	// Get the target index
	idx, err := m.registry.GetIndex(decision.TargetIndex)
	if err != nil {
		return nil, fmt.Errorf("failed to get index: %w", err)
	}

	// Search the target index
	results, err := idx.Search(query, k, 0, 0)
	if err != nil {
		return nil, fmt.Errorf("failed to search index: %w", err)
	}

	// Add routing metadata to results
	for i := range results {
		if results[i].Metadata == nil {
			results[i].Metadata = make(map[string]interface{})
		}
		results[i].Metadata["_routing"] = map[string]interface{}{
			"index":       string(decision.TargetIndex),
			"confidence":  decision.Confidence,
			"decision_id": decision.DecisionID,
		}
	}

	return results, nil
}

// BatchRoute routes multiple queries in parallel
func (m *MultiIndexManager) BatchRoute(ctx context.Context, queries [][]float32) ([]router.RoutingDecision, error) {
	return m.router.BatchRoute(ctx, queries)
}

// GetRouter returns the underlying semantic router
func (m *MultiIndexManager) GetRouter() *router.SemanticRouter {
	return m.router
}

// Close closes the manager and releases resources
func (m *MultiIndexManager) Close() error {
	return m.router.Close()
}
