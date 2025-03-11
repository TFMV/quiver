package dimreduce

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sync"

	"go.uber.org/zap"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

// ReductionMethod represents the method used for dimensionality reduction
type ReductionMethod string

const (
	// PCA is Principal Component Analysis
	PCA ReductionMethod = "PCA"
	// TSNE is t-Distributed Stochastic Neighbor Embedding
	TSNE ReductionMethod = "TSNE"
	// UMAP is Uniform Manifold Approximation and Projection
	UMAP ReductionMethod = "UMAP"
)

// DimReducerConfig holds configuration for dimensionality reduction.
type DimReducerConfig struct {
	// TargetDimension is the desired output dimension
	TargetDimension int
	// Method specifies the dimensionality reduction algorithm to use
	Method ReductionMethod
	// Adaptive enables automatic determination of optimal dimensions
	Adaptive bool
	// MinVarianceExplained is the minimum variance that should be explained (0.0-1.0)
	// Only used when Adaptive is true
	MinVarianceExplained float64
	// MaxDimension is the maximum dimension to consider when using adaptive reduction
	MaxDimension int
	// Logger for logging operations
	Logger *zap.Logger
}

// DefaultDimReducerConfig returns a default configuration
func DefaultDimReducerConfig() DimReducerConfig {
	return DimReducerConfig{
		TargetDimension:      128,
		Method:               PCA,
		Adaptive:             true,
		MinVarianceExplained: 0.95, // 95% variance explained
		MaxDimension:         512,
		Logger:               nil,
	}
}

// DimReducer handles dimensionality reduction operations.
type DimReducer struct {
	config DimReducerConfig
	lock   sync.RWMutex
	logger *zap.Logger

	// PCA specific fields
	pcaComponents *mat.Dense // Principal components (eigenvectors)
	pcaVariance   []float64  // Explained variance for each component
	pcaMean       []float64  // Mean of each feature
}

// NewDimReducer creates a new dimensionality reducer with the given configuration.
func NewDimReducer(config DimReducerConfig) (*DimReducer, error) {
	if config.TargetDimension <= 0 {
		return nil, errors.New("target dimension must be positive")
	}

	if config.Method == "" {
		config.Method = PCA // Default to PCA if not specified
	}

	if config.Logger == nil {
		// Create a no-op logger if none provided
		logger, _ := zap.NewProduction()
		config.Logger = logger
	}

	// Validate adaptive parameters
	if config.Adaptive {
		if config.MinVarianceExplained <= 0 || config.MinVarianceExplained > 1.0 {
			return nil, errors.New("minimum variance explained must be between 0 and 1")
		}
		if config.MaxDimension <= 0 {
			return nil, errors.New("maximum dimension must be positive")
		}
	}

	return &DimReducer{
		config: config,
		logger: config.Logger,
	}, nil
}

// Reduce applies dimensionality reduction to the given vectors.
func (dr *DimReducer) Reduce(vectors [][]float32) ([][]float32, error) {
	dr.lock.Lock()
	defer dr.lock.Unlock()

	dr.logger.Info("Starting dimensionality reduction",
		zap.String("method", string(dr.config.Method)),
		zap.Int("target_dimension", dr.config.TargetDimension),
		zap.Bool("adaptive", dr.config.Adaptive))

	if len(vectors) == 0 {
		return [][]float32{}, nil // Handle empty input
	}

	// Validate input dimensions
	inputDim := len(vectors[0])
	for _, vec := range vectors {
		if len(vec) != inputDim {
			return nil, errors.New("all input vectors must have the same dimension")
		}
		// Check for NaN or Inf values
		for _, val := range vec {
			if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
				return nil, errors.New("input vectors contain NaN or Inf values")
			}
		}
	}

	if inputDim < dr.config.TargetDimension {
		return nil, errors.New("input vector dimension is smaller than target dimension")
	}

	// Special case: If we have only one vector, we can't do PCA
	// Just return a simple projection (take the first N dimensions)
	if len(vectors) == 1 {
		dr.logger.Warn("Only one vector provided, using simple projection instead of PCA")
		result := make([][]float32, 1)
		result[0] = make([]float32, dr.config.TargetDimension)
		for i := 0; i < dr.config.TargetDimension; i++ {
			result[0][i] = vectors[0][i]
		}
		return result, nil
	}

	// Convert [][]float32 to *mat.Dense
	rows := len(vectors)
	cols := inputDim
	data := make([]float64, 0, rows*cols)
	for _, row := range vectors {
		for _, val := range row {
			data = append(data, float64(val))
		}
	}
	matrix := mat.NewDense(rows, cols, data)

	// Apply the appropriate reduction method
	switch dr.config.Method {
	case PCA:
		return dr.reducePCA(matrix, rows, cols)
	case TSNE, UMAP:
		// Not implemented yet, fallback to PCA
		dr.logger.Warn("Method not implemented, falling back to PCA",
			zap.String("requested_method", string(dr.config.Method)))
		return dr.reducePCA(matrix, rows, cols)
	default:
		return nil, errors.New("unsupported dimensionality reduction method")
	}
}

// reducePCA performs PCA dimensionality reduction
func (dr *DimReducer) reducePCA(matrix *mat.Dense, rows, cols int) ([][]float32, error) {
	targetDim := dr.config.TargetDimension

	// Check if we have enough samples for PCA
	if rows < 2 {
		return nil, errors.New("at least 2 samples are required for PCA")
	}

	// Ensure target dimension is less than input dimension
	if targetDim >= cols {
		return nil, fmt.Errorf("target dimension (%d) must be less than input dimension (%d)", targetDim, cols)
	}

	// Center the data (subtract mean from each column)
	centered := mat.NewDense(rows, cols, nil)
	dr.pcaMean = make([]float64, cols)

	for j := 0; j < cols; j++ {
		col := mat.Col(nil, j, matrix)
		mean := stat.Mean(col, nil)
		dr.pcaMean[j] = mean

		for i := 0; i < rows; i++ {
			centered.Set(i, j, matrix.At(i, j)-mean)
		}
	}

	// Check for zero variance features
	zeroVarianceFeatures := false
	for j := 0; j < cols; j++ {
		col := mat.Col(nil, j, centered)
		variance := stat.Variance(col, nil)
		if variance < 1e-10 {
			zeroVarianceFeatures = true
			// Add a small amount of noise to avoid numerical issues
			for i := 0; i < rows; i++ {
				noise := (rand.Float64() * 1e-5) - 5e-6
				centered.Set(i, j, centered.At(i, j)+noise)
			}
		}
	}

	if zeroVarianceFeatures {
		dr.logger.Warn("Found features with near-zero variance, adding small noise to improve numerical stability")
	}

	// Compute covariance matrix
	var cov mat.SymDense
	stat.CovarianceMatrix(&cov, centered, nil)

	// Add a small regularization term to the diagonal to improve conditioning
	for i := 0; i < cols; i++ {
		cov.SetSym(i, i, cov.At(i, i)+1e-6)
	}

	// Perform eigendecomposition
	var eigsym mat.EigenSym
	ok := eigsym.Factorize(&cov, true)
	if !ok {
		// If factorization fails, try with more regularization
		dr.logger.Warn("Initial eigendecomposition failed, trying with stronger regularization")
		for i := 0; i < cols; i++ {
			cov.SetSym(i, i, cov.At(i, i)+1e-4)
		}

		ok = eigsym.Factorize(&cov, true)
		if !ok {
			return nil, errors.New("eigendecomposition failed even with regularization")
		}
	}

	// Get eigenvalues and eigenvectors
	eigenvalues := eigsym.Values(nil)
	var eigenvectors mat.Dense
	eigsym.VectorsTo(&eigenvectors)

	// Check for negative eigenvalues (numerical issues)
	for i, val := range eigenvalues {
		if val < 0 {
			dr.logger.Warn("Found negative eigenvalue, setting to small positive value",
				zap.Int("index", i),
				zap.Float64("value", val))
			eigenvalues[i] = 1e-10
		}
	}

	// Sort eigenvalues and eigenvectors in descending order
	indices := make([]int, len(eigenvalues))
	for i := range indices {
		indices[i] = i
	}

	// Sort indices by eigenvalues in descending order
	for i := 0; i < len(indices)-1; i++ {
		for j := i + 1; j < len(indices); j++ {
			if eigenvalues[indices[i]] < eigenvalues[indices[j]] {
				indices[i], indices[j] = indices[j], indices[i]
			}
		}
	}

	// Reorder eigenvalues and eigenvectors
	sortedEigenvalues := make([]float64, len(eigenvalues))
	sortedEigenvectors := mat.NewDense(cols, cols, nil)

	for i, idx := range indices {
		sortedEigenvalues[i] = eigenvalues[idx]
		for j := 0; j < cols; j++ {
			sortedEigenvectors.Set(j, i, eigenvectors.At(j, idx))
		}
	}

	// Store components and variance for future use
	dr.pcaComponents = sortedEigenvectors
	dr.pcaVariance = sortedEigenvalues

	// Calculate total variance and explained variance ratio
	totalVariance := 0.0
	for _, val := range sortedEigenvalues {
		totalVariance += val
	}

	// Protect against division by zero
	if totalVariance < 1e-10 {
		dr.logger.Warn("Total variance is near zero, setting to small value to avoid division by zero")
		totalVariance = 1e-10
	}

	explainedVarianceRatio := make([]float64, len(sortedEigenvalues))
	for i, val := range sortedEigenvalues {
		explainedVarianceRatio[i] = val / totalVariance
	}

	// Log explained variance
	cumulativeVariance := 0.0
	for i := 0; i < min(10, len(explainedVarianceRatio)); i++ {
		cumulativeVariance += explainedVarianceRatio[i]
		dr.logger.Debug("PCA component variance",
			zap.Int("component", i+1),
			zap.Float64("explained_variance_ratio", explainedVarianceRatio[i]),
			zap.Float64("cumulative_variance", cumulativeVariance))
	}

	// Extract the top k eigenvectors
	components := mat.NewDense(cols, targetDim, nil)
	for i := 0; i < cols; i++ {
		for j := 0; j < targetDim; j++ {
			components.Set(i, j, sortedEigenvectors.At(i, j))
		}
	}

	// Project the data onto the principal components
	transformed := mat.NewDense(rows, targetDim, nil)
	transformed.Mul(centered, components)

	// Convert *mat.Dense back to [][]float32
	reducedVectors := make([][]float32, rows)
	for i := 0; i < rows; i++ {
		reducedVectors[i] = make([]float32, targetDim)
		for j := 0; j < targetDim; j++ {
			reducedVectors[i][j] = float32(transformed.At(i, j))
		}
	}

	dr.logger.Info("PCA reduction completed",
		zap.Int("input_dimensions", cols),
		zap.Int("output_dimensions", targetDim),
		zap.Float64("variance_explained", cumulativeVariance))

	return reducedVectors, nil
}

// AdaptiveReduce dynamically adjusts the dimensionality based on data characteristics.
func (dr *DimReducer) AdaptiveReduce(vectors [][]float32) ([][]float32, error) {
	if !dr.config.Adaptive {
		return dr.Reduce(vectors)
	}

	dr.logger.Info("Starting adaptive dimensionality reduction")

	// Determine the optimal dimension
	adaptiveDimension, err := dr.determineAdaptiveDimension(vectors)
	if err != nil {
		return nil, err
	}

	dr.logger.Info("Adaptive dimension determined",
		zap.Int("adaptive_dimension", adaptiveDimension),
		zap.Int("original_target", dr.config.TargetDimension))

	// Save the original target dimension
	originalDimension := dr.config.TargetDimension

	// Set the new target dimension
	dr.config.TargetDimension = adaptiveDimension

	// Perform reduction with the adaptive dimension
	result, err := dr.Reduce(vectors)

	// Restore the original target dimension
	dr.config.TargetDimension = originalDimension

	return result, err
}

// determineAdaptiveDimension determines the optimal dimension based on data.
func (dr *DimReducer) determineAdaptiveDimension(vectors [][]float32) (int, error) {
	if len(vectors) == 0 {
		return 0, errors.New("cannot determine adaptive dimension for empty input")
	}

	inputDim := len(vectors[0])
	maxDim := min(dr.config.MaxDimension, inputDim)

	// Convert [][]float32 to *mat.Dense
	rows := len(vectors)
	cols := inputDim
	data := make([]float64, 0, rows*cols)
	for _, row := range vectors {
		for _, val := range row {
			data = append(data, float64(val))
		}
	}
	matrix := mat.NewDense(rows, cols, data)

	// Center the data
	centered := mat.NewDense(rows, cols, nil)
	means := make([]float64, cols)

	for j := 0; j < cols; j++ {
		col := mat.Col(nil, j, matrix)
		mean := stat.Mean(col, nil)
		means[j] = mean

		for i := 0; i < rows; i++ {
			centered.Set(i, j, matrix.At(i, j)-mean)
		}
	}

	// Compute covariance matrix
	var cov mat.SymDense
	stat.CovarianceMatrix(&cov, centered, nil)

	// Perform eigendecomposition
	var eigsym mat.EigenSym
	ok := eigsym.Factorize(&cov, true)
	if !ok {
		return dr.config.TargetDimension, errors.New("eigendecomposition failed in adaptive dimension determination")
	}

	// Get eigenvalues
	eigenvalues := eigsym.Values(nil)

	// Sort eigenvalues in descending order
	indices := make([]int, len(eigenvalues))
	for i := range indices {
		indices[i] = i
	}

	for i := 0; i < len(indices)-1; i++ {
		for j := i + 1; j < len(indices); j++ {
			if eigenvalues[indices[i]] < eigenvalues[indices[j]] {
				indices[i], indices[j] = indices[j], indices[i]
			}
		}
	}

	// Reorder eigenvalues
	sortedEigenvalues := make([]float64, len(eigenvalues))
	for i, idx := range indices {
		sortedEigenvalues[i] = eigenvalues[idx]
	}

	// Calculate total variance
	totalVariance := 0.0
	for _, val := range sortedEigenvalues {
		totalVariance += val
	}

	// Find the number of components needed to explain the desired variance
	cumulativeVariance := 0.0
	for i := 0; i < maxDim; i++ {
		cumulativeVariance += sortedEigenvalues[i] / totalVariance

		if cumulativeVariance >= dr.config.MinVarianceExplained {
			// We found the optimal dimension
			optimalDim := i + 1

			dr.logger.Info("Determined adaptive dimension",
				zap.Int("optimal_dimension", optimalDim),
				zap.Float64("explained_variance", cumulativeVariance),
				zap.Float64("target_variance", dr.config.MinVarianceExplained))

			return optimalDim, nil
		}
	}

	// If we couldn't reach the desired variance, use the maximum dimension
	dr.logger.Warn("Could not reach desired variance with maximum dimension",
		zap.Float64("max_explained_variance", cumulativeVariance),
		zap.Float64("target_variance", dr.config.MinVarianceExplained),
		zap.Int("using_dimension", maxDim))

	return maxDim, nil
}

// TransformVector transforms a single vector using the fitted model
func (dr *DimReducer) TransformVector(vector []float32) ([]float32, error) {
	dr.lock.RLock()
	defer dr.lock.RUnlock()

	if dr.pcaComponents == nil || dr.pcaMean == nil {
		return nil, errors.New("model not fitted, call Reduce or AdaptiveReduce first")
	}

	if len(vector) != len(dr.pcaMean) {
		return nil, errors.New("input vector dimension does not match the fitted model")
	}

	// Center the vector
	centered := make([]float64, len(vector))
	for i, val := range vector {
		centered[i] = float64(val) - dr.pcaMean[i]
	}

	// Project onto principal components
	result := make([]float64, dr.config.TargetDimension)
	for i := 0; i < dr.config.TargetDimension; i++ {
		for j := 0; j < len(vector); j++ {
			result[i] += centered[j] * dr.pcaComponents.At(j, i)
		}
	}

	// Convert to float32
	resultFloat32 := make([]float32, len(result))
	for i, val := range result {
		resultFloat32[i] = float32(val)
	}

	return resultFloat32, nil
}

// GetExplainedVarianceRatio returns the explained variance ratio for each component
func (dr *DimReducer) GetExplainedVarianceRatio() ([]float64, error) {
	dr.lock.RLock()
	defer dr.lock.RUnlock()

	if dr.pcaVariance == nil {
		return nil, errors.New("model not fitted, call Reduce or AdaptiveReduce first")
	}

	totalVariance := 0.0
	for _, val := range dr.pcaVariance {
		totalVariance += val
	}

	ratio := make([]float64, len(dr.pcaVariance))
	for i, val := range dr.pcaVariance {
		ratio[i] = val / totalVariance
	}

	return ratio, nil
}

// GetCumulativeExplainedVariance returns the cumulative explained variance
func (dr *DimReducer) GetCumulativeExplainedVariance() ([]float64, error) {
	ratio, err := dr.GetExplainedVarianceRatio()
	if err != nil {
		return nil, err
	}

	cumulative := make([]float64, len(ratio))
	sum := 0.0
	for i, val := range ratio {
		sum += val
		cumulative[i] = sum
	}

	return cumulative, nil
}

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
