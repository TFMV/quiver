package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"path/filepath"
	"time"

	"github.com/TFMV/quiver/pkg/core"
	"github.com/TFMV/quiver/pkg/types"
	"github.com/TFMV/quiver/pkg/vectortypes"
	"github.com/gin-gonic/gin"
)

// Handlers encapsulates all API route handlers
type Handlers struct {
	db *core.DB
}

// NewHandlers creates a new API handlers instance
func NewHandlers(db *core.DB) *Handlers {
	return &Handlers{db: db}
}

// ErrorResponse represents an API error response
type ErrorResponse struct {
	Status  int    `json:"status"`
	Message string `json:"message"`
	Error   string `json:"error,omitempty"`
}

// ===== Health Check =====

// HealthCheck handles GET /api/v1/health
func (h *Handlers) HealthCheck(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status":    "ok",
		"timestamp": time.Now().Format(time.RFC3339),
	})
}

// ===== Collection Management =====

// CreateCollectionRequest represents a request to create a collection
type CreateCollectionRequest struct {
	Name         string `json:"name" binding:"required"`
	Dimension    int    `json:"dimension" binding:"required,gt=0"`
	DistanceType string `json:"distance_type,omitempty"`
}

// CreateCollection handles POST /api/v1/collections
func (h *Handlers) CreateCollection(c *gin.Context) {
	var req CreateCollectionRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, ErrorResponse{
			Status:  http.StatusBadRequest,
			Message: "Invalid request body",
			Error:   err.Error(),
		})
		return
	}

	// Determine distance function based on request
	var distFunc vectortypes.Surface[vectortypes.F32]
	switch req.DistanceType {
	case "euclidean", "l2":
		distFunc = vectortypes.EuclideanSurface
	case "dot_product", "dot":
		distFunc = vectortypes.DotProductSurface
	case "cosine", "cos", "":
		distFunc = vectortypes.CosineSurface
	default:
		c.JSON(http.StatusBadRequest, ErrorResponse{
			Status:  http.StatusBadRequest,
			Message: "Invalid distance type",
			Error:   "Supported types: cosine, euclidean, dot_product",
		})
		return
	}

	// Create the collection
	collection, err := h.db.CreateCollection(req.Name, req.Dimension, distFunc)
	if err != nil {
		status := http.StatusInternalServerError
		if err == core.ErrCollectionExists {
			status = http.StatusConflict
		}

		c.JSON(status, ErrorResponse{
			Status:  status,
			Message: "Failed to create collection",
			Error:   err.Error(),
		})
		return
	}

	stats := collection.Stats()
	c.JSON(http.StatusCreated, gin.H{
		"name":         stats.Name,
		"dimension":    stats.Dimension,
		"created_at":   stats.CreatedAt,
		"vector_count": stats.VectorCount,
	})
}

// ListCollections handles GET /api/v1/collections
func (h *Handlers) ListCollections(c *gin.Context) {
	collections := h.db.ListCollections()
	result := make([]map[string]interface{}, len(collections))

	for i, name := range collections {
		col, err := h.db.GetCollection(name)
		if err != nil {
			continue // Skip this collection if there's an error
		}

		stats := col.Stats()
		result[i] = map[string]interface{}{
			"name":         stats.Name,
			"dimension":    stats.Dimension,
			"created_at":   stats.CreatedAt,
			"vector_count": stats.VectorCount,
		}
	}

	c.JSON(http.StatusOK, gin.H{
		"collections": result,
		"count":       len(result),
	})
}

// GetCollection handles GET /api/v1/collections/:collection
func (h *Handlers) GetCollection(c *gin.Context) {
	name := c.Param("collection")
	collection, err := h.db.GetCollection(name)
	if err != nil {
		c.JSON(http.StatusNotFound, ErrorResponse{
			Status:  http.StatusNotFound,
			Message: "Collection not found",
			Error:   err.Error(),
		})
		return
	}

	stats := collection.Stats()
	c.JSON(http.StatusOK, gin.H{
		"name":         stats.Name,
		"dimension":    stats.Dimension,
		"created_at":   stats.CreatedAt,
		"vector_count": stats.VectorCount,
	})
}

// DeleteCollection handles DELETE /api/v1/collections/:collection
func (h *Handlers) DeleteCollection(c *gin.Context) {
	name := c.Param("collection")
	err := h.db.DeleteCollection(name)
	if err != nil {
		status := http.StatusInternalServerError
		if err == core.ErrCollectionNotFound {
			status = http.StatusNotFound
		}

		c.JSON(status, ErrorResponse{
			Status:  status,
			Message: "Failed to delete collection",
			Error:   err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"status":  "ok",
		"message": fmt.Sprintf("Collection '%s' deleted successfully", name),
	})
}

// GetCollectionStats handles GET /api/v1/collections/:collection/stats
func (h *Handlers) GetCollectionStats(c *gin.Context) {
	name := c.Param("collection")
	collection, err := h.db.GetCollection(name)
	if err != nil {
		c.JSON(http.StatusNotFound, ErrorResponse{
			Status:  http.StatusNotFound,
			Message: "Collection not found",
			Error:   err.Error(),
		})
		return
	}

	stats := collection.Stats()
	c.JSON(http.StatusOK, gin.H{
		"name":         stats.Name,
		"dimension":    stats.Dimension,
		"created_at":   stats.CreatedAt,
		"vector_count": stats.VectorCount,
	})
}

// ===== Vector Operations =====

// AddVectorRequest represents a request to add a vector
type AddVectorRequest struct {
	ID       string          `json:"id" binding:"required"`
	Vector   vectortypes.F32 `json:"vector" binding:"required"`
	Metadata json.RawMessage `json:"metadata,omitempty"`
}

// AddVector handles POST /api/v1/collections/:collection/vectors
func (h *Handlers) AddVector(c *gin.Context) {
	name := c.Param("collection")
	collection, err := h.db.GetCollection(name)
	if err != nil {
		c.JSON(http.StatusNotFound, ErrorResponse{
			Status:  http.StatusNotFound,
			Message: "Collection not found",
			Error:   err.Error(),
		})
		return
	}

	var req AddVectorRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, ErrorResponse{
			Status:  http.StatusBadRequest,
			Message: "Invalid request body",
			Error:   err.Error(),
		})
		return
	}

	err = collection.Add(req.ID, req.Vector, req.Metadata)
	if err != nil {
		status := http.StatusInternalServerError
		if err == core.ErrInvalidDimension {
			status = http.StatusBadRequest
		} else if err == core.ErrVectorAlreadyExist {
			status = http.StatusConflict
		}

		c.JSON(status, ErrorResponse{
			Status:  status,
			Message: "Failed to add vector",
			Error:   err.Error(),
		})
		return
	}

	c.JSON(http.StatusCreated, gin.H{
		"id":      req.ID,
		"status":  "ok",
		"message": "Vector added successfully",
	})
}

// AddVectorBatch handles POST /api/v1/collections/:collection/vectors/batch
func (h *Handlers) AddVectorBatch(c *gin.Context) {
	name := c.Param("collection")
	collection, err := h.db.GetCollection(name)
	if err != nil {
		c.JSON(http.StatusNotFound, ErrorResponse{
			Status:  http.StatusNotFound,
			Message: "Collection not found",
			Error:   err.Error(),
		})
		return
	}

	var reqs []AddVectorRequest
	if err := c.ShouldBindJSON(&reqs); err != nil {
		c.JSON(http.StatusBadRequest, ErrorResponse{
			Status:  http.StatusBadRequest,
			Message: "Invalid request body",
			Error:   err.Error(),
		})
		return
	}

	if len(reqs) == 0 {
		c.JSON(http.StatusBadRequest, ErrorResponse{
			Status:  http.StatusBadRequest,
			Message: "Empty batch",
		})
		return
	}

	// Convert to vectortypes.Vector format
	vectors := make([]vectortypes.Vector, len(reqs))
	for i, req := range reqs {
		vectors[i] = vectortypes.Vector{
			ID:       req.ID,
			Values:   req.Vector,
			Metadata: req.Metadata,
		}
	}

	err = collection.AddBatch(vectors)
	if err != nil {
		status := http.StatusInternalServerError
		if err == core.ErrInvalidDimension || err == core.ErrInvalidMetadata {
			status = http.StatusBadRequest
		} else if err == core.ErrVectorAlreadyExist {
			status = http.StatusConflict
		}

		c.JSON(status, ErrorResponse{
			Status:  status,
			Message: "Failed to add vectors",
			Error:   err.Error(),
		})
		return
	}

	c.JSON(http.StatusCreated, gin.H{
		"status":  "ok",
		"message": fmt.Sprintf("%d vectors added successfully", len(vectors)),
		"count":   len(vectors),
	})
}

// GetVector handles GET /api/v1/collections/:collection/vectors/:id
func (h *Handlers) GetVector(c *gin.Context) {
	name := c.Param("collection")
	id := c.Param("id")

	collection, err := h.db.GetCollection(name)
	if err != nil {
		c.JSON(http.StatusNotFound, ErrorResponse{
			Status:  http.StatusNotFound,
			Message: "Collection not found",
			Error:   err.Error(),
		})
		return
	}

	vector, err := collection.Get(id)
	if err != nil {
		status := http.StatusInternalServerError
		if err == core.ErrVectorNotFound {
			status = http.StatusNotFound
		}

		c.JSON(status, ErrorResponse{
			Status:  status,
			Message: "Failed to get vector",
			Error:   err.Error(),
		})
		return
	}

	// Parse metadata if it exists
	var metadata map[string]interface{}
	if len(vector.Metadata) > 0 {
		if err := json.Unmarshal(vector.Metadata, &metadata); err != nil {
			metadata = map[string]interface{}{}
		}
	}

	c.JSON(http.StatusOK, gin.H{
		"id":       vector.ID,
		"vector":   vector.Values,
		"metadata": metadata,
	})
}

// UpdateVectorRequest represents a request to update a vector
type UpdateVectorRequest struct {
	Vector   vectortypes.F32 `json:"vector,omitempty"`
	Metadata json.RawMessage `json:"metadata,omitempty"`
}

// UpdateVector handles PUT /api/v1/collections/:collection/vectors/:id
func (h *Handlers) UpdateVector(c *gin.Context) {
	name := c.Param("collection")
	id := c.Param("id")

	collection, err := h.db.GetCollection(name)
	if err != nil {
		c.JSON(http.StatusNotFound, ErrorResponse{
			Status:  http.StatusNotFound,
			Message: "Collection not found",
			Error:   err.Error(),
		})
		return
	}

	var req UpdateVectorRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, ErrorResponse{
			Status:  http.StatusBadRequest,
			Message: "Invalid request body",
			Error:   err.Error(),
		})
		return
	}

	if req.Vector == nil && len(req.Metadata) == 0 {
		c.JSON(http.StatusBadRequest, ErrorResponse{
			Status:  http.StatusBadRequest,
			Message: "Nothing to update",
		})
		return
	}

	err = collection.Update(id, req.Vector, req.Metadata)
	if err != nil {
		status := http.StatusInternalServerError
		if err == core.ErrVectorNotFound {
			status = http.StatusNotFound
		} else if err == core.ErrInvalidDimension || err == core.ErrInvalidMetadata {
			status = http.StatusBadRequest
		}

		c.JSON(status, ErrorResponse{
			Status:  status,
			Message: "Failed to update vector",
			Error:   err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"id":      id,
		"status":  "ok",
		"message": "Vector updated successfully",
	})
}

// DeleteVector handles DELETE /api/v1/collections/:collection/vectors/:id
func (h *Handlers) DeleteVector(c *gin.Context) {
	name := c.Param("collection")
	id := c.Param("id")

	collection, err := h.db.GetCollection(name)
	if err != nil {
		c.JSON(http.StatusNotFound, ErrorResponse{
			Status:  http.StatusNotFound,
			Message: "Collection not found",
			Error:   err.Error(),
		})
		return
	}

	err = collection.Delete(id)
	if err != nil {
		status := http.StatusInternalServerError
		if err == core.ErrVectorNotFound {
			status = http.StatusNotFound
		}

		c.JSON(status, ErrorResponse{
			Status:  status,
			Message: "Failed to delete vector",
			Error:   err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"id":      id,
		"status":  "ok",
		"message": "Vector deleted successfully",
	})
}

// DeleteVectorBatch handles POST /api/v1/collections/:collection/vectors/delete/batch
func (h *Handlers) DeleteVectorBatch(c *gin.Context) {
	name := c.Param("collection")
	collection, err := h.db.GetCollection(name)
	if err != nil {
		c.JSON(http.StatusNotFound, ErrorResponse{
			Status:  http.StatusNotFound,
			Message: "Collection not found",
			Error:   err.Error(),
		})
		return
	}

	var ids []string
	if err := c.ShouldBindJSON(&ids); err != nil {
		c.JSON(http.StatusBadRequest, ErrorResponse{
			Status:  http.StatusBadRequest,
			Message: "Invalid request body",
			Error:   err.Error(),
		})
		return
	}

	if len(ids) == 0 {
		c.JSON(http.StatusBadRequest, ErrorResponse{
			Status:  http.StatusBadRequest,
			Message: "Empty ID list",
		})
		return
	}

	err = collection.DeleteBatch(ids)
	if err != nil {
		status := http.StatusInternalServerError
		if err == core.ErrVectorNotFound {
			status = http.StatusNotFound
		}

		c.JSON(status, ErrorResponse{
			Status:  status,
			Message: "Failed to delete vectors",
			Error:   err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"status":  "ok",
		"message": fmt.Sprintf("%d vectors deleted successfully", len(ids)),
		"count":   len(ids),
	})
}

// ===== Search =====

// Search handles POST /api/v1/collections/:collection/search
func (h *Handlers) Search(c *gin.Context) {
	name := c.Param("collection")
	collection, err := h.db.GetCollection(name)
	if err != nil {
		c.JSON(http.StatusNotFound, ErrorResponse{
			Status:  http.StatusNotFound,
			Message: "Collection not found",
			Error:   err.Error(),
		})
		return
	}

	var req types.SearchRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, ErrorResponse{
			Status:  http.StatusBadRequest,
			Message: "Invalid request body",
			Error:   err.Error(),
		})
		return
	}

	// Set defaults if not provided
	if req.TopK <= 0 {
		req.TopK = 10 // Default to 10 results
	}

	// Execute search
	result, err := collection.Search(req)
	if err != nil {
		status := http.StatusInternalServerError
		if err == core.ErrInvalidDimension {
			status = http.StatusBadRequest
		}

		c.JSON(status, ErrorResponse{
			Status:  status,
			Message: "Search failed",
			Error:   err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, result)
}

// ===== Database Operations =====

// GetMetrics handles GET /api/v1/metrics
func (h *Handlers) GetMetrics(c *gin.Context) {
	metrics, err := h.db.GetMetrics()
	if err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{
			Status:  http.StatusInternalServerError,
			Message: "Failed to get metrics",
			Error:   err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"avg_latency_ms": metrics.AvgLatencyMs,
		"qps":            metrics.QPS,
		"cpu_percent":    metrics.CPUPercent,
		"memory_mb":      metrics.MemoryMB,
		"timestamp":      metrics.Timestamp,
	})
}

// BackupRequest represents a request to create a backup
type BackupRequest struct {
	Path string `json:"path" binding:"required"`
}

// CreateBackup handles POST /api/v1/backup
func (h *Handlers) CreateBackup(c *gin.Context) {
	var req BackupRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, ErrorResponse{
			Status:  http.StatusBadRequest,
			Message: "Invalid request body",
			Error:   err.Error(),
		})
		return
	}

	// Ensure path is absolute
	backupPath := req.Path
	if !filepath.IsAbs(backupPath) {
		c.JSON(http.StatusBadRequest, ErrorResponse{
			Status:  http.StatusBadRequest,
			Message: "Backup path must be absolute",
		})
		return
	}

	err := h.db.BackupDatabase(backupPath)
	if err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{
			Status:  http.StatusInternalServerError,
			Message: "Backup failed",
			Error:   err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"status":      "ok",
		"message":     "Backup created successfully",
		"backup_path": backupPath,
		"timestamp":   time.Now().Format(time.RFC3339),
	})
}

// RestoreRequest represents a request to restore from a backup
type RestoreRequest struct {
	Path string `json:"path" binding:"required"`
}

// RestoreBackup handles POST /api/v1/restore
func (h *Handlers) RestoreBackup(c *gin.Context) {
	var req RestoreRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, ErrorResponse{
			Status:  http.StatusBadRequest,
			Message: "Invalid request body",
			Error:   err.Error(),
		})
		return
	}

	// Ensure path is absolute
	backupPath := req.Path
	if !filepath.IsAbs(backupPath) {
		c.JSON(http.StatusBadRequest, ErrorResponse{
			Status:  http.StatusBadRequest,
			Message: "Backup path must be absolute",
		})
		return
	}

	err := h.db.RestoreDatabase(backupPath)
	if err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{
			Status:  http.StatusInternalServerError,
			Message: "Restore failed",
			Error:   err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"status":      "ok",
		"message":     "Database restored successfully",
		"backup_path": backupPath,
		"timestamp":   time.Now().Format(time.RFC3339),
	})
}
