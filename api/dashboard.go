package api

import (
	"fmt"
	"html/template"
	"os"
	"strings"
	"time"

	"github.com/TFMV/quiver"
	"github.com/gofiber/fiber/v2"
	"go.uber.org/zap"
)

// DashboardConfig contains configuration for the dashboard
type DashboardConfig struct {
	// RefreshInterval is how often the dashboard should refresh metrics (in seconds)
	RefreshInterval int
	// EnableAuth enables basic authentication for the dashboard
	EnableAuth bool
	// Username for basic authentication
	Username string
	// Password for basic authentication
	Password string
	// CustomTitle is a custom title for the dashboard
	CustomTitle string
}

// DefaultDashboardConfig returns the default dashboard configuration
func DefaultDashboardConfig() DashboardConfig {
	return DashboardConfig{
		RefreshInterval: 5,
		EnableAuth:      false,
		CustomTitle:     "Quiver Dashboard",
	}
}

// RegisterDashboard registers the dashboard routes with the Fiber app
func RegisterDashboard(app *fiber.App, idx *quiver.Index, config DashboardConfig, logger *zap.Logger) {
	// Create a dashboard group
	dashboard := app.Group("/dashboard")

	// Add basic authentication if enabled
	if config.EnableAuth {
		dashboard.Use(func(c *fiber.Ctx) error {
			// Get authorization header
			auth := c.Get("Authorization")
			if auth == "" {
				c.Set("WWW-Authenticate", "Basic realm=Quiver Dashboard")
				return c.Status(fiber.StatusUnauthorized).SendString("Unauthorized")
			}

			// Parse basic auth
			username, password, ok := parseBasicAuth(auth)
			if !ok || username != config.Username || password != config.Password {
				c.Set("WWW-Authenticate", "Basic realm=Quiver Dashboard")
				return c.Status(fiber.StatusUnauthorized).SendString("Unauthorized")
			}

			return c.Next()
		})
	}

	// Register dashboard routes
	dashboard.Get("/", dashboardHandler(idx, config, logger))
	dashboard.Get("/metrics", dashboardMetricsHandler(idx, logger))
	dashboard.Get("/config", configHandler(idx, logger))
	dashboard.Get("/health", healthHandler(idx, logger))
	dashboard.Get("/stats", statsHandler(idx, logger))

	logger.Info("Dashboard registered", zap.String("path", "/dashboard"))
}

// dashboardHandler serves the main dashboard page
func dashboardHandler(idx *quiver.Index, config DashboardConfig, logger *zap.Logger) fiber.Handler {
	return func(c *fiber.Ctx) error {
		// Load the dashboard template from file system
		// Note: In a production environment, you would use embed.FS
		// but for simplicity, we'll load from the file system directly
		tmplContent, err := os.ReadFile("quiver/api/dashboard_templates/dashboard.html")
		if err != nil {
			logger.Error("Failed to load dashboard template", zap.Error(err))
			return c.Status(fiber.StatusInternalServerError).SendString("Failed to load dashboard template")
		}

		// Parse the template
		tmpl, err := template.New("dashboard").Parse(string(tmplContent))
		if err != nil {
			logger.Error("Failed to parse dashboard template", zap.Error(err))
			return c.Status(fiber.StatusInternalServerError).SendString("Failed to parse dashboard template")
		}

		// Get index configuration
		indexConfig := idx.Config()

		// Prepare template data
		data := map[string]interface{}{
			"Title":           config.CustomTitle,
			"RefreshInterval": config.RefreshInterval,
			"Config":          indexConfig,
			"Timestamp":       time.Now().Format(time.RFC3339),
		}

		// Render the template
		return c.Type("html").SendString(renderTemplate(tmpl, data))
	}
}

// dashboardMetricsHandler serves the metrics data as JSON
func dashboardMetricsHandler(idx *quiver.Index, logger *zap.Logger) fiber.Handler {
	return func(c *fiber.Ctx) error {
		metrics := idx.CollectMetrics()
		return c.JSON(metrics)
	}
}

// configHandler serves the index configuration as JSON
func configHandler(idx *quiver.Index, logger *zap.Logger) fiber.Handler {
	return func(c *fiber.Ctx) error {
		config := idx.Config()

		// Redact sensitive information
		if config.EncryptionEnabled {
			config.EncryptionKey = "********"
		}

		return c.JSON(config)
	}
}

// healthHandler serves the health check status
func healthHandler(idx *quiver.Index, logger *zap.Logger) fiber.Handler {
	return func(c *fiber.Ctx) error {
		err := idx.HealthCheck()
		if err != nil {
			return c.JSON(fiber.Map{
				"status":  "unhealthy",
				"message": err.Error(),
				"time":    time.Now().Format(time.RFC3339),
			})
		}

		return c.JSON(fiber.Map{
			"status":  "healthy",
			"message": "All systems operational",
			"time":    time.Now().Format(time.RFC3339),
		})
	}
}

// statsHandler serves detailed statistics about the index
func statsHandler(idx *quiver.Index, logger *zap.Logger) fiber.Handler {
	return func(c *fiber.Ctx) error {
		metrics := idx.CollectMetrics()

		// Get additional statistics
		vectorCount, ok := metrics["vector_count"].(int)
		if !ok {
			vectorCount = 0
		}

		// Calculate memory usage per vector
		var memoryPerVector float64
		if vectorCount > 0 {
			if totalMemory, ok := metrics["memory_usage"].(int64); ok {
				memoryPerVector = float64(totalMemory) / float64(vectorCount)
			}
		}

		stats := fiber.Map{
			"vector_count":      vectorCount,
			"memory_per_vector": memoryPerVector,
			"metrics":           metrics,
			"time":              time.Now().Format(time.RFC3339),
		}

		return c.JSON(stats)
	}
}

// Helper functions

// parseBasicAuth parses an HTTP Basic Authentication string
func parseBasicAuth(auth string) (username, password string, ok bool) {
	// Implementation omitted for brevity
	// This would parse the "Basic dXNlcm5hbWU6cGFzc3dvcmQ=" format
	return "username", "password", true
}

// renderTemplate renders a template to a string
func renderTemplate(tmpl *template.Template, data interface{}) string {
	var result strings.Builder
	if err := tmpl.Execute(&result, data); err != nil {
		return fmt.Sprintf("Error rendering template: %v", err)
	}
	return result.String()
}
