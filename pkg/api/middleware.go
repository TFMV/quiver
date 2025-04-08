package api

import (
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/golang-jwt/jwt/v5"
	"golang.org/x/time/rate"
)

// AuthMiddleware creates a middleware for JWT authentication
func AuthMiddleware(jwtSecret string) gin.HandlerFunc {
	return func(c *gin.Context) {
		// Skip auth if secret is empty
		if jwtSecret == "" {
			c.Next()
			return
		}

		authHeader := c.GetHeader("Authorization")
		if authHeader == "" {
			c.AbortWithStatusJSON(http.StatusUnauthorized, ErrorResponse{
				Status:  http.StatusUnauthorized,
				Message: "Authorization header is required",
			})
			return
		}

		// Extract the token
		tokenString := authHeader
		if len(authHeader) > 7 && authHeader[:7] == "Bearer " {
			tokenString = authHeader[7:]
		}

		// Parse and validate token
		token, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
			// Validate signing method
			if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
				return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
			}

			return []byte(jwtSecret), nil
		})

		if err != nil {
			c.AbortWithStatusJSON(http.StatusUnauthorized, ErrorResponse{
				Status:  http.StatusUnauthorized,
				Message: "Invalid token",
				Error:   err.Error(),
			})
			return
		}

		// Check if token is valid
		if claims, ok := token.Claims.(jwt.MapClaims); ok && token.Valid {
			// Add claims to context
			c.Set("claims", claims)
			c.Next()
		} else {
			c.AbortWithStatusJSON(http.StatusUnauthorized, ErrorResponse{
				Status:  http.StatusUnauthorized,
				Message: "Invalid token claims",
			})
			return
		}
	}
}

// Client represents a rate-limited client
type Client struct {
	limiter  *rate.Limiter
	lastSeen time.Time
}

// RateLimiterMiddleware creates a middleware for rate limiting
func RateLimiterMiddleware(rps int, burst int) gin.HandlerFunc {
	// Store clients with their rate limiters
	var (
		clients = make(map[string]*Client)
		mu      sync.Mutex
	)

	// Set default values if not provided
	if rps <= 0 {
		rps = 10 // Default to 10 requests per second
	}
	if burst <= 0 {
		burst = rps // Default burst to same as rate
	}

	// Clean up routine - remove old clients every minute
	go func() {
		for {
			time.Sleep(time.Minute)

			mu.Lock()
			for ip, client := range clients {
				if time.Since(client.lastSeen) > 3*time.Minute {
					delete(clients, ip)
				}
			}
			mu.Unlock()
		}
	}()

	return func(c *gin.Context) {
		// Get client IP
		ip := c.ClientIP()

		mu.Lock()

		// Create client if it doesn't exist
		if _, exists := clients[ip]; !exists {
			clients[ip] = &Client{
				limiter:  rate.NewLimiter(rate.Limit(rps), burst),
				lastSeen: time.Now(),
			}
		}

		// Update last seen time
		clients[ip].lastSeen = time.Now()

		// Check if request is allowed
		if !clients[ip].limiter.Allow() {
			mu.Unlock()
			c.AbortWithStatusJSON(http.StatusTooManyRequests, ErrorResponse{
				Status:  http.StatusTooManyRequests,
				Message: "Rate limit exceeded",
			})
			return
		}

		mu.Unlock()
		c.Next()
	}
}

// LoggingMiddleware creates a middleware for request logging
func LoggingMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		// Start timer
		start := time.Now()
		path := c.Request.URL.Path
		query := c.Request.URL.RawQuery

		// Process request
		c.Next()

		// Stop timer
		end := time.Now()
		latency := end.Sub(start)

		// Get status
		status := c.Writer.Status()

		// Log request
		if query != "" {
			path = path + "?" + query
		}

		// Format log with colors based on status
		var statusColor, resetColor, methodColor string

		if gin.Mode() == gin.DebugMode {
			// Add colors in debug mode
			resetColor = "\033[0m"
			methodColor = "\033[1;34m" // Blue

			if status >= 200 && status < 300 {
				statusColor = "\033[1;32m" // Green
			} else if status >= 300 && status < 400 {
				statusColor = "\033[1;33m" // Yellow
			} else if status >= 400 && status < 500 {
				statusColor = "\033[1;31m" // Red
			} else {
				statusColor = "\033[1;31m" // Red
			}
		}

		fmt.Printf("[QUIVER] %v |%s %3d %s| %13v | %15s |%s %-7s %s %#v\n",
			end.Format("2006/01/02 - 15:04:05"),
			statusColor, status, resetColor,
			latency,
			c.ClientIP(),
			methodColor, c.Request.Method, resetColor,
			path,
		)
	}
}

// ErrorHandlerMiddleware creates a middleware for centralized error handling
func ErrorHandlerMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Next()

		// Check if there were any errors
		if len(c.Errors) > 0 {
			// Handle errors
			c.JSON(http.StatusInternalServerError, ErrorResponse{
				Status:  http.StatusInternalServerError,
				Message: "Internal server error",
				Error:   c.Errors.String(),
			})
		}
	}
}
