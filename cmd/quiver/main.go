package main

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/TFMV/quiver/pkg/api"
	"github.com/TFMV/quiver/pkg/core"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var (
	cfgFile   string
	dataDir   string
	logLevel  string
	version   = "0.1.0"    // Will be set during build
	startTime = time.Now() // Track app start time
)

func main() {
	cobra.OnInitialize(initConfig)

	rootCmd := &cobra.Command{
		Use:   "quiver",
		Short: "Quiver - High-performance vector database",
		Long: `Quiver is a high-performance vector database optimized for 
machine learning applications and similarity search.`,
		Version: version,
	}

	// Global flags
	rootCmd.PersistentFlags().StringVar(&cfgFile, "config", "", "config file (default is $HOME/.quiver.yaml)")
	rootCmd.PersistentFlags().StringVar(&dataDir, "data-dir", "./data", "data directory for storing vectors")
	rootCmd.PersistentFlags().StringVar(&logLevel, "log-level", "info", "log level (debug, info, warn, error)")

	// Add commands
	rootCmd.AddCommand(serveCmd)
	rootCmd.AddCommand(backupCmd)
	rootCmd.AddCommand(restoreCmd)
	rootCmd.AddCommand(infoCmd)

	// Execute root command
	if err := rootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}

// initConfig reads in config file and ENV variables if set
func initConfig() {
	if cfgFile != "" {
		// Use config file from the flag
		viper.SetConfigFile(cfgFile)
	} else {
		// Find home directory
		home, err := os.UserHomeDir()
		if err != nil {
			fmt.Println(err)
			os.Exit(1)
		}

		// Search config in home directory with name ".quiver"
		viper.AddConfigPath(home)
		viper.AddConfigPath(".")
		viper.SetConfigName(".quiver")
	}

	// Read environment variables
	viper.AutomaticEnv()
	viper.SetEnvPrefix("QUIVER")

	// Read in config
	if err := viper.ReadInConfig(); err == nil {
		fmt.Println("Using config file:", viper.ConfigFileUsed())
	}

	// Apply config to variables if set
	if viper.IsSet("data_dir") {
		dataDir = viper.GetString("data_dir")
	}

	if viper.IsSet("log_level") {
		logLevel = viper.GetString("log_level")
	}
}

// serveCmd represents the serve command
var serveCmd = &cobra.Command{
	Use:   "serve",
	Short: "Start the Quiver server",
	Long:  `Start the Quiver vector database server with the specified configuration.`,
	RunE: func(cmd *cobra.Command, args []string) error {
		host, _ := cmd.Flags().GetString("host")
		port, _ := cmd.Flags().GetInt("port")
		enableAuth, _ := cmd.Flags().GetBool("auth")
		jwtSecret, _ := cmd.Flags().GetString("jwt-secret")

		// Create data directory if needed
		if err := os.MkdirAll(dataDir, 0755); err != nil {
			return fmt.Errorf("failed to create data directory: %w", err)
		}

		// Initialize database
		fmt.Println("Initializing Quiver database...")

		dbOptions := core.DefaultDBOptions()
		dbOptions.StoragePath = dataDir
		dbOptions.EnablePersistence = true

		db, err := core.NewDB(dbOptions)
		if err != nil {
			return fmt.Errorf("failed to initialize database: %w", err)
		}

		// Initialize server
		fmt.Printf("Starting Quiver server on %s:%d...\n", host, port)

		// Create server config
		serverConfig := api.DefaultServerConfig()
		serverConfig.Host = host
		serverConfig.Port = port
		serverConfig.LogLevel = logLevel

		// Configure auth if enabled
		if enableAuth {
			if jwtSecret == "" {
				return fmt.Errorf("jwt-secret is required when auth is enabled")
			}
			serverConfig.JWTSecret = jwtSecret
		}

		// Start server
		server := api.NewServer(db, serverConfig)
		fmt.Println("Server is running. Press Ctrl+C to stop.")

		// This will block until server is shut down
		server.Start()
		return nil
	},
}

// backupCmd represents the backup command
var backupCmd = &cobra.Command{
	Use:   "backup PATH",
	Short: "Backup the database",
	Long:  `Create a backup of the Quiver database to the specified path.`,
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		backupPath := args[0]

		// Convert to absolute path if needed
		if !filepath.IsAbs(backupPath) {
			absPath, err := filepath.Abs(backupPath)
			if err != nil {
				return fmt.Errorf("failed to resolve backup path: %w", err)
			}
			backupPath = absPath
		}

		// Initialize database
		fmt.Println("Initializing Quiver database...")

		dbOptions := core.DefaultDBOptions()
		dbOptions.StoragePath = dataDir
		dbOptions.EnablePersistence = true

		db, err := core.NewDB(dbOptions)
		if err != nil {
			return fmt.Errorf("failed to initialize database: %w", err)
		}

		// Create backup
		fmt.Printf("Creating backup at %s...\n", backupPath)
		if err := db.BackupDatabase(backupPath); err != nil {
			return fmt.Errorf("backup failed: %w", err)
		}

		fmt.Println("Backup completed successfully")
		return nil
	},
}

// restoreCmd represents the restore command
var restoreCmd = &cobra.Command{
	Use:   "restore PATH",
	Short: "Restore the database from backup",
	Long:  `Restore the Quiver database from a backup at the specified path.`,
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		backupPath := args[0]

		// Convert to absolute path if needed
		if !filepath.IsAbs(backupPath) {
			absPath, err := filepath.Abs(backupPath)
			if err != nil {
				return fmt.Errorf("failed to resolve backup path: %w", err)
			}
			backupPath = absPath
		}

		// Initialize database
		fmt.Println("Initializing Quiver database...")

		dbOptions := core.DefaultDBOptions()
		dbOptions.StoragePath = dataDir
		dbOptions.EnablePersistence = true

		db, err := core.NewDB(dbOptions)
		if err != nil {
			return fmt.Errorf("failed to initialize database: %w", err)
		}

		// Restore from backup
		fmt.Printf("Restoring from backup at %s...\n", backupPath)
		if err := db.RestoreDatabase(backupPath); err != nil {
			return fmt.Errorf("restore failed: %w", err)
		}

		fmt.Println("Database restored successfully")
		return nil
	},
}

// infoCmd represents the info command
var infoCmd = &cobra.Command{
	Use:   "info",
	Short: "Show database information",
	Long:  `Display information about the Quiver database.`,
	RunE: func(cmd *cobra.Command, args []string) error {
		// Initialize database (read-only mode)
		fmt.Println("Initializing Quiver database...")

		dbOptions := core.DefaultDBOptions()
		dbOptions.StoragePath = dataDir
		dbOptions.EnablePersistence = true

		db, err := core.NewDB(dbOptions)
		if err != nil {
			return fmt.Errorf("failed to initialize database: %w", err)
		}

		// Calculate total vector count
		collections := db.ListCollections()
		totalVectors := 0
		for _, colName := range collections {
			col, err := db.GetCollection(colName)
			if err == nil && col != nil {
				totalVectors += col.Count()
			}
		}

		// Print database info
		fmt.Println("\nQuiver Database Information")
		fmt.Println("---------------------------")
		fmt.Printf("Version:          %s\n", version)
		fmt.Printf("Data Directory:   %s\n", dataDir)
		fmt.Printf("Collections:      %d\n", len(collections))
		fmt.Printf("Total Vectors:    %d\n", totalVectors)
		fmt.Printf("Uptime:           %s\n", time.Since(startTime).Round(time.Second))
		fmt.Println("---------------------------")

		// Print collection details if any exist
		if len(collections) > 0 {
			fmt.Println("\nCollections:")
			for _, colName := range collections {
				col, err := db.GetCollection(colName)
				if err == nil && col != nil {
					stats := col.Stats()
					fmt.Printf("- %s: %d vectors, %d dimensions\n",
						colName, stats.VectorCount, stats.Dimension)
				}
			}
		}

		return nil
	},
}

func init() {
	// Add server-specific flags
	serveCmd.Flags().String("host", "localhost", "server host")
	serveCmd.Flags().Int("port", 8080, "server port")
	serveCmd.Flags().Bool("auth", false, "enable JWT authentication")
	serveCmd.Flags().String("jwt-secret", "", "JWT secret key for authentication")
	serveCmd.Flags().Bool("cors", true, "enable CORS")

	// Bind flags to viper
	if err := viper.BindPFlag("host", serveCmd.Flags().Lookup("host")); err != nil {
		fmt.Printf("Error binding host flag: %v\n", err)
	}
	if err := viper.BindPFlag("port", serveCmd.Flags().Lookup("port")); err != nil {
		fmt.Printf("Error binding port flag: %v\n", err)
	}
	if err := viper.BindPFlag("auth", serveCmd.Flags().Lookup("auth")); err != nil {
		fmt.Printf("Error binding auth flag: %v\n", err)
	}
	if err := viper.BindPFlag("jwt_secret", serveCmd.Flags().Lookup("jwt-secret")); err != nil {
		fmt.Printf("Error binding jwt_secret flag: %v\n", err)
	}
	if err := viper.BindPFlag("cors", serveCmd.Flags().Lookup("cors")); err != nil {
		fmt.Printf("Error binding cors flag: %v\n", err)
	}
}
