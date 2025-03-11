# Quiver Dashboard Example

This example demonstrates how to use Quiver's built-in dashboard for monitoring and visualizing your vector database.

## Features

The dashboard provides real-time monitoring of:

- Vector count and memory usage
- Query latency and operation counts
- Index configuration details
- Detailed metrics and statistics

## Running the Example

To run this example:

```bash
cd examples/dashboard
go run main.go
```

Then open your browser to [http://localhost:8080/dashboard](http://localhost:8080/dashboard)

## How It Works

This example:

1. Creates a Quiver index with dimensionality reduction disabled initially
2. Adds 1,000 random vectors with metadata to the index
3. Sets up a dashboard with a 3-second refresh interval
4. Continuously adds more vectors and performs searches in the background
5. Serves the dashboard on port 8080

### Vector Generation Strategy

The example uses a special vector generation strategy to create more structured data:

1. Creates several "concept" vectors that represent different patterns in the vector space
2. Generates each vector as a weighted combination of these concepts plus a small amount of noise
3. This approach creates vectors with meaningful relationships, which is more realistic than purely random data

This structured approach helps avoid issues with dimensionality reduction algorithms like PCA, which can fail when applied to completely random data.

## Customizing the Dashboard

You can customize the dashboard by modifying the `DashboardConfig`:

```go
dashboardConfig := api.DefaultDashboardConfig()
dashboardConfig.CustomTitle = "Your Custom Title"
dashboardConfig.RefreshInterval = 5 // Refresh every 5 seconds
dashboardConfig.EnableAuth = true   // Enable basic authentication
dashboardConfig.Username = "admin"  // Set username
dashboardConfig.Password = "secret" // Set password
```

## Adding to Your Application

To add the dashboard to your own application:

1. Create a Fiber app or get the app from your API server
2. Configure the dashboard with `api.DefaultDashboardConfig()`
3. Register the dashboard with `api.RegisterDashboard(app, idx, dashboardConfig, logger)`

## Screenshots

![Dashboard Overview](https://example.com/dashboard-screenshot.png)

## Learn More

For more information about Quiver's features, check out the [documentation](https://tfmv.github.io/quiver/).
