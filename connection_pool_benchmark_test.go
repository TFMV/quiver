package quiver

import (
	"context"
	"fmt"
	"sync"
	"testing"
)

func BenchmarkConnectionPool(b *testing.B) {
	// Create a new in-memory DuckDB instance
	db, err := NewDuckDB()
	if err != nil {
		b.Fatalf("Failed to create DuckDB: %v", err)
	}
	defer db.Close()

	// Create a connection pool
	pool, err := NewConnectionPool(db, 5, 10)
	if err != nil {
		b.Fatalf("Failed to create connection pool: %v", err)
	}
	defer pool.Close()

	// Create a test table
	conn, err := pool.GetConnection()
	if err != nil {
		b.Fatalf("Failed to get connection: %v", err)
	}
	_, err = conn.Exec(context.Background(), `CREATE TABLE test (id INTEGER, value VARCHAR)`)
	if err != nil {
		b.Fatalf("Failed to create test table: %v", err)
	}
	pool.ReleaseConnection(conn)

	b.Run("Sequential", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			conn, err := pool.GetConnection()
			if err != nil {
				b.Fatalf("Failed to get connection: %v", err)
			}
			query := fmt.Sprintf(`INSERT INTO test VALUES (%d, 'test')`, i)
			_, err = conn.Exec(context.Background(), query)
			if err != nil {
				b.Fatalf("Failed to execute query: %v", err)
			}
			pool.ReleaseConnection(conn)
		}
	})

	b.Run("Parallel", func(b *testing.B) {
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			counter := 0
			for pb.Next() {
				conn, err := pool.GetConnection()
				if err != nil {
					b.Fatalf("Failed to get connection: %v", err)
				}
				query := fmt.Sprintf(`INSERT INTO test VALUES (%d, 'test')`, counter)
				_, err = conn.Exec(context.Background(), query)
				if err != nil {
					b.Fatalf("Failed to execute query: %v", err)
				}
				pool.ReleaseConnection(conn)
				counter++
			}
		})
	})

	b.Run("BatchConnection", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			conn, err := pool.GetBatchConnection()
			if err != nil {
				b.Fatalf("Failed to get batch connection: %v", err)
			}
			query := fmt.Sprintf(`INSERT INTO test VALUES (%d, 'test')`, i)
			_, err = conn.Exec(context.Background(), query)
			if err != nil {
				b.Fatalf("Failed to execute query: %v", err)
			}
			pool.ReleaseBatchConnection(conn)
		}
	})

	b.Run("PreparedStatements", func(b *testing.B) {
		// For prepared statements, we'll use a fixed query since DuckDB doesn't support
		// parameterized queries in the same way as other databases
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			query := fmt.Sprintf(`INSERT INTO test VALUES (%d, 'test')`, i)
			stmt, err := pool.GetPreparedStatement(query)
			if err != nil {
				b.Fatalf("Failed to get prepared statement: %v", err)
			}
			_, err = stmt.conn.Exec(context.Background(), stmt.query)
			if err != nil {
				b.Fatalf("Failed to execute prepared statement: %v", err)
			}
			pool.ReleasePreparedStatement(stmt)
		}
	})

	b.Run("ThreadLocalConnections", func(b *testing.B) {
		// Create a large number of goroutines to test thread-local connections
		var wg sync.WaitGroup
		numGoroutines := 100
		queriesPerGoroutine := b.N / numGoroutines
		if queriesPerGoroutine < 1 {
			queriesPerGoroutine = 1
		}

		b.ResetTimer()
		for i := 0; i < numGoroutines; i++ {
			wg.Add(1)
			go func(id int) {
				defer wg.Done()
				for j := 0; j < queriesPerGoroutine; j++ {
					conn, err := pool.GetConnection()
					if err != nil {
						b.Fatalf("Failed to get connection: %v", err)
					}
					query := fmt.Sprintf(`INSERT INTO test VALUES (%d, 'test')`, id*queriesPerGoroutine+j)
					_, err = conn.Exec(context.Background(), query)
					if err != nil {
						b.Fatalf("Failed to execute query: %v", err)
					}
					pool.ReleaseConnection(conn)
				}
			}(i)
		}
		wg.Wait()
	})
}
