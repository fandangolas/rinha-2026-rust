package uds

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"net"
	"sync"

	"github.com/fandangolas/rinha-de-backend-2026/api/internal/search"
)

// Client implements search.Searcher by forwarding queries to a Server over a
// Unix domain socket. Connections are pooled; safe for concurrent use.
type Client struct {
	socketPath string
	pool       sync.Pool
}

// NewClient returns a Client that connects to the Server at socketPath.
// Connections are established lazily and reused across calls.
func NewClient(socketPath string) *Client {
	c := &Client{socketPath: socketPath}
	c.pool.New = func() any {
		conn, err := net.Dial("unix", socketPath)
		if err != nil {
			return nil
		}
		return conn
	}
	return c
}

// Search sends query to the Server and returns the k nearest neighbors.
// If the pooled connection is broken it retries once with a fresh connection.
func (c *Client) Search(query search.Vector, k int) ([]search.Neighbor, error) {
	conn, err := c.getConn()
	if err != nil {
		return nil, err
	}

	neighbors, err := doSearch(conn, query, k)
	if err != nil {
		conn.Close()
		// Retry once with a fresh connection (handles server restart / stale conn).
		if conn, err = net.Dial("unix", c.socketPath); err != nil {
			return nil, err
		}
		if neighbors, err = doSearch(conn, query, k); err != nil {
			conn.Close()
			return nil, err
		}
	}

	c.pool.Put(conn)
	return neighbors, nil
}

// Close is a no-op; pooled connections are closed by the GC or on error.
func (c *Client) Close() error { return nil }

func (c *Client) getConn() (net.Conn, error) {
	if v := c.pool.Get(); v != nil {
		return v.(net.Conn), nil
	}
	return net.Dial("unix", c.socketPath)
}

// doSearch performs one request/response exchange on conn.
func doSearch(conn net.Conn, query search.Vector, k int) ([]search.Neighbor, error) {
	var req [reqBytes]byte
	for i, f := range query {
		binary.LittleEndian.PutUint32(req[i*4:], math.Float32bits(f))
	}
	binary.LittleEndian.PutUint32(req[56:], uint32(k))

	if _, err := conn.Write(req[:]); err != nil {
		return nil, fmt.Errorf("write: %w", err)
	}

	var hdr [respHeaderLen]byte
	if _, err := io.ReadFull(conn, hdr[:]); err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}
	n := int(binary.LittleEndian.Uint32(hdr[:]))
	if n == 0 {
		return nil, nil
	}

	body := make([]byte, n*neighborLen)
	if _, err := io.ReadFull(conn, body); err != nil {
		return nil, fmt.Errorf("read body: %w", err)
	}

	neighbors := make([]search.Neighbor, n)
	for i := range neighbors {
		off := i * neighborLen
		neighbors[i] = search.Neighbor{
			Distance: math.Float32frombits(binary.LittleEndian.Uint32(body[off:])),
			IsFraud:  body[off+4] == 1,
		}
	}
	return neighbors, nil
}
