// Package hnsw provides an HNSW-based approximate nearest-neighbor searcher.
// This implementation is a stub — switch to it if IVF cannot hit p99 < 1ms
// under benchmark. Building the HNSW index requires a shared-mmap setup
// (see docs) because the graph overhead (~150-250 MB) must be shared between
// both API containers to stay within the 350 MB total RAM budget.
package hnsw

import (
	"errors"

	"github.com/fandangolas/rinha-de-backend-2026/api/internal/search"
)

var ErrNotImplemented = errors.New("hnsw: not yet implemented — use IVF searcher")

// Searcher is the HNSW nearest-neighbor searcher.
type Searcher struct{}

// Load opens a pre-built HNSW index from path and memory-maps it.
// ef controls the search-time quality/speed trade-off (higher = better recall).
func Load(_ string, _ int) (*Searcher, error) {
	return nil, ErrNotImplemented
}

func (s *Searcher) Search(_ search.Vector, _ int) ([]search.Neighbor, error) {
	return nil, ErrNotImplemented
}

func (s *Searcher) Close() error { return nil }
