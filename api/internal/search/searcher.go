package search

// Dims is the fixed number of dimensions for all transaction vectors.
const Dims = 14

// Vector is a normalized 14-dimensional transaction fingerprint.
// Values are in [0.0, 1.0] except the sentinel value -1.0 used for
// dimensions 5 and 6 when last_transaction is null.
type Vector [Dims]float32

// Neighbor is a single result from a k-nearest-neighbor search.
type Neighbor struct {
	Distance float32
	IsFraud  bool
}

// Searcher is the interface that both IVF and HNSW implementations satisfy.
// All implementations must be safe for concurrent use by multiple goroutines.
type Searcher interface {
	// Search returns the k nearest neighbors to query from the reference dataset.
	Search(query Vector, k int) ([]Neighbor, error)

	// Close releases any resources held by the searcher (e.g. mmap regions).
	Close() error
}
