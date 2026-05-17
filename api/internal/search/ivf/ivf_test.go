package ivf_test

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"sort"
	"testing"

	"github.com/fandangolas/rinha-de-backend-2026/api/internal/search"
	"github.com/fandangolas/rinha-de-backend-2026/api/internal/search/ivf"
)

// buildTestIndex writes a minimal IVF index to a temp file and returns its path.
//
// Layout:
//
//	Centroid 0: [0, 0, ..., 0]   (14 dims) — "low" cluster
//	Centroid 1: [1, 1, ..., 1]   (14 dims) — "high" cluster
//
//	Cluster 0 (legit):  v0=[0.1,...], v1=[0.2,...], v2=[0.3,...]
//	Cluster 1 (fraud):  v3=[0.7,...], v4=[0.8,...], v5=[0.9,...]
//
// Quantized (f*127, truncated):
//
//	0.1→12, 0.2→25, 0.3→38, 0.7→88, 0.8→101, 0.9→114
func buildTestIndex(t *testing.T) (path string, cleanup func()) {
	t.Helper()
	const (
		numCents     = 2
		numVecs      = 6
		defaultProbe = 2
	)

	var buf bytes.Buffer

	// Header
	hdr := make([]byte, 32)
	ivf.WriteHeader(hdr, numVecs, numCents, defaultProbe)
	buf.Write(hdr)

	// Centroids: 2 × 14 float32
	for _, val := range []float32{0.0, 1.0} {
		for d := 0; d < search.Dims; d++ {
			b := make([]byte, 4)
			binary.LittleEndian.PutUint32(b, math.Float32bits(val))
			buf.Write(b)
		}
	}

	// Cluster offsets: [0, 3, 6]  (uint64)
	for _, off := range []uint64{0, 3, 6} {
		b := make([]byte, 8)
		binary.LittleEndian.PutUint64(b, off)
		buf.Write(b)
	}

	// Quantized vectors: 6 × 14 int8
	// 0.1*127=12.7→12, 0.2*127=25.4→25, 0.3*127=38.1→38,
	// 0.7*127=88.9→88, 0.8*127=101.6→101, 0.9*127=114.3→114
	for _, q := range []int8{12, 25, 38, 88, 101, 114} {
		for d := 0; d < search.Dims; d++ {
			buf.WriteByte(byte(q))
		}
	}

	// Labels: 0=legit (cluster 0), 1=fraud (cluster 1)
	buf.Write([]byte{0, 0, 0, 1, 1, 1})

	f, err := os.CreateTemp(t.TempDir(), "ivf-*.bin")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.Write(buf.Bytes()); err != nil {
		f.Close()
		t.Fatal(err)
	}
	f.Close()
	return f.Name(), func() {}
}

func loadTest(t *testing.T, probes int) *ivf.Searcher {
	t.Helper()
	path, _ := buildTestIndex(t)
	s, err := ivf.Load(path, probes)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	return s
}

// queryAll returns a Vector with every dimension set to val.
func queryAll(val float32) search.Vector {
	var v search.Vector
	for i := range v {
		v[i] = val
	}
	return v
}

func distances(neighbors []search.Neighbor) []float32 {
	ds := make([]float32, len(neighbors))
	for i, n := range neighbors {
		ds[i] = n.Distance
	}
	sort.Slice(ds, func(i, j int) bool { return ds[i] < ds[j] })
	return ds
}

// TestSearchNearestIsExactMatch verifies that querying with a stored vector
// returns a neighbor with distance 0 as the best match.
func TestSearchNearestIsExactMatch(t *testing.T) {
	s := loadTest(t, 2)

	// Query matches v0 exactly (quantized [12,...,12]).
	neighbors, err := s.Search(queryAll(0.1), 3)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(neighbors) != 3 {
		t.Fatalf("want 3 neighbors, got %d", len(neighbors))
	}

	ds := distances(neighbors)
	if ds[0] != 0 {
		t.Errorf("best neighbor distance: want 0, got %v", ds[0])
	}
}

// TestSearchFraudLabels verifies fraud/legit labels are returned correctly.
func TestSearchFraudLabels(t *testing.T) {
	s := loadTest(t, 2)

	t.Run("low cluster returns legit", func(t *testing.T) {
		neighbors, err := s.Search(queryAll(0.1), 3)
		if err != nil {
			t.Fatal(err)
		}
		for _, n := range neighbors {
			if n.IsFraud {
				t.Errorf("expected legit neighbor, got fraud (dist=%v)", n.Distance)
			}
		}
	})

	t.Run("high cluster returns fraud", func(t *testing.T) {
		neighbors, err := s.Search(queryAll(0.9), 3)
		if err != nil {
			t.Fatal(err)
		}
		for _, n := range neighbors {
			if !n.IsFraud {
				t.Errorf("expected fraud neighbor, got legit (dist=%v)", n.Distance)
			}
		}
	})
}

// TestSearchKLargerThanIndex returns at most numVecs results, not k.
func TestSearchKLargerThanIndex(t *testing.T) {
	s := loadTest(t, 2)
	neighbors, err := s.Search(queryAll(0.5), 100)
	if err != nil {
		t.Fatal(err)
	}
	if len(neighbors) > 6 {
		t.Errorf("want ≤ 6 neighbors (index size), got %d", len(neighbors))
	}
}

// TestSearchSingleProbeOnlyReturnsOneCluster verifies that probes=1 limits
// the search to one cluster (3 neighbors max from a 6-vector index).
func TestSearchSingleProbeOnlyReturnsOneCluster(t *testing.T) {
	s := loadTest(t, 1)
	neighbors, err := s.Search(queryAll(0.1), 10)
	if err != nil {
		t.Fatal(err)
	}
	// Only one cluster (3 vectors) should be scanned.
	if len(neighbors) != 3 {
		t.Errorf("want 3 neighbors (1 cluster), got %d", len(neighbors))
	}
	for _, n := range neighbors {
		if n.IsFraud {
			t.Errorf("cluster 0 should be all legit")
		}
	}
}

// BenchmarkSearch measures IVF search throughput using the real index.
// Run with: go test -bench=BenchmarkSearch -benchtime=5s ./api/internal/search/ivf/
func BenchmarkSearch(b *testing.B) {
	const realIndex = "/tmp/rinha-data/index.ivf.bin"
	if _, err := os.Stat(realIndex); err != nil {
		b.Skip("real index not available at " + realIndex)
	}
	s, err := ivf.Load(realIndex, 5)
	if err != nil {
		b.Fatal(err)
	}
	defer s.Close()

	query := queryAll(0.3)

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, _ = s.Search(query, 5)
		}
	})
}
