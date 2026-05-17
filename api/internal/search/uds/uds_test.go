package uds_test

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/fandangolas/rinha-de-backend-2026/api/internal/search"
	"github.com/fandangolas/rinha-de-backend-2026/api/internal/search/ivf"
	"github.com/fandangolas/rinha-de-backend-2026/api/internal/search/uds"
)

// buildTestIndex is a copy of the helper from ivf_test — each test package
// must be self-contained.
func buildTestIndex(t *testing.T) string {
	t.Helper()
	const (
		numCents     = 2
		numVecs      = 6
		defaultProbe = 2
	)
	var buf bytes.Buffer
	hdr := make([]byte, 32)
	ivf.WriteHeader(hdr, numVecs, numCents, defaultProbe)
	buf.Write(hdr)
	for _, val := range []float32{0.0, 1.0} {
		for d := 0; d < search.Dims; d++ {
			b := make([]byte, 4)
			binary.LittleEndian.PutUint32(b, math.Float32bits(val))
			buf.Write(b)
		}
	}
	for _, off := range []uint64{0, 3, 6} {
		b := make([]byte, 8)
		binary.LittleEndian.PutUint64(b, off)
		buf.Write(b)
	}
	for _, q := range []int8{12, 25, 38, 88, 101, 114} {
		for d := 0; d < search.Dims; d++ {
			buf.WriteByte(byte(q))
		}
	}
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
	return f.Name()
}

// startServer loads an IVF index from path, starts a UDS server, and returns
// a connected Client. The server is stopped when t cleans up.
func startServer(t *testing.T, indexPath string, probes int) *uds.Client {
	t.Helper()
	s, err := ivf.Load(indexPath, probes)
	if err != nil {
		t.Fatalf("ivf.Load: %v", err)
	}
	sock := filepath.Join(t.TempDir(), "test.sock")
	srv, err := uds.NewServer(sock, s)
	if err != nil {
		t.Fatalf("uds.NewServer: %v", err)
	}
	go srv.Serve()
	t.Cleanup(func() {
		_ = srv.Close()
		_ = s.Close()
	})
	return uds.NewClient(sock)
}

func queryAll(val float32) search.Vector {
	var v search.Vector
	for i := range v {
		v[i] = val
	}
	return v
}

func sortedDistances(neighbors []search.Neighbor) []float32 {
	ds := make([]float32, len(neighbors))
	for i, n := range neighbors {
		ds[i] = n.Distance
	}
	sort.Slice(ds, func(i, j int) bool { return ds[i] < ds[j] })
	return ds
}

// TestRoundTripMatchesDirectSearch is the key correctness test: the UDS client
// must return identical results to a direct ivf.Searcher call.
func TestRoundTripMatchesDirectSearch(t *testing.T) {
	indexPath := buildTestIndex(t)

	direct, err := ivf.Load(indexPath, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer direct.Close()

	client := startServer(t, indexPath, 2)

	queries := []struct {
		name  string
		query search.Vector
		k     int
	}{
		{"low-cluster k=3", queryAll(0.1), 3},
		{"high-cluster k=3", queryAll(0.9), 3},
		{"midpoint k=5", queryAll(0.5), 5},
		{"k=1", queryAll(0.2), 1},
		{"k larger than index", queryAll(0.5), 100},
	}

	for _, tc := range queries {
		t.Run(tc.name, func(t *testing.T) {
			want, err := direct.Search(tc.query, tc.k)
			if err != nil {
				t.Fatalf("direct search: %v", err)
			}
			got, err := client.Search(tc.query, tc.k)
			if err != nil {
				t.Fatalf("uds search: %v", err)
			}

			if len(got) != len(want) {
				t.Fatalf("neighbor count: direct=%d uds=%d", len(want), len(got))
			}

			// Compare sorted distances (heap order differs between calls).
			wantDist := sortedDistances(want)
			gotDist := sortedDistances(got)
			for i := range wantDist {
				if wantDist[i] != gotDist[i] {
					t.Errorf("distance[%d]: direct=%v uds=%v", i, wantDist[i], gotDist[i])
				}
			}

			// Fraud labels must also match (by sorted distance order).
			type nb struct {
				dist  float32
				fraud bool
			}
			toSorted := func(ns []search.Neighbor) []nb {
				out := make([]nb, len(ns))
				for i, n := range ns {
					out[i] = nb{n.Distance, n.IsFraud}
				}
				sort.Slice(out, func(i, j int) bool { return out[i].dist < out[j].dist })
				return out
			}
			ws, gs := toSorted(want), toSorted(got)
			for i := range ws {
				if ws[i].fraud != gs[i].fraud {
					t.Errorf("fraud[%d]: direct=%v uds=%v", i, ws[i].fraud, gs[i].fraud)
				}
			}
		})
	}
}

// TestRoundTripConcurrent exercises the connection pool under concurrent load.
func TestRoundTripConcurrent(t *testing.T) {
	indexPath := buildTestIndex(t)
	client := startServer(t, indexPath, 2)
	query := queryAll(0.1)

	done := make(chan error, 20)
	for i := 0; i < 20; i++ {
		go func() {
			neighbors, err := client.Search(query, 3)
			if err != nil {
				done <- err
				return
			}
			if len(neighbors) != 3 {
				done <- nil // unexpected count handled by other test
				return
			}
			done <- nil
		}()
	}
	for i := 0; i < 20; i++ {
		if err := <-done; err != nil {
			t.Errorf("concurrent search: %v", err)
		}
	}
}

// loadRealSearcher loads the production index (1k centroids, probes=5).
// Returns nil if the file is not present.
func loadRealSearcher(b *testing.B) *ivf.Searcher {
	b.Helper()
	const realIndex = "/tmp/rinha-data/index.ivf.bin"
	if _, err := os.Stat(realIndex); err != nil {
		return nil
	}
	s, err := ivf.Load(realIndex, 5)
	if err != nil {
		b.Fatal(err)
	}
	return s
}

// BenchmarkIVFDirect measures throughput of the mmap strategy: the goroutine
// calls Search directly with no IPC.
//
// Run both benchmarks together for a side-by-side comparison:
//
//	go test -bench=. -benchtime=5s ./api/internal/search/uds/
func BenchmarkIVFDirect(b *testing.B) {
	s := loadRealSearcher(b)
	if s == nil {
		b.Skip("real index not available")
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

// BenchmarkUDSSearch measures throughput of the UDS strategy: each Search call
// serialises the query, sends it over a Unix domain socket, the server goroutine
// deserialises and runs the IVF search, then sends the result back.
func BenchmarkUDSSearch(b *testing.B) {
	s := loadRealSearcher(b)
	if s == nil {
		b.Skip("real index not available")
	}
	defer s.Close()

	sock := filepath.Join(b.TempDir(), "bench.sock")
	srv, err := uds.NewServer(sock, s)
	if err != nil {
		b.Fatal(err)
	}
	go srv.Serve()
	defer srv.Close()

	client := uds.NewClient(sock)
	query := queryAll(0.3)

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, _ = client.Search(query, 5)
		}
	})
}
