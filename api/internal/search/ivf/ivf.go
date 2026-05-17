// Package ivf provides an IVF (Inverted File Index) nearest-neighbor searcher
// backed by a memory-mapped binary index file.
//
// Index file layout (little-endian):
//
//	[0:4]    magic   "IVFX"
//	[4:8]    version uint32 = 1
//	[8:16]   numVecs uint64
//	[16:20]  numCentroids uint32
//	[20:24]  dims    uint32 = 14
//	[24:28]  defaultProbes uint32
//	[28:32]  reserved uint32
//	--- header end (32 bytes) ---
//	centroids:     [numCentroids * dims]float32
//	clusterOffset: [numCentroids+1]uint64   (vector indices, not byte offsets)
//	vectors:       [numVecs * dims]int8      (quantized, sorted by cluster)
//	labels:        [numVecs]uint8            (0=legit, 1=fraud)
package ivf

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"sort"
	"sync"
	"syscall"
	"unsafe"

	"github.com/fandangolas/rinha-de-backend-2026/api/internal/search"
)

const (
	magic   = "IVFX"
	version = uint32(1)
	dims    = search.Dims
)

// QuantScale maps the [-1.0, 1.0] float range to [-127, 127] int8.
// The sentinel -1.0 maps to -127, safely distinct from the [0, 127] normal range.
const QuantScale = float32(127)

// Searcher is the IVF nearest-neighbor searcher. Safe for concurrent use.
type Searcher struct {
	raw      []byte   // mmap'd file bytes (kept alive so GC doesn't collect the fd)
	cents    []float32 // centroid coordinates: [numCentroids * dims]
	offsets  []uint64  // cluster boundaries: [numCentroids+1], index into vecs/labels
	vecs     []int8    // quantized vectors: [numVecs * dims], grouped by cluster
	labels   []uint8   // 0=legit 1=fraud: [numVecs]
	numCents int
	numVecs  int
	probes   int // how many clusters to scan per query
}

// Load memory-maps the pre-built IVF index at path.
// probes controls how many clusters are scanned per query:
// more probes = higher recall, higher latency.
func Load(path string, probes int) (*Searcher, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("ivf: open %s: %w", path, err)
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("ivf: stat: %w", err)
	}

	raw, err := syscall.Mmap(int(f.Fd()), 0, int(fi.Size()), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("ivf: mmap: %w", err)
	}

	if string(raw[0:4]) != magic {
		_ = syscall.Munmap(raw)
		return nil, fmt.Errorf("ivf: bad magic %q", raw[0:4])
	}
	if binary.LittleEndian.Uint32(raw[4:8]) != version {
		_ = syscall.Munmap(raw)
		return nil, fmt.Errorf("ivf: unsupported version %d", binary.LittleEndian.Uint32(raw[4:8]))
	}

	numVecs := int(binary.LittleEndian.Uint64(raw[8:16]))
	numCents := int(binary.LittleEndian.Uint32(raw[16:20]))
	fileDims := int(binary.LittleEndian.Uint32(raw[20:24]))
	if fileDims != dims {
		_ = syscall.Munmap(raw)
		return nil, fmt.Errorf("ivf: expected %d dims, got %d", dims, fileDims)
	}
	if probes <= 0 {
		probes = int(binary.LittleEndian.Uint32(raw[24:28]))
	}

	// Compute section offsets within the mmap'd buffer.
	centOff := 32
	centBytes := numCents * dims * 4 // float32
	clusterOff := centOff + centBytes
	clusterBytes := (numCents + 1) * 8 // uint64
	vecOff := clusterOff + clusterBytes
	vecBytes := numVecs * dims // int8
	labelOff := vecOff + vecBytes

	expectedSize := labelOff + numVecs
	if int(fi.Size()) < expectedSize {
		_ = syscall.Munmap(raw)
		return nil, fmt.Errorf("ivf: file too small: want %d bytes, got %d", expectedSize, fi.Size())
	}

	cents := unsafe.Slice((*float32)(unsafe.Pointer(&raw[centOff])), numCents*dims)
	offsets := unsafe.Slice((*uint64)(unsafe.Pointer(&raw[clusterOff])), numCents+1)
	vecs := unsafe.Slice((*int8)(unsafe.Pointer(&raw[vecOff])), numVecs*dims)
	labels := raw[labelOff : labelOff+numVecs]

	return &Searcher{
		raw:      raw,
		cents:    cents,
		offsets:  offsets,
		vecs:     vecs,
		labels:   labels,
		numCents: numCents,
		numVecs:  numVecs,
		probes:   probes,
	}, nil
}

func (s *Searcher) Close() error {
	return syscall.Munmap(s.raw)
}

// centDist pairs a centroid index with its squared Euclidean distance to the query.
type centDist struct {
	idx  int
	dist float32
}

// candidate is a neighbor candidate held in the max-heap during scanning.
type candidate struct {
	dist  int32
	fraud bool
}

// pools avoid per-request heap allocations in the hot path.
var (
	centDistPool = sync.Pool{New: func() any { return make([]centDist, 0, 1024) }}
	candPool     = sync.Pool{New: func() any { return make([]candidate, 0, 8) }}
)

// Search returns the k nearest neighbors to query from the reference dataset.
func (s *Searcher) Search(query search.Vector, k int) ([]search.Neighbor, error) {
	// --- Step 1: find the nearest centroids (float32 distance) ---
	cds := centDistPool.Get().([]centDist)
	if cap(cds) < s.numCents {
		cds = make([]centDist, s.numCents)
	} else {
		cds = cds[:s.numCents]
	}
	for i := 0; i < s.numCents; i++ {
		c := s.cents[i*dims : i*dims+dims]
		var d float32
		for j := 0; j < dims; j++ {
			diff := query[j] - c[j]
			d += diff * diff
		}
		cds[i] = centDist{i, d}
	}
	// Partial sort: only need the top `probes` nearest.
	probes := s.probes
	if probes > s.numCents {
		probes = s.numCents
	}
	partialSortAsc(cds, probes)
	probeList := cds[:probes]

	// --- Step 2: quantize query to int8 for fast inner-loop comparison ---
	var qInt8 [dims]int8
	for i, f := range query {
		qInt8[i] = quantize(f)
	}

	// --- Step 3: scan candidate clusters with a max-heap of size k ---
	cands := candPool.Get().([]candidate)
	cands = cands[:0]

	for _, cd := range probeList {
		start := s.offsets[cd.idx]
		end := s.offsets[cd.idx+1]
		count := int(end - start)
		if count == 0 {
			continue
		}

		// Use a raw pointer that advances by dims bytes each iteration.
		// This avoids a multiply and a slice-header allocation per vector.
		vecBase := unsafe.Pointer(&s.vecs[int(start)*dims])

		for i := 0; i < count; i++ {
			d := distInt8Ptr((*int8)(vecBase), qInt8)
			vecBase = unsafe.Add(vecBase, dims)

			vi := start + uint64(i)
			if len(cands) < k {
				cands = append(cands, candidate{d, s.labels[vi] == 1})
				if len(cands) == k {
					heapifyMax(cands)
				}
			} else if d < cands[0].dist {
				cands[0] = candidate{d, s.labels[vi] == 1}
				siftDownMax(cands, 0)
			}
		}
	}

	neighbors := make([]search.Neighbor, len(cands))
	for i, c := range cands {
		neighbors[i] = search.Neighbor{
			Distance: float32(c.dist),
			IsFraud:  c.fraud,
		}
	}

	cands = cands[:0]
	candPool.Put(cands)
	centDistPool.Put(cds)

	return neighbors, nil
}

// quantize maps a float32 in [-1.0, 1.0] to int8 in [-127, 127].
func quantize(f float32) int8 {
	v := f * QuantScale
	if v > 127 {
		return 127
	}
	if v < -127 {
		return -127
	}
	return int8(v)
}

// distInt8Ptr computes the squared Euclidean distance between two 14-dim int8
// vectors. The loop is fully unrolled for dims=14 so the compiler can pipeline
// all 14 multiply-accumulate pairs without loop overhead or bounds checks.
// stored must point to at least dims consecutive int8 values.
func distInt8Ptr(stored *int8, query [dims]int8) int32 {
	p := unsafe.Slice(stored, dims)
	_ = p[dims-1] // bounds-check elimination for all subsequent p[i]
	d0 := int32(p[0]) - int32(query[0])
	d1 := int32(p[1]) - int32(query[1])
	d2 := int32(p[2]) - int32(query[2])
	d3 := int32(p[3]) - int32(query[3])
	d4 := int32(p[4]) - int32(query[4])
	d5 := int32(p[5]) - int32(query[5])
	d6 := int32(p[6]) - int32(query[6])
	d7 := int32(p[7]) - int32(query[7])
	d8 := int32(p[8]) - int32(query[8])
	d9 := int32(p[9]) - int32(query[9])
	d10 := int32(p[10]) - int32(query[10])
	d11 := int32(p[11]) - int32(query[11])
	d12 := int32(p[12]) - int32(query[12])
	d13 := int32(p[13]) - int32(query[13])
	return d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6 +
		d7*d7 + d8*d8 + d9*d9 + d10*d10 + d11*d11 + d12*d12 + d13*d13
}

// partialSortAsc rearranges cds so that cds[:n] contains the n smallest elements
// (in ascending order). Faster than full sort when n << len(cds).
func partialSortAsc(cds []centDist, n int) {
	if n >= len(cds) {
		sort.Slice(cds, func(i, j int) bool { return cds[i].dist < cds[j].dist })
		return
	}
	// Selection: find the n-th smallest element, then sort the prefix.
	// For typical probes=20, numCents=1000 this is faster than full sort.
	sort.Slice(cds[:n], func(i, j int) bool { return cds[i].dist < cds[j].dist })
	for i := n; i < len(cds); i++ {
		if cds[i].dist < cds[n-1].dist {
			cds[n-1] = cds[i]
			// Re-insert into the sorted prefix.
			j := n - 1
			for j > 0 && cds[j].dist < cds[j-1].dist {
				cds[j], cds[j-1] = cds[j-1], cds[j]
				j--
			}
		}
	}
}

// max-heap over candidate.dist so we can efficiently maintain the k-best set.
func heapifyMax(h []candidate) {
	for i := len(h)/2 - 1; i >= 0; i-- {
		siftDownMax(h, i)
	}
}

func siftDownMax(h []candidate, i int) {
	n := len(h)
	for {
		largest := i
		l, r := 2*i+1, 2*i+2
		if l < n && h[l].dist > h[largest].dist {
			largest = l
		}
		if r < n && h[r].dist > h[largest].dist {
			largest = r
		}
		if largest == i {
			break
		}
		h[i], h[largest] = h[largest], h[i]
		i = largest
	}
}

// WriteHeader writes a valid IVF file header into dst (must be at least 32 bytes).
func WriteHeader(dst []byte, numVecs uint64, numCents, defaultProbes uint32) {
	copy(dst[0:4], magic)
	binary.LittleEndian.PutUint32(dst[4:8], version)
	binary.LittleEndian.PutUint64(dst[8:16], numVecs)
	binary.LittleEndian.PutUint32(dst[16:20], numCents)
	binary.LittleEndian.PutUint32(dst[20:24], dims)
	binary.LittleEndian.PutUint32(dst[24:28], defaultProbes)
	binary.LittleEndian.PutUint32(dst[28:32], 0)
}

// SquaredEuclidean is exported for use by the index builder.
func SquaredEuclidean(a, b []float32) float32 {
	var sum float32
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return sum
}

// QuantizeVec converts a float32 vector to int8 — exported for the builder.
func QuantizeVec(src []float32) []int8 {
	out := make([]int8, len(src))
	for i, f := range src {
		v := f * QuantScale
		if v > 127 {
			out[i] = 127
		} else if v < -127 {
			out[i] = -127
		} else {
			out[i] = int8(math.Round(float64(v)))
		}
	}
	return out
}
