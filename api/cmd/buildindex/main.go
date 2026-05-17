// buildindex builds the IVF binary index from references.json.gz.
//
// Usage:
//
//	buildindex -in references.json.gz -out index.ivf.bin \
//	           -centroids 1000 -sample 0.1 -iters 20 -probes 20
//
// The tool runs in two passes:
//  1. Sample ~10% of vectors → run k-means to find centroids.
//  2. Full pass → assign every vector to its nearest centroid,
//     quantize to int8, and write the binary index.
//
// Memory budget at 3 M vectors, 1000 centroids:
//   sample (10%): 300K × 14 × 4B  ≈  17 MB
//   clusters:     3M   × 15B       ≈  45 MB  (int8 vec + bool)
//   centroids:    1000 × 14 × 4B  ≈  56 KB
package main

import (
	"bufio"
	"compress/gzip"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"

	"github.com/fandangolas/rinha-de-backend-2026/api/internal/search"
	"github.com/fandangolas/rinha-de-backend-2026/api/internal/search/ivf"
)

const dims = search.Dims

func main() {
	inPath := flag.String("in", "references.json.gz", "path to references.json.gz")
	outPath := flag.String("out", "index.ivf.bin", "output binary index path")
	numCentroids := flag.Int("centroids", 1000, "number of IVF clusters")
	sampleRate := flag.Float64("sample", 0.1, "fraction of vectors to use for k-means (0.0–1.0)")
	iters := flag.Int("iters", 20, "k-means iterations")
	probes := flag.Int("probes", 20, "default probes written into the index header")
	seed := flag.Int64("seed", 42, "random seed")
	flag.Parse()

	rng := rand.New(rand.NewSource(*seed))

	// ---- Pass 1: sample vectors and run k-means -------------------------
	log.Printf("pass 1: sampling %.0f%% of vectors for k-means...", *sampleRate*100)
	sample, err := readSample(*inPath, *sampleRate, rng)
	if err != nil {
		log.Fatalf("sample: %v", err)
	}
	log.Printf("  sampled %d vectors", len(sample))

	log.Printf("running k-means (k=%d, iters=%d)...", *numCentroids, *iters)
	centroids := kMeans(sample, *numCentroids, *iters, rng)
	log.Printf("  k-means done")

	// ---- Pass 2: assign all vectors and collect per-cluster data --------
	// Vectors are quantized to int8 at this stage to keep memory ~45 MB
	// rather than ~240 MB if stored as float32.
	log.Printf("pass 2: assigning all vectors to clusters...")
	clusters := make([][]entry, *numCentroids)
	totalVecs, err := assignAll(*inPath, centroids, clusters)
	if err != nil {
		log.Fatalf("assign: %v", err)
	}
	log.Printf("  total vectors: %d", totalVecs)

	// ---- Write binary index ---------------------------------------------
	log.Printf("writing index to %s...", *outPath)
	if err := writeIndex(*outPath, centroids, clusters, uint64(totalVecs), uint32(*numCentroids), uint32(*probes)); err != nil {
		log.Fatalf("write: %v", err)
	}
	log.Printf("done — index written to %s", *outPath)
}

// entry holds a single reference vector quantized to int8, plus its fraud label.
// Using a fixed-size array avoids a heap allocation per vector and keeps the
// cluster slices compact (~15 bytes/entry vs ~80 bytes with []float32).
type entry struct {
	vec   [dims]int8
	fraud bool
}

// refLine is the JSON structure of each line in references.json.gz.
type refLine struct {
	Vector []float32 `json:"vector"`
	Label  string    `json:"label"`
}

func openGZ(path string) (*bufio.Reader, func(), error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	gz, err := gzip.NewReader(f)
	if err != nil {
		f.Close()
		return nil, nil, err
	}
	return bufio.NewReaderSize(gz, 1<<20), func() { gz.Close(); f.Close() }, nil
}

func readSample(path string, rate float64, rng *rand.Rand) ([][]float32, error) {
	r, closeFn, err := openGZ(path)
	if err != nil {
		return nil, err
	}
	defer closeFn()

	dec := json.NewDecoder(r)
	if err := consumeArrayOpen(dec); err != nil {
		return nil, err
	}
	var sample [][]float32
	for dec.More() {
		var line refLine
		if err := dec.Decode(&line); err != nil {
			break
		}
		if rng.Float64() < rate {
			vec := make([]float32, dims)
			copy(vec, line.Vector)
			sample = append(sample, vec)
		}
	}
	return sample, nil
}

func assignAll(path string, centroids [][]float32, clusters [][]entry) (int, error) {
	r, closeFn, err := openGZ(path)
	if err != nil {
		return 0, err
	}
	defer closeFn()

	dec := json.NewDecoder(r)
	if err := consumeArrayOpen(dec); err != nil {
		return 0, err
	}
	total := 0
	for dec.More() {
		var line refLine
		if err := dec.Decode(&line); err != nil {
			break
		}
		ci := nearestCentroid(line.Vector, centroids)

		var e entry
		q := ivf.QuantizeVec(line.Vector)
		copy(e.vec[:], q)
		e.fraud = line.Label == "fraud"
		clusters[ci] = append(clusters[ci], e)
		total++

		if total%500_000 == 0 {
			log.Printf("  assigned %d vectors...", total)
		}
	}
	return total, nil
}

func nearestCentroid(vec []float32, centroids [][]float32) int {
	best := 0
	bestDist := float32(math.MaxFloat32)
	for i, c := range centroids {
		d := ivf.SquaredEuclidean(vec, c)
		if d < bestDist {
			bestDist = d
			best = i
		}
	}
	return best
}

// kMeans runs Lloyd's algorithm on sample vectors.
func kMeans(sample [][]float32, k, iters int, rng *rand.Rand) [][]float32 {
	perm := rng.Perm(len(sample))
	centroids := make([][]float32, k)
	for i := 0; i < k; i++ {
		c := make([]float32, dims)
		copy(c, sample[perm[i]])
		centroids[i] = c
	}

	assignments := make([]int, len(sample))

	for iter := 0; iter < iters; iter++ {
		for vi, v := range sample {
			assignments[vi] = nearestCentroid(v, centroids)
		}

		newCents := make([][]float32, k)
		counts := make([]int, k)
		for i := range newCents {
			newCents[i] = make([]float32, dims)
		}
		for vi, v := range sample {
			ci := assignments[vi]
			counts[ci]++
			for j := range v {
				newCents[ci][j] += v[j]
			}
		}
		for ci, c := range newCents {
			if counts[ci] > 0 {
				for j := range c {
					c[j] /= float32(counts[ci])
				}
				centroids[ci] = c
			}
		}
		log.Printf("  k-means iter %d/%d", iter+1, iters)
	}
	return centroids
}

func writeIndex(path string, centroids [][]float32, clusters [][]entry, numVecs uint64, numCents, defaultProbes uint32) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := bufio.NewWriterSize(f, 1<<20)

	// Header (32 bytes).
	hdr := make([]byte, 32)
	ivf.WriteHeader(hdr, numVecs, numCents, defaultProbes)
	if _, err := w.Write(hdr); err != nil {
		return err
	}

	// Centroids: [numCents * dims]float32.
	for _, c := range centroids {
		for _, f32 := range c {
			if err := writeFloat32(w, f32); err != nil {
				return err
			}
		}
	}

	// Cluster offsets: [numCents+1]uint64.
	offsets := make([]uint64, int(numCents)+1)
	var cur uint64
	for i, cl := range clusters {
		offsets[i] = cur
		cur += uint64(len(cl))
	}
	offsets[numCents] = cur
	for _, off := range offsets {
		if err := writeUint64(w, off); err != nil {
			return err
		}
	}

	// Vector data: [numVecs * dims]int8 (already quantized at assignment time).
	for _, cl := range clusters {
		for _, e := range cl {
			for _, b := range e.vec {
				if err := w.WriteByte(byte(b)); err != nil {
					return err
				}
			}
		}
	}

	// Label data: [numVecs]uint8.
	for _, cl := range clusters {
		for _, e := range cl {
			label := uint8(0)
			if e.fraud {
				label = 1
			}
			if err := w.WriteByte(label); err != nil {
				return err
			}
		}
	}

	return w.Flush()
}

func writeFloat32(w *bufio.Writer, v float32) error {
	var buf [4]byte
	binary.LittleEndian.PutUint32(buf[:], math.Float32bits(v))
	_, err := w.Write(buf[:])
	return err
}

func writeUint64(w *bufio.Writer, v uint64) error {
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], v)
	_, err := w.Write(buf[:])
	return err
}

// consumeArrayOpen reads the opening '[' token from a JSON array stream.
// references.json.gz is a single JSON array, not NDJSON.
func consumeArrayOpen(dec *json.Decoder) error {
	tok, err := dec.Token()
	if err != nil {
		return fmt.Errorf("expected '[': %w", err)
	}
	if tok != json.Delim('[') {
		return fmt.Errorf("expected '[', got %v", tok)
	}
	return nil
}
