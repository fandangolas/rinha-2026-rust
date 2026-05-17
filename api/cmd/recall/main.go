// recall measures the accuracy of an IVF approximate search against a ground-truth
// full scan of the same (quantized) index.
//
// Usage:
//
//	recall -index /data/index.ivf.bin -probes 10 -n 500
//
// Ground truth is obtained by setting probes=numCentroids (scan every cluster).
// Query vectors are sampled directly from the stored int8 data so they follow
// the real distribution. Because a stored vector has distance 0 to itself,
// it always appears in both GT and approx top-5; we account for this in the
// neighbour-set comparison.
//
// The primary metric for this competition is decision accuracy:
//   - False Negative (FN): GT says reject (fraud), approx says approve → costs 3× in scoring
//   - False Positive (FP): GT says approve (legit), approx says reject → costs 1× in scoring
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"sort"
	"syscall"
	"unsafe"

	"github.com/fandangolas/rinha-de-backend-2026/api/internal/search"
	"github.com/fandangolas/rinha-de-backend-2026/api/internal/search/ivf"
)

const dims = search.Dims

func main() {
	indexPath := flag.String("index", "/tmp/rinha-data/index5k.ivf.bin", "path to IVF index")
	probes := flag.Int("probes", 10, "number of probes to evaluate")
	n := flag.Int("n", 500, "number of query vectors to sample")
	seed := flag.Int64("seed", 42, "random seed")
	k := flag.Int("k", 5, "number of neighbours")
	flag.Parse()

	// --- Load raw index data for sampling ---
	f, err := os.Open(*indexPath)
	if err != nil {
		log.Fatalf("open: %v", err)
	}
	fi, _ := f.Stat()
	raw, err := syscall.Mmap(int(f.Fd()), 0, int(fi.Size()), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		log.Fatalf("mmap: %v", err)
	}
	defer syscall.Munmap(raw)
	f.Close()

	numVecs := int(binary.LittleEndian.Uint64(raw[8:16]))
	numCents := int(binary.LittleEndian.Uint32(raw[16:20]))
	log.Printf("index: %d vectors, %d centroids", numVecs, numCents)

	// Locate the vector and label sections.
	vecOff := 32 + numCents*dims*4 + (numCents+1)*8
	labelOff := vecOff + numVecs*dims

	storedVecs := unsafe.Slice((*int8)(unsafe.Pointer(&raw[vecOff])), numVecs*dims)
	labels := raw[labelOff : labelOff+numVecs]

	// --- Create two searchers ---
	log.Printf("loading GT searcher (probes=%d) …", numCents)
	gt, err := ivf.Load(*indexPath, numCents)
	if err != nil {
		log.Fatalf("gt load: %v", err)
	}
	defer gt.Close()

	log.Printf("loading approx searcher (probes=%d) …", *probes)
	approx, err := ivf.Load(*indexPath, *probes)
	if err != nil {
		log.Fatalf("approx load: %v", err)
	}
	defer approx.Close()

	// Dataset fraud ratio (for context).
	var datasetFraud int
	for _, l := range labels {
		if l == 1 {
			datasetFraud++
		}
	}
	fmt.Printf("\nDataset: %d vectors, %.1f%% fraud\n\n",
		numVecs, float64(datasetFraud)/float64(numVecs)*100)

	// --- Run recall measurement ---
	rng := rand.New(rand.NewSource(*seed))

	type result struct {
		gtScore    float64
		approxScore float64
	}

	results := make([]result, 0, *n)

	for i := 0; i < *n; i++ {
		// Sample a random stored vector as query (real distribution).
		vi := rng.Intn(numVecs)
		base := vi * dims

		// Dequantize: int8 → float32
		var query search.Vector
		for j := 0; j < dims; j++ {
			query[j] = float32(storedVecs[base+j]) / ivf.QuantScale
		}

		gtNeighbors, _ := gt.Search(query, *k)
		approxNeighbors, _ := approx.Search(query, *k)

		gtScore := fraudScore(gtNeighbors)
		approxScore := fraudScore(approxNeighbors)

		results = append(results, result{gtScore, approxScore})
	}

	// --- Compute metrics ---
	var (
		decisionMatch int
		fp, fn        int
		totalFraud    int // GT fraud queries
		totalLegit    int // GT legit queries
		scoreErrors   []float64
	)

	for _, r := range results {
		gtApproved := r.gtScore < 0.6
		approxApproved := r.approxScore < 0.6

		if !gtApproved {
			totalFraud++
		} else {
			totalLegit++
		}

		scoreErrors = append(scoreErrors, math.Abs(r.gtScore-r.approxScore))

		if approxApproved == gtApproved {
			decisionMatch++
		} else if !gtApproved && approxApproved {
			fn++ // missed fraud
		} else {
			fp++ // blocked legit
		}
	}

	// Score error percentiles.
	sort.Float64s(scoreErrors)
	pct := func(p float64) float64 {
		idx := int(float64(len(scoreErrors))*p/100) - 1
		if idx < 0 {
			idx = 0
		}
		return scoreErrors[idx]
	}

	// Weighted error (matches competition formula: FP=1, FN=3)
	weightedErr := float64(fp*1+fn*3) / float64(len(results))

	fmt.Printf("─────────────────────────────────────────\n")
	fmt.Printf("Queries:          %d\n", len(results))
	fmt.Printf("GT fraud queries: %d (%.1f%%)\n", totalFraud, float64(totalFraud)/float64(len(results))*100)
	fmt.Printf("GT legit queries: %d (%.1f%%)\n", totalLegit, float64(totalLegit)/float64(len(results))*100)
	fmt.Printf("─────────────────────────────────────────\n")
	fmt.Printf("Decision accuracy:  %6.2f%%  (%d/%d correct)\n",
		float64(decisionMatch)/float64(len(results))*100, decisionMatch, len(results))
	fmt.Printf("False negatives:    %6.2f%%  (%d — approx approves fraud)\n",
		float64(fn)/float64(len(results))*100, fn)
	fmt.Printf("False positives:    %6.2f%%  (%d — approx blocks legit)\n",
		float64(fp)/float64(len(results))*100, fp)
	fmt.Printf("Weighted error:     %6.4f  (FP×1 + FN×3) / N\n", weightedErr)
	fmt.Printf("─────────────────────────────────────────\n")
	fmt.Printf("Fraud score error:\n")
	fmt.Printf("  p50:  %.3f\n", pct(50))
	fmt.Printf("  p90:  %.3f\n", pct(90))
	fmt.Printf("  p99:  %.3f\n", pct(99))
	fmt.Printf("  mean: %.3f\n", mean(scoreErrors))
	fmt.Printf("─────────────────────────────────────────\n")

	// Estimate detection score component using the competition formula.
	// E = FP*1 + FN*3 (HTTP errors assumed 0 for this test)
	E := float64(fp*1 + fn*3)
	N := float64(len(results))
	eps := math.Max(E/N, 0.001)
	failureRate := float64(fp+fn) / N
	var detScore float64
	if failureRate > 0.15 {
		detScore = -3000
	} else {
		detScore = 1000*math.Log10(1/eps) - 300*math.Log10(1+E)
	}
	fmt.Printf("Est. detection score component: %+.0f / +3000\n", detScore)
	fmt.Printf("─────────────────────────────────────────\n")
}

func fraudScore(neighbors []search.Neighbor) float64 {
	var n int
	for _, nb := range neighbors {
		if nb.IsFraud {
			n++
		}
	}
	return float64(n) / float64(len(neighbors))
}

func mean(xs []float64) float64 {
	var s float64
	for _, x := range xs {
		s += x
	}
	return s / float64(len(xs))
}
