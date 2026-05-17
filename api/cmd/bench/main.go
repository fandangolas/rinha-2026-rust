// bench is a latency benchmark for the /fraud-score endpoint.
//
// Usage:
//
//	bench -url http://localhost:9999 -n 5000 -c 10 -probes 20,30,50
//
// It generates synthetic but varied transaction requests, reports p50/p95/p99/p999,
// and can sweep multiple IVF_PROBES values if the server is restarted between runs.
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

func main() {
	url := flag.String("url", "http://localhost:9999", "base URL of the API")
	n := flag.Int("n", 5000, "total number of requests")
	c := flag.Int("c", 10, "concurrent workers")
	warmup := flag.Int("warmup", 200, "warmup requests (excluded from stats)")
	flag.Parse()

	client := &http.Client{
		Transport: &http.Transport{
			MaxIdleConnsPerHost: *c + 4,
		},
		Timeout: 5 * time.Second,
	}

	// Wait for the server to be ready.
	waitReady(*url+"/ready", client)

	rng := rand.New(rand.NewSource(42))
	requests := makeRequests(*n+*warmup, rng)

	fmt.Printf("warming up (%d requests)…\n", *warmup)
	runBatch(client, *url+"/fraud-score", requests[:*warmup], *c, true)

	fmt.Printf("benchmarking %d requests with concurrency %d…\n", *n, *c)
	latencies := runBatch(client, *url+"/fraud-score", requests[*warmup:], *c, false)

	printStats(latencies)
}

func waitReady(url string, client *http.Client) {
	fmt.Printf("waiting for %s … ", url)
	for {
		resp, err := client.Get(url)
		if err == nil && resp.StatusCode < 300 {
			resp.Body.Close()
			fmt.Println("ready")
			return
		}
		if resp != nil {
			resp.Body.Close()
		}
		time.Sleep(200 * time.Millisecond)
	}
}

func runBatch(client *http.Client, url string, reqs [][]byte, concurrency int, silent bool) []time.Duration {
	latencies := make([]time.Duration, 0, len(reqs))
	var mu sync.Mutex
	var errors atomic.Int64

	queue := make(chan []byte, len(reqs))
	for _, r := range reqs {
		queue <- r
	}
	close(queue)

	var wg sync.WaitGroup
	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for body := range queue {
				start := time.Now()
				resp, err := client.Post(url, "application/json", bytes.NewReader(body))
				elapsed := time.Since(start)

				if err != nil || resp.StatusCode != 200 {
					errors.Add(1)
					if resp != nil {
						resp.Body.Close()
					}
					continue
				}
				io.Copy(io.Discard, resp.Body)
				resp.Body.Close()

				mu.Lock()
				latencies = append(latencies, elapsed)
				mu.Unlock()
			}
		}()
	}
	wg.Wait()

	if !silent && errors.Load() > 0 {
		fmt.Printf("  errors: %d\n", errors.Load())
	}
	return latencies
}

func printStats(latencies []time.Duration) {
	if len(latencies) == 0 {
		fmt.Println("no successful requests")
		return
	}
	sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })

	n := len(latencies)
	var total time.Duration
	for _, d := range latencies {
		total += d
	}

	pct := func(p float64) time.Duration {
		idx := int(float64(n)*p/100) - 1
		if idx < 0 {
			idx = 0
		}
		if idx >= n {
			idx = n - 1
		}
		return latencies[idx]
	}

	bar := func(d time.Duration) string {
		ms := float64(d.Microseconds()) / 1000.0
		width := int(ms / 0.1) // 1 char per 0.1ms
		if width > 60 {
			width = 60
		}
		return strings.Repeat("█", width)
	}

	fmt.Printf("\n%-8s %8s\n", "percentile", "latency")
	fmt.Println(strings.Repeat("─", 40))
	for _, p := range []float64{50, 90, 95, 99, 99.9} {
		d := pct(p)
		fmt.Printf("p%-7.1f %8.3f ms  %s\n", p, float64(d.Microseconds())/1000.0, bar(d))
	}
	fmt.Println(strings.Repeat("─", 40))
	fmt.Printf("%-8s %8.3f ms\n", "mean", float64((total/time.Duration(n)).Microseconds())/1000.0)
	fmt.Printf("%-8s %8.3f ms\n", "min", float64(latencies[0].Microseconds())/1000.0)
	fmt.Printf("%-8s %8.3f ms\n", "max", float64(latencies[n-1].Microseconds())/1000.0)
	fmt.Printf("%-8s %8d\n", "n", n)
}

// MCC codes from mcc_risk.json
var mccCodes = []string{"5411", "5812", "5912", "5944", "7801", "7802", "7995", "4511", "5311", "5999"}

func makeRequests(n int, rng *rand.Rand) [][]byte {
	merchants := []string{"MERC-001", "MERC-002", "MERC-003", "MERC-004", "MERC-005",
		"MERC-006", "MERC-007", "MERC-008", "MERC-009", "MERC-010"}

	reqs := make([][]byte, n)
	baseTime := time.Date(2026, 3, 11, 14, 0, 0, 0, time.UTC)

	for i := 0; i < n; i++ {
		merchant := merchants[rng.Intn(len(merchants))]
		knownCount := rng.Intn(4)
		known := make([]string, knownCount)
		for j := range known {
			known[j] = merchants[rng.Intn(len(merchants))]
		}

		txTime := baseTime.Add(time.Duration(rng.Intn(86400)) * time.Second)

		var lastTx any
		if rng.Float64() > 0.3 { // 70% have last transaction
			lastTxTime := txTime.Add(-time.Duration(rng.Intn(1440)) * time.Minute)
			lastTx = map[string]any{
				"timestamp":       lastTxTime.Format(time.RFC3339),
				"km_from_current": rng.Float64() * 500,
			}
		}

		payload := map[string]any{
			"id": fmt.Sprintf("bench-%d", i),
			"transaction": map[string]any{
				"amount":       rng.Float64() * 5000,
				"installments": rng.Intn(12) + 1,
				"requested_at": txTime.Format(time.RFC3339),
			},
			"customer": map[string]any{
				"avg_amount":      rng.Float64() * 2000,
				"tx_count_24h":    rng.Intn(20),
				"known_merchants": known,
			},
			"merchant": map[string]any{
				"id":         merchant,
				"mcc":        mccCodes[rng.Intn(len(mccCodes))],
				"avg_amount": rng.Float64() * 3000,
			},
			"terminal": map[string]any{
				"is_online":    rng.Float64() > 0.5,
				"card_present": rng.Float64() > 0.3,
				"km_from_home": rng.Float64() * 200,
			},
			"last_transaction": lastTx,
		}

		b, _ := json.Marshal(payload)
		reqs[i] = b
	}
	return reqs
}
