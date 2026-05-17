package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strconv"

	"github.com/fandangolas/rinha-de-backend-2026/api/internal/handler"
	"github.com/fandangolas/rinha-de-backend-2026/api/internal/search"
	"github.com/fandangolas/rinha-de-backend-2026/api/internal/search/hnsw"
	"github.com/fandangolas/rinha-de-backend-2026/api/internal/search/ivf"
	"github.com/fandangolas/rinha-de-backend-2026/api/internal/search/uds"
	"github.com/fandangolas/rinha-de-backend-2026/api/internal/vectorize"
	"github.com/valyala/fasthttp"
)

func main() {
	norm := mustLoadNorm(env("NORM_PATH", "/data/normalization.json"))
	mccRisk := mustLoadMCCRisk(env("MCC_RISK_PATH", "/data/mcc_risk.json"))

	searcher := mustLoadSearcher()

	h := handler.New(searcher, norm, mccRisk)

	addr := ":" + env("PORT", "9999")
	log.Printf("listening on %s (searcher=%s, probes=%s)", addr, env("SEARCHER", "ivf"), env("IVF_PROBES", "20"))

	if err := fasthttp.ListenAndServe(addr, h.Router()); err != nil {
		log.Fatalf("server: %v", err)
	}
}

// mustLoadSearcher reads SEARCHER env var ("ivf" or "hnsw") and loads the index.
// Swap to "hnsw" once the HNSW implementation is complete.
func mustLoadSearcher() search.Searcher {
	kind := env("SEARCHER", "ivf")
	switch kind {
	case "ivf":
		path := env("INDEX_PATH", "/data/index.ivf.bin")
		probes := mustInt(env("IVF_PROBES", "20"))
		s, err := ivf.Load(path, probes)
		if err != nil {
			log.Fatalf("ivf: load %s: %v", path, err)
		}
		return s

	case "hnsw":
		path := env("INDEX_PATH", "/data/index.hnsw.bin")
		ef := mustInt(env("HNSW_EF", "50"))
		s, err := hnsw.Load(path, ef)
		if err != nil {
			log.Fatalf("hnsw: load %s: %v", path, err)
		}
		return s

	case "uds":
		// Load IVF index locally and serve it over a Unix domain socket.
		// The HTTP handler becomes a UDS client, adding one socket round-trip
		// per query. Used to benchmark IPC overhead vs direct mmap access.
		path := env("INDEX_PATH", "/data/index.ivf.bin")
		probes := mustInt(env("IVF_PROBES", "20"))
		backend, err := ivf.Load(path, probes)
		if err != nil {
			log.Fatalf("uds: load ivf %s: %v", path, err)
		}
		socketPath := env("UDS_PATH", "/tmp/search.sock")
		srv, err := uds.NewServer(socketPath, backend)
		if err != nil {
			log.Fatalf("uds: listen %s: %v", socketPath, err)
		}
		go srv.Serve()
		log.Printf("uds server listening on %s", socketPath)
		return uds.NewClient(socketPath)

	default:
		log.Fatalf("unknown SEARCHER %q — must be 'ivf' or 'hnsw'", kind)
		return nil
	}
}

func mustLoadNorm(path string) *vectorize.Normalization {
	data, err := os.ReadFile(path)
	if err != nil {
		log.Fatalf("normalization: read %s: %v", path, err)
	}
	var n vectorize.Normalization
	if err := json.Unmarshal(data, &n); err != nil {
		log.Fatalf("normalization: parse: %v", err)
	}
	return &n
}

func mustLoadMCCRisk(path string) vectorize.MCCRisk {
	data, err := os.ReadFile(path)
	if err != nil {
		log.Fatalf("mcc_risk: read %s: %v", path, err)
	}
	var m vectorize.MCCRisk
	if err := json.Unmarshal(data, &m); err != nil {
		log.Fatalf("mcc_risk: parse: %v", err)
	}
	return m
}

func env(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func mustInt(s string) int {
	v, err := strconv.Atoi(s)
	if err != nil {
		panic(fmt.Sprintf("mustInt(%q): %v", s, err))
	}
	return v
}
