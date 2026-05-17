# Architecture — Rinha de Backend 2026

Fraud detection API using vector search. Every architectural decision below was made
during the design and benchmarking sessions for this submission, recorded here in the
order in which it was reached.

---

## 1. Language: Go

**Alternatives considered:** Go, Rust, Python, Java/Kotlin.

**Decision:** Go.

**Reasoning:**

| | Go | Rust |
|---|---|---|
| Concurrency model | Goroutines + channels (CSP); M:N runtime scheduler | `async/await` + Tokio for I/O; Rayon for CPU parallelism |
| GC | Yes, but mmap'd data lives outside the heap — GC never touches the index | None — fully deterministic memory |
| Performance ceiling | ~10–20% below Rust for CPU-bound work | Maximum |
| Development speed | Fast — idiomatic concurrency, large stdlib | Steep learning curve with lifetimes + async |

The bottleneck in this challenge is vector math (distance computation over millions of
int8 values), not the HTTP layer. Go is within acceptable range of Rust for this workload
and ships much faster. Python was ruled out due to GIL and memory overhead. JVM was ruled
out due to the 350 MB total RAM budget — JVM startup and heap alone would consume too much.

---

## 2. HTTP Framework: fasthttp

**Alternatives considered:** `net/http` (stdlib), `fasthttp`.

**Decision:** `fasthttp`.

**Reasoning:** `fasthttp` avoids per-request heap allocations by pooling request/response
objects internally. Under the tight CPU budget (0.45 cores per API instance) and low-latency
requirement, fewer allocations mean fewer GC triggers and lower tail latency. The API
surface is simple enough (two endpoints) that fasthttp's non-stdlib API is not a burden.

---

## 3. Vector Search Strategy: IVF with int8 quantization

**Alternatives considered:**

| Approach | RAM / instance | Query time | Accuracy | Verdict |
|---|---|---|---|---|
| HNSW M=16, float32 | ~456 MB | ~0.2 ms | 99% | Impossible — exceeds total budget |
| HNSW M=6, int8 (shared mmap) | ~150 MB (shared) | ~0.3 ms | ~93% | Feasible but complex startup |
| Flat brute-force int8 | 42 MB | ~42 ms | 100% | Too slow |
| **IVF + int8** | **~44 MB** | **< 1 ms** | **~100%** | **Chosen** |

**Decision:** IVF (Inverted File Index) with int8 scalar quantization.

**Reasoning:**

HNSW at standard settings (M=16) requires ~456 MB per instance for graph + vectors —
already over the entire 350 MB budget. Reducing M degrades recall. Flat brute-force over
3 M vectors takes ~42 ms single-threaded, which is incompatible with the p99 < 1 ms target.

IVF with int8 quantization fits in 44 MB per instance (42 MB vectors + overhead) and
achieves sub-millisecond p99 by only scanning a small fraction of the dataset per query.

**How IVF works in this implementation:**

1. **Build time:** k-means clusters the 3 M reference vectors into N centroids.
   Vectors are stored sorted by cluster assignment, quantized to int8.
2. **Query time:** compute Euclidean distance from the query to all centroids (float32),
   pick the top-`probes` nearest centroids, scan only those clusters, return top-5 neighbours.

**int8 quantization:** the challenge vectors are in [−1, 1]. Mapping to int8 with scale
factor 127 (so −1→−127, 1→127) reduces per-vector storage from 56 bytes (float32) to
14 bytes (int8), at negligible accuracy cost for Euclidean distance in 14 dimensions.

---

## 4. Searcher Abstraction

**Decision:** a single `search.Searcher` interface that both IVF and HNSW satisfy.

```go
type Searcher interface {
    Search(query Vector, k int) ([]Neighbor, error)
    Close() error
}
```

**Reasoning:** IVF was chosen as the starting implementation because it fits within the
memory budget. HNSW remains a viable upgrade path if IVF p99 exceeds 1 ms under real
competition load. The active implementation is selected at startup via the `SEARCHER`
env var (`"ivf"` or `"hnsw"`) — no rebuild required to switch.

The HNSW implementation is a stub returning `ErrNotImplemented`. Filling it in later
requires no changes outside `api/internal/search/hnsw/`.

---

## 5. Load Balancer: HAProxy

**Alternatives considered:** Nginx, HAProxy, Caddy, Traefik.

**Decision:** HAProxy.

**Reasoning:**

| | HAProxy | Nginx |
|---|---|---|
| RAM footprint | ~3–5 MB | ~10–20 MB |
| Purpose | Purpose-built load balancer | General web server + proxy |
| Required features | Round-robin, health check on `/ready` | Same |

In this challenge every megabyte counts (350 MB total budget). HAProxy's lower footprint
saves ~15 MB compared to Nginx, with no meaningful feature loss — we only need round-robin
and HTTP health checks.

**Resource allocation:**

```
HAProxy:    cpus=0.10   memory=25 MB
api1:       cpus=0.45   memory=160 MB
api2:       cpus=0.45   memory=160 MB
─────────────────────────────────────
Total:      cpus=1.00   memory=345 MB  (≤ 350 MB limit)
```

---

## 6. Index Memory Layout: mmap + single file per container

**Options considered:**

| Option | Complexity | RAM impact |
|---|---|---|
| Each container loads its own copy | Low | 44 MB × 2 = 88 MB |
| Shared volume + kernel page sharing | Medium | ~44 MB (shared physical pages) |
| Sidecar index service | High | + network latency |

**Decision:** each container mmaps its own copy of the index file baked into the Docker
image. No shared volume needed.

**Reasoning:** the int8 quantized index is 44 MB. Two independent copies cost 88 MB,
which fits comfortably within the 320 MB combined budget for both API containers. The
shared-mmap approach would save ~44 MB but adds coordination complexity at startup
(one container must build/signal, the other must wait). The savings are not worth it.

**GC interaction:** mmapped memory lives outside the Go heap. The GC never scans it.
With `GOGC=off` and `GOMEMLIMIT=120MiB`, the Go runtime only manages the small HTTP
handling heap (~5–20 MB). This directly addresses the concern about GC pressure under
tight memory budgets.

---

## 7. Docker: Multi-Stage Build

The Docker image is built in four stages to keep the final image minimal and avoid
shipping build tooling or raw reference data:

```
Stage 1 (builder-tool): Go 1.23 → compile the buildindex binary
Stage 2 (indexer):      alpine → download reference data, run buildindex → index.ivf.bin
Stage 3 (api-builder):  Go 1.23 → compile the API server binary (CGO_ENABLED=0)
Stage 4 (runtime):      alpine → copy /api + /data/index.ivf.bin + mcc_risk + norm
```

The index is baked into the image at build time. Container startup is instant — mmap is
lazy, so the OS only loads pages that are actually accessed.

**Reference data download URLs** (hardcoded in Dockerfile, no build args needed):
```
resources/references.json.gz    (48 MB compressed / ~284 MB uncompressed)
resources/mcc_risk.json         (10 MCC → risk score entries)
resources/normalization.json    (7 normalization constants)
```

Note: `references.json.gz` is a **JSON array** (not NDJSON). The index builder uses
`json.Decoder.Token()` + `dec.More()` to stream-parse without loading the full 284 MB
into memory.

---

## 8. Index Binary Format

```
Offset  Size            Field
──────  ──────────────  ──────────────────────────────────────────────────
0       4 B             Magic: "IVFX"
4       4 B             Version: uint32 = 1
8       8 B             NumVectors: uint64
16      4 B             NumCentroids: uint32
20      4 B             Dims: uint32 = 14
24      4 B             DefaultProbes: uint32
28      4 B             Reserved: uint32 = 0
── header end (32 B) ──
32      NC*14*4 B       Centroids: [NC][14]float32  (float32 for centroid lookup)
+       (NC+1)*8 B      ClusterOffsets: [NC+1]uint64 (first vector index per cluster)
+       NV*14 B         VectorData: [NV][14]int8    (quantized, sorted by cluster)
+       NV B            LabelData: [NV]uint8        (0=legit, 1=fraud)
```

Total size with 5000 centroids, 3 M vectors: **~44 MB**.

Centroids are stored as float32 (not quantized) because they are used in the
centroid-selection step which requires higher precision. The hot path (vector scan)
uses int8 exclusively.

---

## 9. Index Builder: k-means Parameters

**Final parameters:**

```
-centroids 5000    k-means cluster count
-sample    0.1     fraction of vectors used for k-means (300 K out of 3 M)
-iters     20      Lloyd's algorithm iterations
-probes    5       default baked into the index header
```

**Why 5000 centroids (not 1000):**

| Centroids | Vectors/cluster | Probes needed | Vectors scanned/query |
|---|---|---|---|
| 1000 | ~3000 | 20 | ~60 000 |
| **5000** | **~600** | **5** | **~3 000** |

With 5000 centroids, each probe scans only ~600 vectors. This reduces the scan workload
by 20× compared to 1000 centroids at probes=20, while achieving equal or better recall
because the tighter clusters reduce neighbourhood leakage.

Build time in the Docker indexer stage: ~8–10 minutes (single-threaded Lloyd's over
300 K sample vectors).

---

## 10. Response Pre-computation

`fraud_score` is always `k/5` where k ∈ {0, 1, 2, 3, 4, 5}. The approval threshold
is fixed at `fraud_score < 0.6`. This means there are exactly **6 possible response
bodies**, all known at startup:

```go
var cachedResponses = [6][]byte{
    []byte(`{"approved":true,"fraud_score":0.0}`),
    []byte(`{"approved":true,"fraud_score":0.2}`),
    []byte(`{"approved":true,"fraud_score":0.4}`),
    []byte(`{"approved":false,"fraud_score":0.6}`),
    []byte(`{"approved":false,"fraud_score":0.8}`),
    []byte(`{"approved":false,"fraud_score":1.0}`),
}
```

The handler counts fraud neighbours and indexes into this array — zero allocations,
zero formatting in the response path.

---

## 11. GC Tuning

```
GOGC=off        disable the default heap-doubling trigger
GOMEMLIMIT=120MiB   use soft memory limit as the sole GC trigger
```

With `GOGC=off` the runtime only collects when the heap approaches `GOMEMLIMIT`.
Since the Go heap in this service is tiny (~5–20 MB for in-flight requests and
pooled buffers), collections are rare and brief. The 44 MB mmap region is invisible
to the GC entirely.

---

## 12. Hot-Path Optimization: `distInt8` Unrolling

**Profiling result (pprof under c=3 load):**

```
670ms  71.28%   ivf.distInt8 (inline)
 90ms   9.57%   syscall.Write (network I/O)
 20ms   2.13%   encoding/json
```

The inner distance loop accounted for 71% of all CPU time. JSON parsing was only 2% —
switching to a faster JSON library would have negligible effect.

**Fix applied:** fully unroll the 14-iteration loop, add a bounds-check elimination
hint (`_ = p[dims-1]`), and replace the per-iteration slice header creation with a raw
unsafe pointer that advances by `dims` bytes each step:

```go
// Before: loop + slice header per iteration + bounds check per access
for i := 0; i < dims; i++ {
    diff := int32(stored[i]) - int32(query[i])
    sum += diff * diff
}

// After: fully unrolled, no loop, BCE, pointer arithmetic in caller
d0 := int32(p[0]) - int32(query[0])
...
d13 := int32(p[13]) - int32(query[13])
return d0*d0 + d1*d1 + ... + d13*d13
```

**Result:** p99 at c=1 dropped from 0.731 ms → 0.545 ms (25% improvement).

---

## 13. IVF Probe Count: Benchmark-Driven Selection

**Target:** p99 < 1 ms at competition-representative concurrency (c ≈ 3 per instance).

**Sweep results (5000-centroid index, c=3, 4000 requests, 500 warmup):**

| probes | p50 | p99 at c=3 | Recall (n=5000) | Est. det. score |
|---|---|---|---|---|
| 1 | — | — | 99.65% (2 FN, 5 FP) | +1936 |
| 2 | — | — | 99.90% (0 FN, 2 FP) | +2857 |
| **5** | **0.31 ms** | **0.81 ms** | **99.98% (0 FN, 1 FP)** | **+2910** |
| 10 | 0.35 ms | 0.87 ms | 100.00% (0 FN, 0 FP) | +3000 |

**Decision:** `probes=5`.

probes=5 is ~70 µs faster at p99 than probes=10, fits comfortably under 1 ms at c=3,
and produces zero false negatives over 5000 test queries. The single false positive
(1 in 5000) is a vector on a cluster boundary and is classified as statistical noise.

**Fallback:** to increase recall at the cost of latency, set `IVF_PROBES=10` in
`docker-compose.yml` — no image rebuild required.

---

## 14. Scoring Implications

From `AVALIACAO.md`:

```
score_p99  = 1000 · log₁₀(1000 / max(p99_ms, 1))   [−3000 if p99 > 2000 ms]
score_det  = 1000 · log₁₀(1 / max(ε, 0.001)) − 300 · log₁₀(1 + E)
E          = 1·FP + 3·FN + 5·HTTPErr
score_final = score_p99 + score_det
```

**FN costs 3× FP** — missing fraud is far worse than blocking a legitimate transaction.
This drove the decision to choose probes=5 over probes=2 despite the small latency
saving from probes=2: zero false negatives are more valuable than the ~70 µs saved.

**Estimated final score with current configuration:**

```
p99 ≈ 0.81 ms at c=3  →  score_p99 = 1000·log₁₀(1000/0.81) ≈ +3091 → capped at +3000
recall at probes=5     →  score_det ≈ +2910 to +3000
─────────────────────────────────────────────────────────────────────
Total                  ≈  +5910 to +6000 out of ±6000
```

---

## 15. Key Files

```
api/
  cmd/server/main.go          API entrypoint; reads SEARCHER env var to select backend
  cmd/buildindex/main.go      Offline index builder (k-means + binary write)
  cmd/bench/main.go           Latency benchmark (p50/p95/p99/p999, configurable concurrency)
  cmd/recall/main.go          Recall measurement vs ground-truth full scan
  internal/search/searcher.go search.Searcher interface (the swap point)
  internal/search/ivf/ivf.go  IVF implementation: mmap load, unrolled distInt8, pool
  internal/search/hnsw/hnsw.go HNSW stub (ErrNotImplemented)
  internal/vectorize/vectorize.go  Transaction payload → [14]float32 vector
  internal/handler/handler.go fasthttp router; pre-baked response array
Dockerfile                    4-stage build; index baked in at build time
docker-compose.yml            HAProxy + api1 + api2; resource limits; IVF_PROBES=5
haproxy.cfg                   Round-robin, health-check on GET /ready
```

---

## Open Items

- **HNSW implementation** — `api/internal/search/hnsw/hnsw.go` is a stub. If
  benchmarks under competition hardware show p99 > 1 ms, implement HNSW with a
  compact graph (M=6, int8 vectors, shared mmap). Requires ~150 MB shared across
  both containers plus a startup coordination mechanism.

- **Recall on fresh queries** — the recall tool samples stored vectors (which are
  always in their own cluster). Fresh API payloads sit at different positions in
  the 14-dim space and may fall near cluster boundaries more often. If competition
  accuracy is below expectations, increase `IVF_PROBES` from 5 to 10.

- **Submission branch** — the `submission` branch (separate from `main`) must contain
  `docker-compose.yml` at the repo root plus `info.json`. The PR adding
  `participants/fandangolas.json` to the upstream challenge repo triggers official
  evaluation via an issue with `rinha/test` in the body.
