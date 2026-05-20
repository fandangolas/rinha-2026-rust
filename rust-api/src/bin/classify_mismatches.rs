/// classify_mismatches — offline ground-truth classifier for IVF search errors.
///
/// For each mismatched transaction in test/diag-mismatches.jsonl:
///   1. Re-vectorize using the same logic as production (vectorize.rs).
///   2. Run a diagnostic IVF search (probes=3, k=5, int8) that also returns
///      which cluster indices were scanned and all candidate (vector-index, dist) pairs.
///   3. Run brute-force float32 5-NN over the full references.json.gz corpus.
///   4. Classify the error:
///        PROBE-MISS  — true 5-NN includes a vector not in the 3 scanned clusters.
///        TIE-FLIP    — true 5-NN is fully within scanned candidates but int8
///                      ranking swapped them (different fraud_count result).
///        VECTORIZE   — even with true float32 5-NN the expected answer is wrong
///                      (vectorizer or test-data label issue).
///        OTHER       — none of the above.
///   5. Write test/diag-classified.jsonl (one JSON line per mismatch).
///   6. Print a summary with histograms.
///
/// Running:
///   cd rust-api
///   cargo run --release --bin classify_mismatches -- \
///       --mismatches ../test/diag-mismatches.jsonl \
///       --index      ../data/index.ivf.bin \
///       --refs       ../data/references.json.gz \
///       --norm       ../data/normalization.json \
///       --mcc        ../data/mcc_risk.json \
///       --output     ../test/diag-classified.jsonl
///
/// Default paths assume running from the rust-api/ directory.

use std::{
    cmp::Ordering,
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Read, Write},
    slice,
    time::Instant,
};

use flate2::read::GzDecoder;
use serde::{Deserialize, Serialize};

// Pull in the production vectorize module unchanged.
// The binary is in src/bin/, one level below src/ where vectorize lives.
#[path = "../vectorize.rs"]
mod vectorize;

// ─── IVF index constants (must match search.rs exactly) ─────────────────────

const DIMS: usize = 14;
const MAGIC: &[u8; 4] = b"IVFX";
const QUANT_SCALE: f32 = 127.0;
const PROBES: usize = 3;
const K: usize = 5;

// ─── CLI ────────────────────────────────────────────────────────────────────

struct Args {
    mismatches_path: String,
    index_path: String,
    refs_path: String,
    norm_path: String,
    mcc_path: String,
    output_path: String,
}

fn parse_args() -> Args {
    let raw: Vec<String> = std::env::args().collect();
    let mut mismatches_path = "../test/diag-mismatches.jsonl".to_string();
    let mut index_path = "../data/index.ivf.bin".to_string();
    let mut refs_path = "../data/references.json.gz".to_string();
    let mut norm_path = "../data/normalization.json".to_string();
    let mut mcc_path = "../data/mcc_risk.json".to_string();
    let mut output_path = "../test/diag-classified.jsonl".to_string();

    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "--mismatches" => { i += 1; mismatches_path = raw[i].clone(); }
            "--index"      => { i += 1; index_path      = raw[i].clone(); }
            "--refs"       => { i += 1; refs_path        = raw[i].clone(); }
            "--norm"       => { i += 1; norm_path        = raw[i].clone(); }
            "--mcc"        => { i += 1; mcc_path         = raw[i].clone(); }
            "--output"     => { i += 1; output_path      = raw[i].clone(); }
            other => { eprintln!("unknown flag: {other}"); std::process::exit(1); }
        }
        i += 1;
    }

    Args { mismatches_path, index_path, refs_path, norm_path, mcc_path, output_path }
}

// ─── Mismatch record from JSONL ──────────────────────────────────────────────

#[derive(Deserialize)]
struct Mismatch {
    idx: usize,
    id: String,
    expected_approved: bool,
    actual_approved: bool,
    actual_fraud_score: f32,
    transaction: serde_json::Value,
}

// ─── IVF index (mmap-free version for diagnostic use) ───────────────────────

struct IvfIndex {
    num_cents: usize,
    num_vecs: usize,
    default_probes: usize,
    centroids: Vec<[f32; DIMS]>,         // [num_cents][DIMS]
    offsets: Vec<u64>,                   // [num_cents + 1]
    vecs: Vec<[i8; DIMS]>,               // [num_vecs][DIMS]  int8-quantized
    labels: Vec<bool>,                   // [num_vecs]  true=fraud
}

impl IvfIndex {
    fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;

        if bytes.len() < 32 {
            return Err("file too small for header".into());
        }
        if &bytes[0..4] != MAGIC {
            return Err(format!("bad magic: {:?}", &bytes[0..4]).into());
        }
        let version = u32::from_le_bytes(bytes[4..8].try_into()?);
        if version != 1 {
            return Err(format!("unsupported version {version}").into());
        }

        let num_vecs  = u64::from_le_bytes(bytes[8..16].try_into()?)  as usize;
        let num_cents = u32::from_le_bytes(bytes[16..20].try_into()?) as usize;
        let file_dims = u32::from_le_bytes(bytes[20..24].try_into()?) as usize;
        if file_dims != DIMS {
            return Err(format!("expected {DIMS} dims, got {file_dims}").into());
        }
        let default_probes = u32::from_le_bytes(bytes[24..28].try_into()?) as usize;

        let cent_off   = 32;
        let offset_off = cent_off + num_cents * DIMS * 4;
        let vec_off    = offset_off + (num_cents + 1) * 8;
        let label_off  = vec_off + num_vecs * DIMS;

        if bytes.len() < label_off + num_vecs {
            return Err(format!(
                "file too small: need {}, got {}",
                label_off + num_vecs, bytes.len()
            ).into());
        }

        // Decode centroids (f32 little-endian).
        let cents_f32: &[f32] = unsafe {
            slice::from_raw_parts(
                bytes[cent_off..].as_ptr() as *const f32,
                num_cents * DIMS,
            )
        };
        let mut centroids = vec![[0f32; DIMS]; num_cents];
        for ci in 0..num_cents {
            centroids[ci].copy_from_slice(&cents_f32[ci * DIMS..(ci + 1) * DIMS]);
        }

        // Decode offsets (u64 little-endian).
        let off_raw: &[u64] = unsafe {
            slice::from_raw_parts(
                bytes[offset_off..].as_ptr() as *const u64,
                num_cents + 1,
            )
        };
        let offsets: Vec<u64> = off_raw.to_vec();

        // Decode int8 vectors.
        let vecs_raw: &[i8] = unsafe {
            slice::from_raw_parts(
                bytes[vec_off..].as_ptr() as *const i8,
                num_vecs * DIMS,
            )
        };
        let mut vecs = vec![[0i8; DIMS]; num_vecs];
        for vi in 0..num_vecs {
            vecs[vi].copy_from_slice(&vecs_raw[vi * DIMS..(vi + 1) * DIMS]);
        }

        // Decode labels (u8, 0=legit, 1=fraud).
        let labels: Vec<bool> = bytes[label_off..label_off + num_vecs]
            .iter()
            .map(|&b| b == 1)
            .collect();

        Ok(IvfIndex {
            num_cents,
            num_vecs,
            default_probes,
            centroids,
            offsets,
            vecs,
            labels,
        })
    }
}

// ─── Diagnostic search result ────────────────────────────────────────────────

struct DiagResult {
    /// Cluster indices selected for probing (in centroid-distance order).
    probed_cluster_indices: Vec<usize>,
    /// All vector indices scanned across the probed clusters.
    scanned_vector_indices: Vec<usize>,
    /// The k=5 returned neighbors: (global_vec_index, int8_dist, is_fraud).
    returned_neighbors: Vec<(usize, i32, bool)>,
    /// fraud_count from the IVF search.
    ivf_fraud_count: usize,
}

// ─── Diagnostic search (replicates search.rs exactly, adds instrumentation) ──

fn diagnostic_search(index: &IvfIndex, query: &[f32; DIMS]) -> DiagResult {
    let probes = PROBES.min(index.num_cents);

    // Step 1: centroid distances.
    let mut cent_dists: Vec<(usize, f32)> = (0..index.num_cents)
        .map(|ci| {
            let c = &index.centroids[ci];
            let d: f32 = (0..DIMS).map(|j| {
                let diff = query[j] - c[j];
                diff * diff
            }).sum();
            (ci, d)
        })
        .collect();

    // Partial sort: bring the `probes` nearest to the front (same as search.rs).
    if probes < cent_dists.len() {
        cent_dists.select_nth_unstable_by(probes - 1, |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal)
        });
    }
    cent_dists[..probes].sort_unstable_by(|a, b| {
        a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal)
    });

    let probed_cluster_indices: Vec<usize> = cent_dists[..probes].iter().map(|(ci, _)| *ci).collect();

    // Step 2: quantize query.
    let q_int8: [i8; DIMS] = std::array::from_fn(|i| quantize(query[i]));

    // Step 3: scan probed clusters, maintain max-heap of k best.
    let mut cands: Vec<(usize, i32, bool)> = Vec::with_capacity(K + 1); // (vec_idx, dist, is_fraud)
    let mut scanned_vector_indices: Vec<usize> = Vec::new();

    for &ci in &probed_cluster_indices {
        let start = index.offsets[ci] as usize;
        let end   = index.offsets[ci + 1] as usize;

        for vi in start..end {
            scanned_vector_indices.push(vi);
            let d = dist_int8(&index.vecs[vi], &q_int8);
            let is_fraud = index.labels[vi];

            if cands.len() < K {
                cands.push((vi, d, is_fraud));
                if cands.len() == K {
                    heapify_max_cands(&mut cands);
                }
            } else if d < cands[0].1 {
                cands[0] = (vi, d, is_fraud);
                sift_down_max_cands(&mut cands, 0);
            }
        }
    }

    let ivf_fraud_count = cands.iter().filter(|(_, _, fraud)| *fraud).count();

    DiagResult {
        probed_cluster_indices,
        scanned_vector_indices,
        returned_neighbors: cands,
        ivf_fraud_count,
    }
}

// ─── Brute-force float32 5-NN ────────────────────────────────────────────────

/// Returns the true 5-NN as (global_vec_index, f32_dist, is_fraud).
/// Streams references.json.gz once for each query batch, but since we only
/// have ~136 queries we download + buffer all reference vectors in RAM (~200 MB).
fn brute_force_5nn(
    refs: &[([f32; DIMS], bool)],
    query: &[f32; DIMS],
) -> Vec<(usize, f32, bool)> {
    // Max-heap of size 5: (negative_dist, global_idx, is_fraud)
    // We use a simple partial-sort at the end for clarity over speed.
    let mut best: Vec<(usize, f32, bool)> = Vec::with_capacity(K + 1);

    for (idx, (vec, is_fraud)) in refs.iter().enumerate() {
        let d: f32 = (0..DIMS).map(|j| {
            let diff = query[j] - vec[j];
            diff * diff
        }).sum();

        if best.len() < K {
            best.push((idx, d, *is_fraud));
            if best.len() == K {
                best.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            }
        } else if d < best[0].1 {
            best[0] = (idx, d, *is_fraud);
            // Re-sort (small vec; fine for diagnostic use).
            best.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        }
    }

    // Return sorted nearest-first.
    best.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    best
}

// ─── Reference corpus loader ─────────────────────────────────────────────────

#[derive(Deserialize)]
struct RefLine {
    vector: Vec<f32>,
    label: String,
}

struct JsonArrayUnwrapper<R: Read> {
    inner: R,
    started: bool,
    depth: i32,
    done: bool,
}

impl<R: Read> JsonArrayUnwrapper<R> {
    fn new(inner: R) -> Self {
        Self { inner, started: false, depth: 0, done: false }
    }
}

impl<R: Read> Read for JsonArrayUnwrapper<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.done || buf.is_empty() { return Ok(0); }
        loop {
            let n = self.inner.read(buf)?;
            if n == 0 { self.done = true; return Ok(0); }
            let mut out_pos = 0usize;
            for i in 0..n {
                let b = buf[i];
                if !self.started {
                    if b == b'[' { self.started = true; }
                    continue;
                }
                match b {
                    b'{' | b'[' => { self.depth += 1; buf[out_pos] = b; out_pos += 1; }
                    b'}' | b']' if self.depth > 0 => { self.depth -= 1; buf[out_pos] = b; out_pos += 1; }
                    b']' => { self.done = true; break; }
                    b',' if self.depth == 0 => { buf[out_pos] = b' '; out_pos += 1; }
                    _ => { buf[out_pos] = b; out_pos += 1; }
                }
            }
            if out_pos > 0 || self.done { return Ok(out_pos); }
        }
    }
}

fn load_references(path: &str) -> Result<Vec<([f32; DIMS], bool)>, Box<dyn std::error::Error>> {
    eprintln!("loading references from {path} ...");
    let t = Instant::now();

    let file = File::open(path)?;
    let gz   = GzDecoder::new(file);
    let reader = JsonArrayUnwrapper::new(gz);

    let mut refs: Vec<([f32; DIMS], bool)> = Vec::with_capacity(3_000_000);
    let stream = serde_json::Deserializer::from_reader(reader).into_iter::<RefLine>();
    for result in stream {
        let rec = result?;
        if rec.vector.len() < DIMS { continue; }
        let mut v = [0f32; DIMS];
        v.copy_from_slice(&rec.vector[..DIMS]);
        refs.push((v, rec.label == "fraud"));
        if refs.len() % 500_000 == 0 {
            eprintln!("  loaded {}k references ({:.1}s)", refs.len() / 1000, t.elapsed().as_secs_f32());
        }
    }

    eprintln!("  loaded {} references total in {:.1}s", refs.len(), t.elapsed().as_secs_f32());
    Ok(refs)
}

// ─── Classification logic ─────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Classification {
    TieFlip,
    ProbeMiss,
    Vectorize,
    Other,
}

/// For each probe-miss case, compute the rank of the missed true-neighbor's
/// cluster in centroid-distance order (1-based: rank 1 = the nearest cluster
/// that was NOT scanned). We need all 1000 centroid distances to compute this.
fn cluster_rank_of_vec_in_centroid_order(
    index: &IvfIndex,
    query: &[f32; DIMS],
    true_vec_idx: usize,
) -> usize {
    // Find which cluster this vector belongs to by scanning offsets.
    let cluster_of_vec = find_cluster(index, true_vec_idx);

    // Compute distances from query to all centroids, sort, return 1-based rank.
    let mut cent_dists: Vec<(usize, f32)> = (0..index.num_cents)
        .map(|ci| {
            let c = &index.centroids[ci];
            let d: f32 = (0..DIMS).map(|j| { let d = query[j] - c[j]; d * d }).sum();
            (ci, d)
        })
        .collect();
    cent_dists.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    cent_dists.iter().position(|(ci, _)| *ci == cluster_of_vec).unwrap_or(999) + 1
}

fn find_cluster(index: &IvfIndex, vec_idx: usize) -> usize {
    for ci in 0..index.num_cents {
        let start = index.offsets[ci] as usize;
        let end   = index.offsets[ci + 1] as usize;
        if vec_idx >= start && vec_idx < end {
            return ci;
        }
    }
    panic!("vec_idx {vec_idx} not found in any cluster (num_vecs={})", index.num_vecs);
}

// ─── Output record ────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct ClassifiedRecord {
    idx: usize,
    id: String,
    classification: Classification,
    expected_approved: bool,
    actual_approved_ivf: bool,
    query_vec: [f32; DIMS],
    true_5nn: Vec<TrueNeighbor>,
    ivf_5nn: Vec<IvfNeighbor>,
    fraud_count_true: usize,
    fraud_count_ivf: usize,
    cluster_indices_scanned: Vec<usize>,
    /// For each true-NN vector, its rank in centroid-distance order (1-based).
    /// Rank ≤ PROBES means the cluster was probed; rank > PROBES is a miss.
    ranks_of_true_in_scanned: Vec<usize>,
    /// For probe-miss: rank of the nearest un-probed true-neighbor cluster.
    probe_miss_cluster_rank: Option<usize>,
    /// For tie-flip: int8 distance of the displaced true-5th vs the int8-ranked 5th.
    tie_flip_int8_gap: Option<i32>,
}

#[derive(Serialize)]
struct TrueNeighbor {
    vec_idx: usize,
    float32_dist: f32,
    is_fraud: bool,
    cluster_idx: usize,
    cluster_rank_in_cent_order: usize,
}

#[derive(Serialize)]
struct IvfNeighbor {
    vec_idx: usize,
    int8_dist: i32,
    is_fraud: bool,
}

// ─── Arithmetic helpers (replicate search.rs exactly) ────────────────────────

#[inline(always)]
fn quantize(f: f32) -> i8 {
    let v = f * QUANT_SCALE;
    if v > 127.0 { 127 } else if v < -127.0 { -127 } else { v as i8 }
}

#[inline(always)]
fn dist_int8(stored: &[i8; DIMS], query: &[i8; DIMS]) -> i32 {
    let mut sum = 0i32;
    for i in 0..DIMS {
        let d = stored[i] as i32 - query[i] as i32;
        sum += d * d;
    }
    sum
}

fn heapify_max_cands(h: &mut [(usize, i32, bool)]) {
    let n = h.len();
    for i in (0..n / 2).rev() { sift_down_max_cands(h, i); }
}

fn sift_down_max_cands(h: &mut [(usize, i32, bool)], mut i: usize) {
    let n = h.len();
    loop {
        let mut largest = i;
        let l = 2 * i + 1;
        let r = 2 * i + 2;
        if l < n && h[l].1 > h[largest].1 { largest = l; }
        if r < n && h[r].1 > h[largest].1 { largest = r; }
        if largest == i { break; }
        h.swap(i, largest);
        i = largest;
    }
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();

    // ── Load normalization and MCC risk ──────────────────────────────────────
    let norm_str = std::fs::read_to_string(&args.norm_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", args.norm_path));
    let norm: vectorize::Normalization = serde_json::from_str(&norm_str)
        .unwrap_or_else(|e| panic!("parse normalization: {e}"));

    let mcc_str = std::fs::read_to_string(&args.mcc_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", args.mcc_path));
    let mcc_risk: vectorize::MccRisk = serde_json::from_str(&mcc_str)
        .unwrap_or_else(|e| panic!("parse mcc_risk: {e}"));

    // ── Load IVF index ───────────────────────────────────────────────────────
    eprintln!("loading IVF index from {} ...", args.index_path);
    let index = IvfIndex::load(&args.index_path)
        .unwrap_or_else(|e| panic!("load index: {e}"));
    eprintln!(
        "  index: {} vecs, {} centroids, default_probes={}",
        index.num_vecs, index.num_cents, index.default_probes
    );

    // ── Load reference corpus ────────────────────────────────────────────────
    let refs = load_references(&args.refs_path)
        .unwrap_or_else(|e| panic!("load references: {e}"));

    // ── Load mismatch records ────────────────────────────────────────────────
    let mismatches_file = File::open(&args.mismatches_path)
        .unwrap_or_else(|e| panic!("open {}: {e}", args.mismatches_path));
    let reader = BufReader::new(mismatches_file);
    let mut mismatches: Vec<Mismatch> = Vec::new();
    for (line_no, line) in reader.lines().enumerate() {
        let line = line.unwrap();
        let line = line.trim();
        if line.is_empty() { continue; }
        match serde_json::from_str::<Mismatch>(line) {
            Ok(m) => mismatches.push(m),
            Err(e) => eprintln!("  line {line_no}: parse error: {e}"),
        }
    }
    eprintln!("loaded {} mismatch records from {}", mismatches.len(), args.mismatches_path);

    // ── Classify each mismatch ───────────────────────────────────────────────
    let output_file = File::create(&args.output_path)
        .unwrap_or_else(|e| panic!("create {}: {e}", args.output_path));
    let mut writer = BufWriter::new(output_file);

    // Accumulate stats for final histograms.
    let mut count_tie_flip:  usize = 0;
    let mut count_probe_miss: usize = 0;
    let mut count_vectorize:  usize = 0;
    let mut count_other:      usize = 0;

    // probe-miss histogram: cluster rank → count
    let mut probe_miss_rank_hist: HashMap<usize, usize> = HashMap::new();
    // tie-flip histogram: int8 gap → count
    let mut tie_flip_gap_hist: HashMap<i32, usize> = HashMap::new();

    let total = mismatches.len();
    for (i, mismatch) in mismatches.iter().enumerate() {
        let t = Instant::now();

        // 1. Re-vectorize using production logic.
        let req: vectorize::Request = serde_json::from_value(mismatch.transaction.clone())
            .unwrap_or_else(|e| panic!("idx {}: deserialize request: {e}", mismatch.idx));
        let query_vec = vectorize::vectorize(&req, &norm, &mcc_risk);

        // 2. Diagnostic IVF search.
        let diag = diagnostic_search(&index, &query_vec);

        // 3. Brute-force float32 5-NN.
        let t_bf = Instant::now();
        let true_nn = brute_force_5nn(&refs, &query_vec);
        let bf_elapsed = t_bf.elapsed();

        let fraud_count_true = true_nn.iter().filter(|(_, _, f)| *f).count();
        let true_approved = fraud_count_true < 3;

        // 4. Classify.
        // Compute set membership: which true-NN vectors are in the scanned set?
        let scanned_set: std::collections::HashSet<usize> =
            diag.scanned_vector_indices.iter().copied().collect();

        let true_nn_idxs: Vec<usize> = true_nn.iter().map(|(vi, _, _)| *vi).collect();
        let true_nn_all_scanned = true_nn_idxs.iter().all(|vi| scanned_set.contains(vi));

        // Compute cluster rank for each true neighbor.
        let ranks_of_true: Vec<usize> = true_nn_idxs.iter().map(|&vi| {
            cluster_rank_of_vec_in_centroid_order(&index, &query_vec, vi)
        }).collect();

        let classification;
        let probe_miss_cluster_rank;
        let tie_flip_int8_gap;

        if !true_approved == mismatch.expected_approved {
            // Even with true float32 5-NN the answer matches the unexpected
            // (i.e. true 5-NN says the opposite of expected_approved).
            // This is a VECTORIZE or label issue.
            classification = Classification::Vectorize;
            probe_miss_cluster_rank = None;
            tie_flip_int8_gap = None;
            count_vectorize += 1;
        } else if !true_nn_all_scanned {
            // At least one true neighbor is outside all scanned clusters.
            classification = Classification::ProbeMiss;

            // Find the missed cluster with the best (lowest) rank.
            let missed_rank = true_nn_idxs.iter()
                .zip(ranks_of_true.iter())
                .filter(|(&vi, _)| !scanned_set.contains(&vi))
                .map(|(_, &rank)| rank)
                .min()
                .unwrap_or(999);
            probe_miss_cluster_rank = Some(missed_rank);
            tie_flip_int8_gap = None;
            count_probe_miss += 1;
            *probe_miss_rank_hist.entry(missed_rank).or_insert(0) += 1;
        } else if diag.ivf_fraud_count != fraud_count_true {
            // All true neighbors were scanned but int8 ranking changed fraud_count.
            classification = Classification::TieFlip;
            probe_miss_cluster_rank = None;

            // Compute the int8 distances for the true 5-NN.
            let q_int8: [i8; DIMS] = std::array::from_fn(|j| quantize(query_vec[j]));
            let true_5th_int8_dist = true_nn_idxs.get(4).map(|&vi| {
                dist_int8(&index.vecs[vi], &q_int8)
            });
            // The actual 5th placed by IVF.
            let ivf_5th_dist = diag.returned_neighbors.iter()
                .map(|(_, d, _)| *d)
                .min();
            let gap = match (true_5th_int8_dist, ivf_5th_dist) {
                (Some(a), Some(b)) => (a - b).abs(),
                _ => 0,
            };
            tie_flip_int8_gap = Some(gap);
            count_tie_flip += 1;
            *tie_flip_gap_hist.entry(gap).or_insert(0) += 1;
        } else {
            classification = Classification::Other;
            probe_miss_cluster_rank = None;
            tie_flip_int8_gap = None;
            count_other += 1;
        }

        // Build true-NN detail records.
        let true_5nn: Vec<TrueNeighbor> = true_nn.iter()
            .zip(ranks_of_true.iter())
            .map(|((vi, dist, is_fraud), &rank)| TrueNeighbor {
                vec_idx: *vi,
                float32_dist: *dist,
                is_fraud: *is_fraud,
                cluster_idx: find_cluster(&index, *vi),
                cluster_rank_in_cent_order: rank,
            })
            .collect();

        let ivf_5nn: Vec<IvfNeighbor> = diag.returned_neighbors.iter()
            .map(|(vi, d, fraud)| IvfNeighbor {
                vec_idx: *vi,
                int8_dist: *d,
                is_fraud: *fraud,
            })
            .collect();

        let record = ClassifiedRecord {
            idx: mismatch.idx,
            id: mismatch.id.clone(),
            classification: classification.clone(),
            expected_approved: mismatch.expected_approved,
            actual_approved_ivf: mismatch.actual_approved,
            query_vec,
            true_5nn,
            ivf_5nn,
            fraud_count_true,
            fraud_count_ivf: diag.ivf_fraud_count,
            cluster_indices_scanned: diag.probed_cluster_indices,
            ranks_of_true_in_scanned: ranks_of_true,
            probe_miss_cluster_rank,
            tie_flip_int8_gap,
        };

        let line = serde_json::to_string(&record)
            .unwrap_or_else(|e| panic!("serialize record: {e}"));
        writeln!(writer, "{line}")
            .unwrap_or_else(|e| panic!("write output: {e}"));

        eprintln!(
            "  [{}/{total}] idx={} class={:?} brute_force_time={:.1}s",
            i + 1,
            mismatch.idx,
            classification,
            bf_elapsed.as_secs_f32()
        );
    }

    writer.flush().unwrap();
    eprintln!("wrote classified records to {}", args.output_path);

    // ── Summary ──────────────────────────────────────────────────────────────
    println!();
    println!("════════════════════════════════════════════════════════════");
    println!(" MISMATCH CLASSIFICATION SUMMARY");
    println!("════════════════════════════════════════════════════════════");
    println!("  Total mismatches classified : {total}");
    println!("  TIE-FLIP                    : {count_tie_flip}  ({:.1}%)", 100.0 * count_tie_flip as f64 / total as f64);
    println!("  PROBE-MISS                  : {count_probe_miss}  ({:.1}%)", 100.0 * count_probe_miss as f64 / total as f64);
    println!("  VECTORIZE / label           : {count_vectorize}  ({:.1}%)", 100.0 * count_vectorize as f64 / total as f64);
    println!("  OTHER                       : {count_other}  ({:.1}%)", 100.0 * count_other as f64 / total as f64);
    println!();

    if !probe_miss_rank_hist.is_empty() {
        println!("  PROBE-MISS — cluster rank histogram");
        println!("  (rank = position of true neighbor's cluster in centroid-distance");
        println!("   order; ranks 1-{PROBES} are probed, ranks >{PROBES} are missed)");
        let mut ranks: Vec<usize> = probe_miss_rank_hist.keys().copied().collect();
        ranks.sort_unstable();
        for rank in ranks {
            let n = probe_miss_rank_hist[&rank];
            println!("    rank {:3} : {:3} mismatches  {}", rank, n, "█".repeat(n));
        }
        println!();
    }

    if !tie_flip_gap_hist.is_empty() {
        println!("  TIE-FLIP — int8 distance gap histogram");
        println!("  (gap = |int8_dist(true_5th) - int8_dist(ivf_5th)|)");
        let mut gaps: Vec<i32> = tie_flip_gap_hist.keys().copied().collect();
        gaps.sort_unstable();
        for gap in gaps {
            let n = tie_flip_gap_hist[&gap];
            println!("    gap {:4} : {:3} mismatches  {}", gap, n, "█".repeat(n));
        }
        println!();
    }

    println!("════════════════════════════════════════════════════════════");
    println!("  ADC re-rank recommendation:");
    if count_probe_miss > count_tie_flip {
        println!("  Dominant error = PROBE-MISS → ADC re-rank alone is insufficient.");
        println!("  Raising probes or using a larger centroid count will help more.");
    } else if count_tie_flip >= count_probe_miss {
        println!("  Dominant error = TIE-FLIP → ADC re-rank (float32 re-scoring of");
        println!("  scanned candidates) should fix most of these errors.");
    }
    println!("════════════════════════════════════════════════════════════");
}
