/// buildindex — Rust port of api/cmd/buildindex/main.go
///
/// Reads references.json.gz (a single JSON array of 3M transaction records),
/// runs k-means to find `centroids` cluster centroids, assigns every vector,
/// quantizes float32 → int8, and writes the IVF binary index that
/// rust-api/src/search.rs mmap-reads at startup.
///
/// Binary format (little-endian, defined by search.rs):
///   [0:4]   magic "IVFX"
///   [4:8]   version u32 = 1
///   [8:16]  num_vecs u64
///   [16:20] num_centroids u32
///   [20:24] dims u32 = 14
///   [24:28] default_probes u32
///   [28:32] reserved u32 = 0
///   centroids:      [num_centroids * dims] f32  (little-endian)
///   cluster_offset: [num_centroids + 1]    u64  (vector indices, not byte offsets)
///   vectors:        [num_vecs * dims]       i8   (sorted by cluster id)
///   labels:         [num_vecs]              u8   (0=legit, 1=fraud)
use std::{
    fs::File,
    io::{BufWriter, Read, Write},
    time::Instant,
};

use flate2::read::GzDecoder;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::Deserialize;

const DIMS: usize = 14;
/// QUANT_SCALE must match the constant in search.rs (127.0).
/// The reader quantizes the query with: v = f * 127.0; clamp to [-127, 127]; truncate.
/// The writer applies the identical transform to stored vectors.
const QUANT_SCALE: f32 = 127.0;
const MAGIC: &[u8; 4] = b"IVFX";
const VERSION: u32 = 1;

fn main() {
    let args = parse_args();
    let t0 = Instant::now();

    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);

    // --- Pass 1: sample vectors, run k-means ---
    eprintln!(
        "pass 1: sampling {:.0}% of vectors for k-means...",
        args.sample_rate * 100.0
    );
    let sample = read_sample(&args.in_path, args.sample_rate, &mut rng)
        .unwrap_or_else(|e| panic!("failed to read sample: {e}"));
    eprintln!("  sampled {} vectors", sample.len());

    eprintln!(
        "running k-means (k={}, iters={})...",
        args.num_centroids, args.iters
    );
    let centroids = kmeans(&sample, args.num_centroids, args.iters, &mut rng);
    eprintln!("  k-means done ({:.1}s elapsed)", t0.elapsed().as_secs_f32());

    // --- Pass 2: assign all vectors to clusters ---
    eprintln!("pass 2: assigning all vectors to clusters...");
    let mut clusters: Vec<Vec<Entry>> = (0..args.num_centroids).map(|_| Vec::new()).collect();
    let total_vecs = assign_all(&args.in_path, &centroids, &mut clusters)
        .unwrap_or_else(|e| panic!("failed to assign vectors: {e}"));
    eprintln!("  total vectors: {total_vecs}");

    // --- Write binary index ---
    eprintln!("writing index to {}...", args.out_path);
    write_index(
        &args.out_path,
        &centroids,
        &clusters,
        total_vecs as u64,
        args.num_centroids as u32,
        args.default_probes as u32,
    )
    .unwrap_or_else(|e| panic!("failed to write index: {e}"));

    eprintln!(
        "done — index written to {} ({:.1}s total)",
        args.out_path,
        t0.elapsed().as_secs_f32()
    );
}

// ---------------------------------------------------------------------------
// CLI parsing
// ---------------------------------------------------------------------------

struct Args {
    in_path: String,
    out_path: String,
    num_centroids: usize,
    sample_rate: f64,
    iters: usize,
    default_probes: usize,
    seed: u64,
}

fn parse_args() -> Args {
    let raw: Vec<String> = std::env::args().collect();
    let mut in_path = "references.json.gz".to_string();
    let mut out_path = "index.ivf.bin".to_string();
    let mut num_centroids: usize = 1000;
    let mut sample_rate: f64 = 0.1;
    let mut iters: usize = 20;
    let mut default_probes: usize = 20;
    let mut seed: u64 = 42;

    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "-in" => {
                i += 1;
                in_path = raw[i].clone();
            }
            "-out" => {
                i += 1;
                out_path = raw[i].clone();
            }
            "-centroids" => {
                i += 1;
                num_centroids = raw[i].parse().expect("-centroids must be a positive integer");
            }
            "-sample" => {
                i += 1;
                sample_rate = raw[i].parse().expect("-sample must be a float in (0, 1]");
            }
            "-iters" => {
                i += 1;
                iters = raw[i].parse().expect("-iters must be a positive integer");
            }
            "-probes" => {
                i += 1;
                default_probes = raw[i].parse().expect("-probes must be a positive integer");
            }
            "-seed" => {
                i += 1;
                seed = raw[i].parse().expect("-seed must be a u64");
            }
            other => {
                eprintln!("unknown flag: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    Args {
        in_path,
        out_path,
        num_centroids,
        sample_rate,
        iters,
        default_probes,
        seed,
    }
}

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// A single reference vector quantized to int8 + its fraud label.
/// Fixed-size avoids per-entry heap allocation; ~15 bytes/entry.
struct Entry {
    vec: [i8; DIMS],
    is_fraud: bool,
}

#[derive(Deserialize)]
struct RefLine {
    vector: Vec<f32>,
    label: String,
}

// ---------------------------------------------------------------------------
// JSON array streaming
//
// references.json.gz contains a single compact JSON array:
//   [{"vector":[...],"label":"legit"},{"vector":[...],"label":"fraud"},...]
//
// `JsonArrayUnwrapper` is a `Read` adapter that strips the outer `[` / `]`
// and replaces top-level commas (i.e. commas between array elements, not
// commas inside objects/arrays) with spaces, so that `serde_json`'s
// `StreamDeserializer` (which skips only whitespace) can process the
// elements as a sequence of top-level values.
// ---------------------------------------------------------------------------

struct JsonArrayUnwrapper<R: Read> {
    inner: R,
    /// Have we consumed the opening `[`?
    started: bool,
    /// Nesting depth of `{` / `[` encountered so far. Top-level commas
    /// (depth == 0) are the array-element separators to replace with spaces.
    depth: i32,
    /// True once we have returned EOF.
    done: bool,
}

impl<R: Read> JsonArrayUnwrapper<R> {
    fn new(inner: R) -> Self {
        Self {
            inner,
            started: false,
            depth: 0,
            done: false,
        }
    }
}

impl<R: Read> Read for JsonArrayUnwrapper<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.done || buf.is_empty() {
            return Ok(0);
        }

        // Loop until we've produced at least one output byte (or reached true
        // EOF / end of array). Without this loop, returning 0 when we skipped
        // bytes (e.g. the opening `[`) would be misinterpreted as EOF by
        // callers that use the `Bytes` iterator (serde_json's IoRead).
        loop {
            let n = self.inner.read(buf)?;
            if n == 0 {
                self.done = true;
                return Ok(0);
            }

            let mut out_pos = 0usize;

            for i in 0..n {
                let b = buf[i];

                if !self.started {
                    // Skip bytes until we see the opening `[`.
                    if b == b'[' {
                        self.started = true;
                    }
                    continue;
                }

                // Once we've seen the opening `[`, track depth and rewrite
                // top-level commas → spaces and eat the closing `]`.
                match b {
                    b'{' | b'[' => {
                        self.depth += 1;
                        buf[out_pos] = b;
                        out_pos += 1;
                    }
                    b'}' | b']' if self.depth > 0 => {
                        self.depth -= 1;
                        buf[out_pos] = b;
                        out_pos += 1;
                    }
                    b']' => {
                        // depth == 0: closing bracket of the outer array.
                        self.done = true;
                        break;
                    }
                    b',' if self.depth == 0 => {
                        // Top-level comma: element separator → space.
                        buf[out_pos] = b' ';
                        out_pos += 1;
                    }
                    _ => {
                        buf[out_pos] = b;
                        out_pos += 1;
                    }
                }
            }

            if out_pos > 0 || self.done {
                return Ok(out_pos);
            }
            // out_pos == 0 and not done: all bytes in this chunk were skipped
            // (e.g. we just consumed the `[`). Read another chunk.
        }
    }
}

// ---------------------------------------------------------------------------
// I/O helpers
// ---------------------------------------------------------------------------

fn open_gz_array(path: &str) -> Result<JsonArrayUnwrapper<GzDecoder<File>>, std::io::Error> {
    let file = File::open(path)?;
    let gz = GzDecoder::new(file);
    Ok(JsonArrayUnwrapper::new(gz))
}

// ---------------------------------------------------------------------------
// Pass 1: sample
// ---------------------------------------------------------------------------

fn read_sample(
    path: &str,
    rate: f64,
    rng: &mut ChaCha8Rng,
) -> Result<Vec<[f32; DIMS]>, Box<dyn std::error::Error>> {
    let reader = open_gz_array(path)?;
    let mut sample = Vec::new();

    let stream = serde_json::Deserializer::from_reader(reader).into_iter::<RefLine>();
    for result in stream {
        let rec = result?;
        if rec.vector.len() < DIMS {
            continue;
        }
        if rng.random::<f64>() >= rate {
            continue;
        }
        let mut v = [0f32; DIMS];
        v.copy_from_slice(&rec.vector[..DIMS]);
        sample.push(v);
    }

    Ok(sample)
}

// ---------------------------------------------------------------------------
// K-means (Lloyd's algorithm)
// ---------------------------------------------------------------------------

fn kmeans(
    sample: &[[f32; DIMS]],
    k: usize,
    iters: usize,
    rng: &mut ChaCha8Rng,
) -> Vec<[f32; DIMS]> {
    assert!(
        sample.len() >= k,
        "sample size {} < k={k}; lower -sample or -centroids",
        sample.len()
    );

    // Initialise centroids by sampling k distinct vectors (Forgy init).
    let mut indices: Vec<usize> = (0..sample.len()).collect();
    indices.shuffle(rng);
    let mut centroids: Vec<[f32; DIMS]> = indices[..k].iter().map(|&i| sample[i]).collect();

    let mut assignments = vec![0usize; sample.len()];

    for iter in 0..iters {
        // Assign each sample to nearest centroid.
        for (vi, v) in sample.iter().enumerate() {
            assignments[vi] = nearest_centroid(v, &centroids);
        }

        // Recompute centroids as the mean of their assigned vectors.
        // Use f64 accumulators to avoid precision loss over many additions.
        let mut sums = vec![[0f64; DIMS]; k];
        let mut counts = vec![0usize; k];
        for (vi, v) in sample.iter().enumerate() {
            let ci = assignments[vi];
            counts[ci] += 1;
            for j in 0..DIMS {
                sums[ci][j] += v[j] as f64;
            }
        }
        for ci in 0..k {
            if counts[ci] > 0 {
                let n = counts[ci] as f64;
                for j in 0..DIMS {
                    centroids[ci][j] = (sums[ci][j] / n) as f32;
                }
            }
            // If a centroid has no assignments, leave it in place (rare edge
            // case at low sample rates; matches Go's behaviour).
        }

        eprintln!("  k-means iter {}/{iters}", iter + 1);
    }

    centroids
}

// ---------------------------------------------------------------------------
// Pass 2: assign all vectors
// ---------------------------------------------------------------------------

fn assign_all(
    path: &str,
    centroids: &[[f32; DIMS]],
    clusters: &mut Vec<Vec<Entry>>,
) -> Result<usize, Box<dyn std::error::Error>> {
    let reader = open_gz_array(path)?;
    let mut total = 0usize;

    let stream = serde_json::Deserializer::from_reader(reader).into_iter::<RefLine>();
    for result in stream {
        let rec = result?;
        if rec.vector.len() < DIMS {
            continue;
        }

        let mut v = [0f32; DIMS];
        v.copy_from_slice(&rec.vector[..DIMS]);

        let ci = nearest_centroid(&v, centroids);
        let entry = Entry {
            vec: quantize_vec(&v),
            is_fraud: rec.label == "fraud",
        };
        clusters[ci].push(entry);
        total += 1;

        if total % 500_000 == 0 {
            eprintln!("  assigned {total} vectors...");
        }
    }

    Ok(total)
}

// ---------------------------------------------------------------------------
// Index writer
// ---------------------------------------------------------------------------

fn write_index(
    path: &str,
    centroids: &[[f32; DIMS]],
    clusters: &[Vec<Entry>],
    num_vecs: u64,
    num_centroids: u32,
    default_probes: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(path)?;
    let mut w = BufWriter::with_capacity(1 << 20, file);

    // --- Header (32 bytes, little-endian) ---
    w.write_all(MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
    w.write_all(&num_vecs.to_le_bytes())?;
    w.write_all(&num_centroids.to_le_bytes())?;
    w.write_all(&(DIMS as u32).to_le_bytes())?;
    w.write_all(&default_probes.to_le_bytes())?;
    w.write_all(&0u32.to_le_bytes())?; // reserved

    // --- Centroids: [num_centroids * DIMS] f32 ---
    for c in centroids {
        for &f in c.iter() {
            w.write_all(&f.to_le_bytes())?;
        }
    }

    // --- Cluster offsets: [num_centroids + 1] u64 ---
    // offsets[i]..offsets[i+1] is the range of vectors in cluster i.
    // offsets[num_centroids] == num_vecs (sentinel).
    let mut offsets = vec![0u64; clusters.len() + 1];
    let mut cur: u64 = 0;
    for (i, cluster) in clusters.iter().enumerate() {
        offsets[i] = cur;
        cur += cluster.len() as u64;
    }
    offsets[clusters.len()] = cur;
    debug_assert_eq!(
        cur, num_vecs,
        "cluster vector count {cur} != declared num_vecs {num_vecs}"
    );
    for &off in &offsets {
        w.write_all(&off.to_le_bytes())?;
    }

    // --- Vectors: [num_vecs * DIMS] i8 (sorted by cluster id) ---
    for cluster in clusters {
        for entry in cluster {
            // i8 bytes are written as their unsigned bit-pattern (same as Go's byte(b)).
            for &b in &entry.vec {
                w.write_all(&[b as u8])?;
            }
        }
    }

    // --- Labels: [num_vecs] u8 (0=legit, 1=fraud) ---
    for cluster in clusters {
        for entry in cluster {
            w.write_all(&[entry.is_fraud as u8])?;
        }
    }

    w.flush()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Arithmetic helpers
// ---------------------------------------------------------------------------

/// Returns the index of the centroid nearest to `v` (squared Euclidean distance).
fn nearest_centroid(v: &[f32; DIMS], centroids: &[[f32; DIMS]]) -> usize {
    let mut best = 0;
    let mut best_dist = f32::MAX;
    for (i, c) in centroids.iter().enumerate() {
        let d = squared_euclidean(v, c);
        if d < best_dist {
            best_dist = d;
            best = i;
        }
    }
    best
}

#[inline(always)]
fn squared_euclidean(a: &[f32; DIMS], b: &[f32; DIMS]) -> f32 {
    let mut sum = 0f32;
    for i in 0..DIMS {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

/// Quantize a float32 vector to int8 using the same formula as search.rs
/// `quantize()`: `v = f * 127.0; clamp to [-127, 127]; truncate`.
///
/// Note: the Go writer used `math.Round` instead of truncation. We match the
/// Rust reader's formula exactly so query quantization and stored-vector
/// quantization use the same rounding mode.
fn quantize_vec(v: &[f32; DIMS]) -> [i8; DIMS] {
    let mut out = [0i8; DIMS];
    for i in 0..DIMS {
        out[i] = quantize_scalar(v[i]);
    }
    out
}

#[inline(always)]
fn quantize_scalar(f: f32) -> i8 {
    let v = f * QUANT_SCALE;
    if v > 127.0 {
        127
    } else if v < -127.0 {
        -127
    } else {
        v as i8
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- quantize_scalar ---

    #[test]
    fn quantize_zero_maps_to_zero() {
        assert_eq!(quantize_scalar(0.0), 0);
    }

    #[test]
    fn quantize_positive_one_clamps_to_127() {
        assert_eq!(quantize_scalar(1.0), 127);
    }

    #[test]
    fn quantize_negative_one_clamps_to_neg127() {
        assert_eq!(quantize_scalar(-1.0), -127);
    }

    #[test]
    fn quantize_overflow_positive_clamps() {
        assert_eq!(quantize_scalar(2.0), 127);
    }

    #[test]
    fn quantize_overflow_negative_clamps() {
        assert_eq!(quantize_scalar(-2.0), -127);
    }

    #[test]
    fn quantize_half_truncates_not_rounds() {
        // 0.5 * 127.0 = 63.5; truncation → 63, not 64
        assert_eq!(quantize_scalar(0.5), 63);
    }

    #[test]
    fn quantize_matches_search_rs_formula() {
        // Verify we match the exact formula from search.rs: v as i8 (truncation).
        for &f in &[-0.9f32, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9] {
            let v = f * 127.0;
            let expected = v as i8;
            assert_eq!(
                quantize_scalar(f),
                expected,
                "mismatch at f={f}: got {}, want {expected}",
                quantize_scalar(f)
            );
        }
    }

    // --- quantize_vec ---

    #[test]
    fn quantize_vec_all_zeros() {
        let v = [0f32; DIMS];
        assert_eq!(quantize_vec(&v), [0i8; DIMS]);
    }

    #[test]
    fn quantize_vec_element_wise() {
        let mut v = [0f32; DIMS];
        v[0] = 0.5;
        v[1] = -0.5;
        let q = quantize_vec(&v);
        assert_eq!(q[0], 63);
        assert_eq!(q[1], -63);
        for i in 2..DIMS {
            assert_eq!(q[i], 0);
        }
    }

    // --- squared_euclidean ---

    #[test]
    fn squared_euclidean_identical_vectors_is_zero() {
        let v = [0.1f32; DIMS];
        assert_eq!(squared_euclidean(&v, &v), 0.0);
    }

    #[test]
    fn squared_euclidean_unit_difference() {
        let a = [0f32; DIMS];
        let mut b = [0f32; DIMS];
        b[0] = 1.0;
        // Only first dimension differs by 1.0; sum = 1.0.
        assert!((squared_euclidean(&a, &b) - 1.0).abs() < 1e-6);
    }

    // --- nearest_centroid ---

    #[test]
    fn nearest_centroid_returns_closest() {
        let query = [0.0f32; DIMS];
        let mut far = [10.0f32; DIMS];
        far[0] = 10.0;
        let near = [0.1f32; DIMS];
        let centroids = vec![far, near];
        assert_eq!(nearest_centroid(&query, &centroids), 1);
    }

    // --- JsonArrayUnwrapper ---

    #[test]
    fn unwrapper_strips_outer_brackets_and_emits_objects() {
        let input = br#"[{"a":1},{"b":2}]"#;
        let mut unwrapper = JsonArrayUnwrapper::new(input.as_slice());
        let mut out = Vec::new();
        unwrapper.read_to_end(&mut out).unwrap();
        let s = std::str::from_utf8(&out).unwrap().trim().to_string();
        // Should contain both objects separated by a space, no outer brackets.
        assert!(s.contains(r#"{"a":1}"#), "got: {s}");
        assert!(s.contains(r#"{"b":2}"#), "got: {s}");
        assert!(!s.starts_with('['), "should not start with [, got: {s}");
    }

    #[test]
    fn unwrapper_preserves_commas_inside_arrays_in_objects() {
        // The "vector" field is an array — its commas must NOT be replaced.
        let input = br#"[{"vector":[1,2,3],"label":"ok"}]"#;
        let mut unwrapper = JsonArrayUnwrapper::new(input.as_slice());
        let mut out = Vec::new();
        unwrapper.read_to_end(&mut out).unwrap();
        let s = std::str::from_utf8(&out).unwrap();
        assert!(s.contains("[1,2,3]"), "nested array commas must be preserved, got: {s}");
    }

    #[test]
    fn unwrapper_single_element_array() {
        let input = br#"[{"x":42}]"#;
        let mut unwrapper = JsonArrayUnwrapper::new(input.as_slice());
        let mut out = Vec::new();
        unwrapper.read_to_end(&mut out).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(parsed["x"], 42);
    }

    #[test]
    fn unwrapper_multiple_elements_parseable_as_stream() {
        let input = br#"[{"vector":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.0,0.0,0.0,0.0],"label":"legit"},{"vector":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"label":"fraud"}]"#;
        let unwrapper = JsonArrayUnwrapper::new(input.as_slice());
        let records: Vec<RefLine> = serde_json::Deserializer::from_reader(unwrapper)
            .into_iter::<RefLine>()
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].label, "legit");
        assert_eq!(records[1].label, "fraud");
    }

    // --- write_index header contract ---

    #[test]
    fn write_index_header_magic_and_layout() {
        let centroids: Vec<[f32; DIMS]> = vec![[0.0f32; DIMS]];
        let entry = Entry {
            vec: [0i8; DIMS],
            is_fraud: false,
        };
        let clusters: Vec<Vec<Entry>> = vec![vec![entry]];

        let tmp = tempfile::NamedTempFile::new().expect("tempfile");
        let path = tmp.path().to_str().unwrap();

        write_index(path, &centroids, &clusters, 1, 1, 3).expect("write_index");

        let bytes = std::fs::read(path).expect("read back");

        // Magic
        assert_eq!(&bytes[0..4], b"IVFX", "wrong magic");
        // Version
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 1);
        // num_vecs
        assert_eq!(u64::from_le_bytes(bytes[8..16].try_into().unwrap()), 1);
        // num_centroids
        assert_eq!(u32::from_le_bytes(bytes[16..20].try_into().unwrap()), 1);
        // dims
        assert_eq!(
            u32::from_le_bytes(bytes[20..24].try_into().unwrap()),
            DIMS as u32
        );
        // default_probes
        assert_eq!(u32::from_le_bytes(bytes[24..28].try_into().unwrap()), 3);
        // reserved
        assert_eq!(u32::from_le_bytes(bytes[28..32].try_into().unwrap()), 0);
    }

    #[test]
    fn write_index_cluster_offsets_sentinel() {
        let centroids: Vec<[f32; DIMS]> = vec![[0.0f32; DIMS], [1.0f32; DIMS]];
        let clusters: Vec<Vec<Entry>> = vec![
            vec![Entry { vec: [0i8; DIMS], is_fraud: false }],
            vec![
                Entry { vec: [1i8; DIMS], is_fraud: true },
                Entry { vec: [2i8; DIMS], is_fraud: false },
            ],
        ];
        let num_vecs = 3u64;

        let tmp = tempfile::NamedTempFile::new().expect("tempfile");
        let path = tmp.path().to_str().unwrap();
        write_index(path, &centroids, &clusters, num_vecs, 2, 3).expect("write_index");

        let bytes = std::fs::read(path).expect("read back");

        // offset_off = 32 + num_centroids * DIMS * 4
        let offset_off = 32 + 2 * DIMS * 4;
        let off0 = u64::from_le_bytes(bytes[offset_off..offset_off + 8].try_into().unwrap());
        let off1 =
            u64::from_le_bytes(bytes[offset_off + 8..offset_off + 16].try_into().unwrap());
        let off2 =
            u64::from_le_bytes(bytes[offset_off + 16..offset_off + 24].try_into().unwrap());

        assert_eq!(off0, 0, "cluster 0 starts at 0");
        assert_eq!(off1, 1, "cluster 1 starts after cluster 0's 1 vector");
        assert_eq!(off2, 3, "sentinel == num_vecs");
    }

    #[test]
    fn write_index_vectors_sorted_by_cluster() {
        let centroids: Vec<[f32; DIMS]> = vec![[0.0f32; DIMS], [1.0f32; DIMS]];
        let clusters: Vec<Vec<Entry>> = vec![
            vec![Entry {
                vec: {
                    let mut a = [0i8; DIMS];
                    a[0] = 10;
                    a
                },
                is_fraud: false,
            }],
            vec![Entry {
                vec: {
                    let mut a = [0i8; DIMS];
                    a[0] = 20;
                    a
                },
                is_fraud: true,
            }],
        ];

        let tmp = tempfile::NamedTempFile::new().expect("tempfile");
        let path = tmp.path().to_str().unwrap();
        write_index(path, &centroids, &clusters, 2, 2, 3).expect("write_index");

        let bytes = std::fs::read(path).expect("read back");

        // vec_off = 32 + num_centroids * DIMS * 4 + (num_centroids + 1) * 8
        let vec_off = 32 + 2 * DIMS * 4 + 3 * 8;
        assert_eq!(bytes[vec_off] as i8, 10, "cluster-0 vector first byte");
        assert_eq!(bytes[vec_off + DIMS] as i8, 20, "cluster-1 vector first byte");

        let label_off = vec_off + 2 * DIMS;
        assert_eq!(bytes[label_off], 0, "cluster-0 label = legit");
        assert_eq!(bytes[label_off + 1], 1, "cluster-1 label = fraud");
    }

    #[test]
    fn kmeans_produces_k_centroids() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let sample: Vec<[f32; DIMS]> = (0..20)
            .map(|i| {
                let mut v = [0f32; DIMS];
                for j in 0..DIMS {
                    v[j] = (i as f32 + j as f32) / 100.0;
                }
                v
            })
            .collect();
        let centroids = kmeans(&sample, 3, 5, &mut rng);
        assert_eq!(centroids.len(), 3);
    }

    #[test]
    fn kmeans_is_deterministic() {
        let sample: Vec<[f32; DIMS]> = (0..50)
            .map(|i| {
                let mut v = [0f32; DIMS];
                for j in 0..DIMS {
                    v[j] = (i as f32 * 0.01) + (j as f32 * 0.001);
                }
                v
            })
            .collect();
        let c1 = kmeans(&sample, 5, 3, &mut ChaCha8Rng::seed_from_u64(42));
        let c2 = kmeans(&sample, 5, 3, &mut ChaCha8Rng::seed_from_u64(42));
        assert_eq!(c1, c2, "same seed must produce identical centroids");
    }
}
