/// Per-phase timing histograms, gated behind the METRICS env-var.
///
/// When ENABLED=false (the default / production path):
///   - handler calls `m.then(Instant::now)` which is `None` — Instant::now()
///     is never called, zero extra instructions after branch prediction.
///
/// When ENABLED=true:
///   - each phase: one Instant::now() (~20 ns vDSO) + three Relaxed atomic adds
///   - no allocation, no mutex, no contention between Tokio tasks
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};

// Set once in main() from the METRICS env-var, read on every request.
pub static ENABLED: AtomicBool = AtomicBool::new(false);

// ── Histogram ─────────────────────────────────────────────────────────────
// Bucket upper bounds in µs.  Last bucket is open-ended (≥5000 µs = ≥5 ms).
const BOUNDS: [u64; 7] = [10, 50, 100, 250, 500, 1_000, 5_000];
const LABELS: [&str; 8] = ["<10µs", "<50µs", "<100µs", "<250µs", "<500µs", "<1ms", "<5ms", "≥5ms"];
const N: usize = 8;

pub struct PhaseStats {
    name:    &'static str,
    sum_us:  AtomicU64,
    count:   AtomicU64,
    buckets: [AtomicU64; N],
}

impl PhaseStats {
    pub const fn new(name: &'static str) -> Self {
        Self {
            name,
            sum_us:  AtomicU64::new(0),
            count:   AtomicU64::new(0),
            buckets: [
                AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
                AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
                AtomicU64::new(0), AtomicU64::new(0),
            ],
        }
    }

    #[inline]
    pub fn record(&self, us: u64) {
        self.sum_us.fetch_add(us, Relaxed);
        self.count.fetch_add(1, Relaxed);
        let idx = BOUNDS.iter().position(|&b| us < b).unwrap_or(N - 1);
        self.buckets[idx].fetch_add(1, Relaxed);
    }

    fn report(&self) -> String {
        let n = self.count.load(Relaxed);
        if n == 0 {
            return format!("{:<12}  (no data)\n", self.name);
        }

        let sum = self.sum_us.load(Relaxed);
        let avg = sum / n;

        // Approximate p50, p99 from bucket cumulative counts.
        let pct = |pct: f64| -> &'static str {
            let target = (n as f64 * pct) as u64;
            let mut cum = 0u64;
            for i in 0..N {
                cum += self.buckets[i].load(Relaxed);
                if cum >= target { return LABELS[i]; }
            }
            LABELS[N - 1]
        };

        let mut s = format!(
            "{:<12}  n={:<7}  avg={:<6}µs  p50≈{:<8}  p99≈{:<8}  | ",
            self.name, n, avg, pct(0.50), pct(0.99)
        );

        for i in 0..N {
            let c = self.buckets[i].load(Relaxed);
            if c > 0 {
                s += &format!("{}:{} ", LABELS[i], c);
            }
        }
        s += "\n";
        s
    }
}

// ── Global phase stats ────────────────────────────────────────────────────
pub static PARSE:     PhaseStats = PhaseStats::new("parse");
pub static VECTORIZE: PhaseStats = PhaseStats::new("vectorize");
pub static SEARCH:    PhaseStats = PhaseStats::new("search");
pub static TOTAL:     PhaseStats = PhaseStats::new("total");

// ── /metrics response ─────────────────────────────────────────────────────
pub fn report_all() -> String {
    if !ENABLED.load(Relaxed) {
        return "metrics collection is disabled.\nSet METRICS=true and restart to enable.\n".to_string();
    }

    let mut out = String::with_capacity(1024);
    out += "# ── per-request phase breakdown (µs) ──────────────────────────────\n";
    out += "# parse      = serde_json::from_slice  (JSON → Request struct)\n";
    out += "# vectorize  = feature extraction      (Request → [f32; 14])\n";
    out += "# search     = IVF k-NN search         (centroid scan + cluster scan)\n";
    out += "# total      = full handler wall time  (parse + vectorize + search + overhead)\n";
    out += "#\n";
    out += &PARSE.report();
    out += &VECTORIZE.report();
    out += &SEARCH.report();
    out += &TOTAL.report();
    out += "\n";
    out += &cfs_stats();
    out
}

// ── CFS throttle counters ─────────────────────────────────────────────────
// Reads /sys/fs/cgroup/cpu.stat (cgroup v2, default on modern Linux/Docker).
// Key fields:
//   nr_throttled   — number of periods the container was throttled
//   throttled_usec — total µs spent throttled (= stall time)
//
// If nr_throttled is large → CFS is starving the container under load.
fn cfs_stats() -> String {
    // cgroup v2 (modern Docker / systemd)
    let paths = [
        "/sys/fs/cgroup/cpu.stat",
        "/sys/fs/cgroup/cpu,cpuacct/cpu.stat", // cgroup v1 fallback
    ];

    for path in &paths {
        if let Ok(content) = std::fs::read_to_string(path) {
            let mut out = format!("# ── CFS throttle counters ({path}) ─────────────────────────────\n");
            for line in content.lines() {
                // Surface the most diagnostic fields prominently.
                let tag = if line.starts_with("nr_throttled")
                    || line.starts_with("throttled_usec")
                    || line.starts_with("nr_periods")
                    || line.starts_with("nr_burst")
                    || line.starts_with("burst_usec")
                {
                    "*** "
                } else {
                    "    "
                };
                out += &format!("{tag}{line}\n");
            }
            out += "# *** = fields most relevant to CFS stall diagnosis\n";
            return out;
        }
    }

    "# CFS stats: /sys/fs/cgroup/cpu.stat not found (not Linux, or unusual cgroup mount)\n".to_string()
}
