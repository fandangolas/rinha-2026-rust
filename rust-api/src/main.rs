mod search;
mod timing;
mod vectorize;

use axum::{
    body::Bytes,
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use std::{env, sync::Arc, time::Instant};
use std::sync::atomic::Ordering::Relaxed;

// Pre-computed response bodies: fraud_score = fraud_count/5, threshold < 0.6.
// Identical to the Go handler — no allocation or formatting in the hot path.
static RESPONSES: [&[u8]; 6] = [
    b"{\"approved\":true,\"fraud_score\":0.0}",
    b"{\"approved\":true,\"fraud_score\":0.2}",
    b"{\"approved\":true,\"fraud_score\":0.4}",
    b"{\"approved\":false,\"fraud_score\":0.6}",
    b"{\"approved\":false,\"fraud_score\":0.8}",
    b"{\"approved\":false,\"fraud_score\":1.0}",
];

struct AppState {
    searcher: search::Searcher,
    norm: vectorize::Normalization,
    mcc_risk: vectorize::MccRisk,
}

async fn ready() -> StatusCode {
    StatusCode::OK
}

async fn metrics() -> String {
    timing::report_all()
}

async fn fraud_score(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    // Single atomic load; branch predictor caches it — ~0 cost when false.
    let m = timing::ENABLED.load(Relaxed);

    // bool::then(f) calls f() only when m=true, returning Option<Instant>.
    // When m=false, Instant::now() is never invoked.
    let t_total = m.then(Instant::now);

    // ── 1. JSON parse ────────────────────────────────────────────────────
    let t = m.then(Instant::now);
    let req: vectorize::Request = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
    };
    if let Some(t) = t { timing::PARSE.record(t.elapsed().as_micros() as u64); }

    // ── 2. Vectorize ─────────────────────────────────────────────────────
    let t = m.then(Instant::now);
    let query = vectorize::vectorize(&req, &state.norm, &state.mcc_risk);
    if let Some(t) = t { timing::VECTORIZE.record(t.elapsed().as_micros() as u64); }

    // ── 3. IVF search ────────────────────────────────────────────────────
    let t = m.then(Instant::now);
    let neighbors = state.searcher.search(&query, 5);
    if let Some(t) = t { timing::SEARCH.record(t.elapsed().as_micros() as u64); }

    let fraud_count = neighbors.iter().filter(|n| n.is_fraud).count();

    if let Some(t) = t_total { timing::TOTAL.record(t.elapsed().as_micros() as u64); }

    (
        [(header::CONTENT_TYPE, "application/json")],
        Bytes::from_static(RESPONSES[fraud_count]),
    )
        .into_response()
}

fn env_var(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_string())
}

#[tokio::main]
async fn main() {
    let norm_path = env_var("NORM_PATH", "/data/normalization.json");
    let norm: vectorize::Normalization = {
        let data = std::fs::read_to_string(&norm_path)
            .unwrap_or_else(|e| panic!("normalization: read {norm_path}: {e}"));
        serde_json::from_str(&data).unwrap_or_else(|e| panic!("normalization: parse: {e}"))
    };

    let mcc_path = env_var("MCC_RISK_PATH", "/data/mcc_risk.json");
    let mcc_risk: vectorize::MccRisk = {
        let data = std::fs::read_to_string(&mcc_path)
            .unwrap_or_else(|e| panic!("mcc_risk: read {mcc_path}: {e}"));
        serde_json::from_str(&data).unwrap_or_else(|e| panic!("mcc_risk: parse: {e}"))
    };

    let index_path = env_var("INDEX_PATH", "/data/index.ivf.bin");
    let probes: usize = env_var("IVF_PROBES", "20")
        .parse()
        .expect("IVF_PROBES must be a positive integer");
    let searcher = search::Searcher::load(&index_path, probes)
        .unwrap_or_else(|e| panic!("ivf: load {index_path}: {e}"));

    let state = Arc::new(AppState { searcher, norm, mcc_risk });

    // METRICS=true  → record per-phase histograms + expose /metrics
    // METRICS=false → zero overhead; /metrics returns "disabled"
    let metrics_on = env_var("METRICS", "false").eq_ignore_ascii_case("true");
    timing::ENABLED.store(metrics_on, Relaxed);
    eprintln!("metrics={metrics_on} (GET /metrics for live histograms)");

    let app = Router::new()
        .route("/ready", get(ready))
        .route("/fraud-score", post(fraud_score))
        .route("/metrics", get(metrics))
        .with_state(state);

    let socket_path = env_var("SOCKET_PATH", "");

    if !socket_path.is_empty() {
        // UDS mode: HAProxy connects via Unix socket.
        // Also bind a TCP listener on 9999 for Docker health checks —
        // debian:bookworm-slim has no bash, so /dev/tcp tricks don't work.
        // wget (installed in the image) hits this TCP port instead.
        let port = env_var("PORT", "9999");
        let health_addr = format!("0.0.0.0:{port}");
        let health_app = Router::new()
            .route("/ready", get(ready));
        let health_listener = tokio::net::TcpListener::bind(&health_addr)
            .await
            .unwrap_or_else(|e| panic!("bind health-tcp {health_addr}: {e}"));
        eprintln!("health check on tcp:{health_addr}");
        tokio::spawn(async move {
            axum::serve(health_listener, health_app)
                .await
                .expect("health server error");
        });

        serve_uds(app, &socket_path, probes).await;
    } else {
        serve_tcp(app, probes).await;
    }
}

/// Serve HTTP/1.1 over a Unix domain socket.
///
/// The vendored axum 0.7 pins `axum::serve` to `TcpListener`, so we drive
/// the accept loop ourselves using `hyper::server::conn::http1` (already in
/// the vendor tree as a dep of axum).
#[cfg(unix)]
async fn serve_uds(app: axum::Router, socket_path: &str, probes: usize) {
    use hyper::server::conn::http1;
    use hyper_util::rt::TokioIo;
    use hyper_util::service::TowerToHyperService;
    use std::os::unix::fs::PermissionsExt;

    // Clean up any socket file left by a previous container run.
    let _ = std::fs::remove_file(socket_path);

    let listener = tokio::net::UnixListener::bind(socket_path)
        .unwrap_or_else(|e| panic!("bind uds {socket_path}: {e}"));

    // 0o666 lets HAProxy (same or different uid) connect without privilege.
    std::fs::set_permissions(socket_path, std::fs::Permissions::from_mode(0o666))
        .unwrap_or_else(|e| panic!("chmod {socket_path}: {e}"));

    eprintln!("listening on uds:{socket_path} (probes={probes})");

    loop {
        let (stream, _addr) = listener.accept().await.expect("uds accept failed");
        let io = TokioIo::new(stream);
        let svc = TowerToHyperService::new(app.clone());
        tokio::spawn(async move {
            if let Err(e) = http1::Builder::new()
                .keep_alive(true)
                .serve_connection(io, svc)
                .await
            {
                // Most errors here are benign (client closed before response).
                let s = e.to_string();
                if !s.contains("connection closed") && !s.contains("broken pipe") {
                    eprintln!("uds conn error: {e}");
                }
            }
        });
    }
}

// On non-unix platforms (Windows) the UDS path is unreachable; fall through
// to TCP.  This silences the "unused" warning on the socket_path variable.
#[cfg(not(unix))]
async fn serve_uds(_app: axum::Router, socket_path: &str, probes: usize) {
    eprintln!("UDS not supported on this platform, ignoring SOCKET_PATH={socket_path}");
    // Caller will fall through to serve_tcp in practice (socket_path is empty on Windows).
    let _ = probes;
}

async fn serve_tcp(app: axum::Router, probes: usize) {
    let port = env_var("PORT", "9999");
    let addr = format!("0.0.0.0:{port}");
    eprintln!("listening on tcp:{addr} (probes={probes})");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|e| panic!("bind tcp {addr}: {e}"));
    axum::serve(listener, app).await.expect("server error");
}
