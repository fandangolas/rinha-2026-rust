mod search;
mod vectorize;

use axum::{
    body::Bytes,
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use std::{env, sync::Arc};

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

async fn fraud_score(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    let req: vectorize::Request = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
    };

    let query = vectorize::vectorize(&req, &state.norm, &state.mcc_risk);

    let neighbors = state.searcher.search(&query, 5);
    let fraud_count = neighbors.iter().filter(|n| n.is_fraud).count();

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

    let app = Router::new()
        .route("/ready", get(ready))
        .route("/fraud-score", post(fraud_score))
        .with_state(state);

    let port = env_var("PORT", "9999");
    let addr = format!("0.0.0.0:{port}");
    eprintln!(
        "listening on {addr} (probes={probes}, workers={})",
        env_var("TOKIO_WORKER_THREADS", "auto")
    );

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|e| panic!("bind {addr}: {e}"));
    axum::serve(listener, app)
        .await
        .expect("server error");
}
