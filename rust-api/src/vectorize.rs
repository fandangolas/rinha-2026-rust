use chrono::Timelike;
use serde::Deserialize;
use std::collections::HashMap;

pub type MccRisk = HashMap<String, f32>;

#[derive(Deserialize)]
pub struct Normalization {
    pub max_amount: f32,
    pub max_installments: f32,
    pub amount_vs_avg_ratio: f32,
    pub max_minutes: f32,
    pub max_km: f32,
    pub max_tx_count_24h: f32,
    pub max_merchant_avg_amount: f32,
}

// Only fields we actually use are deserialized; serde ignores unknown JSON keys.
#[derive(Deserialize)]
pub struct Request {
    pub transaction: Transaction,
    pub customer: Customer,
    pub merchant: Merchant,
    pub terminal: Terminal,
    #[serde(rename = "last_transaction", default)]
    pub last_tx: Option<LastTx>,
}

#[derive(Deserialize)]
pub struct Transaction {
    pub amount: f32,
    pub installments: i32,
    pub requested_at: String,
}

#[derive(Deserialize)]
pub struct Customer {
    pub avg_amount: f32,
    pub tx_count_24h: i32,
    pub known_merchants: Vec<String>,
}

#[derive(Deserialize)]
pub struct Merchant {
    pub id: String,
    pub mcc: String,
    pub avg_amount: f32,
}

#[derive(Deserialize)]
pub struct Terminal {
    pub is_online: bool,
    pub card_present: bool,
    pub km_from_home: f32,
}

#[derive(Deserialize)]
pub struct LastTx {
    pub timestamp: String,
    pub km_from_current: f32,
}

const DIMS: usize = 14;
const DEFAULT_MCC_RISK: f32 = 0.5;

/// Converts a transaction request into a 14-dimensional search vector.
/// Mirrors the Go Vectorize function exactly.
pub fn vectorize(req: &Request, norm: &Normalization, mcc_risk: &MccRisk) -> [f32; DIMS] {
    let mut v = [0.0f32; DIMS];

    // dim 0 — normalized transaction amount
    v[0] = clamp(req.transaction.amount / norm.max_amount);

    // dim 1 — normalized installment count
    v[1] = clamp(req.transaction.installments as f32 / norm.max_installments);

    // dim 2 — amount relative to customer average
    if req.customer.avg_amount > 0.0 {
        v[2] = clamp(
            (req.transaction.amount / req.customer.avg_amount) / norm.amount_vs_avg_ratio,
        );
    }

    // dim 3, 4 — time of day / day of week (UTC)
    // On parse failure use a zero-like default (epoch = Thursday, midnight).
    let tx_time = chrono::DateTime::parse_from_rfc3339(&req.transaction.requested_at)
        .unwrap_or_else(|_| {
            chrono::DateTime::parse_from_rfc3339("1970-01-01T00:00:00Z").unwrap()
        });
    let naive = tx_time.naive_utc();
    v[3] = naive.hour() as f32 / 23.0;
    v[4] = {
        use chrono::Datelike;
        naive.weekday().num_days_from_sunday() as f32 / 6.0
    };

    // dim 5, 6 — recency / velocity (sentinel -1.0 when no prior transaction)
    match &req.last_tx {
        None => {
            v[5] = -1.0;
            v[6] = -1.0;
        }
        Some(last) => {
            let last_time = chrono::DateTime::parse_from_rfc3339(&last.timestamp)
                .unwrap_or(tx_time);
            let minutes = (naive - last_time.naive_utc()).num_seconds() as f32 / 60.0;
            v[5] = clamp(minutes / norm.max_minutes);
            v[6] = clamp(last.km_from_current / norm.max_km);
        }
    }

    // dim 7 — distance from home
    v[7] = clamp(req.terminal.km_from_home / norm.max_km);

    // dim 8 — transaction frequency in last 24h
    v[8] = clamp(req.customer.tx_count_24h as f32 / norm.max_tx_count_24h);

    // dim 9 — online transaction flag
    v[9] = bool_f(req.terminal.is_online);

    // dim 10 — card present flag
    v[10] = bool_f(req.terminal.card_present);

    // dim 11 — merchant unknown to customer
    v[11] = bool_f(
        !req.customer
            .known_merchants
            .iter()
            .any(|m| m == &req.merchant.id),
    );

    // dim 12 — MCC risk score
    v[12] = mcc_risk
        .get(&req.merchant.mcc)
        .copied()
        .unwrap_or(DEFAULT_MCC_RISK);

    // dim 13 — normalized merchant average amount
    v[13] = clamp(req.merchant.avg_amount / norm.max_merchant_avg_amount);

    v
}

#[inline(always)]
fn clamp(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

#[inline(always)]
fn bool_f(b: bool) -> f32 {
    if b { 1.0 } else { 0.0 }
}
