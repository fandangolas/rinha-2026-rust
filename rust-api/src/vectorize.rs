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
        // Spec: day_of_week / 6, where mon=0 … sun=6 (ISO Monday-origin).
        // num_days_from_monday() gives Monday=0 … Sunday=6, matching the spec.
        // The previous num_days_from_sunday() yielded Sunday=0 … Saturday=6,
        // shifting every day by 1 and mapping Sunday to 0.0 instead of 1.0.
        naive.weekday().num_days_from_monday() as f32 / 6.0
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

#[cfg(test)]
mod tests {
    use super::*;

    // Minimal normalization and MCC risk fixtures matching the contest's
    // normalization.json defaults (values are irrelevant for dim4 tests but
    // must be non-zero to avoid division-by-zero in other dims).
    fn norm() -> Normalization {
        Normalization {
            max_amount: 10_000.0,
            max_installments: 12.0,
            amount_vs_avg_ratio: 10.0,
            max_minutes: 1440.0,
            max_km: 1000.0,
            max_tx_count_24h: 20.0,
            max_merchant_avg_amount: 10_000.0,
        }
    }

    fn mcc_risk() -> MccRisk {
        HashMap::new() // all lookups fall through to DEFAULT_MCC_RISK = 0.5
    }

    /// Build a minimal Request with only requested_at varying; all other fields
    /// are neutral (zero amounts, empty merchants list, no last_transaction).
    fn request_at(requested_at: &str) -> Request {
        Request {
            transaction: Transaction {
                amount: 0.0,
                installments: 0,
                requested_at: requested_at.to_string(),
            },
            customer: Customer {
                avg_amount: 1.0, // non-zero to avoid zero-division in dim2
                tx_count_24h: 0,
                known_merchants: vec![],
            },
            merchant: Merchant {
                id: "m1".to_string(),
                mcc: "0000".to_string(),
                avg_amount: 0.0,
            },
            terminal: Terminal {
                is_online: false,
                card_present: false,
                km_from_home: 0.0,
            },
            last_tx: None,
        }
    }

    // ── Spec: day_of_week formula is `day_of_week(requested_at) / 6`
    //         where mon=0, sun=6.
    //
    // Our current implementation calls `num_days_from_sunday()` which yields
    // sun=0, mon=1, ..., sat=6 — off by one for every day, and a full swing
    // (0.0 vs 1.0) for Sunday specifically.
    //
    // These tests assert the CORRECT spec-compliant values and therefore FAIL
    // on the current code.  After the fix they must all pass.

    #[test]
    fn dim4_monday_is_zero() {
        // 2026-03-23 is a Monday.
        let v = vectorize(&request_at("2026-03-23T12:00:00Z"), &norm(), &mcc_risk());
        assert_eq!(v[4], 0.0 / 6.0, "Monday must map to dim4=0.0 (spec: mon=0)");
    }

    #[test]
    fn dim4_tuesday_is_one_sixth() {
        // 2026-03-24 is a Tuesday.
        let v = vectorize(&request_at("2026-03-24T12:00:00Z"), &norm(), &mcc_risk());
        let expected = 1.0_f32 / 6.0;
        assert!(
            (v[4] - expected).abs() < 1e-6,
            "Tuesday must map to dim4≈{expected:.6} (spec: tue=1), got {}",
            v[4]
        );
    }

    #[test]
    fn dim4_wednesday_is_two_sixths() {
        // 2026-03-25 is a Wednesday.
        let v = vectorize(&request_at("2026-03-25T12:00:00Z"), &norm(), &mcc_risk());
        let expected = 2.0_f32 / 6.0;
        assert!(
            (v[4] - expected).abs() < 1e-6,
            "Wednesday must map to dim4≈{expected:.6} (spec: wed=2), got {}",
            v[4]
        );
    }

    #[test]
    fn dim4_thursday_is_three_sixths() {
        // 2026-03-26 is a Thursday.
        let v = vectorize(&request_at("2026-03-26T12:00:00Z"), &norm(), &mcc_risk());
        let expected = 3.0_f32 / 6.0;
        assert!(
            (v[4] - expected).abs() < 1e-6,
            "Thursday must map to dim4≈{expected:.6} (spec: thu=3), got {}",
            v[4]
        );
    }

    #[test]
    fn dim4_friday_is_four_sixths() {
        // 2026-03-27 is a Friday.
        let v = vectorize(&request_at("2026-03-27T12:00:00Z"), &norm(), &mcc_risk());
        let expected = 4.0_f32 / 6.0;
        assert!(
            (v[4] - expected).abs() < 1e-6,
            "Friday must map to dim4≈{expected:.6} (spec: fri=4), got {}",
            v[4]
        );
    }

    #[test]
    fn dim4_saturday_is_five_sixths() {
        // 2026-03-28 is a Saturday.
        let v = vectorize(&request_at("2026-03-28T12:00:00Z"), &norm(), &mcc_risk());
        let expected = 5.0_f32 / 6.0;
        assert!(
            (v[4] - expected).abs() < 1e-6,
            "Saturday must map to dim4≈{expected:.6} (spec: sat=5), got {}",
            v[4]
        );
    }

    #[test]
    fn dim4_sunday_is_one() {
        // 2026-03-15 is a Sunday.
        // With num_days_from_sunday() the current code returns 0.0 — a full 1.0
        // swing from the correct value of 1.0.  This is the worst-case mismatch.
        let v = vectorize(&request_at("2026-03-15T12:00:00Z"), &norm(), &mcc_risk());
        assert_eq!(v[4], 6.0 / 6.0, "Sunday must map to dim4=1.0 (spec: sun=6), got {}", v[4]);
    }

    // Regression: other dims must be unaffected by the day_of_week fix.

    #[test]
    fn dim3_hour_of_day_unaffected() {
        // Midnight → 0/23 = 0.0; 23:00 → 23/23 = 1.0
        let v_midnight = vectorize(&request_at("2026-03-23T00:00:00Z"), &norm(), &mcc_risk());
        let v_evening = vectorize(&request_at("2026-03-23T23:00:00Z"), &norm(), &mcc_risk());
        assert_eq!(v_midnight[3], 0.0, "midnight must give dim3=0.0");
        assert!((v_evening[3] - 1.0).abs() < 1e-6, "23:00 must give dim3=1.0");
    }
}
