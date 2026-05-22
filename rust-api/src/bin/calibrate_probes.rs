use serde::Deserialize;
use std::{fs::File, io::BufReader, time::Instant};

#[path = "../vectorize.rs"]
mod vectorize;
#[path = "../search.rs"]
mod search;

#[derive(Deserialize)]
struct TestData {
    entries: Vec<TestEntry>,
}

#[derive(Deserialize)]
struct TestEntry {
    expected_approved: bool,
    request: serde_json::Value,
}

fn main() {
    let norm_path = "/Users/silveira.nic/dev/personal/rinha-2026-rust/data/normalization.json";
    let mcc_path = "/Users/silveira.nic/dev/personal/rinha-2026-rust/data/mcc_risk.json";
    let index_path = "/Users/silveira.nic/dev/personal/rinha-2026-rust/data/index.ivf.bin";
    let test_data_path = "/Users/silveira.nic/dev/personal/rinha-2026-rust/test/test-data.json";

    println!("Loading normalization and MCC risk...");
    let norm: vectorize::Normalization = serde_json::from_reader(
        BufReader::new(File::open(norm_path).unwrap())
    ).unwrap();
    let mcc_risk: vectorize::MccRisk = serde_json::from_reader(
        BufReader::new(File::open(mcc_path).unwrap())
    ).unwrap();

    println!("Loading test data from {}...", test_data_path);
    let test_data: TestData = serde_json::from_reader(
        BufReader::new(File::open(test_data_path).unwrap())
    ).unwrap();
    println!("Loaded {} test entries.", test_data.entries.len());

    // Pre-vectorize all test entries to make search loop super fast.
    println!("Pre-vectorizing query requests...");
    let t_vec = Instant::now();
    let queries: Vec<(bool, [f32; 14])> = test_data.entries.iter().map(|entry| {
        let req: vectorize::Request = serde_json::from_value(entry.request.clone()).unwrap();
        let query_vec = vectorize::vectorize(&req, &norm, &mcc_risk);
        (entry.expected_approved, query_vec)
    }).collect();
    println!("Vectorization done in {:.2}s", t_vec.elapsed().as_secs_f32());

    // Try different probe values
    let probe_values = [70, 71, 72, 73, 74, 75];
    for &probes in &probe_values {
        println!("\nTesting IVF search with probes = {}...", probes);
        let searcher = search::Searcher::load(index_path, probes).unwrap();
        
        let t_start = Instant::now();
        let mut mismatches = 0;
        let mut false_positives = 0;
        let mut false_negatives = 0;

        for &(expected_approved, ref query) in &queries {
            let neighbors = searcher.search(query, 5);
            let fraud_count = neighbors.iter().filter(|n| n.is_fraud).count();
            let approved = fraud_count < 3;

            if approved != expected_approved {
                mismatches += 1;
                if expected_approved {
                    false_positives += 1; // expected approved (legit), but we denied it
                } else {
                    false_negatives += 1; // expected denied (fraud), but we approved it
                }
            }
        }

        let elapsed = t_start.elapsed().as_secs_f32();
        let error_rate = (mismatches as f64) / (queries.len() as f64) * 100.0;
        println!(
            "Results for probes = {}: mismatches = {} (FP = {}, FN = {}), error rate = {:.4}%, time = {:.2}s ({:.1} us/query)",
            probes,
            mismatches,
            false_positives,
            false_negatives,
            error_rate,
            elapsed,
            (elapsed * 1_000_000.0) / (queries.len() as f32)
        );

        if mismatches == 0 {
            println!("SUCCESS: 0.00% failure rate achieved at probes = {}!", probes);
        }
    }
}
