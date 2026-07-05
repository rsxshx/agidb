//! Retrieval metrics + latency percentiles.

use serde::Serialize;

use crate::corpus::{BenchQuery, QueryClass};

#[derive(Serialize, Default, Clone)]
pub struct ClassMetrics {
    pub queries: usize,
    pub hit_at_1: f64,
    pub hit_at_5: f64,
    pub mrr_at_10: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
}

#[derive(Serialize)]
pub struct SystemReport {
    pub system: String,
    pub episodes: usize,
    pub ingest_ms: f64,
    pub ingest_per_sec: f64,
    pub disk_bytes: u64,
    pub overall: ClassMetrics,
    pub exact: ClassMetrics,
    pub single_entity: ClassMetrics,
    pub noisy: ClassMetrics,
    pub temporal: ClassMetrics,
}

pub struct Sample {
    pub class: QueryClass,
    pub rank_of_first_relevant: Option<usize>,
    pub latency_ms: f64,
}

pub fn score(q: &BenchQuery, ranked: &[u64], latency_ms: f64) -> Sample {
    let rank = ranked
        .iter()
        .position(|id| q.relevant.contains(id))
        .map(|p| p + 1);
    Sample {
        class: q.class.clone(),
        rank_of_first_relevant: rank,
        latency_ms,
    }
}

pub fn aggregate(samples: &[&Sample]) -> ClassMetrics {
    if samples.is_empty() {
        return ClassMetrics::default();
    }
    let n = samples.len() as f64;
    let hit1 = samples
        .iter()
        .filter(|s| s.rank_of_first_relevant == Some(1))
        .count() as f64
        / n;
    let hit5 = samples
        .iter()
        .filter(|s| s.rank_of_first_relevant.map(|r| r <= 5).unwrap_or(false))
        .count() as f64
        / n;
    let mrr = samples
        .iter()
        .map(|s| {
            s.rank_of_first_relevant
                .map(|r| 1.0 / r as f64)
                .unwrap_or(0.0)
        })
        .sum::<f64>()
        / n;
    let mut lat: Vec<f64> = samples.iter().map(|s| s.latency_ms).collect();
    lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f64| lat[((lat.len() as f64 - 1.0) * p) as usize];
    ClassMetrics {
        queries: samples.len(),
        hit_at_1: hit1,
        hit_at_5: hit5,
        mrr_at_10: mrr,
        p50_ms: pct(0.50),
        p95_ms: pct(0.95),
    }
}
