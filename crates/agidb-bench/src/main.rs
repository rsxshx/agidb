//! Deterministic retrieval benchmark: agidb vs SQLite FTS5 (BM25) vs a
//! naive full scan, over a seeded synthetic corpus.
//!
//! Honesty notes (constitution article X):
//! - The corpus is synthetic and templated. It measures lexical +
//!   structural retrieval, temporal filtering, and typo robustness. It
//!   does NOT measure semantic paraphrase (no system here embeds
//!   meaning) — do not quote these numbers as semantic recall quality.
//! - Covered metrics: hit@k / MRR (F1-style), p50/p95 latency,
//!   noisy-cue degradation, ingest throughput, disk footprint.
//!   Not covered (need external corpora / LLM APIs): BLEU, LLM-judge,
//!   token cost. RESULTS.md must say so.

mod corpus;
mod metrics;
mod systems;

use clap::Parser;
use metrics::{aggregate, score, Sample, SystemReport};
use systems::{time, AgidbSystem, Fts5System, ScanSystem, System};

#[derive(Parser)]
struct Cli {
    #[arg(long, default_value_t = 10_000)]
    episodes: usize,
    /// Total queries (split equally across 4 classes).
    #[arg(long, default_value_t = 200)]
    queries: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Output directory for results JSON + RESULTS.md.
    #[arg(long, default_value = "bench")]
    out: std::path::PathBuf,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let mut rng = corpus::Rng::new(cli.seed);
    let docs = corpus::build_corpus(cli.episodes, &mut rng);
    let queries = corpus::build_queries(&docs, cli.queries / 4, &mut rng);
    eprintln!(
        "corpus: {} docs ({} supersession pairs), {} queries",
        docs.len(),
        docs.iter().filter(|d| d.superseded_by.is_some()).count(),
        queries.len()
    );

    std::fs::create_dir_all(&cli.out)?;
    let work = tempfile_dir()?;

    let mut reports = Vec::new();
    {
        let mut sys = AgidbSystem::open(&work.join("agidb"))?;
        reports.push(run(&mut sys, &docs, &queries)?);
    }
    {
        let mut sys = Fts5System::open(&work.join("fts5.db"))?;
        reports.push(run(&mut sys, &docs, &queries)?);
    }
    {
        let mut sys = ScanSystem::new();
        reports.push(run(&mut sys, &docs, &queries)?);
    }

    let json = serde_json::to_string_pretty(&reports)?;
    let json_path = cli
        .out
        .join(format!("results-{}k.json", cli.episodes / 1000));
    std::fs::write(&json_path, &json)?;
    eprintln!("wrote {}", json_path.display());

    print_markdown(&reports);
    Ok(())
}

fn run(
    sys: &mut dyn System,
    docs: &[corpus::Doc],
    queries: &[corpus::BenchQuery],
) -> anyhow::Result<SystemReport> {
    eprintln!("== {} ==", sys.name());
    let (ingest_result, ingest_ms) = time(|| sys.ingest(docs));
    ingest_result?;
    let mut samples: Vec<Sample> = Vec::with_capacity(queries.len());
    for q in queries {
        let (ranked, ms) = time(|| sys.query(q));
        samples.push(score(q, &ranked?, ms));
    }
    let by_class = |c: corpus::QueryClass| {
        let filtered: Vec<&Sample> = samples.iter().filter(|s| s.class == c).collect();
        aggregate(&filtered)
    };
    let all: Vec<&Sample> = samples.iter().collect();
    Ok(SystemReport {
        system: sys.name().to_string(),
        episodes: docs.len(),
        ingest_ms,
        ingest_per_sec: docs.len() as f64 / (ingest_ms / 1000.0),
        disk_bytes: sys.disk_bytes(),
        overall: aggregate(&all),
        exact: by_class(corpus::QueryClass::Exact),
        single_entity: by_class(corpus::QueryClass::SingleEntity),
        noisy: by_class(corpus::QueryClass::Noisy),
        temporal: by_class(corpus::QueryClass::Temporal),
    })
}

fn print_markdown(reports: &[SystemReport]) {
    println!("| system | ingest/s | disk MB | hit@1 | hit@5 | MRR | p50 ms | p95 ms |");
    println!("|---|---|---|---|---|---|---|---|");
    for r in reports {
        println!(
            "| {} | {:.0} | {:.1} | {:.3} | {:.3} | {:.3} | {:.2} | {:.2} |",
            r.system,
            r.ingest_per_sec,
            r.disk_bytes as f64 / 1e6,
            r.overall.hit_at_1,
            r.overall.hit_at_5,
            r.overall.mrr_at_10,
            r.overall.p50_ms,
            r.overall.p95_ms
        );
    }
    for (label, pick) in [
        ("exact", 0usize),
        ("single-entity", 1),
        ("noisy", 2),
        ("temporal", 3),
    ] {
        println!("\n### {label}");
        println!("| system | hit@1 | hit@5 | MRR | p95 ms |");
        println!("|---|---|---|---|---|");
        for r in reports {
            let m = match pick {
                0 => &r.exact,
                1 => &r.single_entity,
                2 => &r.noisy,
                _ => &r.temporal,
            };
            println!(
                "| {} | {:.3} | {:.3} | {:.3} | {:.2} |",
                r.system, m.hit_at_1, m.hit_at_5, m.mrr_at_10, m.p95_ms
            );
        }
    }
}

fn tempfile_dir() -> anyhow::Result<std::path::PathBuf> {
    let dir = std::env::temp_dir().join(format!("agidb-bench-{}", std::process::id()));
    if dir.exists() {
        std::fs::remove_dir_all(&dir)?;
    }
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}
