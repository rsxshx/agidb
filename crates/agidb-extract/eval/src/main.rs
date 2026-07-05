//! Phase-3 gold-set evaluation harness.
//!
//! Loads `observations.jsonl`, runs `agidb_extract::Extractor` over each
//! row, computes triple-level precision / recall / F1 against the
//! human-labelled triples, and writes a JSON report.
//!
//! Two modes:
//!   - default: load real models + run extraction. Errors if the model
//!     cache is empty (first run downloads ~hundreds of MB via
//!     `model_manager`; pin the resulting SHAs into `agidb_extract::models`
//!     after that).
//!   - `--dry-run`: skip model load, treat every row as extracting zero
//!     triples. Useful for verifying the JSONL schema + the harness
//!     plumbing without any models. Produces a P/R/F1 = 0 report.
//!
//! The phase-3 exit gate is **F1 ≥ 0.85 on a 100-sample gold set**.
//! Today's committed gold set is a 3-row placeholder; replace per plan
//! task 15.

use std::collections::HashSet;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use anyhow::{Context, Result};
use chrono::Utc;
use clap::Parser;
use serde::{Deserialize, Serialize};

use agidb_core::types::{ExtractContext, TextExtractor};
use agidb_extract::{Extractor, ExtractorConfig};

#[derive(Parser, Debug)]
#[command(about = "agidb-extract phase-3 gold-set evaluation")]
struct Cli {
    /// Path to the JSONL gold set.
    #[arg(
        long,
        default_value = "crates/agidb-extract/eval/gold/observations.jsonl"
    )]
    gold: PathBuf,

    /// Where to write the JSON report.
    #[arg(long, default_value = "crates/agidb-extract/eval/results/latest.json")]
    out: PathBuf,

    /// Skip Extractor model load; treat every row as zero-triple extraction.
    /// Useful for schema-validation runs without any cached models.
    #[arg(long)]
    dry_run: bool,

    /// F1 exit gate for CI. The build fails (exit 2) below this when
    /// not in dry-run. Default is the phase-3 commitment: 0.85.
    #[arg(long, default_value_t = 0.85)]
    gate: f64,
}

#[derive(Deserialize, Debug, Clone)]
struct GoldRow {
    text: String,
    triples: Vec<GoldTriple>,
    // `notes` is metadata for the labeller, ignored by scoring; serde
    // skips unknown fields by default so we don't even need to declare it.
}

#[derive(Deserialize, Serialize, Clone, Eq, PartialEq, Hash, Debug)]
struct GoldTriple {
    subject: String,
    predicate: String,
    object: String,
}

#[derive(Serialize)]
struct Report {
    precision: f64,
    recall: f64,
    f1: f64,
    n: usize,
    dry_run: bool,
    per_row: Vec<RowReport>,
}

#[derive(Serialize)]
struct RowReport {
    text: String,
    expected: Vec<GoldTriple>,
    extracted: Vec<GoldTriple>,
    tp: usize,
    fp: usize,
    #[serde(rename = "fn")]
    fn_count: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let rows = load_gold(&cli.gold)?;

    // Eager model load so a config problem fails fast — before we walk
    // the gold set. None in dry-run.
    let extractor: Option<Extractor> =
        if cli.dry_run {
            None
        } else {
            Some(Extractor::new(ExtractorConfig::default()).with_context(|| {
                "loading Extractor (run with --dry-run if models aren't cached)"
            })?)
        };

    let mut per_row = Vec::with_capacity(rows.len());
    let (mut tp_total, mut fp_total, mut fn_total) = (0usize, 0usize, 0usize);

    for row in &rows {
        let extracted = if let Some(ext) = &extractor {
            extract_for_row(ext, &row.text)?
        } else {
            Vec::new()
        };
        let (tp, fp, fn_) = score_row(&row.triples, &extracted);
        tp_total += tp;
        fp_total += fp;
        fn_total += fn_;
        per_row.push(RowReport {
            text: row.text.clone(),
            expected: row.triples.clone(),
            extracted,
            tp,
            fp,
            fn_count: fn_,
        });
    }

    let precision = ratio(tp_total, tp_total + fp_total);
    let recall = ratio(tp_total, tp_total + fn_total);
    let f1 = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };

    let report = Report {
        precision,
        recall,
        f1,
        n: rows.len(),
        dry_run: cli.dry_run,
        per_row,
    };

    if let Some(parent) = cli.out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&cli.out, serde_json::to_string_pretty(&report)?)?;

    println!(
        "P={:.3} R={:.3} F1={:.3} (n={}, dry_run={}) → {}",
        precision,
        recall,
        f1,
        rows.len(),
        cli.dry_run,
        cli.out.display()
    );

    // For unattended CI runs we want non-zero exit when the F1 gate
    // isn't met AND we're not in dry-run. (Skip when there's no real
    // extraction — dry-run F1 is always 0 by design.)
    if !cli.dry_run && f1 < cli.gate {
        eprintln!("F1 {:.3} below the gate ({:.3})", f1, cli.gate);
        std::process::exit(2);
    }
    Ok(())
}

fn load_gold(path: &PathBuf) -> Result<Vec<GoldRow>> {
    let f = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut rows = Vec::new();
    for (i, line) in BufReader::new(f).lines().enumerate() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let row: GoldRow = serde_json::from_str(line)
            .with_context(|| format!("parse line {} of gold set: {}", i + 1, line))?;
        rows.push(row);
    }
    Ok(rows)
}

fn extract_for_row(ext: &Extractor, text: &str) -> Result<Vec<GoldTriple>> {
    let ctx = ExtractContext {
        observation_time: Utc::now(),
        relation_hint_types: Vec::new(),
    };
    let extraction = ext.extract(text, &ctx)?;
    Ok(extraction
        .triples
        .into_iter()
        .map(|t| GoldTriple {
            subject: t.subject,
            predicate: t.predicate,
            object: t.object,
        })
        .collect())
}

fn score_row(expected: &[GoldTriple], extracted: &[GoldTriple]) -> (usize, usize, usize) {
    let e: HashSet<_> = expected.iter().cloned().collect();
    let x: HashSet<_> = extracted.iter().cloned().collect();
    let tp = e.intersection(&x).count();
    let fp = x.difference(&e).count();
    let fn_ = e.difference(&x).count();
    (tp, fp, fn_)
}

fn ratio(num: usize, den: usize) -> f64 {
    if den == 0 {
        0.0
    } else {
        num as f64 / den as f64
    }
}
