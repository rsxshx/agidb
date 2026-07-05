//! agidb CLI — open, observe, recall, consolidate, export, import, serve.
//!
//! A thin human-readable shell over the [`agidb`] facade. Every command
//! opens the store at `<db>`, runs one operation, prints the result, and
//! exits. `serve` launches the MCP stdio server for Claude Desktop /
//! Cursor integration.
//!
//! Examples:
//!   agidb observe ./mem.agidb "Sarah recommended Bawri in Bandra"
//!   agidb recall ./mem.agidb "what did sarah say about bawri?"
//!   agidb consolidate ./mem.agidb
//!   agidb stats ./mem.agidb
//!   agidb serve ./mem.agidb

use std::path::PathBuf;

use agidb::{Agidb, AgidbConfig, ExtractorSetup, Query, Tier};
use agidb::{Belief, BeliefId, ConceptId, EpisodeId, Goal, GoalId, UnlearnTarget};
use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "agidb",
    version,
    about = "Embedded, content-addressable memory database for AI agents.",
    long_about = None
)]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Record a new observation (runs layer-2 extraction when a model is loaded).
    Observe {
        /// Store directory.
        db: PathBuf,
        /// The observation text.
        text: String,
        /// Provenance source label (default: "user").
        #[arg(long)]
        source: Option<String>,
        /// Skip model load; store a text-only episode (fast, deterministic).
        #[arg(long)]
        offline: bool,
    },
    /// Recall by cue. Never returns the empty set under the default tier floor.
    Recall {
        db: PathBuf,
        cue: String,
        /// Max matches (default 10).
        #[arg(long, default_value_t = 10)]
        k: u32,
        /// Confidence floor.
        #[arg(long)]
        min_confidence: Option<f32>,
        /// Deepest tier allowed: exact|similarity|gist|nearest (default nearest).
        #[arg(long)]
        tier_floor: Option<String>,
    },
    /// Run one consolidation pass: cluster episodes, mint semantic atoms,
    /// detect contradictions.
    Consolidate { db: PathBuf },
    /// Fetch a single episode by id.
    Get { db: PathBuf, id: u64 },
    /// List up to `limit` episodes.
    List {
        db: PathBuf,
        #[arg(default_value_t = 20)]
        limit: u32,
    },
    /// Print store row counts + signature file size.
    Stats { db: PathBuf },
    /// Dump every episode (with its HV) as JSON lines into `file`.
    Export { db: PathBuf, file: PathBuf },
    /// Import JSON lines produced by `export`.
    Import { db: PathBuf, file: PathBuf },
    /// Launch the MCP stdio server for Claude Desktop / Cursor.
    Serve {
        db: PathBuf,
        #[arg(long)]
        offline: bool,
    },
    // -- goals (floor 6) --------------------------------------------------
    /// Set a new goal. Prints the minted GoalId.
    GoalSet {
        db: PathBuf,
        description: String,
        #[arg(long)]
        offline: bool,
    },
    /// List all goals (or only active with --active).
    GoalList {
        db: PathBuf,
        #[arg(long)]
        active: bool,
    },
    /// Show one goal.
    GoalGet { db: PathBuf, id: u64 },
    /// Complete a goal with optional evidence episode ids.
    GoalComplete {
        db: PathBuf,
        id: u64,
        #[arg(long)]
        evidence: Vec<u64>,
    },
    /// Abandon a goal.
    GoalAbandon {
        db: PathBuf,
        id: u64,
        reason: String,
    },
    // -- beliefs (floor 6) ------------------------------------------------
    /// Assert a belief. Use --subject/--predicate/--object to make it
    /// queryable by subject; --confidence sets the initial grade.
    BeliefAssert {
        db: PathBuf,
        claim: String,
        #[arg(long)]
        subject: Option<String>,
        #[arg(long)]
        predicate: Option<String>,
        #[arg(long)]
        object: Option<String>,
        #[arg(long, default_value_t = 0.5)]
        confidence: f32,
    },
    /// List all beliefs (or only beliefs about --subject).
    BeliefList {
        db: PathBuf,
        #[arg(long)]
        subject: Option<String>,
    },
    /// Revise a belief with new evidence (--evidence ID --supports|--contradicts).
    BeliefRevise {
        db: PathBuf,
        id: u64,
        #[arg(long)]
        evidence: u64,
        #[arg(long)]
        supports: bool,
        #[arg(long, default_value = "new evidence")]
        reason: String,
    },
    /// Withdraw a belief.
    BeliefWithdraw {
        db: PathBuf,
        id: u64,
        reason: String,
    },
    /// Show the append-only revision history of a belief.
    BeliefHistory { db: PathBuf, id: u64 },
    // -- unlearn (phase 11) -----------------------------------------------
    /// Forget a concept and everything referencing it. Non-destructive
    /// (tombstoned); recoverable within 30 days.
    UnlearnConcept {
        db: PathBuf,
        concept_id: u64,
        reason: String,
    },
    /// Forget a single episode.
    UnlearnEpisode {
        db: PathBuf,
        id: u64,
        reason: String,
    },
    /// Forget everything from a source (GDPR Article 17).
    UnlearnSource {
        db: PathBuf,
        source: String,
        reason: String,
    },
    /// Show the unlearn (tombstone) history.
    UnlearnHistory { db: PathBuf },
    /// Restore a tombstoned unlearn within the 30-day window.
    Restore { db: PathBuf, audit_event_id: u64 },
    // -- self-model (phase 10) --------------------------------------------
    /// What did I learn? Prints the learning-event log (optionally since
    /// an ISO-8601 timestamp).
    WhatDidILearn {
        db: PathBuf,
        #[arg(long)]
        since: Option<String>,
    },
    /// Show the current self-vector (hamming weight + history count).
    SelfVector { db: PathBuf },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Default to `info` but honor `RUST_LOG` so the recording script
    // (and users debugging) can set `RUST_LOG=error` to silence the
    // model-manager noise.
    use tracing_subscriber::EnvFilter;
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(filter)
        .init();

    let cli = Cli::parse();
    match cli.cmd {
        Command::Observe {
            db,
            text,
            source,
            offline,
        } => {
            let ag = open(&db, offline).await?;
            let src = source.unwrap_or_else(|| "user".into());
            let id = ag.observe_with(&text, &src).await?;
            println!("observed episode {} (source={})", id, src);
            print_extractor_banner(&ag);
        }
        Command::Recall {
            db,
            cue,
            k,
            min_confidence,
            tier_floor,
        } => {
            let ag = open(&db, true).await?;
            let mut q = Query::cue(&cue).with_k(k as usize);
            if let Some(mc) = min_confidence {
                q = q.with_min_confidence(mc);
            }
            if let Some(tf) = tier_floor.as_deref() {
                q = q.with_tier_floor(parse_tier(tf)?);
            }
            let r = ag.recall(q).await?;
            println!("tier_used: {:?}  elapsed_ms: {}", r.tier_used, r.elapsed_ms);
            println!("matches ({}):", r.matches.len());
            for m in &r.matches {
                println!(
                    "  [{:.2}] ep{} ({:?}){} {}",
                    m.confidence,
                    m.episode_id.raw(),
                    m.source_tier,
                    if m.superseded { " [superseded]" } else { "" },
                    m.text,
                );
            }
            if !r.semantic_atoms.is_empty() {
                println!("semantic_atoms ({}):", r.semantic_atoms.len());
                for a in &r.semantic_atoms {
                    println!(
                        "  [{:.2}] atom{} (evidence={}) {}",
                        a.confidence,
                        a.atom_id.raw(),
                        a.evidence_count,
                        a.statement,
                    );
                }
            }
        }
        Command::Consolidate { db } => {
            let ag = open(&db, true).await?;
            let r = ag.consolidate().await?;
            println!(
                "consolidated: scanned={}, atoms_created={}, contradictions={}, elapsed_ms={}",
                r.episodes_scanned,
                r.semantic_atoms_created,
                r.contradictions_detected,
                r.elapsed_ms,
            );
        }
        Command::Get { db, id } => {
            let ag = open(&db, true).await?;
            match ag.get_episode(id).await? {
                Some(ep) => {
                    println!("episode {}", ep.id);
                    println!("  text: {}", ep.text);
                    println!("  confidence: {:.2}", ep.confidence);
                    println!(
                        "  valid_time: {} ..= {:?}",
                        ep.valid_time.start.to_rfc3339(),
                        ep.valid_time.end.map(|e| e.to_rfc3339())
                    );
                    println!("  triples ({}):", ep.triples.len());
                    for t in &ep.triples {
                        println!(
                            "    ({:.2}) {} | {} | {}",
                            t.confidence, t.subject, t.predicate, t.object
                        );
                    }
                    if let Some(sb) = ep.superseded_by {
                        println!("  superseded_by: {}", sb);
                    }
                }
                None => println!("episode {} not found", id),
            }
        }
        Command::List { db, limit } => {
            let ag = open(&db, true).await?;
            for ep in ag.list_episodes(limit as usize).await? {
                println!("ep{} [{:.2}] {}", ep.id.raw(), ep.confidence, ep.text);
            }
        }
        Command::Stats { db } => {
            let ag = open(&db, true).await?;
            let s = ag.stats().await?;
            println!("episodes:             {}", s.episodes);
            println!("concepts:             {}", s.concepts);
            println!("semantic_atoms:       {}", s.semantic_atoms);
            println!("consolidation_passes: {}", s.consolidation_passes);
            println!(
                "signatures:           {} ({} bytes)",
                s.signatures,
                s.signatures * 1024
            );
        }
        Command::Export { db, file } => {
            let ag = open(&db, true).await?;
            ag.export_jsonl(&file).await?;
            println!("exported to {}", file.display());
        }
        Command::Import { db, file } => {
            let ag = open(&db, true).await?;
            let n = ag.import_jsonl(&file).await?;
            println!("imported {} episodes from {}", n, file.display());
        }
        Command::Serve { db, offline } => {
            let ctx = if offline {
                agidb_mcp::AgidbContext::open_null(db.to_str().expect("utf8 db path"))?
            } else {
                agidb_mcp::AgidbContext::open(db.to_str().expect("utf8 db path"))?
            };
            let server = agidb_mcp::McpServer::new(ctx);
            server.run_stdio()?;
        }
        // -- goals ---------------------------------------------------------
        Command::GoalSet {
            db,
            description,
            offline,
        } => {
            let ag = open(&db, offline).await?;
            let id = ag.set_goal(Goal::new(&description)).await?;
            println!("set goal {} — {:?}", id.raw(), description);
        }
        Command::GoalList { db, active } => {
            let ag = open(&db, true).await?;
            let goals = if active {
                ag.active_goals().await?
            } else {
                ag.all_goals().await?
            };
            if goals.is_empty() {
                println!("(no goals)");
            }
            for g in goals {
                println!(
                    "goal{} [{:?}] {}",
                    g.id.raw(),
                    g.state.kind(),
                    g.description
                );
            }
        }
        Command::GoalGet { db, id } => {
            let ag = open(&db, true).await?;
            match ag.get_goal(GoalId::new(id)).await? {
                Some(g) => {
                    println!(
                        "goal{} [{:?}] {}",
                        g.id.raw(),
                        g.state.kind(),
                        g.description
                    );
                    if let Some(p) = g.parent_id {
                        println!("  parent: goal{}", p.raw());
                    }
                    for (i, c) in g.success_criteria.iter().enumerate() {
                        println!(
                            "  criterion[{}] {}{}",
                            i,
                            if c.met { "[met] " } else { "" },
                            c.description
                        );
                    }
                }
                None => println!("goal {} not found", id),
            }
        }
        Command::GoalComplete { db, id, evidence } => {
            let ag = open(&db, true).await?;
            ag.complete_goal(
                GoalId::new(id),
                evidence.into_iter().map(EpisodeId::new).collect(),
            )
            .await?;
            println!("completed goal {}", id);
        }
        Command::GoalAbandon { db, id, reason } => {
            let ag = open(&db, true).await?;
            ag.abandon_goal(GoalId::new(id), reason.clone()).await?;
            println!("abandoned goal {} — {}", id, reason);
        }
        // -- beliefs -------------------------------------------------------
        Command::BeliefAssert {
            db,
            claim,
            subject,
            predicate,
            object,
            confidence,
        } => {
            let ag = open(&db, true).await?;
            let mut b = Belief::new(&claim).with_confidence(confidence);
            if let (Some(s), Some(p), Some(o)) = (subject, predicate, object) {
                b = b.with_triple(s, p, o);
            }
            let id = ag.assert_belief(b).await?;
            println!("asserted belief {} — {:?}", id.raw(), claim);
        }
        Command::BeliefList { db, subject } => {
            let ag = open(&db, true).await?;
            let beliefs = if let Some(s) = subject.as_deref() {
                ag.what_do_i_believe(s).await?
            } else {
                ag.all_beliefs().await?
            };
            if beliefs.is_empty() {
                println!("(no beliefs)");
            }
            for b in beliefs {
                let state = if b.is_withdrawn() {
                    "withdrawn"
                } else {
                    "active"
                };
                println!(
                    "belief{} [{:.2}] {} — {}",
                    b.id.raw(),
                    b.confidence,
                    state,
                    b.claim
                );
                if !b.evidence.is_empty() {
                    println!(
                        "  evidence: {:?}",
                        b.evidence.iter().map(|e| e.raw()).collect::<Vec<_>>()
                    );
                }
                if !b.contradictions.is_empty() {
                    println!(
                        "  contradictions: {:?}",
                        b.contradictions.iter().map(|e| e.raw()).collect::<Vec<_>>()
                    );
                }
            }
        }
        Command::BeliefRevise {
            db,
            id,
            evidence,
            supports,
            reason,
        } => {
            let ag = open(&db, true).await?;
            let r = ag
                .revise_belief(
                    BeliefId::new(id),
                    EpisodeId::new(evidence),
                    supports,
                    reason.clone(),
                )
                .await?;
            let verdict = if r.withdrawn { " → WITHDRAWN" } else { "" };
            println!(
                "revised belief {}: {:.2} → {:.2}{} — {}",
                id, r.previous_confidence, r.new_confidence, verdict, reason,
            );
        }
        Command::BeliefWithdraw { db, id, reason } => {
            let ag = open(&db, true).await?;
            ag.withdraw_belief(BeliefId::new(id), reason.clone())
                .await?;
            println!("withdrew belief {} — {}", id, reason);
        }
        Command::BeliefHistory { db, id } => {
            let ag = open(&db, true).await?;
            let hist = ag.belief_history(BeliefId::new(id)).await?;
            if hist.is_empty() {
                println!("(no revisions for belief {})", id);
            }
            for (i, r) in hist.iter().enumerate() {
                println!(
                    "[{}] {} {:.2} → {:.2} — {}",
                    i,
                    r.timestamp.to_rfc3339(),
                    r.previous_confidence,
                    r.new_confidence,
                    r.reason,
                );
            }
        }
        // -- unlearn -------------------------------------------------------
        Command::UnlearnConcept {
            db,
            concept_id,
            reason,
        } => {
            let ag = open(&db, true).await?;
            let report = ag
                .unlearn(
                    UnlearnTarget::Concept(ConceptId::new(concept_id)),
                    reason.clone(),
                )
                .await?;
            println!(
                "unlearned concept {}: episodes={}, beliefs={}, beliefs_revised={}, atoms={}, sv_drift={}, expiry={}",
                concept_id, report.episodes_removed, report.beliefs_removed,
                report.beliefs_revised, report.semantic_atoms_affected,
                report.self_vector_drift_hamming, report.tombstone_expiry.to_rfc3339(),
            );
            println!(
                "  reason: {}  audit_event_id: {}",
                reason, report.audit_event_id
            );
        }
        Command::UnlearnEpisode { db, id, reason } => {
            let ag = open(&db, true).await?;
            let report = ag
                .unlearn(UnlearnTarget::Episode(EpisodeId::new(id)), reason.clone())
                .await?;
            println!(
                "unlearned episode {}: beliefs_revised={}, sv_drift={}, audit_event_id: {}",
                id, report.beliefs_revised, report.self_vector_drift_hamming, report.audit_event_id,
            );
        }
        Command::UnlearnSource { db, source, reason } => {
            let ag = open(&db, true).await?;
            let report = ag
                .unlearn(UnlearnTarget::BySource(source.clone()), reason.clone())
                .await?;
            println!(
                "unlearned source '{}': episodes={}, beliefs={}, audit_event_id: {}",
                source, report.episodes_removed, report.beliefs_removed, report.audit_event_id,
            );
        }
        Command::UnlearnHistory { db } => {
            let ag = open(&db, true).await?;
            let tombs = ag.unlearn_history().await?;
            if tombs.is_empty() {
                println!("(no tombstones)");
            }
            for t in tombs {
                println!(
                    "kind={} id={} tombstoned={} reason={} audit={}",
                    t.kind,
                    t.id,
                    t.tombstoned_at.to_rfc3339(),
                    t.reason,
                    t.audit_event_id
                );
            }
        }
        Command::Restore { db, audit_event_id } => {
            let ag = open(&db, true).await?;
            let n = ag.restore_within_window(audit_event_id).await?;
            println!(
                "restored {} tombstones from audit event {}",
                n, audit_event_id
            );
        }
        // -- self-model ----------------------------------------------------
        Command::WhatDidILearn { db, since } => {
            let ag = open(&db, true).await?;
            let events = if let Some(s) = since.as_deref() {
                let dt = chrono::DateTime::parse_from_rfc3339(s)
                    .map_err(|e| anyhow::anyhow!("invalid timestamp: {e}"))?
                    .with_timezone(&chrono::Utc);
                ag.what_did_i_learn(dt).await?
            } else {
                ag.all_learning_events().await?
            };
            if events.is_empty() {
                println!("(no learning events)");
            }
            for e in &events {
                println!(
                    "[{}] {} — {}",
                    e.timestamp().to_rfc3339(),
                    e.kind_label(),
                    summarize_event(e)
                );
            }
        }
        Command::SelfVector { db } => {
            let ag = open(&db, true).await?;
            let sv = ag.self_vector().await?;
            let weight: u32 = (0..1024).map(|i| sv.0[i].count_ones()).sum();
            let hist = ag.self_vector_history().await?;
            println!("self_vector hamming_weight: {}/8192", weight);
            println!("history snapshots: {}", hist.len());
        }
    }
    Ok(())
}

async fn open(db: &std::path::Path, offline: bool) -> Result<Agidb> {
    // Non-observe commands never run layer-2 extraction, so they default
    // to the null extractor for instant startup. Only `observe` passes
    // offline=false to attempt the real GLiNER load.
    let cfg = AgidbConfig::new(db).with_extractor(if offline {
        ExtractorSetup::Null
    } else {
        ExtractorSetup::Auto
    });
    Ok(Agidb::open_with(cfg).await?)
}

fn print_extractor_banner(ag: &Agidb) {
    if ag.extractor_loaded() {
        eprintln!("  (layer-2 extractor: GLiNER + heuristics — structured triples active)");
    } else {
        eprintln!("  (layer-2 extractor: none — text-only episode; pass without --offline once to load GLiNER)");
    }
}

fn parse_tier(s: &str) -> Result<Tier> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "exact" => Tier::Exact,
        "similarity" | "sim" => Tier::Similarity,
        "gist" => Tier::Gist,
        "nearest" | "nearestneighbor" | "nn" => Tier::NearestNeighbor,
        other => anyhow::bail!("unknown tier '{other}', expected exact|similarity|gist|nearest"),
    })
}

fn summarize_event(e: &agidb::LearningEvent) -> String {
    use agidb::LearningEvent;
    match e {
        LearningEvent::EpisodeStored { id, .. } => format!("episode {}", id.raw()),
        LearningEvent::GoalSet {
            id, description, ..
        } => format!("goal {} — {}", id.raw(), description),
        LearningEvent::GoalStateChanged { id, from, to, .. } => {
            format!("goal {} {} → {}", id.raw(), from, to)
        }
        LearningEvent::BeliefAsserted {
            id,
            claim,
            confidence,
            ..
        } => format!("belief {} [{:.2}] — {}", id.raw(), confidence, claim),
        LearningEvent::BeliefRevised {
            id,
            previous_confidence,
            new_confidence,
            ..
        } => format!(
            "belief {} {:.2} → {:.2}",
            id.raw(),
            previous_confidence,
            new_confidence
        ),
        LearningEvent::BeliefWithdrawn { id, reason, .. } => {
            format!("belief {} — {}", id.raw(), reason)
        }
        LearningEvent::SemanticAtomFormed {
            atom_id,
            evidence_count,
            ..
        } => format!("atom {} (evidence={})", atom_id.raw(), evidence_count),
        LearningEvent::ContradictionDetected { count, .. } => format!("{} contradictions", count),
        LearningEvent::ConsolidationRun {
            atoms_created,
            contradictions,
            ..
        } => format!("atoms={}, contradictions={}", atoms_created, contradictions),
        LearningEvent::Unlearned {
            target,
            cascade_size,
            self_vector_drift,
            ..
        } => format!(
            "target={}, cascade={}, sv_drift={}",
            target, cascade_size, self_vector_drift
        ),
        LearningEvent::SelfVectorUpdated { drift_hamming, .. } => {
            format!("drift={}", drift_hamming)
        }
    }
}
