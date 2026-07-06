//! Phase 6 — consolidation invariants.
//!
//! Cluster + atom creation + contradiction detection. Decay and
//! compaction are deferred so they don't appear here.

use agidb_core::consolidate::{ConsolidationReport, MIN_EVIDENCE};
use agidb_core::episode::encode_episode_signature;
use agidb_core::store::{Store, StoreConfig, CONSOLIDATION_LOG};
use agidb_core::types::{Episode, EpisodeId, Provenance, Query, TimeRange, Triple};
use chrono::{Duration, TimeZone, Utc};
use tempfile::TempDir;

// --- helpers ---------------------------------------------------------------

fn fresh_store() -> (Store, TempDir) {
    let dir = TempDir::new().expect("tempdir");
    let store = Store::open(StoreConfig::at(dir.path())).expect("open");
    (store, dir)
}

fn t(year: i32, month: u32, day: u32) -> chrono::DateTime<Utc> {
    Utc.with_ymd_and_hms(year, month, day, 0, 0, 0).unwrap()
}

fn observe_triple(
    store: &mut Store,
    id: u64,
    subj: &str,
    pred: &str,
    obj: &str,
    at: chrono::DateTime<Utc>,
) -> EpisodeId {
    let ep_id = EpisodeId::new(id);
    let triples = vec![Triple {
        subject: subj.into(),
        predicate: pred.into(),
        object: obj.into(),
        confidence: 0.9,
        episode_id: ep_id,
    }];
    let sig = encode_episode_signature(&triples, Some(at));
    let ep = Episode {
        id: ep_id,
        text: format!("{subj} {pred} {obj}"),
        signature_offset: 0,
        gist_offset: 0,
        embedding_offset: 0,
        triples,
        valid_time: TimeRange::point(at),
        t_tx_start: at,
        provenance: Provenance {
            source: "test".into(),
            ..Provenance::default()
        },
        confidence: 0.9,
        superseded_by: None,
    };
    store.observe(ep, &sig).expect("observe")
}

// --- empty store -----------------------------------------------------------

#[test]
fn consolidate_empty_store_returns_zero_atoms() {
    let (mut store, _dir) = fresh_store();
    let r: ConsolidationReport = store.consolidate().expect("consolidate");
    assert_eq!(r.episodes_scanned, 0);
    assert_eq!(r.semantic_atoms_created, 0);
    assert_eq!(r.contradictions_detected, 0);
}

// --- min-evidence threshold ------------------------------------------------

#[test]
fn consolidate_below_min_evidence_creates_no_atom() {
    let (mut store, _dir) = fresh_store();
    let at = t(2026, 5, 14);
    // Two identical triples — same signature — but only 2 episodes,
    // below MIN_EVIDENCE = 3.
    observe_triple(&mut store, 1, "Sarah", "recommended", "Bawri", at);
    observe_triple(&mut store, 2, "Sarah", "recommended", "Bawri", at);

    let r = store.consolidate().expect("consolidate");
    assert_eq!(r.episodes_scanned, 2);
    assert_eq!(
        r.semantic_atoms_created, 0,
        "clusters below MIN_EVIDENCE must not produce atoms"
    );
}

#[test]
fn consolidate_at_min_evidence_creates_one_atom() {
    assert_eq!(MIN_EVIDENCE, 3, "this test pins MIN_EVIDENCE = 3");
    let (mut store, _dir) = fresh_store();
    let at = t(2026, 5, 14);
    for id in 1..=3u64 {
        observe_triple(&mut store, id, "Sarah", "recommended", "Bawri", at);
    }
    let r = store.consolidate().expect("consolidate");
    assert_eq!(r.episodes_scanned, 3);
    assert_eq!(
        r.semantic_atoms_created, 1,
        "exactly one atom should be created for one cluster of 3"
    );
}

// --- idempotency -----------------------------------------------------------

#[test]
fn consolidate_is_idempotent_on_unchanged_store() {
    let (mut store, _dir) = fresh_store();
    let at = t(2026, 5, 14);
    for id in 1..=3u64 {
        observe_triple(&mut store, id, "Sarah", "recommended", "Bawri", at);
    }
    let first = store.consolidate().expect("consolidate");
    let second = store.consolidate().expect("consolidate");
    assert_eq!(first.semantic_atoms_created, 1);
    assert_eq!(
        second.semantic_atoms_created, 0,
        "a second pass over the same store should create no new atoms"
    );
}

// --- consolidation log -----------------------------------------------------

#[test]
fn consolidate_writes_a_log_entry() {
    use redb::ReadableTable;
    let (mut store, _dir) = fresh_store();
    let _ = store.consolidate().expect("consolidate");
    let tx = store.db.begin_read().expect("begin read");
    let table = tx.open_table(CONSOLIDATION_LOG).expect("open log");
    let count = table.iter().expect("iter").count();
    assert_eq!(
        count, 1,
        "consolidate() should append exactly one log entry"
    );
}

// --- contradiction detection -----------------------------------------------

#[test]
fn contradiction_detection_supersedes_older_fact() {
    let (mut store, _dir) = fresh_store();
    let t1 = t(2026, 1, 1);
    let t2 = t1 + Duration::days(30);

    // Two facts about Sarah's favorite thai place, second supersedes
    // first.
    let older = observe_triple(&mut store, 1, "Sarah", "favorite_thai", "Bawri", t1);
    let _newer = observe_triple(&mut store, 2, "Sarah", "favorite_thai", "Olive", t2);

    let r = store.consolidate().expect("consolidate");
    assert!(
        r.contradictions_detected >= 1,
        "(subject, predicate) collision with different objects must be flagged"
    );

    let older_ep = store.get_episode(older).unwrap().expect("present");
    assert!(
        older_ep.superseded_by.is_some(),
        "older fact must carry a supersession link after consolidate"
    );
    assert!(
        older_ep.valid_time.end.is_some(),
        "older fact's valid_time must be closed"
    );
}

// --- recall integration ----------------------------------------------------

#[test]
fn consolidated_atom_surfaces_in_recall_via_concept_token() {
    let (mut store, _dir) = fresh_store();
    let at = t(2026, 5, 14);
    for id in 1..=3u64 {
        observe_triple(&mut store, id, "Sarah", "recommended", "Bawri", at);
    }
    let r = store.consolidate().expect("consolidate");
    assert_eq!(r.semantic_atoms_created, 1);

    let recall = store.recall(&Query::cue("Sarah")).expect("recall");
    assert_eq!(
        recall.semantic_atoms.len(),
        1,
        "recall by a concept token must surface the consolidated atom for that concept"
    );
    let m = &recall.semantic_atoms[0];
    assert_eq!(m.evidence_count, 3);
    assert_eq!(
        m.evidence.len(),
        3,
        "atom must carry the full evidence list of source EpisodeIds"
    );
}

// --- exit-criterion-shaped reduction test ----------------------------------

#[test]
fn consolidation_reduces_redundant_episodes_into_atoms() {
    // Phase 6 exit criterion shape: a controlled-redundancy dataset
    // collapses to a much smaller atom count without losing coverage
    // of the underlying concepts.
    //
    // 20 unique (subject, predicate, object) triples, each repeated
    // MIN_EVIDENCE times → 60 episodes total → expected ≥ 20 atoms.
    // A 30 %+ "reduction" (atoms / episodes ≤ 0.7) holds by inspection.
    let (mut store, _dir) = fresh_store();
    let at = t(2026, 5, 14);
    let people = ["alice", "bob", "carol", "dave"];
    let verbs = ["recommends", "dislikes"];
    let places = ["spicelounge", "havana", "kafe"];
    // Iterate every (person, verb) × every place — 4 × 2 × 3 = 24 unique
    // triples. Each is observed MIN_EVIDENCE times for a total of 72 eps.
    let mut id = 1u64;
    for p in &people {
        for v in &verbs {
            for pl in &places {
                for _ in 0..MIN_EVIDENCE {
                    observe_triple(&mut store, id, p, v, pl, at);
                    id += 1;
                }
            }
        }
    }
    let total_episodes = (id - 1) as u32;
    assert_eq!(total_episodes, 72);

    let r = store.consolidate().expect("consolidate");
    assert_eq!(r.episodes_scanned, total_episodes);
    assert!(
        r.semantic_atoms_created >= 24,
        "expected ≥ 24 atoms (one per unique triple), got {}",
        r.semantic_atoms_created
    );

    // 30 % reduction → atoms ≤ 0.7 × episodes.
    let ratio = r.semantic_atoms_created as f32 / total_episodes as f32;
    assert!(
        ratio <= 0.7,
        "atom/episode ratio {:.2} must be ≤ 0.7 (≥ 30% reduction)",
        ratio
    );
}
