//! Phase 4 — tiered recall invariants.
//!
//! Covers tier-A (exact concept lookup), tier-C (gist similarity in
//! the high-confidence band), tier-D (nearest-neighbor low-confidence
//! fallback), the `tier_floor` clamp, the `k` cap, the `as_of`
//! bi-temporal filter, and a 100-episode synthetic smoke run.

use agidb_core::episode::{encode_episode_signature, encode_gist_signature};
use agidb_core::store::{Store, StoreConfig};
use agidb_core::types::{Episode, EpisodeId, Provenance, Query, Tier, TimeRange, Triple};
use agidb_core::unlearn::UnlearnTarget;
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

fn make_episode(
    id: u64,
    text: &str,
    subj: &str,
    pred: &str,
    obj: &str,
    at: chrono::DateTime<Utc>,
) -> Episode {
    let ep_id = EpisodeId::new(id);
    Episode {
        id: ep_id,
        text: text.into(),
        signature_offset: 0,
        gist_offset: 0,
        embedding_offset: 0,
        triples: vec![Triple {
            subject: subj.into(),
            predicate: pred.into(),
            object: obj.into(),
            confidence: 0.9,
            episode_id: ep_id,
        }],
        valid_time: TimeRange::point(at),
        t_tx_start: at,
        provenance: Provenance {
            source: "test".into(),
            ..Provenance::default()
        },
        confidence: 0.9,
        superseded_by: None,
    }
}

fn observe_with_encoding(store: &mut Store, ep: Episode) -> EpisodeId {
    let sig = encode_episode_signature(&ep.triples, Some(ep.valid_time.start));
    store.observe(ep, &sig).expect("observe")
}

// --- encoding determinism --------------------------------------------------

#[test]
fn encode_episode_signature_is_deterministic() {
    let ep = make_episode(
        1,
        "Sarah recommended Bawri",
        "Sarah",
        "recommended",
        "Bawri",
        t(2026, 5, 14),
    );
    let a = encode_episode_signature(&ep.triples, Some(ep.valid_time.start));
    let b = encode_episode_signature(&ep.triples, Some(ep.valid_time.start));
    assert_eq!(a, b, "same inputs must produce the same HV");
}

#[test]
fn encode_episode_signature_differs_for_different_objects() {
    let valid_from = t(2026, 5, 14);
    let a = make_episode(
        1,
        "Sarah recommended Bawri",
        "Sarah",
        "recommended",
        "Bawri",
        valid_from,
    );
    let b = make_episode(
        2,
        "Sarah recommended Olive",
        "Sarah",
        "recommended",
        "Olive",
        valid_from,
    );
    let sig_a = encode_episode_signature(&a.triples, Some(valid_from));
    let sig_b = encode_episode_signature(&b.triples, Some(valid_from));
    assert_ne!(
        sig_a, sig_b,
        "different objects must produce different signatures"
    );
}

#[test]
fn encode_gist_signature_is_case_insensitive() {
    let a = encode_gist_signature("Sarah recommended Bawri");
    let b = encode_gist_signature("sarah recommended bawri");
    assert_eq!(a, b, "gist tokenization is lower-cased");
}

// --- tier A: exact concept lookup ------------------------------------------

#[test]
fn tier_a_returns_episode_when_concept_token_in_cue() {
    let (mut store, _dir) = fresh_store();
    observe_with_encoding(
        &mut store,
        make_episode(
            1,
            "Sarah recommended Bawri",
            "Sarah",
            "recommended",
            "Bawri",
            t(2026, 5, 14),
        ),
    );
    observe_with_encoding(
        &mut store,
        make_episode(
            2,
            "Alice mentioned Olive",
            "Alice",
            "mentioned",
            "Olive",
            t(2026, 5, 14),
        ),
    );

    let r = store
        .recall(&Query::cue("what did Sarah say?"))
        .expect("recall");
    assert_eq!(
        r.tier_used,
        Tier::Exact,
        "matching concept token must land tier A"
    );
    assert!(r.matches.iter().any(|m| m.episode_id == EpisodeId::new(1)));
    assert!(
        r.matches.iter().all(|m| m.confidence >= 0.99),
        "tier A must return confidence ≈ 1.0"
    );
}

#[test]
fn tier_a_filters_by_as_of() {
    let (mut store, _dir) = fresh_store();
    let t1 = t(2026, 1, 1);
    let t2 = t1 + Duration::days(60);
    let id_a = observe_with_encoding(
        &mut store,
        make_episode(
            10,
            "Sarah recommended Bawri",
            "Sarah",
            "recommended",
            "Bawri",
            t1,
        ),
    );
    let id_b = observe_with_encoding(
        &mut store,
        make_episode(
            11,
            "Sarah recommended Olive",
            "Sarah",
            "recommended",
            "Olive",
            t2,
        ),
    );
    store.supersede(id_a, id_b).expect("supersede");

    let as_of_after = t2 + Duration::days(1);
    let r = store
        .recall(&Query::cue("Sarah").with_as_of(as_of_after))
        .expect("recall");
    assert!(r.matches.iter().any(|m| m.episode_id == id_b));
    assert!(
        !r.matches.iter().any(|m| m.episode_id == id_a),
        "superseded episode must be filtered out by as_of"
    );
}

// --- tier C / D fall-through -----------------------------------------------

#[test]
fn recall_falls_through_to_gist_when_no_concept_match() {
    let (mut store, _dir) = fresh_store();
    // Entity names (MainCafe, CentralPark) are disjoint from cue
    // tokens so tier A misses. The cue shares "cafe", "park",
    // "noon", "opens" with the episode text, so the token-level
    // inverted index (tier L) matches — the cascade short-circuits
    // at L before tier C. This is the new "better answer" path:
    // tier L is more precise than tier C for lexical overlap.
    observe_with_encoding(
        &mut store,
        make_episode(
            1,
            "the cafe near the park opens at noon every day",
            "MainCafe",
            "located_near",
            "CentralPark",
            t(2026, 5, 14),
        ),
    );

    let r = store
        .recall(&Query::cue("cafe park noon opens"))
        .expect("recall");
    assert_eq!(
        r.tier_used,
        Tier::Lexical,
        "tier L must fire on lexical token overlap"
    );
    assert!(
        !r.matches.is_empty(),
        "tier L must return at least one match"
    );
    assert!(
        r.matches
            .iter()
            .all(|m| m.confidence >= 0.55 && m.confidence <= 0.95),
        "tier L confidence must sit in [0.55, 0.95]"
    );
}

#[test]
fn recall_never_returns_empty_under_default_floor() {
    let (mut store, _dir) = fresh_store();
    observe_with_encoding(
        &mut store,
        make_episode(
            1,
            "the cat sat on the mat",
            "cat",
            "sat_on",
            "mat",
            t(2026, 5, 14),
        ),
    );

    let r = store
        .recall(&Query::cue("completely-unrelated-asdfghjkl-cue"))
        .expect("recall");
    assert!(
        !r.matches.is_empty(),
        "recall must never return empty under default tier_floor"
    );
    assert_eq!(r.tier_used, Tier::NearestNeighbor, "should land at tier D");
    assert!(
        r.matches.iter().all(|m| m.confidence <= 0.3),
        "tier D caps confidence at 0.3"
    );
}

// --- tier_floor ------------------------------------------------------------

#[test]
fn tier_floor_exact_skips_fuzzy_tiers() {
    let (mut store, _dir) = fresh_store();
    observe_with_encoding(
        &mut store,
        make_episode(
            1,
            "the cat sat on the mat",
            "cat",
            "sat_on",
            "mat",
            t(2026, 5, 14),
        ),
    );

    // No tier-A match for this cue; tier_floor=Exact disables fuzzy
    // tiers, so the result must be empty.
    let r = store
        .recall(&Query::cue("dogs and birds").with_tier_floor(Tier::Exact))
        .expect("recall");
    assert!(
        r.matches.is_empty(),
        "tier_floor=Exact with no concept match must return no rows"
    );
}

// --- k cap + min_confidence ------------------------------------------------

#[test]
fn recall_respects_k_cap() {
    let (mut store, _dir) = fresh_store();
    for i in 0..20 {
        observe_with_encoding(
            &mut store,
            make_episode(
                100 + i,
                "the cat sat on the mat",
                "cat",
                "sat_on",
                "mat",
                t(2026, 5, 14),
            ),
        );
    }
    let r = store.recall(&Query::cue("cat").with_k(5)).expect("recall");
    assert_eq!(r.matches.len(), 5, "k=5 must cap the result count");
}

#[test]
fn recall_respects_min_confidence() {
    let (mut store, _dir) = fresh_store();
    observe_with_encoding(
        &mut store,
        make_episode(
            1,
            "the cat sat on the mat",
            "cat",
            "sat_on",
            "mat",
            t(2026, 5, 14),
        ),
    );

    let r = store
        .recall(&Query::cue("unrelated-asdf").with_min_confidence(0.5))
        .expect("recall");
    assert!(
        r.matches.is_empty(),
        "min_confidence=0.5 must drop tier-D matches whose confidence is capped below 0.3"
    );
}

// --- synthetic smoke -------------------------------------------------------

#[test]
fn synthetic_100_episodes_recall_smoke() {
    let (mut store, _dir) = fresh_store();
    let people = ["alice", "bob", "carol", "dave", "eve"];
    let verbs = ["recommended", "disliked", "mentioned"];
    let places = ["bawri", "olive", "trishna", "pali", "mahesh"];

    let t0 = t(2026, 5, 14);
    let mut id = 1u64;
    for p in &people {
        for v in &verbs {
            for pl in &places {
                let text = format!("{p} {v} {pl}");
                observe_with_encoding(&mut store, make_episode(id, &text, p, v, pl, t0));
                id += 1;
            }
        }
    }
    // 5 * 3 * 5 = 75 episodes (under 100; keeps the test fast).

    // Query a known person → tier A returns all 15 of their episodes.
    let r = store
        .recall(&Query::cue("alice").with_k(50))
        .expect("recall");
    assert_eq!(r.tier_used, Tier::Exact);
    assert_eq!(
        r.matches.len(),
        15,
        "alice should appear in 15 episodes (3 verbs × 5 places)"
    );

    // Query a known place → tier A returns all 15 of those episodes
    // (because the place names are also stored as concept tokens via
    // the triple's object slot).
    let r = store
        .recall(&Query::cue("trishna").with_k(50))
        .expect("recall");
    assert_eq!(r.tier_used, Tier::Exact);
    assert_eq!(r.matches.len(), 15);

    // A vague cue → tier C/D fallback returns *something*.
    let r = store.recall(&Query::cue("dinner plans")).expect("recall");
    assert!(!r.matches.is_empty(), "fallback must produce something");
}

// --- tier B: structured similarity ------------------------------------------

#[test]
fn tier_b_matches_case_insensitive_entity_mentions() {
    // "sarah" (lowercase) misses tier A's case-sensitive concept
    // lookup but resolves case-insensitively in tier B, whose
    // structured cue signature overlaps the stored role-bound episode
    // signature. Distractor entities stay in the ≈0.5 noise band.
    let (mut store, _dir) = fresh_store();
    observe_with_encoding(
        &mut store,
        make_episode(
            1,
            "Sarah recommended Bawri",
            "Sarah",
            "recommended",
            "Bawri",
            t(2026, 5, 14),
        ),
    );
    for i in 0..10u64 {
        observe_with_encoding(
            &mut store,
            make_episode(
                100 + i,
                &format!("Visitor{i} toured Museum{i}"),
                &format!("Visitor{i}"),
                "toured",
                &format!("Museum{i}"),
                t(2026, 5, 14),
            ),
        );
    }

    let r = store.recall(&Query::cue("sarah")).expect("recall");
    assert_eq!(
        r.tier_used,
        Tier::Similarity,
        "lowercase entity cue must land tier B, got {:?} with {:?}",
        r.tier_used,
        r.matches
            .iter()
            .map(|m| (&m.text, m.confidence))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        r.matches[0].episode_id,
        EpisodeId::new(1),
        "the Sarah episode must rank first"
    );
    assert!(
        r.matches[0].confidence >= 0.6 && r.matches[0].confidence <= 0.95,
        "tier B confidence must sit in the [0.6, 0.95] band, got {}",
        r.matches[0].confidence
    );
    assert!(
        !r.matches.iter().any(|m| m.episode_id != EpisodeId::new(1)),
        "distractor entities must stay below the tier-B floor"
    );
}

#[test]
fn tier_b_falls_through_when_no_concept_resolves() {
    let (mut store, _dir) = fresh_store();
    observe_with_encoding(
        &mut store,
        make_episode(
            1,
            "the cafe near the park opens at noon every day",
            "MainCafe",
            "located_near",
            "CentralPark",
            t(2026, 5, 14),
        ),
    );
    // No cue token resolves to a concept even fuzzily. Tier A misses;
    // tier L *also* matches because "noon" / "opens" / "every" / "day"
    // appear in both cue and episode text. Tier L short-circuits the
    // cascade — that's the new better answer for lexical overlap.
    let r = store
        .recall(&Query::cue("opens at noon every day"))
        .expect("recall");
    assert_eq!(
        r.tier_used,
        Tier::Lexical,
        "tier L must short-circuit tier B on shared content tokens"
    );
    assert!(!r.matches.is_empty());
}

// --- scan directory: tombstones + reopen -------------------------------------

#[test]
fn tombstoned_episode_excluded_from_every_tier_and_restorable() {
    let (mut store, _dir) = fresh_store();
    let id = observe_with_encoding(
        &mut store,
        make_episode(
            1,
            "the cat sat on the mat",
            "cat",
            "sat_on",
            "mat",
            t(2026, 5, 14),
        ),
    );

    let report = store
        .unlearn(UnlearnTarget::Episode(id), "test forget")
        .expect("unlearn");
    let r = store.recall(&Query::cue("cat sat mat")).expect("recall");
    assert!(
        r.matches.is_empty(),
        "tombstoned episode must be invisible to tiers A-D"
    );

    store
        .restore_within_window(report.audit_event_id)
        .expect("restore");
    let r = store.recall(&Query::cue("cat sat mat")).expect("recall");
    assert!(
        r.matches.iter().any(|m| m.episode_id == id),
        "restored episode must be recallable again"
    );
}

#[test]
fn recall_works_after_reopen_from_persisted_gist_signatures() {
    let dir = TempDir::new().expect("tempdir");
    {
        let mut store = Store::open(StoreConfig::at(dir.path())).expect("open");
        observe_with_encoding(
            &mut store,
            make_episode(
                1,
                "the cafe near the park opens at noon every day",
                "MainCafe",
                "located_near",
                "CentralPark",
                t(2026, 5, 14),
            ),
        );
    }
    // Reopen — the scan directory must rebuild from disk and the gist
    // scan must run off the persisted HVs.
    let store = Store::open(StoreConfig::at(dir.path())).expect("reopen");
    let r = store
        .recall(&Query::cue("cafe park noon opens"))
        .expect("recall");
    // Reopen must rebuild the TOKENS table from EPISODES (same path
    // the scan directory rebuild uses); tier L fires on the lexical
    // overlap of the cue and the stored text.
    assert_eq!(r.tier_used, Tier::Lexical);
    assert!(!r.matches.is_empty());
}

#[test]
fn gist_hv_not_duplicated_when_signature_is_the_gist() {
    let (mut store, _dir) = fresh_store();
    // Extraction-less episode: caller passes the gist as the episode
    // signature → exactly one HV stored.
    let mut ep = make_episode(1, "raw text only", "x", "y", "z", t(2026, 5, 14));
    ep.triples = vec![];
    let gist = encode_gist_signature("raw text only");
    store.observe(ep, &gist).expect("observe");
    assert_eq!(store.stats().expect("stats").signatures, 1);

    // Structured episode: distinct structured + gist HVs → two more.
    observe_with_encoding(
        &mut store,
        make_episode(
            2,
            "Sarah recommended Bawri",
            "Sarah",
            "recommended",
            "Bawri",
            t(2026, 5, 14),
        ),
    );
    assert_eq!(store.stats().expect("stats").signatures, 3);
}
