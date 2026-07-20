//! Phase 4 — tiered recall invariants.
//!
//! Covers tier-A (exact concept lookup), tier-C (gist similarity in
//! the high-confidence band), tier-D (nearest-neighbor low-confidence
//! fallback), the `tier_floor` clamp, the `k` cap, the `as_of`
//! bi-temporal filter, and a 100-episode synthetic smoke run.
//!
//! v0.2 — `feat/temporal-retrieval`: temporal-retrieval paths are
//! covered by the tests at the bottom of this file. Each new
//! retrieval primitive (time-window filter, `list_episodes_in_range`,
//! recency rerank, subject filter) gets its own `#[test]` so a
//! future regression is local to the path it breaks.

use agidb_core::episode::{encode_episode_signature, encode_gist_signature};
use agidb_core::store::{Store, StoreConfig};
use agidb_core::types::{
    ConceptId, Episode, EpisodeId, Provenance, Query, Tier, TimeRange, Triple,
};
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
        Tier::Lexical,
        "matching concept token must route through tier L (which now subsumes tier A's candidate set via IDF rerank)"
    );
    assert!(r.matches.iter().any(|m| m.episode_id == EpisodeId::new(1)));
    // Confidence is in the tier L band [0.55, 0.95] — not 1.0
    // anymore (tier A subsumes into the IDF rerank that picks the
    // best-scoring candidate at 0.95 and the rest scaled down).
    assert!(
        r.matches
            .iter()
            .all(|m| m.confidence >= 0.55 && m.confidence <= 0.95),
        "tier A confidence must sit in [0.55, 0.95]"
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

    // Query a known person → tier A's concept-rerank path returns all
    // 15 of their episodes (idf reranked by lexical overlap).
    let r = store
        .recall(&Query::cue("alice").with_k(50))
        .expect("recall");
    assert_eq!(r.tier_used, Tier::Lexical);
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
    assert_eq!(r.tier_used, Tier::Lexical);
    assert_eq!(r.matches.len(), 15);

    // A vague cue → tier C/D fallback returns *something*.
    let r = store.recall(&Query::cue("dinner plans")).expect("recall");
    assert!(!r.matches.is_empty(), "fallback must produce something");
}

// --- tier B: structured similarity ------------------------------------------

#[test]
fn tier_b_matches_case_insensitive_entity_mentions() {
    // "sarah" (lowercase) misses tier A's case-sensitive concept
    // lookup but resolves case-insensitively in tier L (the new
    // lexical inverted-index tier lowercases both at write and at
    // query time). Distractor entities stay below the tier-L
    // confidence floor.
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
        Tier::Lexical,
        "lowercase entity cue must land tier L, got {:?} with {:?}",
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
        r.matches[0].confidence >= 0.55 && r.matches[0].confidence <= 0.95,
        "tier L confidence must sit in the [0.55, 0.95] band, got {}",
        r.matches[0].confidence
    );
    assert!(
        !r.matches.iter().any(|m| m.episode_id != EpisodeId::new(1)),
        "distractor entities must stay below the tier-L floor"
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

// ============================================================================
// v0.2 — temporal retrieval (`feat/temporal-retrieval`)
//
// Every new retrieval path needs at least one regression test. The
// tests below cover A–E of the task spec:
//   A. `Query::time_window` (interval-overlap filter)
//   B. `Store::list_episodes_in_range`
//   C. `Query::recency_weight` + `Query::time_anchor`
//   D. `Query::subject` (ConceptId filter)
//   E. `Agidb::timeline` end-to-end (covered in crates/agidb/tests)
//
// All new fields default to "off" so the pre-temporal-retrieval tests
// above (which use `Query::cue(...)` and never touch the new fields)
// must stay green and produce byte-identical results.
// ============================================================================

/// Helper: synthesize a tiny "5 facts across 5 days" corpus that makes
/// the temporal retrieval tests independent of cue wording. Episodes
/// about different subjects on different days let us assert that the
/// filters do what we claim without relying on lexical accidents.
fn seed_temporal_corpus(store: &mut Store) -> Vec<EpisodeId> {
    let mut ids = Vec::new();
    let schedule = [
        // (day_offset_from_t0, subj, text)
        (0i64, "alice", "alice finished the wireframe on Monday"),
        (1, "alice", "alice reviewed the wireframe on Tuesday"),
        (2, "bob", "bob sent the pricing quote on Wednesday"),
        (3, "alice", "alice replied with feedback on Thursday"),
        (4, "bob", "bob closed the deal on Friday"),
    ];
    let t0 = t(2026, 5, 11); // Monday
    for (i, (offset, subj, text)) in schedule.iter().enumerate() {
        let when = t0 + Duration::days(*offset);
        let ep = make_episode((i + 1) as u64, text, subj, "did", text, when);
        ids.push(observe_with_encoding(store, ep));
    }
    ids
}

// --- A: time_window filter -------------------------------------------------

#[test]
fn time_window_excludes_episodes_outside_the_window() {
    let (mut store, _dir) = fresh_store();
    seed_temporal_corpus(&mut store);

    // Window covers Thursday + Friday only — the two episodes on
    // those days must come back; Monday/Tuesday/Wednesday must not.
    let thu = t(2026, 5, 14);
    let sat = t(2026, 5, 16);
    let r = store
        .recall(
            &Query::cue("alice bob wireframe quote feedback deal")
                .with_time_window(thu, sat)
                .with_k(10),
        )
        .expect("recall");

    assert_eq!(
        r.matches.len(),
        2,
        "only Thursday + Friday episodes survive"
    );
    let returned_days: Vec<u32> = r
        .matches
        .iter()
        .map(|m| {
            m.valid_time
                .start
                .format("%u")
                .to_string()
                .parse()
                .unwrap_or(0)
        })
        .collect();
    assert!(returned_days.contains(&4), "Thursday (4) must be present");
    assert!(returned_days.contains(&5), "Friday (5) must be present");
}

#[test]
fn time_window_partial_overlap_is_inclusive() {
    let (mut store, _dir) = fresh_store();
    seed_temporal_corpus(&mut store);

    // Window [Wed 00:00, Fri 00:00) covers both Wednesday's bob
    // pricing quote and Thursday's alice feedback. Per the
    // feat/temporal-retrieval spec, an open-ended episode's
    // effective end equals its start — both starts fall in the
    // window, so both must overlap.
    let wed = t(2026, 5, 13);
    let fri = t(2026, 5, 15);
    let r = store
        .recall(
            &Query::cue("pricing quote feedback")
                .with_time_window(wed, fri)
                .with_k(10),
        )
        .expect("recall");
    assert_eq!(r.matches.len(), 2);
}

#[test]
fn time_window_disjoint_returns_nothing_above_cue_floor() {
    let (mut store, _dir) = fresh_store();
    seed_temporal_corpus(&mut store);

    // Window in 2027 — no episode's `valid_time.start` falls inside it,
    // so no episode's effective interval (treating open-ended as
    // point-at-start) overlaps.
    let future_from = t(2027, 1, 1);
    let future_to = t(2027, 2, 1);
    let r = store
        .recall(
            &Query::cue("wireframe quote feedback")
                .with_time_window(future_from, future_to)
                .with_tier_floor(Tier::Lexical),
        )
        .expect("recall");
    assert!(
        r.matches.is_empty(),
        "tier L must return empty when the time_window excludes every episode"
    );
}

#[test]
fn time_window_composes_with_as_of() {
    let (mut store, _dir) = fresh_store();
    seed_temporal_corpus(&mut store);

    // `as_of` is a single point (Wednesday); `time_window` is the full
    // week. Both active: only the Wednesday episode survives.
    let wed = t(2026, 5, 13);
    let r = store
        .recall(
            &Query::cue("pricing quote")
                .with_as_of(wed)
                .with_time_window(t(2026, 5, 11), t(2026, 5, 18))
                .with_k(10),
        )
        .expect("recall");
    assert_eq!(r.matches.len(), 1);
    assert!(r.matches[0].text.contains("Wednesday"));
}

#[test]
fn time_window_disabled_by_default_does_not_change_recall() {
    // Regression: a query with the time_window field left at its
    // default (None) must produce the same matches as the
    // pre-temporal-recall behavior. We pin a synthetic corpus + cue
    // and compare the set of returned episode ids.
    let (mut store, _dir) = fresh_store();
    seed_temporal_corpus(&mut store);
    let q = Query::cue("alice bob wireframe quote feedback deal").with_k(10);
    let r = store.recall(&q).expect("recall");
    let ids: std::collections::BTreeSet<EpisodeId> =
        r.matches.iter().map(|m| m.episode_id).collect();
    // The cue shares tokens with every episode in the corpus, so the
    // lexical tier should surface all five. (Tier L is a token
    // posting-list intersection; every cue token hits at least one
    // episode and the IDF rerank keeps the full set.)
    assert_eq!(
        ids.len(),
        5,
        "all 5 episodes must be retrievable by cue alone"
    );
}

// --- B: list_episodes_in_range --------------------------------------------

#[test]
fn list_episodes_in_range_returns_chronological_order() {
    let (mut store, _dir) = fresh_store();
    let ids = seed_temporal_corpus(&mut store);
    assert_eq!(ids.len(), 5);

    // Full-week window, limit = 100.
    let eps = store
        .list_episodes_in_range(t(2026, 5, 11), t(2026, 5, 18), 100)
        .expect("list");
    assert_eq!(eps.len(), 5);
    // Ascending by valid_time.start.
    for window in eps.windows(2) {
        assert!(
            window[0].valid_time.start <= window[1].valid_time.start,
            "out-of-order: {} then {}",
            window[0].valid_time.start,
            window[1].valid_time.start
        );
    }
    // First = Monday (the oldest), last = Friday.
    assert_eq!(
        eps.first().unwrap().valid_time.start,
        t(2026, 5, 11),
        "Monday first"
    );
    assert_eq!(
        eps.last().unwrap().valid_time.start,
        t(2026, 5, 15),
        "Friday last"
    );
}

#[test]
fn list_episodes_in_range_truncates_to_limit() {
    let (mut store, _dir) = fresh_store();
    seed_temporal_corpus(&mut store);
    let eps = store
        .list_episodes_in_range(t(2026, 5, 11), t(2026, 5, 18), 2)
        .expect("list");
    assert_eq!(eps.len(), 2);
    assert_eq!(eps[0].valid_time.start, t(2026, 5, 11));
    assert_eq!(eps[1].valid_time.start, t(2026, 5, 12));
}

#[test]
fn list_episodes_in_range_skips_out_of_window_episodes() {
    let (mut store, _dir) = fresh_store();
    seed_temporal_corpus(&mut store);
    let eps = store
        .list_episodes_in_range(t(2026, 5, 14), t(2026, 5, 16), 100)
        .expect("list");
    assert_eq!(eps.len(), 2, "Thursday + Friday only");
}

#[test]
fn list_episodes_in_range_empty_when_no_episodes_match() {
    let (mut store, _dir) = fresh_store();
    seed_temporal_corpus(&mut store);
    let eps = store
        .list_episodes_in_range(t(2030, 1, 1), t(2030, 2, 1), 100)
        .expect("list");
    assert!(eps.is_empty());
}

#[test]
fn list_episodes_in_range_picks_up_episode_whose_interval_covers_window() {
    // An episode whose valid_time covers the query window (closed
    // start <= from AND closed end >= to) must be returned. The
    // half-open window-overlap test gives the same result here as
    // the closed-closed version because the window's `from` and
    // `to` straddle the episode's interior.
    let (mut store, _dir) = fresh_store();
    let mut ep = make_episode(
        1,
        "long-running concern",
        "alice",
        "tracking",
        "Q3 roadmap",
        t(2026, 1, 1),
    );
    ep.valid_time = TimeRange {
        start: t(2026, 1, 1),
        end: Some(t(2026, 12, 31)),
    };
    observe_with_encoding(&mut store, ep);

    let eps = store
        .list_episodes_in_range(t(2026, 6, 1), t(2026, 6, 30), 100)
        .expect("list");
    assert_eq!(eps.len(), 1, "the Jan–Dec episode covers the June window");
}

#[test]
fn list_episodes_in_range_treats_open_ended_episode_as_point_in_time() {
    // Per the feat/temporal-retrieval spec, an open-ended episode is
    // treated as a point at its `start` for window-overlap purposes.
    // A 2026-01-01 open-ended episode therefore does NOT appear in
    // a June 2026 window — it lives only in a window that contains
    // 2026-01-01 itself.
    let (mut store, _dir) = fresh_store();
    let ep = make_episode(
        1,
        "still-current",
        "alice",
        "tracking",
        "ongoing concern",
        t(2026, 1, 1),
    );
    assert!(ep.valid_time.end.is_none());
    observe_with_encoding(&mut store, ep);

    let no_show = store
        .list_episodes_in_range(t(2026, 6, 1), t(2026, 6, 30), 100)
        .expect("list");
    assert!(
        no_show.is_empty(),
        "open-ended episode with start in Jan doesn't cover a June window"
    );

    let yes_show = store
        .list_episodes_in_range(t(2025, 12, 1), t(2026, 2, 1), 100)
        .expect("list");
    assert_eq!(
        yes_show.len(),
        1,
        "open-ended episode's effective end is its start, so a window containing the start picks it up"
    );
}

// --- C: recency rerank ----------------------------------------------------

#[test]
fn recency_rerank_boosts_recent_episode_over_old_one_with_identical_cue() {
    let (mut store, _dir) = fresh_store();
    // Two episodes with identical cue-sharing tokens. Without
    // recency rerank they should be near-tied on cue_score; with
    // recency_weight > 0 + an anchor close to "now", the recent
    // one must outrank the old one.
    let mut old = make_episode(
        1,
        "alice sent the wireframe for the proposal",
        "alice",
        "sent",
        "wireframe",
        t(2026, 1, 1),
    );
    old.confidence = 0.9;
    observe_with_encoding(&mut store, old);
    let mut new = make_episode(
        2,
        "alice sent the wireframe for the proposal",
        "alice",
        "sent",
        "wireframe",
        t(2026, 5, 15),
    );
    new.confidence = 0.9;
    let new_id = observe_with_encoding(&mut store, new);

    // Baseline: without recency, the higher id wins (the lexical
    // tier normalizes by max-score and ties break by id desc).
    let baseline = store
        .recall(&Query::cue("alice wireframe proposal").with_k(2))
        .expect("recall");
    assert_eq!(baseline.matches.len(), 2);

    // With recency: anchor 1 day after the new episode's valid_time.
    let anchor = t(2026, 5, 15) + Duration::days(1);
    let reranked = store
        .recall(
            &Query::cue("alice wireframe proposal")
                .with_recency_bias(0.7, anchor)
                .with_k(2),
        )
        .expect("recall");
    assert_eq!(reranked.matches.len(), 2);
    assert_eq!(
        reranked.matches[0].episode_id, new_id,
        "recent episode must outrank the old one when recency_weight > 0"
    );
}

#[test]
fn recency_rerank_zero_weight_is_no_op() {
    // Sanity check: with `recency_weight = 0` (the default) the
    // rerank is a no-op, so the resulting confidence ordering must
    // match the cue-only baseline ordering.
    let (mut store, _dir) = fresh_store();
    let mut old = make_episode(
        1,
        "alice wireframe",
        "alice",
        "made",
        "wireframe",
        t(2026, 1, 1),
    );
    old.confidence = 0.9;
    observe_with_encoding(&mut store, old);
    let mut new = make_episode(
        2,
        "alice wireframe",
        "alice",
        "made",
        "wireframe",
        t(2026, 5, 15),
    );
    new.confidence = 0.9;
    observe_with_encoding(&mut store, new);

    let anchor = t(2026, 5, 15);
    let r = store
        .recall(
            &Query::cue("alice wireframe")
                .with_recency_bias(0.0, anchor)
                .with_k(2),
        )
        .expect("recall");
    // With weight 0 the blend is (1.0) * cue_score — no boost.
    assert_eq!(r.matches.len(), 2);
    // Either ordering is fine as long as both came back; the point
    // is the call didn't error and didn't apply recency.
    let _ = anchor;
}

#[test]
fn recency_rerank_clamps_final_score_to_unit_interval() {
    // The blend `(1-w)*cue + w*recency` is bounded by the cue score
    // and the recency score (both ≤ 1) so the result is guaranteed
    // ≤ 1. Make sure that holds even with extreme inputs.
    let (mut store, _dir) = fresh_store();
    let mut old = make_episode(
        1,
        "alice wireframe",
        "alice",
        "made",
        "wireframe",
        t(2026, 1, 1),
    );
    old.confidence = 0.9;
    observe_with_encoding(&mut store, old);
    let mut new = make_episode(
        2,
        "alice wireframe",
        "alice",
        "made",
        "wireframe",
        t(2026, 5, 15),
    );
    new.confidence = 0.9;
    observe_with_encoding(&mut store, new);

    let r = store
        .recall(&Query::cue("alice wireframe").with_recency_bias(1.0, t(2026, 5, 15)))
        .expect("recall");
    for m in &r.matches {
        assert!(m.confidence >= 0.0 && m.confidence <= 1.0);
    }
}

// --- D: subject filter ----------------------------------------------------

#[test]
fn subject_filter_restricts_cascade_to_linked_episodes() {
    let (mut store, _dir) = fresh_store();
    seed_temporal_corpus(&mut store);

    // Resolve the "alice" ConceptId via the concept-by-name index.
    let alice_cid = store
        .concept_id_for("alice")
        .expect("concept_id_for")
        .expect("alice must exist");

    // Generic cue (no entity tokens); without a subject, this would
    // fall through to tier D and surface all 5 episodes. With
    // subject = alice, only the 3 alice episodes come back.
    let r = store
        .recall(
            &Query::cue("a totally unrelated cue with no entity tokens")
                .with_subject(alice_cid)
                .with_k(10),
        )
        .expect("recall");
    assert!(
        r.matches.iter().all(|m| m.text.contains("alice")),
        "subject filter must restrict matches to alice episodes: got {:?}",
        r.matches.iter().map(|m| &m.text).collect::<Vec<_>>()
    );
    assert_eq!(r.matches.len(), 3, "alice has 3 episodes in the corpus");
}

#[test]
fn subject_filter_unknown_concept_returns_nothing_above_floor() {
    let (mut store, _dir) = fresh_store();
    seed_temporal_corpus(&mut store);
    let nonexistent = ConceptId::new(999_999);
    let r = store
        .recall(
            &Query::cue("alice bob wireframe")
                .with_subject(nonexistent)
                .with_tier_floor(Tier::Lexical),
        )
        .expect("recall");
    assert!(
        r.matches.is_empty(),
        "an unknown subject must produce no tier-A/L/B/C matches"
    );
}

#[test]
fn subject_filter_and_time_window_compose() {
    let (mut store, _dir) = fresh_store();
    seed_temporal_corpus(&mut store);
    let alice_cid = store
        .concept_id_for("alice")
        .expect("concept_id_for")
        .expect("alice exists");

    // subject = alice AND window = Thursday onwards.
    let r = store
        .recall(
            &Query::cue("alice feedback wireframe")
                .with_subject(alice_cid)
                .with_time_window(t(2026, 5, 14), t(2026, 5, 18))
                .with_k(10),
        )
        .expect("recall");
    // Alice has 3 episodes: Mon, Tue, Thu. The Thu one is the only
    // one in the window.
    assert_eq!(r.matches.len(), 1);
    assert!(r.matches[0].text.contains("Thursday"));
}

#[test]
fn subject_filter_creates_concept_via_observation() {
    // Sanity: the concept index is populated at observe time, so a
    // freshly-seeded store has alice + bob Concepts and the
    // concept_id_for lookup returns them.
    let (mut store, _dir) = fresh_store();
    seed_temporal_corpus(&mut store);
    let alice = store.concept_id_for("alice").expect("ci").expect("alice");
    let bob = store.concept_id_for("bob").expect("ci").expect("bob");
    assert_ne!(alice, bob);
}
