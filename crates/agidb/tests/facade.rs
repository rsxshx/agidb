//! Integration test for the `Agidb` facade: observe → recall →
//! consolidate → stats round-trip on a temp store with the null
//! extractor (deterministic, no network).
//!
//! v0.2 — `feat/temporal-retrieval`: timeline() round-trip test at
//! the bottom of the file. Exercises the new
//! `Agidb::timeline(subject, from, to, limit)` convenience path
//! end-to-end through the public facade.

use agidb::{
    Agidb, AgidbConfig, Entity, ExtractContext, ExtractedTriple, Extraction, ExtractorSetup, Query,
    TextExtractor, Tier,
};
use chrono::{Duration, TimeZone, Utc};
use std::sync::Arc;

type CoreResult<T> = agidb_core::Result<T>;

/// Deterministic extractor that synthesizes one `(subject, predicate,
/// object)` triple per observation, where `subject` is the first
/// whitespace-delimited token of the text. Used to seed concept
/// links for the timeline test without pulling in GLiNER.
#[derive(Debug)]
struct SubjectFirstExtractor;

impl TextExtractor for SubjectFirstExtractor {
    fn extract(&self, text: &str, _ctx: &ExtractContext) -> CoreResult<Extraction> {
        let subject = text
            .split_whitespace()
            .next()
            .unwrap_or("unknown")
            .to_string();
        let object = text.to_string();
        let span_len = subject.len();
        let raw = Entity {
            text: subject.clone(),
            entity_type: "unknown".into(),
            span: (0, span_len),
            confidence: 1.0,
            canonical_name: Some(subject.clone()),
        };
        Ok(Extraction {
            triples: vec![ExtractedTriple {
                subject: subject.clone(),
                predicate: "did".into(),
                object,
                confidence: 0.9,
            }],
            valid_time: None,
            raw_entities: vec![raw],
        })
    }
}

fn t(year: i32, month: u32, day: u32) -> chrono::DateTime<Utc> {
    Utc.with_ymd_and_hms(year, month, day, 0, 0, 0).unwrap()
}

#[tokio::test]
async fn observe_recall_consolidate_round_trip() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cfg = AgidbConfig::new(dir.path()).with_extractor(ExtractorSetup::Null);
    let db = Agidb::open_with(cfg).await.expect("open");

    // text-only episodes still get gist signatures → tier C/D recall.
    let id1 = db
        .observe("Sarah recommended Bawri in Bandra")
        .await
        .unwrap();
    let _id2 = db
        .observe("Sarah said Bawri is a thai place")
        .await
        .unwrap();
    assert_eq!(id1.raw(), 1);

    // constitution article VI: recall never returns the empty set.
    let r = db
        .recall_cue("what thai place did sarah mention?")
        .await
        .unwrap();
    assert!(!r.matches.is_empty(), "recall must never return empty");
    // elapsed_ms is a u32 wall-clock measurement.
    let _ = r.elapsed_ms;

    // consolidate is idempotent and safe on a tiny store.
    let c = db.consolidate().await.unwrap();
    assert_eq!(c.episodes_scanned, 2);

    // stats reflect what was written.
    let s = db.stats().await.unwrap();
    assert_eq!(s.episodes, 2);
    assert_eq!(s.signatures, 2);
    assert_eq!(s.consolidation_passes, 1);
}

#[tokio::test]
async fn get_episode_and_list_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = AgidbConfig::new(dir.path()).with_extractor(ExtractorSetup::Null);
    let db = Agidb::open_with(cfg).await.unwrap();

    db.observe("alice likes rust").await.unwrap();
    db.observe("bob likes rust").await.unwrap();

    let got = db.get_episode(1).await.unwrap().expect("ep1 exists");
    assert_eq!(got.text, "alice likes rust");

    let listed = db.list_episodes(10).await.unwrap();
    assert_eq!(listed.len(), 2);
    assert_eq!(listed[0].id.raw(), 1);
}

#[tokio::test]
async fn export_import_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = AgidbConfig::new(dir.path()).with_extractor(ExtractorSetup::Null);
    let db = Agidb::open_with(cfg).await.unwrap();
    db.observe("a fact about the world").await.unwrap();
    db.observe("another fact").await.unwrap();

    let path = dir.path().join("dump.jsonl");
    db.export_jsonl(&path).await.unwrap();

    let dir2 = tempfile::tempdir().unwrap();
    let cfg2 = AgidbConfig::new(dir2.path()).with_extractor(ExtractorSetup::Null);
    let db2 = Agidb::open_with(cfg2).await.unwrap();
    let n = db2.import_jsonl(&path).await.unwrap();
    assert_eq!(n, 2);
    let s = db2.stats().await.unwrap();
    assert_eq!(s.episodes, 2);
}

#[tokio::test]
async fn tier_floor_caps_the_cascade() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = AgidbConfig::new(dir.path()).with_extractor(ExtractorSetup::Null);
    let db = Agidb::open_with(cfg).await.unwrap();
    db.observe("something unrelated to the cue").await.unwrap();

    let q = Query::cue("zzz no match").with_tier_floor(Tier::Gist);
    let r = db.recall(q).await.unwrap();
    // floor at Gist forbids the NearestNeighbor fallback; with no gist
    // match, the cascade returns no matches (the never-empty guarantee
    // only holds under the default NearestNeighbor floor).
    assert!(r.matches.is_empty() || r.tier_used.depth() <= Tier::Gist.depth());
}

// ============================================================================
// v0.2 — temporal retrieval (`feat/temporal-retrieval`): timeline() end-to-end
// ============================================================================

#[tokio::test]
async fn timeline_returns_episodes_in_chronological_order() {
    // Seed via a low-level Store so we control each episode's
    // `valid_time` directly. The facade's `observe` would use
    // `Utc::now()`, which is fine for the real timeline use case
    // but awkward for a deterministic test.
    use agidb_core::episode::encode_episode_signature;
    use agidb_core::store::{Store, StoreConfig};
    use agidb_core::types::{Episode, EpisodeId, Provenance, TimeRange, Triple};

    let dir = tempfile::tempdir().expect("tempdir");
    let mut store = Store::open(StoreConfig::at(dir.path())).expect("open");

    // Seed 3 episodes: alice on Mon/Tue/Wed, bob on Thu.
    let alice_id_1 = store
        .observe(
            Episode {
                id: EpisodeId::new(1),
                text: "alice finished the wireframe on Monday".into(),
                signature_offset: 0,
                gist_offset: 0,
                embedding_offset: 0,
                triples: vec![Triple {
                    subject: "alice".into(),
                    predicate: "did".into(),
                    object: "wireframe".into(),
                    confidence: 0.9,
                    episode_id: EpisodeId::new(1),
                }],
                valid_time: TimeRange::point(t(2026, 5, 11)),
                t_tx_start: t(2026, 5, 11),
                provenance: Provenance::default(),
                confidence: 0.9,
                superseded_by: None,
            },
            &encode_episode_signature(
                &[Triple {
                    subject: "alice".into(),
                    predicate: "did".into(),
                    object: "wireframe".into(),
                    confidence: 0.9,
                    episode_id: EpisodeId::new(1),
                }],
                Some(t(2026, 5, 11)),
            ),
        )
        .expect("observe");

    let _alice_id_2 = store
        .observe(
            Episode {
                id: EpisodeId::new(2),
                text: "alice reviewed the wireframe on Tuesday".into(),
                signature_offset: 0,
                gist_offset: 0,
                embedding_offset: 0,
                triples: vec![Triple {
                    subject: "alice".into(),
                    predicate: "did".into(),
                    object: "wireframe".into(),
                    confidence: 0.9,
                    episode_id: EpisodeId::new(2),
                }],
                valid_time: TimeRange::point(t(2026, 5, 12)),
                t_tx_start: t(2026, 5, 12),
                provenance: Provenance::default(),
                confidence: 0.9,
                superseded_by: None,
            },
            &encode_episode_signature(
                &[Triple {
                    subject: "alice".into(),
                    predicate: "did".into(),
                    object: "wireframe".into(),
                    confidence: 0.9,
                    episode_id: EpisodeId::new(2),
                }],
                Some(t(2026, 5, 12)),
            ),
        )
        .expect("observe");

    let _bob_id = store
        .observe(
            Episode {
                id: EpisodeId::new(3),
                text: "bob sent the pricing quote on Thursday".into(),
                signature_offset: 0,
                gist_offset: 0,
                embedding_offset: 0,
                triples: vec![Triple {
                    subject: "bob".into(),
                    predicate: "did".into(),
                    object: "quote".into(),
                    confidence: 0.9,
                    episode_id: EpisodeId::new(3),
                }],
                valid_time: TimeRange::point(t(2026, 5, 14)),
                t_tx_start: t(2026, 5, 14),
                provenance: Provenance::default(),
                confidence: 0.9,
                superseded_by: None,
            },
            &encode_episode_signature(
                &[Triple {
                    subject: "bob".into(),
                    predicate: "did".into(),
                    object: "quote".into(),
                    confidence: 0.9,
                    episode_id: EpisodeId::new(3),
                }],
                Some(t(2026, 5, 14)),
            ),
        )
        .expect("observe");

    // Full-week timeline, no subject filter → all 3 episodes in
    // chronological order.
    let all = store
        .list_episodes_in_range(t(2026, 5, 11), t(2026, 5, 16), 100)
        .expect("list");
    assert_eq!(all.len(), 3);
    assert!(all
        .windows(2)
        .all(|w| w[0].valid_time.start <= w[1].valid_time.start));

    // limit truncates.
    let two = store
        .list_episodes_in_range(t(2026, 5, 11), t(2026, 5, 16), 2)
        .expect("list");
    assert_eq!(two.len(), 2);
    assert_eq!(two[0].id, alice_id_1);

    // `_alice_id_2` is referenced to silence the unused warning. The
    // chronological-order assertion above checks `id` order only
    // indirectly (via valid_time.start), so we use the variable to
    // keep the test compileable if we add more granular checks later.
    let _ = _alice_id_2;
    let _ = _bob_id;
}

#[tokio::test]
async fn timeline_with_subject_filter_via_facade() {
    // End-to-end via the public `Agidb::timeline` path. The
    // SubjectFirstExtractor creates a concept per subject token
    // ("alice" / "bob") so the subject filter has something to
    // match against. We seed 3 alice + 2 bob episodes at
    // `Utc::now()` (the facade uses now), then ask for "alice"
    // episodes in a wide window. We can't pin exact dates this
    // way, but the *filtering* is what we're testing — the result
    // set must be all-alice, never bob.
    let dir = tempfile::tempdir().expect("tempdir");
    let extractor: Arc<dyn agidb::TextExtractor + Send + Sync> = Arc::new(SubjectFirstExtractor);
    let cfg = AgidbConfig::new(dir.path()).with_extractor(ExtractorSetup::Custom(extractor));
    let db = Agidb::open_with(cfg).await.expect("open");

    db.observe("alice finished the wireframe").await.unwrap();
    db.observe("alice reviewed the wireframe").await.unwrap();
    db.observe("alice replied with feedback").await.unwrap();
    db.observe("bob sent the pricing quote").await.unwrap();
    db.observe("bob closed the deal").await.unwrap();

    // Wide window covering now±1 day.
    let now = Utc::now();
    let timeline = db
        .timeline(
            Some("alice"),
            now - Duration::days(1),
            now + Duration::days(1),
            100,
        )
        .await
        .expect("timeline");

    assert_eq!(
        timeline.len(),
        3,
        "alice must surface 3 episodes, none of bob's"
    );
    assert!(timeline.iter().all(|ep| ep.text.starts_with("alice")));

    // Unknown subject returns empty (no error).
    let empty = db
        .timeline(Some("nobody"), now - Duration::days(1), now, 10)
        .await
        .expect("timeline");
    assert!(empty.is_empty());
}
