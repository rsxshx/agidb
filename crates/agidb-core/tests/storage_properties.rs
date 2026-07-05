//! Phase 2 RED suite — the invariants `Store` + `SignatureFile` must
//! satisfy.
//!
//! Scaffold state: every test in this file panics from `todo!()` inside
//! the Store / SignatureFile API. As phase 2 implementation lands, the
//! panics turn into asserts; the asserts turn green.
//!
//! Coverage:
//! - `Store::open` round-trips an empty store
//! - `SignatureFile::append` returns an offset that `read` accepts
//! - Signatures survive a close + reopen
//! - `Store::observe` → `Store::get_episode` round-trips an Episode
//! - Concepts resolve by canonical name *and* alias to the same id
//! - Bi-temporal supersession closes the older valid_time and writes
//!   the `superseded_by` link in one transaction
//! - Tier-A exact recall respects `as_of` time
//! - `export_jsonl` → fresh store + `import_jsonl` round-trips

use agidb_core::error::AgidbError;
use agidb_core::hdc::HV;
use agidb_core::signatures::SignatureFile;
use agidb_core::store::{Store, StoreConfig};
use agidb_core::types::{Episode, EpisodeId, Provenance, TimeRange, Triple};
use chrono::{Duration, TimeZone, Utc};
use tempfile::TempDir;

// --- helpers ---------------------------------------------------------------

fn fresh_store() -> (Store, TempDir) {
    let dir = TempDir::new().expect("tempdir create");
    let store = Store::open(StoreConfig::at(dir.path())).expect("open");
    (store, dir)
}

fn fresh_signature_file() -> (SignatureFile, TempDir) {
    let dir = TempDir::new().expect("tempdir create");
    let sf = SignatureFile::open(dir.path().join("signatures.dat")).expect("open sigfile");
    (sf, dir)
}

fn sample_episode(id: u64, valid_from: chrono::DateTime<Utc>) -> Episode {
    Episode {
        id: EpisodeId::new(id),
        text: format!("test observation #{id}"),
        signature_offset: 0,
        gist_offset: 0,
        triples: vec![Triple {
            subject: "Sarah".into(),
            predicate: "recommended".into(),
            object: "Bawri".into(),
            confidence: 0.92,
            episode_id: EpisodeId::new(id),
        }],
        valid_time: TimeRange::point(valid_from),
        t_tx_start: valid_from,
        provenance: Provenance {
            source: "test".into(),
            ..Provenance::default()
        },
        confidence: 0.92,
        superseded_by: None,
    }
}

// --- store open / round-trip ----------------------------------------------

#[test]
fn open_creates_an_empty_store() {
    let (store, _dir) = fresh_store();
    // No assertion beyond construction — phase 2 ensures `open` succeeds
    // on a new path and creates the manifest table.
    drop(store);
}

#[test]
fn episode_roundtrip_through_observe_then_get() {
    let (mut store, _dir) = fresh_store();
    let ep = sample_episode(1, Utc.with_ymd_and_hms(2026, 5, 14, 0, 0, 0).unwrap());
    let hv = HV::from_name(&ep.text);

    let assigned = store.observe(ep.clone(), &hv).expect("observe");
    let fetched = store.get_episode(assigned).expect("get").expect("present");

    assert_eq!(fetched.text, ep.text);
    assert_eq!(fetched.triples, ep.triples);
    assert_eq!(fetched.valid_time, ep.valid_time);
}

// --- signature file --------------------------------------------------------

#[test]
fn signature_file_append_then_read_returns_same_bits() {
    let (mut sf, _dir) = fresh_signature_file();
    let hv = HV::from_name("Bawri");
    let offset = sf.append(&hv).expect("append");
    let back = sf.read(offset).expect("read");
    assert_eq!(back, hv);
}

#[test]
fn signature_file_offsets_are_stable_across_reopen() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("signatures.dat");

    let a = HV::from_name("a");
    let b = HV::from_name("b");
    let off_a;
    let off_b;
    {
        let mut sf = SignatureFile::open(&path).expect("open");
        off_a = sf.append(&a).expect("append a");
        off_b = sf.append(&b).expect("append b");
        sf.flush().expect("flush");
    }
    let sf = SignatureFile::open(&path).expect("reopen");
    assert_eq!(sf.read(off_a).unwrap(), a, "a survives reopen");
    assert_eq!(sf.read(off_b).unwrap(), b, "b survives reopen");
}

#[test]
fn signature_file_out_of_bounds_offset_is_typed_error() {
    let (sf, _dir) = fresh_signature_file();
    match sf.read(999_999_999) {
        Err(AgidbError::SignatureOutOfBounds { .. }) => {}
        other => panic!("expected SignatureOutOfBounds, got {other:?}"),
    }
}

// --- concepts --------------------------------------------------------------

#[test]
fn observing_registers_concepts_for_every_subject_and_object() {
    // Phase 2 records a Concept for each canonical entity name seen
    // in a triple's subject or object. Alias auto-deduction (resolving
    // "sarah" to "Sarah Kelly") is layer-2 extraction work and lives
    // in phase 3.
    let (mut store, _dir) = fresh_store();
    let ep = sample_episode(1, Utc.with_ymd_and_hms(2026, 5, 14, 0, 0, 0).unwrap());
    let _ = store
        .observe(ep, &HV::from_name("Sarah recommended Bawri"))
        .expect("observe");

    let sarah = store
        .concept_id_for("Sarah")
        .expect("lookup subject")
        .expect("Sarah resolves");
    let bawri = store
        .concept_id_for("Bawri")
        .expect("lookup object")
        .expect("Bawri resolves");
    assert_ne!(
        sarah, bawri,
        "distinct entities must get distinct ConceptIds"
    );

    // Unknown name returns None, not an error.
    assert!(
        store.concept_id_for("not-an-entity").unwrap().is_none(),
        "unknown name must resolve to None"
    );
}

/// Phase 3 will add this test back with `observe_with_aliases` semantics
/// once layer-2 alias resolution lands.
#[test]
#[ignore = "phase 3: alias auto-deduction is layer-2 extraction work"]
fn concept_resolves_by_canonical_and_alias_to_same_id() {}

// --- bi-temporal supersession ----------------------------------------------

#[test]
fn supersession_closes_older_valid_time_and_writes_link() {
    let (mut store, _dir) = fresh_store();
    let t1 = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
    let t2 = t1 + Duration::days(30);

    let older = sample_episode(10, t1);
    let newer = sample_episode(11, t2);

    let older_id = store.observe(older.clone(), &HV::from_name("v1")).unwrap();
    let newer_id = store.observe(newer.clone(), &HV::from_name("v2")).unwrap();

    store.supersede(older_id, newer_id).expect("supersede");

    let fetched_older = store.get_episode(older_id).unwrap().expect("present");
    assert_eq!(
        fetched_older.superseded_by,
        Some(newer_id),
        "older episode must link to its successor"
    );
    assert!(
        fetched_older.valid_time.end.is_some(),
        "older valid_time.end must be set on supersession (was {:?})",
        fetched_older.valid_time.end
    );

    let fetched_newer = store.get_episode(newer_id).unwrap().expect("present");
    assert!(
        fetched_newer.superseded_by.is_none(),
        "newer episode must not be superseded"
    );
}

#[test]
fn recall_exact_filters_by_as_of_time() {
    let (mut store, _dir) = fresh_store();
    let t1 = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
    let t2 = t1 + Duration::days(30);
    let mid = t1 + Duration::days(10);
    let after = t2 + Duration::days(10);

    let older_id = store
        .observe(sample_episode(20, t1), &HV::from_name("v1"))
        .unwrap();
    let newer_id = store
        .observe(sample_episode(21, t2), &HV::from_name("v2"))
        .unwrap();
    store.supersede(older_id, newer_id).unwrap();

    // The concept the two episodes share — once concept lookup is wired
    // in phase 2 this will produce both rows below.
    let concept = store
        .concept_id_for("Sarah")
        .unwrap()
        .expect("concept exists");

    let at_mid = store.recall_exact(concept, Some(mid)).expect("recall mid");
    let at_after = store
        .recall_exact(concept, Some(after))
        .expect("recall after");

    assert!(
        at_mid.iter().any(|e| e.id == older_id),
        "as_of mid must include the older episode (valid in [t1, t2))"
    );
    assert!(
        at_after.iter().any(|e| e.id == newer_id),
        "as_of after must include the newer episode"
    );
    assert!(
        !at_after.iter().any(|e| e.id == older_id),
        "as_of after must NOT include the superseded older episode"
    );
}

// --- export / import round-trip --------------------------------------------

#[test]
fn export_then_import_roundtrips_episodes() {
    let (mut store_a, dir_a) = fresh_store();
    let t = Utc.with_ymd_and_hms(2026, 5, 14, 0, 0, 0).unwrap();
    for i in 0..5 {
        let ep = sample_episode(100 + i, t + Duration::seconds(i as i64));
        store_a
            .observe(ep, &HV::from_name(&format!("e{i}")))
            .unwrap();
    }
    let mut jsonl = Vec::new();
    store_a.export_jsonl(&mut jsonl).expect("export");
    drop(store_a);
    drop(dir_a);

    let (mut store_b, _dir_b) = fresh_store();
    let imported = store_b.import_jsonl(jsonl.as_slice()).expect("import");
    assert_eq!(imported, 5, "imported count must match exported");
    for i in 0..5 {
        let ep = store_b
            .get_episode(EpisodeId::new(100 + i))
            .unwrap()
            .expect("present after import");
        assert_eq!(ep.text, format!("test observation #{}", 100 + i));
    }
}
