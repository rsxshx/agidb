//! Floor 1 — sensory buffer + surprise gating invariants.

use agidb_core::sensory::SURPRISE_PROMOTION_THRESHOLD;
use agidb_core::store::{Store, StoreConfig};
use agidb_core::types::Query;
use tempfile::TempDir;

fn fresh_store() -> (Store, TempDir) {
    let dir = TempDir::new().expect("tempdir");
    let store = Store::open(StoreConfig::at(dir.path())).expect("open");
    (store, dir)
}

#[test]
fn first_frame_on_empty_store_is_maximally_surprising_and_promoted() {
    let (mut store, _dir) = fresh_store();
    let obs = store
        .observe_sensory("Sarah recommended Bawri in Bandra")
        .expect("sense");
    assert!(
        (obs.surprise - 1.0).abs() < 1e-6,
        "empty store → surprise 1.0"
    );
    assert!(obs.promoted.is_some(), "must promote above threshold");
}

#[test]
fn duplicate_text_is_not_promoted() {
    let (mut store, _dir) = fresh_store();
    store
        .observe_sensory("Sarah recommended Bawri in Bandra")
        .expect("first");
    let second = store
        .observe_sensory("Sarah recommended Bawri in Bandra")
        .expect("second");
    assert!(
        second.surprise < SURPRISE_PROMOTION_THRESHOLD,
        "identical text must not be surprising, got {}",
        second.surprise
    );
    assert!(second.promoted.is_none());
}

#[test]
fn novel_text_after_familiar_text_is_promoted() {
    let (mut store, _dir) = fresh_store();
    store
        .observe_sensory("Sarah recommended Bawri in Bandra")
        .expect("first");
    let novel = store
        .observe_sensory("quarterly compliance audit deadline moved to friday")
        .expect("novel");
    assert!(
        novel.surprise >= SURPRISE_PROMOTION_THRESHOLD,
        "disjoint text must stay surprising, got {}",
        novel.surprise
    );
    assert!(novel.promoted.is_some());
}

#[test]
fn empty_text_scores_zero_and_is_never_promoted() {
    let (mut store, _dir) = fresh_store();
    let obs = store.observe_sensory("   ").expect("sense");
    assert_eq!(obs.surprise, 0.0);
    assert!(obs.promoted.is_none());
}

#[test]
fn promoted_frame_is_recallable_as_episode() {
    let (mut store, _dir) = fresh_store();
    let obs = store
        .observe_sensory("the tokyo offsite starts on monday morning")
        .expect("sense");
    let id = obs.promoted.expect("promoted");
    let r = store
        .recall(&Query::cue("tokyo offsite monday"))
        .expect("recall");
    assert!(r.matches.iter().any(|m| m.episode_id == id));
}

#[test]
fn ring_buffer_evicts_beyond_capacity() {
    let (mut store, _dir) = fresh_store();
    for i in 0..1010u32 {
        store
            .observe_sensory(&format!("frame number {i} carries payload {}", i * 7))
            .expect("sense");
    }
    let frames = store.sensory_frames(2000).expect("frames");
    assert_eq!(frames.len(), 1000, "ring buffer must cap at capacity");
    assert!(frames[0].text.contains("frame number 1009"));
    assert!(frames.last().unwrap().text.contains("frame number 10"));
}

#[test]
fn frames_survive_reopen() {
    let dir = TempDir::new().expect("tempdir");
    {
        let mut store = Store::open(StoreConfig::at(dir.path())).expect("open");
        store
            .observe_sensory("persist me across reopen")
            .expect("sense");
    }
    let store = Store::open(StoreConfig::at(dir.path())).expect("reopen");
    let frames = store.sensory_frames(10).expect("frames");
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].text, "persist me across reopen");
}
