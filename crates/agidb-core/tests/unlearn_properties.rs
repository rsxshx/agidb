//! Phase 11 — non-destructive cascading unlearn invariants.
//! Phase 10 — self-vector + learning-log invariants.
//!
//! Constitution article XVI: unlearn is non-destructive at audit. The
//! `LearningEvent::Unlearned` record is permanent. Tombstoned data is
//! recoverable within 30 days. Self-vector subtraction removes centroid
//! contamination.

use agidb_core::episode::encode_episode_signature;
use agidb_core::hdc::HV;
use agidb_core::learning_log::LearningEvent;
use agidb_core::self_model::{hv_ema_update, hv_subtract};
use agidb_core::store::{Store, StoreConfig};
use agidb_core::types::*;
use agidb_core::unlearn::{UnlearnReport, UnlearnTarget};
use chrono::Utc;
use tempfile::TempDir;

fn fresh_store() -> (Store, TempDir) {
    let dir = TempDir::new().expect("tempdir");
    let mut store = Store::open(StoreConfig::at(dir.path())).expect("open");
    store.init_self_vector().expect("init self-vector");
    (store, dir)
}

fn observe_triple(store: &mut Store, id: u64, subj: &str, pred: &str, obj: &str) -> EpisodeId {
    let ep_id = EpisodeId::new(id);
    let triples = vec![Triple {
        subject: subj.into(),
        predicate: pred.into(),
        object: obj.into(),
        confidence: 0.9,
        episode_id: ep_id,
    }];
    let sig = encode_episode_signature(&triples, None);
    let ep = Episode {
        id: ep_id,
        text: format!("{subj} {pred} {obj}"),
        signature_offset: 0,
        gist_offset: 0,
        triples,
        valid_time: TimeRange::point(Utc::now()),
        t_tx_start: Utc::now(),
        provenance: Provenance::default(),
        confidence: 0.9,
        superseded_by: None,
    };
    store.observe(ep, &sig).expect("observe");
    ep_id
}

// ---------------------------------------------------------------------------
// Learning log (phase 10).
// ---------------------------------------------------------------------------

#[test]
fn observe_emits_episode_stored_event() {
    let (mut store, _d) = fresh_store();
    let _ = observe_triple(&mut store, 1, "Sarah", "likes", "thai");

    let events = store.all_learning_events().expect("events");
    assert!(
        events
            .iter()
            .any(|e| matches!(e, LearningEvent::EpisodeStored { id, .. } if id.raw() == 1)),
        "observe must emit EpisodeStored"
    );
}

#[test]
fn what_did_i_learn_filters_by_time() {
    let (mut store, _d) = fresh_store();
    let _ = observe_triple(&mut store, 1, "A", "is", "A");
    let before = Utc::now();
    let _ = observe_triple(&mut store, 2, "B", "is", "B");

    let since = store.what_did_i_learn(before).expect("learn");
    assert!(
        since.iter().all(|e| e.timestamp() >= before),
        "what_did_i_learn must only return events after the cutoff"
    );
    assert!(since
        .iter()
        .any(|e| matches!(e, LearningEvent::EpisodeStored { id, .. } if id.raw() == 2)));
}

#[test]
fn goal_and_belief_ops_emit_learning_events() {
    let (mut store, _d) = fresh_store();
    let gid = store.set_goal(Goal::new("test goal")).expect("set");
    store.complete_goal(gid, vec![]).expect("complete");
    let bid = store
        .assert_belief(
            Belief::new("test belief")
                .with_triple("X", "is", "Y")
                .with_confidence(0.7),
        )
        .expect("assert");
    store.withdraw_belief(bid, "testing").expect("withdraw");

    let events = store.all_learning_events().expect("events");
    assert!(events
        .iter()
        .any(|e| matches!(e, LearningEvent::GoalSet { .. })));
    assert!(events
        .iter()
        .any(|e| matches!(e, LearningEvent::GoalStateChanged { to, .. } if to == "Completed")));
    assert!(events
        .iter()
        .any(|e| matches!(e, LearningEvent::BeliefAsserted { .. })));
    assert!(events
        .iter()
        .any(|e| matches!(e, LearningEvent::BeliefWithdrawn { .. })));
}

// ---------------------------------------------------------------------------
// Self-vector (phase 10).
// ---------------------------------------------------------------------------

#[test]
fn self_vector_starts_zero_and_drifts_on_consolidate() {
    let (mut store, _d) = fresh_store();
    let sv0 = store.self_vector().expect("sv");
    assert_eq!(
        sv0.hamming(&HV([0u8; 1024])),
        0,
        "fresh self-vector is zero"
    );

    // Observe 3 identical episodes → consolidate mints an atom →
    // self-vector should drift away from zero.
    for i in 1..=3 {
        observe_triple(&mut store, i, "Sarah", "likes", "thai");
    }
    store.consolidate().expect("consolidate");
    let sv1 = store.self_vector().expect("sv");
    assert!(
        sv1.hamming(&HV([0u8; 1024])) > 0,
        "self-vector must drift after consolidation"
    );
}

#[test]
fn self_vector_history_snapshots_are_retrievable() {
    let (mut store, _d) = fresh_store();
    for i in 1..=3 {
        observe_triple(&mut store, i, "A", "is", "A");
    }
    store.consolidate().expect("consolidate");
    let hist = store.self_vector_history().expect("history");
    assert!(
        !hist.is_empty(),
        "consolidation must snapshot the self-vector"
    );
}

#[test]
fn hv_ema_update_moves_toward_bundle() {
    let current = HV::from_name("current");
    let bundle = HV::from_name("target");
    let updated = hv_ema_update(&current, &bundle, 0.5);
    // With alpha=0.5, the updated HV should be closer to the bundle than
    // the original was.
    let sim_before = current.similarity(&bundle);
    let sim_after = updated.similarity(&bundle);
    assert!(
        sim_after > sim_before,
        "EMA must move toward bundle: {sim_before} → {sim_after}"
    );
}

#[test]
fn hv_subtract_removes_alignment() {
    let base = HV::from_name("base");
    let addition = HV::from_name("added");
    let combined = hv_ema_update(&base, &addition, 1.0); // full add
    let sim_after_add = combined.similarity(&addition);
    let subtracted = hv_subtract(&combined, &addition);
    let sim_after_sub = subtracted.similarity(&addition);
    assert!(
        sim_after_sub < sim_after_add,
        "subtract must reduce alignment: {sim_after_add} → {sim_after_sub}"
    );
}

// ---------------------------------------------------------------------------
// Unlearn (phase 11).
// ---------------------------------------------------------------------------

#[test]
fn unlearn_episode_tombstones_it_and_excludes_from_recall() {
    let (mut store, _d) = fresh_store();
    let _ = observe_triple(&mut store, 1, "Sarah", "likes", "thai");
    let _ = observe_triple(&mut store, 2, "Marco", "likes", "go");

    // Both are recallable initially.
    let r = store.recall(&Query::cue("Sarah")).expect("recall");
    assert!(r.matches.iter().any(|m| m.episode_id.raw() == 1));

    // Unlearn episode 1.
    let report = store
        .unlearn(UnlearnTarget::Episode(EpisodeId::new(1)), "test")
        .expect("unlearn");
    assert_eq!(report.episodes_removed, 1);

    // Episode 1 is excluded from recall.
    let r = store.recall(&Query::cue("Sarah")).expect("recall");
    assert!(
        !r.matches.iter().any(|m| m.episode_id.raw() == 1),
        "tombstoned episode must not appear in recall"
    );

    // The audit event is permanent.
    let events = store.all_learning_events().expect("events");
    assert!(
        events.iter().any(
            |e| matches!(e, LearningEvent::Unlearned { target, .. } if target.contains("episode:1"))
        ),
        "Unlearned event must be in the audit log"
    );
}

#[test]
fn unlearn_concept_cascades_to_episodes_and_beliefs() {
    let (mut store, _d) = fresh_store();
    let _ = observe_triple(&mut store, 1, "Sarah", "likes", "thai");
    let _ = observe_triple(&mut store, 2, "Sarah", "lives_in", "Bandra");
    let _ = observe_triple(&mut store, 3, "Marco", "likes", "go");

    // Assert a belief about Sarah.
    let _bid = store
        .assert_belief(
            Belief::new("Sarah likes thai food")
                .with_triple("Sarah", "likes", "thai food")
                .with_confidence(0.8)
                .with_evidence(vec![EpisodeId::new(1)]),
        )
        .expect("assert");

    // Find Sarah's concept id.
    let cid = store
        .concept_id_for("Sarah")
        .expect("lookup")
        .expect("exists");

    // Unlearn the concept.
    let report = store
        .unlearn(UnlearnTarget::Concept(cid), "GDPR forget request")
        .expect("unlearn");
    assert!(
        report.episodes_removed >= 2,
        "should tombstone Sarah's episodes: got {}",
        report.episodes_removed
    );
    assert!(
        report.beliefs_removed + report.beliefs_revised >= 1,
        "belief about Sarah should be affected"
    );

    // Sarah is gone from recall.
    let r = store.recall(&Query::cue("Sarah")).expect("recall");
    assert!(
        !r.matches.iter().any(|m| m.text.contains("Sarah")),
        "no Sarah episodes after unlearn concept"
    );

    // Marco is still there.
    let r = store.recall(&Query::cue("Marco")).expect("recall");
    assert!(
        r.matches.iter().any(|m| m.text.contains("Marco")),
        "unrelated episodes must survive concept unlearn"
    );
}

#[test]
fn unlearn_subtracts_from_self_vector() {
    let (mut store, _d) = fresh_store();
    for i in 1..=3 {
        observe_triple(&mut store, i, "Sarah", "likes", "thai");
    }
    store.consolidate().expect("consolidate");
    let sv_before = store.self_vector().expect("sv");

    // Get Sarah's concept sig for verification.
    let cid = store
        .concept_id_for("Sarah")
        .expect("lookup")
        .expect("exists");
    let _ = cid;

    let report = store
        .unlearn(UnlearnTarget::Concept(cid), "forget Sarah")
        .expect("unlearn");
    assert!(
        report.self_vector_drift_hamming > 0,
        "self-vector must drift on unlearn"
    );

    let sv_after = store.self_vector().expect("sv");
    assert_ne!(
        sv_before.hamming(&sv_after),
        0,
        "self-vector must change after unlearn"
    );
}

#[test]
fn unlearn_by_source_gdpr_article_17() {
    let (mut store, _d) = fresh_store();
    // Observe with a specific source.
    let ep1 = Episode {
        id: EpisodeId::new(1),
        text: "Sarah likes thai".into(),
        signature_offset: 0,
        gist_offset: 0,
        triples: vec![Triple {
            subject: "Sarah".into(),
            predicate: "likes".into(),
            object: "thai".into(),
            confidence: 0.9,
            episode_id: EpisodeId::new(1),
        }],
        valid_time: TimeRange::point(Utc::now()),
        t_tx_start: Utc::now(),
        provenance: Provenance {
            source: "tool:gmail".into(),
            ..Provenance::default()
        },
        confidence: 0.9,
        superseded_by: None,
    };
    let sig = encode_episode_signature(&ep1.triples, None);
    store.observe(ep1, &sig).expect("observe");

    // Another episode from a different source.
    observe_triple(&mut store, 2, "Marco", "likes", "go");

    let report = store
        .unlearn(
            UnlearnTarget::BySource("tool:gmail".into()),
            "GDPR Article 17",
        )
        .expect("unlearn");
    assert_eq!(report.episodes_removed, 1);
    assert_eq!(report.reason, "GDPR Article 17");

    // Gmail episode is gone; Marco survives.
    let r = store.recall(&Query::cue("Sarah")).expect("recall");
    assert!(!r.matches.iter().any(|m| m.episode_id.raw() == 1));
    let r = store.recall(&Query::cue("Marco")).expect("recall");
    assert!(r.matches.iter().any(|m| m.episode_id.raw() == 2));
}

#[test]
fn restore_within_window_reverses_tombstone() {
    let (mut store, _d) = fresh_store();
    let _ = observe_triple(&mut store, 1, "Sarah", "likes", "thai");

    let report = store
        .unlearn(UnlearnTarget::Episode(EpisodeId::new(1)), "mistake")
        .expect("unlearn");

    // Episode is excluded.
    let r = store.recall(&Query::cue("Sarah")).expect("recall");
    assert!(!r.matches.iter().any(|m| m.episode_id.raw() == 1));

    // Restore within the window.
    let restored = store
        .restore_within_window(report.audit_event_id)
        .expect("restore");
    assert_eq!(restored, 1, "should restore 1 tombstone");

    // Episode is back.
    let r = store.recall(&Query::cue("Sarah")).expect("recall");
    assert!(
        r.matches.iter().any(|m| m.episode_id.raw() == 1),
        "restored episode must reappear in recall"
    );
}

#[test]
fn unlearn_audit_log_is_permanent() {
    let (mut store, _d) = fresh_store();
    let _ = observe_triple(&mut store, 1, "Sarah", "likes", "thai");
    let report = store
        .unlearn(UnlearnTarget::Episode(EpisodeId::new(1)), "test")
        .expect("unlearn");

    // The Unlearned event survives — it's in the learning log.
    let events = store.all_learning_events().expect("events");
    let unlearned = events
        .iter()
        .find(|e| matches!(e, LearningEvent::Unlearned { .. }));
    assert!(unlearned.is_some(), "Unlearned event must be permanent");

    // Even after restore, the audit event persists (restore only
    // removes the tombstone, not the audit record).
    store
        .restore_within_window(report.audit_event_id)
        .expect("restore");
    let events = store.all_learning_events().expect("events");
    assert!(
        events
            .iter()
            .any(|e| matches!(e, LearningEvent::Unlearned { .. })),
        "Unlearned audit record must persist after restore"
    );
}

#[test]
fn unlearn_cascade_corrects_beliefs_with_tombstoned_evidence() {
    let (mut store, _d) = fresh_store();
    let _ = observe_triple(&mut store, 1, "Sarah", "likes", "thai");
    let _ = observe_triple(&mut store, 2, "Sarah", "likes", "thai");
    let bid = store
        .assert_belief(
            Belief::new("Sarah likes thai")
                .with_triple("Sarah", "likes", "thai")
                .with_confidence(0.8)
                .with_evidence(vec![EpisodeId::new(1), EpisodeId::new(2)]),
        )
        .expect("assert");
    let before = store.get_belief(bid).unwrap().unwrap();
    assert_eq!(before.evidence.len(), 2);

    // Unlearn episode 1 — the belief's evidence should be corrected.
    let report = store
        .unlearn(UnlearnTarget::Episode(EpisodeId::new(1)), "test")
        .expect("unlearn");
    assert!(
        report.beliefs_revised > 0,
        "belief with tombstoned evidence should be revised"
    );

    let after = store.get_belief(bid).unwrap().unwrap();
    assert!(
        !after.evidence.contains(&EpisodeId::new(1)),
        "tombstoned evidence must be removed"
    );
    assert!(
        after.confidence < before.confidence,
        "confidence must drop with less evidence"
    );
}

#[test]
fn unlearn_report_carries_full_cascade_summary() {
    let (mut store, _d) = fresh_store();
    let _ = observe_triple(&mut store, 1, "Sarah", "likes", "thai");
    let _ = observe_triple(&mut store, 2, "Sarah", "lives", "Bandra");
    let cid = store.concept_id_for("Sarah").unwrap().unwrap();

    let report: UnlearnReport = store
        .unlearn(UnlearnTarget::Concept(cid), "test")
        .expect("unlearn");
    assert!(!report.target.is_empty());
    assert!(report.episodes_removed > 0);
    assert!(report.tombstone_expiry > Utc::now());
}
