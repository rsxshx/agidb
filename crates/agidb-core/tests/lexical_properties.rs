//! Lexical-tier inverted-index invariants.

use agidb_core::store::{Store, StoreConfig};
use agidb_core::types::{Episode, EpisodeId, Provenance, Query, TimeRange, Triple};
use chrono::{TimeZone, Utc};
use tempfile::TempDir;

fn make_episode(id: u64, text: &str) -> Episode {
    let ep_id = EpisodeId::new(id);
    Episode {
        id: ep_id,
        text: text.to_string(),
        signature_offset: 0,
        gist_offset: 0,
        embedding_offset: 0,
        triples: vec![Triple {
            subject: "x".into(),
            predicate: "y".into(),
            object: "z".into(),
            confidence: 0.9,
            episode_id: ep_id,
        }],
        valid_time: TimeRange::point(Utc.with_ymd_and_hms(2026, 5, 14, 0, 0, 0).unwrap()),
        t_tx_start: Utc.with_ymd_and_hms(2026, 5, 14, 0, 0, 0).unwrap(),
        provenance: Provenance::default(),
        confidence: 0.9,
        superseded_by: None,
    }
}

#[test]
fn tokens_inverted_index_rebuilt_across_reopen() {
    // Use unique nonsense tokens so tier A (concept lookup) doesn't
    // short-circuit and so we can assert the Lexical tier is the one
    // that fired.
    let dir = TempDir::new().unwrap();
    let sig_bytes: [u8; 1024] = std::array::from_fn(|i| (i % 7) as u8);
    let signature = agidb_core::hdc::HV(sig_bytes);

    {
        let mut store = Store::open(StoreConfig::at(dir.path())).unwrap();
        for (id, text) in [
            (1u64, "zylphorium alpha bravo charlie delta echo"),
            (2, "foxtrot golf hotel india juliet kilo lima"),
            (3, "mike november oscar papa quebec romeo"),
        ] {
            store.observe(make_episode(id, text), &signature).unwrap();
        }
    }
    let store = Store::open(StoreConfig::at(dir.path())).unwrap();
    let r = store.recall(&Query::cue("zylphorium")).unwrap();
    assert!(
        !r.matches.is_empty(),
        "token-level recall must produce matches after reopen"
    );
    assert_eq!(
        r.tier_used,
        agidb_core::types::Tier::Lexical,
        "tier L must fire on a token-only cue — got {:?}",
        r.tier_used
    );
    assert_eq!(
        r.matches[0].episode_id,
        EpisodeId::new(1),
        "the only episode containing 'zylphorium' must rank first"
    );
}

#[test]
fn tier_l_ranks_episodes_by_token_overlap_count() {
    // Episode 1 shares 2 cue tokens ("zylphorium", "charlie").
    // Episode 2 shares 1 ("zylphorium"). Episode 3 shares 0.
    // Tier L must rank 1, 2 (and not 3).
    let dir = TempDir::new().unwrap();
    let sig_bytes: [u8; 1024] = std::array::from_fn(|i| (i % 11) as u8);
    let signature = agidb_core::hdc::HV(sig_bytes);
    let mut store = Store::open(StoreConfig::at(dir.path())).unwrap();
    store
        .observe(
            make_episode(1, "zylphorium alpha bravo charlie delta"),
            &signature,
        )
        .unwrap();
    store
        .observe(make_episode(2, "zylphorium foxtrot golf hotel"), &signature)
        .unwrap();
    store
        .observe(make_episode(3, "india juliet kilo lima mike"), &signature)
        .unwrap();
    let r = store.recall(&Query::cue("zylphorium charlie")).unwrap();
    assert_eq!(
        r.tier_used,
        agidb_core::types::Tier::Lexical,
        "tier L must fire on token overlap — got {:?}",
        r.tier_used
    );
    let ids: Vec<u64> = r.matches.iter().map(|m| m.episode_id.raw()).collect();
    assert_eq!(
        ids.first().copied(),
        Some(1),
        "episode 1 (2 token matches) must rank first; got ids={:?}",
        ids
    );
    assert!(ids.contains(&2), "episode 2 (1 token match) must appear");
    assert!(!ids.contains(&3), "episode 3 (no match) must be excluded");
    // Confidence must descend by match count.
    let confs: Vec<f32> = r.matches.iter().map(|m| m.confidence).collect();
    for w in confs.windows(2) {
        assert!(w[0] >= w[1], "confidence must descend: {:?}", confs);
    }
}

#[test]
fn tier_l_confidence_reflects_cue_coverage_not_rank() {
    // The IDF rerank used to normalize each score against the *observed
    // top score*, so whatever ranked first reported ~0.95 no matter how
    // little of the cue it actually matched. Confidence has to mean
    // "how much of the cue did this episode account for", otherwise the
    // number is a rank position wearing a confidence's clothes.
    let dir = TempDir::new().unwrap();
    let sig_bytes: [u8; 1024] = std::array::from_fn(|i| (i % 11) as u8);
    let signature = agidb_core::hdc::HV(sig_bytes);
    let mut store = Store::open(StoreConfig::at(dir.path())).unwrap();
    store
        .observe(make_episode(1, "zylphorium quandric"), &signature)
        .unwrap();
    store
        .observe(make_episode(2, "vosmeric plinthos"), &signature)
        .unwrap();

    // A cue whose every token is present in episode 1.
    let full = store.recall(&Query::cue("zylphorium quandric")).unwrap();
    let full_top = full.matches[0].confidence;
    assert!(
        full_top > 0.90,
        "an episode accounting for the whole cue should report high \
         confidence, got {full_top}"
    );

    // The case that exposes rank-relative scoring: a cue where the
    // *best* match is still a weak one. Episode 1 accounts for one of
    // five cue tokens and nothing outranks it, so normalizing against
    // the observed top score hands it a top-of-band 0.95.
    let weak = store
        .recall(&Query::cue("zylphorium kestrelic morvath dunlish farrow"))
        .unwrap();
    let weak_top = weak.matches[0].confidence;

    assert!(
        weak_top < full_top,
        "the sole weak match ({weak_top}) must not tie the full-cue match \
         ({full_top}) just because nothing outranked it"
    );
    assert!(
        weak_top < 0.80,
        "an episode matching one of five cue tokens must not report high \
         confidence merely for being first, got {weak_top}"
    );
}

#[test]
fn tier_l_ignores_tombstoned_episodes() {
    let dir = TempDir::new().unwrap();
    let sig_bytes: [u8; 1024] = std::array::from_fn(|i| (i % 13) as u8);
    let signature = agidb_core::hdc::HV(sig_bytes);
    let mut store = Store::open(StoreConfig::at(dir.path())).unwrap();
    store
        .observe(make_episode(1, "zylphorium alpha"), &signature)
        .unwrap();
    store
        .observe(make_episode(2, "zylphorium bravo"), &signature)
        .unwrap();
    store
        .unlearn(
            agidb_core::unlearn::UnlearnTarget::Episode(EpisodeId::new(1)),
            "test",
        )
        .unwrap();
    let r = store.recall(&Query::cue("zylphorium")).unwrap();
    assert_eq!(
        r.matches[0].episode_id,
        EpisodeId::new(2),
        "tombstoned episode must be filtered out of tier L"
    );
}
