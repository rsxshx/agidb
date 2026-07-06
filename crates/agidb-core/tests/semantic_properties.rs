//! Semantic tier — static embedding invariants.

use agidb_core::semantic::{cosine, Embedder};
use agidb_core::store::{Store, StoreConfig};
use agidb_core::types::{Episode, EpisodeId, Provenance, Query, TimeRange, Triple};
use chrono::{TimeZone, Utc};
use tempfile::TempDir;

#[test]
fn embedder_returns_fixed_dim_vector() {
    let emb = agidb_core::semantic::default_embedder();
    assert_eq!(emb.dim(), 256, "default embedder must produce 256-dim");
    let v = emb.embed("Sarah recommended Bawri");
    assert_eq!(v.len(), 256);
    assert!(v.iter().any(|&x| x != 0.0), "must not be all zeros");
}

#[test]
fn deterministic_and_seeded() {
    let emb = agidb_core::semantic::default_embedder();
    let a = emb.embed("pad thai for dinner");
    let b = emb.embed("pad thai for dinner");
    assert_eq!(a, b, "same input must produce same vector");

    let c = emb.embed("thai food tonight");
    let cos = cosine(&a, &c);
    assert!(cos > 0.1 && cos < 0.99, "paraphrase cosine = {cos}");
}

#[test]
fn unrelated_texts_have_low_similarity() {
    let emb = agidb_core::semantic::default_embedder();
    let food = emb.embed("Sarah recommended Bawri");
    let code = emb.embed("the build pipeline is broken again");
    let cos = cosine(&food, &code);
    assert!(cos < 0.3, "unrelated cosine = {cos} (must stay below 0.3)");
}

#[test]
fn projection_preserves_cosine_sign() {
    let emb = agidb_core::semantic::default_embedder();
    let a = emb.embed("Sarah recommended Bawri");
    let b = emb.embed("Bawri is Sarah's pick");
    let hv_a = emb.project(&a);
    let hv_b = emb.project(&b);
    let sim = hv_a.similarity(&hv_b);
    assert!(sim > 0.55, "expected > 0.55, got {sim}");
}

#[test]
fn projection_flips_unrelated_to_low_overlap() {
    let emb = agidb_core::semantic::default_embedder();
    let food = emb.embed("Sarah recommended Bawri");
    let code = emb.embed("the build pipeline is broken");
    let hv_food = emb.project(&food);
    let hv_code = emb.project(&code);
    let sim = hv_food.similarity(&hv_code);
    assert!(sim < 0.55, "expected < 0.55, got {sim}");
}

#[test]
fn projection_is_deterministic_across_instances() {
    let e1 = agidb_core::semantic::default_embedder();
    let e2 = agidb_core::semantic::default_embedder();
    let v = vec![0.1; e1.dim()];
    assert_eq!(e1.project(&v), e2.project(&v));
}

fn make_episode(id: u64, subject: &str, predicate: &str, object: &str) -> Episode {
    let ep_id = EpisodeId::new(id);
    Episode {
        id: ep_id,
        text: format!("{subject} {predicate} {object}"),
        signature_offset: 0,
        gist_offset: 0,
        embedding_offset: 0,
        triples: vec![Triple {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
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
fn episode_with_embedder_persists_at_least_structured_and_gist() {
    let dir = TempDir::new().unwrap();
    let sig_bytes: [u8; 1024] = std::array::from_fn(|i| ((i * 13) % 256) as u8);
    let signature = agidb_core::hdc::HV(sig_bytes);
    let embedder = agidb_core::semantic::default_embedder();

    {
        let mut store = Store::open(StoreConfig::at(dir.path())).unwrap();
        let ep = make_episode(1, "Sarah", "recommends", "Bawri");
        store
            .observe_with_embedder(ep, &signature, Some(&embedder))
            .unwrap();
    }
    let store2 = Store::open(StoreConfig::at(dir.path())).unwrap();
    let stats = store2.stats().unwrap();
    assert!(
        stats.signatures >= 2,
        "expected >= 2 signatures (structured + one of {{gist,embedding}}); got {}",
        stats.signatures
    );
}

#[test]
fn tier_e_finds_paraphrase_without_structured_token_overlap() {
    let dir = TempDir::new().unwrap();
    let sig_bytes: [u8; 1024] = std::array::from_fn(|i| ((i * 13) % 256) as u8);
    let signature = agidb_core::hdc::HV(sig_bytes);

    {
        let mut store = Store::open(StoreConfig::at(dir.path())).unwrap();
        store.embedder = Some(std::sync::Arc::new(agidb_core::semantic::default_embedder()));
        // Decoy that shares only the noun ("restaurant") with the cue
        // so gist/tier-C would already hit. Tier E is what proves the
        // paraphrase-only retrieval.
        let decoy = make_episode(2, "Alice", "dislikes", "restaurant");
        store
            .observe_with_embedder(
                decoy,
                &signature,
                Some(&agidb_core::semantic::default_embedder()),
            )
            .unwrap();
        let ep = make_episode(1, "Sarah", "recommends", "Bawri");
        store
            .observe_with_embedder(
                ep,
                &signature,
                Some(&agidb_core::semantic::default_embedder()),
            )
            .unwrap();
    }
    let mut store = Store::open(StoreConfig::at(dir.path())).unwrap();
    store.embedder = Some(std::sync::Arc::new(agidb_core::semantic::default_embedder()));
    let r = store
        .recall(&Query::cue("good thai place suggestion"))
        .unwrap();
    assert_eq!(
        r.tier_used,
        agidb_core::types::Tier::Semantic,
        "tier E must fire — got {:?}",
        r.tier_used
    );
    let bawri_first = r
        .matches
        .first()
        .map(|m| m.text.contains("Bawri"))
        .unwrap_or(false);
    assert!(bawri_first, "Bawri episode must rank first");
}
