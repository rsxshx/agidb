//! Semantic tier — static embedding invariants.
//!
//! The persistence test (`episode_persists_semantic_hv_across_reopen`)
//! lives behind `#[cfg(...)]` until Task 3 lands `embedding_offset` on
//! `Episode` and `observe_with_embedder` on `Store`.

use agidb_core::semantic::{cosine, Embedder};

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
    // Cosine must be positive but not 1.0 — paraphrase ≠ duplicate.
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
    assert_eq!(
        e1.project(&v),
        e2.project(&v),
        "projection must be seeded — same seed, same matrix"
    );
}