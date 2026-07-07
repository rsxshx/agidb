//! Model2Vec (`potion-base-8M`) embedder invariants.

use agidb_core::model2vec::Model2VecEmbedder;
use agidb_core::semantic::{cosine, Embedder};

/// Whether the cache is present in this environment. The model2vec
/// tests download ~30 MB on first run — skipped if the cache is absent
/// (CI without network) so the rest of the suite stays green.
fn embedder() -> Option<Model2VecEmbedder> {
    let dir = agidb_core::model2vec::cache_dir();
    if dir.join("model.safetensors").exists() && dir.join("vocab.txt").exists() {
        Model2VecEmbedder::load(&dir.join("model.safetensors"), &dir.join("vocab.txt")).ok()
    } else {
        None
    }
}

#[test]
fn potion_loads_with_correct_dim_and_vocab() {
    let Some(emb) = embedder() else {
        eprintln!("model2vec cache not present; skipping");
        return;
    };
    assert_eq!(emb.dim(), 256, "potion-base-8M is 256-dim");
}

#[test]
fn potion_projects_cosine_to_hv_similarity() {
    let Some(emb) = embedder() else {
        eprintln!("model2vec cache not present; skipping");
        return;
    };
    let a = emb.embed("Sarah recommended Bawri");
    let b = emb.embed("Bawri is Sarah's pick");
    let c = emb.embed("the build pipeline is broken again");

    let cos_ab = cosine(&a, &b);
    let cos_ac = cosine(&a, &c);
    eprintln!("DEBUG cos_ab = {cos_ab}, cos_ac = {cos_ac}");
    assert!(
        cos_ab > cos_ac,
        "related pair ({cos_ab}) must outscore unrelated ({cos_ac})"
    );

    let hv_a = emb.project(&a);
    let hv_b = emb.project(&b);
    let hv_c = emb.project(&c);
    let sim_ab = hv_a.similarity(&hv_b);
    let sim_ac = hv_a.similarity(&hv_c);
    eprintln!("DEBUG sim_ab = {sim_ab}, sim_ac = {sim_ac}");
    assert!(sim_ab > sim_ac, "projection must preserve the rank");
    assert!(
        sim_ab > 0.6,
        "related pair projected sim must clear 0.6, got {sim_ab}"
    );
    assert!(
        sim_ac < 0.55,
        "unrelated pair projected sim must stay under 0.55, got {sim_ac}"
    );
    // The [CLS]/[SEP] bias made unrelated pairs project close to 0.5
    // similarity; this is a known limitation we will document.
}

#[test]
fn potion_handles_thai_food_paraphrase() {
    let Some(emb) = embedder() else {
        eprintln!("model2vec cache not present; skipping");
        return;
    };
    // Closer paraphrases (rephrasings of the same fact) should land
    // in the model's positive-cosine half. The bench will measure how
    // far the threshold has to drop to admit them at scale.
    let a = emb.embed("Sarah recommended Bawri");
    let b = emb.embed("Bawri is Sarah's recommendation");
    let cos = cosine(&a, &b);
    eprintln!("DEBUG close-paraphrase cosine = {cos}");
    assert!(
        cos > 0.3,
        "close paraphrase should outscore random; got {cos}"
    );
}
