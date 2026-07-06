//! Static-text embedding tier.
//!
//! Loads a deterministic offline embedder (no model inference, no network,
//! no LLM) that produces a fixed-dim float vector per text input. The
//! vector is then projected onto an 8192-bit hypervector via a Charikar
//! random projection (seed-frozen) so the recall cascade can scan it
//! the same way it scans structured signatures.
//!
//! The default embedder is a hand-rolled feature-hash + n-gram counter
//! (zero dependencies, ~256-dim output). Plan A (model2vec/potion-class)
//! can be wired in by passing an alternate `Embedder` to
//! `Store::observe_with_embedder` — the projection and downstream scan
//! tier are embedder-agnostic.

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::episode::tokenize;
use crate::hdc::{D_U64, HV};

/// Deterministic offline text embedder. Trait-bounded so callers can
/// swap the implementation (feature-hash, model2vec, anything else)
/// without changing the recall cascade.
pub trait Embedder: Send + Sync {
    /// Fixed output dimensionality.
    fn dim(&self) -> usize;
    /// Produce an L2-normalized float vector for `text`.
    fn embed(&self, text: &str) -> Vec<f32>;
    /// Project a vector to an 8192-bit HV via Charikar random projection.
    /// Implementations cache the projection matrix internally.
    fn project(&self, vec: &[f32]) -> HV;
    /// Convenience: embed + project in one call.
    fn project_text(&self, text: &str) -> HV {
        self.project(&self.embed(text))
    }
}

/// Cosine similarity between two equal-length vectors. Returns 0.0 if
/// either vector is zero or lengths differ.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b.iter()) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        (dot / denom) as f32
    }
}

/// Stable hash of bytes using the stdlib's `DefaultHasher` (SipHash-ish;
/// same as `HV::from_name` uses, so the embedder and the HDC kernel see
/// the same hash function).
fn stable_hash(bytes: &[u8]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut h);
    h.finish()
}

/// Zero-dependency feature-hash embedder. Lowercases, tokenizes, then
/// hashes (token, bigram, trigram) features into a 256-dim L2-normalized
/// vector. Plans A (model2vec) drops in here without changing the
/// projection code.
pub struct FeatureHashEmbedder {
    dim: usize,
}

impl FeatureHashEmbedder {
    pub fn new() -> Self {
        Self { dim: 256 }
    }
}

impl Default for FeatureHashEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

impl Embedder for FeatureHashEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        let tokens: Vec<String> = tokenize(text)
            .into_iter()
            .map(|t| t.to_lowercase())
            .collect();
        let mut v = vec![0f32; self.dim];
        for (i, t) in tokens.iter().enumerate() {
            // unigram
            let h = stable_hash(t.as_bytes());
            v[(h as usize) % self.dim] += 1.0;
            // bigram
            if i + 1 < tokens.len() {
                let bg = format!("{}_{}", t, tokens[i + 1]);
                v[(stable_hash(bg.as_bytes()) as usize) % self.dim] += 0.7;
            }
            // trigram (small weight — sparse signal but useful for phrase overlap)
            if i + 2 < tokens.len() {
                let tg = format!("{}_{}_{}", t, tokens[i + 1], tokens[i + 2]);
                v[(stable_hash(tg.as_bytes()) as usize) % self.dim] += 0.4;
            }
        }
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
        v
    }

    fn project(&self, _vec: &[f32]) -> HV {
        // The default embedder delegates to `ProjectionEmbedder::project`
        // via the trait object; this stub returns zero so tests that
        // talk to a bare `FeatureHashEmbedder` directly won't accidentally
        // hit a different projection seed.
        HV::zero()
    }
}

/// Embedder trait object that owns the Charikar projection matrix
/// (seed-frozen) so callers can wrap a feature function and get the
/// canonical projection for free.
///
/// Projection (Charikar 2000): For each output bit `i ∈ [0, D)`,
/// compute `h_i = sign(Σ_{k=0}^{D-1} r_{ik} · v_k)` where `r_{ik} ∈
/// {+1, -1}` is sampled from a fixed seed. With a dense, mean-zero,
/// L2-normalized input vector this preserves cosine: independent
/// vectors stay near 0.5 similarity, correlated vectors rise above
/// the noise floor. Matrix size: `D × embedder.dim × i8 = 2 MiB` for
/// `D = 8192` and `dim = 256`.
pub struct ProjectionEmbedder {
    embed: Arc<dyn Fn(&str) -> Vec<f32> + Send + Sync>,
    dim: usize,
    projection: Vec<[i8; 256]>,
}

const CHARIKAR_SEED: u64 = 0xC417_4E5C_4152_4B41; // "CHARIKA" in ASCII

impl ProjectionEmbedder {
    pub fn new(embed: Arc<dyn Fn(&str) -> Vec<f32> + Send + Sync>, embedder_dim: usize) -> Self {
        assert!(embedder_dim <= 256, "projection rows are [i8; 256]");
        let mut rng = StdRng::seed_from_u64(CHARIKAR_SEED);
        let mut projection: Vec<[i8; 256]> = Vec::with_capacity(crate::hdc::D);
        for _ in 0..crate::hdc::D {
            let mut row = [0i8; 256];
            for x in row.iter_mut() {
                *x = if rng.gen::<bool>() { 1 } else { -1 };
            }
            projection.push(row);
        }
        Self {
            embed,
            dim: embedder_dim,
            projection,
        }
    }
}

impl Embedder for ProjectionEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }
    fn embed(&self, text: &str) -> Vec<f32> {
        (self.embed)(text)
    }
    fn project(&self, vec: &[f32]) -> HV {
        let mut bits = [0u64; D_U64];
        for (i, row) in self.projection.iter().enumerate() {
            // i32 sum avoids overflow on large positive dot products.
            let s: i32 = row[..self.dim]
                .iter()
                .zip(vec.iter())
                .map(|(r, x)| (*r as i32) * (*x * 1024.0) as i32)
                .sum();
            if s > 0 {
                bits[i / 64] |= 1u64 << (i % 64);
            }
        }
        HV::from_u64s(bits)
    }
}

/// Build the default embedder: feature-hash + Charikar binary projection.
/// Returns ~256 KiB of projection matrix + ~1 KiB of feature counters.
pub fn default_embedder() -> ProjectionEmbedder {
    let inner = Arc::new(FeatureHashEmbedder::new());
    let embed: Arc<dyn Fn(&str) -> Vec<f32> + Send + Sync> = {
        let inner = inner.clone();
        Arc::new(move |t: &str| inner.embed(t))
    };
    ProjectionEmbedder::new(embed, inner.dim())
}
