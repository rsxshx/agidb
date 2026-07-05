//! Layer 2 — extraction.
//!
//! Turns raw text into structured triples, canonicalized entities, and
//! parsed time anchors. Wraps `gline-rs` (NER) + a ported GLiREL
//! relation extractor + a ported chrono_english-based temporal parser.
//! Built so the agidb-core engine stays extraction-blind: callers go
//! through `Extractor::extract` directly or through the `observe_text`
//! free function (added in plan task 12).
//!
//! Layered per the phase-3 design:
//! `docs/superpowers/specs/2026-05-23-phase-3-extraction-design.md`.

pub mod aliases;
pub mod error;
pub mod extractor;
pub mod heuristic_relations;
pub mod model_manager;
pub mod models;
pub mod ner;
pub mod predicates;
pub mod temporal;

pub use crate::extractor::{Extractor, ExtractorConfig};

// Remaining future modules:
//
// pub mod ner;            — plan task 9
// pub mod glirel;         — plan task 10
// pub mod extractor;      — plan task 11
//
// `observe_text` and `ObserveContext` are added to this file in plan
// task 12 once `extractor` lands.

pub use crate::error::{ExtractError, Result};

// ---------------------------------------------------------------------------
// observe_text — the high-level "text in → Episode stored" orchestration.
// Plan task 12 of phase 3.
// ---------------------------------------------------------------------------

use agidb_core::episode::{encode_episode_signature, encode_gist_signature};
use agidb_core::store::Store;
use agidb_core::types::{
    Episode, EpisodeId, ExtractContext, Provenance, TextExtractor, TimeRange, Triple,
};
use agidb_core::AgidbError;
use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// Context passed to [`observe_text`] — the observation time anchors any
/// relative time anchors the extractor parses, and the provenance is
/// attached to the resulting Episode.
#[derive(Clone, Debug)]
pub struct ObserveContext {
    pub observation_time: DateTime<Utc>,
    pub provenance: Provenance,
}

impl Default for ObserveContext {
    fn default() -> Self {
        Self {
            observation_time: Utc::now(),
            provenance: Provenance::default(),
        }
    }
}

/// Extract → resolve aliases → mint episode id → build + store Episode.
///
/// The high-level entry point for layer-2 → layer-3 integration. Generic
/// over `T: TextExtractor` so test code can substitute a `MockExtractor`
/// without loading any ONNX models.
///
/// Behavior at the boundaries:
///   - **Empty extraction:** the episode is still stored, with
///     `triples = vec![]`, `confidence = 0.5`, and signature computed
///     via [`encode_gist_signature`] so the raw text remains useful for
///     tier-C / gist recall.
///   - **Fuzzy-matched entity:** when the alias resolver merges a typo'd
///     mention into an existing concept, the stored triple uses the
///     existing concept's canonical name, not the raw mention.
///   - **Missing time anchor:** falls back to `ctx.observation_time`.
pub fn observe_text<T: TextExtractor>(
    store: &mut Store,
    extractor: &T,
    text: &str,
    ctx: ObserveContext,
) -> std::result::Result<EpisodeId, AgidbError> {
    let xctx = ExtractContext {
        observation_time: ctx.observation_time,
        relation_hint_types: vec![],
    };
    let ex = extractor.extract(text, &xctx)?;

    // 1. Resolve aliases for each NER entity and build a raw-text →
    //    canonical-name map. The map is what propagates into triples.
    let resolver = crate::aliases::AliasResolver::new();
    let mut canonical: HashMap<String, String> = HashMap::new();
    for e in &ex.raw_entities {
        let id = resolver.resolve(store, &e.text, &e.entity_type)?;
        let name = store
            .concept_canonical_name(id)?
            .unwrap_or_else(|| e.text.clone());
        canonical.insert(e.text.clone(), name);
    }

    // 2. Mint episode id.
    let episode_id = store.next_episode_id()?;

    // 3. Build Triples — canonical names where mapped, raw otherwise.
    let triples: Vec<Triple> = ex
        .triples
        .iter()
        .map(|et| Triple {
            subject: canonical
                .get(&et.subject)
                .cloned()
                .unwrap_or_else(|| et.subject.clone()),
            predicate: et.predicate.clone(),
            object: canonical
                .get(&et.object)
                .cloned()
                .unwrap_or_else(|| et.object.clone()),
            confidence: et.confidence,
            episode_id,
        })
        .collect();

    // 4. Resolve valid_time.
    let valid_time = ex
        .valid_time
        .unwrap_or_else(|| TimeRange::point(ctx.observation_time));

    // 5. Compute signature + confidence.
    let (signature, confidence) = if triples.is_empty() {
        (encode_gist_signature(text), 0.5)
    } else {
        let conf = geomean(triples.iter().map(|t| t.confidence));
        (
            encode_episode_signature(&triples, Some(valid_time.start)),
            conf,
        )
    };

    // 6. Build + store Episode.
    let episode = Episode {
        id: episode_id,
        text: text.to_string(),
        signature_offset: 0, // overwritten by Store::observe
        gist_offset: 0,      // overwritten by Store::observe
        triples,
        valid_time,
        t_tx_start: Utc::now(),
        provenance: ctx.provenance,
        confidence,
        superseded_by: None,
    };
    store.observe(episode, &signature)
}

/// Geometric mean of a non-empty positive iterator; `0.5` on empty input.
fn geomean<I: IntoIterator<Item = f32>>(iter: I) -> f32 {
    let v: Vec<f32> = iter.into_iter().filter(|x| *x > 0.0).collect();
    if v.is_empty() {
        return 0.5;
    }
    let log_sum: f32 = v.iter().map(|x| x.ln()).sum();
    (log_sum / v.len() as f32).exp()
}
