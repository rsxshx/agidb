//! Tiered recall — the layer-1 read path.
//!
//! Implements [`Store::recall`] as the four-tier cascade documented in
//! `docs/architecture/layer-1-recall.md`:
//!
//! - **Tier A — Exact**: cue tokens looked up in the concept index;
//!   any matching episode is returned with confidence 1.0.
//! - **Tier B — Similarity**: structured HDC signature similarity.
//!   The cue's tokens are resolved to known concepts (case-insensitive,
//!   fuzzy within edit distance 1); their role-bound HVs are bundled
//!   into a structured cue signature that is POPCOUNT-scanned against
//!   every episode's stored structured signature. Confidence band
//!   [0.6, 0.95].
//! - **Tier C — Gist**: stored gist HV (token bundle) similarity,
//!   similarity floor 0.55, confidence band [0.3, 0.6].
//! - **Tier D — NearestNeighbor**: top-k by gist similarity regardless
//!   of threshold, confidence capped at 0.3.
//!
//! All scanning tiers sweep the in-memory scan directory
//! ([`crate::store::ScanEntry`]) and read HVs straight out of the
//! `signatures.dat` mmap — no per-row redb reads, no re-encoding of
//! episode text at query time. Only the top-k survivors are hydrated
//! into full `Episode` rows.
//!
//! Per [constitution](../../.specify/memory/constitution.md) article VI,
//! recall never returns the empty set under the default `tier_floor`.
//!
//! ## Temporal retrieval (v0.2 — `feat/temporal-retrieval`)
//!
//! Before this change, recall was purely cue-driven: time was a filter
//! (`Query::as_of`), never a retrieval key or ranking signal. homn's
//! recall eval showed that landed at 40% on temporal queries (vs 80%
//! factual / 100% commitment), because the cue for a temporal question
//! (e.g. "when did I send the pricing quote") typically shares no
//! tokens with the answer episode (which contains "Friday" or a date).
//!
//! The fix layers four orthogonal temporal retrieval features on top
//! of the existing cascade without changing its tiers:
//!
//! 1. **`Query::time_window`** — an interval-overlap filter applied at
//!    every tier that sweeps the scan directory. An episode survives
//!    iff `valid_time.start <= to` and `valid_time.end.unwrap_or(start)`
//!    is at least `from`. Distinct from `as_of` (point-containment);
//!    both can be active together.
//! 2. **`Query::subject`** — a `ConceptId` filter that restricts the
//!    cascade to episodes linked to that concept via the
//!    `concept_episodes` multimap. Lets callers ask "every episode
//!    about X" as a structured filter rather than a lexical guess.
//! 3. **`Query::recency_weight` / `Query::time_anchor`** — a
//!    post-cascade rerank blend: `final = (1 - w) * cue_score +
//!    w * recency`, where `recency = exp(-|valid_time.start - anchor|
//!    / half_life)` with a default 7-day half-life. `weight = 0.0`
//!    preserves the pre-temporal-recall behavior exactly.
//! 4. **`Store::list_episodes_in_range(from, to, limit)`** — a
//!    chronological listing keyed on `valid_time.start`, exposed
//!    through `Agidb::timeline(subject, from, to, limit)`.
//!
//! All four default to "off" so existing callers see no behavior
//! change.

use crate::episode::{encode_query_signature, role_obj, role_subj, tokenize};
use crate::error::{AgidbError, Result};
use crate::hdc::HV;
use crate::store::{ScanEntry, Store, SEMANTIC_ATOMS};
use crate::types::*;
use chrono::{DateTime, Utc};
use redb::ReadableTable;
use roaring::RoaringBitmap;
use std::collections::HashSet;
use std::time::Instant;

/// Tier-B phi floor. Phi is density-corrected correlation: unrelated
/// episodes score ≈ 0 ± 0.011 (one σ at D=8192); a cue sharing one
/// entity with a stored episode scores ≈ 0.12–0.17 depending on how
/// many triples share the bundle. 0.06 is >5σ above noise and below
/// the thinnest genuine signal.
const TIER_B_PHI_FLOOR: f32 = 0.06;

/// Phi at (or above) which tier-B confidence saturates at the top of
/// its band.
const TIER_B_PHI_HI: f32 = 0.30;

/// Tier-E phi floor. Tier E (semantic) reads the Charikar-projected
/// static-text embedding HV and scores by the same phi kernel — but
/// the embedding signal is broader (paraphrase > role-bound overlap),
/// so the floor sits a notch below B. Unrelated pairs still land ≈0
/// ±0.011 (D=8192); paraphrase cosines in our probe sit in 0.10–0.25
/// which translates to phi in the same range. Floor 0.04 is >3σ above
/// noise and below the thinnest genuine paraphrase.
const TIER_E_PHI_FLOOR: f32 = 0.06;

/// Phi at which tier-E confidence saturates at the top of its band.
const TIER_E_PHI_HI: f32 = 0.20;

/// Tier-C similarity floor. Two random HVs have expected similarity
/// ≈ 0.5; the floor sits a few percent above that to keep noise out
/// of the high-confidence band.
const TIER_C_SIM_FLOOR: f32 = 0.55;

/// Linear map ranges for confidence calibration.
const TIER_B_BAND: (f32, f32) = (0.6, 0.95);
const TIER_E_BAND: (f32, f32) = (0.4, 0.7);
const TIER_C_BAND: (f32, f32) = (0.3, 0.6);
const TIER_D_CAP: f32 = 0.3;

/// Cap on how many cue tokens are resolved against the concept table
/// when building the tier-B structured cue signature. Keeps worst-case
/// fuzzy lookups bounded on adversarially long cues.
const TIER_B_MAX_CUE_CONCEPTS: usize = 8;

impl Store {
    /// Run a recall against the store. Per the constitution, never
    /// returns an empty `Recall::matches` under the default `tier_floor`
    /// of `NearestNeighbor`.
    ///
    /// `Recall::semantic_atoms` also carries any consolidated atoms
    /// whose anchoring concept matches a cue token — this is how phase
    /// 6 surfaces consolidated knowledge alongside raw episodes.
    pub fn recall(&self, query: &Query) -> Result<Recall> {
        let started = Instant::now();
        let mut matches = self.run_cascade(query)?;
        let semantic_atoms = self.semantic_atoms_for_cue(query)?;

        // Phase 9 — goal-biased reweighting. When `goal_bias_weight > 0`,
        // active goals' HDC signatures up-weight episode matches that are
        // semantically related to those goals. Attention as a cognitive
        // function: the agent attends to what it wants.
        let (goal_biased, active_goal_ids) = if query.goal_bias_weight > 0.0 {
            self.apply_goal_bias(&mut matches, query.goal_bias_weight)?;
            let ids: Vec<GoalId> = self.active_goals()?.into_iter().map(|g| g.id).collect();
            (!ids.is_empty(), ids)
        } else {
            (false, Vec::new())
        };

        // v0.2 — temporal retrieval: recency rerank. When
        // `recency_weight > 0.0` AND `time_anchor` is set, blend the
        // cue-similarity score with a recency score. Default 0.0
        // preserves the byte-identical pre-temporal-recall behavior.
        if query.recency_weight > 0.0 {
            if let Some(anchor) = query.time_anchor {
                self.apply_recency_rerank(
                    &mut matches,
                    query.recency_weight,
                    anchor,
                    query.recency_half_life,
                )?;
            }
        }

        // Re-sort after biasing and re-apply min_confidence / k.
        matches.retain(|m| m.confidence >= query.min_confidence);
        matches.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        matches.truncate(query.k);

        let tier_used = matches
            .iter()
            .map(|m| m.source_tier)
            .min_by_key(|t| t.depth())
            .unwrap_or(Tier::NearestNeighbor);
        let elapsed_ms = started.elapsed().as_millis().min(u32::MAX as u128) as u32;
        Ok(Recall {
            matches,
            semantic_atoms,
            goal_biased,
            active_goals: active_goal_ids,
            tier_used,
            elapsed_ms,
        })
    }

    /// Goal-biased reweighting pass (phase 9). For each match, read its
    /// episode signature and compute the max similarity to any active
    /// goal's signature, then nudge confidence up by
    /// `weight * max_goal_sim * (1 - confidence)`. Bias never reduces
    /// confidence and never exceeds 1.0.
    fn apply_goal_bias(&self, matches: &mut [RecallMatch], weight: f32) -> Result<()> {
        let goals = self.active_goals()?;
        if goals.is_empty() {
            return Ok(());
        }
        let goal_sigs: Vec<HV> = goals
            .iter()
            .filter_map(|g| self.signatures.read(g.signature_offset).ok())
            .collect();
        if goal_sigs.is_empty() {
            return Ok(());
        }
        for m in matches.iter_mut() {
            // The scan directory has the signature offset — no redb read.
            let Some(entry) = self.scan_entry(m.episode_id.raw()) else {
                continue;
            };
            let Ok(ep_sig) = self.signatures.read(entry.sig_offset) else {
                continue;
            };
            let max_goal_sim = goal_sigs
                .iter()
                .map(|gs| ep_sig.similarity(gs))
                .fold(0.0f32, f32::max);
            let boost = weight * max_goal_sim * (1.0 - m.confidence);
            m.confidence = (m.confidence + boost).clamp(0.0, 1.0);
        }
        Ok(())
    }

    /// v0.2 — temporal retrieval: recency rerank. Blend each match's
    /// cue-similarity confidence with a recency score based on how
    /// close the episode's `valid_time.start` is to `anchor`:
    ///
    /// ```text
    /// recency = exp(-|valid_time.start - anchor| / half_life)
    /// final   = (1 - w) * cue_score + w * recency
    /// ```
    ///
    /// `weight = 0.0` (the default) disables this pass so callers that
    /// don't care about time see byte-identical behavior. The blend
    /// happens after goal-bias so the final sort reflects every
    /// reranking signal. Symmetric around the anchor — an episode
    /// dated a week before *or* after scores the same — which matches
    /// the way homn phrases "around then"-style queries.
    fn apply_recency_rerank(
        &self,
        matches: &mut [RecallMatch],
        weight: f32,
        anchor: DateTime<Utc>,
        half_life_seconds: f32,
    ) -> Result<()> {
        if matches.is_empty() || weight <= 0.0 {
            return Ok(());
        }
        let half_life = half_life_seconds.max(1.0);
        for m in matches.iter_mut() {
            let Some(entry) = self.scan_entry(m.episode_id.raw()) else {
                continue;
            };
            let dt_seconds = (anchor - entry.valid_start).num_seconds().abs() as f32;
            // `(-dt/half_life).exp()` stays in (0, 1] for any non-negative
            // dt. dt of 0 → 1.0; dt = half_life → ~0.368; dt = 4*half_life
            // → ~0.018. With the default 7-day half-life an episode
            // dated 7 days from the anchor keeps ~37% of its recency
            // signal — a soft, easily-tuned knob.
            let recency = (-dt_seconds / half_life).exp();
            let cue_score = m.confidence;
            m.confidence = ((1.0 - weight) * cue_score + weight * recency).clamp(0.0, 1.0);
        }
        Ok(())
    }

    /// v0.2 — temporal retrieval: load the set of episode ids linked
    /// to `concept` via the `concept_episodes` multimap, restricted to
    /// the query's temporal filters. Used by the cascade when
    /// `Query::subject` is set, so the subject filter can be applied
    /// in O(1) per scan entry without a per-tier concept resolution
    /// pass.
    fn subject_episode_set(&self, concept: ConceptId, query: &Query) -> Result<HashSet<EpisodeId>> {
        let tx = self.db.begin_read()?;
        let concept_episodes = tx.open_multimap_table(crate::store::CONCEPT_EPISODES)?;
        let mut set = HashSet::new();
        for raw in concept_episodes.get(concept.raw())? {
            let raw_id = raw?.value();
            // Subject filter + temporal filter can both be enforced
            // from the scan directory alone — `scan_entry` has the
            // valid_start / valid_end / tombstoned bit without
            // deserializing the full `Episode` row. Keeps the
            // subject-filter precomputation O(N) where N is the size
            // of the concept's posting list, not N + per-row decode.
            let Some(entry) = self.scan_entry(raw_id) else {
                continue;
            };
            if entry.tombstoned {
                continue;
            }
            if let Some(t) = query.as_of {
                if !entry.valid_at(t) {
                    continue;
                }
            }
            if let Some((from, to)) = query.time_window {
                if !entry.overlaps_window(from, to) {
                    continue;
                }
            }
            set.insert(EpisodeId::new(raw_id));
        }
        Ok(set)
    }

    /// Look up every `SemanticAtom` whose anchoring concept matches a
    /// token in the cue. O(N) over atoms today; a concept→atoms
    /// inverted index is a phase-6 follow-up.
    fn semantic_atoms_for_cue(&self, query: &Query) -> Result<Vec<SemanticMatch>> {
        let mut wanted: HashSet<ConceptId> = HashSet::new();
        for token in tokenize(&query.cue) {
            if let Some(cid) = self.concept_id_for(&token)? {
                wanted.insert(cid);
            }
        }
        if wanted.is_empty() {
            return Ok(Vec::new());
        }
        let tx = self.db.begin_read()?;
        let table = tx.open_table(SEMANTIC_ATOMS)?;
        let mut out = Vec::new();
        for entry in table.iter()? {
            let (_, v) = entry?;
            let bytes = v.value();
            let atom: SemanticAtom = bincode::deserialize(&bytes)
                .map_err(|e| AgidbError::Internal(format!("decode atom: {e}")))?;
            if wanted.contains(&atom.concept) {
                out.push(SemanticMatch::from(atom));
            }
        }
        Ok(out)
    }

    fn run_cascade(&self, query: &Query) -> Result<Vec<RecallMatch>> {
        // v0.2 — temporal retrieval: resolve the subject filter once
        // and reuse it across every tier that sweeps the scan
        // directory. Subject episodes also have to satisfy the query's
        // temporal filters (as_of / time_window) — `subject_episode_set`
        // applies both up front so the per-tier filter is a single
        // `HashSet::contains` lookup.
        let subject_filter: Option<HashSet<EpisodeId>> = match query.subject {
            Some(cid) => Some(self.subject_episode_set(cid, query)?),
            None => None,
        };

        // Tier A — concept candidates reranked by IDF-weighted lexical
        // overlap. This catches both "the cue token is a known concept"
        // (high-confidence A) and "the cue tokens overlap the stored
        // text by IDF" (lexical rerank, tier L). One cascade step.
        if Tier::Exact.depth() <= query.tier_floor.depth() {
            let a = self.tier_a_exact(query, subject_filter.as_ref())?;
            if !a.is_empty() {
                return Ok(self.finalize(a, query));
            }
        }

        // Tier L — lexical inverted-index posting-list intersection
        // (cue tokens that aren't known concepts). Falls through if
        // tier A already produced matches.
        if Tier::Lexical.depth() <= query.tier_floor.depth() {
            let l = self.tier_l_lexical(query, subject_filter.as_ref())?;
            if !l.is_empty() {
                return Ok(self.finalize(l, query));
            }
        }

        // Tier B — structured similarity. Resolve cue tokens to known
        // concepts (case-insensitive + fuzzy), bundle their role-bound
        // HVs into a structured cue signature, and phi-score the stored
        // episode signatures (density-corrected; raw similarity inflates
        // on AND-bundled sparse vectors — see phi_from_counts).
        if Tier::Similarity.depth() <= query.tier_floor.depth() {
            if let Some(cue_sig) = self.structured_cue_signature(&query.cue)? {
                let scored = self.scan_phi(&cue_sig, query, subject_filter.as_ref());
                let b = self.band_matches(
                    &scored,
                    query,
                    TIER_B_PHI_FLOOR,
                    TIER_B_PHI_HI,
                    TIER_B_BAND,
                    Tier::Similarity,
                )?;
                if !b.is_empty() {
                    return Ok(self.finalize(b, query));
                }
            }
        }

        // Tier E — semantic similarity via a Charikar-projected
        // static-text embedding. Sits below B (broader paraphrase
        // signal, lower precision than structured role-bound matching)
        // and above tier C (gist) because the embedding signal is
        // cleaner than token-bundle overlap on paraphrase queries.
        if Tier::Semantic.depth() <= query.tier_floor.depth() {
            if let Some(emb) = self.embedder.as_ref() {
                let query_hv = emb.project_text(&query.cue);
                let scored = self.scan_phi_with_pick(
                    &query_hv,
                    query,
                    |e| e.embedding_offset,
                    |e| e.embedding_popcount,
                    subject_filter.as_ref(),
                );
                let e = self.band_matches(
                    &scored,
                    query,
                    TIER_E_PHI_FLOOR,
                    TIER_E_PHI_HI,
                    TIER_E_BAND,
                    Tier::Semantic,
                )?;
                if !e.is_empty() {
                    return Ok(self.finalize(e, query));
                }
            }
        }

        // Tier C — gist similarity in the mid-confidence band. Tier C/D
        // score by raw hamming similarity over gist bundles. Known caveat:
        // majority-bundling an even token count produces density-skewed
        // gists that can inflate raw similarity between sparse pairs
        // (see phi_is_density_robust_where_raw_similarity_is_not in
        // hdc_properties). Acceptable at current floors; switching C/D to
        // phi is a calibrated follow-up.
        if Tier::Gist.depth() <= query.tier_floor.depth() {
            let query_hv = encode_query_signature(&query.cue);
            let scored =
                self.scan_signatures(&query_hv, query, |e| e.gist_offset, subject_filter.as_ref());
            let c = self.band_matches(
                &scored,
                query,
                TIER_C_SIM_FLOOR,
                1.0,
                TIER_C_BAND,
                Tier::Gist,
            )?;
            if !c.is_empty() {
                return Ok(self.finalize(c, query));
            }

            // Tier D — nearest neighbor (no threshold, low confidence)
            if Tier::NearestNeighbor.depth() <= query.tier_floor.depth() {
                let d = self.tier_d_matches(&scored, query)?;
                return Ok(self.finalize(d, query));
            }
        }

        Ok(vec![])
    }

    fn tier_a_exact(
        &self,
        query: &Query,
        subject_filter: Option<&HashSet<EpisodeId>>,
    ) -> Result<Vec<RecallMatch>> {
        // Tier A returns the candidate SET (every episode whose
        // any-cue-token is a known concept). The cascade then
        // reranks this set through tier L's IDF scoring so the most
        // token-overlapping episode surfaces first. Confidence 1.0
        // is preserved for backwards compat.
        //
        // v0.2 — temporal retrieval: when `subject_filter` is set,
        // the supplied set already contains the exact set of
        // subject episodes that pass the temporal filters, so we
        // skip the cue-token concept resolution entirely and use
        // the supplied set as the candidate pool. The subject
        // path then returns every linked episode ranked by IDF
        // against the cue (so "alice bob wireframe" surfaces
        // the alice-wireframe episodes first) — if the cue
        // shares no tokens with any subject episode, the fallback
        // returns every subject episode in id order. This is the
        // "every episode about X" path — no lexical guess required.
        let mut seen: HashSet<EpisodeId> = HashSet::new();
        if let Some(subject_set) = subject_filter {
            for &id in subject_set {
                if self
                    .scan_entry(id.raw())
                    .map(|e| e.tombstoned)
                    .unwrap_or(true)
                {
                    continue;
                }
                seen.insert(id);
            }
        } else {
            for token in tokenize(&query.cue) {
                let Some(cid) = self.concept_id_for(&token)? else {
                    continue;
                };
                for ep in self.recall_exact(cid, query.as_of)? {
                    if self
                        .scan_entry(ep.id.raw())
                        .map(|e| e.tombstoned)
                        .unwrap_or(false)
                    {
                        continue;
                    }
                    // v0.2 — temporal retrieval: enforce the time_window
                    // overlap filter even though `recall_exact` only
                    // checks `as_of`. They can coexist; `as_of` is
                    // already handled inside `recall_exact`.
                    if !query.valid_time_passes(&ep.valid_time) {
                        continue;
                    }
                    seen.insert(ep.id);
                }
            }
        }
        // Rerank the concept-matched candidate set via tier L IDF.
        let reranked = self.rerank_via_idf(&seen, query, subject_filter.is_some())?;
        Ok(reranked)
    }

    /// IDF-weighted rerank over an explicit episode-id candidate set.
    /// Used by tier A (concept candidates) and exposed for future
    /// tiers that want to seed from a precomputed index.
    ///
    /// `subject_is_filter = true` (v0.2 — temporal retrieval):
    /// the caller is in "all episodes about X" mode, so if the cue
    /// shares no tokens with any candidate we fall back to
    /// returning every candidate in id order. Otherwise the cue
    /// must drive the ranking.
    fn rerank_via_idf(
        &self,
        candidates: &HashSet<EpisodeId>,
        query: &Query,
        subject_is_filter: bool,
    ) -> Result<Vec<RecallMatch>> {
        if candidates.is_empty() {
            return Ok(vec![]);
        }
        let cue_tokens: Vec<String> = tokenize(&query.cue)
            .into_iter()
            .map(|t| t.to_lowercase())
            .collect();
        if cue_tokens.is_empty() {
            return Ok(vec![]);
        }

        let tx = self.db.begin_read()?;
        let table = tx.open_table(crate::store::TOKENS)?;
        let total_docs = self.scan_entries().len().max(1) as f32;
        let mut scores: std::collections::HashMap<u64, f32> = std::collections::HashMap::new();
        // See `tier_l_lexical`: normalize confidence against the cue's
        // achievable IDF mass, not the best score observed, so the
        // number stays absolute rather than rank-relative.
        let mut achievable = 0.0f32;
        // A cue token the store has never indexed is maximally rare —
        // score it as if df == 1. It can never be matched, so it only
        // ever raises the denominator, which is the point: asking about
        // five things when the store knows one of them must not report
        // the same confidence as asking about the one thing it knows.
        let unseen_idf = total_docs.ln() + 0.5;
        for token in &cue_tokens {
            let Some(bytes) = table.get(token.as_str())?.map(|v| v.value()) else {
                achievable += unseen_idf;
                continue;
            };
            let bitmap = RoaringBitmap::deserialize_from(bytes.as_slice())
                .map_err(|e| AgidbError::Internal(format!("tokens decode: {e}")))?;
            let df = bitmap.len() as f32;
            if df == 0.0 {
                achievable += unseen_idf;
                continue;
            }
            let idf = (total_docs / df).ln() + 0.5;
            achievable += idf;
            for id in bitmap.iter() {
                if !candidates.contains(&EpisodeId::new(id as u64)) {
                    continue;
                }
                *scores.entry(id as u64).or_insert(0.0) += idf;
            }
        }
        drop(table);
        drop(tx);

        // v0.2 — temporal retrieval: in subject-filter mode every
        // candidate must come back, even if its text shares no
        // tokens with the cue. Score-0 entries get hydrated with
        // the lowest tier-L confidence so the result is sorted by
        // cue-overlap within the subject set. Without this, a cue
        // that misses most of the subject's episodes would return
        // a single top match instead of "every episode about X".
        if subject_is_filter {
            for &id in candidates {
                scores.entry(id.raw()).or_insert(0.0);
            }
        }

        let mut scored: Vec<(EpisodeId, f32)> = scores
            .into_iter()
            .map(|(id, s)| (EpisodeId::new(id), s))
            .collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.raw().cmp(&b.0.raw()))
        });

        let mut out = Vec::new();
        for (id, score) in scored {
            if out.len() >= query.k {
                break;
            }
            let Some(ep) = self.get_episode(id)? else {
                continue;
            };
            // v0.2 — temporal retrieval: enforce the time_window
            // filter at hydration time. The IDF rerank runs against
            // the in-memory candidate set (built from cue-token
            // posting lists) without consulting `valid_time`, so the
            // filter has to be reapplied once we have the row.
            if !query.valid_time_passes(&ep.valid_time) {
                continue;
            }
            let confidence = if achievable > 0.0 {
                0.55 + (0.95 - 0.55) * (score / achievable)
            } else {
                0.55
            };
            let confidence = confidence.clamp(0.55, 0.95);
            out.push(into_match(ep, confidence, Tier::Lexical));
        }
        // If IDF didn't rank anything (e.g., only one cue token which
        // was a known concept but no other tokens), fall back to the
        // insertion order so tier A at least matches historical
        // behavior. In subject-filter mode the fallback returns every
        // candidate — "every episode about X" — so a cue that shares
        // no tokens with the subject's episodes still gets a
        // meaningful answer.
        if out.is_empty() {
            let mut sorted_ids: Vec<EpisodeId> = candidates.iter().copied().collect();
            sorted_ids.sort_by_key(|i| i.raw());
            let take_count = if subject_is_filter {
                query.k.max(sorted_ids.len())
            } else {
                query.k
            };
            for id in sorted_ids.into_iter().take(take_count) {
                let Some(ep) = self.get_episode(id)? else {
                    continue;
                };
                // v0.2 — temporal retrieval: same time_window check
                // applies to the fallback so a stale subject set
                // (built before the filter, hypothetically) can't
                // surface out-of-window episodes.
                if !query.valid_time_passes(&ep.valid_time) {
                    continue;
                }
                out.push(into_match(ep, 1.0, Tier::Exact));
                if out.len() >= query.k {
                    break;
                }
            }
        }
        Ok(out)
    }

    /// Tier L — token-level posting-list intersection. Same pattern as
    /// BM25's posting-list lookup, keyed on the canonical tokens from
    /// `crate::episode::tokenize`. Ranks candidates by how many cue
    /// tokens share a posting list with them; ties broken by episode
    /// id (deterministic).
    ///
    /// v0.2 — temporal retrieval: when `subject_filter` is set, only
    /// episodes in the subject set are kept as candidates. The
    /// time_window overlap filter is enforced via the scan directory
    /// entries (every entry's `valid_start` / `valid_end` are in
    /// memory so the check is a constant-time comparison).
    fn tier_l_lexical(
        &self,
        query: &Query,
        subject_filter: Option<&HashSet<EpisodeId>>,
    ) -> Result<Vec<RecallMatch>> {
        let cue_tokens: Vec<String> = tokenize(&query.cue)
            .into_iter()
            .map(|t| t.to_lowercase())
            .collect();
        if cue_tokens.is_empty() {
            return Ok(vec![]);
        }
        let now = query.as_of.unwrap_or_else(chrono::Utc::now);

        // One read per unique cue token: fetch each token's bitmap
        // and its document frequency. Score each candidate episode
        // with the standard BM25-style inverse-document-frequency
        // weight (sum of `log(N / df)` across the cue tokens whose
        // posting list contains the candidate). This is the fix for
        // the 0.000 exact loss on the templated bench: a generic
        // token ("any", "favorites") that hits 1000+ episodes gets a
        // ~log(1) weight ≈ 0, while a rare entity token ("bombay") that
        // hits 10 episodes gets log(10000/10) ≈ 7 — and the candidate
        // with both rare tokens ranks above the candidate with one
        // generic + one rare.
        let tx = self.db.begin_read()?;
        let table = tx.open_table(crate::store::TOKENS)?;
        let total_docs = self.scan_entries().len().max(1) as f32;
        let mut scores: std::collections::HashMap<u64, f32> = std::collections::HashMap::new();
        // Sum of IDF across every cue token that exists in the index —
        // the score a hypothetical episode containing the whole cue
        // would earn. Confidence is normalized against *this*, not
        // against the best score actually observed, so the number means
        // "how much of the cue did this episode account for" rather
        // than "did this episode come first".
        let mut achievable = 0.0f32;
        // A cue token the store has never indexed is maximally rare —
        // score it as if df == 1. It can never be matched, so it only
        // ever raises the denominator, which is the point: asking about
        // five things when the store knows one of them must not report
        // the same confidence as asking about the one thing it knows.
        let unseen_idf = total_docs.ln() + 0.5;
        for token in &cue_tokens {
            let Some(bytes) = table.get(token.as_str())?.map(|v| v.value()) else {
                achievable += unseen_idf;
                continue;
            };
            let bitmap = RoaringBitmap::deserialize_from(bytes.as_slice())
                .map_err(|e| AgidbError::Internal(format!("tokens decode: {e}")))?;
            let df = bitmap.len() as f32;
            if df == 0.0 {
                achievable += unseen_idf;
                continue;
            }
            // Standard IDF — small +0.5 keeps high-idf tokens from
            // dominating when their df is 1 (log(N/1) can be very
            // large); we just need ranking, not calibrated weights.
            let idf = (total_docs / df).ln() + 0.5;
            achievable += idf;
            for id in bitmap.iter() {
                // v0.2 — subject filter: only accumulate score for
                // episodes that pass the subject constraint.
                if let Some(set) = subject_filter {
                    if !set.contains(&EpisodeId::new(id as u64)) {
                        continue;
                    }
                }
                *scores.entry(id as u64).or_insert(0.0) += idf;
            }
        }
        drop(table);
        drop(tx);

        if scores.is_empty() {
            return Ok(vec![]);
        }

        // Tombstone + bi-temporal filter via the scan directory. The
        // time_window overlap check (v0.2) uses
        // `TimeRange::overlaps_window` directly on the scan entry's
        // `valid_start` / `valid_end` so it stays an in-memory op.
        let window = query.time_window;
        let mut candidates: Vec<(u64, f32)> = scores
            .into_iter()
            .filter(|(id, _)| {
                let Some(entry) = self.scan_entry(*id) else {
                    return false;
                };
                if entry.tombstoned {
                    return false;
                }
                if let Some(t) = query.as_of {
                    if !entry.valid_at(t) {
                        return false;
                    }
                }
                if let Some((from, to)) = window {
                    if !entry.overlaps_window(from, to) {
                        return false;
                    }
                }
                let _ = now; // keep the closure tidy
                true
            })
            .collect();
        // Sort: score DESC, then id ASC for determinism.
        candidates.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });

        // Confidence band for tier L — [0.55, 0.95], linear in the
        // fraction of the cue's achievable IDF mass this episode
        // accounts for. Normalizing against `achievable` rather than
        // the observed top score is what keeps the number absolute:
        // an episode that matches one rare token out of five reports
        // ~0.6 whether or not something better exists, instead of
        // being promoted to 0.95 for merely sorting first.
        let mut out = Vec::new();
        for &(id, score) in &candidates {
            if out.len() >= query.k {
                break;
            }
            let Some(ep) = self.get_episode(EpisodeId::new(id))? else {
                continue;
            };
            let confidence = if achievable > 0.0 {
                0.55 + (0.95 - 0.55) * (score / achievable)
            } else {
                0.55
            };
            let confidence = confidence.clamp(0.55, 0.95);
            out.push(into_match(ep, confidence, Tier::Lexical));
        }
        Ok(out)
    }
    /// Build the tier-B structured cue signature: resolve cue tokens to
    /// known concepts (exact, then case-insensitive, then fuzzy within
    /// edit distance 1) and bundle each concept's subject-role and
    /// object-role bindings. Mirrors how `bind_triple` encodes stored
    /// episodes, so a cue entity overlaps the stored signature no matter
    /// which side of the triple it appeared on.
    ///
    /// Returns `None` when no cue token resolves to a known concept —
    /// the cascade then falls straight through to tier C.
    fn structured_cue_signature(&self, cue: &str) -> Result<Option<HV>> {
        let mut names: Vec<String> = Vec::new();
        let mut seen: HashSet<ConceptId> = HashSet::new();
        for token in tokenize(cue) {
            if names.len() >= TIER_B_MAX_CUE_CONCEPTS {
                break;
            }
            let folded = token.to_lowercase();
            let cid = match self.concept_id_for(&token)? {
                Some(c) => Some(c),
                None => match self.concept_id_for_ci(&folded)? {
                    Some(c) => Some(c),
                    None => self
                        .fuzzy_concept_candidates(&folded, 1)?
                        .into_iter()
                        .next(),
                },
            };
            let Some(cid) = cid else { continue };
            if !seen.insert(cid) {
                continue;
            }
            if let Some(name) = self.concept_canonical_name(cid)? {
                names.push(name);
            }
        }
        if names.is_empty() {
            return Ok(None);
        }
        let mut parts: Vec<HV> = Vec::with_capacity(names.len() * 2);
        for name in &names {
            let hv = HV::from_name(name);
            parts.push(role_subj().bind(&hv));
            parts.push(role_obj().bind(&hv));
        }
        Ok(Some(HV::bundle(&parts)))
    }

    /// Sweep the scan directory, POPCOUNT-comparing `query_hv` against
    /// the HV each entry's `pick` offset points at in the mmap. Applies
    /// the tombstone and bi-temporal filters from the directory itself —
    /// zero redb reads, zero text re-encoding. Returns `(similarity,
    /// episode_id)` sorted by similarity descending.
    ///
    /// v0.2 — temporal retrieval: also enforces the query's
    /// `time_window` overlap filter (when set) and the `subject_filter`
    /// (when set). Both checks are O(1) per entry so the scan stays
    /// linear in the number of episodes.
    fn scan_signatures(
        &self,
        query_hv: &HV,
        query: &Query,
        pick: impl Fn(&ScanEntry) -> u64,
        subject_filter: Option<&HashSet<EpisodeId>>,
    ) -> Vec<(f32, u64)> {
        let mut scored: Vec<(f32, u64)> = Vec::with_capacity(self.scan_entries().len());
        for entry in self.scan_entries() {
            if entry.tombstoned {
                continue;
            }
            if let Some(t) = query.as_of {
                if !entry.valid_at(t) {
                    continue;
                }
            }
            if let Some((from, to)) = query.time_window {
                if !entry.overlaps_window(from, to) {
                    continue;
                }
            }
            if let Some(set) = subject_filter {
                if !set.contains(&EpisodeId::new(entry.id)) {
                    continue;
                }
            }
            let Ok(hv) = self.signatures.read(pick(entry)) else {
                continue;
            };
            scored.push((query_hv.similarity(&hv), entry.id));
        }
        top_sorted(scored)
    }

    /// Tier-B scan: phi-scored sweep of the structured episode
    /// signatures. Uses the query popcount (computed once) and each
    /// entry's cached `sig_popcount`, so the per-pair cost stays one
    /// POPCOUNT pass (the hamming) — same as the raw-similarity scan.
    fn scan_phi(
        &self,
        query_hv: &HV,
        query: &Query,
        subject_filter: Option<&HashSet<EpisodeId>>,
    ) -> Vec<(f32, u64)> {
        self.scan_phi_with_pick(
            query_hv,
            query,
            |e| e.sig_offset,
            |e| e.sig_popcount,
            subject_filter,
        )
    }

    /// Tier E / generic phi scan: like `scan_phi` but reads from an
    /// arbitrary offset per ScanEntry so tier B (structured sig) and
    /// tier E (embedding) can share the same scoring kernel.
    ///
    /// `pb_closure` selects which cached popcount to use as `pb` in
    /// the phi calculation — `sig_popcount` for tier B (the
    /// structured signature's density), `embedding_popcount` for tier
    /// E (the projected embedding's own density). Using the wrong one
    /// miscalibrates phi enough to mask the signal.
    ///
    /// v0.2 — temporal retrieval: subject filter is forwarded to keep
    /// every tier consistent.
    fn scan_phi_with_pick(
        &self,
        query_hv: &HV,
        query: &Query,
        pick: impl Fn(&ScanEntry) -> u64,
        pb_closure: impl Fn(&ScanEntry) -> u32,
        subject_filter: Option<&HashSet<EpisodeId>>,
    ) -> Vec<(f32, u64)> {
        let n = crate::hdc::D as f64;
        let pa = query_hv.popcount() as f64;
        if pa <= 0.0 || pa >= n {
            return Vec::new();
        }
        let mut scored: Vec<(f32, u64)> = Vec::with_capacity(self.scan_entries().len());
        for entry in self.scan_entries() {
            if entry.tombstoned {
                continue;
            }
            if let Some(t) = query.as_of {
                if !entry.valid_at(t) {
                    continue;
                }
            }
            if let Some((from, to)) = query.time_window {
                if !entry.overlaps_window(from, to) {
                    continue;
                }
            }
            if let Some(set) = subject_filter {
                if !set.contains(&EpisodeId::new(entry.id)) {
                    continue;
                }
            }
            let Ok(hv) = self.signatures.read(pick(entry)) else {
                continue;
            };
            if pick(entry) == 0 {
                continue;
            }
            let phi = crate::hdc::phi_from_counts(
                n,
                pa,
                pb_closure(entry) as f64,
                query_hv.hamming(&hv) as f64,
            );
            scored.push((phi, entry.id));
        }
        top_sorted(scored)
    }

    /// Turn the top of a scored scan into matches within a confidence
    /// band, hydrating only the episodes that survive the floor.
    fn band_matches(
        &self,
        scored: &[(f32, u64)],
        query: &Query,
        sim_floor: f32,
        sim_hi: f32,
        band: (f32, f32),
        tier: Tier,
    ) -> Result<Vec<RecallMatch>> {
        let mut out = Vec::new();
        for &(sim, id) in scored.iter().take_while(|(s, _)| *s >= sim_floor) {
            if out.len() >= query.k {
                break;
            }
            let Some(ep) = self.get_episode(EpisodeId::new(id))? else {
                continue;
            };
            let confidence = calibrate_band(sim, sim_floor, sim_hi, band);
            out.push(into_match(ep, confidence, tier));
        }
        Ok(out)
    }

    fn tier_d_matches(&self, scored: &[(f32, u64)], query: &Query) -> Result<Vec<RecallMatch>> {
        let mut out = Vec::new();
        for &(sim, id) in scored.iter().take(query.k) {
            let Some(ep) = self.get_episode(EpisodeId::new(id))? else {
                continue;
            };
            let confidence = (sim * TIER_D_CAP).clamp(0.0, TIER_D_CAP);
            out.push(into_match(ep, confidence, Tier::NearestNeighbor));
        }
        Ok(out)
    }

    /// Sort by confidence descending, apply `min_confidence`, and
    /// truncate to `k`. Final shape returned to the caller.
    fn finalize(&self, mut matches: Vec<RecallMatch>, query: &Query) -> Vec<RecallMatch> {
        matches.retain(|m| m.confidence >= query.min_confidence);
        matches.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        matches.truncate(query.k);
        matches
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn into_match(ep: Episode, confidence: f32, tier: Tier) -> RecallMatch {
    RecallMatch {
        episode_id: ep.id,
        text: ep.text,
        triples: ep.triples,
        confidence,
        valid_time: ep.valid_time,
        provenance: ep.provenance,
        superseded: ep.superseded_by.is_some(),
        source_tier: tier,
    }
}

/// Linearly map a similarity score from `[sim_lo, sim_hi]` into the
/// confidence band `(conf_lo, conf_hi)`. Values outside `[sim_lo, sim_hi]`
/// are clamped to the corresponding band edge.
fn calibrate_band(sim: f32, sim_lo: f32, sim_hi: f32, band: (f32, f32)) -> f32 {
    if sim <= sim_lo {
        return band.0;
    }
    if sim >= sim_hi {
        return band.1;
    }
    let t = (sim - sim_lo) / (sim_hi - sim_lo);
    band.0 + t * (band.1 - band.0)
}

/// Partial-select the top slice before sorting — callers only consume
/// the band floor + k head of the list, so fully sorting a large store's
/// scores per query is wasted work.
fn top_sorted(mut scored: Vec<(f32, u64)>) -> Vec<(f32, u64)> {
    const TOP: usize = 512;
    if scored.len() > TOP {
        scored.select_nth_unstable_by(TOP - 1, |a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(TOP);
    }
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored
}
