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

use crate::episode::{encode_query_signature, role_obj, role_subj, tokenize};
use crate::error::{AgidbError, Result};
use crate::hdc::HV;
use crate::store::{ScanEntry, Store, SEMANTIC_ATOMS};
use crate::types::*;
use redb::ReadableTable;
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

/// Tier-C similarity floor. Two random HVs have expected similarity
/// ≈ 0.5; the floor sits a few percent above that to keep noise out
/// of the high-confidence band.
const TIER_C_SIM_FLOOR: f32 = 0.55;

/// Linear map ranges for confidence calibration.
const TIER_B_BAND: (f32, f32) = (0.6, 0.95);
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
        // Tier A — exact concept lookup
        if Tier::Exact.depth() <= query.tier_floor.depth() {
            let a = self.tier_a_exact(query)?;
            if !a.is_empty() {
                return Ok(self.finalize(a, query));
            }
        }

        // Tier B — structured similarity. Resolve cue tokens to known
        // concepts (case-insensitive + fuzzy), bundle their role-bound
        // HVs into a structured cue signature, and phi-score the stored
        // episode signatures (density-corrected; raw similarity inflates
        // on AND-bundled sparse vectors — see phi_from_counts).
        if Tier::Similarity.depth() <= query.tier_floor.depth() {
            if let Some(cue_sig) = self.structured_cue_signature(&query.cue)? {
                let scored = self.scan_phi(&cue_sig, query);
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

        // Tier C — gist similarity in the mid-confidence band. Tier C/D
        // score by raw hamming similarity over gist bundles. Known caveat:
        // majority-bundling an even token count produces density-skewed
        // gists that can inflate raw similarity between sparse pairs
        // (see phi_is_density_robust_where_raw_similarity_is_not in
        // hdc_properties). Acceptable at current floors; switching C/D to
        // phi is a calibrated follow-up.
        if Tier::Gist.depth() <= query.tier_floor.depth() {
            let query_hv = encode_query_signature(&query.cue);
            let scored = self.scan_signatures(&query_hv, query, |e| e.gist_offset);
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

    fn tier_a_exact(&self, query: &Query) -> Result<Vec<RecallMatch>> {
        let mut out = Vec::new();
        let mut seen: HashSet<EpisodeId> = HashSet::new();
        for token in tokenize(&query.cue) {
            let Some(cid) = self.concept_id_for(&token)? else {
                continue;
            };
            for ep in self.recall_exact(cid, query.as_of)? {
                // Phase 11 — skip tombstoned episodes. The scan
                // directory answers without touching redb.
                if self
                    .scan_entry(ep.id.raw())
                    .map(|e| e.tombstoned)
                    .unwrap_or(false)
                {
                    continue;
                }
                if seen.insert(ep.id) {
                    out.push(into_match(ep, 1.0, Tier::Exact));
                }
            }
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
    fn scan_signatures(
        &self,
        query_hv: &HV,
        query: &Query,
        pick: impl Fn(&ScanEntry) -> u64,
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
    fn scan_phi(&self, query_hv: &HV, query: &Query) -> Vec<(f32, u64)> {
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
            let Ok(hv) = self.signatures.read(entry.sig_offset) else {
                continue;
            };
            let phi = crate::hdc::phi_from_counts(
                n,
                pa,
                entry.sig_popcount as f64,
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
