//! Phase 6 — consolidation. The analog of biological sleep.
//!
//! Three operations per pass:
//!
//! 1. **Cluster** every episode by hamming similarity of its stored
//!    signature. Clusters of size ≥ [`MIN_EVIDENCE`] are candidates for
//!    consolidation.
//! 2. **Create a `SemanticAtom`** for each candidate cluster, bundling
//!    the member signatures and linking back to every source
//!    `EpisodeId` for provenance.
//! 3. **Detect contradictions**. Triples sharing `(subject, predicate)`
//!    with different objects across overlapping valid-time windows
//!    are superseded — the older episode gains a `superseded_by` link
//!    + a closed `valid_time.end`.
//!
//! Decay and compaction are deferred to a follow-up; they don't move
//! the phase-6 exit criterion (atom count), only the on-disk size.
//!
//! Every pass writes a [`ConsolidationLogEntry`] into
//! [`crate::store::CONSOLIDATION_LOG`] so the audit trail is durable
//! and queryable.

use crate::error::{AgidbError, Result};
use crate::hdc::HV;
use crate::store::{ConsolidationLogEntry, Store, CONSOLIDATION_LOG, MANIFEST, SEMANTIC_ATOMS};
use crate::types::*;
use chrono::{DateTime, Utc};
use redb::ReadableTable;
use std::collections::{BTreeMap, HashSet};
use std::time::Instant;

/// Hamming similarity floor for clustering. Episodes whose stored
/// signatures share ≥ 95 % of bits cluster together. Random HV pairs
/// hover near 0.5, identical bundle inputs land at 1.0. Two episodes
/// that share two of three triple slots ("Alice recommended X" vs
/// "Bob recommended X") land in the 0.7–0.85 band — well below this
/// floor, so they stay separate clusters. The whole point of an atom
/// is "we've already seen *this exact thing*", not "something
/// adjacent in concept space".
pub const CLUSTER_SIM_THRESHOLD: f32 = 0.95;

/// Minimum cluster size that triggers semantic-atom creation.
pub const MIN_EVIDENCE: usize = 3;

/// Manifest key for the monotonic atom-id counter.
const KEY_NEXT_ATOM_ID: &str = "next_atom_id";

/// What a single `consolidate()` pass did. Returned synchronously to
/// the caller and persisted as a [`ConsolidationLogEntry`] for the
/// audit trail.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConsolidationReport {
    pub at: DateTime<Utc>,
    pub episodes_scanned: u32,
    pub semantic_atoms_created: u32,
    pub contradictions_detected: u32,
    pub atoms_decayed: u32,
    pub bytes_reclaimed: u64,
    pub elapsed_ms: u32,
}

impl Store {
    /// Run one consolidation pass synchronously. Safe to call at any
    /// time; idempotent in the sense that running back-to-back passes
    /// on an unchanged store creates no new atoms (every cluster has
    /// either already been atomized or remains below `MIN_EVIDENCE`).
    pub fn consolidate(&mut self) -> Result<ConsolidationReport> {
        let started = Instant::now();
        let at = Utc::now();

        let episodes = self.load_episodes_with_signatures()?;
        let episodes_scanned = episodes.len() as u32;

        let already_consolidated = self.collect_consolidated_episode_ids()?;

        let clusters = cluster_by_similarity(&episodes, &already_consolidated);
        let semantic_atoms_created = self.materialize_atoms(&clusters, &episodes, at)?;
        let contradictions_detected = self.detect_and_supersede_contradictions(&episodes)?;

        let elapsed_ms = started.elapsed().as_millis().min(u32::MAX as u128) as u32;
        let report = ConsolidationReport {
            at,
            episodes_scanned,
            semantic_atoms_created,
            contradictions_detected,
            atoms_decayed: 0,
            bytes_reclaimed: 0,
            elapsed_ms,
        };

        self.write_log_entry(&ConsolidationLogEntry {
            at,
            episodes_scanned: report.episodes_scanned,
            atoms_created: report.semantic_atoms_created,
            contradictions: report.contradictions_detected,
            atoms_decayed: report.atoms_decayed,
            bytes_reclaimed: report.bytes_reclaimed,
        })?;

        // Phase 10 — update the self-vector via EMA toward the bundle of
        // newly-created atom signatures. Skip if no atoms were created.
        if semantic_atoms_created > 0 {
            let atom_sigs = self.load_atom_signatures()?;
            if !atom_sigs.is_empty() {
                let bundle = crate::hdc::HV::bundle(&atom_sigs);
                let _drift =
                    self.update_self_vector(&bundle, crate::self_model::SELF_VECTOR_ALPHA)?;
            }
        }

        // Phase 10 — emit a learning event for this consolidation pass.
        self.record_event(crate::learning_log::LearningEvent::ConsolidationRun {
            atoms_created: semantic_atoms_created,
            contradictions: contradictions_detected,
            at,
        })?;

        Ok(report)
    }

    // --- internal helpers -------------------------------------------------

    fn load_episodes_with_signatures(&self) -> Result<Vec<(Episode, HV)>> {
        use crate::store::EPISODES;
        let tx = self.db.begin_read()?;
        let table = tx.open_table(EPISODES)?;
        let mut out = Vec::new();
        for entry in table.iter()? {
            let (k, v) = entry?;
            // Article XVI: an unlearned episode must not be able to
            // re-enter the store's content through a semantic atom
            // minted after the erasure. Skipping it here is what keeps
            // consolidation from laundering tombstoned text.
            if self.is_tombstoned(k.value()) {
                continue;
            }
            let bytes = v.value();
            let ep: Episode = bincode::deserialize(&bytes)
                .map_err(|e| AgidbError::Internal(format!("decode ep: {e}")))?;
            let sig = self.signatures.read(ep.signature_offset)?;
            out.push((ep, sig));
        }
        Ok(out)
    }

    /// Collect every `EpisodeId` that already appears in an existing
    /// `SemanticAtom`'s evidence list. Used to skip re-consolidating
    /// the same cluster on repeated passes.
    fn collect_consolidated_episode_ids(&self) -> Result<HashSet<EpisodeId>> {
        let tx = self.db.begin_read()?;
        let table = tx.open_table(SEMANTIC_ATOMS)?;
        let mut out = HashSet::new();
        for entry in table.iter()? {
            let (_, v) = entry?;
            let bytes = v.value();
            let atom: SemanticAtom = bincode::deserialize(&bytes)
                .map_err(|e| AgidbError::Internal(format!("decode atom: {e}")))?;
            for id in atom.evidence {
                out.insert(id);
            }
        }
        Ok(out)
    }

    fn materialize_atoms(
        &mut self,
        clusters: &[Vec<usize>],
        episodes: &[(Episode, HV)],
        at: DateTime<Utc>,
    ) -> Result<u32> {
        let mut created = 0u32;
        for cluster in clusters {
            if cluster.len() < MIN_EVIDENCE {
                continue;
            }
            // Bundle the cluster signatures into one atom signature.
            let hvs: Vec<HV> = cluster.iter().map(|&i| episodes[i].1).collect();
            let atom_hv = HV::bundle(&hvs);

            // Anchor the atom to the concept of the first triple of
            // the representative episode (cluster centroid is overkill
            // for phase 6 — picking the first member is correct in
            // spirit since cluster members share concept space).
            let representative = &episodes[cluster[0]].0;
            let concept_name = match representative.triples.first() {
                Some(t) => &t.subject,
                None => continue, // no concept anchor → no atom
            };
            let concept = match self.concept_id_for(concept_name)? {
                Some(c) => c,
                None => continue,
            };

            let atom_id = self.next_atom_id()?;
            let signature_offset = self.signatures.append(&atom_hv)?;

            let atom = SemanticAtom {
                id: atom_id,
                statement: format!(
                    "consolidated atom anchored to {concept_name} from {} episodes",
                    cluster.len()
                ),
                concept,
                evidence: cluster.iter().map(|&i| episodes[i].0.id).collect(),
                evidence_count: cluster.len() as u32,
                confidence: 0.9,
                last_referenced: at,
                signature_offset,
            };

            let tx = self.db.begin_write()?;
            {
                let mut atoms = tx.open_table(SEMANTIC_ATOMS)?;
                let bytes = bincode::serialize(&atom)
                    .map_err(|e| AgidbError::Internal(format!("encode atom: {e}")))?;
                atoms.insert(atom_id.raw(), bytes)?;
            }
            tx.commit()?;

            created += 1;
        }
        Ok(created)
    }

    fn detect_and_supersede_contradictions(&mut self, episodes: &[(Episode, HV)]) -> Result<u32> {
        // Group every triple by (subject, predicate). Each group with
        // ≥ 2 entries is a candidate for supersession — the rule is
        // "same subject + predicate but different object across
        // overlapping valid time" per the bi-temporal contract.
        let mut by_sp: BTreeMap<(String, String), Vec<TripleSlot>> = BTreeMap::new();
        for (ep, _) in episodes {
            if ep.superseded_by.is_some() {
                continue;
            }
            for tr in &ep.triples {
                by_sp
                    .entry((tr.subject.clone(), tr.predicate.clone()))
                    .or_default()
                    .push(TripleSlot {
                        episode_id: ep.id,
                        object: tr.object.clone(),
                        valid_start: ep.valid_time.start,
                    });
            }
        }

        let mut count = 0u32;
        for (_, mut entries) in by_sp {
            if entries.len() < 2 {
                continue;
            }
            entries.sort_by_key(|e| e.valid_start);
            for i in 1..entries.len() {
                let older = &entries[i - 1];
                let newer = &entries[i];
                if older.object == newer.object {
                    continue;
                }
                // Re-read the older episode through the store so we
                // see the current superseded_by state (we may have
                // already superseded it in this loop).
                if let Some(ep) = self.get_episode(older.episode_id)? {
                    if ep.superseded_by.is_none() {
                        self.supersede(older.episode_id, newer.episode_id)?;
                        count += 1;
                    }
                }
            }
        }
        Ok(count)
    }

    /// Load the signatures of all semantic atoms — used by the
    /// self-vector EMA update.
    fn load_atom_signatures(&self) -> Result<Vec<crate::hdc::HV>> {
        let tx = self.db.begin_read()?;
        let table = tx.open_table(SEMANTIC_ATOMS)?;
        let mut out = Vec::new();
        for entry in table.iter()? {
            let (_, v) = entry?;
            let atom: SemanticAtom = bincode::deserialize(&v.value())
                .map_err(|e| AgidbError::Internal(format!("decode atom: {e}")))?;
            if let Ok(sig) = self.signatures.read(atom.signature_offset) {
                out.push(sig);
            }
        }
        Ok(out)
    }

    fn write_log_entry(&self, entry: &ConsolidationLogEntry) -> Result<()> {
        let tx = self.db.begin_write()?;
        {
            let mut log = tx.open_table(CONSOLIDATION_LOG)?;
            // Key by milliseconds since epoch so the log is naturally
            // ordered. Collisions on the same millisecond are
            // vanishingly rare for a synchronous consolidate() and
            // would be a no-op overwrite anyway.
            let key = entry.at.timestamp_millis() as u64;
            let bytes = bincode::serialize(entry)
                .map_err(|e| AgidbError::Internal(format!("encode log: {e}")))?;
            log.insert(key, bytes)?;
        }
        tx.commit()?;
        Ok(())
    }

    fn next_atom_id(&self) -> Result<SemanticAtomId> {
        let tx = self.db.begin_write()?;
        let id_value: u64;
        {
            let mut manifest = tx.open_table(MANIFEST)?;
            let raw = manifest.get(KEY_NEXT_ATOM_ID)?.map(|v| v.value());
            id_value = match raw {
                Some(bytes) => bincode::deserialize(&bytes)
                    .map_err(|e| AgidbError::Internal(format!("decode atom-id: {e}")))?,
                None => 1,
            };
            let next = bincode::serialize(&(id_value + 1))
                .map_err(|e| AgidbError::Internal(format!("encode atom-id: {e}")))?;
            manifest.insert(KEY_NEXT_ATOM_ID, next)?;
        }
        tx.commit()?;
        Ok(SemanticAtomId::new(id_value))
    }
}

/// One row of `(EpisodeId, object, valid_start)` used by the
/// contradiction-detection grouping. Factored out so the BTreeMap
/// signature stays under clippy's "very complex type" threshold.
struct TripleSlot {
    episode_id: EpisodeId,
    object: String,
    valid_start: DateTime<Utc>,
}

// ---------------------------------------------------------------------------
// Clustering
// ---------------------------------------------------------------------------

/// Greedy single-pass clustering: pick the first un-assigned episode,
/// pull every other un-assigned episode within similarity threshold
/// into the same cluster, repeat. O(N²) worst case; good enough for
/// the phase-6 entry. A locality-sensitive hash (LSH) pre-filter
/// lives in the phase-6 follow-up.
fn cluster_by_similarity(episodes: &[(Episode, HV)], skip: &HashSet<EpisodeId>) -> Vec<Vec<usize>> {
    let mut clusters: Vec<Vec<usize>> = Vec::new();
    let mut assigned: HashSet<usize> = HashSet::new();
    for i in 0..episodes.len() {
        if assigned.contains(&i) || skip.contains(&episodes[i].0.id) {
            continue;
        }
        let mut cluster = vec![i];
        assigned.insert(i);
        for j in (i + 1)..episodes.len() {
            if assigned.contains(&j) || skip.contains(&episodes[j].0.id) {
                continue;
            }
            if episodes[i].1.similarity(&episodes[j].1) >= CLUSTER_SIM_THRESHOLD {
                cluster.push(j);
                assigned.insert(j);
            }
        }
        clusters.push(cluster);
    }
    clusters
}
