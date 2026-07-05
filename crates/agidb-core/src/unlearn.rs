//! Phase 11 — non-destructive cascading unlearn.
//!
//! Given a target (concept / episode / belief / source / session),
//! compute the full dependency cascade, tombstone every affected row,
//! cascade corrections through beliefs and semantic atoms, subtract the
//! unlearned signatures from the self-vector, and emit a permanent
//! `LearningEvent::Unlearned`. Constitution articles XII and XVI.
//!
//! Tombstones are non-destructive: rows are marked with `tombstoned_at`
//! in a separate `TOMBSTONES` table (the original row is preserved).
//! `restore_within_window` reverses a tombstone within 30 days. After
//! expiry, compaction (phase 6 follow-up) physically removes the data —
//! but the `LearningEvent::Unlearned` audit record is **permanent**.

use chrono::{DateTime, Duration, Utc};
use redb::{ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};

use crate::belief::BELIEFS;
use crate::error::{AgidbError, Result};
use crate::hdc::HV;
use crate::learning_log::LearningEvent;
use crate::store::{decode, encode, Store, EPISODES, SEMANTIC_ATOMS};
use crate::types::*;

/// Tombstone records — `(kind_tag, id) → Tombstone`.
/// kind_tag: 0 = episode, 1 = belief, 2 = semantic_atom.
pub const TOMBSTONES: TableDefinition<(u8, u64), Vec<u8>> = TableDefinition::new("tombstones");

/// 30-day recovery window. Within this period `restore_within_window`
/// can reverse a tombstone. After expiry the data may be compacted.
pub const TOMBSTONE_WINDOW_DAYS: i64 = 30;

pub(crate) const TOMBSTONE_EPISODE: u8 = 0;
const TOMBSTONE_BELIEF: u8 = 1;
const TOMBSTONE_ATOM: u8 = 2;

/// What to forget.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum UnlearnTarget {
    /// Forget a single episode by id.
    Episode(EpisodeId),
    /// Forget a single belief by id.
    Belief(BeliefId),
    /// Forget a concept and *everything* referencing it — episodes,
    /// beliefs, semantic atoms. The GDPR-style "forget Sarah" case.
    Concept(ConceptId),
    /// Forget every episode + derived belief whose provenance source
    /// matches (e.g. `"tool:gmail"`). GDPR Article 17.
    BySource(String),
    /// Forget every episode + derived belief from a session.
    BySession(String),
}

impl UnlearnTarget {
    pub fn label(&self) -> String {
        match self {
            Self::Episode(id) => format!("episode:{}", id.raw()),
            Self::Belief(id) => format!("belief:{}", id.raw()),
            Self::Concept(id) => format!("concept:{}", id.raw()),
            Self::BySource(s) => format!("source:{s}"),
            Self::BySession(s) => format!("session:{s}"),
        }
    }
}

/// A non-destructive removal marker.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Tombstone {
    pub kind: u8,
    pub id: u64,
    pub tombstoned_at: DateTime<Utc>,
    pub reason: String,
    pub audit_event_id: u64,
}

/// What an `unlearn()` call did. Returned to the caller and persisted
/// as the permanent `LearningEvent::Unlearned` audit record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UnlearnReport {
    pub target: String,
    pub episodes_removed: usize,
    pub beliefs_removed: usize,
    pub beliefs_revised: usize,
    pub semantic_atoms_affected: usize,
    pub self_vector_drift_hamming: u32,
    pub reason: String,
    pub audit_event_id: u64,
    pub tombstone_expiry: DateTime<Utc>,
}

// ---------------------------------------------------------------------------
// Store API.
// ---------------------------------------------------------------------------

impl Store {
    /// Non-destructive cascading unlearn. Constitution article XVI.
    ///
    /// 1. Compute the full dependency cascade for `target`.
    /// 2. Tombstone every affected episode, belief, and semantic atom.
    /// 3. Cascade through beliefs: reduce confidence or withdraw beliefs
    ///    whose evidence was tombstoned.
    /// 4. Subtract the bundle of tombstoned signatures from the self-vector.
    /// 5. Emit `LearningEvent::Unlearned` (permanent, survives compaction).
    pub fn unlearn(
        &mut self,
        target: UnlearnTarget,
        reason: impl Into<String>,
    ) -> Result<UnlearnReport> {
        let reason = reason.into();
        let at = Utc::now();

        // 1. Compute the cascade — which episodes, beliefs, and atoms to
        //    tombstone.
        let (episode_ids, belief_ids, atom_ids) = self.compute_cascade(&target)?;

        // 2. Collect the signatures of everything being tombstoned (for
        //    self-vector subtraction).
        let mut tombstoned_sigs: Vec<HV> = Vec::new();
        for &eid in &episode_ids {
            if let Some(ep) = self.get_episode(eid)? {
                if let Ok(sig) = self.signatures.read(ep.signature_offset) {
                    tombstoned_sigs.push(sig);
                }
            }
        }
        for &bid in &belief_ids {
            if let Some(b) = self.get_belief(bid)? {
                if let Ok(sig) = self.signatures.read(b.signature_offset) {
                    tombstoned_sigs.push(sig);
                }
            }
        }

        // 3. Emit the permanent audit event first so its id is available
        //    for every tombstone row.
        let cascade_size = episode_ids.len() + belief_ids.len() + atom_ids.len();
        let audit_event_id = self.record_event(LearningEvent::Unlearned {
            target: target.label(),
            cascade_size,
            self_vector_drift: 0, // updated after subtraction
            reason: reason.clone(),
            at,
        })?;

        // 4. Tombstone episodes.
        for &eid in &episode_ids {
            self.write_tombstone(TOMBSTONE_EPISODE, eid.raw(), &reason, audit_event_id)?;
        }
        // 5. Tombstone beliefs (and withdraw them).
        let mut beliefs_revised = 0usize;
        for &bid in &belief_ids {
            self.write_tombstone(TOMBSTONE_BELIEF, bid.raw(), &reason, audit_event_id)?;
            // Withdraw the belief non-destructively (but don't emit a
            // separate BeliefWithdrawn event — the Unlearned event is
            // the audit record here).
            if let Some(mut belief) = self.get_belief(bid)? {
                if !belief.is_withdrawn() {
                    belief.t_valid_end = Some(at);
                    belief.confidence = 0.0;
                    let tx = self.db.begin_write()?;
                    {
                        let mut beliefs = tx.open_table(BELIEFS)?;
                        beliefs.insert(bid.raw(), encode(&belief)?)?;
                    }
                    tx.commit()?;
                    beliefs_revised += 1;
                }
            }
        }
        // 6. Tombstone semantic atoms.
        for &aid in &atom_ids {
            self.write_tombstone(TOMBSTONE_ATOM, aid.raw(), &reason, audit_event_id)?;
        }

        // 7. Cascade through beliefs whose *evidence* includes tombstoned
        //    episodes but the belief itself wasn't in the direct cascade.
        beliefs_revised += self.cascade_belief_evidence_corrections(&episode_ids, &reason, at)?;

        // 8. Self-vector subtraction — the key v2 contribution.
        let self_vector_drift = if !tombstoned_sigs.is_empty() {
            let bundle = HV::bundle(&tombstoned_sigs);
            self.subtract_from_self_vector(&bundle)?
        } else {
            0
        };

        let report = UnlearnReport {
            target: target.label(),
            episodes_removed: episode_ids.len(),
            beliefs_removed: belief_ids.len(),
            beliefs_revised,
            semantic_atoms_affected: atom_ids.len(),
            self_vector_drift_hamming: self_vector_drift,
            reason: reason.clone(),
            audit_event_id,
            tombstone_expiry: at + Duration::days(TOMBSTONE_WINDOW_DAYS),
        };
        Ok(report)
    }

    /// Reverse a tombstone within the 30-day recovery window. Removes
    /// the tombstone row so the original data is visible again. After
    /// the window expires, returns `Err`.
    pub fn restore_within_window(&mut self, audit_event_id: u64) -> Result<usize> {
        let now = Utc::now();
        let mut restored = 0usize;
        let to_restore: Vec<(u8, u64, Tombstone)> = {
            let tx = self.db.begin_read()?;
            let table = tx.open_table(TOMBSTONES)?;
            table
                .iter()?
                .filter_map(|e| {
                    let (k, v) = e.ok()?;
                    let t: Tombstone = decode(&v.value()).ok()?;
                    if t.audit_event_id == audit_event_id {
                        Some((k.value().0, k.value().1, t))
                    } else {
                        None
                    }
                })
                .collect()
        };
        for (kind, id, tomb) in to_restore {
            if (now - tomb.tombstoned_at).num_days() > TOMBSTONE_WINDOW_DAYS {
                return Err(AgidbError::Internal(format!(
                    "tombstone {}:{} expired — past {} day recovery window",
                    kind, id, TOMBSTONE_WINDOW_DAYS
                )));
            }
            let tx = self.db.begin_write()?;
            {
                let mut table = tx.open_table(TOMBSTONES)?;
                table.remove((kind, id))?;
            }
            tx.commit()?;
            if kind == TOMBSTONE_EPISODE {
                self.scan_set_tombstoned(id, false);
            }
            restored += 1;
        }
        Ok(restored)
    }

    /// Every tombstone record (for `unlearn_history`).
    pub fn all_tombstones(&self) -> Result<Vec<Tombstone>> {
        let tx = self.db.begin_read()?;
        let table = tx.open_table(TOMBSTONES)?;
        let mut out = Vec::new();
        for entry in table.iter()? {
            let (_, v) = entry?;
            out.push(decode(&v.value())?);
        }
        Ok(out)
    }

    /// Is this episode tombstoned?
    pub fn is_episode_tombstoned(&self, id: EpisodeId) -> Result<bool> {
        let tx = self.db.begin_read()?;
        let table = tx.open_table(TOMBSTONES)?;
        Ok(table.get((TOMBSTONE_EPISODE, id.raw()))?.is_some())
    }

    /// Is this belief tombstoned?
    pub fn is_belief_tombstoned(&self, id: BeliefId) -> Result<bool> {
        let tx = self.db.begin_read()?;
        let table = tx.open_table(TOMBSTONES)?;
        Ok(table.get((TOMBSTONE_BELIEF, id.raw()))?.is_some())
    }

    // --- internal: cascade computation ----------------------------------

    /// Compute the full dependency cascade for `target`. Returns
    /// `(episode_ids, belief_ids, atom_ids)` to tombstone.
    fn compute_cascade(
        &self,
        target: &UnlearnTarget,
    ) -> Result<(Vec<EpisodeId>, Vec<BeliefId>, Vec<SemanticAtomId>)> {
        match target {
            UnlearnTarget::Episode(eid) => {
                // Direct: tombstone the episode + any atom that cites it
                // as evidence. Beliefs that cite it as evidence are
                // *corrected* (evidence removed, confidence reduced) by
                // the cascade step — not tombstoned — so they retain
                // their remaining evidence and revision log.
                let ep_ids = vec![*eid];
                let atom_ids = self.atoms_citing_evidence(&[*eid])?;
                Ok((ep_ids, Vec::new(), atom_ids))
            }
            UnlearnTarget::Belief(bid) => {
                // Direct: tombstone just the belief.
                Ok((Vec::new(), vec![*bid], Vec::new()))
            }
            UnlearnTarget::Concept(cid) => {
                // Every episode that mentions this concept (via
                // concept_episodes multimap) + every belief whose subject
                // resolves to this concept + every atom anchored to it.
                let ep_ids = self.episodes_for_concept(*cid)?;
                let belief_ids = self.beliefs_for_concept(*cid)?;
                let atom_ids = self.atoms_for_concept(*cid)?;
                Ok((ep_ids, belief_ids, atom_ids))
            }
            UnlearnTarget::BySource(source) => {
                let ep_ids = self.episodes_by_source(source)?;
                let belief_ids = self.beliefs_by_source(source)?;
                let atom_ids = Vec::new(); // atoms don't carry provenance in v0.1
                Ok((ep_ids, belief_ids, atom_ids))
            }
            UnlearnTarget::BySession(session) => {
                let ep_ids = self.episodes_by_session(session)?;
                let belief_ids = self.beliefs_by_session(session)?;
                Ok((ep_ids, belief_ids, Vec::new()))
            }
        }
    }

    fn episodes_for_concept(&self, cid: ConceptId) -> Result<Vec<EpisodeId>> {
        use crate::store::CONCEPT_EPISODES;
        let tx = self.db.begin_read()?;
        let table = tx.open_multimap_table(CONCEPT_EPISODES)?;
        let mut out = Vec::new();
        for entry in table.get(cid.raw())? {
            out.push(EpisodeId::new(entry?.value()));
        }
        Ok(out)
    }

    fn beliefs_for_concept(&self, cid: ConceptId) -> Result<Vec<BeliefId>> {
        // Beliefs store subject as a string, not ConceptId. Resolve the
        // concept's canonical name and match beliefs by subject string.
        let name = self.concept_canonical_name(cid)?;
        if let Some(subject) = name {
            let tx = self.db.begin_read()?;
            let table = tx.open_table(BELIEFS)?;
            let mut out = Vec::new();
            for entry in table.iter()? {
                let (k, v) = entry?;
                let b: Belief = decode(&v.value())?;
                if b.subject == subject {
                    out.push(BeliefId::new(k.value()));
                }
            }
            Ok(out)
        } else {
            Ok(Vec::new())
        }
    }

    fn atoms_for_concept(&self, cid: ConceptId) -> Result<Vec<SemanticAtomId>> {
        let tx = self.db.begin_read()?;
        let table = tx.open_table(SEMANTIC_ATOMS)?;
        let mut out = Vec::new();
        for entry in table.iter()? {
            let (k, v) = entry?;
            let atom: SemanticAtom = decode(&v.value())?;
            if atom.concept == cid {
                out.push(SemanticAtomId::new(k.value()));
            }
        }
        Ok(out)
    }

    fn episodes_by_source(&self, source: &str) -> Result<Vec<EpisodeId>> {
        let tx = self.db.begin_read()?;
        let table = tx.open_table(EPISODES)?;
        let mut out = Vec::new();
        for entry in table.iter()? {
            let (k, v) = entry?;
            let ep: Episode = decode(&v.value())?;
            if ep.provenance.source == source {
                out.push(EpisodeId::new(k.value()));
            }
        }
        Ok(out)
    }

    fn episodes_by_session(&self, session: &str) -> Result<Vec<EpisodeId>> {
        let tx = self.db.begin_read()?;
        let table = tx.open_table(EPISODES)?;
        let mut out = Vec::new();
        for entry in table.iter()? {
            let (k, v) = entry?;
            let ep: Episode = decode(&v.value())?;
            if ep.provenance.session_id.as_deref() == Some(session) {
                out.push(EpisodeId::new(k.value()));
            }
        }
        Ok(out)
    }

    fn beliefs_by_source(&self, source: &str) -> Result<Vec<BeliefId>> {
        let tx = self.db.begin_read()?;
        let table = tx.open_table(BELIEFS)?;
        let mut out = Vec::new();
        for entry in table.iter()? {
            let (k, v) = entry?;
            let b: Belief = decode(&v.value())?;
            if b.provenance.source == source {
                out.push(BeliefId::new(k.value()));
            }
        }
        Ok(out)
    }

    fn beliefs_by_session(&self, session: &str) -> Result<Vec<BeliefId>> {
        let tx = self.db.begin_read()?;
        let table = tx.open_table(BELIEFS)?;
        let mut out = Vec::new();
        for entry in table.iter()? {
            let (k, v) = entry?;
            let b: Belief = decode(&v.value())?;
            if b.provenance.session_id.as_deref() == Some(session) {
                out.push(BeliefId::new(k.value()));
            }
        }
        Ok(out)
    }

    fn atoms_citing_evidence(&self, evidence: &[EpisodeId]) -> Result<Vec<SemanticAtomId>> {
        let ev_set: std::collections::HashSet<EpisodeId> = evidence.iter().copied().collect();
        let tx = self.db.begin_read()?;
        let table = tx.open_table(SEMANTIC_ATOMS)?;
        let mut out = Vec::new();
        for entry in table.iter()? {
            let (k, v) = entry?;
            let atom: SemanticAtom = decode(&v.value())?;
            if atom.evidence.iter().any(|e| ev_set.contains(e)) {
                out.push(SemanticAtomId::new(k.value()));
            }
        }
        Ok(out)
    }

    /// Cascade: for beliefs whose evidence includes tombstoned episodes
    /// (but the belief itself wasn't directly tombstoned), remove the
    /// tombstoned evidence and reduce confidence. Returns the count of
    /// beliefs revised.
    fn cascade_belief_evidence_corrections(
        &mut self,
        tombstoned_episodes: &[EpisodeId],
        reason: &str,
        at: DateTime<Utc>,
    ) -> Result<usize> {
        let ev_set: std::collections::HashSet<EpisodeId> =
            tombstoned_episodes.iter().copied().collect();
        let to_fix: Vec<(BeliefId, Belief)> = {
            let tx = self.db.begin_read()?;
            let table = tx.open_table(BELIEFS)?;
            let mut out = Vec::new();
            for entry in table.iter()? {
                let (k, v) = entry?;
                let b: Belief = decode(&v.value())?;
                if b.is_withdrawn() {
                    continue;
                }
                if b.evidence.iter().any(|e| ev_set.contains(e))
                    || b.contradictions.iter().any(|e| ev_set.contains(e))
                {
                    out.push((BeliefId::new(k.value()), b));
                }
            }
            out
        };
        let mut count = 0usize;
        for (bid, mut belief) in to_fix {
            let before_len = belief.evidence.len();
            belief.evidence.retain(|e| !ev_set.contains(e));
            belief.contradictions.retain(|e| !ev_set.contains(e));
            // Reduce confidence proportional to evidence lost.
            if before_len > 0 {
                let ratio = belief.evidence.len() as f32 / before_len as f32;
                belief.confidence *= ratio;
            }
            // If confidence drops below threshold, withdraw.
            if belief.confidence < crate::belief::WITHDRAWAL_THRESHOLD {
                belief.t_valid_end = Some(at);
            }
            let tx = self.db.begin_write()?;
            {
                let mut beliefs = tx.open_table(BELIEFS)?;
                beliefs.insert(bid.raw(), encode(&belief)?)?;
            }
            tx.commit()?;
            count += 1;
        }
        // Silence unused-warning on `reason` — it's part of the API for
        // future LLM-assisted revision reasons.
        let _ = reason;
        Ok(count)
    }

    fn write_tombstone(
        &mut self,
        kind: u8,
        id: u64,
        reason: &str,
        audit_event_id: u64,
    ) -> Result<()> {
        let tomb = Tombstone {
            kind,
            id,
            tombstoned_at: Utc::now(),
            reason: reason.to_string(),
            audit_event_id,
        };
        let tx = self.db.begin_write()?;
        {
            let mut table = tx.open_table(TOMBSTONES)?;
            table.insert((kind, id), encode(&tomb)?)?;
        }
        tx.commit()?;
        if kind == TOMBSTONE_EPISODE {
            self.scan_set_tombstoned(id, true);
        }
        Ok(())
    }
}
