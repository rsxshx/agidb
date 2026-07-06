//! Domain types stored by layer 3 and exposed through the public API.
//!
//! Every term here corresponds to a glossary entry in `CONTEXT.md`. Drift
//! between this file and the glossary is a signal for `/grill-with-docs`.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Identifiers
// ---------------------------------------------------------------------------

macro_rules! id_newtype {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(
            Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize,
        )]
        #[repr(transparent)]
        pub struct $name(pub u64);

        impl $name {
            pub const fn new(raw: u64) -> Self {
                Self(raw)
            }

            pub const fn raw(self) -> u64 {
                self.0
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}({})", stringify!($name), self.0)
            }
        }
    };
}

id_newtype!(
    /// Identifier for an [`Episode`].
    EpisodeId
);

id_newtype!(
    /// Identifier for a [`Concept`].
    ConceptId
);

id_newtype!(
    /// Identifier for a [`SemanticAtom`].
    SemanticAtomId
);

id_newtype!(
    /// Identifier for a [`Triple`] row.
    TripleId
);

id_newtype!(
    /// Identifier for a [`Goal`] (floor 6). Phase 9.
    GoalId
);

id_newtype!(
    /// Identifier for a [`Belief`] (floor 6). Phase 9.
    BeliefId
);

// ---------------------------------------------------------------------------
// Time & provenance
// ---------------------------------------------------------------------------

/// Closed-open valid-time interval `[start, end)`. `end = None` means
/// "currently valid"; the consolidation loop sets it when a fact is
/// superseded.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: DateTime<Utc>,
    pub end: Option<DateTime<Utc>>,
}

impl TimeRange {
    pub fn point(t: DateTime<Utc>) -> Self {
        Self {
            start: t,
            end: None,
        }
    }

    /// `true` iff `at` is inside `[start, end)`.
    pub fn contains(&self, at: DateTime<Utc>) -> bool {
        at >= self.start && self.end.is_none_or(|e| at < e)
    }
}

/// Attribution for a write. Every Episode and SemanticAtom traces back
/// to a Provenance record.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    pub source: String,
    pub session_id: Option<String>,
    pub trace_id: Option<String>,
    /// Free-form bag for caller-defined annotations. Kept as
    /// `BTreeMap` for deterministic serialization.
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
}

impl Default for Provenance {
    fn default() -> Self {
        Self {
            source: "unknown".into(),
            session_id: None,
            trace_id: None,
            metadata: BTreeMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Triples, episodes, concepts, semantic atoms
// ---------------------------------------------------------------------------

/// A `(subject, predicate, object)` tuple extracted by layer 2. Each
/// triple carries its own confidence score and a back-reference to the
/// source episode.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Triple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f32,
    pub episode_id: EpisodeId,
}

/// A single stored observation. The fundamental unit of episodic memory.
///
/// Bi-temporal: `valid_time` says when the fact was true in the world,
/// `t_tx_start` says when agidb learned it. `superseded_by` is set by
/// the consolidation loop when a later contradicting fact replaces this
/// one — the row is preserved, never deleted (constitution article V).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Episode {
    pub id: EpisodeId,
    pub text: String,
    /// Byte offset into `signatures.dat` where this episode's HV lives.
    pub signature_offset: u64,
    /// Byte offset into `signatures.dat` where the gist HV of `text`
    /// lives. Written by `Store::observe` so the tier-C/D scan reads
    /// stored signatures instead of re-encoding text per query.
    /// `#[serde(default)]` keeps pre-v2 JSONL exports importable.
    #[serde(default)]
    pub gist_offset: u64,
    /// Byte offset into `signatures.dat` where the projected semantic
    /// embedding of `text` lives. Written by `Store::observe` when an
    /// embedder is supplied; falls back to `signature_offset` for
    /// stores pre-v3. `#[serde(default)]` keeps pre-v3 imports loadable.
    #[serde(default)]
    pub embedding_offset: u64,
    pub triples: Vec<Triple>,
    pub valid_time: TimeRange,
    /// Transaction time — when agidb learned this fact.
    pub t_tx_start: DateTime<Utc>,
    pub provenance: Provenance,
    pub confidence: f32,
    /// If non-empty, this episode was superseded by the listed one. The
    /// `t_valid_end` of `valid_time` is set in the same write.
    pub superseded_by: Option<EpisodeId>,
}

/// A canonical entity. Multiple aliases collapse to the same `ConceptId`
/// so "Sarah" and "sarah_kelly" point at the same vector.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Concept {
    pub id: ConceptId,
    pub canonical_name: String,
    pub aliases: Vec<String>,
    pub entity_type: String,
}

/// A consolidated fact produced by the consolidation loop. Always links
/// back to the source episodes via `evidence` for provenance.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SemanticAtom {
    pub id: SemanticAtomId,
    pub statement: String,
    pub concept: ConceptId,
    pub evidence: Vec<EpisodeId>,
    pub evidence_count: u32,
    pub confidence: f32,
    pub last_referenced: DateTime<Utc>,
    /// Byte offset into `signatures.dat` where the atom's bundle HV lives.
    pub signature_offset: u64,
}

// ---------------------------------------------------------------------------
// Procedural memory
// ---------------------------------------------------------------------------

/// A typed episode shape representing a workflow or skill — procedural
/// memory. Answers "how do I do X?"
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Procedure {
    pub name: String,
    pub description: String,
    pub trigger: String,
    pub preconditions: Vec<String>,
    pub steps: Vec<ProcedureStep>,
    pub postconditions: Vec<String>,
    pub provenance: Option<Provenance>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProcedureStep {
    pub description: String,
    pub tool: Option<String>,
    /// Tool-specific args, opaque to agidb. Encoded as JSON-compatible
    /// text so we don't pull `serde_json::Value` into the type for
    /// callers that don't need it.
    pub args: Option<String>,
}

// ---------------------------------------------------------------------------
// Cognitive primitives — Goals and Beliefs (floor 6). Phase 9.
// ---------------------------------------------------------------------------

/// One testable success criterion on a [`Goal`]. Free-form description
/// plus optional evidence episodes that satisfy it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SuccessCriterion {
    pub description: String,
    /// Episode ids that the agent claims satisfy this criterion.
    pub evidence: Vec<EpisodeId>,
    /// `true` once the agent marks the criterion met.
    pub met: bool,
}

/// Lifecycle state of a [`Goal`]. `Completed` and `Abandoned` are
/// terminal (constitution article XV). Every transition emits a
/// `LearningEvent::GoalStateChanged` (phase 10).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum GoalState {
    Active,
    Paused {
        since: DateTime<Utc>,
        reason: String,
    },
    Completed {
        at: DateTime<Utc>,
        evidence: Vec<EpisodeId>,
    },
    Abandoned {
        at: DateTime<Utc>,
        reason: String,
    },
}

impl GoalState {
    /// `true` for `Completed` and `Abandoned` — no further transitions
    /// are allowed out of a terminal state.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            GoalState::Completed { .. } | GoalState::Abandoned { .. }
        )
    }

    /// `true` iff the goal is currently biasing recall (only `Active`).
    pub fn is_active(&self) -> bool {
        matches!(self, GoalState::Active)
    }
}

/// What the agent wants. Floor 6. A typed state machine with optional
/// parent-child hierarchy and an HDC signature for goal-biased retrieval.
///
/// Constitution article XV: goals are a first-class typed substrate
/// primitive, not text inside an episode.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Goal {
    pub id: GoalId,
    pub parent_id: Option<GoalId>,
    pub description: String,
    pub state: GoalState,
    pub success_criteria: Vec<SuccessCriterion>,
    pub deadline: Option<DateTime<Utc>>,
    /// Byte offset into `signatures.dat` where this goal's HV lives.
    pub signature_offset: u64,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub provenance: Provenance,
}

impl Goal {
    /// Construct a new in-memory goal with `Active` state and an empty
    /// criteria list. `id`, `signature_offset`, and timestamps are
    /// populated by [`crate::store::Store::set_goal`] on persist.
    pub fn new(description: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: GoalId::new(0),
            parent_id: None,
            description: description.into(),
            state: GoalState::Active,
            success_criteria: Vec::new(),
            deadline: None,
            signature_offset: 0,
            created_at: now,
            updated_at: now,
            provenance: Provenance::default(),
        }
    }

    pub fn with_parent(mut self, parent: GoalId) -> Self {
        self.parent_id = Some(parent);
        self
    }

    pub fn with_deadline(mut self, deadline: DateTime<Utc>) -> Self {
        self.deadline = Some(deadline);
        self
    }

    pub fn with_success_criterion(mut self, description: impl Into<String>) -> Self {
        self.success_criteria.push(SuccessCriterion {
            description: description.into(),
            evidence: Vec::new(),
            met: false,
        });
        self
    }

    pub fn with_provenance(mut self, provenance: Provenance) -> Self {
        self.provenance = provenance;
        self
    }
}

/// A patch applied by [`crate::store::Store::revise_goal`]. Any `None`
/// field is left unchanged.
#[derive(Clone, Debug, Default)]
pub struct GoalPatch {
    pub description: Option<String>,
    pub deadline: Option<Option<DateTime<Utc>>>,
    pub success_criteria: Option<Vec<SuccessCriterion>>,
}

/// What a [`crate::store::Store::revise_belief`] call did. Returned to
/// the caller and persisted as a `BeliefRevision` on the belief's
/// append-only log (constitution article XVII).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RevisionReport {
    pub belief_id: BeliefId,
    pub previous_confidence: f32,
    pub new_confidence: f32,
    pub triggering_evidence: Option<EpisodeId>,
    pub reason: String,
    /// `true` iff the belief was withdrawn (confidence dropped below the
    /// withdrawal threshold).
    pub withdrawn: bool,
}

/// One entry in a [`Belief`]'s append-only `revision_log`. Replaying the
/// log reconstructs the current confidence. Constitution article XVII.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BeliefRevision {
    pub timestamp: DateTime<Utc>,
    pub previous_confidence: f32,
    pub new_confidence: f32,
    pub triggering_evidence: Option<EpisodeId>,
    pub reason: String,
}

/// What the agent thinks is true. Floor 6. A graded, revisable claim:
/// confidence in `[0.0, 1.0]`, supporting `evidence` and conflicting
/// `contradictions` (both lists of `EpisodeId`), and an append-only
/// `revision_log`. Constitution article XVII: beliefs are revised, never
/// overwritten.
///
/// `subject` and `predicate` are stored as strings (canonical concept
/// names) so beliefs are queryable by subject without a separate index
/// in v0.1; a `ConceptId`-keyed index is a phase-9 follow-up.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Belief {
    pub id: BeliefId,
    pub claim: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f32,
    pub evidence: Vec<EpisodeId>,
    pub contradictions: Vec<EpisodeId>,
    pub revision_log: Vec<BeliefRevision>,
    /// Byte offset into `signatures.dat` where this belief's HV lives.
    pub signature_offset: u64,
    pub t_valid_start: DateTime<Utc>,
    pub t_valid_end: Option<DateTime<Utc>>,
    pub t_tx_start: DateTime<Utc>,
    pub provenance: Provenance,
}

impl Belief {
    /// Construct a new in-memory belief. `id`, `signature_offset`, and
    /// `t_tx_start` are populated by [`crate::store::Store::assert_belief`].
    pub fn new(claim: impl Into<String>) -> Self {
        Self {
            id: BeliefId::new(0),
            claim: claim.into(),
            subject: String::new(),
            predicate: String::new(),
            object: String::new(),
            confidence: 0.5,
            evidence: Vec::new(),
            contradictions: Vec::new(),
            revision_log: Vec::new(),
            signature_offset: 0,
            t_valid_start: Utc::now(),
            t_valid_end: None,
            t_tx_start: Utc::now(),
            provenance: Provenance::default(),
        }
    }

    pub fn with_confidence(mut self, c: f32) -> Self {
        self.confidence = c.clamp(0.0, 1.0);
        self
    }

    /// Decompose the claim into a `(subject, predicate, object)` triple
    /// shape so `what_do_i_believe(subject)` can find it. Subjects and
    /// objects are matched case-sensitively against canonical concept
    /// names (same convention as tier-A recall).
    pub fn with_triple(
        mut self,
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        self.subject = subject.into();
        self.predicate = predicate.into();
        self.object = object.into();
        self
    }

    pub fn with_evidence(mut self, episodes: Vec<EpisodeId>) -> Self {
        self.evidence = episodes;
        self
    }

    pub fn with_provenance(mut self, provenance: Provenance) -> Self {
        self.provenance = provenance;
        self
    }

    /// `true` iff the belief is currently withdrawn (closed valid-time).
    pub fn is_withdrawn(&self) -> bool {
        self.t_valid_end.is_some()
    }
}

// ---------------------------------------------------------------------------
// Retrieval
// ---------------------------------------------------------------------------

/// Which recall tier produced a match. Lower variants are higher-quality.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Tier {
    /// Canonical entity match via the concept index.
    Exact,
    /// HDC signature similarity (POPCOUNT-driven).
    Similarity,
    /// Raw-text gist fallback.
    Gist,
    /// Best-effort nearest neighbor with `low_confidence` flag.
    NearestNeighbor,
}

impl Tier {
    /// Tier depth used for fall-through ordering. Lower depth = higher
    /// quality; recall starts at depth 0 and descends until the depth
    /// matches `tier_floor` or matches are found.
    pub const fn depth(self) -> u8 {
        match self {
            Tier::Exact => 0,
            Tier::Similarity => 1,
            Tier::Gist => 2,
            Tier::NearestNeighbor => 3,
        }
    }
}

// ---------------------------------------------------------------------------
// Recall
// ---------------------------------------------------------------------------

/// A retrieval request. Only `cue` is required; the rest have defaults
/// that match the v0.1 targets in [`crate::store`].
#[derive(Clone, Debug)]
pub struct Query {
    /// Free-text cue. Tokenized for tier-A concept lookup and encoded
    /// into a gist signature for tier-C/D fallback.
    pub cue: String,
    /// Maximum number of matches to return.
    pub k: usize,
    /// Bi-temporal "as-of" filter. Episodes whose valid_time does not
    /// contain `as_of` are excluded.
    pub as_of: Option<DateTime<Utc>>,
    /// Confidence floor — matches below this are dropped.
    pub min_confidence: f32,
    /// Deepest tier the cascade is allowed to fall through to. Default
    /// is [`Tier::NearestNeighbor`] (i.e. anything goes).
    pub tier_floor: Tier,
    /// Goal-bias weight in `[0.0, 1.0]`. When > 0, active goals' HDC
    /// signatures up-weight episode matches semantically related to
    /// those goals. `0.0` (the default) disables biasing so recall is
    /// purely cue-driven. Phase 9.
    pub goal_bias_weight: f32,
}

impl Query {
    /// Construct a query with sensible defaults: `k = 10`,
    /// `min_confidence = 0.0`, `tier_floor = NearestNeighbor`,
    /// `goal_bias_weight = 0.0`.
    pub fn cue(text: impl Into<String>) -> Self {
        Self {
            cue: text.into(),
            k: 10,
            as_of: None,
            min_confidence: 0.0,
            tier_floor: Tier::NearestNeighbor,
            goal_bias_weight: 0.0,
        }
    }

    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    pub fn with_as_of(mut self, as_of: DateTime<Utc>) -> Self {
        self.as_of = Some(as_of);
        self
    }

    pub fn with_min_confidence(mut self, c: f32) -> Self {
        self.min_confidence = c;
        self
    }

    pub fn with_tier_floor(mut self, t: Tier) -> Self {
        self.tier_floor = t;
        self
    }

    /// Enable goal-biased retrieval with the given weight (clamped to
    /// `[0, 1]`). Active goals up-weight related matches. Phase 9.
    pub fn with_goal_bias(mut self, weight: f32) -> Self {
        self.goal_bias_weight = weight.clamp(0.0, 1.0);
        self
    }
}

/// The result of a recall. Per [constitution](../../.specify/memory/constitution.md)
/// article VI, `Recall::matches` is never empty under the default
/// `tier_floor`; the deepest tier always returns nearest neighbors.
///
/// `semantic_atoms` carries any consolidated `SemanticAtom` rows
/// matching the cue's concept tokens — phase 6 surfaces these as a
/// parallel result lane to the episodic matches.
#[derive(Clone, Debug)]
pub struct Recall {
    pub matches: Vec<RecallMatch>,
    pub semantic_atoms: Vec<SemanticMatch>,
    /// `true` iff goal-biased reweighting was applied. Phase 9.
    pub goal_biased: bool,
    /// The active goals that biased this recall (empty if unbiased).
    /// Phase 9.
    pub active_goals: Vec<GoalId>,
    /// The shallowest tier that contributed at least one match.
    pub tier_used: Tier,
    /// Wall-clock elapsed time of the recall call, in milliseconds.
    pub elapsed_ms: u32,
}

/// One row in a `Recall`. Tier-specific confidence calibration is
/// applied by the tier that produced the match.
#[derive(Clone, Debug, PartialEq)]
pub struct RecallMatch {
    pub episode_id: EpisodeId,
    pub text: String,
    pub triples: Vec<Triple>,
    pub confidence: f32,
    pub valid_time: TimeRange,
    pub provenance: Provenance,
    /// `true` iff this episode has a `superseded_by` link set.
    pub superseded: bool,
    pub source_tier: Tier,
}

/// A consolidated semantic atom surfaced as part of a `Recall`. Always
/// carries the list of source `EpisodeId`s in `evidence` so callers
/// can drill back to the verbatim observations behind the atom (per
/// constitution article VII — provenance always).
#[derive(Clone, Debug, PartialEq)]
pub struct SemanticMatch {
    pub atom_id: SemanticAtomId,
    pub statement: String,
    pub concept: ConceptId,
    pub evidence: Vec<EpisodeId>,
    pub evidence_count: u32,
    pub confidence: f32,
    pub last_referenced: DateTime<Utc>,
}

impl From<SemanticAtom> for SemanticMatch {
    fn from(a: SemanticAtom) -> Self {
        Self {
            atom_id: a.id,
            statement: a.statement,
            concept: a.concept,
            evidence: a.evidence,
            evidence_count: a.evidence_count,
            confidence: a.confidence,
            last_referenced: a.last_referenced,
        }
    }
}

// ---------------------------------------------------------------------------
// Layer-2-facing types (consumed by agidb-extract; defined here so agidb-core
// has no dependency on agidb-extract). Phase 3.
// ---------------------------------------------------------------------------

/// Caller-supplied context for an extraction call. The `observation_time`
/// anchors any relative time expressions ("yesterday", "last weekend")
/// the extractor parses out of the input.
#[derive(Clone, Debug)]
pub struct ExtractContext {
    pub observation_time: DateTime<Utc>,
    /// Optional whitelist of relation types the extractor should focus
    /// on. Empty = use the extractor's built-in default vocabulary.
    pub relation_hint_types: Vec<String>,
}

/// An entity identified by layer-2 NER. The `canonical_name` is
/// populated by the alias resolver after extraction; until then it is
/// `None`.
#[derive(Clone, Debug)]
pub struct Entity {
    pub text: String,
    pub entity_type: String,
    /// `(start, end)` byte offsets into the original input string.
    pub span: (usize, usize),
    pub confidence: f32,
    pub canonical_name: Option<String>,
}

/// A triple the extractor has identified but NOT yet bound to an
/// [`EpisodeId`]. The `subject` / `object` strings are post-alias-
/// resolution canonical names; the `predicate` is post-canonicalization.
/// Promoted to a [`Triple`] by `agidb_extract::observe_text` once an
/// episode id is minted.
#[derive(Clone, Debug)]
pub struct ExtractedTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f32,
}

/// The output of a single [`TextExtractor::extract`] call.
#[derive(Clone, Debug)]
pub struct Extraction {
    pub triples: Vec<ExtractedTriple>,
    pub valid_time: Option<TimeRange>,
    pub raw_entities: Vec<Entity>,
}

/// Layer-2 → layer-3 boundary. Any extractor that implements this can
/// be passed to `agidb_extract::observe_text`. Defined here in
/// `agidb-core` so the engine stays extraction-blind.
pub trait TextExtractor {
    fn extract(&self, text: &str, ctx: &ExtractContext) -> crate::Result<Extraction>;
}

// ---------------------------------------------------------------------------
// Tests — round-trip the schema invariants
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_range_contains_point_inside() {
        let start = "2026-05-01T00:00:00Z".parse::<DateTime<Utc>>().unwrap();
        let end = "2026-06-01T00:00:00Z".parse::<DateTime<Utc>>().unwrap();
        let range = TimeRange {
            start,
            end: Some(end),
        };
        let probe = "2026-05-14T00:00:00Z".parse::<DateTime<Utc>>().unwrap();
        assert!(range.contains(probe));
    }

    #[test]
    fn time_range_unbounded_end_contains_future() {
        let start = "2026-05-01T00:00:00Z".parse::<DateTime<Utc>>().unwrap();
        let range = TimeRange { start, end: None };
        let probe = "2099-01-01T00:00:00Z".parse::<DateTime<Utc>>().unwrap();
        assert!(range.contains(probe));
    }

    #[test]
    fn id_newtype_display_round_trips() {
        let e = EpisodeId::new(42);
        assert_eq!(format!("{e}"), "EpisodeId(42)");
        assert_eq!(e.raw(), 42);
    }
}
