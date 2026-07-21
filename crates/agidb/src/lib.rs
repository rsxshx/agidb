//! agidb — the cognitive substrate for autonomous AI agents.
//!
//! This is the umbrella crate. It exposes a single high-level [`Agidb`]
//! facade that wires layer 2 ([`agidb_extract`]) onto layer 1 + 3
//! ([`agidb_core`]) so the whole pipeline — text in, structured triples
//! out, signed, indexed, bi-temporally stamped, and recallable by cue —
//! is one object with the API the README promises:
//!
//! ```no_run
//! # async fn demo() -> anyhow::Result<()> {
//! let db = agidb::Agidb::open("./memory.agidb").await?;
//!
//! db.observe("Sarah recommended Bawri in Bandra last weekend").await?;
//!
//! let recall = db.recall_cue("what thai place did sarah mention?").await?;
//! for m in &recall.matches {
//!     println!("[{:.2}] {}", m.confidence, m.text);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! No SQL, no embedding API calls, no separate vector db. One function.
//!
//! The lower-level surfaces are re-exported as [`core`] and [`extract`] for
//! callers that need direct access to the engine or the extractor.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

// Internal-only imports (not re-exported).
use agidb_core::store::Store;
use agidb_core::Result as CoreResult;
use agidb_extract::{observe_text, Extractor};

// ---------------------------------------------------------------------------
// Re-exports — the stable public surface (also in scope internally).
// ---------------------------------------------------------------------------

pub use agidb_core as core;
pub use agidb_extract as extract;

pub use agidb_core::belief::{self, revise_confidence, WITHDRAWAL_THRESHOLD};
pub use agidb_core::consolidate::ConsolidationReport;
pub use agidb_core::goal::{self, GoalStateKind};
pub use agidb_core::hdc;
pub use agidb_core::learning_log::{self, LearningEvent};
pub use agidb_core::self_model::{
    self, hv_ema_update, hv_subtract, SelfVectorSnapshot, SELF_VECTOR_ALPHA,
};
pub use agidb_core::store::{Stats, StoreConfig};
pub use agidb_core::types::{
    Belief, BeliefId, BeliefRevision, Concept, ConceptId, Entity, Episode, EpisodeId,
    ExtractContext, ExtractedTriple, Extraction, Goal, GoalId, GoalPatch, GoalState, Provenance,
    Query, Recall, RecallMatch, RevisionReport, SemanticAtom, SemanticAtomId, SemanticMatch,
    SuccessCriterion, TextExtractor, Tier, TimeRange, Triple,
};
pub use agidb_core::unlearn::{self, Tombstone, UnlearnReport, UnlearnTarget};
pub use agidb_extract::{ExtractorConfig, ObserveContext};

// ---------------------------------------------------------------------------
// Extractor selection.
// ---------------------------------------------------------------------------

/// The layer-2 extractor the facade uses to turn raw text into triples.
///
/// `Real` wraps a loaded GLiNER + heuristics [`Extractor`]; `Null` stores
/// text-only episodes (signatures fall back to the gist path) so the
/// server still works when model artifacts are missing.
enum FacadeExtractor {
    Real(Arc<dyn TextExtractor + Send + Sync>),
    Null,
}

impl TextExtractor for FacadeExtractor {
    fn extract(&self, text: &str, ctx: &ExtractContext) -> CoreResult<Extraction> {
        match self {
            Self::Real(e) => e.extract(text, ctx),
            Self::Null => Ok(Extraction {
                triples: Vec::new(),
                valid_time: None,
                raw_entities: Vec::new(),
            }),
        }
    }
}

/// Configuration for [`Agidb::open_with`].
#[derive(Clone)]
pub struct AgidbConfig {
    /// Root directory of the store (created if missing).
    pub root: PathBuf,
    /// Layer-2 extractor setup.
    pub extractor: ExtractorSetup,
}

/// Which extractor to load.
#[derive(Clone)]
#[allow(clippy::large_enum_variant)]
pub enum ExtractorSetup {
    /// Try to load the real GLiNER-backed [`Extractor`]; fall back to
    /// text-only if model artifacts are unavailable (the default).
    Auto,
    /// Skip the model load entirely — observe stores text-only episodes.
    /// Fastest startup; use for demos / tests that don't need NER.
    Null,
    /// Load the real extractor with an explicit config.
    With(ExtractorConfig),
    /// Use a caller-supplied extractor (e.g. a deterministic mock for
    /// tests, or a domain-specific extractor). Lets users plug in any
    /// `TextExtractor` without depending on GLiNER.
    Custom(Arc<dyn TextExtractor + Send + Sync>),
}

impl AgidbConfig {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            extractor: ExtractorSetup::Auto,
        }
    }

    pub fn with_extractor(mut self, setup: ExtractorSetup) -> Self {
        self.extractor = setup;
        self
    }
}

// ---------------------------------------------------------------------------
// The facade.
// ---------------------------------------------------------------------------

/// The single object users hold. Owns the store (behind a mutex) and the
/// extractor. All methods are async; the blocking redb work runs on a
/// `spawn_blocking` thread so the facade is safe to share across an
/// async runtime.
pub struct Agidb {
    store: Arc<Mutex<Store>>,
    extractor: Arc<FacadeExtractor>,
    root: PathBuf,
}

impl Agidb {
    /// Open or create a store at `root` and auto-load the extractor
    /// (falls back to text-only if the model cache is cold).
    pub async fn open(root: impl Into<PathBuf>) -> CoreResult<Self> {
        Self::open_with(AgidbConfig::new(root)).await
    }

    /// Open with an explicit [`AgidbConfig`].
    pub async fn open_with(cfg: AgidbConfig) -> CoreResult<Self> {
        let root = cfg.root.clone();
        let store = tokio::task::spawn_blocking(move || Store::open(StoreConfig::at(root)))
            .await
            .map_err(|e| agidb_core::AgidbError::Internal(format!("store open join: {e}")))??;

        let extractor: FacadeExtractor = match cfg.extractor {
            ExtractorSetup::Null => FacadeExtractor::Null,
            ExtractorSetup::Custom(e) => FacadeExtractor::Real(e),
            ExtractorSetup::With(ecfg) => match Extractor::new(ecfg) {
                Ok(e) => FacadeExtractor::Real(Arc::new(e)),
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "no Extractor loaded; falling back to text-only observe"
                    );
                    FacadeExtractor::Null
                }
            },
            ExtractorSetup::Auto => match Extractor::new(ExtractorConfig::default()) {
                Ok(e) => {
                    tracing::info!("loaded real Extractor (GLiNER + heuristics)");
                    FacadeExtractor::Real(Arc::new(e))
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "model cache cold; falling back to text-only observe. \
                         Set AGIDB_OFFLINE=0 and rerun with network access to download GLiNER, \
                         or pass ExtractorSetup::Null to silence this."
                    );
                    FacadeExtractor::Null
                }
            },
        };

        Ok(Self {
            store: Arc::new(Mutex::new(store)),
            extractor: Arc::new(extractor),
            root: cfg.root,
        })
    }

    /// `true` iff observe is running real layer-2 extraction (vs.
    /// text-only). Useful for demo banners.
    pub fn extractor_loaded(&self) -> bool {
        matches!(self.extractor.as_ref(), FacadeExtractor::Real(_))
    }

    /// Path the store was opened at.
    pub fn root(&self) -> &std::path::Path {
        &self.root
    }

    // -- write -------------------------------------------------------------

    /// Record an observation attributed to `source = "user"`.
    pub async fn observe(&self, text: &str) -> CoreResult<EpisodeId> {
        self.observe_with(text, "user").await
    }

    /// Record an observation with an explicit provenance source label
    /// (e.g. `"agent"`, `"tool:gmail"`).
    pub async fn observe_with(&self, text: &str, source: &str) -> CoreResult<EpisodeId> {
        let store = self.store.clone();
        let extractor = self.extractor.clone();
        let text = text.to_string();
        let source = source.to_string();
        tokio::task::spawn_blocking(move || {
            let mut store = store.lock().expect("store mutex poisoned");
            let ctx = ObserveContext {
                observation_time: chrono::Utc::now(),
                provenance: Provenance {
                    source,
                    ..Provenance::default()
                },
            };
            observe_text(&mut store, extractor.as_ref(), &text, ctx)
        })
        .await
        .map_err(|e| agidb_core::AgidbError::Internal(format!("observe join: {e}")))?
    }

    /// Record an observation with an explicit [`ObserveContext`] (observation_time +
    /// provenance). Use this when the activity happened at a *known* time other than now —
    /// e.g. replaying captured screen/audio frames where `valid_time` must be the capture
    /// timestamp, not the ingest timestamp, so bi-temporal / time-window recall works.
    pub async fn observe_with_context(
        &self,
        text: &str,
        ctx: ObserveContext,
    ) -> CoreResult<EpisodeId> {
        let store = self.store.clone();
        let extractor = self.extractor.clone();
        let text = text.to_string();
        tokio::task::spawn_blocking(move || {
            let mut store = store.lock().expect("store mutex poisoned");
            observe_text(&mut store, extractor.as_ref(), &text, ctx)
        })
        .await
        .map_err(|e| agidb_core::AgidbError::Internal(format!("observe join: {e}")))?
    }

    // -- read --------------------------------------------------------------

    /// Run a tiered recall. Per the constitution, never returns the empty
    /// set under the default `tier_floor`.
    pub async fn recall(&self, query: Query) -> CoreResult<Recall> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || {
            let store = store.lock().expect("store mutex poisoned");
            store.recall(&query)
        })
        .await
        .map_err(|e| agidb_core::AgidbError::Internal(format!("recall join: {e}")))?
    }

    /// Convenience: recall with just a cue string and default options.
    pub async fn recall_cue(&self, cue: impl Into<String>) -> CoreResult<Recall> {
        self.recall(Query::cue(cue)).await
    }

    /// Fetch a single episode by id.
    pub async fn get_episode(&self, id: u64) -> CoreResult<Option<Episode>> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || {
            let store = store.lock().expect("store mutex poisoned");
            store.get_episode(EpisodeId::new(id))
        })
        .await
        .map_err(|e| agidb_core::AgidbError::Internal(format!("get_episode join: {e}")))?
    }

    /// List up to `limit` episodes in ascending id order.
    pub async fn list_episodes(&self, limit: usize) -> CoreResult<Vec<Episode>> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || {
            let store = store.lock().expect("store mutex poisoned");
            store.list_episodes(limit)
        })
        .await
        .map_err(|e| agidb_core::AgidbError::Internal(format!("list_episodes join: {e}")))?
    }

    /// v0.2 — temporal retrieval: return episodes whose `valid_time`
    /// overlaps the half-open window `[from, to)`, optionally
    /// restricted to a subject concept. Backed by
    /// [`Store::list_episodes_in_range`] — the direct layer-3 path
    /// homn's `timeline(subject, from, to)` MCP tool calls.
    ///
    /// When `subject` is `Some(name)`, the name is resolved to a
    /// [`ConceptId`] via the concept-by-name index and the result is
    /// filtered to episodes linked to that concept. An unknown name
    /// returns an empty list (no error) so a stale MCP caller doesn't
    /// crash the assistant. Episodes are returned in chronological
    /// order (by `valid_time.start` ascending, then id ascending as
    /// the deterministic tiebreaker).
    pub async fn timeline(
        &self,
        subject: Option<&str>,
        from: chrono::DateTime<chrono::Utc>,
        to: chrono::DateTime<chrono::Utc>,
        limit: usize,
    ) -> CoreResult<Vec<Episode>> {
        let store = self.store.clone();
        let subject = subject.map(|s| s.to_string());
        tokio::task::spawn_blocking(move || {
            let store = store.lock().expect("store mutex poisoned");
            let mut episodes = store.list_episodes_in_range(from, to, limit)?;
            if let Some(name) = subject {
                let Some(cid) = store.concept_id_for(&name)? else {
                    return Ok(Vec::new());
                };
                let linked = store.recall_exact(cid, None)?;
                let linked_ids: std::collections::HashSet<EpisodeId> =
                    linked.into_iter().map(|e| e.id).collect();
                episodes.retain(|e| linked_ids.contains(&e.id));
            }
            Ok(episodes)
        })
        .await
        .map_err(|e| agidb_core::AgidbError::Internal(format!("timeline join: {e}")))?
    }

    // -- consolidation -----------------------------------------------------

    /// Run one consolidation pass: cluster recent episodes, mint
    /// semantic atoms, detect contradictions.
    pub async fn consolidate(&self) -> CoreResult<ConsolidationReport> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || {
            let mut store = store.lock().expect("store mutex poisoned");
            store.consolidate()
        })
        .await
        .map_err(|e| agidb_core::AgidbError::Internal(format!("consolidate join: {e}")))?
    }

    // -- admin -------------------------------------------------------------

    /// Row counts + signature file size.
    pub async fn stats(&self) -> CoreResult<Stats> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || {
            let store = store.lock().expect("store mutex poisoned");
            store.stats()
        })
        .await
        .map_err(|e| agidb_core::AgidbError::Internal(format!("stats join: {e}")))?
    }

    /// Dump every episode (with its HV) as JSON lines into `path`.
    pub async fn export_jsonl(&self, path: impl Into<PathBuf>) -> CoreResult<()> {
        let store = self.store.clone();
        let path = path.into();
        tokio::task::spawn_blocking(move || {
            let store = store.lock().expect("store mutex poisoned");
            let file = std::fs::File::create(&path)?;
            let mut writer = std::io::BufWriter::new(file);
            store.export_jsonl(&mut writer)
        })
        .await
        .map_err(|e| agidb_core::AgidbError::Internal(format!("export join: {e}")))?
    }

    /// Import JSON lines produced by [`Self::export_jsonl`]. Returns the
    /// number of episodes imported.
    pub async fn import_jsonl(&self, path: impl Into<PathBuf>) -> CoreResult<u32> {
        let store = self.store.clone();
        let path = path.into();
        tokio::task::spawn_blocking(move || {
            let mut store = store.lock().expect("store mutex poisoned");
            let file = std::fs::File::open(&path)?;
            let reader = std::io::BufReader::new(file);
            store.import_jsonl(reader)
        })
        .await
        .map_err(|e| agidb_core::AgidbError::Internal(format!("import join: {e}")))?
    }

    /// Flush the signature file to disk.
    pub async fn flush(&self) -> CoreResult<()> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || {
            let store = store.lock().expect("store mutex poisoned");
            store.flush()
        })
        .await
        .map_err(|e| agidb_core::AgidbError::Internal(format!("flush join: {e}")))?
    }

    // -- goals (floor 6, phase 9) -----------------------------------------

    /// Set a new goal. Returns the minted [`GoalId`].
    pub async fn set_goal(&self, goal: Goal) -> CoreResult<GoalId> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || {
            let mut store = store.lock().expect("store mutex poisoned");
            store.set_goal(goal)
        })
        .await
        .map_err(|e| agidb_core::AgidbError::Internal(format!("set_goal join: {e}")))?
    }

    /// Apply a partial update to a goal.
    pub async fn revise_goal(&self, id: GoalId, patch: GoalPatch) -> CoreResult<()> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || {
            let mut store = store.lock().expect("store mutex poisoned");
            store.revise_goal(id, patch)
        })
        .await
        .map_err(|e| agidb_core::AgidbError::Internal(format!("revise_goal join: {e}")))?
    }

    /// Mark a goal completed with evidence episodes. Terminal.
    pub async fn complete_goal(&self, id: GoalId, evidence: Vec<EpisodeId>) -> CoreResult<()> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || {
            let mut store = store.lock().expect("store mutex poisoned");
            store.complete_goal(id, evidence)
        })
        .await
        .map_err(|e| agidb_core::AgidbError::Internal(format!("complete_goal join: {e}")))?
    }

    /// Abandon a goal with a reason. Terminal.
    pub async fn abandon_goal(
        &self,
        id: GoalId,
        reason: impl Into<String> + Send + 'static,
    ) -> CoreResult<()> {
        let store = self.store.clone();
        let reason = reason.into();
        tokio::task::spawn_blocking(move || {
            let mut store = store.lock().expect("store mutex poisoned");
            store.abandon_goal(id, reason)
        })
        .await
        .map_err(|e| agidb_core::AgidbError::Internal(format!("abandon_goal join: {e}")))?
    }

    /// Pause an active goal.
    pub async fn pause_goal(
        &self,
        id: GoalId,
        reason: impl Into<String> + Send + 'static,
    ) -> CoreResult<()> {
        let store = self.store.clone();
        let reason = reason.into();
        tokio::task::spawn_blocking(move || {
            let mut store = store.lock().expect("store mutex poisoned");
            store.pause_goal(id, reason)
        })
        .await
        .map_err(|e| agidb_core::AgidbError::Internal(format!("pause_goal join: {e}")))?
    }

    /// Resume a paused goal.
    pub async fn resume_goal(&self, id: GoalId) -> CoreResult<()> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || {
            let mut store = store.lock().expect("store mutex poisoned");
            store.resume_goal(id)
        })
        .await
        .map_err(|e| agidb_core::AgidbError::Internal(format!("resume_goal join: {e}")))?
    }

    /// Fetch a single goal by id.
    pub async fn get_goal(&self, id: GoalId) -> CoreResult<Option<Goal>> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || {
            let store = store.lock().expect("store mutex poisoned");
            store.get_goal(id)
        })
        .await
        .map_err(|e| agidb_core::AgidbError::Internal(format!("get_goal join: {e}")))?
    }

    /// Every goal in `Active` state — the goals that bias recall.
    pub async fn active_goals(&self) -> CoreResult<Vec<Goal>> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || {
            let store = store.lock().expect("store mutex poisoned");
            store.active_goals()
        })
        .await
        .map_err(|e| agidb_core::AgidbError::Internal(format!("active_goals join: {e}")))?
    }

    /// Every goal (any state).
    pub async fn all_goals(&self) -> CoreResult<Vec<Goal>> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || {
            let store = store.lock().expect("store mutex poisoned");
            store.all_goals()
        })
        .await
        .map_err(|e| agidb_core::AgidbError::Internal(format!("all_goals join: {e}")))?
    }

    // -- beliefs (floor 6, phase 9) ---------------------------------------

    /// Assert a new belief. Re-asserting the same `(subject, predicate)`
    /// revises the existing belief instead of duplicating.
    pub async fn assert_belief(&self, belief: Belief) -> CoreResult<BeliefId> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || {
            let mut store = store.lock().expect("store mutex poisoned");
            store.assert_belief(belief)
        })
        .await
        .map_err(|e| agidb_core::AgidbError::Internal(format!("assert_belief join: {e}")))?
    }

    /// Revise a belief with new evidence. `supports = true` raises
    /// confidence; `false` lowers it. Returns a [`RevisionReport`].
    pub async fn revise_belief(
        &self,
        id: BeliefId,
        new_evidence: EpisodeId,
        supports: bool,
        reason: impl Into<String> + Send + 'static,
    ) -> CoreResult<RevisionReport> {
        let store = self.store.clone();
        let reason = reason.into();
        tokio::task::spawn_blocking(move || {
            let mut store = store.lock().expect("store mutex poisoned");
            store.revise_belief(id, new_evidence, supports, reason)
        })
        .await
        .map_err(|e| agidb_core::AgidbError::Internal(format!("revise_belief join: {e}")))?
    }

    /// Withdraw a belief explicitly (non-destructive). Idempotent.
    pub async fn withdraw_belief(
        &self,
        id: BeliefId,
        reason: impl Into<String> + Send + 'static,
    ) -> CoreResult<()> {
        let store = self.store.clone();
        let reason = reason.into();
        tokio::task::spawn_blocking(move || {
            let mut store = store.lock().expect("store mutex poisoned");
            store.withdraw_belief(id, reason)
        })
        .await
        .map_err(|e| agidb_core::AgidbError::Internal(format!("withdraw_belief join: {e}")))?
    }

    /// Fetch a single belief by id.
    pub async fn get_belief(&self, id: BeliefId) -> CoreResult<Option<Belief>> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || {
            let store = store.lock().expect("store mutex poisoned");
            store.get_belief(id)
        })
        .await
        .map_err(|e| agidb_core::AgidbError::Internal(format!("get_belief join: {e}")))?
    }

    /// Every non-withdrawn belief about `subject`.
    pub async fn what_do_i_believe(&self, subject: &str) -> CoreResult<Vec<Belief>> {
        let store = self.store.clone();
        let subject = subject.to_string();
        tokio::task::spawn_blocking(move || {
            let store = store.lock().expect("store mutex poisoned");
            store.what_do_i_believe(&subject)
        })
        .await
        .map_err(|e| agidb_core::AgidbError::Internal(format!("what_do_i_believe join: {e}")))?
    }

    /// The append-only revision log for a belief.
    pub async fn belief_history(&self, id: BeliefId) -> CoreResult<Vec<BeliefRevision>> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || {
            let store = store.lock().expect("store mutex poisoned");
            store.belief_history(id)
        })
        .await
        .map_err(|e| agidb_core::AgidbError::Internal(format!("belief_history join: {e}")))?
    }

    /// Every belief (any state, including withdrawn).
    pub async fn all_beliefs(&self) -> CoreResult<Vec<Belief>> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || {
            let store = store.lock().expect("store mutex poisoned");
            store.all_beliefs()
        })
        .await
        .map_err(|e| agidb_core::AgidbError::Internal(format!("all_beliefs join: {e}")))?
    }

    // -- unlearn (phase 11) ----------------------------------------------

    /// Non-destructive cascading unlearn. Constitution article XVI.
    pub async fn unlearn(
        &self,
        target: UnlearnTarget,
        reason: impl Into<String> + Send + 'static,
    ) -> CoreResult<UnlearnReport> {
        let store = self.store.clone();
        let reason = reason.into();
        tokio::task::spawn_blocking(move || {
            let mut store = store.lock().expect("store mutex poisoned");
            store.unlearn(target, reason)
        })
        .await
        .map_err(|e| agidb_core::AgidbError::Internal(format!("unlearn join: {e}")))?
    }

    /// Restore tombstoned rows within the 30-day recovery window.
    pub async fn restore_within_window(&self, audit_event_id: u64) -> CoreResult<usize> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || {
            let mut store = store.lock().expect("store mutex poisoned");
            store.restore_within_window(audit_event_id)
        })
        .await
        .map_err(|e| agidb_core::AgidbError::Internal(format!("restore join: {e}")))?
    }

    /// Every tombstone record (unlearn history).
    pub async fn unlearn_history(&self) -> CoreResult<Vec<Tombstone>> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || {
            let store = store.lock().expect("store mutex poisoned");
            store.all_tombstones()
        })
        .await
        .map_err(|e| agidb_core::AgidbError::Internal(format!("unlearn_history join: {e}")))?
    }

    // -- self-model introspection (phase 10) ------------------------------

    /// Every learning event since `since` (the self-model audit log).
    pub async fn what_did_i_learn(
        &self,
        since: chrono::DateTime<chrono::Utc>,
    ) -> CoreResult<Vec<LearningEvent>> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || {
            let store = store.lock().expect("store mutex poisoned");
            store.what_did_i_learn(since)
        })
        .await
        .map_err(|e| agidb_core::AgidbError::Internal(format!("what_did_i_learn join: {e}")))?
    }

    /// Every learning event (no time filter).
    pub async fn all_learning_events(&self) -> CoreResult<Vec<LearningEvent>> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || {
            let store = store.lock().expect("store mutex poisoned");
            store.all_learning_events()
        })
        .await
        .map_err(|e| agidb_core::AgidbError::Internal(format!("all_learning_events join: {e}")))?
    }

    /// The current self-vector (8192-bit HV).
    pub async fn self_vector(&self) -> CoreResult<agidb_core::hdc::HV> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || {
            let store = store.lock().expect("store mutex poisoned");
            store.self_vector()
        })
        .await
        .map_err(|e| agidb_core::AgidbError::Internal(format!("self_vector join: {e}")))?
    }

    /// Self-vector snapshot history.
    pub async fn self_vector_history(&self) -> CoreResult<Vec<SelfVectorSnapshot>> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || {
            let store = store.lock().expect("store mutex poisoned");
            store.self_vector_history()
        })
        .await
        .map_err(|e| agidb_core::AgidbError::Internal(format!("self_vector_history join: {e}")))?
    }

    /// Record a raw sensory frame; promotes to episodic memory when
    /// surprising. Floor 1.
    pub async fn observe_sensory(
        &self,
        text: impl Into<String> + Send + 'static,
    ) -> CoreResult<agidb_core::sensory::SensoryObservation> {
        let store = self.store.clone();
        let text = text.into();
        tokio::task::spawn_blocking(move || {
            let mut store = store.lock().expect("store mutex poisoned");
            store.observe_sensory(&text)
        })
        .await
        .map_err(|e| agidb_core::AgidbError::Internal(format!("observe_sensory join: {e}")))?
    }

    /// Surprise score of `text` against recent memory, in [0, 1].
    pub async fn surprise_score(
        &self,
        text: impl Into<String> + Send + 'static,
    ) -> CoreResult<f32> {
        let store = self.store.clone();
        let text = text.into();
        tokio::task::spawn_blocking(move || {
            let store = store.lock().expect("store mutex poisoned");
            store.surprise_score(&text)
        })
        .await
        .map_err(|e| agidb_core::AgidbError::Internal(format!("surprise_score join: {e}")))?
    }

    /// The most recent sensory frames, newest first.
    pub async fn sensory_frames(
        &self,
        limit: usize,
    ) -> CoreResult<Vec<agidb_core::sensory::SensoryFrame>> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || {
            let store = store.lock().expect("store mutex poisoned");
            store.sensory_frames(limit)
        })
        .await
        .map_err(|e| agidb_core::AgidbError::Internal(format!("sensory_frames join: {e}")))?
    }
}
