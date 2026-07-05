# agidb — Architecture

> The architecture in one document. Three engineering layers, seven cognitive
> floors, the write path (v2.0 + v2.1 multimodal), the read path, the
> consolidation loop, the unlearn loop with self-vector subtraction, and the
> key design choices.

## The shape of agidb

agidb is built in **three engineering layers** that implement **seven cognitive floors**. The engineering layers tell you *how* it's built. The cognitive floors tell you *what* it stores. Both are required to understand the system fully.

```
┌────────────────────────────────────────────────────────────────────┐
│                   THE SEVEN COGNITIVE FLOORS                       │
│                  (what the user experiences)                       │
├────────────────────────────────────────────────────────────────────┤
│  7. Self-model         introspection, learning log, attention,     │
│                        self-vector EMA                             │
│  6. Goals + Beliefs    typed state machines, revisable claims      │
│  5. Procedural         skills with execution traces                │
│  4. Semantic           consolidated facts from episodes            │
│  3. Episodic           events with bi-temporal stamps              │
│  2. Working            active context, session-scoped              │
│  1. Sensory            raw signal buffer, surprise-gated           │
│                        v2.1: multimodal (video + audio + text)     │
└────────────────────────────────────────────────────────────────────┘
                              ▲
┌────────────────────────────────────────────────────────────────────┐
│                  THE THREE ENGINEERING LAYERS                      │
│                       (how it's built)                             │
├────────────────────────────────────────────────────────────────────┤
│  LAYER 1 — RECALL                                                  │
│  The mind-like layer. HDC signatures, binding, bundling,           │
│  hamming-distance retrieval, tiered confidence, goal-biased        │
│  weighting. This is what the user experiences directly.            │
├────────────────────────────────────────────────────────────────────┤
│  LAYER 2 — EXTRACTION                                              │
│  The scaffolding. v2.0: GLiNER ONNX entity/relation extraction,    │
│  belief extraction, time-anchor parsing.                           │
│  v2.1: V-JEPA 2 (video), Wav2Vec-BERT (audio), Llama-3.2-3B        │
│  (text). All project to 8192-bit HV via random projection.         │
├────────────────────────────────────────────────────────────────────┤
│  LAYER 3 — STORAGE                                                 │
│  The plumbing. redb for metadata + bi-temporal indexes. mmap'd     │
│  flat files for signatures. Append-only logs for the self-model    │
│  audit trail. Crash-safe, ACID, Rust.                              │
└────────────────────────────────────────────────────────────────────┘
```

## The three engineering layers

### Layer 3 — Storage

The plumbing. Layer 3 persists everything reliably. It exposes a typed API to the upper layers, knows nothing about cognition or extraction.

**Storage components:**
- **redb** — pure-Rust embedded ACID key-value store with MVCC and savepoints. Holds 14 tables in v2.0, 16 in v2.1.
- **mmap'd `signatures.dat`** — flat file of 1024-byte HDC signatures, indexed by offset. POPCOUNT-scanned in bulk.
- **`learning_events`** — append-only redb table for self-model audit.
- **`tombstones`** — non-destructive removal tracking with 30-day recovery window.
- **`self_vector_history`** (v2.1) — EMA snapshots of the self-vector across consolidation epochs.
- **`encoder_versions`** (v2.1) — versioned config for V-JEPA 2, Wav2Vec-BERT, Llama-3.2-3B model hashes and projection matrices.

**Why this stack:** redb is the right embedded ACID layer in Rust. mmap for signatures because they are fixed-size, never updated in place, and POPCOUNT-friendly. Append-only logs for audit because audit trails must be tamper-evident.

### Layer 2 — Extraction

The scaffolding. Layer 2 turns natural language (and in v2.1, raw video and audio) into the structured signatures that layer 1 binds into episodes.

**Extraction components (v2.0):**
- **GLiNER ONNX** — entity + relation extraction. Local, no API key, no hallucination. Output: ranked triples with confidence.
- **Time anchor parser** — "last weekend" → 2026-05-09.
- **Alias resolver** — "Sarah" and "Sarah Lee" → same `ConceptId`.
- **Predicate canonicalizer** — "recommended", "suggested" → canonical `recommends`.
- **Belief extractor** — high-confidence assertions extracted as `Belief` candidates.

**Extraction components (v2.1, multimodal):**
- **V-JEPA 2 Gigantic-256** — 64-frame video window → 1024d dense latent via Meta's frozen ViT. ONNX or Candle backend.
- **Wav2Vec-BERT 2.0** — 60s audio chunk → 1024d dense latent via Meta's frozen audio model.
- **Llama-3.2-3B** — 1024-token text window → 2048d latent via Meta's frozen small language model.
- **HDC projection** — Charikar 2002 thresholded random projection: `s = sign(R · x)` where R ∈ {-1,+1}^(8192 × D) is a fixed seeded random matrix. Maps dense latents to 8192-bit signatures with Johnson-Lindenstrauss distance preservation. Deterministic, training-free.
- **VSA multimodal binding** — role-filler XOR binding of modality signatures into one episode signature. Factorable: video/audio/text components can be recovered from the bound episode.

**Why this stack:** GLiNER is the best local extractor for v2.0. For v2.1, V-JEPA 2 + Wav2Vec-BERT + Llama-3.2-3B is the encoder stack used by Meta FAIR's TRIBE v2 (the brain-encoding foundation model that won Algonauts 2025). Matching the encoder stack means agidb's internal signatures can be benchmarked against TRIBE-predicted cortical activations on matched stimuli — see [brain-alignment.md](./brain-alignment.md).

### Layer 1 — Recall

The mind-like layer. Layer 1 takes structured input from layer 2 and stores/retrieves it as HDC signatures over layer 3's storage. This is what the user experiences directly.

**Recall components:**
- **HDC kernel** — `bind` (XOR), `bundle` (per-bit majority), `hamming` (POPCOUNT). 8192-bit signatures.
- **Episode encoder** — bind entities into role-filler patterns, bundle triples into one signature per episode. Extended in v2.1 to multimodal episodes.
- **Tiered cascade** — A (exact) → B (similarity) → C (gist) → D (nearest-neighbor). Never returns empty.
- **Goal-bias reweighter** — active goals up-weight relevant matches via HDC similarity to goal signatures.
- **Attention tracer** — every recall emits a trace of which signatures activated and why.

**Why this stack:** HDC because the math is settled, deterministic, encoder-free, and POPCOUNT-fast. Tiered cascade because real cues are noisy and recall must degrade gracefully. Goal-biasing because attention is cognitive function — agents attend to what they want.

## The seven cognitive floors

The biological mapping. These are *what* agidb stores and exposes through the API. Each floor is a typed shape with its own retrieval semantics.

### Floor 1 — Sensory memory

**What it is:** raw signal buffer, sub-second to minutes residence, surprise-gated promotion to episodic memory.

**Storage:** `SensoryFrame` ring buffer in redb. Default capacity 1000 frames or 60 seconds. Frames carry raw text (v2.0) or multimodal blob refs (v2.1).

**Surprise gating (v2.0):** `surprise = 1 - hamming_similarity(new_sig, bundle(recent_beliefs))`. Promotion threshold default 0.4.

**Surprise gating (v2.1, brain-calibrated):** threshold θ_brain empirically fit against neural surprise predicted by TRIBE v2 on associative cortex (TPJ, dlPFC, DMN) over 720-subject fMRI ground truth. Replaces magic threshold with measurement-grounded calibration. See [brain-alignment.md](./brain-alignment.md).

**API (v2.0):**
- `observe_sensory(frame: SensoryFrame) -> SensoryId`
- `working_state() -> SensoryBuffer`
- `surprise_score(frame: &SensoryFrame) -> f32`

**API (v2.1):**
- `observe_multimodal(video: VideoClip, audio: AudioClip, text: String) -> EpisodeId`
- `surprise_score_brain_calibrated(frame: &MultimodalFrame) -> f32`

### Floor 2 — Working memory

**What it is:** active context, capacity-bounded (~7 items), session-scoped, recency-weighted.

**Storage:** no separate table — implemented as a session + recency boost on top of episodic retrieval.

**API:**
- `recall(Query::with_session(session_id))` — session-scoped retrieval

### Floor 3 — Episodic memory

**What it is:** autobiographical events with full context: when, where, who, what.

**Storage:** `Episode` table in redb. Bi-temporal stamps, HDC signature, provenance, confidence, tombstone flag. v2.1: multimodal signature components recoverable via VSA unbinding.

**API:**
- `observe(text: String, ctx: ObserveContext) -> EpisodeId`
- `observe_multimodal(...) -> EpisodeId` (v2.1)
- `recall(Query) -> Recall`
- `supersede(old: EpisodeId, new: EpisodeId) -> Result<()>`
- `between(t0: Time, t1: Time) -> Vec<Episode>`

### Floor 4 — Semantic memory

**What it is:** decoupled general knowledge. Facts consolidated from episodes.

**Storage:** `SemanticAtom` table in redb. Produced by consolidation worker. Carries evidence count, source episodes, confidence, canonical (subject, predicate, object).

**API:**
- `consolidate() -> ConsolidationReport`
- `what_about(concept: ConceptId) -> Vec<SemanticMatch>`

### Floor 5 — Procedural memory

**What it is:** skills and workflows with execution traces.

**Storage:** `Procedure` table + `ExecutionTrace` log.

**API:**
- `observe_procedure(p: Procedure) -> ProcedureId`
- `record_execution(p: ProcedureId, trace: ExecutionTrace) -> Result<()>`
- `procedure_stats(p: ProcedureId) -> ProcedureStats`
- `recall_procedure(query: Query) -> Vec<Procedure>`

### Floor 6 — Goals + Beliefs

**What it is:** what the agent wants and what the agent thinks is true.

**Goals storage:** `Goal` table. State machine (Active / Paused / Completed / Abandoned), parent-child hierarchy, success criteria, deadlines, HDC signature.

**Beliefs storage:** `Belief` + `belief_revisions` tables. Confidence, evidence, contradictions, append-only revision log.

**API:**
- `set_goal(g: Goal) -> GoalId` / `revise_goal(id, patch) -> Result<()>`
- `active_goals() -> Vec<Goal>` / `goal_tree(root) -> GoalTree`
- `assert_belief(b: Belief) -> BeliefId`
- `revise_belief(evidence) -> RevisionReport`
- `what_do_i_believe(about: ConceptId) -> Vec<Belief>`

### Floor 7 — Self-model

**What it is:** the agent's audit log of its own development + a slowly-drifting self-representation.

**Storage:**
- `learning_events` table — append-only log of every introspectable event.
- `self_vector_history` (v2.1) — EMA snapshots of the self-vector across consolidation epochs.

**Self-vector (v2):** 8192-bit hypervector representing "what kind of agent am I right now." Updated each consolidation epoch as `self_vector ← (1-α) self_vector + α bundle(consolidated_atoms)` with α ≈ 0.05. Inspired by V-JEPA 2's EMA target network and TRIBE v2's per-subject embedding layer.

**Critical unlearn behavior:** unlearn cascades through the self-vector. When episodes are tombstoned, `self_vector ← self_vector - α · bundle(tombstoned_signatures)`. Without this, the self-model still "remembers" unlearned concepts as centroid contamination. With it, unlearn is real.

**API:**
- `what_did_i_learn(since: DateTime<Utc>) -> Vec<LearningEvent>`
- `attention_trace(recall_id: RecallId) -> AttentionTrace`
- `self_vector() -> HV` (v2)
- `self_vector_at(time: DateTime<Utc>) -> HV` (v2.1, replay from history)

## The write path — `observe()` (v2.0, text-only)

```
USER  db.observe("Sarah recommended Bawri in Bandra last weekend")
  │
  ▼  FLOOR 1 — SENSORY BUFFER
  │  1. Record raw text in sensory ring buffer with timestamp
  │  2. Compute surprise score against current beliefs
  │     surprise = 1 - similarity(new_signature, bundled_beliefs)
  │  3. If surprise > threshold OR explicit observe() call: promote
  │
  ▼  LAYER 2 — EXTRACTION
  │  1. GLiNER ONNX (~150ms CPU) extracts entities + relations
  │  2. Resolve "last weekend" → 2026-05-09 (valid time)
  │  3. Attach confidence scores; canonicalize entity names
  │  4. Optionally extract beliefs
  │
  ▼  LAYER 1 — BINDING
  │  1. Look up / assign 8192-bit HVs per concept
  │  2. Bind triples into role-filler patterns
  │  3. Bundle triples into one episode signature
  │  4. Also compute a raw-text gist signature
  │
  ▼  LAYER 3 — STORAGE
  │  1. Append 1KB signature to signatures.dat
  │  2. Write episode row to redb
  │  3. Update inverted index + concept index
  │  4. Emit LearningEvent::EpisodeStored to floor 7 log
  │  5. fsync, return EpisodeId
  │
  ▼
USER gets EpisodeId
```

Total time: ~200ms, dominated by GLiNER inference.

## The write path — `observe_multimodal()` (v2.1, brain-aligned)

```
USER  db.observe_multimodal(video_30s, audio_30s, "sarah said bawri")
  │
  ▼  FLOOR 1 — SENSORY BUFFER (multimodal)
  │  1. Record raw modalities in sensory ring buffer
  │
  ▼  LAYER 2 — MULTIMODAL EXTRACTION
  │  1. V-JEPA 2 inference on 64-frame video → 1024d latent (~1.5s CPU)
  │  2. Wav2Vec-BERT inference on 60s audio → 1024d latent (~400ms CPU)
  │  3. Llama-3.2-3B inference on text → 2048d latent (~200ms CPU)
  │  4. Charikar 2002 random projection: each latent → 8192-bit HV
  │
  ▼  FLOOR 1 — BRAIN-CALIBRATED SURPRISE
  │  1. Compute predicted next signature from sliding window
  │  2. surprise = 1 - hamming_similarity(new_sig, predicted)
  │  3. If surprise > θ_brain (calibrated against TRIBE v2): promote
  │
  ▼  LAYER 1 — MULTIMODAL BINDING (VSA)
  │  episode = ROLE_video ⊕ sig_video
  │         XOR ROLE_audio ⊕ sig_audio
  │         XOR ROLE_text  ⊕ sig_text
  │         XOR ROLE_goal  ⊕ sig_goal
  │         XOR ROLE_belief ⊕ sig_belief
  │         XOR ROLE_time  ⊕ sig_time
  │  (Factorable: any modality can be recovered via XOR with its role HV)
  │
  ▼  LAYER 3 — STORAGE
  │  Same as v2.0 + multimodal metadata
  │  Emit LearningEvent::MultimodalEpisodeStored
  │
  ▼
USER gets EpisodeId
```

Total time: ~2s CPU on a laptop, ~500ms on M2 ANE / RTX 4090. Dominated by V-JEPA 2 inference.

## The read path — `recall()` with goal-bias

```
USER  db.recall("what thai place did sarah mention?")
  │
  ▼  LAYER 2 — PARTIAL EXTRACTION (if natural-language cue)
  │  Extract partial triple shape: (Sarah, recommends, ?thai_place)
  │
  ▼  FLOOR 6 — ACTIVE GOAL LOOKUP
  │  Active goals: [find a thai place for the team dinner]
  │  Compute goal signature bundle: GoalSig
  │
  ▼  LAYER 1 — TIERED RETRIEVAL
  │
  ▼  TIER A — EXACT       canonical entity match via concept index
  │                       confidence 1.0
  │
  ▼  TIER B — SIMILARITY  HDC structured-signature similarity,
  │                       POPCOUNT over inverted-index intersection
  │                       confidence band [0.6, 0.95]
  │
  ▼  TIER C — GIST        raw-text gist signature similarity
  │                       confidence band [0.3, 0.6]
  │
  ▼  TIER D — NEAREST     best-effort nearest neighbors,
  │                       low_confidence flag, confidence ≤ 0.3
  │
  ▼  GOAL-BIAS REWEIGHTING
  │  For each match m:
  │    bias = similarity(m.signature, GoalSig) * GOAL_BIAS_WEIGHT
  │    m.confidence *= (1 + bias)
  │  Re-rank by adjusted confidence
  │
  ▼  LAYER 3 — HYDRATION
  │  Fetch rows, apply bi-temporal `as_of` filter,
  │  apply supersession filter, attach provenance + beliefs
  │
  ▼  FLOOR 7 — ATTENTION TRACE
  │  Emit LearningEvent::AttentionTraced
  │
  ▼
USER gets Recall {
  matches: [Bawri (0.94, goal-biased)],
  semantic_atoms: [Sarah likes thai food (0.82)],
  beliefs: [Bawri is a thai restaurant (0.95)],
  active_goals: [find_thai_place],
  tier_used: B,
  elapsed_ms: 32,
  attention_trace: Some(trace_id)
}
```

Target: **p95 under 50ms** on a laptop with 100k episodes. Zero network calls. No LLM in the read path. No V-JEPA inference in the read path — only stored signatures are scanned.

## The consolidation loop — `consolidate()`

`consolidate()` is an **explicit pass** invoked by the caller (CLI
`agidb consolidate <db>`, MCP `memory_consolidate`, or API). The
self-scheduling background worker that calls it on a cadence is roadmap,
not in the engine yet.

```
1. SCAN         Scan recent episodic signatures (last 7 days).

2. SURPRISE     For each episode, compute surprise score against
                current beliefs and semantic atoms.

3. CLUSTER      Cluster by hamming distance (similarity ≥ 0.95).
                Clusters of N ≥ 3 are consolidation candidates.

4. ATOMS        Bundle each cluster into a SemanticAtom with
                evidence_count = N and links back to source episodes.

5. BELIEFS      Promote high-confidence semantic atoms to beliefs:
                  evidence_count ≥ 5, no contradictions, confidence ≥ 0.8
                Add to Belief table with revision_log entry.

6. CONTRADICT   Same (subject, predicate), overlapping valid time,
                different object → older fact superseded.
                Affected beliefs get BeliefRevision entries.

7. SELF-VECTOR (v2)
                self_vector ← (1-α) self_vector + α bundle(consolidated_atoms)
                Append snapshot to self_vector_history table.

8. DECAY        Unreferenced atoms (last_referenced > 90 days) decay
                by factor λ; archived below the floor.

9. COMPACT      Rewrite signatures.dat to drop tombstoned and archived entries.

10. AUDIT       Emit LearningEvents for every action.
```

Phase 6 implements steps 1, 3, 4, 6, 10. Phase 9 adds steps 2 and 5. Phase 10 adds step 7. Steps 8 and 9 are deferred follow-ups.

## The unlearn loop — `unlearn()` with self-vector subtraction

```
USER  db.unlearn(UnlearnTarget::Concept("Sarah"), "user requested forget")
  │
  ▼  IDENTIFY CASCADE
  │  1. Find all episodes referencing Sarah's ConceptId
  │  2. Find all beliefs with Sarah as subject or in evidence
  │  3. Find all semantic atoms derived from those episodes
  │  4. Find all procedures triggered by those events
  │  5. Compute the full dependency graph
  │
  ▼  TOMBSTONE (non-destructive)
  │  1. Mark each affected row with t_tombstoned = now
  │  2. Signatures invalidated in mmap (compacted later)
  │  3. Concept HV marked withdrawn
  │  4. Inverted-index entries removed
  │
  ▼  CASCADE THROUGH BELIEFS + ATOMS + PROCEDURES
  │  1. Beliefs whose evidence drops below threshold:
  │     - confidence reduced proportionally, OR
  │     - belief withdrawn entirely (emit BeliefRevision)
  │  2. Semantic atoms recomputed without removed evidence
  │     (or atom withdrawn if evidence falls below 3)
  │  3. Procedures with broken trigger chains marked degraded
  │
  ▼  SELF-VECTOR SUBTRACTION (v2, critical)
  │  self_vector ← self_vector - α · bundle(tombstoned_signatures)
  │  Without this step the self-model still "remembers" the
  │  unlearned concept as centroid contamination.
  │  Append corrected snapshot to self_vector_history.
  │
  ▼  AUDIT (permanent)
  │  Emit LearningEvent::Unlearned with:
  │  - target_ref = "Concept:Sarah"
  │  - cascade_size = 47 episodes, 12 beliefs, 3 atoms, 1 procedure
  │  - self_vector_drift = || self_vector_before - self_vector_after ||
  │  - reason = "user requested forget"
  │  - dependency_graph_id for full audit trail
  │
  ▼
USER gets UnlearnReport {
  episodes_removed: 47,
  beliefs_removed: 8,
  beliefs_revised: 4,
  semantic_atoms_affected: 3,
  procedures_affected: 1,
  signatures_invalidated: 47,
  self_vector_drift_hamming: 184,
  audit_log_entry: LearningEventId(...),
  tombstone_expiry: 2026-06-19
}
```

## Key design choices

| Choice | Alternative | Why |
|---|---|---|
| HDC signatures as the primary representation | dense embeddings | deterministic, no model dependency, 8× smaller, POPCOUNT-fast, encoder-free |
| 8192-bit signatures | 1024 or 16384 | balance: enough capacity for 100M concepts, fits in two cache lines, fast POPCOUNT |
| Goals + beliefs as first-class types | text-stored in episodes | enables typed retrieval, state-machine semantics, revision audit |
| GLiNER for text extraction | LLM-based extraction | local, no API key, no hallucination at write time |
| **V-JEPA 2 for video (v2.1)** | dense pixel models | latent-space prediction philosophy, 1.2B params, SOTA SSv2, TRIBE-aligned |
| **Wav2Vec-BERT for audio (v2.1)** | whisper | TRIBE v2 uses this; shared encoder = free brain-alignment |
| **Llama-3.2-3B for text encoding (v2.1)** | other small LMs | TRIBE v2 uses this; smaller cost than 8B+ models |
| **Charikar 2002 random projection (v2.1)** | learned quantization | deterministic, training-free, JL distance preservation guarantee |
| **VSA role-filler binding for multimodal (v2.1)** | attention fusion | factorable: modality components recoverable from bound episode |
| LLM allowed at write time only | no LLM ever / LLM at read | belief revision needs semantic judgment; read path must stay deterministic |
| redb for metadata | sqlite, rocksdb, sled | pure Rust, ACID, MVCC, no FFI |
| mmap'd flat files for signatures | signatures in redb | fixed-size, POPCOUNT-friendly bulk scan |
| Bi-temporal supersession | overwrite on update | preserves history, auditable, mirrors legal/financial systems |
| Non-destructive unlearn with audit | hard DELETE | compliance, security, recoverability |
| **Self-vector subtraction on unlearn (v2)** | tombstones only | otherwise self-model retains centroid contamination from unlearned data |
| Tiered recall with explicit confidence | binary hit/miss | graceful degradation |
| Goal-biased retrieval | uniform retrieval | matches cognitive function |
| Attention trace per recall | no introspection | enables self-model floor 7 |
| **Brain-calibrated surprise gating (v2.1)** | hand-tuned threshold | grounded in 720-subject fMRI ground truth |
| Single binary, embedded | client-server | runs offline, no infra, sqlite-like deployment |
| Rust top to bottom | Python + Rust extensions | sub-50ms p95, no GC |
| MCP server as first-class interface | REST API | reaches agents directly |
| Self-model audit log | no introspection | enables "what did I learn?" |
| Self-vector EMA in self-model | log-only | inspired by V-JEPA 2 EMA target + TRIBE per-subject embedding |
| `LearningEvent` enum (closed set) | open event types | type-safety, exhaustive matching |
| 30-day tombstone window | immediate hard delete | recovery from accidental unlearn |

## What the architecture enables

| Capability | Enabled by |
|---|---|
| Sub-50ms recall with no API calls | HDC + redb + mmap (layer 1 + 3) |
| Bi-temporal "as of" queries | bi-temporal columns on every fact (layer 3) |
| Goal-biased retrieval | HDC goal signatures + reweight pass (layer 1 + floor 6) |
| Belief revision with audit | revision_log + LearningEvent (floor 6 + 7) |
| "What did I learn?" introspection | append-only LearningEvent log (floor 7) |
| **"What kind of agent am I?" introspection (v2)** | self-vector EMA (floor 7) |
| Right-to-be-forgotten compliance | non-destructive cascading unlearn + permanent audit |
| **Real unlearn (not just hiding) (v2)** | self-vector subtraction step |
| Surprise-gated noise filtering | HDC similarity + sensory ring buffer (floor 1) |
| **Brain-calibrated surprise (v2.1)** | TRIBE v2 calibration against 720-subject fMRI |
| Procedure improvement over time | execution traces with success rates (floor 5) |
| Provenance back to original observation | every signature has back-pointer to source episode |
| No catastrophic forgetting on update | bi-temporal supersession instead of overwrite |
| **Multimodal episode storage (v2.1)** | VSA role-filler binding |
| **Factorable multimodal retrieval (v2.1)** | XOR unbinding recovers individual modality signatures |
| **Brain-alignment evaluation (v2.1)** | shared encoder stack with TRIBE v2 + BAMS RSA benchmark |
