# agidb — Layer 1: Recall

> The mind-like layer. HDC signatures, binding, bundling, hamming-distance
> retrieval, tiered confidence, goal-biased weighting, attention tracing.
> The substrate's most-touched code path; the one the user experiences.

## What layer 1 is

Layer 1 is where memories become hypervectors and retrieval becomes bit-overlap counting. It's the layer that makes agidb feel different from a vector database. Where other systems do "embed → similarity search → rerank → answer," layer 1 does "bind into HDC → POPCOUNT scan → tiered confidence → answer."

Layer 1 sits on top of layer 3 (storage) and uses output from layer 2 (extraction). Its public-facing surface is `recall()`, the read-path function the agent calls most often.

## The HDC kernel

### Hypervectors

8192-bit binary hypervectors. 1024 bytes each. 64-byte aligned for SIMD.

```rust
#[repr(align(64))]
pub struct HV {
    pub bits: [u64; 128],   // 128 × 64 = 8192 bits
}
```

### The three core operations

**1. Bind (`⊗` or `XOR`):**
```rust
pub fn bind(&self, other: &HV) -> HV {
    HV { bits: array::from_fn(|i| self.bits[i] ^ other.bits[i]) }
}
```
Bind two HVs into a third HV that is **dissimilar to both**. Used for role-filler patterns: `bind(ROLE_SUBJ, Sarah_HV)` is "the subject is Sarah," similar to neither ROLE_SUBJ alone nor Sarah_HV alone. Self-inverse: `bind(bind(a, b), b) = a`.

**2. Bundle (majority vote):**
```rust
pub fn bundle(hvs: &[HV]) -> HV {
    let mut counts = [0i32; 8192];
    for hv in hvs {
        for bit_idx in 0..8192 {
            if hv.get_bit(bit_idx) { counts[bit_idx] += 1; } else { counts[bit_idx] -= 1; }
        }
    }
    let mut result = HV::zero();
    for bit_idx in 0..8192 {
        if counts[bit_idx] > 0 { result.set_bit(bit_idx); }
    }
    result
}
```
Bundle multiple HVs into one HV that is **similar to all of them**. Per-bit majority vote, thresholded to binary. Used to compose multiple bindings (multiple role-filler pairs in an episode signature).

**3. Hamming (POPCOUNT distance):**
```rust
pub fn hamming(&self, other: &HV) -> u32 {
    self.bits.iter().zip(&other.bits)
        .map(|(a, b)| (a ^ b).count_ones())
        .sum()
}
pub fn similarity(&self, other: &HV) -> f32 {
    1.0 - (self.hamming(other) as f32) / 8192.0
}
```
Hamming distance is bit-overlap counting. Similarity is `1 - hamming/8192` ∈ [0, 1]. POPCOUNT is a single CPU instruction on modern hardware; AVX-512 has VPOPCNTQ; ARM NEON has CNT. Portable Rust path uses `u64::count_ones()`.

### Why binary, not real-valued

| | binary BSC (agidb) | real-valued HRR |
|---|---|---|
| bind operation | XOR (1 instruction) | circular convolution (FFT) |
| similarity | hamming (POPCOUNT) | cosine (multiply-add) |
| storage | 1 KB per HV | 32 KB per HV (8192 × float32) |
| ops/second | ~10× faster | slower |
| theoretical capacity | excellent for compositional structure | excellent for analog scalars |
| factorability | clean (XOR self-inverse) | requires resonator networks |

For agidb's use case (compositional structure over discrete concepts), BSC wins on every metric except analog scalars. v2.3 adds HRR as a secondary format for analog values (temperatures, scores, probabilities); v2.1 stays pure BSC.

### Why 8192 bits

- **Capacity:** at 8192 bits with random hypervectors, the probability of two unrelated HVs having hamming distance > 4000 is essentially 1.0. So similarity > 0.51 is meaningful signal.
- **Cache friendliness:** 1024 bytes = 16 × 64-byte cache lines. POPCOUNT scan over 100k signatures is ~5ms on Zen 4 portable path, ~1.5ms with AVX-512.
- **Charikar 2002 JL bound:** for embedding distances of typical ML latents (1024-2048 dims), 8192 bits provides ε ≈ 0.05 distortion.
- **Round number:** debugging is easier with bit indices that fit in i13 + sign.

### Encoder API

```rust
pub fn from_name(name: &str) -> HV {
    let hash = blake3::hash(name.as_bytes());
    HV::from_seed(hash.as_bytes())
}
pub fn from_seed(seed: &[u8; 32]) -> HV {
    let mut rng = ChaCha20Rng::from_seed(*seed);
    let mut hv = HV::zero();
    for bit_idx in 0..8192 {
        if rng.gen_bool(0.5) { hv.set_bit(bit_idx); }
    }
    hv
}
```

`from_name("Sarah")` deterministically produces the same 8192-bit HV every time. This is how agidb assigns concept HVs without a learned codebook.

### v2.1 extension: HDC projection of dense latents

In v2.1, dense latents from V-JEPA 2 (1024d), Wav2Vec-BERT (1024d), and Llama-3.2-3B (2048d) project to 8192-bit HVs via Charikar 2002 thresholded random projection. See [brain-alignment.md](./brain-alignment.md) for the math. The projected HVs participate in layer 1 retrieval identically to text-derived signatures.

## Episode encoding (binding triples into one signature)

A stored episode has zero or more triples. Each triple binds to a partial pattern; all triples bundle into the episode signature.

```rust
pub fn bind_triple(triple: &Triple) -> HV {
    let subj = concept_hv(&triple.subject);
    let pred = predicate_hv(&triple.predicate);
    let obj  = value_hv(&triple.object);

    ROLE_SUBJ.bind(&subj)
        ^ ROLE_PRED.bind(&pred)
        ^ ROLE_OBJ.bind(&obj)
}

pub fn encode_episode_signature(triples: &[Triple]) -> HV {
    let triple_sigs: Vec<HV> = triples.iter().map(bind_triple).collect();
    if triple_sigs.is_empty() {
        return HV::zero();
    }
    bundle(&triple_sigs)
}
```

`ROLE_SUBJ`, `ROLE_PRED`, `ROLE_OBJ` are workspace-init seeded random HVs, fixed for the database lifetime.

Why bundling: a single episode often has multiple triples. Bundling produces one signature similar to all the triple bindings, retrievable by any of them. Example: episode "Sarah said Bawri is thai and Bawri is in Bandra" has two triples. The bundled signature matches partial queries about Sarah, Bawri, thai, or Bandra.

### v2.1: multimodal episode encoding

In v2.1, the episode signature additionally binds modality components:

```rust
pub fn encode_multimodal_episode(
    text_triples: &[Triple],
    video_sig: Option<HV>,
    audio_sig: Option<HV>,
    text_sig: Option<HV>,
    goal_id: Option<GoalId>,
    belief_ids: &[BeliefId],
    time_bucket: TimeBucket,
) -> HV {
    let mut episode = encode_episode_signature(text_triples);

    if let Some(sv) = video_sig { episode ^= ROLE_VIDEO.bind(&sv); }
    if let Some(sa) = audio_sig { episode ^= ROLE_AUDIO.bind(&sa); }
    if let Some(st) = text_sig  { episode ^= ROLE_TEXT.bind(&st); }
    if let Some(g) = goal_id    { episode ^= ROLE_GOAL.bind(&goal_signature(g)); }
    for b in belief_ids         { episode ^= ROLE_BELIEF.bind(&belief_signature(*b)); }
    episode ^= ROLE_TIME.bind(&time_signature(time_bucket));

    episode
}
```

The bundle stays factorable: any component can be recovered via XOR with its ROLE_* HV, cleaned up via nearest-neighbor lookup in the relevant codebook. See [neurosymbolic.md](./neurosymbolic.md) for the full factorization mechanics.

## The tiered cascade

Recall never returns the empty set (constitution article VI). It falls through tiers until it finds matches or hits the `tier_floor`:

```
Tier A — Exact         canonical entity match via concept index
                       confidence 1.0
Tier B — Similarity    HDC structured signature similarity,
                       POPCOUNT over inverted-index intersection
                       confidence band [0.6, 0.95]
Tier C — Gist          raw-text gist signature similarity
                       confidence band [0.3, 0.6]
Tier D — Nearest       best-effort nearest neighbors,
                       low_confidence flag, confidence ≤ 0.3
```

### Tier A — Exact concept match

If the cue contains a known concept name (resolved via the concept index), retrieve all episodes referencing that concept's `ConceptId`. Returned with confidence 1.0.

```rust
async fn tier_a(&self, query: &Query) -> Result<Vec<RecallMatch>> {
    if let Some(name) = &query.entity_name {
        if let Some(concept_id) = self.store.lookup_concept(name).await? {
            return self.store.episodes_for_concept(concept_id).await;
        }
    }
    Ok(vec![])
}
```

Phase 4 ships this.

### Tier B — Structured signature similarity

If extraction (layer 2) can build a partial triple from the cue, encode that as a partial signature and find episodes whose stored signatures have high overlap.

```rust
async fn tier_b(&self, query: &Query) -> Result<Vec<RecallMatch>> {
    let partial_sig = encode_partial_signature(&query.extracted_triples)?;
    let candidates = self.inverted_index_intersection(&partial_sig).await?;
    let mut scored = vec![];
    for ep_id in candidates {
        let ep_sig = self.store.load_signature(ep_id)?;
        let sim = ep_sig.similarity(&partial_sig);
        if sim > 0.6 { scored.push((ep_id, sim)); }
    }
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    Ok(scored.into_iter().take(query.k).map(|(id, conf)| RecallMatch::new(id, conf, Tier::Similarity)).collect())
}
```

Phase 3 (extraction) unlocks tier B. The inverted index by bit-set membership prevents scanning all signatures every query.

### Tier C — Gist signature similarity

Every episode also has a gist signature derived from a hash of bigrams/trigrams of its raw text. Tier C searches by gist similarity when the structured tier finds nothing.

```rust
async fn tier_c(&self, query: &Query) -> Result<Vec<RecallMatch>> {
    let gist_sig = encode_gist_signature(&query.cue_text);
    let mut scored = vec![];
    for (ep_id, ep_gist) in self.store.all_gists().await? {
        let sim = ep_gist.similarity(&gist_sig);
        if sim > 0.3 { scored.push((ep_id, sim)); }
    }
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    Ok(scored.into_iter().take(query.k).map(|(id, conf)| RecallMatch::new(id, conf, Tier::Gist)).collect())
}
```

Phase 4 ships this. For >1M episodes, replace the full scan with LSH (v0.3+).

### Tier D — Nearest-neighbor fallback

Best-effort. Even if tier B and tier C find nothing above their thresholds, return the closest matches with `low_confidence: true`. Agents always get *something* to work with.

```rust
async fn tier_d(&self, query: &Query) -> Result<Vec<RecallMatch>> {
    let cue_sig = encode_gist_signature(&query.cue_text);
    let mut all_episodes = self.store.recent_episodes(query.recency_window).await?;
    all_episodes.sort_by_key(|ep| ep.gist_signature.hamming(&cue_sig));
    Ok(all_episodes.into_iter().take(query.k).map(|ep|
        RecallMatch::new(ep.id, 0.2, Tier::NearestNeighbor)
            .with_low_confidence_flag()
    ).collect())
}
```

Phase 4 ships this.

### The full recall cascade

```rust
pub async fn recall(&self, query: Query) -> Result<Recall> {
    let start = Instant::now();
    let mut matches = self.tier_a(&query).await?;

    if matches.is_empty() && query.tier_floor <= Tier::Similarity {
        matches = self.tier_b(&query).await?;
    }
    if matches.is_empty() && query.tier_floor <= Tier::Gist {
        matches = self.tier_c(&query).await?;
    }
    if matches.is_empty() && query.tier_floor <= Tier::NearestNeighbor {
        matches = self.tier_d(&query).await?;
    }

    let tier_used = matches.first().map(|m| m.tier).unwrap_or(Tier::Exact);
    matches = self.apply_goal_bias(matches).await?;
    matches = self.apply_recency_boost(matches, &query).await?;
    matches = self.apply_bitemporal_filter(matches, &query).await?;
    matches.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
    matches.truncate(query.k);

    let attention_trace = if query.trace_attention {
        Some(self.build_attention_trace(&query, &matches).await?)
    } else { None };
    if let Some(ref t) = attention_trace {
        self.emit_learning_event(LearningEvent::AttentionTraced { recall_id: t.id, signatures_considered: t.candidates.len(), at: Utc::now() }).await?;
    }

    let semantic_atoms = self.recall_semantic_atoms(&query).await?;
    let beliefs = self.recall_beliefs(&query).await?;
    let active_goals = self.active_goals_for_query(&query).await?;

    Ok(Recall {
        matches, semantic_atoms, beliefs, active_goals,
        tier_used,
        elapsed_ms: start.elapsed().as_millis() as u32,
        attention_trace,
    })
}
```

## Goal-biased retrieval (floor 6 → layer 1)

When active goals exist, their HDC signatures up-weight related matches.

```rust
async fn apply_goal_bias(&self, mut matches: Vec<RecallMatch>) -> Result<Vec<RecallMatch>> {
    let active_goals = self.active_goals().await?;
    if active_goals.is_empty() { return Ok(matches); }
    let goal_sig = bundle(&active_goals.iter().map(|g| g.signature.clone()).collect::<Vec<_>>());
    for m in matches.iter_mut() {
        let ep_sig = self.store.load_signature(m.episode_id)?;
        let bias = ep_sig.similarity(&goal_sig) * GOAL_BIAS_WEIGHT;
        m.confidence = (m.confidence * (1.0 + bias)).min(1.0);
        m.goal_biased = bias > 0.05;
    }
    Ok(matches)
}
```

`GOAL_BIAS_WEIGHT` default 0.3. Tunable per-query via `Query::with_goal_bias(weight)`.

Mirrors PFC's role in biasing attention toward goal-relevant memories (Miller & Cohen 2001).

## Bi-temporal filtering

Every recall has implicit (or explicit) `valid_as_of` and `transaction_as_of` timestamps. Episodes outside the valid window or superseded as of the transaction time are filtered.

```rust
async fn apply_bitemporal_filter(
    &self, mut matches: Vec<RecallMatch>, query: &Query
) -> Result<Vec<RecallMatch>> {
    let valid_t = query.valid_as_of.unwrap_or_else(Utc::now);
    let tx_t = query.transaction_as_of.unwrap_or_else(Utc::now);
    matches.retain(|m| {
        let ep = self.store.episode(m.episode_id).unwrap();
        ep.is_valid_at(valid_t) && !ep.is_superseded_at(tx_t) && ep.tombstoned_at.map_or(true, |t| t > tx_t)
    });
    Ok(matches)
}
```

## Temporal retrieval (v0.2 — `feat/temporal-retrieval`)

Pure cue-driven recall scored 40% on temporal queries in homn's
eval (vs 80% factual / 100% commitment), because the cue for a
temporal question — e.g. "when did I send the pricing quote?" —
typically shares no tokens with the answer episode ("Friday").
This release adds four orthogonal temporal retrieval features on top
of the existing cascade without changing its tiers:

| Field on `Query`              | Role                                                    |
|-------------------------------|---------------------------------------------------------|
| `time_window: Option<(from, to)>` | interval-overlap filter applied at every tier      |
| `subject: Option<ConceptId>`  | restrict to episodes linked to a specific concept       |
| `recency_weight: f32`         | blend cue-similarity with a recency score (post-rank)   |
| `time_anchor: Option<DateTime>` | anchor timestamp for the recency decay                |

All four default to "off" so existing callers see byte-identical
behavior.

### `time_window` — interval-overlap filter

```rust
pub fn overlaps_window(&self, from: DateTime<Utc>, to: DateTime<Utc>) -> bool {
    let effective_end = self.end.unwrap_or(self.start);
    self.start <= to && effective_end >= from
}
```

An episode is in-window iff `valid_time.start <= to` AND
`valid_time.end.unwrap_or(start) >= from`. Per the spec an
open-ended episode's effective end equals its start (point-in-time
semantics), so a 2026-01-01 open-ended episode lives only in a
window containing 2026-01-01 itself — that avoids the surprise of
a 2026 fact leaking into a 2027 query just because nobody
superseded it.

The filter applies at every tier that sweeps the scan directory:
`scan_signatures`, `scan_phi`, `scan_phi_with_pick`, `tier_l_lexical`,
`tier_a_exact`'s `rerank_via_idf`. `Query::as_of` and
`time_window` are independent and can be set together.

### `subject` — entity filter

```rust
pub subject: Option<ConceptId>,

let mut seen = HashSet::new();
if let Some(subject_set) = subject_filter {
    for &id in subject_set { seen.insert(id); }
} else { /* original cue-token concept resolution */ }
let reranked = self.rerank_via_idf(&seen, query, /*subject_is_filter=*/true)?;
```

When `subject` is set, the cascade restricts to episodes linked to
that concept via `concept_episodes`. In "subject-filter" mode the
IDF rerank fills in score-0 entries for every candidate so the
result always returns "every episode about X" (sorted by cue
overlap within the subject set) — a cue that shares no tokens
with the subject's episodes still surfaces the full set.

### `recency_weight` + `time_anchor` — recency rerank

```rust
fn apply_recency_rerank(
    matches: &mut [RecallMatch], weight: f32, anchor: DateTime<Utc>, half_life: f32,
) {
    for m in matches.iter_mut() {
        let dt = (anchor - valid_start).num_seconds().abs() as f32;
        let recency = (-dt / half_life).exp();
        m.confidence = (1.0 - weight) * cue_score + weight * recency;
    }
}
```

The blend happens after goal-bias so the final sort reflects every
reranking signal. Default half-life is 7 days. Symmetric around
the anchor — an episode dated a week before *or* after scores the
same — matching homn's "around then"-style phrasing.

### `Store::list_episodes_in_range(from, to, limit)`

```rust
pub fn list_episodes_in_range(
    &self, from: DateTime<Utc>, to: DateTime<Utc>, limit: usize,
) -> Result<Vec<Episode>> {
    // full scan, filter by overlap, sort by valid_time.start ASC, take limit
}
```

Direct chronological listing keyed on `valid_time.start`. Backed
by a full-table scan today; a secondary index on `valid_time.start`
is the planned O(log N + k) follow-up (the cost is noted in the
method's doc comment). Exposed through `Agidb::timeline(subject,
from, to, limit)` as the direct backing for homn's `timeline`
MCP tool.

## Attention trace (floor 7 audit)

When `query.trace_attention = true`, agidb records which signatures were considered, which scored highest, and why each was retained or rejected. The trace lands in floor 7's learning log; the agent can later ask "what was I attending to during recall_id X?"

```rust
pub struct AttentionTrace {
    pub id: RecallId,
    pub query: Query,
    pub candidates: Vec<AttentionCandidate>,
    pub goal_signature: Option<HV>,
    pub recency_window: Duration,
    pub timestamp: DateTime<Utc>,
}

pub struct AttentionCandidate {
    pub episode_id: EpisodeId,
    pub similarity: f32,
    pub goal_bias: f32,
    pub recency_boost: f32,
    pub final_confidence: f32,
    pub retained: bool,
    pub rejection_reason: Option<String>,
}
```

Default off (overhead). Opt-in for debugging, RL exploration tracking, or compliance auditing.

## Performance characteristics

| Operation | 100k episodes | 1M episodes | Implementation |
|---|---|---|---|
| Tier A | < 1ms | < 1ms | concept index hash lookup |
| Tier B (with extraction) | ~10ms | ~50ms | inverted index intersection + POPCOUNT |
| Tier C | ~5ms | ~50ms (with LSH ~10ms) | full gist scan (LSH at scale) |
| Tier D | ~5ms | ~50ms | bounded recency window scan |
| Goal-bias pass | < 1ms | < 5ms | one similarity per match (k=20) |
| Bi-temporal filter | < 1ms | < 5ms | redb point lookups |
| Attention trace build | ~5ms | ~10ms | only if enabled |
| **Total p95** | **< 50ms** | **< 100ms** | target |

These are CPU-only measurements. No GPU. No external services. No LLM calls.

## What this layer doesn't do

- **Extract entities and relations from text.** That's layer 2 (`agidb-extract`).
- **Encode video/audio to dense latents.** That's layer 2 in v2.1 (`agidb-sensory`).
- **Persist anything to disk.** That's layer 3 (`agidb-core::store`).
- **Decide when to consolidate.** That's the consolidation worker (separate module).
- **Run any LLM.** Read path is deterministic math (constitution article IV).

## What enables what (the dependency graph)

```
HDC kernel (phase 1, done)
   ↓
Episode encoding (phase 4, done)
   ↓
Tier A + C + D recall (phase 4, done)
   ↓
Goal-bias + recency + bi-temporal filtering (phase 4, done)

Tier B recall (phase 4, blocked on phase 3)
   ↓
Triple extraction (phase 3) ─────────────┐
                                          ↓
                                  Belief extraction (phase 9)
                                          ↓
                                  Goal-biased recall fully active

Multimodal episode encoding (phase 14) — v2.1
   ↓
Multimodal recall (phase 14) — v2.1 unbind for per-modality search

Attention trace (phase 10)
   ↓
Self-model floor 7 fully active

Cognitive benchmark suite (phase 13)
   ↓
Decision gate (phase 7)

BAMS benchmark (phase 16) — v2.1
   ↓
v2.1 milestone, ICLR 2026 MemAgents paper
```

## Why layer 1 is the wedge

Every other agent memory system embeds its similarity layer in either an LLM API call or a vector DB query. agidb's similarity layer is **CPU-local POPCOUNT over 1KB signatures**. Three orders of magnitude faster than the API-call path. Three orders of magnitude cheaper. Zero external dependencies in the read path.

That's the wedge. Layer 1 is what makes agidb the substrate that gets the latency, the cost, the offline operation, the determinism. Everything else (cognitive primitives, brain-alignment, BAMS) is built on this layer's properties.

The v2.1 multimodal extension stays purely on layer 1: dense latents from V-JEPA 2 / Wav2Vec-BERT / Llama-3.2-3B project to 8192-bit HVs, then layer 1 retrieval works identically. The encoder pipeline is layer 2 in v2.1; the math at the retrieval level is the same as v2.0.

That's why brain-alignment is additive (constitution article XVIII). Layer 1 isn't getting rewritten for v2.1; it's getting a new write-path input source. The fundamental retrieval math stays exactly the same.
