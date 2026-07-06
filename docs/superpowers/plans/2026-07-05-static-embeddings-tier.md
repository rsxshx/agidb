# agidb — Semantic Tier via Static Embeddings

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the exact-class loss (0.000 hit@1 on the 10k benchmark) and the temporal-class loss (0.020) by adding a deterministic, off-the-shelf static text-embedding tier between tier C (gist) and tier D (nearest-neighbor). Hits the assessment's recommendation: a model2vec-class lookup table that costs nothing at read time and adds zero network calls, zero LLM cost, and zero trained-from-scratch weight — meeting the constitution's Article IV "no LLM and no network in the read path" bar.

**Architecture:** Insert a new tier E — semantic — between tier C and tier D in `recall.rs`. Tier E projects a static text embedding (e.g. potion-base-8M, a 120-dim lookup table with a known seed) through a fixed, deterministic Charikar random projection to an 8192-bit HV, then phi-scans the scan directory the same way tier B does. The projection is a frozen `HashMap<i32, Vec<usize>>` (or precomputed `Vec<HV>`) keyed by the sign-bit index. Generated once at startup from a fixed seed; never mutated.

**Tech stack:** Add `tokenizers` (HuggingFace line splitter, model2vec compatibility) and a small statically-loaded `potion-base-8M` vocab file. OR, if the dep footprint is too heavy, ship a hand-rolled "tiny hash-trick" embedder that produces a comparable-surrogate semantic signal at ~30 dim. **Decision deferred to Task 1's probe** — pick whichever gives meaningful paraphrase recall in <200 MB dependency footprint and zero runtime model inference.

## Global constraints (same as end-to-end-substrate-fix plan)

- Rust top to bottom. No Python/JS. ONNX via `ort` only (already present if needed).
- **No LLM, no network, no model inference in the read path.** The static embedder is a loaded vector table — a lookup, not a computation. Article IV.
- Never return the empty set under default `tier_floor` when ≥1 non-tombstoned episode exists. Article VI.
- Conventional commits. No attribution lines.
- TDD: failing test first for every behavior change; `cargo test --workspace` green before commit.
- Files ≤ 800 lines; split new modules by responsibility.
- Don't modify `.specify/memory/constitution.md`.

---

## State of the working tree

Read before Task 1. The end-to-end-substrate-fix plan (`docs/superpowers/plans/2026-07-04-end-to-end-substrate-fix.md`) shipped and its `634460a` commit is HEAD. This plan extends that work without touching it.

Test status from HEAD: 165 passed, 0 failed. 13 MCP tools. 100-row extraction gold set. Floor-1 sensory buffer. Scan-directory mmap recall. Phi-corrected tier B.

The honest measurement this plan must beat: **agidb exact hit@1 = 0.000**, **agidb temporal hit@1 = 0.020** on the 10k synthetic corpus (`bench/RESULTS.md`). Goal: ≥ 0.50 exact, ≥ 0.50 temporal, hold or improve noisy (currently 0.90).

---

### Task 1: Probe — pick the static embedder

**Files:**
- Create: `crates/agidb-core/src/semantic.rs` (sketch)
- Modify: `Cargo.toml` and `crates/agidb-core/Cargo.toml` (add candidate deps)

**Decision criteria:** the chosen embedder must (1) produce a fixed-dim float vector per text input, (2) be loadable as a flat file at startup in ≤ 200 MB total deps, (3) work fully offline (no API calls), (4) have a deterministic output for a given input + seed, (5) come close to BM25 quality on the noisy-cue class we already win.

**Probe steps:**
1. Try `potion-base-8M` (model2vec, ~30 MB, deterministically derived from a static corpus via FastText-style training). If available on HuggingFace, fetch via `hf-hub`.
2. Try the existing `tokenizers` crate as the splitter; rank fallback if not.
3. Try a hand-rolled "feature hash" embedder as plan B: tokenize → 2- and 3-grams → murmur-hash into 256 buckets → L2 normalize. This is the model2vec spirit (count-based + hashing trick) at zero-dep cost. Likely degrades but should be a real semantic signal vs. raw token-bundle gist.
4. Decide based on: compile success, dependency size, and a 5-minute test on 20 paraphrase queries (built ad-hoc) showing non-trivial overlap.

**Output of this task:** a `pub trait Embedder { fn embed(&self, text: &str) -> Vec<f32>; }` in `semantic.rs`, and one concrete implementation (`pub struct FeatureHashEmbedder` for plan B) compiled and tested. If plan A (model2vec) works on the first probe, swap implementations and keep plan B as a doc comment fallback.

- [ ] **Step 1: Write failing test for the embedder trait**

Append to a new `crates/agidb-core/tests/semantic_properties.rs`:

```rust
//! Semantic tier — static embedding invariants.

use agidb_core::semantic::Embedder;

#[test]
fn embedder_returns_fixed_dim_vector() {
    let emb = agidb_core::semantic::default_embedder();
    assert_eq!(emb.dim(), 256, "default embedder must produce 256-dim");
    let v = emb.embed("Sarah recommended Bawri");
    assert_eq!(v.len(), 256);
    assert!(v.iter().any(|&x| x != 0.0), "must not be all zeros");
}

#[test]
fn deterministic_and_seeded() {
    let emb = agidb_core::semantic::default_embedder();
    let a = emb.embed("pad thai for dinner");
    let b = emb.embed("pad thai for dinner");
    assert_eq!(a, b, "same input must produce same vector");

    let c = emb.embed("thai food tonight");
    // Cosine must be positive but not 1.0 — paraphrase ≠ duplicate.
    let cos = agidb_core::semantic::cosine(&a, &c);
    assert!(cos > 0.1 && cos < 0.99, "paraphrase cosine = {cos}");
}

#[test]
fn unrelated_texts_have_low_similarity() {
    let emb = agidb_core::semantic::default_embedder();
    let food = emb.embed("Sarah recommended Bawri");
    let code = emb.embed("the build pipeline is broken again");
    let cos = agidb_core::semantic::cosine(&food, &code);
    assert!(cos < 0.3, "unrelated cosine = {cos} (must stay below 0.3)");
}
```

- [ ] **Step 2: Verify compile error**

Run `cargo test -p agidb-core --test semantic_properties 2>&1 | tail -3`. Expect: `no module semantic`.

- [ ] **Step 3: Stub the module**

Create `crates/agidb-core/src/semantic.rs` with `Embedder` trait + `cosine` + `default_embedder` (panicking) and add `pub mod semantic;` to `lib.rs`. Confirm trait tests can compile and fail at runtime.

- [ ] **Step 4: Probe + choose**

Try `potion-base-8M` via `hf-hub` + `tokenizers` in a side branch first. If it loads cleanly under 200 MB and `cargo build -p agidb-core` finishes in < 60s, go with it. Otherwise implement `FeatureHashEmbedder`:

```rust
pub struct FeatureHashEmbedder {
    dim: usize,
}

impl Embedder for FeatureHashEmbedder {
    fn dim(&self) -> usize { self.dim }
    fn embed(&self, text: &str) -> Vec<f32> {
        // Lowercase + tokenize on whitespace + unicode categories.
        let tokens: Vec<String> = agidb_core::episode::tokenize(text)
            .into_iter()
            .map(|t| t.to_lowercase())
            .collect();
        // Bucket by hashing unigrams + bigrams.
        let mut v = vec![0f32; self.dim];
        for (i, t) in tokens.iter().enumerate() {
            let h = stable_hash(t.as_bytes());
            let idx = (h as usize) % self.dim;
            v[idx] += 1.0;
            if i + 1 < tokens.len() {
                let bigram = format!("{t}_{}", tokens[i + 1]);
                let h2 = stable_hash(bigram.as_bytes());
                let idx2 = (h2 as usize) % self.dim;
                v[idx2] += 0.5;
            }
        }
        // L2 normalize.
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 { for x in v.iter_mut() { *x /= norm; } }
        v
    }
}
```

`stable_hash` is `seahash` (already in workspace via `redb` transitively if not, add as a tiny dep).

- [ ] **Step 5: Re-run tests**

Run `cargo test -p agidb-core --test semantic_properties`. Expect all three green. If cosine for `pad thai for dinner` vs `thai food tonight` is < 0.1, the hashing-trick is too sparse and we need to enrich bigrams to 3-grams or add character n-grams.

- [ ] **Step 6: Decision doc**

Append to the plan file: which embedder was chosen (model2vec vs feature-hash), its footprint, and the measured cosine stats on the 20-query paraphrase probe. No code commits yet to keep the diff focused; this is just the choice.

---

### Task 2: Charikar random projection (embedder → HV)

**Files:**
- Modify: `crates/agidb-core/src/semantic.rs`
- Test: `crates/agidb-core/tests/semantic_properties.rs`

**Math:** For each dimension i in `[0, D)`, sample `r_i ∈ {+1, -1}^{embedder.dim}` from a fixed seed, compute `h_i = (Σ embedder[k] * r_i[k] > 0) ? 1 : 0`. This is the canonical Charikar / Kanerva random projection: any two embedder vectors u, v have `E[hamming_proj(u, v)] = (1 - cos(u, v)) / 2` for D large enough. With D = 8192 and embedder.dim = 256, the variance on the projected Hamming distance is tight enough that cosine ≈ 1 maps to popcount ≪ D/2.

**Implementation:**
- Projection matrix `r: Vec<Vec<i8>>` of shape `[D, embedder.dim]`, generated once at `Embedder::new()` via `StdRng::seed_from_u64(0xC417_4E5C_4152_4B41)` (`CHARIKA`).
- `project(&Vec<f32>) -> HV` does the matvec + sign accumulation.
- Cached: the embedder holds the projection, not the caller.

- [ ] **Step 1: Write failing test**

```rust
#[test]
fn projection_preserves_cosine_sign() {
    let emb = agidb_core::semantic::default_embedder();
    let a = emb.embed("Sarah recommended Bawri");
    let b = emb.embed("Bawri is Sarah's pick");
    let hv_a = emb.project(&a);
    let hv_b = emb.project(&b);
    // Related embedder vectors → hamming agreement >> 0.5 * D
    assert!(hv_a.similarity(&hv_b) > 0.55,
        "expected > 0.55, got {}", hv_a.similarity(&hv_b));
}

#[test]
fn projection_flips_unrelated_to_low_overlap() {
    let emb = agidb_core::semantic::default_embedder();
    let food = emb.embed("Sarah recommended Bawri");
    let code = emb.embed("the build pipeline is broken");
    let hv_food = emb.project(&food);
    let hv_code = emb.project(&code);
    assert!(hv_food.similarity(&hv_code) < 0.55,
        "expected < 0.55, got {}", hv_food.similarity(&hv_code));
}

#[test]
fn projection_is_deterministic_across_instances() {
    let e1 = agidb_core::semantic::default_embedder();
    let e2 = agidb_core::semantic::default_embedder();
    let v = vec![0.1; e1.dim()];
    assert_eq!(e1.project(&v), e2.project(&v),
        "projection must be seeded — same seed, same matrix");
}
```

- [ ] **Step 2: Run to verify failure**

`cargo test -p agidb-core --test semantic_properties`. Expect `no method named project`.

- [ ] **Step 3: Implement projection**

```rust
const CHARIKAR_SEED: u64 = 0xC417_4E5C_4152_4B41; // "CHARIKA"

pub trait Embedder: Sync + Send {
    fn dim(&self) -> usize;
    fn embed(&self, text: &str) -> Vec<f32>;
    fn project(&self, vec: &[f32]) -> HV;
    fn project_text(&self, text: &str) -> HV {
        self.project(&self.embed(text))
    }
}

pub struct StaticEmbedder {
    embed: Box<dyn Fn(&str) -> Vec<f32> + Send + Sync>,
    dim: usize,
    projection: Vec<[i8; 256]>, // [D][embedder.dim]
}

impl StaticEmbedder {
    pub fn new(embed_fn: impl Fn(&str) -> Vec<f32> + Send + Sync + 'static, embedder_dim: usize) -> Self {
        assert!(embedder_dim <= 256);
        let mut rng = StdRng::seed_from_u64(CHARIKAR_SEED);
        let projection = (0..D)
            .map(|_| {
                let mut row = [0i8; 256];
                for x in row.iter_mut() { *x = if rng.gen() { 1 } else { -1 }; }
                row
            })
            .collect();
        Self { embed: Box::new(embed_fn), dim: embedder_dim, projection }
    }
}

impl Embedder for StaticEmbedder {
    fn dim(&self) -> usize { self.dim }
    fn embed(&self, text: &str) -> Vec<f32> { (self.embed)(text) }
    fn project(&self, vec: &[f32]) -> HV {
        let mut bits = [0u64; D_U64];
        for (i, row) in self.projection.iter().enumerate() {
            let s: i32 = row[..self.dim].iter()
                .zip(vec.iter())
                .map(|(r, x)| (*r as i32) * (*x * 1000.0) as i32)
                .sum();
            if s > 0 { set_bit(&mut bits, i); }
        }
        HV::from_u64s(bits)
    }
    fn project_text(&self, text: &str) -> HV {
        self.project(&self.embed(text))
    }
}
```

- [ ] **Step 4: Run tests**

All three new tests green.

- [ ] **Step 5: Expose via lib.rs**

`pub mod semantic` + `pub use semantic::{Embedder, StaticEmbedder, default_embedder}`.

- [ ] **Step 6: Commit**

```
perf(core): static-embedder semantic tier (Charikar projection to 8192-bit HV)

NEW embedder trait: deterministic, fixed-dim, lookup-table-only — no
inference, no network. Charikar random projection from embedder.dim to D
via a seed-frozen projection matrix (seed=CHARIKA).  Embedder choice
noted in plan doc; <X> chosen for ~Y MB total footprint. Adding tier E
between tier C (gist) and tier D (NN) is Task 3.
```

---

### Task 3: Persist semantic HV in signatures.dat + scan directory

**Files:**
- Modify: `crates/agidb-core/src/types.rs` (`Episode` gains `embedding_offset: u64`)
- Modify: `crates/agidb-core/src/store.rs` (`Store::observe` appends + bumps version → 3)
- Modify: `crates/agidb-core/src/recall.rs` (scan directory gains `embedding_offset: u64`)

**Semantics:** Every `observe(Episode, signature, embedder)` writes three HVs into `signatures.dat`: structured signature, gist, and semantic embedding. Pre-v3 stores fail to load with `FormatVersion` error (with the existing JSONL-import escape hatch — `#[serde(default)]` on the new field).

- [ ] **Step 1: Write failing persistence test**

Append to `crates/agidb-core/tests/semantic_properties.rs`:

```rust
#[test]
fn episode_persists_semantic_hv_across_reopen() {
    let dir = tempfile::TempDir::new().unwrap();
    {
        let mut store = agidb_core::store::Store::open(
            agidb_core::store::StoreConfig::at(dir.path())
        ).unwrap();
        let id = store.observe_with_embedder(
            make_episode(1, "Sarah recommended Bawri", "Sarah", "recommended", "Bawri"),
            &sig_for("Sarah", "recommended", "Bawri"),
            &agidb_core::semantic::default_embedder(),
        ).unwrap();
        assert!(id.raw() >= 1);
    }
    let store = agidb_core::store::Store::open(
        agidb_core::store::StoreConfig::at(dir.path())
    ).unwrap();
    let stats = store.stats().unwrap();
    // Each observe appends 3 HVs: structured, gist, embedding.
    assert_eq!(stats.signatures, 3);
}
```

(Helper `make_episode` and `sig_for` come from the existing `recall_properties.rs` module; mirror them.)

- [ ] **Step 2: Verify failure**

`cargo test -p agidb-core --test semantic_properties`. Expect `no method named observe_with_embedder`.

- [ ] **Step 3: Bump format version + extend `Episode`**

```rust
// types.rs
#[serde(default)]
pub embedding_offset: u64,
```

```rust
// store.rs
pub const STORE_FORMAT_VERSION: u32 = 3;
```

`StoreConfig::at` uses the new version automatically.

`Store::observe`:
- accept `embedder: Option<&dyn Embedder>` after `signature: &HV` (defaulting to None preserves the existing test surface).

```rust
pub fn observe(
    &mut self,
    mut episode: Episode,
    signature: &HV,
    embedder: Option<&dyn Embedder>,
) -> Result<EpisodeId> {
    // ... existing signature + gist append ...
    episode.embedding_offset = if let Some(emb) = embedder {
        let hv = emb.project_text(&episode.text);
        if &hv == signature { sig_offset } else { self.signatures.append(&hv)? }
    } else {
        0
    };
    // ... store + scan_push with new field ...
}
```

Or, to keep the existing 2-arg `observe` callers compiling, add `observe_with_embedder` as a thin shim that defaults `signature` to None and stores the text-only path.

- [ ] **Step 4: Update scan entry**

`ScanEntry` gains `embedding_offset: u64`. `rebuild_scan_dir` populates it from `ep.embedding_offset` (defaulting to `ep.signature_offset` for backward-compat when `embedding_offset == 0` from a v2 import).

- [ ] **Step 5: Update `observe()` callers**

- `crates/agidb-extract/src/lib.rs`: keep `observe_text` taking the extractor only — extractors don't have embeddings. For embedding-on-observe, callers must use a newer API. **Defer this concern** to Task 5 (full semantic-tier integration). For Task 3, keep extract-path text-only and only test the new `observe_with_embedder` path explicitly.

- [ ] **Step 6: Update existing test files**

`crates/agidb-core/tests/{recall,storage,unlearn,consolidate}_properties.rs` and `crates/agidb-extract/src/lib.rs`: pass `None` (or set `embedding_offset: 0`) for the pre-existing `observe(Episode, &HV)` callsites. Use `#[serde(default)]` on the new field so old JSONL imports still load.

- [ ] **Step 7: Run tests — green**

`cargo test --workspace`. All existing tests still green; the new persistence test green too.

- [ ] **Step 8: Commit**

```
perf(core): persist semantic embedding HV in signatures.dat (STORE_FORMAT_VERSION=3)

Episode + ScanEntry gain embedding_offset; observe() takes an optional
Embedder and appends a Charikar-projected semantic HV per row. v2 stores
fail to open with FormatVersion; v2 JSONL imports still load via
serde(default).
```

---

### Task 4: Wire semantic tier into recall cascade

**Files:**
- Modify: `crates/agidb-core/src/recall.rs`
- Test: `crates/agidb-core/tests/semantic_properties.rs`

**Cascade order (final):** A (exact, weighted 1.0) → B (structured phi, [0.6, 0.95]) → **NEW E (semantic phi, [0.4, 0.7])** → C (gist raw sim, [0.3, 0.6]) → D (NN, cap 0.3).

Tier E confidence band sits below B because the embedder's paraphrase signal is broader — useful but lower precision than the structured role-bound match.

- [ ] **Step 1: Add `Semantic` variant to `Tier` enum**

```rust
// types.rs
pub enum Tier {
    Exact,
    Similarity,         // tier B
    Semantic,           // tier E — NEW
    Gist,
    NearestNeighbor,
}
```

Bump `Tier::depth()` to put Semantic between Similarity (1) and Gist (2).

- [ ] **Step 2: Write failing tier E test**

```rust
#[test]
fn recall_finds_paraphrase_relationships_no_structured_overlap() {
    let (mut store, _dir) = fresh_store_with_embedder();
    observe_with_embedder(
        &mut store,
        make_episode(1, "Sarah recommended Bawri", "Sarah", "recommended", "Bawri"),
    );
    // Cue has no token overlap with the episode's triple tokens ("Sarah"/"Bawri"/"recommend")
    // but semantically is clearly related.
    let r = store.recall(&Query::cue("good thai place suggestion")).unwrap();
    assert!(!r.matches.is_empty(),
        "paraphrase cue must find the stored episode via tier E");
    assert_eq!(r.tier_used, Tier::Semantic);
}
```

- [ ] **Step 3: Verify failure**

Expect `no variant Semantic for Tier`. Or, if you add the variant first, expect the test to fail because tier E isn't wired.

- [ ] **Step 4: Implement tier E**

Mirror tier B's `scan_phi`, but `pick` is the entry's `embedding_offset` and the floor is `TIER_E_PHI_FLOOR = 0.04` (slightly below B's 0.06 because the semantic signal is broader — paraphrases land lower than direct structured matches).

```rust
const TIER_E_PHI_FLOOR: f32 = 0.04;
const TIER_E_PHI_HI: f32 = 0.20;
const TIER_E_BAND: (f32, f32) = (0.4, 0.7);
```

In `run_cascade`, between the tier B and tier C blocks:

```rust
// Tier E — semantic similarity from a static-text embedding +
// Charikar projection. Sits below B because the signal is broader
// (paraphrase > role-bound overlap).
if Tier::Semantic.depth() <= query.tier_floor.depth() {
    if let Some(emb) = self.embedder.as_ref() {
        let query_hv = emb.project_text(&query.cue);
        let scored = self.scan_phi_with_pick(
            &query_hv,
            query,
            |e| e.embedding_offset,
        );
        let e = self.band_matches(
            &scored, query,
            TIER_E_PHI_FLOOR, TIER_E_PHI_HI,
            TIER_E_BAND, Tier::Semantic,
        )?;
        if !e.is_empty() { return Ok(self.finalize(e, query)); }
    }
}
```

The embedder lives on `Store` as `Option<Arc<dyn Embedder>>`. Optional because text-only stores (the 99% pre-embedding case) have no embedder and skip tier E cleanly.

- [ ] **Step 5: Run the new test — green**

- [ ] **Step 6: Run the full suite**

`cargo test --workspace`. All existing recall tests still green.

- [ ] **Step 7: Commit**

```
perf(core): tier E — semantic similarity via embedded embedding HV

Static-text embedding → Charikar projection → phi scan over scan_dir,
between tier B (structured) and tier C (gist). Confidence band [0.4, 0.7],
floor 0.04. Closes the exact-class loss without breaking the
no-LLM, no-network read path: the embedder is a loaded lookup table.
```

---

### Task 5: Facade + CLI + MCP exposure

**Files:**
- Modify: `crates/agidb/src/lib.rs` (`Agidb::open_with_embedder`, `Agidb::observe_with_embedder`)
- Modify: `crates/agidb-cli/src/main.rs` (`--embedder potion` flag on `observe`)
- Test: `crates/agidb-mcp/tests/dispatch.rs` (new tool `memory_observe_semantic`)

- [ ] **Step 1: Facade**

`Agidb::open_with_embedder(path, embedder: Arc<dyn Embedder>) -> CoreResult<Agidb>`. `observe_with_embedder(text, source)` async wrapper.

- [ ] **Step 2: CLI**

```rust
/// Load a static-text embedder at startup. Choices: potion (planned), hash (default).
#[arg(long, default_value = "hash")]
embedder: String,
```

The hash embedder is the always-on plan B; the `potion` choice loads the model2vec vocab (probed in Task 1 — wire only if it fits the footprint).

- [ ] **Step 3: MCP tool**

```rust
Tool {
    name: "memory_observe_semantic",
    description: "Same as memory_observe but routes through the semantic tier — use when the cue is a paraphrase and not a direct lookup.",
    schema: observe_semantic_schema,
    handler: observe_semantic,
},
```

Handler:

```rust
fn observe_semantic(ctx: &AgidbContext, args: Value) -> Result<ToolResult, McpError> {
    // similar to `observe`, but the embedder is loaded from `~/.config/agidb/embedder.bin` if present.
    // If no embedder is loaded, returns an error pointing the caller at the CLI flag.
}
```

Test: cue "thai place suggestion" against an episode "Sarah recommended Bawri" must return a match.

- [ ] **Step 4: Run all tests, commit**

```
feat(mcp+cli+facade): expose semantic-tier observe through all entry points

Agidb::open_with_embedder, `agidb observe --embedder <kind>`,
memory_observe_semantic MCP tool. Embedder is loaded lazily from
~/.config/agidb/embedder.bin if present; absent → tool returns a clear
error rather than degrading silently.
```

---

### Task 6: Re-benchmark and update RESULTS.md

**Files:**
- Modify: `crates/agidb-bench/src/corpus.rs` (new `QueryClass::Paraphrase`)
- Modify: `crates/agidb-bench/src/main.rs` (run includes tier E)
- Modify: `bench/RESULTS.md` (new section + new numbers)

- [ ] **Step 1: Add paraphrase query class to bench**

```rust
QueryClass::Paraphrase,
```

50 queries per class, equal split. Same doc pool. Cue: paraphrase via canned templates ("good X place suggestion" maps back to the predicate "recommends X"). Benchmark honesty: the paraphrase templates are intentional and only test what a static embedder was bought to fix; document this limitation.

- [ ] **Step 2: Run 10k with all five classes**

`cargo run -p agidb-bench --release -- --episodes 10000 --queries 250 --out bench`. Saves `results-10k.json`.

- [ ] **Step 3: Compare + document**

Append a new section to `bench/RESULTS.md` with: before/after exact + temporal + paraphrase hit@1, the per-class table with the new Paraphrase row, and a Limitations note that the paraphrase templates are designed to be tractable for a static-text embedder.

- [ ] **Step 4: Commit**

```
bench: rerun agidb vs FTS5 with semantic tier + paraphrase query class

10k corpus, 5 classes (exact / single-entity / noisy / temporal /
paraphrase). Tier E recovers exact-class hit@1 from 0.000 → X.XXX and
temporal from 0.020 → X.XXX while preserving noisy win. Results in
bench/RESULTS.md.
```

---

### Task 7: Docs truth pass for the new tier

**Files:**
- Modify: `README.md` (add tier E row to the status table, update status paragraph)
- Modify: `crates/agidb-mcp/src/lib.rs` (add `memory_observe_semantic` to the tool list)

- [ ] **Step 1: README**

| floor | status |
|---|---|
| 1. sensory buffer (surprise-gated) | ✅ shipped |
| 2. working memory (session-scoped recency) | 🚧 planned |
| 3. episodic memory (bi-temporal) | ✅ shipped |
| 4. semantic memory (consolidated atoms) | ✅ shipped |
| 4a. semantic-tier recall (static-embedder + Charikar) | ✅ shipped — see `bench/RESULTS.md` for the exact-class recovery numbers |
| 5. procedural memory | 🚧 types defined, retrieval planned |
| 6. goals + beliefs | ✅ shipped |
| 7. self-model (learning log + self-vector) | ✅ shipped |

Update the status paragraph to reference the new bench numbers.

- [ ] **Step 2: MCP doc**

Add the new tool to the list.

- [ ] **Step 3: Commit**

```
docs: note tier E in README + MCP docs

Seven-floor table gains a 4a row. Status paragraph references new
bench numbers. MCP lib.rs lists memory_observe_semantic.
```

---

## Self-review notes

- **Architecture coherence:** tier E sits where the constitution's "graceful degradation" article wants it — between structured (high precision) and gist (broad coverage). Charikar keeps the no-LLM-in-read-path promise by being a one-time load + per-query matvec over 256×8192 = 2 MiB frozen matrix.
- **Why not model2vec embeddings directly?** Charikar projection is a value-add because it lets the embedder stay small — even a hand-rolled feature-hash embedder at 256 dim gets a usable semantic signal through the projection. If plan A (model2vec) loads cleanly, the Charikar math turns it into an HDC-compatible HV without changing the embedding model.
- **Failure modes:** if the embedder is unseeded across machines, the projection diverges. The single seed constant + the documented freeze prevent this. Tier E degrades cleanly when the embedder is absent (`embedder.is_none()` skips tier E → tier C → tier D).
- **Type consistency:** `embedding_offset: u64` mirrors `signature_offset`/`gist_offset`; `ScanEntry` carries all three; `band_matches` and `scan_phi` already pick by closure, no signature changes needed.
- **Risks:** Task 1's embedder probe may find the model2vec footprint too large for the plan A path; the plan-B feature hash is the documented fallback. Task 4 tier E may need floor tuning per the measured phi distribution; expect 2–3 iterations within `[0.03, 0.10]` before landing.
- **What this plan does NOT do:** train a new relation extractor, add multimodal, write the ICLR paper, add Python bindings. All deferred per the assessment's "scope is the risk" advice.