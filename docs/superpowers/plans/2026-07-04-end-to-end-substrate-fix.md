# agidb End-to-End Substrate Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close every gap between agidb's README claims and its code — fast mmap-scan recall, a real Tier B, a real sensory buffer, the full MCP tool surface, a 100-sample extraction gold set, an honest benchmark against SQLite FTS5 — and publish measured results.

**Architecture:** agidb is an embedded Rust memory database: redb for metadata, an mmap'd `signatures.dat` flat file for 8192-bit hypervectors (HVs), and a tiered recall cascade (A exact → B structured similarity → C gist → D nearest-neighbor). This plan finishes the read path (scan directory over the mmap, density-corrected Tier B), adds the missing floor-1 sensory buffer, exposes the existing goals/beliefs/unlearn engine over MCP, and builds the benchmark harness that proves (or disproves) the performance claims.

**Tech Stack:** Rust (stable, workspace in `/home/rsx/Desktop/projx/agidb`), redb 2, memmap2, roaring, bincode, serde, clap 4, proptest, criterion, rusqlite (new, bundled) for the FTS5 baseline.

## Global Constraints

- **Rust top to bottom.** No Python/JS. The only permitted FFI is ONNX via `ort` (already present in agidb-extract). Do not add new FFI.
- **No LLM and no network calls in the read path** (`recall`, `what_about`, `between`). Constitution Article IV. Everything in this plan is deterministic math.
- **Never return the empty set** under the default `tier_floor` when ≥1 non-tombstoned episode exists (Article VI). Don't break the existing test `recall_never_returns_empty_under_default_floor`.
- **Conventional commits** (`feat:`, `fix:`, `perf:`, `docs:`, `test:`, `chore:`, `bench:`). **No attribution lines** (no Co-Authored-By — disabled globally for this user).
- **TDD**: write the failing test first for every behavior change. Run `cargo test --workspace` before every commit; it must be green.
- Files ≤ 800 lines; split new modules by responsibility.
- Run everything from the repo root `/home/rsx/Desktop/projx/agidb`.
- Do not modify `.specify/memory/constitution.md`.

---

## State of the working tree (read before Task 1)

The following changes are ALREADY MADE and UNCOMMITTED in the working tree. Task 1 finishes and commits them. Do not redo them; do read them.

| File | What changed |
|---|---|
| `crates/agidb-core/src/types.rs` | `Episode` gained `#[serde(default)] pub gist_offset: u64` after `signature_offset` |
| `crates/agidb-core/src/store.rs` | `STORE_FORMAT_VERSION: u32 = 2`; `StoreConfig::at` uses it; new `ScanEntry` struct + `scan_dir: Vec<ScanEntry>` + `scan_pos: HashMap<u64, usize>` on `Store`; `rebuild_scan_dir()` at open; `scan_push`/`scan_entries`/`scan_entry`/`scan_set_tombstoned` helpers; `observe()` computes + appends the gist HV (dedupes when identical to the signature) and pushes a scan entry; `supersede()` updates the entry's `valid_end` |
| `crates/agidb-core/src/unlearn.rs` | `TOMBSTONE_EPISODE` is now `pub(crate)`; `write_tombstone`/`restore_within_window` call `scan_set_tombstoned` |
| `crates/agidb-core/src/recall.rs` | Fully rewritten: scans the scan directory + mmap instead of deserializing every episode; Tier B implemented via `structured_cue_signature` (cue tokens → concepts, exact → case-insensitive → fuzzy dist 1 → bundle of `role_subj⊗HV(name)` and `role_obj⊗HV(name)`); `scan_signatures` with partial top-512 select; `band_matches` hydrates only survivors; tier A + goal-bias use the scan directory instead of per-row redb lookups |
| `crates/agidb-extract/src/lib.rs` | `Episode` literal gained `gist_offset: 0` |
| `crates/agidb-core/tests/{recall,storage,unlearn,consolidate}_properties.rs` | `Episode` literals gained `gist_offset: 0`; `recall_properties.rs` gained 6 new tests (tier B, tombstone exclusion+restore, reopen persistence, gist dedup) |
| `Cargo.toml` + `crates/agidb-cli/src/main.rs` | (pre-existing, unrelated) `tracing-subscriber` env-filter support — keep, commit with Task 1 |

**Test status:** `cargo test --workspace` → everything passes EXCEPT
`tier_b_matches_case_insensitive_entity_mentions` (in `crates/agidb-core/tests/recall_properties.rs`), which fails with *"distractor entities must stay below the tier-B floor"*.

### Why it fails — the density-skew bug (Task 1 fixes this)

`HV::bundle` uses **strict majority**: for an even number of inputs a 1-of-2 or 2-of-4 split resolves to 0. So `bundle([a, b])` is `a AND b` → only ~25% of bits set, not ~50%.

- The Tier-B cue signature is `bundle([SUBJ⊗e, OBJ⊗e])` → **25% dense**.
- A single-triple episode signature is `bundle([triple_hv, TIME⊗date])` → also **25% dense**.
- Raw Hamming agreement between two *independent* 25%-dense vectors is `0.25·0.25 + 0.75·0.75 = 0.625` — far above the 0.53 floor. Every episode, related or not, crosses the floor. That's the failure.

**Fix:** score Tier B with the **phi coefficient** (Pearson correlation on binary vectors), which is density-corrected: independent vectors score ≈ 0 regardless of density; correlated vectors score positive. For query `a`, stored `b`, with `n = 8192`, `pa = popcount(a)`, `pb = popcount(b)`, `h = hamming(a,b)`:

```text
n11 = (pa + pb − h) / 2                      # bits set in both
phi = (n·n11 − pa·pb) / sqrt(pa·pb·(n−pa)·(n−pb))
```

Expected values (verified analytically): unrelated pair ≈ 0 (σ = 1/√8192 ≈ 0.011); cue sharing one entity with a single-triple episode ≈ **0.167**; with a 3-triple episode ≈ **0.117**. Floor **0.06** is >5σ above noise and well below the thinnest signal. Tiers C/D stay on raw similarity (their existing floors are empirically validated by passing tests); the C/D density caveat is documented in a code comment only.

---

### Task 1: Density-corrected Tier B scoring (phi) + commit the hot-path work

**Files:**
- Modify: `crates/agidb-core/src/hdc.rs` (add `popcount`)
- Modify: `crates/agidb-core/src/store.rs` (cache `sig_popcount` in `ScanEntry`)
- Modify: `crates/agidb-core/src/recall.rs` (Tier B scores by phi)
- Test: `crates/agidb-core/tests/hdc_properties.rs` (phi properties), `crates/agidb-core/tests/recall_properties.rs` (already written, currently failing)

**Interfaces:**
- Consumes: `HV::hamming(&self, &HV) -> u32`, `ScanEntry`, `Store::scan_entries()`.
- Produces: `HV::popcount(&self) -> u32`; `pub(crate) fn phi_from_counts(n: f64, pa: f64, pb: f64, hamming: f64) -> f32` in `hdc.rs`; `ScanEntry.sig_popcount: u32`. Task 5's bench and any future tier reuse these.

- [ ] **Step 1: Write failing phi property tests**

Append to `crates/agidb-core/tests/hdc_properties.rs`:

```rust
// --- phi coefficient (density-corrected correlation) ------------------------

#[test]
fn phi_of_identical_vectors_is_one() {
    let a = HV::from_name("phi-self");
    let pa = a.popcount() as f64;
    let phi = agidb_core::hdc::phi_from_counts(8192.0, pa, pa, 0.0);
    assert!((phi - 1.0).abs() < 1e-3, "phi(x,x) must be ≈1, got {phi}");
}

#[test]
fn phi_of_independent_vectors_is_near_zero() {
    // 50 independent pairs: |phi| must stay within 5σ (σ = 1/√8192 ≈ 0.011).
    for i in 0..50 {
        let a = HV::from_name(&format!("phi-a-{i}"));
        let b = HV::from_name(&format!("phi-b-{i}"));
        let phi = agidb_core::hdc::phi_from_counts(
            8192.0,
            a.popcount() as f64,
            b.popcount() as f64,
            a.hamming(&b) as f64,
        );
        assert!(phi.abs() < 0.055, "independent pair {i} phi = {phi}");
    }
}

#[test]
fn phi_is_density_robust_where_raw_similarity_is_not() {
    // AND-bundles are ~25% dense. Raw similarity between two independent
    // sparse vectors inflates to ≈0.625; phi must stay ≈0.
    let a = HV::bundle(&[HV::from_name("sparse-a1"), HV::from_name("sparse-a2")]);
    let b = HV::bundle(&[HV::from_name("sparse-b1"), HV::from_name("sparse-b2")]);
    let raw = a.similarity(&b);
    assert!(raw > 0.55, "precondition: raw similarity inflates, got {raw}");
    let phi = agidb_core::hdc::phi_from_counts(
        8192.0,
        a.popcount() as f64,
        b.popcount() as f64,
        a.hamming(&b) as f64,
    );
    assert!(phi.abs() < 0.055, "phi must not inflate on sparse pairs, got {phi}");
}

#[test]
fn popcount_counts_set_bits() {
    assert_eq!(HV::zero().popcount(), 0);
    let mut bytes = [0u8; 1024];
    bytes[0] = 0b1010_1010;
    bytes[1023] = 0xFF;
    assert_eq!(HV(bytes).popcount(), 12);
}
```

(Check the imports at the top of `hdc_properties.rs`; it already imports `HV`. Add `use agidb_core::hdc::HV;` only if missing.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p agidb-core --test hdc_properties 2>&1 | tail -5`
Expected: compile error — `popcount` and `phi_from_counts` not found.

- [ ] **Step 3: Implement `popcount` and `phi_from_counts` in hdc.rs**

Add inside `impl HV` (after `similarity`):

```rust
    /// Number of set bits. Cached by the store's scan directory so the
    /// phi scoring path doesn't recount per query.
    pub fn popcount(&self) -> u32 {
        self.as_u64s().iter().map(|x| x.count_ones()).sum()
    }
```

Add at module level in `hdc.rs` (after the hamming backends):

```rust
/// Phi coefficient (Pearson correlation over binary vectors) from
/// pre-computed counts. Density-corrected: independent vectors score
/// ≈ 0 regardless of how sparse either side is, which raw hamming
/// similarity does not guarantee (majority-bundling an even number of
/// HVs yields AND-like ~25%-dense vectors that inflate raw agreement).
///
/// `n` is the dimensionality (use `D as f64`), `pa`/`pb` the popcounts
/// of the two vectors, `hamming` their hamming distance. Returns 0.0
/// for degenerate inputs (all-zero / all-one vectors).
pub fn phi_from_counts(n: f64, pa: f64, pb: f64, hamming: f64) -> f32 {
    if pa <= 0.0 || pa >= n || pb <= 0.0 || pb >= n {
        return 0.0;
    }
    let n11 = (pa + pb - hamming) / 2.0;
    let num = n * n11 - pa * pb;
    let den = (pa * pb * (n - pa) * (n - pb)).sqrt();
    (num / den) as f32
}
```

- [ ] **Step 4: Run the phi tests — pass**

Run: `cargo test -p agidb-core --test hdc_properties 2>&1 | tail -5`
Expected: `test result: ok.` (all tests, including the 4 new ones).

- [ ] **Step 5: Cache `sig_popcount` in the scan directory**

In `crates/agidb-core/src/store.rs`:

1. Add the field to `ScanEntry` (after `gist_offset`):

```rust
    /// Cached popcount of the HV at `sig_offset` — used by the tier-B
    /// phi scoring so the scan does one popcount per pair, not two.
    pub sig_popcount: u32,
```

2. In `rebuild_scan_dir()`, compute it when building each entry. Replace the `self.scan_push(ScanEntry { ... })` block with:

```rust
            let sig_popcount = self
                .signatures
                .read(ep.signature_offset)
                .map(|hv| hv.popcount())
                .unwrap_or(0);
            self.scan_push(ScanEntry {
                id: ep.id.raw(),
                sig_offset: ep.signature_offset,
                gist_offset: ep.gist_offset,
                sig_popcount,
                valid_start: ep.valid_time.start,
                valid_end: ep.valid_time.end,
                tombstoned: tombstoned.contains(&ep.id.raw()),
            });
```

(Note: `rebuild_scan_dir` currently borrows `self.db` via an open read
transaction while calling `self.scan_push(&mut self)` — if the borrow
checker complains, collect entries into a local `Vec` first, drop the
transaction, then push. `signatures` is a separate field so
`self.signatures.read(...)` inside the loop is fine.)

3. In `observe()`, the signature HV is already in hand; add `sig_popcount: signature.popcount(),` to the `ScanEntry` literal.

- [ ] **Step 6: Score Tier B with phi in recall.rs**

In `crates/agidb-core/src/recall.rs`:

1. Replace the tier-B constants:

```rust
/// Tier-B phi floor. Phi is density-corrected correlation: unrelated
/// episodes score ≈ 0 ± 0.011 (one σ at D=8192); a cue sharing one
/// entity with a stored episode scores ≈ 0.12–0.17 depending on how
/// many triples share the bundle. 0.06 is >5σ above noise and below
/// the thinnest genuine signal.
const TIER_B_PHI_FLOOR: f32 = 0.06;

/// Phi at (or above) which tier-B confidence saturates at the top of
/// its band.
const TIER_B_PHI_HI: f32 = 0.30;
```

Delete `TIER_B_SIM_FLOOR` / `TIER_B_SIM_HI`. Keep `TIER_B_BAND: (f32, f32) = (0.6, 0.95)`.

2. Add a phi-scoring scan (next to `scan_signatures`):

```rust
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
```

3. Factor the partial-sort tail of `scan_signatures` into a shared helper (module level, next to `calibrate_band`), and call it from both scans:

```rust
/// Partial-select the top slice before sorting — callers only consume
/// the band floor + k head of the list.
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
```

4. In `run_cascade`, the tier-B block becomes:

```rust
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
```

5. Add a comment on the tier-C block noting the known limitation:

```rust
        // Tier C/D score by raw hamming similarity over gist bundles.
        // Known caveat: majority-bundling an even token count produces
        // density-skewed gists that can inflate raw similarity between
        // sparse pairs (see phi_is_density_robust_where_raw_similarity_
        // is_not in hdc_properties). Acceptable at current floors;
        // switching C/D to phi is a calibrated follow-up.
```

- [ ] **Step 7: Run the failing tier-B test — pass**

Run: `cargo test -p agidb-core --test recall_properties 2>&1 | tail -8`
Expected: `test result: ok. 16 passed; 0 failed`.

If `tier_b_matches_case_insensitive_entity_mentions` still fails, print the diagnostics the test's assert message includes (it prints tier + per-match confidence). Debug rule: phi for the Sarah episode should be ≈0.167 and distractors within ±0.055 of 0. Only adjust `TIER_B_PHI_FLOOR` within [0.055, 0.10] if needed; never below 5σ = 0.055.

- [ ] **Step 8: Full workspace test run**

Run: `cargo test --workspace 2>&1 | grep -E "test result|FAILED" | sort | uniq -c`
Expected: all `ok`, 0 failed (≈125+ tests).

- [ ] **Step 9: Clippy + fmt**

Run: `cargo clippy --workspace --all-targets 2>&1 | grep -E "^error|^warning" | head` then `cargo fmt --all`
Expected: no errors; fix any new warnings in files this plan touched.

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "perf(core): scan-directory recall over mmap + density-corrected tier-B similarity

- Episode carries a persisted gist_offset; observe() appends the gist HV
  (deduped when identical to the structured signature)
- In-memory scan directory (id, offsets, popcount, valid time, tombstone)
  rebuilt at open, maintained by observe/supersede/unlearn/restore
- Tier C/D scan sweeps the mmap via the directory: no per-row redb reads,
  no per-query text re-encoding, batch tombstone filtering
- Tier B implemented: cue tokens -> concepts (exact/ci/fuzzy-1) ->
  role-bound structured cue signature, scored by phi (density-corrected
  binary correlation) with cached popcounts
- STORE_FORMAT_VERSION=2 (bincode layout change); old JSONL exports
  still import via serde(default)"
```

---

### Task 2: Sensory buffer (floor 1) with surprise gating

**Files:**
- Create: `crates/agidb-core/src/sensory.rs`
- Modify: `crates/agidb-core/src/lib.rs` (add `pub mod sensory;` next to the other module decls)
- Modify: `crates/agidb-core/src/store.rs` (register table in `open()`)
- Modify: `crates/agidb/src/lib.rs` (facade wrappers)
- Modify: `crates/agidb-cli/src/main.rs` (`sense` + `sensory` subcommands)
- Test: `crates/agidb-core/tests/sensory_properties.rs`

**Interfaces:**
- Consumes: `encode_gist_signature(&str) -> HV`, `Store::scan_entries()`, `Store::observe`, `Store::next_episode_id`, `store::{encode, decode}` (pub(crate)).
- Produces:
  - `pub const SURPRISE_PROMOTION_THRESHOLD: f32 = 0.4`
  - `pub const DEFAULT_SENSORY_CAPACITY: u64 = 1000`
  - `pub struct SensoryFrame { pub id: u64, pub text: String, pub at: DateTime<Utc>, pub surprise: f32, pub promoted: Option<EpisodeId> }`
  - `pub struct SensoryObservation { pub frame_id: u64, pub surprise: f32, pub promoted: Option<EpisodeId> }`
  - `Store::surprise_score(&self, text: &str) -> Result<f32>`
  - `Store::observe_sensory(&mut self, text: &str) -> Result<SensoryObservation>`
  - `Store::sensory_frames(&self, limit: usize) -> Result<Vec<SensoryFrame>>` (newest first)
  - Facade: `Agidb::{observe_sensory, surprise_score, sensory_frames}` (async, same shapes)
  - Task 3's `memory_sense` MCP tool consumes `observe_sensory`.

**Semantics (write these in the module doc):** surprise = `1 − max_similarity(gist(text), gist of the most recent 64 non-tombstoned episodes)`. Empty store → 1.0 (everything is surprising). Empty/whitespace text → 0.0 (never promote). Frames with `surprise ≥ 0.4` are promoted to episodic memory as text-only episodes with `provenance.source = "sensory"`. The frame ring buffer keeps the last 1000 frames in redb. This is deliberately *recall-shaped novelty detection*, not belief-based prediction — the doc comment must say so.

- [ ] **Step 1: Write the failing tests**

Create `crates/agidb-core/tests/sensory_properties.rs`:

```rust
//! Floor 1 — sensory buffer + surprise gating invariants.

use agidb_core::sensory::{SURPRISE_PROMOTION_THRESHOLD};
use agidb_core::store::{Store, StoreConfig};
use agidb_core::types::Query;
use tempfile::TempDir;

fn fresh_store() -> (Store, TempDir) {
    let dir = TempDir::new().expect("tempdir");
    let store = Store::open(StoreConfig::at(dir.path())).expect("open");
    (store, dir)
}

#[test]
fn first_frame_on_empty_store_is_maximally_surprising_and_promoted() {
    let (mut store, _dir) = fresh_store();
    let obs = store
        .observe_sensory("Sarah recommended Bawri in Bandra")
        .expect("sense");
    assert!((obs.surprise - 1.0).abs() < 1e-6, "empty store → surprise 1.0");
    assert!(obs.promoted.is_some(), "must promote above threshold");
}

#[test]
fn duplicate_text_is_not_promoted() {
    let (mut store, _dir) = fresh_store();
    store.observe_sensory("Sarah recommended Bawri in Bandra").expect("first");
    let second = store
        .observe_sensory("Sarah recommended Bawri in Bandra")
        .expect("second");
    assert!(
        second.surprise < SURPRISE_PROMOTION_THRESHOLD,
        "identical text must not be surprising, got {}",
        second.surprise
    );
    assert!(second.promoted.is_none());
}

#[test]
fn novel_text_after_familiar_text_is_promoted() {
    let (mut store, _dir) = fresh_store();
    store.observe_sensory("Sarah recommended Bawri in Bandra").expect("first");
    let novel = store
        .observe_sensory("quarterly compliance audit deadline moved to friday")
        .expect("novel");
    assert!(
        novel.surprise >= SURPRISE_PROMOTION_THRESHOLD,
        "disjoint text must stay surprising, got {}",
        novel.surprise
    );
    assert!(novel.promoted.is_some());
}

#[test]
fn empty_text_scores_zero_and_is_never_promoted() {
    let (mut store, _dir) = fresh_store();
    let obs = store.observe_sensory("   ").expect("sense");
    assert_eq!(obs.surprise, 0.0);
    assert!(obs.promoted.is_none());
}

#[test]
fn promoted_frame_is_recallable_as_episode() {
    let (mut store, _dir) = fresh_store();
    let obs = store
        .observe_sensory("the tokyo offsite starts on monday morning")
        .expect("sense");
    let id = obs.promoted.expect("promoted");
    let r = store
        .recall(&Query::cue("tokyo offsite monday"))
        .expect("recall");
    assert!(r.matches.iter().any(|m| m.episode_id == id));
}

#[test]
fn ring_buffer_evicts_beyond_capacity() {
    let (mut store, _dir) = fresh_store();
    // Capacity is 1000; insert 1010 distinct frames.
    for i in 0..1010u32 {
        store
            .observe_sensory(&format!("frame number {i} carries payload {}", i * 7))
            .expect("sense");
    }
    let frames = store.sensory_frames(2000).expect("frames");
    assert_eq!(frames.len(), 1000, "ring buffer must cap at capacity");
    // Newest first; the oldest surviving frame is #10.
    assert!(frames[0].text.contains("frame number 1009"));
    assert!(frames.last().unwrap().text.contains("frame number 10"));
}

#[test]
fn frames_survive_reopen() {
    let dir = TempDir::new().expect("tempdir");
    {
        let mut store = Store::open(StoreConfig::at(dir.path())).expect("open");
        store.observe_sensory("persist me across reopen").expect("sense");
    }
    let store = Store::open(StoreConfig::at(dir.path())).expect("reopen");
    let frames = store.sensory_frames(10).expect("frames");
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].text, "persist me across reopen");
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p agidb-core --test sensory_properties 2>&1 | tail -5`
Expected: compile error — `agidb_core::sensory` not found.

- [ ] **Step 3: Implement `sensory.rs`**

Create `crates/agidb-core/src/sensory.rs`:

```rust
//! Floor 1 — the sensory buffer.
//!
//! A capacity-bounded ring of raw text frames with surprise-gated
//! promotion to episodic memory. Surprise is *recall-shaped novelty*:
//! `1 − max_similarity(gist(text), gists of the most recent
//! [`SURPRISE_REFERENCE_WINDOW`] episodes)` — deliberately not a
//! belief-based prediction (that is a documented follow-up). Duplicate
//! or near-duplicate signal scores near 0 and stays in the buffer;
//! novel signal scores near 0.5+ and is promoted as a text-only
//! episode with `provenance.source = "sensory"`.

use chrono::{DateTime, Utc};
use redb::{ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};

use crate::episode::encode_gist_signature;
use crate::error::Result;
use crate::hdc::HV;
use crate::store::{decode, encode, Store};
use crate::types::{Episode, EpisodeId, Provenance, TimeRange};

/// Ring of raw frames — `frame_id → SensoryFrame`.
pub const SENSORY_FRAMES: TableDefinition<u64, Vec<u8>> = TableDefinition::new("sensory_frames");

/// Manifest key for the monotonic frame-id counter.
const KEY_NEXT_SENSORY_ID: &str = "next_sensory_id";

/// Maximum frames retained in the ring buffer.
pub const DEFAULT_SENSORY_CAPACITY: u64 = 1000;

/// Frames whose surprise is at or above this are promoted to episodic
/// memory. Random text against unrelated context scores ≈ 0.5;
/// verbatim repetition scores ≈ 0.0.
pub const SURPRISE_PROMOTION_THRESHOLD: f32 = 0.4;

/// How many recent episode gists the surprise score compares against.
const SURPRISE_REFERENCE_WINDOW: usize = 64;

/// One raw frame in the sensory ring.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SensoryFrame {
    pub id: u64,
    pub text: String,
    pub at: DateTime<Utc>,
    pub surprise: f32,
    /// Set when the frame crossed the promotion threshold.
    pub promoted: Option<EpisodeId>,
}

/// What `observe_sensory` did.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SensoryObservation {
    pub frame_id: u64,
    pub surprise: f32,
    pub promoted: Option<EpisodeId>,
}

impl Store {
    /// Surprise of `text` against the most recent episodes' gists.
    /// 1.0 on an empty store (everything is novel); 0.0 for empty or
    /// whitespace-only text (nothing to remember).
    pub fn surprise_score(&self, text: &str) -> Result<f32> {
        let gist = encode_gist_signature(text);
        if gist == HV::zero() {
            return Ok(0.0);
        }
        let mut max_sim = f32::MIN;
        let mut seen = 0usize;
        for entry in self.scan_entries().iter().rev() {
            if entry.tombstoned {
                continue;
            }
            let Ok(hv) = self.signatures.read(entry.gist_offset) else {
                continue;
            };
            max_sim = max_sim.max(gist.similarity(&hv));
            seen += 1;
            if seen >= SURPRISE_REFERENCE_WINDOW {
                break;
            }
        }
        if seen == 0 {
            return Ok(1.0);
        }
        Ok((1.0 - max_sim).clamp(0.0, 1.0))
    }

    /// Record a raw frame; promote it to an episode when its surprise
    /// crosses [`SURPRISE_PROMOTION_THRESHOLD`]. The ring keeps the
    /// last [`DEFAULT_SENSORY_CAPACITY`] frames.
    pub fn observe_sensory(&mut self, text: &str) -> Result<SensoryObservation> {
        let at = Utc::now();
        let surprise = self.surprise_score(text)?;

        let promoted = if surprise >= SURPRISE_PROMOTION_THRESHOLD {
            let id = self.next_episode_id()?;
            let gist = encode_gist_signature(text);
            let episode = Episode {
                id,
                text: text.to_string(),
                signature_offset: 0, // overwritten by observe
                gist_offset: 0,      // overwritten by observe
                triples: vec![],
                valid_time: TimeRange::point(at),
                t_tx_start: at,
                provenance: Provenance {
                    source: "sensory".into(),
                    ..Provenance::default()
                },
                confidence: 0.5,
                superseded_by: None,
            };
            Some(self.observe(episode, &gist)?)
        } else {
            None
        };

        let frame_id = self.next_sensory_id()?;
        let frame = SensoryFrame {
            id: frame_id,
            text: text.to_string(),
            at,
            surprise,
            promoted,
        };
        let tx = self.db.begin_write()?;
        {
            let mut table = tx.open_table(SENSORY_FRAMES)?;
            table.insert(frame_id, encode(&frame)?)?;
            // Monotonic ids ⇒ evicting exactly (id − capacity) keeps
            // the table at ≤ capacity rows.
            if frame_id > DEFAULT_SENSORY_CAPACITY {
                table.remove(frame_id - DEFAULT_SENSORY_CAPACITY)?;
            }
        }
        tx.commit()?;

        Ok(SensoryObservation {
            frame_id,
            surprise,
            promoted,
        })
    }

    /// Up to `limit` frames, newest first.
    pub fn sensory_frames(&self, limit: usize) -> Result<Vec<SensoryFrame>> {
        let tx = self.db.begin_read()?;
        let table = tx.open_table(SENSORY_FRAMES)?;
        let mut out = Vec::new();
        for entry in table.iter()?.rev() {
            if out.len() >= limit {
                break;
            }
            let (_, v) = entry?;
            out.push(decode(&v.value())?);
        }
        Ok(out)
    }

    fn next_sensory_id(&mut self) -> Result<u64> {
        let tx = self.db.begin_write()?;
        let id;
        {
            let mut manifest = tx.open_table(crate::store::MANIFEST)?;
            let raw = manifest.get(KEY_NEXT_SENSORY_ID)?.map(|v| v.value());
            let current: u64 = match raw {
                Some(bytes) => decode(&bytes)?,
                None => 1,
            };
            manifest.insert(KEY_NEXT_SENSORY_ID, encode(&(current + 1))?)?;
            id = current;
        }
        tx.commit()?;
        Ok(id)
    }
}
```

Wiring:
1. `crates/agidb-core/src/lib.rs`: add `pub mod sensory;` (alphabetical with the other mods).
2. `crates/agidb-core/src/store.rs`: `MANIFEST` is already `pub const` — verify; if it is private, make it `pub(crate)`. In `open()`'s table-touch block add `let _ = tx.open_table(crate::sensory::SENSORY_FRAMES)?;` with comment `// Floor 1 — sensory ring buffer.`

- [ ] **Step 4: Run sensory tests — pass**

Run: `cargo test -p agidb-core --test sensory_properties 2>&1 | tail -5`
Expected: `test result: ok. 7 passed`.
Note: `ring_buffer_evicts_beyond_capacity` runs ~2000 redb transactions; if it exceeds ~60s, mark it `#[ignore]` is NOT acceptable — instead reduce the loop to `DEFAULT_SENSORY_CAPACITY + 10` iterations exactly as written (1010) and keep it.

- [ ] **Step 5: Facade wrappers**

In `crates/agidb/src/lib.rs`, add after the self-model section (follow the existing `spawn_blocking` + `Arc<Mutex<Store>>` pattern used by e.g. `stats`):

```rust
    // --- floor 1: sensory buffer -----------------------------------------

    /// Record a raw sensory frame; promotes to episodic memory when
    /// surprising. Floor 1.
    pub async fn observe_sensory(
        &self,
        text: impl Into<String> + Send + 'static,
    ) -> CoreResult<agidb_core::sensory::SensoryObservation> {
        let store = self.store.clone();
        let text = text.into();
        tokio::task::spawn_blocking(move || {
            store.lock().expect("store mutex").observe_sensory(&text)
        })
        .await
        .map_err(join_err)?
    }

    /// Surprise score of `text` against recent memory, in [0, 1].
    pub async fn surprise_score(
        &self,
        text: impl Into<String> + Send + 'static,
    ) -> CoreResult<f32> {
        let store = self.store.clone();
        let text = text.into();
        tokio::task::spawn_blocking(move || {
            store.lock().expect("store mutex").surprise_score(&text)
        })
        .await
        .map_err(join_err)?
    }

    /// The most recent sensory frames, newest first.
    pub async fn sensory_frames(
        &self,
        limit: usize,
    ) -> CoreResult<Vec<agidb_core::sensory::SensoryFrame>> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || {
            store.lock().expect("store mutex").sensory_frames(limit)
        })
        .await
        .map_err(join_err)?
    }
```

Check how the existing facade methods map `JoinError` (there will be a helper or an inline `.map_err`; mirror it exactly — the name `join_err` above is a guess, use whatever `stats()` uses).

- [ ] **Step 6: CLI subcommands**

In `crates/agidb-cli/src/main.rs` add to `enum Command`:

```rust
    /// Record a sensory frame; promotes to an episode when surprising (floor 1).
    Sense {
        db: PathBuf,
        text: String,
    },
    /// Show recent sensory frames, newest first.
    Sensory {
        db: PathBuf,
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
```

and to the `match` in `main` (mirror the surrounding commands' `Agidb::open` usage):

```rust
        Command::Sense { db, text } => {
            let a = Agidb::open(db).await?;
            let obs = a.observe_sensory(text).await?;
            println!("{}", serde_json::to_string_pretty(&obs)?);
        }
        Command::Sensory { db, limit } => {
            let a = Agidb::open(db).await?;
            for f in a.sensory_frames(limit).await? {
                println!(
                    "[{}] surprise={:.2} promoted={} {}",
                    f.id,
                    f.surprise,
                    f.promoted.map(|e| e.raw().to_string()).unwrap_or_else(|| "-".into()),
                    f.text
                );
            }
        }
```

- [ ] **Step 7: Full test + clippy + fmt**

Run: `cargo test --workspace 2>&1 | grep -E "test result|FAILED" | sort | uniq -c` → all ok.
Run: `cargo clippy --workspace --all-targets 2>&1 | grep -c "^error"` → 0. `cargo fmt --all`.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "feat(core): floor-1 sensory buffer with surprise-gated promotion

Ring-buffered SensoryFrame table (capacity 1000), surprise =
1 - max gist similarity over the 64 most recent episodes, promotion
threshold 0.4 into text-only episodes (source=sensory). Facade
observe_sensory/surprise_score/sensory_frames + CLI sense/sensory."
```

---

### Task 3: Full MCP tool surface + server dispatch fix

**Files:**
- Modify: `crates/agidb-mcp/src/context.rs` (generic store accessors)
- Modify: `crates/agidb-mcp/src/tools.rs` (9 new tools)
- Modify: `crates/agidb-mcp/src/lib.rs` (doc list of tools if it enumerates them)
- Test: `crates/agidb-mcp/tests/dispatch.rs` (extend)
- Test: `crates/agidb-server/src/dispatch.rs` (add registry-consistency test in its `mod tests`)

**Interfaces:**
- Consumes: `AgidbContext` (Mutex<Store> + extractor), `Store::{set_goal, active_goals, assert_belief, revise_belief, what_do_i_believe, all_beliefs, unlearn, restore_within_window, what_did_i_learn, all_learning_events, stats, observe_sensory, concept_id_for, concept_id_for_ci}`, types `Goal::new/with_deadline`, `Belief::new/with_confidence/with_triple`, `UnlearnTarget`, `LearningEvent` (Serialize), `Stats` (Serialize), `SensoryObservation` (Serialize).
- Produces: MCP tools `memory_set_goal`, `memory_active_goals`, `memory_assert_belief`, `memory_revise_belief`, `memory_beliefs`, `memory_unlearn`, `memory_what_did_i_learn`, `memory_stats`, `memory_sense`. The agidb-server demo (`tool_for_phase`) resolves against these names.

- [ ] **Step 1: Add generic store accessors to `AgidbContext`**

In `crates/agidb-mcp/src/context.rs`, add to `impl AgidbContext`:

```rust
    /// Run a read-only closure against the store.
    pub fn with_store<T>(
        &self,
        f: impl FnOnce(&Store) -> AgidbResult<T>,
    ) -> AgidbResult<T> {
        let store = self.store.lock().expect("store mutex poisoned");
        f(&store)
    }

    /// Run a mutating closure against the store.
    pub fn with_store_mut<T>(
        &self,
        f: impl FnOnce(&mut Store) -> AgidbResult<T>,
    ) -> AgidbResult<T> {
        let mut store = self.store.lock().expect("store mutex poisoned");
        f(&mut store)
    }
```

- [ ] **Step 2: Write failing dispatch tests**

Read `crates/agidb-mcp/tests/dispatch.rs` first and mirror its setup helper (it uses `AgidbContext::open_null` + `tools::call`). Append:

```rust
#[test]
fn goal_belief_unlearn_stats_tools_round_trip() {
    let dir = tempfile::TempDir::new().unwrap();
    let ctx = agidb_mcp::context::AgidbContext::open_null(dir.path().to_str().unwrap()).unwrap();

    // set_goal → goal_id
    let r = agidb_mcp::tools::call(
        &ctx,
        "memory_set_goal",
        serde_json::json!({ "description": "find a thai place for the team dinner" }),
    )
    .unwrap();
    let v = r.as_json();
    let goal_id = v["goal_id"].as_u64().expect("goal_id");
    assert!(goal_id >= 1);

    // active_goals lists it
    let r = agidb_mcp::tools::call(&ctx, "memory_active_goals", serde_json::json!({})).unwrap();
    let v = r.as_json();
    assert_eq!(v["goals"].as_array().unwrap().len(), 1);

    // assert_belief → belief_id
    let r = agidb_mcp::tools::call(
        &ctx,
        "memory_assert_belief",
        serde_json::json!({ "claim": "Sarah likes thai food", "confidence": 0.8 }),
    )
    .unwrap();
    let v = r.as_json();
    let belief_id = v["belief_id"].as_u64().expect("belief_id");

    // observe an episode to use as revision evidence
    let r = agidb_mcp::tools::call(
        &ctx,
        "memory_observe",
        serde_json::json!({ "text": "Sarah ordered pad thai again" }),
    )
    .unwrap();
    let episode_id = r.as_json()["episode_id"].as_u64().unwrap();

    // revise_belief (supporting evidence) → confidence rises
    let r = agidb_mcp::tools::call(
        &ctx,
        "memory_revise_belief",
        serde_json::json!({
            "belief_id": belief_id,
            "evidence_episode_id": episode_id,
            "supports": true,
            "reason": "she ordered it again"
        }),
    )
    .unwrap();
    let v = r.as_json();
    assert!(v["new_confidence"].as_f64().unwrap() > v["previous_confidence"].as_f64().unwrap());

    // beliefs list
    let r = agidb_mcp::tools::call(&ctx, "memory_beliefs", serde_json::json!({})).unwrap();
    assert_eq!(r.as_json()["beliefs"].as_array().unwrap().len(), 1);

    // stats
    let r = agidb_mcp::tools::call(&ctx, "memory_stats", serde_json::json!({})).unwrap();
    assert_eq!(r.as_json()["episodes"].as_u64().unwrap(), 1);

    // what_did_i_learn (default window) is non-empty
    let r = agidb_mcp::tools::call(&ctx, "memory_what_did_i_learn", serde_json::json!({})).unwrap();
    assert!(!r.as_json()["events"].as_array().unwrap().is_empty());

    // unlearn the episode
    let r = agidb_mcp::tools::call(
        &ctx,
        "memory_unlearn",
        serde_json::json!({
            "target_kind": "episode",
            "target": episode_id.to_string(),
            "reason": "test forget"
        }),
    )
    .unwrap();
    assert_eq!(r.as_json()["episodes_removed"].as_u64().unwrap(), 1);
}

#[test]
fn sense_tool_promotes_novel_text() {
    let dir = tempfile::TempDir::new().unwrap();
    let ctx = agidb_mcp::context::AgidbContext::open_null(dir.path().to_str().unwrap()).unwrap();
    let r = agidb_mcp::tools::call(
        &ctx,
        "memory_sense",
        serde_json::json!({ "text": "novel sensory frame about the tokyo offsite" }),
    )
    .unwrap();
    let v = r.as_json();
    assert!(v["surprise"].as_f64().unwrap() >= 0.4);
    assert!(v["promoted"].as_u64().is_some());
}
```

**Note:** if `ToolResult` has no `as_json()` accessor, check `crates/agidb-mcp/src/protocol.rs` for how the existing dispatch tests read results (there is an existing pattern — the current tests assert on `ToolResult` content). Use that pattern; if none exists, add `impl ToolResult { pub fn as_json(&self) -> serde_json::Value { … } }` parsing the tool's JSON content — mirror how `ToolResult::json` stores it.

- [ ] **Step 3: Run to verify failure**

Run: `cargo test -p agidb-mcp 2>&1 | tail -5`
Expected: FAIL with `unknown tool: memory_set_goal`.

- [ ] **Step 4: Implement the 9 tools**

In `crates/agidb-mcp/src/tools.rs`, extend `registry()` with the entries below and add the sections. Keep the existing 4 tools first; append new ones in this order.

```rust
        Tool {
            name: "memory_set_goal",
            description: "Create a first-class goal (state machine: Active/Paused/Completed/Abandoned). Active goals bias recall.",
            schema: set_goal_schema,
            handler: set_goal,
        },
        Tool {
            name: "memory_active_goals",
            description: "List every goal currently in the Active state.",
            schema: empty_schema,
            handler: active_goals,
        },
        Tool {
            name: "memory_assert_belief",
            description: "Assert a revisable belief with graded confidence and an append-only revision log.",
            schema: assert_belief_schema,
            handler: assert_belief,
        },
        Tool {
            name: "memory_revise_belief",
            description: "Revise a belief with new supporting or contradicting episode evidence; confidence moves and the revision is logged.",
            schema: revise_belief_schema,
            handler: revise_belief,
        },
        Tool {
            name: "memory_beliefs",
            description: "List beliefs, optionally filtered by subject.",
            schema: beliefs_schema,
            handler: beliefs,
        },
        Tool {
            name: "memory_unlearn",
            description: "Non-destructive cascading unlearn (episode/belief/concept/source/session) with a permanent audit record and 30-day restore window.",
            schema: unlearn_schema,
            handler: unlearn,
        },
        Tool {
            name: "memory_what_did_i_learn",
            description: "Introspect the append-only learning log (floor 7). Defaults to the last 24 hours.",
            schema: what_did_i_learn_schema,
            handler: what_did_i_learn,
        },
        Tool {
            name: "memory_stats",
            description: "Store-wide counts: episodes, concepts, atoms, goals, beliefs, consolidation passes, signatures.",
            schema: empty_schema,
            handler: stats,
        },
        Tool {
            name: "memory_sense",
            description: "Record a raw sensory frame (floor 1). Computes a surprise score; frames above 0.4 are promoted to episodic memory.",
            schema: sense_schema,
            handler: sense,
        },
```

Handler implementations (append to the file; add `use agidb_core::types::{Belief, EpisodeId, Goal};`, `use agidb_core::unlearn::UnlearnTarget;`, `use agidb_core::types::BeliefId;` — adjust paths to wherever those types are re-exported; `BeliefId`/`GoalId` live in `agidb_core::types`):

```rust
// ---------------------------------------------------------------------------
// shared
// ---------------------------------------------------------------------------

fn empty_schema() -> Value {
    json!({ "type": "object", "properties": {} })
}

// ---------------------------------------------------------------------------
// memory_set_goal / memory_active_goals
// ---------------------------------------------------------------------------

fn set_goal_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "description": { "type": "string", "description": "What the agent wants." },
            "deadline": { "type": "string", "description": "Optional RFC3339 deadline." }
        },
        "required": ["description"]
    })
}

#[derive(Deserialize)]
struct SetGoalArgs {
    description: String,
    deadline: Option<String>,
}

fn set_goal(ctx: &AgidbContext, args: Value) -> Result<ToolResult, McpError> {
    let args: SetGoalArgs = serde_json::from_value(args)?;
    let mut goal = Goal::new(args.description);
    if let Some(d) = args.deadline {
        let parsed = chrono::DateTime::parse_from_rfc3339(&d)
            .map_err(|e| McpError::InvalidParams(format!("bad deadline: {e}")))?;
        goal = goal.with_deadline(parsed.with_timezone(&chrono::Utc));
    }
    let id = ctx.with_store_mut(|s| s.set_goal(goal))?;
    Ok(ToolResult::json(&json!({ "goal_id": id.raw() })))
}

fn active_goals(ctx: &AgidbContext, _args: Value) -> Result<ToolResult, McpError> {
    let goals = ctx.with_store(|s| s.active_goals())?;
    Ok(ToolResult::json(&json!({
        "goals": goals.iter().map(|g| json!({
            "goal_id": g.id.raw(),
            "description": g.description,
            "state": format!("{:?}", g.state.kind()),
            "created_at": g.created_at.to_rfc3339(),
            "deadline": g.deadline.map(|d| d.to_rfc3339()),
        })).collect::<Vec<_>>()
    })))
}

// ---------------------------------------------------------------------------
// memory_assert_belief / memory_revise_belief / memory_beliefs
// ---------------------------------------------------------------------------

fn assert_belief_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "claim": { "type": "string", "description": "The belief statement." },
            "confidence": { "type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.5 },
            "subject": { "type": "string", "description": "Optional triple decomposition (all three or none)." },
            "predicate": { "type": "string" },
            "object": { "type": "string" }
        },
        "required": ["claim"]
    })
}

#[derive(Deserialize)]
struct AssertBeliefArgs {
    claim: String,
    confidence: Option<f32>,
    subject: Option<String>,
    predicate: Option<String>,
    object: Option<String>,
}

fn assert_belief(ctx: &AgidbContext, args: Value) -> Result<ToolResult, McpError> {
    let args: AssertBeliefArgs = serde_json::from_value(args)?;
    let mut belief = Belief::new(args.claim);
    if let Some(c) = args.confidence {
        belief = belief.with_confidence(c);
    }
    if let (Some(s), Some(p), Some(o)) = (args.subject, args.predicate, args.object) {
        belief = belief.with_triple(s, p, o);
    }
    let id = ctx.with_store_mut(|s| s.assert_belief(belief))?;
    Ok(ToolResult::json(&json!({ "belief_id": id.raw() })))
}

fn revise_belief_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "belief_id": { "type": "integer", "minimum": 1 },
            "evidence_episode_id": { "type": "integer", "minimum": 1 },
            "supports": { "type": "boolean", "description": "true = supporting evidence, false = contradicting." },
            "reason": { "type": "string" }
        },
        "required": ["belief_id", "evidence_episode_id", "supports", "reason"]
    })
}

#[derive(Deserialize)]
struct ReviseBeliefArgs {
    belief_id: u64,
    evidence_episode_id: u64,
    supports: bool,
    reason: String,
}

fn revise_belief(ctx: &AgidbContext, args: Value) -> Result<ToolResult, McpError> {
    let args: ReviseBeliefArgs = serde_json::from_value(args)?;
    let report = ctx.with_store_mut(|s| {
        s.revise_belief(
            agidb_core::types::BeliefId::new(args.belief_id),
            EpisodeId::new(args.evidence_episode_id),
            args.supports,
            args.reason,
        )
    })?;
    Ok(ToolResult::json(&json!({
        "belief_id": report.belief_id.raw(),
        "previous_confidence": report.previous_confidence,
        "new_confidence": report.new_confidence,
        "withdrawn": report.withdrawn,
    })))
}

fn beliefs_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "about": { "type": "string", "description": "Optional subject filter (canonical concept name)." }
        }
    })
}

#[derive(Deserialize)]
struct BeliefsArgs {
    about: Option<String>,
}

fn beliefs(ctx: &AgidbContext, args: Value) -> Result<ToolResult, McpError> {
    let args: BeliefsArgs = serde_json::from_value(args)?;
    let list = match args.about {
        Some(subject) => ctx.with_store(|s| s.what_do_i_believe(&subject))?,
        None => ctx.with_store(|s| s.all_beliefs())?,
    };
    Ok(ToolResult::json(&json!({
        "beliefs": list.iter().map(|b| json!({
            "belief_id": b.id.raw(),
            "claim": b.claim,
            "confidence": b.confidence,
            "withdrawn": b.is_withdrawn(),
            "evidence_count": b.evidence.len(),
            "revision_count": b.revision_log.len(),
        })).collect::<Vec<_>>()
    })))
}

// ---------------------------------------------------------------------------
// memory_unlearn
// ---------------------------------------------------------------------------

fn unlearn_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "target_kind": {
                "type": "string",
                "enum": ["episode", "belief", "concept", "source", "session"],
                "description": "What to forget. concept cascades to everything referencing it."
            },
            "target": {
                "type": "string",
                "description": "Episode/belief id (integer as string), concept name, source label, or session id."
            },
            "reason": { "type": "string" }
        },
        "required": ["target_kind", "target", "reason"]
    })
}

#[derive(Deserialize)]
struct UnlearnArgs {
    target_kind: String,
    target: String,
    reason: String,
}

fn unlearn(ctx: &AgidbContext, args: Value) -> Result<ToolResult, McpError> {
    let args: UnlearnArgs = serde_json::from_value(args)?;
    let target = match args.target_kind.as_str() {
        "episode" => UnlearnTarget::Episode(EpisodeId::new(parse_id(&args.target)?)),
        "belief" => UnlearnTarget::Belief(agidb_core::types::BeliefId::new(parse_id(&args.target)?)),
        "concept" => {
            let cid = ctx.with_store(|s| {
                Ok(match s.concept_id_for(&args.target)? {
                    Some(c) => Some(c),
                    None => s.concept_id_for_ci(&args.target.to_lowercase())?,
                })
            })?;
            match cid {
                Some(c) => UnlearnTarget::Concept(c),
                None => {
                    return Ok(ToolResult::error(format!(
                        "unknown concept: {}",
                        args.target
                    )))
                }
            }
        }
        "source" => UnlearnTarget::BySource(args.target.clone()),
        "session" => UnlearnTarget::BySession(args.target.clone()),
        other => {
            return Err(McpError::InvalidParams(format!(
                "bad target_kind: {other}"
            )))
        }
    };
    let report = ctx.with_store_mut(|s| s.unlearn(target, args.reason))?;
    Ok(ToolResult::json(&serde_json::to_value(report)?))
}

fn parse_id(s: &str) -> Result<u64, McpError> {
    s.trim()
        .parse::<u64>()
        .map_err(|_| McpError::InvalidParams(format!("expected numeric id, got {s:?}")))
}

// ---------------------------------------------------------------------------
// memory_what_did_i_learn / memory_stats / memory_sense
// ---------------------------------------------------------------------------

fn what_did_i_learn_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "since": { "type": "string", "description": "RFC3339 timestamp; defaults to 24h ago." },
            "limit": { "type": "integer", "minimum": 1, "default": 100 }
        }
    })
}

#[derive(Deserialize)]
struct WhatDidILearnArgs {
    since: Option<String>,
    #[serde(default = "default_learn_limit")]
    limit: usize,
}

fn default_learn_limit() -> usize {
    100
}

fn what_did_i_learn(ctx: &AgidbContext, args: Value) -> Result<ToolResult, McpError> {
    let args: WhatDidILearnArgs = serde_json::from_value(args)?;
    let since = match args.since {
        Some(s) => chrono::DateTime::parse_from_rfc3339(&s)
            .map_err(|e| McpError::InvalidParams(format!("bad since: {e}")))?
            .with_timezone(&chrono::Utc),
        None => chrono::Utc::now() - chrono::Duration::hours(24),
    };
    let mut events = ctx.with_store(|s| s.what_did_i_learn(since))?;
    events.truncate(args.limit);
    Ok(ToolResult::json(&json!({
        "events": events.iter().map(|e| json!({
            "kind": e.kind_label(),
            "at": e.timestamp().to_rfc3339(),
            "detail": serde_json::to_value(e).unwrap_or(Value::Null),
        })).collect::<Vec<_>>()
    })))
}

fn stats(ctx: &AgidbContext, _args: Value) -> Result<ToolResult, McpError> {
    let s = ctx.with_store(|st| st.stats())?;
    Ok(ToolResult::json(&serde_json::to_value(s)?))
}

fn sense_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "text": { "type": "string", "description": "Raw sensory input." }
        },
        "required": ["text"]
    })
}

#[derive(Deserialize)]
struct SenseArgs {
    text: String,
}

fn sense(ctx: &AgidbContext, args: Value) -> Result<ToolResult, McpError> {
    let args: SenseArgs = serde_json::from_value(args)?;
    let obs = ctx.with_store_mut(|s| s.observe_sensory(&args.text))?;
    Ok(ToolResult::json(&json!({
        "frame_id": obs.frame_id,
        "surprise": obs.surprise,
        "promoted": obs.promoted.map(|e| e.raw()),
    })))
}
```

Type-path fixes to expect: `UnlearnReport` derives `Serialize` (verified) so `to_value(report)` works. `LearningEvent` derives `Serialize` (verified). `Stats` derives `Serialize` (verified). `with_store` closures return `AgidbResult<T>` so wrap plain values in `Ok(...)` where needed.

- [ ] **Step 5: Run MCP tests — pass**

Run: `cargo test -p agidb-mcp 2>&1 | tail -5`
Expected: all ok including the 2 new tests.

- [ ] **Step 6: Registry-consistency test in agidb-server**

In `crates/agidb-server/src/dispatch.rs` `mod tests`, add:

```rust
    #[test]
    fn every_demo_phase_maps_to_a_registered_tool() {
        let registered: Vec<String> = agidb_mcp::tools::list()
            .into_iter()
            .map(|t| t.name)
            .collect();
        for phase in [
            "observe", "recall", "consolidate", "set_goal",
            "assert_belief", "revise_belief", "stats", "unlearn",
        ] {
            let tool = tool_for_phase(phase);
            assert!(
                registered.iter().any(|n| n == tool),
                "phase {phase} maps to unregistered tool {tool}"
            );
        }
    }
```

Run: `cargo test -p agidb-server 2>&1 | tail -5` → ok (this test would have failed before Step 4; it now locks the demo UI to real tools).

- [ ] **Step 7: Update the MCP crate docs**

`crates/agidb-mcp/src/lib.rs` has a doc list of the 4 tools — extend it to all 13, one line each (same phrasing as the tool descriptions).

- [ ] **Step 8: Full test + clippy + fmt + commit**

```bash
cargo test --workspace 2>&1 | grep -E "test result|FAILED" | sort | uniq -c
cargo clippy --workspace --all-targets 2>&1 | grep -c "^error"   # expect 0
cargo fmt --all
git add -A
git commit -m "feat(mcp): expose goals, beliefs, unlearn, learning log, stats, sensory over MCP

9 new tools (memory_set_goal, memory_active_goals, memory_assert_belief,
memory_revise_belief, memory_beliefs, memory_unlearn,
memory_what_did_i_learn, memory_stats, memory_sense) with JSON-Schema
inputs + dispatch tests; agidb-server demo phases now resolve to real
registered tools (regression-tested)."
```

---

### Task 4: 100-sample extraction gold set with an honest F1 gate

**Files:**
- Modify: `crates/agidb-extract/eval/gold/observations.jsonl` (replace the 3 placeholder rows)
- Modify: `crates/agidb-extract/eval/src/main.rs` (make the gate a CLI arg)

**Interfaces:**
- Consumes: `GoldRow { text, triples: [GoldTriple { subject, predicate, object }] }` (serde ignores extra fields; a `notes` field per row is allowed and encouraged).
- Produces: `eval/gold/observations.jsonl` with exactly 100 rows; eval binary flag `--gate <f32>` (default 0.85) controlling the exit-code threshold.

**Authoring rules (follow exactly):**
1. 100 rows total. Distribution: 30 `recommends`, 20 `likes`, 15 `works_at`, 15 `located_in`, 10 rows with **two** triples in one sentence, 10 rows with **zero** triples (small talk, questions, bare exclamations — `"triples": []`).
2. Surface forms must be drawn from `crates/agidb-extract/src/predicates.rs` `Default` table **for 70 rows** (so the heuristic can canonicalize) and be out-of-vocabulary paraphrases for the rest (e.g. "raved about", "can't stop talking about") — these measure honest misses.
3. Subjects/objects are proper-noun-ish entities (people, places, companies, dishes) with varied casing and 10 rows containing typos or aliases ("Sarrah", "Bombay" vs "Mumbai").
4. Every gold predicate must be the **canonical** form (`recommends`, `likes`, `works_at`, `located_in`) — check the full canonical list in `predicates.rs` `Default::default()` before authoring, and use only canonicals that exist there.
5. ~15 rows include a relative time phrase ("last weekend", "two days ago") — the extractor's temporal parser is separate from triple scoring, but natural text should include them.
6. Each row gets a `notes` field stating its category, e.g. `"notes": "in-vocab recommends"` — scoring ignores it.

Example rows (write these styles, not these exact rows, 100 times):

```jsonl
{"text": "Sarah recommended Bawri in Bandra last weekend", "triples": [{"subject": "Sarah", "predicate": "recommends", "object": "Bawri"}], "notes": "in-vocab recommends + temporal"}
{"text": "Marco works at Stripe now", "triples": [{"subject": "Marco", "predicate": "works_at", "object": "Stripe"}], "notes": "in-vocab works_at"}
{"text": "Priya loves the masala dosa at Rameshwaram Cafe", "triples": [{"subject": "Priya", "predicate": "likes", "object": "masala dosa"}], "notes": "in-vocab likes; object is a dish"}
{"text": "Ankit is based in Berlin and works for Zalando", "triples": [{"subject": "Ankit", "predicate": "located_in", "object": "Berlin"}, {"subject": "Ankit", "predicate": "works_at", "object": "Zalando"}], "notes": "two triples one sentence"}
{"text": "Dev raved about Trishna to everyone at the offsite", "triples": [{"subject": "Dev", "predicate": "recommends", "object": "Trishna"}], "notes": "OOV paraphrase — expected heuristic miss"}
{"text": "what time is the standup tomorrow?", "triples": [], "notes": "zero-triple question"}
```

- [ ] **Step 1: Make the F1 gate a CLI argument**

In `crates/agidb-extract/eval/src/main.rs`: add to the clap `Cli` struct:

```rust
    /// F1 exit gate for CI. The build fails (exit 1) below this.
    #[arg(long, default_value_t = 0.85)]
    gate: f64,
```

and change the check at the bottom from the hardcoded `0.85` to:

```rust
    if !cli.dry_run && f1 < cli.gate {
        eprintln!("F1 {:.3} below the gate ({:.3})", f1, cli.gate);
        std::process::exit(1);
    }
```

- [ ] **Step 2: Author the 100 rows**

Replace `crates/agidb-extract/eval/gold/observations.jsonl` wholesale following the authoring rules. Validate JSON:

Run: `python3 -c "import json,sys; rows=[json.loads(l) for l in open('crates/agidb-extract/eval/gold/observations.jsonl') if l.strip()]; print(len(rows))"` → `100`
(If python3 is unavailable: `cargo run -p agidb-extract-eval -- --dry-run` parses every row.)

- [ ] **Step 3: Run the eval in dry-run (no model) to validate plumbing**

Run: `cargo run -p agidb-extract-eval -- --dry-run 2>&1 | tail -3`
Expected: `P=0.000 R=0.000 F1=0.000 (n=100, dry_run=true)` and exit 0. (Check the actual eval package name in `crates/agidb-extract/eval/Cargo.toml` — it may be `agidb-extract-eval` or similar; use that.)

- [ ] **Step 4: Run the real eval and record the honest number**

The real run needs the GLiNER model (~downloads on first use):
Run: `cargo run -p agidb-extract-eval -- --gate 0.0 2>&1 | tail -5`
Record `P/R/F1` verbatim. **Do not edit gold rows to raise the score.** Whatever the number is, it goes in `bench/RESULTS.md` (Task 5 creates the file — if running Task 4 first, create `bench/RESULTS.md` with just this section) under the heading `## Extraction F1 (100-sample gold set)`, with the eval command line and date.

If the model can't download in this environment, run dry-run only and record: "real-model F1 not yet measured — heuristic extractor, gold set authored, CI gate parameterized". Do NOT claim a number.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "test(extract): 100-sample gold set + parameterized F1 gate

70 in-vocabulary rows across recommends/likes/works_at/located_in,
20 OOV-paraphrase rows (honest misses), 10 zero-triple rows; measured
F1 recorded in bench/RESULTS.md instead of a hardcoded aspiration."
```

---

### Task 5: Real benchmark harness — agidb vs SQLite FTS5 vs naive scan

**Files:**
- Modify: `crates/agidb-bench/Cargo.toml`
- Replace: `crates/agidb-bench/src/main.rs`
- Create: `crates/agidb-bench/src/corpus.rs`
- Create: `crates/agidb-bench/src/systems.rs`
- Create: `crates/agidb-bench/src/metrics.rs`
- Create: `bench/RESULTS.md` (generated + committed)
- Create: `bench/results-10k.json` (generated + committed)

**Interfaces:**
- Consumes: `Store::{open, observe, supersede, recall}`, `encode_episode_signature`, `Query::{cue, with_k, with_as_of}`, `Tier`.
- Produces: `cargo run -p agidb-bench --release -- --episodes 10000 --queries 200 --out bench/` writes `bench/results-10k.json` + regenerates the measured tables in `bench/RESULTS.md`.

**Design (all deterministic, seed fixed at 42):**
- **Corpus:** N episodes rendered from `"{person} {surface_verb} {place} {date_phrase}"` over pools of 40 people / 40 places / 4 canonical predicates with 2–3 surface verbs each; valid times spread deterministically across 2026. Plus N/100 supersession pairs (same person+place, verb changes at a later date; older superseded).
- **Queries (4 classes, equal split):**
  - `exact`: "what did {person} think of {place}" — relevant = episodes containing both.
  - `single-entity`: "did {person} recommend anything" — relevant = all episodes with that person.
  - `noisy`: exact-class cue with 1 character dropped from the person and 1 swapped in the place — relevant unchanged. (This is the noisy-cue metric from constitution Art. X.)
  - `temporal`: for a supersession pair, cue = "what did {person} think of {place}" with `as_of` before the supersession → relevant = the OLD episode only. SQL baselines get the same filter via date columns (fair).
- **Systems:**
  - `agidb`: `Store::observe` with structured signatures (mirrors `observe_text` minus the ONNX model — deterministic, offline).
  - `fts5`: rusqlite, `CREATE VIRTUAL TABLE ... USING fts5`, BM25 ranking, token-OR match, date columns for temporal.
  - `scan`: SQLite table full-scan scored by matched-token count in Rust (the "no index" floor).
- **Metrics per system:** ingest wall-time + throughput, on-disk bytes, and per query-class: hit@1, hit@5, MRR@10, p50/p95 query latency. Plus overall.
- **Honesty requirements (constitution Art. X):** the JSON + RESULTS.md must state: corpus is synthetic + templated (favors lexical systems; no paraphrase semantics), which of the constitution's six metrics are covered (F1-style hit metrics ✅, p95 ✅, noisy-cue ✅; BLEU / LLM-judge / token-cost pending — they need external corpora/APIs), and any class agidb loses. Never delete a losing row.

- [ ] **Step 1: Cargo.toml**

Replace the `[dependencies]` of `crates/agidb-bench/Cargo.toml` with:

```toml
[dependencies]
agidb-core = { workspace = true }
anyhow = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
chrono = { workspace = true }
clap = { version = "4", features = ["derive"] }
rusqlite = { version = "0.32", features = ["bundled"] }
```

(Check the root `Cargo.toml` `[workspace.dependencies]` — if `clap` and `chrono` are listed there, use `{ workspace = true }` for them. Remove `agidb-extract`, `tokio`, `tracing*` — the harness is synchronous and model-free. Update the `description` field to: `"Deterministic retrieval benchmark: agidb vs SQLite FTS5 (BM25) vs naive scan — hit@k, MRR, latency percentiles, ingest throughput, on-disk size."`)

- [ ] **Step 2: corpus.rs**

```rust
//! Deterministic synthetic corpus + query set. Seeded splitmix64 —
//! same seed, same corpus, byte for byte, forever.

use chrono::{DateTime, Duration, TimeZone, Utc};

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn next(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    pub fn pick<'a, T>(&mut self, s: &'a [T]) -> &'a T {
        &s[self.below(s.len())]
    }
}

pub const PEOPLE: [&str; 40] = [
    "Sarah", "Marco", "Priya", "Ankit", "Dev", "Alice", "Bob", "Carol", "Dan", "Eve",
    "Farid", "Grace", "Hana", "Ivan", "Julia", "Kenji", "Lena", "Miguel", "Nadia", "Omar",
    "Pilar", "Quinn", "Ravi", "Sofia", "Tarun", "Uma", "Viktor", "Wei", "Ximena", "Yuki",
    "Zainab", "Arjun", "Bianca", "Chetan", "Daria", "Emil", "Fatima", "Gustav", "Helga", "Iris",
];

pub const PLACES: [&str; 40] = [
    "Bawri", "Trishna", "Olive", "Pali", "Mahesh", "Soam", "Britannia", "Gajalee", "Dakshin", "Yauatcha",
    "Masque", "Bombay Canteen", "Kissa", "Subko", "Blue Tokai", "Araku", "Naru", "Izumi", "Gymkhana", "Dishoom",
    "Hoppers", "Brat", "Noma", "Ikoyi", "Septime", "Attica", "Quintonil", "Maido", "Narisawa", "Odette",
    "Alchemist", "Hisa", "Franceschetta", "Etxebarri", "Diverxo", "Steirereck", "Frantzen", "Disfrutar", "Trivet", "Ariana",
];

/// (canonical predicate, surface verbs)
pub const PREDICATES: [(&str, &[&str]); 4] = [
    ("recommends", &["recommended", "suggested", "pitched"]),
    ("likes", &["likes", "loves", "enjoys"]),
    ("works_at", &["works at", "is employed by"]),
    ("located_in", &["is located in", "is based in"]),
];

#[derive(Clone, Debug)]
pub struct Doc {
    pub id: u64,
    pub text: String,
    pub person: String,
    pub place: String,
    pub predicate: &'static str,
    pub valid_start: DateTime<Utc>,
    /// Set on the older half of a supersession pair.
    pub superseded_by: Option<u64>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum QueryClass {
    Exact,
    SingleEntity,
    Noisy,
    Temporal,
}

#[derive(Clone, Debug)]
pub struct BenchQuery {
    pub class: QueryClass,
    pub cue: String,
    pub as_of: Option<DateTime<Utc>>,
    /// Any of these ids in the top-k counts as a hit.
    pub relevant: Vec<u64>,
}

pub fn epoch() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap()
}

pub fn build_corpus(n: usize, rng: &mut Rng) -> Vec<Doc> {
    let mut docs = Vec::with_capacity(n);
    for id in 1..=(n as u64) {
        let person = rng.pick(&PEOPLE).to_string();
        let place = rng.pick(&PLACES).to_string();
        let (canonical, surfaces) = rng.pick(&PREDICATES);
        let surface = rng.pick(surfaces);
        let day = rng.below(300) as i64;
        let valid_start = epoch() + Duration::days(day);
        docs.push(Doc {
            id,
            text: format!("{person} {surface} {place} on day {day}"),
            person,
            place,
            predicate: canonical,
            valid_start,
            superseded_by: None,
        });
    }
    // Supersession pairs: for every 100th doc, append a newer doc that
    // supersedes it (same person+place, different verb, +30 days).
    let n_pairs = n / 100;
    for k in 0..n_pairs {
        let old_idx = k * 100; // deterministic
        let old = docs[old_idx].clone();
        let new_id = (docs.len() + 1) as u64;
        let (canonical, surfaces) = PREDICATES[(k + 1) % PREDICATES.len()];
        let surface = surfaces[0];
        let valid_start = old.valid_start + Duration::days(30);
        docs.push(Doc {
            id: new_id,
            text: format!("{} {surface} {} after reconsidering", old.person, old.place),
            person: old.person.clone(),
            place: old.place.clone(),
            predicate: canonical,
            valid_start,
            superseded_by: None,
        });
        docs[old_idx].superseded_by = Some(new_id);
    }
    docs
}

fn typo_drop(s: &str, rng: &mut Rng) -> String {
    if s.len() < 3 {
        return s.to_string();
    }
    let pos = 1 + rng.below(s.len() - 2);
    let mut out = String::with_capacity(s.len());
    for (i, c) in s.chars().enumerate() {
        if i != pos {
            out.push(c);
        }
    }
    out
}

pub fn build_queries(docs: &[Doc], per_class: usize, rng: &mut Rng) -> Vec<BenchQuery> {
    let mut queries = Vec::new();
    let plain: Vec<&Doc> = docs.iter().filter(|d| d.superseded_by.is_none()).collect();

    let relevant_both = |person: &str, place: &str| -> Vec<u64> {
        docs.iter()
            .filter(|d| d.person == person && d.place == place)
            .map(|d| d.id)
            .collect()
    };

    for _ in 0..per_class {
        let d = *rng.pick(&plain);
        queries.push(BenchQuery {
            class: QueryClass::Exact,
            cue: format!("what did {} think of {}", d.person, d.place),
            as_of: None,
            relevant: relevant_both(&d.person, &d.place),
        });
    }
    for _ in 0..per_class {
        let d = *rng.pick(&plain);
        queries.push(BenchQuery {
            class: QueryClass::SingleEntity,
            cue: format!("did {} recommend anything", d.person),
            as_of: None,
            relevant: docs.iter().filter(|x| x.person == d.person).map(|x| x.id).collect(),
        });
    }
    for _ in 0..per_class {
        let d = *rng.pick(&plain);
        let person = typo_drop(&d.person, rng);
        let place = typo_drop(&d.place, rng);
        queries.push(BenchQuery {
            class: QueryClass::Noisy,
            cue: format!("what did {person} think of {place}"),
            as_of: None,
            relevant: relevant_both(&d.person, &d.place),
        });
    }
    // Temporal: query each supersession pair before the new fact.
    let pairs: Vec<&Doc> = docs.iter().filter(|d| d.superseded_by.is_some()).collect();
    for i in 0..per_class {
        let old = pairs[i % pairs.len()];
        queries.push(BenchQuery {
            class: QueryClass::Temporal,
            cue: format!("what did {} think of {}", old.person, old.place),
            as_of: Some(old.valid_start + Duration::days(1)),
            relevant: vec![old.id],
        });
    }
    queries
}
```

- [ ] **Step 3: systems.rs**

```rust
//! The three systems under test. Each ingests the same corpus and
//! answers each query with a ranked id list (top 10).

use std::path::Path;
use std::time::Instant;

use agidb_core::episode::encode_episode_signature;
use agidb_core::store::{Store, StoreConfig};
use agidb_core::types::{Episode, EpisodeId, Provenance, Query, TimeRange, Triple};
use anyhow::Result;
use rusqlite::Connection;

use crate::corpus::{BenchQuery, Doc};

pub const K: usize = 10;

pub trait System {
    fn name(&self) -> &'static str;
    fn ingest(&mut self, docs: &[Doc]) -> Result<()>;
    fn query(&mut self, q: &BenchQuery) -> Result<Vec<u64>>;
    fn disk_bytes(&self) -> u64;
}

// --- agidb -----------------------------------------------------------------

pub struct AgidbSystem {
    store: Store,
    root: std::path::PathBuf,
}

impl AgidbSystem {
    pub fn open(root: &Path) -> Result<Self> {
        Ok(Self {
            store: Store::open(StoreConfig::at(root))?,
            root: root.to_path_buf(),
        })
    }
}

impl System for AgidbSystem {
    fn name(&self) -> &'static str {
        "agidb"
    }

    fn ingest(&mut self, docs: &[Doc]) -> Result<()> {
        for d in docs {
            let ep_id = EpisodeId::new(d.id);
            let triples = vec![Triple {
                subject: d.person.clone(),
                predicate: d.predicate.to_string(),
                object: d.place.clone(),
                confidence: 0.9,
                episode_id: ep_id,
            }];
            let sig = encode_episode_signature(&triples, Some(d.valid_start));
            let ep = Episode {
                id: ep_id,
                text: d.text.clone(),
                signature_offset: 0,
                gist_offset: 0,
                triples,
                valid_time: TimeRange::point(d.valid_start),
                t_tx_start: d.valid_start,
                provenance: Provenance {
                    source: "bench".into(),
                    ..Provenance::default()
                },
                confidence: 0.9,
                superseded_by: None,
            };
            self.store.observe(ep, &sig)?;
        }
        // Apply supersessions after all rows exist.
        for d in docs {
            if let Some(newer) = d.superseded_by {
                self.store
                    .supersede(EpisodeId::new(d.id), EpisodeId::new(newer))?;
            }
        }
        Ok(())
    }

    fn query(&mut self, q: &BenchQuery) -> Result<Vec<u64>> {
        let mut query = Query::cue(q.cue.clone()).with_k(K);
        if let Some(t) = q.as_of {
            query = query.with_as_of(t);
        }
        let r = self.store.recall(&query)?;
        Ok(r.matches.iter().map(|m| m.episode_id.raw()).collect())
    }

    fn disk_bytes(&self) -> u64 {
        dir_bytes(&self.root)
    }
}

// --- SQLite FTS5 (BM25) ------------------------------------------------------

pub struct Fts5System {
    conn: Connection,
    path: std::path::PathBuf,
}

impl Fts5System {
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch(
            "CREATE TABLE docs (
                 id INTEGER PRIMARY KEY,
                 text TEXT NOT NULL,
                 valid_start INTEGER NOT NULL,
                 valid_end INTEGER
             );
             CREATE VIRTUAL TABLE fts USING fts5(text, content='docs', content_rowid='id');",
        )?;
        Ok(Self {
            conn,
            path: path.to_path_buf(),
        })
    }
}

impl System for Fts5System {
    fn name(&self) -> &'static str {
        "sqlite-fts5"
    }

    fn ingest(&mut self, docs: &[Doc]) -> Result<()> {
        let tx = self.conn.transaction()?;
        for d in docs {
            tx.execute(
                "INSERT INTO docs (id, text, valid_start, valid_end) VALUES (?1, ?2, ?3, NULL)",
                rusqlite::params![d.id as i64, d.text, d.valid_start.timestamp()],
            )?;
            tx.execute(
                "INSERT INTO fts (rowid, text) VALUES (?1, ?2)",
                rusqlite::params![d.id as i64, d.text],
            )?;
        }
        // Supersession = close the older row's valid_end (same rule agidb uses).
        for d in docs {
            if let Some(newer) = d.superseded_by {
                tx.execute(
                    "UPDATE docs SET valid_end =
                       (SELECT valid_start - 1 FROM docs WHERE id = ?1)
                     WHERE id = ?2",
                    rusqlite::params![newer as i64, d.id as i64],
                )?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    fn query(&mut self, q: &BenchQuery) -> Result<Vec<u64>> {
        // Token-OR match, BM25 rank. Tokens are quoted to disarm FTS
        // operators.
        let tokens: Vec<String> = q
            .cue
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
            .map(|t| format!("\"{}\"", t.replace('"', "")))
            .collect();
        if tokens.is_empty() {
            return Ok(vec![]);
        }
        let match_expr = tokens.join(" OR ");
        let (sql, rows): (String, Vec<u64>) = if let Some(t) = q.as_of {
            let sql = "SELECT f.rowid FROM fts f JOIN docs d ON d.id = f.rowid
                       WHERE fts MATCH ?1
                         AND d.valid_start <= ?2
                         AND (d.valid_end IS NULL OR d.valid_end >= ?2)
                       ORDER BY bm25(fts) LIMIT ?3";
            let mut stmt = self.conn.prepare_cached(sql)?;
            let ids = stmt
                .query_map(
                    rusqlite::params![match_expr, t.timestamp(), K as i64],
                    |r| r.get::<_, i64>(0),
                )?
                .collect::<std::result::Result<Vec<i64>, _>>()?;
            (sql.into(), ids.into_iter().map(|i| i as u64).collect())
        } else {
            let sql = "SELECT rowid FROM fts WHERE fts MATCH ?1 ORDER BY bm25(fts) LIMIT ?2";
            let mut stmt = self.conn.prepare_cached(sql)?;
            let ids = stmt
                .query_map(rusqlite::params![match_expr, K as i64], |r| {
                    r.get::<_, i64>(0)
                })?
                .collect::<std::result::Result<Vec<i64>, _>>()?;
            (sql.into(), ids.into_iter().map(|i| i as u64).collect())
        };
        let _ = sql;
        Ok(rows)
    }

    fn disk_bytes(&self) -> u64 {
        std::fs::metadata(&self.path).map(|m| m.len()).unwrap_or(0)
    }
}

// --- naive scan (the no-index floor) -----------------------------------------

pub struct ScanSystem {
    rows: Vec<(u64, String, i64, Option<i64>)>,
}

impl ScanSystem {
    pub fn new() -> Self {
        Self { rows: Vec::new() }
    }
}

impl System for ScanSystem {
    fn name(&self) -> &'static str {
        "naive-scan"
    }

    fn ingest(&mut self, docs: &[Doc]) -> Result<()> {
        for d in docs {
            self.rows
                .push((d.id, d.text.to_lowercase(), d.valid_start.timestamp(), None));
        }
        let ends: Vec<(u64, i64)> = docs
            .iter()
            .filter_map(|d| {
                d.superseded_by.map(|newer| {
                    let end = docs
                        .iter()
                        .find(|x| x.id == newer)
                        .map(|x| x.valid_start.timestamp() - 1)
                        .unwrap_or(i64::MAX);
                    (d.id, end)
                })
            })
            .collect();
        for (id, end) in ends {
            if let Some(row) = self.rows.iter_mut().find(|r| r.0 == id) {
                row.3 = Some(end);
            }
        }
        Ok(())
    }

    fn query(&mut self, q: &BenchQuery) -> Result<Vec<u64>> {
        let tokens: Vec<String> = q
            .cue
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
            .map(String::from)
            .collect();
        let mut scored: Vec<(usize, u64)> = self
            .rows
            .iter()
            .filter(|(_, _, start, end)| match q.as_of {
                Some(t) => {
                    let ts = t.timestamp();
                    *start <= ts && end.map(|e| e >= ts).unwrap_or(true)
                }
                None => true,
            })
            .map(|(id, text, _, _)| {
                let score = tokens.iter().filter(|t| text.contains(*t)).count();
                (score, *id)
            })
            .filter(|(s, _)| *s > 0)
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        Ok(scored.into_iter().take(K).map(|(_, id)| id).collect())
    }

    fn disk_bytes(&self) -> u64 {
        self.rows
            .iter()
            .map(|(_, t, _, _)| t.len() as u64 + 24)
            .sum()
    }
}

// --- shared -------------------------------------------------------------------

pub fn dir_bytes(root: &Path) -> u64 {
    fn walk(p: &Path, acc: &mut u64) {
        if let Ok(entries) = std::fs::read_dir(p) {
            for e in entries.flatten() {
                let path = e.path();
                if path.is_dir() {
                    walk(&path, acc);
                } else if let Ok(m) = e.metadata() {
                    *acc += m.len();
                }
            }
        }
    }
    let mut acc = 0;
    walk(root, &mut acc);
    acc
}

pub fn time<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let t0 = Instant::now();
    let out = f();
    (out, t0.elapsed().as_secs_f64() * 1000.0)
}
```

- [ ] **Step 4: metrics.rs**

```rust
//! Retrieval metrics + latency percentiles.

use serde::Serialize;

use crate::corpus::{BenchQuery, QueryClass};

#[derive(Serialize, Default, Clone)]
pub struct ClassMetrics {
    pub queries: usize,
    pub hit_at_1: f64,
    pub hit_at_5: f64,
    pub mrr_at_10: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
}

#[derive(Serialize)]
pub struct SystemReport {
    pub system: String,
    pub episodes: usize,
    pub ingest_ms: f64,
    pub ingest_per_sec: f64,
    pub disk_bytes: u64,
    pub overall: ClassMetrics,
    pub exact: ClassMetrics,
    pub single_entity: ClassMetrics,
    pub noisy: ClassMetrics,
    pub temporal: ClassMetrics,
}

pub struct Sample {
    pub class: QueryClass,
    pub rank_of_first_relevant: Option<usize>, // 1-based
    pub latency_ms: f64,
}

pub fn score(q: &BenchQuery, ranked: &[u64], latency_ms: f64) -> Sample {
    let rank = ranked
        .iter()
        .position(|id| q.relevant.contains(id))
        .map(|p| p + 1);
    Sample {
        class: q.class.clone(),
        rank_of_first_relevant: rank,
        latency_ms,
    }
}

pub fn aggregate(samples: &[&Sample]) -> ClassMetrics {
    if samples.is_empty() {
        return ClassMetrics::default();
    }
    let n = samples.len() as f64;
    let hit1 = samples
        .iter()
        .filter(|s| s.rank_of_first_relevant == Some(1))
        .count() as f64
        / n;
    let hit5 = samples
        .iter()
        .filter(|s| s.rank_of_first_relevant.map(|r| r <= 5).unwrap_or(false))
        .count() as f64
        / n;
    let mrr = samples
        .iter()
        .map(|s| s.rank_of_first_relevant.map(|r| 1.0 / r as f64).unwrap_or(0.0))
        .sum::<f64>()
        / n;
    let mut lat: Vec<f64> = samples.iter().map(|s| s.latency_ms).collect();
    lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f64| lat[((lat.len() as f64 - 1.0) * p) as usize];
    ClassMetrics {
        queries: samples.len(),
        hit_at_1: hit1,
        hit_at_5: hit5,
        mrr_at_10: mrr,
        p50_ms: pct(0.50),
        p95_ms: pct(0.95),
    }
}
```

- [ ] **Step 5: main.rs**

Replace `crates/agidb-bench/src/main.rs`:

```rust
//! Deterministic retrieval benchmark: agidb vs SQLite FTS5 (BM25) vs a
//! naive full scan, over a seeded synthetic corpus.
//!
//! Honesty notes (constitution article X):
//! - The corpus is synthetic and templated. It measures lexical +
//!   structural retrieval, temporal filtering, and typo robustness. It
//!   does NOT measure semantic paraphrase (no system here embeds
//!   meaning) — do not quote these numbers as semantic recall quality.
//! - Covered metrics: hit@k / MRR (F1-style), p50/p95 latency,
//!   noisy-cue degradation, ingest throughput, disk footprint.
//!   Not covered (need external corpora / LLM APIs): BLEU, LLM-judge,
//!   token cost. RESULTS.md must say so.

mod corpus;
mod metrics;
mod systems;

use clap::Parser;
use metrics::{aggregate, score, Sample, SystemReport};
use systems::{time, AgidbSystem, Fts5System, ScanSystem, System};

#[derive(Parser)]
struct Cli {
    #[arg(long, default_value_t = 10_000)]
    episodes: usize,
    /// Total queries (split equally across 4 classes).
    #[arg(long, default_value_t = 200)]
    queries: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Output directory for results JSON + RESULTS.md.
    #[arg(long, default_value = "bench")]
    out: std::path::PathBuf,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let mut rng = corpus::Rng::new(cli.seed);
    let docs = corpus::build_corpus(cli.episodes, &mut rng);
    let queries = corpus::build_queries(&docs, cli.queries / 4, &mut rng);
    eprintln!(
        "corpus: {} docs ({} supersession pairs), {} queries",
        docs.len(),
        docs.iter().filter(|d| d.superseded_by.is_some()).count(),
        queries.len()
    );

    std::fs::create_dir_all(&cli.out)?;
    let work = tempfile_dir()?;

    let mut reports = Vec::new();
    // agidb
    {
        let mut sys = AgidbSystem::open(&work.join("agidb"))?;
        reports.push(run(&mut sys, &docs, &queries)?);
    }
    // sqlite fts5
    {
        let mut sys = Fts5System::open(&work.join("fts5.db"))?;
        reports.push(run(&mut sys, &docs, &queries)?);
    }
    // naive scan
    {
        let mut sys = ScanSystem::new();
        reports.push(run(&mut sys, &docs, &queries)?);
    }

    let json = serde_json::to_string_pretty(&reports)?;
    let json_path = cli
        .out
        .join(format!("results-{}k.json", cli.episodes / 1000));
    std::fs::write(&json_path, &json)?;
    eprintln!("wrote {}", json_path.display());

    print_markdown(&reports);
    Ok(())
}

fn run(
    sys: &mut dyn System,
    docs: &[corpus::Doc],
    queries: &[corpus::BenchQuery],
) -> anyhow::Result<SystemReport> {
    eprintln!("== {} ==", sys.name());
    let (ingest_result, ingest_ms) = time(|| sys.ingest(docs));
    ingest_result?;
    let mut samples: Vec<Sample> = Vec::with_capacity(queries.len());
    for q in queries {
        let (ranked, ms) = time(|| sys.query(q));
        samples.push(score(q, &ranked?, ms));
    }
    let by_class = |c: corpus::QueryClass| {
        let filtered: Vec<&Sample> = samples.iter().filter(|s| s.class == c).collect();
        aggregate(&filtered)
    };
    let all: Vec<&Sample> = samples.iter().collect();
    Ok(SystemReport {
        system: sys.name().to_string(),
        episodes: docs.len(),
        ingest_ms,
        ingest_per_sec: docs.len() as f64 / (ingest_ms / 1000.0),
        disk_bytes: sys.disk_bytes(),
        overall: aggregate(&all),
        exact: by_class(corpus::QueryClass::Exact),
        single_entity: by_class(corpus::QueryClass::SingleEntity),
        noisy: by_class(corpus::QueryClass::Noisy),
        temporal: by_class(corpus::QueryClass::Temporal),
    })
}

fn print_markdown(reports: &[SystemReport]) {
    println!("| system | ingest/s | disk MB | hit@1 | hit@5 | MRR | p50 ms | p95 ms |");
    println!("|---|---|---|---|---|---|---|---|");
    for r in reports {
        println!(
            "| {} | {:.0} | {:.1} | {:.3} | {:.3} | {:.3} | {:.2} | {:.2} |",
            r.system,
            r.ingest_per_sec,
            r.disk_bytes as f64 / 1e6,
            r.overall.hit_at_1,
            r.overall.hit_at_5,
            r.overall.mrr_at_10,
            r.overall.p50_ms,
            r.overall.p95_ms
        );
    }
    for (label, pick) in [
        ("exact", 0usize),
        ("single-entity", 1),
        ("noisy", 2),
        ("temporal", 3),
    ] {
        println!("\n### {label}");
        println!("| system | hit@1 | hit@5 | MRR | p95 ms |");
        println!("|---|---|---|---|---|");
        for r in reports {
            let m = match pick {
                0 => &r.exact,
                1 => &r.single_entity,
                2 => &r.noisy,
                _ => &r.temporal,
            };
            println!(
                "| {} | {:.3} | {:.3} | {:.3} | {:.2} |",
                r.system, m.hit_at_1, m.hit_at_5, m.mrr_at_10, m.p95_ms
            );
        }
    }
}

fn tempfile_dir() -> anyhow::Result<std::path::PathBuf> {
    let dir = std::env::temp_dir().join(format!("agidb-bench-{}", std::process::id()));
    if dir.exists() {
        std::fs::remove_dir_all(&dir)?;
    }
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}
```

Also remove the `[[bin]] name = "bench-all"` block or rename it to `agidb-bench` (keep the path). Adjust `main.rs` module decls if the compiler wants `crate::corpus` paths (they are `mod` in main, so `corpus::` works as written).

- [ ] **Step 6: Compile + run small**

Run: `cargo run -p agidb-bench --release -- --episodes 1000 --queries 100 2>&1 | tail -25`
Expected: the markdown tables print; agidb should win or tie `noisy` and `temporal`; FTS5 will likely win raw `p50` at this scale. **Whatever the numbers are, they are the numbers.**

- [ ] **Step 7: Full run + write RESULTS.md**

Run: `cargo run -p agidb-bench --release -- --episodes 10000 --queries 200 --out bench 2>&1 | tee /tmp/bench-10k.txt`

Create `bench/RESULTS.md` containing: date, machine (`uname -a`, CPU model from `lscpu | grep "Model name"`), commit hash, the exact command, the pasted markdown tables, the extraction F1 section from Task 4, and a **Limitations** section verbatim including: synthetic/templated corpus; lexical-structural only (no paraphrase semantics measured); BLEU / LLM-judge / token-cost metrics not yet covered; single-threaded; and an explicit sentence naming every class where agidb lost.

- [ ] **Step 8: Criterion recall bench in core (supports the latency claim)**

Create `crates/agidb-core/benches/recall_scan.rs`:

```rust
//! End-to-end recall latency over a populated store — the number the
//! "<50ms p95" claim rests on. Run: cargo bench -p agidb-core --bench recall_scan

use agidb_core::episode::encode_episode_signature;
use agidb_core::store::{Store, StoreConfig};
use agidb_core::types::{Episode, EpisodeId, Provenance, Query, TimeRange, Triple};
use chrono::{TimeZone, Utc};
use criterion::{criterion_group, criterion_main, Criterion};

fn populated_store(n: u64) -> (Store, tempfile::TempDir) {
    let dir = tempfile::TempDir::new().unwrap();
    let mut store = Store::open(StoreConfig::at(dir.path())).unwrap();
    let t0 = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
    for i in 1..=n {
        let id = EpisodeId::new(i);
        let triples = vec![Triple {
            subject: format!("Person{}", i % 500),
            predicate: "recommends".into(),
            object: format!("Place{}", i % 700),
            confidence: 0.9,
            episode_id: id,
        }];
        let sig = encode_episode_signature(&triples, Some(t0));
        let ep = Episode {
            id,
            text: format!("Person{} recommended Place{} enthusiastically", i % 500, i % 700),
            signature_offset: 0,
            gist_offset: 0,
            triples,
            valid_time: TimeRange::point(t0),
            t_tx_start: t0,
            provenance: Provenance::default(),
            confidence: 0.9,
            superseded_by: None,
        };
        store.observe(ep, &sig).unwrap();
    }
    (store, dir)
}

fn bench_recall(c: &mut Criterion) {
    let (store, _dir) = populated_store(10_000);
    c.bench_function("recall_gist_10k", |b| {
        b.iter(|| {
            store
                .recall(&Query::cue("recommended enthusiastically somewhere"))
                .unwrap()
        })
    });
    c.bench_function("recall_tier_a_10k", |b| {
        b.iter(|| store.recall(&Query::cue("Person42")).unwrap())
    });
}

criterion_group!(benches, bench_recall);
criterion_main!(benches);
```

Register in `crates/agidb-core/Cargo.toml` next to the existing `hdc_scan` bench entry:

```toml
[[bench]]
name = "recall_scan"
harness = false
```

Run: `cargo bench -p agidb-core --bench recall_scan 2>&1 | grep -A2 "recall_"` and paste the medians into `bench/RESULTS.md` under `## Microbenchmarks`. If `recall_gist_10k` median exceeds 50ms, record it honestly and file the number — do not tune constants to hide it.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "bench: deterministic retrieval benchmark vs SQLite FTS5 + naive scan

Seeded synthetic corpus (10k docs, supersession pairs), 4 query classes
(exact / single-entity / noisy-cue / temporal as-of), hit@k + MRR +
latency percentiles + ingest throughput + disk size. Criterion recall
bench for the core scan path. Raw results + limitations committed under
bench/."
```

---

### Task 6: Docs truth pass

**Files:**
- Modify: `README.md`
- Modify: `docs/phases/README.md` (status flags only)
- Modify: `docs/PROJECT.md` (status table only)
- Modify: `docs/product/roadmap.md` (status flags only)
- Modify: `docs/architecture/architecture.md` (two wording fixes)

**Interfaces:** none (prose). Every claim below is checked against what Tasks 1–5 actually shipped.

- [ ] **Step 1: README — consolidation honesty**

Replace every instance of "surprise-gated background worker" phrasing. Line 48 bullet 4 becomes:

```markdown
4. **sleep-like consolidation** — an explicit `consolidate()` pass (CLI/MCP/API) that clusters repeated episodic patterns into semantic atoms and supersedes contradictions; scheduling it in the background is on the roadmap, not in the engine yet
```

In "key properties", the "automatic consolidation" bullet becomes:

```markdown
- **consolidation on demand.** `consolidate()` clusters repeated patterns into semantic atoms and flags contradictions. It runs when you call it (CLI, MCP tool, or API); a self-scheduling background worker is roadmap.
```

- [ ] **Step 2: README — seven floors status table**

After the "what agidb is" paragraph (line ~15), replace the sentence claiming all seven are first-class with a status table:

```markdown
| floor | status |
|---|---|
| 1. sensory buffer (surprise-gated) | ✅ shipped |
| 2. working memory (session-scoped recency) | 🚧 planned |
| 3. episodic memory (bi-temporal) | ✅ shipped |
| 4. semantic memory (consolidated atoms) | ✅ shipped |
| 5. procedural memory | 🚧 types defined, retrieval planned |
| 6. goals + beliefs | ✅ shipped |
| 7. self-model (learning log + self-vector) | ✅ shipped |
```

- [ ] **Step 3: README — extractor honesty**

Replace "ships with the GLiNER extractor so observe runs out of the box" (line ~67) with:

```markdown
the CLI downloads the GLiNER extractor (~hundreds of MB, one time, sha256-verified) on the first `observe`; pass `--offline` to store text-only episodes with zero downloads. `recall`, `consolidate`, and every read-path command never need a model. Relation extraction is currently a curated heuristic (GLiNER provides entities); a learned relation extractor is roadmap.
```

- [ ] **Step 4: README — benchmarks section**

Add before "## documentation":

```markdown
## benchmarks

Honest numbers or none. The deterministic retrieval benchmark (agidb vs
SQLite FTS5 BM25 vs naive scan — hit@k, MRR, latency percentiles, ingest
throughput, disk size, noisy-cue and temporal classes) lives in
[`bench/RESULTS.md`](bench/RESULTS.md) with raw JSON alongside, and states
its limitations (synthetic corpus, lexical-structural only). The
constitution's full six-metric stack (adds BLEU, LLM-judge, token cost on
LongMemEval-style corpora) is not yet run — claims wait for numbers.
```

Delete or rewrite any sentence elsewhere in the README asserting benchmark results that don't exist in `bench/RESULTS.md`.

- [ ] **Step 5: README — v2.1 / brain-alignment status**

At the top of the "## brain-alignment (v2.1+)" section insert:

```markdown
> **Status: design documents only.** No multimodal code exists in this
> repository yet (no V-JEPA, no Wav2Vec, no BAMS harness). The original
> ICLR 2026 workshop target is stale and has been dropped from the
> critical path; the design docs remain as the v2.1 plan.
```

Fix the "## status" section to list what is actually implemented (phases 0–6 core, 9–11 cognitive primitives, floor-1 sensory buffer, 13 MCP tools, benchmark harness) and what is not (Python bindings, working-memory floor, procedural retrieval, multimodal).

- [ ] **Step 6: Phase/status doc sync**

In `docs/phases/README.md`, `docs/PROJECT.md` §4/§7 status tables, and `docs/product/roadmap.md`: set phase 3 = 🟨 partial (NER real, relations heuristic, gold set shipped), phase 5 = 🟨 partial (MCP shipped with 13 tools; Python bindings absent), phases 9/10/11 = ✅ (goals/beliefs; sensory+self-model; unlearn), phase 13 = 🟨 (retrieval benchmark shipped; LongMemEval harness pending). Do not rewrite prose bodies — status flags and one-line notes only.

In `docs/architecture/architecture.md`: (1) in the consolidation section, keep the step list but change the intro line to say the pass is invoked explicitly; (2) in the recall section, note tier B scores by density-corrected phi correlation.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "docs: truth pass — claims now match the code

Consolidation described as explicit pass, seven-floor status table,
extractor download reality, benchmarks section points at measured
results only, v2.1 brain-alignment marked design-only, phase statuses
synced."
```

---

### Task 7: Final end-to-end verification

**Files:** none new (verification only; fixes as needed).

- [ ] **Step 1: Full quality gate**

```bash
cargo fmt --all -- --check        # expect: no diff
cargo clippy --workspace --all-targets 2>&1 | grep -c "^error"   # expect 0
cargo test --workspace 2>&1 | grep -E "test result" | awk -F'[ ;]' '{p+=$4; f+=$6} END {print p" passed, "f" failed"}'
```
Expected: ≥140 passed, 0 failed.

- [ ] **Step 2: Example runs end to end**

Run: `cargo run -p agidb --example sarah_bawri 2>&1 | tail -15`
Expected: observe → recall (tier A) → consolidate → atom surfaces; exit 0.

- [ ] **Step 3: CLI demo end to end**

```bash
cargo build --release -p agidb-cli
B=target/release/agidb
D=$(mktemp -d)/mem
$B observe $D "Sarah recommended Bawri in Bandra last weekend" --offline
$B observe $D "Sarah said Bawri is a thai restaurant" --offline
$B observe $D "Marco asked the team to pick a thai place for dinner" --offline
$B recall  $D "what thai place did sarah mention?"
$B sense   $D "Sarah recommended Bawri in Bandra last weekend"   # duplicate → low surprise
$B sense   $D "the quarterly audit deadline moved to friday"     # novel → promoted
$B sensory $D
$B consolidate $D
$B stats   $D
```
Expected: recall returns the Bawri episodes with confidence + tier; first `sense` prints `"promoted": null` with surprise < 0.4; second prints a promoted episode id; `stats` shows episodes=4 (3 observed + 1 promoted), signatures ≥ 4.

- [ ] **Step 4: MCP smoke over stdio**

```bash
printf '%s\n' \
 '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
 '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' \
 | $B serve $(mktemp -d)/mcp 2>/dev/null | python3 -c "import sys,json; [print(len(json.loads(l).get('result',{}).get('tools',[]))) for l in sys.stdin if 'tools' in l]"
```
Expected: `13` (4 original + 9 new tools). (Adjust the `serve` invocation to the CLI's actual serve syntax from `--help`.)

- [ ] **Step 5: Working tree clean, history reviewed**

```bash
git status --short          # expect: empty
git log --oneline -8        # expect the 6 commits from Tasks 1-6 in order
```

- [ ] **Step 6: Record completion**

Append a dated entry at the bottom of `bench/RESULTS.md`: "End-to-end verification passed: <date>, <commit hash>, all workspace tests green, CLI + MCP smoke passed." Commit:

```bash
git add -A
git commit -m "chore: record end-to-end verification in bench/RESULTS.md"
```

---

## Self-review notes (already applied)

- **Spec coverage:** recall hot path (Task 1), Tier B (Task 1), sensory floor (Task 2), MCP surface + server dispatch (Task 3), gold set (Task 4), benchmark + "efficiency vs other DBs" + published data (Task 5), docs honesty (Task 6), end-to-end (Task 7). The static-embedding semantic upgrade (model2vec-class) is deliberately **out of scope** — it is the right next bet but is a separate plan; nothing here blocks it.
- **Type consistency:** `ScanEntry.sig_popcount` (Task 1) is consumed by `scan_phi` (Task 1) only; `SensoryObservation` (Task 2) is consumed by the `memory_sense` tool (Task 3) and the CLI; `phi_from_counts(n, pa, pb, hamming) -> f32` signature is identical in tests and impl; bench consumes only public `agidb-core` APIs that exist before Task 5.
- **Known judgment calls an executor may hit:** (1) borrow checker in `rebuild_scan_dir` — collect-then-push pattern is pre-approved; (2) facade `join_err` helper name — mirror whatever `stats()` uses; (3) eval package name — read `crates/agidb-extract/eval/Cargo.toml`; (4) `ToolResult::as_json` — mirror the existing dispatch tests' result-reading pattern; (5) tier-B floor may be tuned only within [0.055, 0.10].
