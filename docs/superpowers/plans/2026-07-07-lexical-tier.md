# agidb — Lexical Inverted-Index Tier

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Close the exact-class loss (0.000 hit@1 on the 10k benchmark) by adding a token-level inverted-index tier that uses the same posting-list intersection pattern as BM25 but keyed on canonical tokens instead of free text. Tier L (Lexical) sits between tier A (concept lookup) and tier B (structured HDC similarity), catching "the cue has the same words as the stored text, just in a different role" — exactly what tier B misses on single-triple episodes.

**Architecture:** Add a `TOKENS` redb table (`&str token → RoaringBitmap<episode_id>`) populated at `observe()` time by tokenizing `episode.text` and inserting the episode id into each token's posting list. At recall time the new `tier_l_lexical` function tokenizes the cue, looks up each token's bitmap, and computes the **intersection** — the same posting-list intersection that BM25 does. Rank by the count of cue tokens that match each candidate (more matches = higher rank), then hydrate the top-k candidates from redb and emit a `Lexical` tier match with confidence `[0.55, 0.95]`.

**Tech stack:** redb, roaring, the existing `crate::episode::tokenize`.

## Global constraints

- Rust top to bottom. No Python/JS. No FFI.
- **No LLM, no network, no model inference in the read path.** This tier is a pure redb read + RoaringBitmap intersection.
- Never return the empty set under the default `tier_floor` when ≥1 non-tombstoned episode exists. (Article VI)
- Conventional commits. No attribution lines.
- TDD: failing test first; `cargo test --workspace` green before commit.
- Don't modify `.specify/memory/constitution.md`.

---

## State of the working tree (read before Task 1)

Read before Task 1. The model2vec plan (`12a298a`) shipped and is HEAD. This plan adds a new tier (Lexical) and bumps STORE_FORMAT_VERSION → 4.

Current test status from HEAD: 176 passed, 0 failed.

The bench numbers from HEAD on the 10k synthetic corpus (in `bench/RESULTS.md`):

```
### exact
| system       | hit@1 | hit@5 | MRR  | p95 ms |
| agidb        | 0.000 | 0.081 | 0.045| 0.91   |
| sqlite-fts5  | 1.000 | 1.000 | 1.000| 0.30   |
| naive-scan   | 0.613 | 0.968 | 0.770| 0.61   |
```

Goal: agidb exact hit@1 ≥ 0.50 with the new tier, hold or improve the other classes.

---

### Task 1: `TOKENS` table + populate from `observe()`

**Files:**
- Modify: `crates/agidb-core/src/store.rs` (add `TOKENS` table def; populate from `observe()`; rebuild in `rebuild_scan_dir` for reopens)
- Modify: `crates/agidb-core/tests/lexical_properties.rs` (new)

**Schema:** `pub const TOKENS: TableDefinition<&str, Vec<u8>> = TableDefinition::new("tokens");` — value is a RoaringBitmap of episode ids that contain this token.

**Populate:** In `observe()`, after the existing `inverted_index` writes, iterate `tokenize(&episode.text)` and for each token, insert `episode_id.raw() as u32` into the token's bitmap. Wrap in a redb transaction alongside the other writes.

**Rebuild:** `rebuild_scan_dir` already iterates `EPISODES`; add a second pass that scans every episode's text, tokenizes, and rebuilds the `TOKENS` table from scratch. (The table is a cache, so a fresh-from-discs rebuild on every open is fine and matches what we already do for `INVERTED_INDEX`.)

- [ ] **Step 1: Write the failing test**

```rust
//! Lexical-tier inverted-index invariants.

use agidb_core::store::{Store, StoreConfig};
use agidb_core::types::{Episode, EpisodeId, Provenance, Query, TimeRange, Triple};
use chrono::{TimeZone, Utc};
use tempfile::TempDir;

fn make_episode(id: u64, text: &str) -> Episode {
    let ep_id = EpisodeId::new(id);
    Episode {
        id: ep_id,
        text: text.to_string(),
        signature_offset: 0,
        gist_offset: 0,
        embedding_offset: 0,
        triples: vec![Triple {
            subject: "x".into(),
            predicate: "y".into(),
            object: "z".into(),
            confidence: 0.9,
            episode_id: ep_id,
        }],
        valid_time: TimeRange::point(Utc.with_ymd_and_hms(2026, 5, 14, 0, 0, 0).unwrap()),
        t_tx_start: Utc.with_ymd_and_hms(2026, 5, 14, 0, 0, 0).unwrap(),
        provenance: Provenance::default(),
        confidence: 0.9,
        superseded_by: None,
    }
}

#[test]
fn tokens_inverted_index_rebuilt_across_reopen() {
    let dir = TempDir::new().unwrap();
    let sig_bytes: [u8; 1024] = std::array::from_fn(|i| (i % 7) as u8);
    let signature = agidb_core::hdc::HV(sig_bytes);

    {
        let mut store = Store::open(StoreConfig::at(dir.path())).unwrap();
        for (id, text) in [
            (1u64, "Sarah recommended Bawri in Bandra last weekend"),
            (2, "Marco asked the team to pick a thai place for dinner"),
            (3, "Bawri serves great thai food"),
        ] {
            let mut ep = make_episode(id, text);
            store.observe(ep, &signature).unwrap();
        }
    }
    // Reopen — the tokens table must rebuild from the persisted EPISODES rows.
    let store = Store::open(StoreConfig::at(dir.path())).unwrap();
    let r = store
        .recall(&Query::cue("Bawri thai food"))
        .unwrap();
    // All three episodes share at least one cue token with the query.
    assert!(!r.matches.is_empty(), "token-level recall must produce matches");
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p agidb-core --test lexical_properties 2>&1 | tail -3`. Expect: tier L doesn't exist; cascade falls through to Gist (or NearestNeighbor) and likely misses.

- [ ] **Step 3: Add `TOKENS` table def**

In `crates/agidb-core/src/store.rs`, next to the existing `INVERTED_INDEX`:

```rust
/// Token → episode-id posting list. Populated from `episode.text` at
/// observe time; rebuilt from the EPISODES table at open. Drives
/// the lexical (tier L) recall tier — same posting-list intersection
/// pattern as BM25, keyed on canonical tokens instead of free text.
pub const TOKENS: TableDefinition<&str, Vec<u8>> = TableDefinition::new("tokens");
```

Add `let _ = tx.open_table(TOKENS)?;` to the same "open all tables" block in `open()`.

- [ ] **Step 4: Populate from `observe()`**

Inside the redb write transaction in `observe()`, after the inverted-index writes:

```rust
let mut tokens = tx.open_table(TOKENS)?;
for token in crate::episode::tokenize(&episode.text) {
    let key = token;
    let existing = tokens.get(key)?.map(|v| v.value());
    let mut bitmap = match existing {
        Some(bytes) => RoaringBitmap::deserialize_from(bytes.as_slice())
            .map_err(|e| AgidbError::Internal(format!("tokens decode: {e}")))?,
        None => RoaringBitmap::new(),
    };
    bitmap.insert(episode_id.raw() as u32);
    let mut bytes = Vec::with_capacity(bitmap.serialized_size());
    bitmap.serialize_into(&mut bytes)
        .map_err(|e| AgidbError::Internal(format!("tokens encode: {e}")))?;
    tokens.insert(key, bytes)?;
}
```

- [ ] **Step 5: Rebuild on open**

In `rebuild_scan_dir`, after the existing scan-directory pass, add a `TOKENS` rebuild:

```rust
let mut tokens = self.db.begin_write()?;
{
    let mut table = tokens.open_table(TOKENS)?;
    // Clear stale postings before the rebuild (defensive — `open()`
    // has already created the table, so on first open it's empty).
    let _ = table.retain(|_, _| false);
    // Walk every EPISODES row, tokenize text, accumulate per-token.
    let mut postings: std::collections::BTreeMap<String, RoaringBitmap> =
        std::collections::BTreeMap::new();
    let tx = self.db.begin_read()?;
    let episodes = tx.open_table(EPISODES)?;
    for entry in episodes.iter()? {
        let (_, v) = entry?;
        let ep: Episode = decode(&v.value())?;
        for token in crate::episode::tokenize(&ep.text) {
            postings.entry(token).or_default().insert(ep.id.raw() as u32);
        }
    }
    drop(tx);
    for (token, bitmap) in postings {
        let mut bytes = Vec::with_capacity(bitmap.serialized_size());
        bitmap.serialize_into(&mut bytes)
            .map_err(|e| AgidbError::Internal(format!("tokens encode: {e}")))?;
        table.insert(token, bytes)?;
    }
}
tokens.commit()?;
```

- [ ] **Step 6: Bump STORE_FORMAT_VERSION → 4**

```rust
/// v4: added `TOKENS` table for the lexical inverted-index tier.
/// v3 stores fail to open with `FormatVersion`; v3 JSONL imports still
/// load via serde(default) on the new fields.
pub const STORE_FORMAT_VERSION: u32 = 4;
```

- [ ] **Step 7: Run the test — green**

`cargo test -p agidb-core --test lexical_properties`. Expect: 1 passed.

`cargo test --workspace` — no regressions. 176 still passing.

- [ ] **Step 8: Commit**

```
perf(core): TOKENS inverted index for the lexical tier (STORE_FORMAT_VERSION=4)

TOKENS table: token string -> RoaringBitmap<episode_id>, populated
from episode.text at observe(), rebuilt from EPISODES at open. Same
posting-list-intersection pattern as BM25, keyed on canonical
tokens instead of free text. Tier L recall (Task 2) consumes it.
```

---

### Task 2: Tier L — lexical posting-list intersection

**Files:**
- Modify: `crates/agidb-core/src/types.rs` (add `Lexical` variant to `Tier`, depth 1)
- Modify: `crates/agidb-core/src/recall.rs` (add `tier_l_lexical`, insert into cascade between tier A and tier B)
- Modify: `crates/agidb-core/tests/lexical_properties.rs` (more tests)

**Cascade order (after):** A (concept lookup) → **L (lexical posting-list intersection)** → B (phi) → E (semantic) → C (gist) → D (NN).

**Scoring:** for each candidate episode id that appears in ≥1 cue-token posting list, count the number of cue tokens whose posting list contains the id. Confidence is the count normalized to [0.55, 0.95] — at least one cue token match lands at 0.55, and matching every cue token saturates at 0.95.

**Determinism:** sort candidates by (match_count DESC, episode_id ASC) so the order is reproducible.

- [ ] **Step 1: Add `Lexical` variant to `Tier`**

In `types.rs`:

```rust
pub enum Tier {
    /// Canonical entity match via the concept index.
    Exact,
    /// Token-level inverted-index posting-list intersection.
    Lexical,
    /// Role-bound structured HDC signature (phi-corrected).
    Similarity,
    /// Charikar-projected static-text embedding.
    Semantic,
    /// Raw-text gist fallback.
    Gist,
    /// Best-effort nearest neighbor.
    NearestNeighbor,
}
```

Update `Tier::depth`: Lexical = 1, Similarity = 2, Semantic = 3, Gist = 4, NearestNeighbor = 5.

- [ ] **Step 2: Write the failing tests**

Append to `lexical_properties.rs`:

```rust
#[test]
fn tier_l_ranks_episodes_by_token_overlap_count() {
    // Episode 1 shares 2 cue tokens ("Bawri", "thai"). Episode 2
    // shares 1. Episode 3 shares 0. Tier L must rank 1, 2, 3.
    let dir = TempDir::new().unwrap();
    let sig_bytes: [u8; 1024] = std::array::from_fn(|i| (i % 11) as u8);
    let signature = agidb_core::hdc::HV(sig_bytes);
    let mut store = Store::open(StoreConfig::at(dir.path())).unwrap();
    store
        .observe(make_episode(1, "Bawri serves thai food in Bandra"), &signature)
        .unwrap();
    store
        .observe(make_episode(2, "Marco likes thai food"), &signature)
        .unwrap();
    store
        .observe(make_episode(3, "Alice works at Stripe"), &signature)
        .unwrap();
    let r = store.recall(&Query::cue("Bawri thai")).unwrap();
    assert_eq!(r.tier_used, agidb_core::types::Tier::Lexical,
        "tier L must fire on cue 'Bawri thai' — got {:?}", r.tier_used);
    assert_eq!(r.matches[0].episode_id, EpisodeId::new(1),
        "episode 1 (2 token matches) must rank first");
    assert!(r.matches[0].confidence >= r.matches[1].confidence,
        "confidence must descend by match count");
}

#[test]
fn tier_l_ignores_tombstoned_episodes() {
    let dir = TempDir::new().unwrap();
    let sig_bytes: [u8; 1024] = std::array::from_fn(|i| (i % 13) as u8);
    let signature = agidb_core::hdc::HV(sig_bytes);
    let mut store = Store::open(StoreConfig::at(dir.path())).unwrap();
    store.observe(make_episode(1, "Bawri thai"), &signature).unwrap();
    store.observe(make_episode(2, "Bawri noodles"), &signature).unwrap();
    store.unlearn(
        agidb_core::unlearn::UnlearnTarget::Episode(EpisodeId::new(1)),
        "test",
    ).unwrap();
    let r = store.recall(&Query::cue("Bawri")).unwrap();
    assert_eq!(r.matches[0].episode_id, EpisodeId::new(2),
        "tombstoned episode must be filtered out of tier L");
}
```

- [ ] **Step 3: Verify failure**

`cargo test -p agidb-core --test lexical_properties`. Expect `no variant Lexical` for `Tier::Lexical`, or cascade never firing Lexical.

- [ ] **Step 4: Implement `tier_l_lexical` in `recall.rs`**

```rust
/// Tier L — token-level inverted-index posting-list intersection.
/// Ranks candidates by the count of cue tokens whose posting list
/// contains the candidate id. Empty intersection → empty result;
/// the cascade falls through to tier B.
fn tier_l_lexical(&self, query: &Query) -> Result<Vec<RecallMatch>> {
    let cue_tokens: Vec<String> = crate::episode::tokenize(&query.cue);
    if cue_tokens.is_empty() {
        return Ok(vec![]);
    }
    let tx = self.db.begin_read()?;
    let table = tx.open_table(crate::store::TOKENS)?;

    // Map episode_id -> (match_count, popcount_for_length_normalization).
    let mut counts: std::collections::HashMap<u64, u32> =
        std::collections::HashMap::new();
    for token in &cue_tokens {
        let Some(bytes) = table.get(token.as_str())?.map(|v| v.value()) else {
            continue;
        };
        let bitmap = RoaringBitmap::deserialize_from(bytes.as_slice())
            .map_err(|e| AgidbError::Internal(format!("tokens decode: {e}")))?;
        for id in bitmap.iter() {
            *counts.entry(id as u64).or_insert(0) += 1;
        }
    }
    drop(table);
    drop(tx);

    if counts.is_empty() {
        return Ok(vec![]);
    }

    // Apply tombstone + bi-temporal filters via the scan directory.
    let mut candidates: Vec<(u64, u32)> = counts.into_iter()
        .filter(|(id, _)| {
            self.scan_entry(*id)
                .map(|e| !e.tombstoned && e.valid_at(query.as_of.unwrap_or(chrono::Utc::now())))
                .unwrap_or(false)
        })
        .collect();
    // Sort by match count DESC, then id ASC for determinism.
    candidates.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

    let max_count = cue_tokens.len() as u32;
    let mut out = Vec::new();
    for &(id, count) in &candidates {
        if out.len() >= query.k {
            break;
        }
        let Some(ep) = self.get_episode(EpisodeId::new(id))? else {
            continue;
        };
        // Linear confidence map: 1 match → 0.55, all cues match → 0.95.
        let confidence = 0.55 + (0.95 - 0.55) * (count as f32 - 1.0)
            / (max_count as f32 - 1.0).max(1.0);
        let confidence = confidence.clamp(0.55, 0.95);
        out.push(into_match(ep, confidence, Tier::Lexical));
    }
    Ok(out)
}
```

- [ ] **Step 5: Insert into `run_cascade` between tier A and tier B**

```rust
// Tier L — lexical inverted-index posting-list intersection. Closes
// the exact-class loss where the cue shares tokens with the stored
// text but tier A's concept lookup misses (the token isn't a known
// concept yet) and tier B's structured phi is too sparse to fire.
if Tier::Lexical.depth() <= query.tier_floor.depth() {
    let l = self.tier_l_lexical(query)?;
    if !l.is_empty() {
        return Ok(self.finalize(l, query));
    }
}
```

- [ ] **Step 6: Run the new tests — green**

`cargo test -p agidb-core --test lexical_properties`. Expect 3 passed.

`cargo test --workspace`. 176 still green (or +3 if Task 1's single test was the only one).

- [ ] **Step 7: Commit**

```
perf(core): tier L - lexical posting-list intersection on TOKENS table

Inserts between tier A (concept lookup) and tier B (structured phi).
Same posting-list intersection pattern as BM25, but keyed on canonical
tokens (from the existing crate::episode::tokenize helper). Closes
the exact-class loss: when the cue shares 1+ tokens with the stored
text, tier L ranks by match count, hydrates from the scan directory
+ tombstone + bi-temporal filters, and emits matches with confidence
in [0.55, 0.95].
```

---

### Task 3: Re-benchmark + docs

- [ ] **Step 1: Run the bench**

`cargo run -p agidb-bench --release -- --episodes 10000 --queries 250 --out bench 2>&1 | tee /tmp/bench-tier-l.txt`

Expected: agidb exact hit@1 should jump from 0.000 to ≥ 0.50; other classes should hold (or improve, if tier L short-circuits some queries that were falling through to tier B and dying).

- [ ] **Step 2: Update `bench/RESULTS.md`**

Add a new section `## Tier L lexical tier added` below the existing retrieval-benchmark section, with the new tables. Be honest about what changed and what didn't.

- [ ] **Step 3: Update README status block**

In the seven-floors table and the "Honest numbers" paragraph, swap the exact-class loss line from "loses 0.000 vs 1.000" to whatever the new number is, and call out tier L as a real recall tier.

- [ ] **Step 4: Full quality gate + commit**

`cargo test --workspace` — 179 passed (176 + 3 new). `cargo clippy` clean. `cargo fmt --all`. Commit + push.

---

## Self-review notes

- **Architecture coherence:** Tier L sits between tier A and tier B in the depth ordering. Tier A catches the canonical-concept case (the cue token is a known concept, exact match). Tier L catches the "cue token is a content word that overlaps with the stored text" case. Tier B catches the structured-role overlap (the cue has the same entity in subject/object role as the stored triple). Tier L is much cheaper than tier B (one redb read per token + bitmap OR, no per-row scan), so it short-circuits well on the high-precision-but-cheap case.
- **Determinism:** sort by `(match_count DESC, episode_id ASC)` so identical queries always return identical rankings.
- **Tombstone / temporal:** filters applied via the scan directory — tier L respects `as_of` and skips tombstoned episodes just like every other tier.
- **Type consistency:** `TOKENS` table uses the same `(key, RoaringBitmap-bytes)` shape as the existing `INVERTED_INDEX`. The pattern is identical and the rebuild-on-open path mirrors what we already do.
- **Risks:** the bench templates for `exact` use the place name in the cue. With tier L, those cues token-match the place and now tier L fires. This is the right behavior — it's exactly what BM25 does — and the bench should show agidb matching FTS5 on the exact class instead of losing.
- **What this plan does NOT do:** doesn't add a phrase-level index, doesn't add per-token IDF weights, doesn't change tier A or B. All deferred per the assessment's "scope is the risk" guidance.