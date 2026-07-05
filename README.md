# agidb

> **agidb is the cognitive substrate for autonomous AI agents — content-addressable hyperdimensional memory, first-class goals and beliefs, bi-temporal supersession, sleep-like consolidation, and a non-destructive unlearn primitive. One Rust binary, one API, no query language. The database AGI systems will run on top of.**

[![Crates.io](https://img.shields.io/crates/v/agidb.svg)](https://crates.io/crates/agidb)
[![Docs](https://docs.rs/agidb/badge.svg)](https://docs.rs/agidb)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

## what agidb is

agidb is a new category of database — a **cognitive substrate** designed for autonomous AI agents that need to *remember, reason, revise, and forget* across days, weeks, and years.

most databases force you to write queries: SQL, Cypher, JSON paths, vector embeddings. agidb doesn't. you give it observations in plain text (or in v2.1+, raw video and audio), and it figures out the rest. retrieval is one function call, runs in milliseconds on a laptop, makes zero network calls in the read path, and never returns "no results" — only matches with explicit confidence scores.

agidb stores the seven things an autonomous AI system needs to persist: sensory input, working memory, episodic memories, semantic facts, procedural skills, **first-class goals and beliefs**, and a self-model audit log. it is the first database where all seven are first-class types with their own retrieval semantics — not bolted on with schemas.

```rust
let db = Agidb::open("./memory.agidb").await?;

db.observe("Sarah recommended Bawri in Bandra last weekend").await?;
db.assert_belief(Belief::new("Sarah likes thai food").with_confidence(0.8)).await?;
db.set_goal(Goal::new("find a thai place for the team dinner")).await?;

let result = db.recall("what thai place did sarah mention?").await?;
// → Bawri, confidence 0.94, with provenance back to the original observation,
//   goal-biased because "find a thai place" is currently active.
```

no SQL. no Cypher. no embedding API calls. no separate vector db. no rerank step. one function.

## the lineage — agidb v2 succeeds sochdb v1

agidb is the v2 successor to **sochdb**, the embedded memory database we shipped through phases 0-2, 4, and 6 (HDC kernel, bi-temporal storage, episode binding, tiered recall, consolidation). every line of sochdb's code carries forward. the HDC math is the same. the storage layout is the same. bi-temporal supersession is the same. **agidb is sochdb extended with the cognitive primitives an AGI substrate requires** — goals, beliefs, sensory buffering, self-model, neurosymbolic interface, and a first-class unlearn API.

if you used sochdb v1, agidb v2 is a strict superset. no migrations required for the substrate features you already use.

## why agidb exists

today's agent memory pipeline is a six-step mess: embed every conversation → store vectors → embed the query → similarity search → rerank with another LLM call → stuff chunks into the prompt. it's slow (p95 1-3 seconds), expensive (API calls per query), has no temporal grounding, weak provenance, no graceful degradation, no consolidation, and certainly no first-class concept of *what the agent wants* or *what the agent believes*.

an autonomous AI agent is not a search-over-documents application. it is a system that remembers, reasons, plans, learns, and forgets — and needs infrastructure shaped like that. agidb is shaped like that.

agidb replaces the six-step pipeline with one local function call, and adds five primitives no other database has:

1. **content-addressable storage via hyperdimensional signatures** — memories are 8192-bit binary fingerprints; retrieval is bit-overlap counting, not query parsing
2. **bi-temporal supersession** — facts don't overwrite, they supersede; query "as of" any historical date
3. **first-class goals and beliefs** — typed state machines for goals; revisable beliefs with audit trails
4. **sleep-like consolidation** — an explicit `consolidate()` pass (CLI/MCP/API) that clusters repeated episodic patterns into semantic atoms and supersedes contradictions; scheduling it in the background is on the roadmap, not in the engine yet
5. **non-destructive unlearn** — cascading removal with full audit trail; right-to-be-forgotten as a first-class operation

## key properties

- **embedded.** one Rust binary, runs locally, no server required. like sqlite, not like postgres.
- **content-addressable.** retrieval by partial cue, not by query. like remembering a song from humming three notes.
- **bi-temporal by default.** every fact carries valid-time (when true) and transaction-time (when learned). contradictions supersede; nothing is overwritten silently.
- **all seven cognitive tiers.** sensory, working, episodic, semantic, procedural, goals/beliefs, self-model — status below.
- **no LLM in the read path.** recall is deterministic math over stored signatures. zero API keys, zero network calls, zero hallucination at retrieval time. (LLMs may participate at write time for belief revision and consolidation, never at read.)
- **tiered confidence.** `recall()` never returns empty. exact → similarity → gist → nearest-neighbor, each with explicit confidence scores.
- **consolidation on demand.** `consolidate()` clusters repeated patterns into semantic atoms and flags contradictions. It runs when you call it (CLI, MCP tool, or API); a self-scheduling background worker is roadmap.
- **non-destructive unlearn.** every fact is removable with a cascading audit log; nothing silently disappears.
- **full provenance.** every claim traces back to the verbatim observation that produced it. every belief revision logs why. no opaque embeddings, no untraceable facts.
- **introspectable.** the self-model log records every learning event; the agent can ask "what did I learn this week?"
- **multimodal-ready (v2.1+).** V-JEPA 2 for video, Wav2Vec-BERT for audio, GLiNER for text — all extracted at write time, fused into one HDC episode signature via VSA binding.

| floor | status |
|---|---|
| 1. sensory buffer (surprise-gated) | ✅ shipped |
| 2. working memory (session-scoped recency) | 🚧 planned |
| 3. episodic memory (bi-temporal) | ✅ shipped |
| 4. semantic memory (consolidated atoms) | ✅ shipped |
| 5. procedural memory | 🚧 types defined, retrieval planned |
| 6. goals + beliefs | ✅ shipped |
| 7. self-model (learning log + self-vector) | ✅ shipped |

## quickstart

install the CLI with one command — works on linux (x86_64 / aarch64) and macOS (Intel / Apple silicon). no dependencies, ~35 MB. The CLI downloads the GLiNER extractor (~hundreds of MB, one time, sha256-verified) on the first `observe`; pass `--offline` to store text-only episodes with zero downloads. `recall`, `consolidate`, and every read-path command never need a model. Relation extraction is currently a curated heuristic (GLiNER provides entities); a learned relation extractor is roadmap.

```bash
# pin a release (recommended for reproducibility)
curl -fsSL https://github.com/rohansx/agidb/releases/download/v0.1.0-dev.1/install.sh | sh

# or latest from master
curl -fsSL https://raw.githubusercontent.com/rohansx/agidb/master/install.sh | sh
```

the script auto-detects OS + arch, downloads the matching binary from the latest GitHub release, verifies its sha256 against the published checksums.txt, and drops it at `/usr/local/bin/agidb` (or `~/.local/bin/agidb` if you don't have write access). pass `--tag vX.Y.Z` to pin a specific version (skips the GitHub API lookup), `--to ~/bin` to choose a different install dir, `--repo rohansx/agidb` for a fork.

**why the pinned URL is preferred:** raw.githubusercontent.com serves `master/install.sh` through a CDN that can lag several minutes behind pushes. The release-pinned URL (`/releases/download/vX.Y.Z/install.sh`) is always in sync with the binaries it installs.

verify:

```bash
agidb --version                                    # agidb 0.1.0-dev
agidb observe ./mem.agidb "Sarah recommended Bawri"  # creates ./mem.agidb
agidb recall  ./mem.agidb "what thai place?"
agidb stats   ./mem.agidb
```

rust library users:

```bash
cargo add agidb
```

```rust
use agidb::{Agidb, Query, Goal, Belief};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let db = Agidb::open("./memory.agidb").await?;

    // floor 3 — episodic memory
    db.observe("Sarah told me about Letta's three-tier memory at PyCon", None).await?;

    // floor 6 — beliefs with confidence
    db.assert_belief(
        Belief::new("Letta uses OS-inspired memory tiering")
            .with_confidence(0.9)
            .with_evidence_from("Sarah's PyCon talk")
    ).await?;

    // floor 6 — goals with state machine
    let goal = db.set_goal(
        Goal::new("evaluate Letta vs agidb for next agent build")
    ).await?;

    // floor 1-7 — unified recall
    let recall = db.recall(Query::cue("what did sarah say about memory?")).await?;
    for r in recall.matches {
        println!("[{:.2}] {}", r.confidence, r.text);
    }

    // floor 7 — introspection
    let log = db.what_did_i_learn(since_yesterday()).await?;

    Ok(())
}
```

## demo

The fastest way to see agidb work is the bundled example — it runs the
"Sarah recommends Bawri" scenario end-to-end (observe → tier-A exact
recall → sleep-like consolidation mints a semantic atom → atom surfaces
in recall), fully offline and deterministic. with the CLI installed:

```bash
mkdir mem && cd mem
agidb observe  . "Sarah recommended Bawri in Bandra last weekend"
agidb observe  . "Sarah said Bawri is a thai restaurant"
agidb observe  . "Marco asked the team to pick a thai place for dinner"
agidb recall   . "what thai place did Sarah mention?"
agidb consolidate .
agidb stats    .
```

library users (after `cargo add agidb`):

```bash
cargo run --example sarah_bawri
```

There is also a real CLI:

```bash
# record an observation (auto-loads GLiNER; --offline stores text-only)
agidb observe ./mem.agidb "Sarah recommended Bawri in Bandra last weekend"
agidb recall  ./mem.agidb "what did Sarah mention?"
agidb consolidate ./mem.agidb
agidb stats  ./mem.agidb
agidb list   ./mem.agidb
agidb export ./mem.agidb dump.jsonl
# expose agidb to Claude Desktop / Cursor over MCP stdio:
agidb serve  ./mem.agidb
```

The umbrella crate's [`Agidb`](https://docs.rs/agidb) facade ties the
whole pipeline together — text in, structured triples out, signed,
indexed, bi-temporally stamped, and recallable by cue, in one object:

```rust
use agidb::Agidb;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let db = Agidb::open("./memory.agidb").await?;
    db.observe("Sarah recommended Bawri in Bandra last weekend").await?;
    let r = db.recall_cue("what did Sarah mention?").await?;
    for m in &r.matches {
        println!("[{:.2}] {}", m.confidence, m.text);
    }
    db.consolidate().await?; // sleep → semantic atoms
    Ok(())
}
```

## the architecture in one paragraph

agidb is built in three engineering layers that implement seven cognitive floors. **layer 1** is the mind-like layer: every memory becomes an 8192-bit hyperdimensional signature, and retrieval is bit-overlap counting (POPCOUNT) over stored signatures. **layer 2** is the scaffolding: GLiNER ONNX extracts entities and relations locally so signatures encode meaning rather than phrasing; from v2.1, V-JEPA 2 + Wav2Vec-BERT extend this to video and audio. **layer 3** is the storage: redb for metadata and bi-temporal indexes, mmap'd flat files for signatures, append-only logs for the self-model audit trail. these three engineering layers together implement the seven floors of an AGI substrate: sensory buffer, working memory, episodic memory, semantic memory, procedural memory, goals + beliefs, and the self-model. the user only ever sees the seven floors.

See [architecture.md](docs/architecture/architecture.md) for the full picture.

## brain-alignment (v2.1+)

agidb v2.1 introduces a **brain-aligned mode**: multimodal sensory encoding via Meta's V-JEPA 2 (video), Wav2Vec-BERT (audio), and Llama-3.2-3B (text) — the same encoder stack used by Meta FAIR's TRIBE v2 brain-encoding foundation model that won the Algonauts 2025 fMRI prediction competition.

this matters for three reasons:

1. **brain-calibrated surprise gating.** the threshold that promotes sensory frames to episodic memory can be empirically calibrated against neural surprise signals predicted by TRIBE v2 over 720-subject fMRI data. this gives agidb's sensory floor a measurement-grounded threshold instead of a magic number.
2. **representational alignment with human cortex.** because agidb shares the encoder stack with TRIBE v2, its episode signatures can be directly compared against TRIBE-predicted cortical activations on matched stimuli via representational similarity analysis (RSA).
3. **BAMS — the brain-aligned memory benchmark.** agidb v2.1 ships with BAMS, a new evaluation suite measuring agent memory systems against TRIBE-derived cortical ground truth across six functional networks (DMN, visual, auditory, language, dorsal attention, frontoparietal). first benchmark of its kind. published at ICLR 2026 MemAgents workshop.

See [brain-alignment.md](docs/architecture/brain-alignment.md) and [bams-benchmark.md](docs/architecture/bams-benchmark.md).

> **Status: design documents only.** No multimodal code exists in this repository yet (no V-JEPA, no Wav2Vec, no BAMS harness). The original ICLR 2026 workshop target is stale and has been dropped from the critical path; the design docs remain as the v2.1 plan.

## benchmarks

Honest numbers or none. The deterministic retrieval benchmark (agidb vs SQLite FTS5 BM25 vs naive scan — hit@k, MRR, latency percentiles, ingest throughput, disk size, noisy-cue and temporal classes) lives in [`bench/RESULTS.md`](bench/RESULTS.md) with raw JSON alongside, and states its limitations (synthetic corpus, lexical-structural only). The constitution's full six-metric stack (adds BLEU, LLM-judge, token cost on LongMemEval-style corpora) is not yet run — claims wait for numbers.

## documentation

| doc | what's in it |
|---|---|
| [overview.md](docs/product/overview.md) | product overview, who it's for, comparisons, why now |
| [PROJECT.md](docs/PROJECT.md) | the master reference, 12 sections, all phases |
| [architecture.md](docs/architecture/architecture.md) | the three engineering layers + seven floors, data flow, design choices |
| [biological-mapping.md](docs/product/biological-mapping.md) | the seven floors mapped to cognitive psychology |
| [layer-1-recall.md](docs/architecture/layer-1-recall.md) | HDC signatures, binding, bundling, tiered retrieval, goal-biased recall |
| [layer-2-extraction.md](docs/architecture/layer-2-extraction.md) | GLiNER, V-JEPA 2, Wav2Vec-BERT, triples, belief extraction |
| [layer-3-storage.md](docs/architecture/layer-3-storage.md) | redb schema, mmap signatures, bi-temporal model, on-disk layout |
| [cognitive-primitives.md](docs/architecture/cognitive-primitives.md) | goals, beliefs, sensory buffer, self-model, unlearn API |
| [neurosymbolic.md](docs/architecture/neurosymbolic.md) | the bidirectional interface between signatures and structured beliefs |
| [brain-alignment.md](docs/architecture/brain-alignment.md) | **new in v2** — V-JEPA 2 + TRIBE v2 integration, brain-calibrated surprise gating |
| [bams-benchmark.md](docs/architecture/bams-benchmark.md) | **new in v2** — the brain-aligned memory similarity benchmark, paper plan |
| [tech-spec.md](docs/spec/tech-spec.md) | the full Rust API, types, trait, performance targets |
| [agi-trajectory.md](docs/product/agi-trajectory.md) | the 5-year roadmap from v2.0 substrate to v2.5 AGI-grade |
| [roadmap.md](docs/product/roadmap.md) | the near-term phase plan, weeks 1-52 (extended through BAMS at month 12) |
| [CONSTITUTION.md](docs/spec/constitution.md) | the immutable principles governing every decision |

## status

agidb v2 is pre-alpha. **Shipped in this repo:** the HDC kernel + bi-temporal storage + tiered recall cascade (A exact → B phi-corrected structured similarity → C gist → D nearest-neighbor) + consolidation + goals/beliefs (state machine + revisable belief log + revision math) + non-destructive unlearn with 30-day restore window + self-model (learning log + self-vector) + floor-1 sensory buffer with surprise-gated promotion + 13-tool MCP server (Claude Desktop / Cursor) + deterministic benchmark harness (agidb vs SQLite FTS5 BM25 vs naive scan — see [`bench/RESULTS.md`](bench/RESULTS.md)).

**Planned (no code yet):** floor-2 working memory, floor-5 procedural retrieval, multimodal encoding (V-JEPA 2 / Wav2Vec-BERT), BAMS brain-alignment benchmark suite, Python bindings, background consolidation scheduler. The ICLR 2026 MemAgents workshop target is dropped from the critical path; v2.1 multimodal and BAMS remain as design docs under `docs/architecture/`.

**Honest numbers from the 10k-ep benchmark:**
- agidb wins **noisy queries** decisively (0.90 hit@1 vs 0.00 for BM25/scan).
- agidb ties on **single-entity** queries (1.00 all three).
- agidb loses **exact** (0.00 vs 1.00 FTS5) and **temporal** (0.02 vs 0.50 FTS5) — structural limitations of HDC token-bundle scoring, captured honestly in `bench/RESULTS.md`.

**Honest number from the 100-row extraction gold set:** F1 = 0.592 (P=0.865, R=0.450). The extractor is conservative; OOV paraphrase misses are correctly labeled as expected misses in the gold set.

## license

Apache-2.0
