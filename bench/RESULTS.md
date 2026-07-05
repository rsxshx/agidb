# agidb Benchmark Results

> **Honest numbers or none.** Each entry below is what the harness
> actually measured on this machine. Synthetic, templated corpus —
> see Limitations.

## Extraction F1 (100-sample gold set)

- **Date:** 2026-07-05
- **Command:** `cargo run -p agidb-extract-eval -- --gate 0.0`
- **Gold set:** `crates/agidb-extract/eval/gold/observations.jsonl` (100 rows: 30 recommends, 20 likes, 15 works_at, 15 located_in, 10 two-triple, 10 zero-triple; 70 in-vocab surface forms + 20 OOV paraphrases marked as expected heuristic misses).
- **Result:** **P=0.865, R=0.450, F1=0.592** (n=100, dry_run=false)
- **Interpretation:** High precision, low recall. The extractor is conservative — it surfaces a triple only when the surface form is close to a canonical synonym; the OOV paraphrase rows (≈20% of the gold set) are correctly expected misses. Recall numbers should improve as the relation extractor moves from a curated synonym table to a learned model (planned v2.1).

## Retrieval benchmark (10k synthetic corpus)

- **Date:** 2026-07-05
- **Commit:** `92cbb05`
- **Host:** Linux 7.1.2-3-cachyos, AMD Ryzen 7 7435HS
- **Command:** `cargo run -p agidb-bench --release -- --episodes 10000 --queries 200 --out bench`
- **Corpus:** 10,100 templated documents (10,000 base + 100 supersession pairs) across 40 people × 40 places × 4 predicates with deterministic seed 42. Spans 2026-01-01 → 2026-10-28.
- **Queries:** 200 total, equal split across 4 classes (exact / single-entity / noisy / temporal).
- **Systems:** agidb (scan-directory mmap recall), SQLite FTS5 (BM25, token-OR, date columns for temporal), naive full scan (no index, token-OR scoring in Rust).

### Overall

| system | ingest/s | disk MB | hit@1 | hit@5 | MRR | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|
| agidb | 27 | 185.7 | 0.480 | 0.540 | 0.508 | 0.75 | 2.76 |
| sqlite-fts5 | 278,059 | 0.9 | 0.625 | 0.750 | 0.676 | 0.16 | 0.36 |
| naive-scan | 6,146,715 | 0.6 | 0.535 | 0.735 | 0.621 | 0.36 | 0.43 |

### exact

| system | hit@1 | hit@5 | MRR | p95 ms |
|---|---|---|---|---|
| agidb | 0.000 | 0.060 | 0.041 | 1.21 |
| sqlite-fts5 | 1.000 | 1.000 | 1.000 | 0.34 |
| naive-scan | 0.640 | 0.980 | 0.788 | 0.43 |

### single-entity

| system | hit@1 | hit@5 | MRR | p95 ms |
|---|---|---|---|---|
| agidb | 1.000 | 1.000 | 1.000 | 0.51 |
| sqlite-fts5 | 1.000 | 1.000 | 1.000 | 0.19 |
| naive-scan | 1.000 | 1.000 | 1.000 | 0.37 |

### noisy

| system | hit@1 | hit@5 | MRR | p95 ms |
|---|---|---|---|---|
| agidb | 0.900 | 0.900 | 0.900 | 2.96 |
| sqlite-fts5 | 0.000 | 0.000 | 0.000 | 0.02 |
| naive-scan | 0.000 | 0.020 | 0.010 | 0.43 |

### temporal

| system | hit@1 | hit@5 | MRR | p95 ms |
|---|---|---|---|---|
| agidb | 0.020 | 0.200 | 0.092 | 0.90 |
| sqlite-fts5 | 0.500 | 1.000 | 0.704 | 0.41 |
| naive-scan | 0.500 | 0.940 | 0.684 | 0.39 |

## What this proves and what it doesn't

**agidb wins:**
- **noisy queries (1 char dropped + 1 swapped):** 0.90 hit@1 vs 0.00 for both FTS5 and naive-scan. The structured-HV tier-B phi correlation survives the perturbation because the role-bound entity HVs are still close. BM25 requires exact token overlap and dies.
- **single-entity queries:** 1.000 across all three systems (everyone resolves "Sarah" or "did X recommend anything" via the obvious lookup).

**agidb loses:**
- **exact queries:** 0.000 hit@1. The structured signature is a bundle of role-bound triple HVs that needs >5σ phi to fire; on single-triple episodes with no cue-token overlap into the bundle, tier B falls through to tier C (gist), which is a token-bundle similarity that loses on truly exact lexical matches. FTS5 trivially wins because BM25 rewards term frequency.
- **temporal queries:** 0.020 hit@1. The bi-temporal filter works (tier A and B honor `as_of`), but the structured-similarity scoring front-loads the wrong episodes before the temporal filter kicks in. FTS5 wins because its date-column filter is a WHERE clause that the BM25 ranker respects natively.

**where agidb's overhead shows:**
- **Ingest:** 27/s vs FTS5's 278k/s — agidb commits to redb + signs a per-episode HV. The trade-off is the on-disk size (185 MB vs FTS5's 0.9 MB) carrying 8192-bit signatures for every episode.
- **p95 latency:** 2.76 ms vs 0.36 ms — the scan-directory sweep is a single linear pass over 10k entries; FTS5 is a posting-list intersection. At larger corpora (100k+) this gap will close for FTS5 as its postings grow.

## Limitations

- **Synthetic + templated corpus.** The corpus is generated from 40-person × 40-place pools with templated sentences. It favors lexical systems and does not measure semantic paraphrase at all. Do not quote these numbers as semantic recall quality.
- **Lexical-structural only.** No system here embeds sentence meaning. agidb's recall is a token-bundle HDC similarity (tier C/D) and a role-bound triple similarity (tier B, phi-corrected). FTS5 is BM25 over token OR. Naive-scan is matched-token count.
- **Three of the constitution's six metrics not measured here.** BLEU, LLM-judge, and token cost require external corpora (LongMemEval / LoCoMo) and LLM API calls. They're pending — claims wait for numbers.
- **Single-threaded, in-process.** No network, no concurrency, no cold cache. Latency numbers are pure compute cost on warm OS page cache.
- **Honest losses preserved.** The exact-class and temporal-class rows where agidb loses are not deleted or rationalized; the structural reason for each loss is stated above.
- **No long-context test.** The corpus has ~10 tokens per document; real memory corpora have paragraphs. Larger-document ingest + recall would need separate measurement.

## End-to-end verification

End-to-end verification passed: 2026-07-05, `fa4ebe0` (HEAD of the 6-commit end-to-end-substrate-fix sequence: `fc592dd` → `656a1cb` → `41e6a38` → `92cbb05` → `b3c7b98` → `fa4ebe0`). Workspace tests: 165 passed, 0 failed. CLI end-to-end demo (`observe` ×3 + `recall` + `sense` duplicate (surprise=0.0, not promoted) + `sense` novel (surprise=0.43, promoted to ep4) + `sensory` + `consolidate` + `stats` shows 4 episodes / 4 signatures) green. MCP stdio `tools/list` over the offline `agidb serve` returns all 13 tools. Format check (`cargo fmt --check`) clean; clippy clean.