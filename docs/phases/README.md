# agidb — phases

> the agidb v2 build plan, one file per phase. each phase has a single owner, a single exit criterion, and a hard date.

## timeline

| # | phase | weeks | status |
|---|---|---|---|
| 0 | [setup](./phase-0-setup.md) | — | ✅ complete (inherited from sochdb v1) |
| 1 | [HDC kernel](./phase-1-hdc-kernel.md) | — | ✅ complete (inherited) |
| 2 | [storage](./phase-2-storage.md) | — | ✅ complete (inherited) |
| 3 | [extraction (GLiNER)](./phase-3-extraction.md) | 1-4 | 🟨 partial — NER real, relations heuristic, 100-row gold set shipped, F1=0.592 measured |
| 4 | [binding + recall](./phase-4-binding-recall.md) | — | ✅ complete (inherited; phi-corrected tier B added) |
| 5 | [MCP + Python](./phase-5-mcp-python.md) | 5-8 | 🟨 partial — MCP shipped with 13 tools; Python bindings absent |
| 6 | [consolidation](./phase-6-consolidation.md) | — | ✅ complete (inherited) |
| 7 | [decision gate](./phase-7-decision-gate.md) | 11-13 | ⬜ not started — **binding** |
| 8 | [hardening + launch](./phase-8-hardening-launch.md) | 31-36 | ⬜ not started — v2.0 ship |
| 9 | [cognitive primitives](./phase-9-cognitive-primitives.md) | 13-18 | ✅ complete — goals + beliefs shipped (state machine + revisable log + revision math) |
| 10 | [sensory + self-model](./phase-10-sensory-self-model.md) | 19-22 | ✅ complete — floor-1 sensory buffer + surprise gating + self-model log + self-vector |
| 11 | [unlearn API](./phase-11-unlearn.md) | 23-25 | ✅ complete — cascading unlearn with 30-day restore window |
| 12 | [neurosymbolic interface](./phase-12-neurosymbolic.md) | 26-27 | ⬜ not started |
| 13 | [cognitive benchmarks](./phase-13-cognitive-benchmarks.md) | 28-30 | 🟨 partial — retrieval benchmark shipped (agidb vs FTS5 vs scan, 10k corpus, 4 query classes); LongMemEval harness pending |
| 14 | [multimodal sensory](./phase-14-multimodal-sensory.md) | 37-42 | ⬜ not started — v2.1 (gated) |
| 15 | [brain-calibrated surprise](./phase-15-brain-calibrated-surprise.md) | 43-46 | ⬜ not started — v2.1 (gated) |
| 16 | [BAMS benchmark + ICLR paper](./phase-16-bams-benchmark.md) | 47-52 | ⬜ not started — v2.1 (gated), ICLR 2026 target dropped |

## the rule

a phase exits only when its exit criterion is met **on a reproducible benchmark**. partial implementations do not exit a phase. they are tracked but they do not unblock the next phase.

## status

phases 0, 1, 2, 4, and 6 are complete — inherited from sochdb v1 and verified by 100+ passing tests.

phase 3 is **partial** — the 100-row human-labelled gold set is now shipped (`crates/agidb-extract/eval/gold/observations.jsonl`), the eval F1 gate is parameterized (`--gate <f1>`), and the measured F1 is 0.592 (P=0.865, R=0.450) recorded in `bench/RESULTS.md`. The relation extractor remains a curated heuristic synonym table; an ONNX-based relation extractor (glirel.rs or relex.rs port) is the next lift before phase 3 exits. See [`phase-3-extraction.md`](./phase-3-extraction.md).

phase 5 is **partial** — the MCP stdio server now exposes 13 tools (memory_observe / memory_recall / memory_consolidate / memory_get_episode / memory_set_goal / memory_active_goals / memory_assert_belief / memory_revise_belief / memory_beliefs / memory_unlearn / memory_what_did_i_learn / memory_stats / memory_sense). Python pyo3 bindings and pip wheels are untouched. See [`phase-5-mcp-python.md`](./phase-5-mcp-python.md).

phases 9, 10, 11 are **complete** — goals + beliefs (state machine + revisable belief log + revision math), floor-1 sensory buffer with surprise-gated promotion, self-model audit log + self-vector history, and cascading unlearn with a 30-day restore window + permanent audit record. Originally phases 12, 14–16 are not started; phase 12 (neurosymbolic) and phase 13 (LongMemEval harness) remain the open work for the v0.2 wedge.

the retrieval benchmark harness (phase 13 prep) is shipped: `agidb-bench` runs a deterministic 10k-ep synthetic corpus through agidb vs SQLite FTS5 vs naive-scan across four query classes (exact / single-entity / noisy-cue / temporal) with hit@k, MRR, p50/p95 latency, ingest throughput, and disk-size metrics. Numbers in `bench/RESULTS.md`; raw JSON alongside. The constitution's full six-metric stack (BLEU, LLM-judge, token cost) requires external corpora + LLM APIs and is not yet run.

weeks 9-10 are a benchmark-harness build that is phase-7 prep, not a separate phase.

note: "weeks" count from the agidb v2 kickoff. a pre-week-0 rebrand (sochdb→agidb) precedes week 1 — namespace lock, crate renames, and the GitHub org move happen before the week-counter starts.

## see also

- [../product/roadmap.md](../product/roadmap.md) — the narrative version of this plan with the risk register
- [../spec/constitution.md](../spec/constitution.md) — the immutable principles each phase must honor
