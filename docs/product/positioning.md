# agidb — Positioning

> The product framing, sharpened. What agidb *is* to the person deciding
> whether to adopt it, led by what actually ships today rather than the
> five-year thesis. This document is the source of truth for the landing
> page and the README pitch; [agi-trajectory.md](./agi-trajectory.md) holds
> the long horizon, [overview.md](./overview.md) holds the full comparison,
> this holds the wedge.

## The one line

**agidb is the state and continuity layer for AI agents** — one embedded Rust file that holds what an agent knows, wants, and believes, with temporal truth, full provenance, and an auditable right-to-be-forgotten. No query language, no LLM in the read path, no separate vector store.

The constitutional one-liner (Article I) does not change: *"the cognitive substrate for autonomous AI agents — content-addressable hyperdimensional memory, first-class goals and beliefs, bi-temporal supersession, sleep-like consolidation, and a non-destructive unlearn primitive."* This document is the same claim aimed at the buyer instead of the architect. "Cognitive substrate" is what it **is**; "state and continuity layer" is what it **does for you today**.

## The reframe: an LLM is a stateless CPU

A large language model is a stateless processor. Prompt in, tokens out, nothing retained. What turns that processor into an *agent* is persistent state:

- what it has seen and learned (**memory**)
- what it is trying to do (**intent**)
- what it holds to be true, and how sure (**world model**)
- who it is and how it has changed (**identity**)
- what it is allowed to keep, and proof of what it dropped (**governance**)

Today every agent framework hand-rolls this state across a vector store, a JSON scratchpad, a graph DB, and a prompt template — three sets of credentials, 1–3 second recall latencies, no temporal grounding, weak provenance, and destructive deletes. agidb is the single substrate shaped like the state an agent actually has.

## The five kinds of state

| State | What it holds | Status in code |
|---|---|---|
| **Memory** | episodic events, semantic facts, procedural skills, working set | episodic + semantic **shipped**; procedural (types only) and working memory **not built** |
| **Intent** | goals as typed state machines, parent/child, success criteria | **shipped** — `set_goal`/`revise`/`complete`/`abandon`/`pause`/`resume` |
| **World model** | beliefs with confidence + revision log; bi-temporal truth; contradiction supersession | **shipped** — revisable beliefs, `valid_at`, `superseded_by` |
| **Identity** | self-model audit log, self-vector EMA, "what did I learn?" | **shipped** — learning log + self-vector |
| **Governance** | provenance on every claim, non-destructive cascading unlearn, permanent audit | **shipped** — this is the strongest, least-marketed differentiator |

The honest read: three of the five are substantially built, the other two are partial. Memory's missing pieces (working set, procedural retrieval) are small and near-term. The reframe does not overclaim — it re-prioritizes what already exists.

## The killer capability: context assembly

The read operation an agent actually needs is not `recall(query) → matches`. It is:

```
context_for(current_task, token_budget) → a ready-to-inject prompt block
```

— deterministically assembled from active goals, relevant beliefs, recalled episodes, and procedural hints, packed to a token budget, with provenance attached to every line. This is the "no query language" promise (Article IX) finally delivered for the way LLM agents work: the agent says what it is doing, agidb returns the state it needs to do it, and because it is pure math over stored signatures it stays constitutional (no LLM in the read path, Article IV).

No competitor offers context assembly as a primitive — they return chunks and leave the packing to the caller. It is also what makes agidb's value *measurable*: the same agent, same model, run with agidb versus with a vector store, on a long-horizon task — better task completion at lower token cost. That single comparison should drive the roadmap. **`context_for` is designed, not yet built; it is the top of the post-decision-gate backlog.**

## Why the wedge holds

Mem0, Letta, Zep, Cognee sit **above** the LLM as Python frameworks doing search-over-documents with an LLM in the loop. agidb sits **beneath** the agent as an embedded Rust substrate. Different layer, different shape. What none of them have, and what agidb has today:

1. **First-class intent.** Goals are typed state machines, not text rows. An agent can ask "what am I working on, and what's blocked?" and get a structured answer.
2. **A revisable world model.** Beliefs carry confidence and a full revision log. "What did I believe last week, and what changed my mind?" is a query, not an archaeology dig.
3. **Temporal truth.** Bi-temporal supersession: ask "what was true on Tuesday?" separately from "what did I know on Tuesday?" Facts supersede; nothing is silently overwritten.
4. **Governance you can put in front of an auditor.** Every claim traces to the verbatim observation that produced it. `unlearn()` removes data from retrieval, cascades through dependent beliefs and atoms, subtracts from the self-vector, and leaves a permanent audit entry — the data goes, the *fact that data was removed* stays. This is right-to-be-forgotten as a first-class operation, and it is the differentiator regulated buyers (legal, health, finance) cannot get anywhere else.

Governance is the positioning nobody can copy quickly, and it targets the buyers with budgets. Lead with it.

## What is shipped vs. what is roadmap

Stated plainly, because credibility with technical buyers depends on it.

**Shipped and tested (in this repo, ~180 passing tests):**
- HDC kernel — 8192-bit signatures, bind/bundle/hamming with SIMD paths
- Storage — redb (ACID) + mmap signature file, bi-temporal schema
- Tiered recall cascade — exact → lexical (inverted index + IDF) → HDC similarity → semantic (static embedder) → gist → nearest-neighbor, goal-biased
- Goals (state machine), beliefs (revision log), supersession, consolidation
- Non-destructive cascading unlearn + 30-day restore + permanent audit
- Learning log, self-vector, surprise-gated sensory buffer
- CLI (~30 commands), 13-tool MCP server, HTTP/WS demo server, benchmark harness

**Research roadmap — designed, not built (design docs only):**
- **`context_for` context assembly** — the killer read API above
- **Working memory** and **procedural retrieval** (`recall_procedure`) — the two missing memory floors
- **Python / TypeScript bindings** — required for real adoption and for running incumbent benchmarks
- **Dense retrieval tier** — an HNSW/ANN vector tier to fix paraphrase recall (currently loses to SQLite FTS5; requires an ADR amending Article II)
- **Multimodal sensory** (V-JEPA 2 + Wav2Vec-BERT), **brain-alignment / BAMS** — the v2.1 thesis; a research program and a paper, not a product feature, and explicitly off the critical path

The landing page mirrors this split exactly: shipped features lead; the roadmap is present, honest, and clearly labeled "not yet shipped."

## The known gaps we do not hide

- **Paraphrase retrieval loses to lexical BM25** (hit@1 0.032 vs 0.048) on the synthetic bench. The HDC-only representation cannot do paraphrase; the fix is a dense tier, which needs a constitutional amendment. This is the single most important retrieval fix.
- **The decision gate has not run.** The constitution (Article XIII) requires benchmarking against Mem0/Zep/Letta *before* committing to the cognitive-primitive bet. The primitives got built; the gate is still open. Running it — with Python bindings and LongMemEval/LoCoMo — is the gating milestone before any launch.
- **No distribution yet.** Rust-only, no Python/TS. Agents aren't written in Rust; without bindings the reachable market is tiny.

## The near-term plan this positioning implies

1. **Ship Python bindings.** Highest ROI in the repo — unlocks benchmarks, design partners, users.
2. **Add a dense retrieval tier** (ADR amending Article II) and re-run the bench so paraphrase is competitive.
3. **Run the decision gate** against the incumbents on LongMemEval + LoCoMo, thresholds committed first, raw results published.
4. **Build `context_for`** and the two missing memory floors — turn "memory library" into "state layer."
5. **Launch on governance** — "agent memory you can put in front of an auditor."

Brain-alignment, multimodal, and the causal layer are revived only after traction or funding. See [agi-trajectory.md](./agi-trajectory.md) for the long horizon and [roadmap.md](./roadmap.md) for the phase plan.
