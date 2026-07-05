//! Tool definitions for the agidb MCP server.
//!
//! Each tool has a JSON-Schema input shape, a stable name, and a handler
//! that takes the unified [`AgidbContext`] (store + extractor) plus the
//! caller's parsed args, and returns a structured result.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use agidb_core::types::{Belief, EpisodeId, Goal, Query};
use agidb_core::unlearn::UnlearnTarget;

use crate::context::AgidbContext;
use crate::protocol::{McpError, ToolDescriptor, ToolResult};

pub type ToolFn = fn(&AgidbContext, Value) -> Result<ToolResult, McpError>;

pub struct Tool {
    pub name: &'static str,
    pub description: &'static str,
    pub schema: fn() -> Value,
    pub handler: ToolFn,
}

/// The full tool registry exposed by this server. Order is the order
/// `tools/list` returns; stable so clients can cache.
pub fn registry() -> Vec<Tool> {
    vec![
        Tool {
            name: "memory_observe",
            description:
                "Record a new observation. Runs layer-2 extraction (or stores text-only when no \
                 model is loaded) and persists an Episode with bi-temporal stamps.",
            schema: observe_schema,
            handler: observe,
        },
        Tool {
            name: "memory_recall",
            description:
                "Tiered recall against the store. Never returns the empty set; the deepest tier \
                 (NearestNeighbor) always emits at least one match unless `tier_floor` caps it.",
            schema: recall_schema,
            handler: recall,
        },
        Tool {
            name: "memory_consolidate",
            description:
                "Run the consolidation worker once: cluster recent episodes into SemanticAtoms, \
                 detect contradictions, write an audit-log entry.",
            schema: consolidate_schema,
            handler: consolidate,
        },
        Tool {
            name: "memory_get_episode",
            description: "Fetch a single Episode by id.",
            schema: get_episode_schema,
            handler: get_episode,
        },
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
    ]
}

/// Render the registry as the `tools/list` MCP response payload.
pub fn list() -> Vec<ToolDescriptor> {
    registry()
        .into_iter()
        .map(|t| ToolDescriptor {
            name: t.name.to_string(),
            description: t.description.to_string(),
            input_schema: (t.schema)(),
        })
        .collect()
}

/// Dispatch a `tools/call` request to the appropriate handler.
pub fn call(ctx: &AgidbContext, name: &str, args: Value) -> Result<ToolResult, McpError> {
    for tool in registry() {
        if tool.name == name {
            return (tool.handler)(ctx, args);
        }
    }
    Err(McpError::InvalidParams(format!("unknown tool: {name}")))
}

// ---------------------------------------------------------------------------
// memory_observe
// ---------------------------------------------------------------------------

fn observe_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The raw observation to record."
            },
            "source": {
                "type": "string",
                "description": "Caller-supplied provenance label (e.g. 'user', 'tool:gmail').",
                "default": "mcp"
            }
        },
        "required": ["text"]
    })
}

#[derive(Deserialize)]
struct ObserveArgs {
    text: String,
    #[serde(default = "default_source")]
    source: String,
}

fn default_source() -> String {
    "mcp".into()
}

#[derive(Serialize)]
struct ObserveResult {
    episode_id: u64,
}

fn observe(ctx: &AgidbContext, args: Value) -> Result<ToolResult, McpError> {
    let args: ObserveArgs = serde_json::from_value(args)?;
    let id = ctx.observe_text(&args.text, &args.source)?;
    Ok(ToolResult::json(&serde_json::to_value(ObserveResult {
        episode_id: id.raw(),
    })?))
}

// ---------------------------------------------------------------------------
// memory_recall
// ---------------------------------------------------------------------------

fn recall_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "cue": {
                "type": "string",
                "description": "The retrieval cue. Tokenized for tier-A concept lookup and \
                                encoded into a gist signature for tier-C/D fallback."
            },
            "k": {
                "type": "integer",
                "minimum": 1,
                "description": "Maximum matches to return.",
                "default": 10
            },
            "min_confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Confidence floor; matches below this are dropped.",
                "default": 0.0
            }
        },
        "required": ["cue"]
    })
}

#[derive(Deserialize)]
struct RecallArgs {
    cue: String,
    #[serde(default = "default_k")]
    k: usize,
    #[serde(default)]
    min_confidence: f32,
}

fn default_k() -> usize {
    10
}

#[derive(Serialize)]
struct RecallResult {
    tier_used: String,
    elapsed_ms: u32,
    matches: Vec<RecallMatchOut>,
    semantic_atoms: Vec<SemanticOut>,
}

#[derive(Serialize)]
struct RecallMatchOut {
    episode_id: u64,
    text: String,
    confidence: f32,
    tier: String,
}

#[derive(Serialize)]
struct SemanticOut {
    atom_id: u64,
    statement: String,
    confidence: f32,
}

fn recall(ctx: &AgidbContext, args: Value) -> Result<ToolResult, McpError> {
    let args: RecallArgs = serde_json::from_value(args)?;
    let query = Query::cue(args.cue)
        .with_k(args.k)
        .with_min_confidence(args.min_confidence);
    let r = ctx.recall(&query)?;
    let payload = RecallResult {
        tier_used: format!("{:?}", r.tier_used),
        elapsed_ms: r.elapsed_ms,
        matches: r
            .matches
            .into_iter()
            .map(|m| RecallMatchOut {
                episode_id: m.episode_id.raw(),
                text: m.text,
                confidence: m.confidence,
                tier: format!("{:?}", m.source_tier),
            })
            .collect(),
        semantic_atoms: r
            .semantic_atoms
            .into_iter()
            .map(|a| SemanticOut {
                atom_id: a.atom_id.raw(),
                statement: a.statement,
                confidence: a.confidence,
            })
            .collect(),
    };
    Ok(ToolResult::json(&serde_json::to_value(payload)?))
}

// ---------------------------------------------------------------------------
// memory_consolidate
// ---------------------------------------------------------------------------

fn consolidate_schema() -> Value {
    json!({ "type": "object", "properties": {} })
}

#[derive(Serialize)]
struct ConsolidateResult {
    episodes_scanned: u32,
    semantic_atoms_created: u32,
    contradictions_detected: u32,
}

fn consolidate(ctx: &AgidbContext, _args: Value) -> Result<ToolResult, McpError> {
    let r = ctx.consolidate()?;
    let payload = ConsolidateResult {
        episodes_scanned: r.episodes_scanned,
        semantic_atoms_created: r.semantic_atoms_created,
        contradictions_detected: r.contradictions_detected,
    };
    Ok(ToolResult::json(&serde_json::to_value(payload)?))
}

// ---------------------------------------------------------------------------
// memory_get_episode
// ---------------------------------------------------------------------------

fn get_episode_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "id": { "type": "integer", "minimum": 0, "description": "EpisodeId" }
        },
        "required": ["id"]
    })
}

#[derive(Deserialize)]
struct GetEpisodeArgs {
    id: u64,
}

fn get_episode(ctx: &AgidbContext, args: Value) -> Result<ToolResult, McpError> {
    let args: GetEpisodeArgs = serde_json::from_value(args)?;
    match ctx.get_episode(args.id)? {
        Some(ep) => Ok(ToolResult::json(&json!({
            "id": ep.id.raw(),
            "text": ep.text,
            "confidence": ep.confidence,
            "triples": ep.triples.iter().map(|t| json!({
                "subject": t.subject,
                "predicate": t.predicate,
                "object": t.object,
                "confidence": t.confidence,
            })).collect::<Vec<_>>(),
            "valid_time": {
                "start": ep.valid_time.start.to_rfc3339(),
                "end": ep.valid_time.end.map(|e| e.to_rfc3339()),
            },
        }))),
        None => Ok(ToolResult::error(format!("episode {} not found", args.id))),
    }
}

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

fn parse_id(s: &str) -> Result<u64, McpError> {
    s.trim()
        .parse::<u64>()
        .map_err(|_| McpError::InvalidParams(format!("expected numeric id, got {s:?}")))
}

fn unlearn(ctx: &AgidbContext, args: Value) -> Result<ToolResult, McpError> {
    let args: UnlearnArgs = serde_json::from_value(args)?;
    let target = match args.target_kind.as_str() {
        "episode" => UnlearnTarget::Episode(EpisodeId::new(parse_id(&args.target)?)),
        "belief" => {
            UnlearnTarget::Belief(agidb_core::types::BeliefId::new(parse_id(&args.target)?))
        }
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
        other => return Err(McpError::InvalidParams(format!("bad target_kind: {other}"))),
    };
    let report = ctx.with_store_mut(|s| s.unlearn(target, args.reason))?;
    Ok(ToolResult::json(&serde_json::to_value(report)?))
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
