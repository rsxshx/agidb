//! Frame dispatch — sits between the WebSocket loop and the
//! (sync) `agidb_mcp::McpServer::handle_request`. The MCP server is
//! pure + blocking, so we drive it from inside `spawn_blocking` to
//! keep the tokio runtime responsive.

use agidb_mcp::protocol::{error_code, JsonRpcRequest, JsonRpcResponse};
use serde_json::{json, Value};

use crate::AppState;

/// Run one inbound JSON-RPC frame, return the response frame to send
/// back. Returns `None` for notifications (no reply per the spec).
///
/// All store-mutating work is funnelled through `spawn_blocking` so the
/// tokio runtime isn't blocked. The MCP server itself is sync and
/// thread-safe (each `AgidbContext` owns its redb + signature file
/// exclusively); we serialize concurrent writers via `AppState::write_lock`
/// so the redb writer lock never deadlocks across two WS frames.
pub async fn handle_ws_frame(state: &AppState, raw: &str) -> Option<JsonRpcResponse> {
    // Parse first (in async land — cheap, no I/O).
    let req: JsonRpcRequest = match serde_json::from_str(raw) {
        Ok(r) => r,
        Err(e) => {
            let id = extract_id(raw).unwrap_or(Value::Null);
            return Some(JsonRpcResponse::err(
                id,
                error_code::PARSE_ERROR,
                format!("invalid JSON-RPC: {e}"),
            ));
        }
    };

    let is_notification = req.id.is_none();

    // Heavy lifting off the runtime.
    let server = state.server.clone();
    let _guard = state.write_lock.lock().await; // serialize writers
    let resp = tokio::task::spawn_blocking(move || server.handle_request(req))
        .await
        .ok()
        .flatten();

    if is_notification {
        None
    } else {
        resp
    }
}

/// Best-effort `id` extractor for parse-error responses.
fn extract_id(raw: &str) -> Option<Value> {
    serde_json::from_str::<Value>(raw)
        .ok()
        .and_then(|v| v.get("id").cloned())
}

/// Phase alias table — the canonical 8 sarah_bawri demo phases and
/// which MCP tool(s) each one exercises, plus the input each one sends.
///
/// Used by both the server (`GET /tools` exposes this) and the landing
/// page's WebSocket handler, so the demo's chip set maps 1:1 to real calls.
#[allow(dead_code)] // consumed by the landing page's JS (server-side helpers)
pub fn phase_aliases() -> [(&'static str, &'static [&'static str]); 8] {
    [
        ("observe", &["observe"]),
        ("set_goal", &["set_goal", "goal"]),
        ("assert_belief", &["assert_belief", "belief"]),
        ("recall", &["recall"]),
        ("consolidate", &["consolidate", "sleep"]),
        (
            "revise_belief",
            &["revise_belief", "revise", "what_do_i_believe"],
        ),
        ("stats", &["stats", "self_vector"]),
        ("unlearn", &["unlearn", "forget sarah"]),
    ]
}

/// Public-facing JSON snapshot of `phase_aliases` for the GET /tools
/// endpoint. Lets the landing page render chips that match the
/// server's resolver.
pub fn phase_aliases_json() -> Value {
    let mut m = serde_json::Map::new();
    for (k, vs) in phase_aliases() {
        m.insert(k.into(), json!(vs));
    }
    Value::Object(m)
}

/// Resolve a user-typed alias to the canonical phase key. Falls back to
/// fuzzy containment so free-text like 'remember sarah likes thai'
/// routes to `observe` when it contains 'remember' or starts with
/// 'observe'.
#[allow(dead_code)] // exercised by tests + landing page can call via lib API
pub fn resolve_phase(q: &str) -> Option<&'static str> {
    let q = q.trim().to_ascii_lowercase();
    let q = q.trim_matches(|c: char| c == '"' || c == '\'');
    if q.is_empty() {
        return None;
    }
    // Build the alias → phase map statically (no Value lifetime to fight).
    let aliases: &[(&str, &[&str])] = &[
        ("observe", &["observe"]),
        ("set_goal", &["set_goal", "goal"]),
        ("assert_belief", &["assert_belief", "belief"]),
        ("recall", &["recall"]),
        ("consolidate", &["consolidate", "sleep"]),
        (
            "revise_belief",
            &["revise_belief", "revise", "what_do_i_believe"],
        ),
        ("stats", &["stats", "self_vector"]),
        ("unlearn", &["unlearn", "forget sarah"]),
    ];
    for (phase, list) in aliases {
        for a in *list {
            if q == *a || q.starts_with(&format!("{a} ")) || q.starts_with(&format!("{a}(")) {
                return Some(phase);
            }
        }
    }
    // Fuzzy fallbacks.
    if q.starts_with("observe") || q.starts_with("remember") || q.starts_with("store") {
        return Some("observe");
    }
    if q.starts_with("recall")
        || q.starts_with("search")
        || q.starts_with("find")
        || q.starts_with("what")
    {
        return Some("recall");
    }
    if q.starts_with("forget") || q.starts_with("unlearn") || q.starts_with("delete") {
        return Some("unlearn");
    }
    if q.starts_with("consolidate") || q.starts_with("sleep") {
        return Some("consolidate");
    }
    if q.starts_with("set_goal") || q == "goal" || q.contains("goal ") {
        return Some("set_goal");
    }
    if q.starts_with("assert_belief") || q.starts_with("believe") || q.contains("belief ") {
        return Some("assert_belief");
    }
    if q.starts_with("revise") || q.starts_with("what_do_i_believe") {
        return Some("revise_belief");
    }
    if q == "stats" || q.starts_with("self_vector") {
        return Some("stats");
    }
    None
}

/// Build the `arguments` Value for a given phase + the user-supplied
/// override string (only `observe` consumes it).
///
/// The point of doing this server-side: the *client* doesn't need to
/// know the MCP tool schemas. The browser sends `{ "phase": "observe",
/// "text": "sarah recommended bawri" }` and we translate it to the
/// right `tools/call` call. Same wire shape as before, real backend.
#[allow(dead_code)] // consumed by the landing page's JS (server-side helpers)
pub fn args_for_phase(phase: &str, user_text: Option<&str>) -> Value {
    match phase {
        "observe" => json!({
            "text": user_text.unwrap_or("Sarah recommended Bawri in Bandra"),
            "source": "demo-ui"
        }),
        "set_goal" => json!({
            "description": "find a thai place for the team dinner",
            "priority":    "active"
        }),
        "assert_belief" => json!({
            "claim":      "Sarah likes thai food",
            "confidence": 0.8
        }),
        "recall" => json!({
            "cue": user_text.unwrap_or("what thai place did Sarah mention?"),
            "k":   5
        }),
        "consolidate" => json!({}),
        "revise_belief" => json!({
            "claim": "Sarah dislikes thai food",
            "confidence": 0.6,
            "supersedes": "Sarah likes thai food"
        }),
        "stats" => json!({}),
        "unlearn" => json!({
            "target":   "Concept(Sarah)",
            "reason":   "user requested forget"
        }),
        _ => json!({}),
    }
}

/// Map phase → MCP tool name.
#[allow(dead_code)] // consumed by the landing page's JS (server-side helpers)
pub fn tool_for_phase(phase: &str) -> &'static str {
    match phase {
        "observe" => "memory_observe",
        "recall" => "memory_recall",
        "consolidate" => "memory_consolidate",
        "set_goal" => "memory_set_goal",
        "assert_belief" => "memory_assert_belief",
        "revise_belief" => "memory_revise_belief",
        "stats" => "memory_stats",
        "unlearn" => "memory_unlearn",
        _ => "memory_observe", // safe default for unknown phases
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_exact_alias() {
        assert_eq!(resolve_phase("observe"), Some("observe"));
        assert_eq!(resolve_phase("RECALL"), Some("recall"));
        assert_eq!(resolve_phase("  recall "), Some("recall"));
        assert_eq!(resolve_phase("\"recall\""), Some("recall"));
    }

    #[test]
    fn resolve_with_trailing_text() {
        // "recall sarah" should resolve to recall (cuing the agent).
        assert_eq!(resolve_phase("recall sarah"), Some("recall"));
        assert_eq!(resolve_phase("forget sarah"), Some("unlearn"));
        assert_eq!(
            resolve_phase("observe sarah recommended bawri"),
            Some("observe")
        );
    }

    #[test]
    fn resolve_unknown_returns_none() {
        assert_eq!(resolve_phase("help"), None);
        assert_eq!(resolve_phase("xyzzy"), None);
    }

    #[test]
    fn fuzzy_fallback_works() {
        assert_eq!(resolve_phase("remember sarah"), Some("observe"));
        assert_eq!(resolve_phase("what did sarah say?"), Some("recall"));
    }

    #[test]
    fn args_for_observe_uses_user_text() {
        let args = args_for_phase("observe", Some("hello world"));
        assert_eq!(args["text"], "hello world");
        assert_eq!(args["source"], "demo-ui");
    }

    #[test]
    fn args_for_observe_falls_back() {
        let args = args_for_phase("observe", None);
        assert!(args["text"].as_str().unwrap().contains("Sarah"));
    }

    #[test]
    fn tool_for_phase_maps() {
        assert_eq!(tool_for_phase("observe"), "memory_observe");
        assert_eq!(tool_for_phase("recall"), "memory_recall");
        assert_eq!(tool_for_phase("consolidate"), "memory_consolidate");
    }

    #[test]
    fn every_demo_phase_maps_to_a_registered_tool() {
        let registered: Vec<String> = agidb_mcp::tools::list()
            .into_iter()
            .map(|t| t.name)
            .collect();
        for phase in [
            "observe",
            "recall",
            "consolidate",
            "set_goal",
            "assert_belief",
            "revise_belief",
            "stats",
            "unlearn",
        ] {
            let tool = tool_for_phase(phase);
            assert!(
                registered.iter().any(|n| n == tool),
                "phase {phase} maps to unregistered tool {tool}"
            );
        }
    }
}
